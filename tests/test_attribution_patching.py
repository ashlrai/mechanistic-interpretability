"""Unit tests for attribution patching — math, ranking, artifacts.

All tests use a fake backend that returns synthetic activations and gradients.
No real model is loaded.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from mech_interp.experiments.attribution_patching import (
    AttributionPatchingExperiment,
    _expand_site_entry,
    _resolve_hook_sites,
)
from mech_interp.orchestration.proposal_generators import AttributionPatchingProposalGenerator
from mech_interp.types import (
    ActivationPatchRequest,
    ActivationPatchSiteResult,
    CrossModelProbeRequest,
    CrossModelProbeResult,
    ExperimentRun,
    ExperimentSpec,
    RunStatus,
    utc_now,
)

# ---------------------------------------------------------------------------
# Fake backend
# ---------------------------------------------------------------------------

_HOOK_A = "blocks.0.hook_resid_pre"
_HOOK_B = "blocks.1.hook_resid_pre"
_HOOK_C = "blocks.2.hook_mlp_out"


class FakeAttributionBackend:
    """Synthetic backend for attribution patching unit tests.

    Attributes
    ----------
    clean_acts:
        Per-site tensors returned by ``capture_activations``.
    corrupted_grads:
        Per-site gradient values returned as part of ``run_with_grad_cache``.
    corrupted_acts:
        Per-site corrupted activation values.
    """

    name = "transformerlens"

    def __init__(
        self,
        *,
        clean_acts: dict[str, list[list[float]]] | None = None,
        corrupted_acts: dict[str, list[list[float]]] | None = None,
        corrupted_grads: dict[str, list[list[float]]] | None = None,
    ) -> None:
        # Default: 1 batch × 3 seq × 4 d_model tensors
        default_clean = [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8], [0.0, 0.1, 0.2, 0.3]]
        default_corrupt = [[0.0, 0.1, 0.2, 0.3], [0.4, 0.5, 0.6, 0.7], [0.0, 0.0, 0.1, 0.2]]
        default_grad = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]

        self._clean_acts = clean_acts or {
            _HOOK_A: default_clean,
            _HOOK_B: default_clean,
            _HOOK_C: default_clean,
        }
        self._corrupt_acts = corrupted_acts or {
            _HOOK_A: default_corrupt,
            _HOOK_B: default_corrupt,
            _HOOK_C: default_corrupt,
        }
        self._grads = corrupted_grads or {
            _HOOK_A: default_grad,
            _HOOK_B: default_grad,
            _HOOK_C: default_grad,
        }

    def load(self) -> None:
        pass

    def capture_activations(self, prompts: list[str], sites: list[str]) -> dict[str, Any]:
        return {
            site: np.array(self._clean_acts[site])
            for site in sites
            if site in self._clean_acts
        }

    def run_intervention(self, prompt: str, interventions: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    def run_activation_patching(
        self, request: ActivationPatchRequest
    ) -> list[ActivationPatchSiteResult]:
        # Minimal exact patching for the correlation test.
        results = []
        for pair in request.prompt_pairs:
            for site in request.hook_sites:
                # Synthetic recovery: higher layer index → lower recovery.
                layer = int(site.split(".")[1]) if site.startswith("blocks.") else 0
                recovery = max(0.0, 1.0 - layer * 0.15)
                results.append(
                    ActivationPatchSiteResult(
                        pair_id=pair.id,
                        hook_site=site,
                        clean_logit_diff=2.0,
                        corrupted_logit_diff=-1.0,
                        patched_logit_diff=2.0 * recovery - 1.0 * (1.0 - recovery),
                        recovery_fraction=recovery,
                    )
                )
        return results

    def run_cross_model_probe(
        self, request: CrossModelProbeRequest
    ) -> list[CrossModelProbeResult]:
        raise NotImplementedError

    def run_with_grad_cache(
        self,
        *,
        prompt: str,
        hook_sites: list[str],
        correct_token: str,
        incorrect_token: str,
        target_position: int = -1,
        clean_cache: dict[str, Any],
    ) -> dict[str, float]:
        """Compute attribution = (clean - corrupted) · grad for each site."""
        scores: dict[str, float] = {}
        for site in hook_sites:
            if site not in self._corrupt_acts or site not in self._grads:
                scores[site] = 0.0
                continue
            clean_np = np.array(clean_cache.get(site, self._clean_acts.get(site, [[]])))
            corrupt_np = np.array(self._corrupt_acts[site])
            grad_np = np.array(self._grads[site])
            scores[site] = float(np.sum((clean_np - corrupt_np) * grad_np))
        return scores


def _make_run(tmp_path: Path, run_id: int = 1) -> ExperimentRun:
    return ExperimentRun(
        id=run_id,
        spec_name="test",
        family="attribution_patching",
        backend="transformerlens",
        status=RunStatus.RUNNING,
        artifact_dir=tmp_path,
        created_at=utc_now(),
    )


def _make_spec(
    hook_sites: list[Any] | None = None,
    extra_params: dict[str, Any] | None = None,
) -> ExperimentSpec:
    params: dict[str, Any] = {
        "model": "gpt2-small",
        "prompt_pairs": [
            {
                "id": "capital-france",
                "clean_prompt": "The capital of France is Paris",
                "corrupted_prompt": "The capital of France is Rome",
                "correct_token": " Paris",
                "incorrect_token": " Rome",
            }
        ],
        "hook_sites": hook_sites
        if hook_sites is not None
        else [_HOOK_A, _HOOK_B, _HOOK_C],
        "seed": 42,
        "artifact_policy": {"write_report": False},
    }
    if extra_params:
        params.update(extra_params)
    return ExperimentSpec(
        name="test-attribution",
        family="attribution_patching",
        backend="transformerlens",
        parameters=params,
    )


# ---------------------------------------------------------------------------
# Math correctness
# ---------------------------------------------------------------------------


def test_attribution_dot_product_sign_convention() -> None:
    """Attribution = (clean - corrupted) · grad; verify sign matches manual calc."""
    # clean > corrupted everywhere → diff > 0; grad > 0 → positive attribution
    backend = FakeAttributionBackend(
        clean_acts={_HOOK_A: [[1.0, 0.0]], _HOOK_B: [[0.0, 0.0]], _HOOK_C: [[0.0, 0.0]]},
        corrupted_acts={_HOOK_A: [[0.0, 0.0]], _HOOK_B: [[0.0, 0.0]], _HOOK_C: [[0.0, 0.0]]},
        corrupted_grads={_HOOK_A: [[1.0, 1.0]], _HOOK_B: [[0.0, 0.0]], _HOOK_C: [[0.0, 0.0]]},
    )
    # diff = [[1,0]], grad = [[1,1]] → dot = 1*1 + 0*1 = 1.0
    score = backend.run_with_grad_cache(
        prompt="test",
        hook_sites=[_HOOK_A],
        correct_token=" Paris",
        incorrect_token=" Rome",
        target_position=-1,
        clean_cache=backend.capture_activations(["test"], [_HOOK_A]),
    )
    assert math.isclose(score[_HOOK_A], 1.0, rel_tol=1e-5)


def test_attribution_negative_sign() -> None:
    """When clean < corrupted the attribution should be negative."""
    backend = FakeAttributionBackend(
        clean_acts={_HOOK_A: [[0.0, 0.0]], _HOOK_B: [[0.0, 0.0]], _HOOK_C: [[0.0, 0.0]]},
        corrupted_acts={_HOOK_A: [[1.0, 0.0]], _HOOK_B: [[0.0, 0.0]], _HOOK_C: [[0.0, 0.0]]},
        corrupted_grads={_HOOK_A: [[1.0, 0.0]], _HOOK_B: [[0.0, 0.0]], _HOOK_C: [[0.0, 0.0]]},
    )
    # diff = -1, grad = 1 → attribution = -1
    score = backend.run_with_grad_cache(
        prompt="test",
        hook_sites=[_HOOK_A],
        correct_token=" Paris",
        incorrect_token=" Rome",
        target_position=-1,
        clean_cache=backend.capture_activations(["test"], [_HOOK_A]),
    )
    assert score[_HOOK_A] < 0.0


def test_attribution_zero_grad_gives_zero() -> None:
    backend = FakeAttributionBackend(
        clean_acts={_HOOK_A: [[1.0, 2.0]], _HOOK_B: [[0.0, 0.0]], _HOOK_C: [[0.0, 0.0]]},
        corrupted_acts={_HOOK_A: [[0.0, 0.0]], _HOOK_B: [[0.0, 0.0]], _HOOK_C: [[0.0, 0.0]]},
        corrupted_grads={_HOOK_A: [[0.0, 0.0]], _HOOK_B: [[0.0, 0.0]], _HOOK_C: [[0.0, 0.0]]},
    )
    score = backend.run_with_grad_cache(
        prompt="test",
        hook_sites=[_HOOK_A],
        correct_token=" Paris",
        incorrect_token=" Rome",
        target_position=-1,
        clean_cache=backend.capture_activations(["test"], [_HOOK_A]),
    )
    assert score[_HOOK_A] == 0.0


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------


def test_attribution_ranking_by_abs(tmp_path: Path) -> None:
    """Sites should be ranked by abs(attribution) descending."""
    backend = FakeAttributionBackend(
        clean_acts={
            _HOOK_A: [[3.0, 0.0]],
            _HOOK_B: [[1.0, 0.0]],
            _HOOK_C: [[2.0, 0.0]],
        },
        corrupted_acts={
            _HOOK_A: [[0.0, 0.0]],
            _HOOK_B: [[0.0, 0.0]],
            _HOOK_C: [[0.0, 0.0]],
        },
        corrupted_grads={
            _HOOK_A: [[1.0, 0.0]],
            _HOOK_B: [[1.0, 0.0]],
            _HOOK_C: [[1.0, 0.0]],
        },
    )
    spec = _make_spec()
    run = _make_run(tmp_path)
    result = AttributionPatchingExperiment(backend=backend).run(spec, run)

    assert result.status == RunStatus.SUCCEEDED
    ranked = json.loads(
        Path(result.artifacts["attribution_ranked_json"]).read_text(encoding="utf-8")
    )
    # HOOK_A diff=3, HOOK_C diff=2, HOOK_B diff=1 → that order
    assert ranked[0]["hook_site"] == _HOOK_A
    assert ranked[1]["hook_site"] == _HOOK_C
    assert ranked[2]["hook_site"] == _HOOK_B
    assert ranked[0]["rank"] == 1
    assert ranked[2]["rank"] == 3


def test_attribution_ranking_deterministic_across_seeds(tmp_path: Path) -> None:
    """Same inputs → same ranking regardless of seed parameter."""
    backend = FakeAttributionBackend()

    def _run(seed: int) -> list[dict[str, Any]]:
        spec = _make_spec(extra_params={"seed": seed})
        run = _make_run(tmp_path / f"seed{seed}")
        result = AttributionPatchingExperiment(backend=backend).run(spec, run)
        parsed: list[dict[str, Any]] = json.loads(
            Path(result.artifacts["attribution_ranked_json"]).read_text(encoding="utf-8")
        )
        return parsed

    ranked_42 = _run(42)
    ranked_99 = _run(99)
    assert [r["hook_site"] for r in ranked_42] == [r["hook_site"] for r in ranked_99]


# ---------------------------------------------------------------------------
# Artifact serialisation
# ---------------------------------------------------------------------------


def test_attribution_artifacts_written(tmp_path: Path) -> None:
    backend = FakeAttributionBackend()
    spec = _make_spec(extra_params={"artifact_policy": {"write_report": True}})
    run = _make_run(tmp_path)

    result = AttributionPatchingExperiment(backend=backend).run(spec, run)

    assert result.status == RunStatus.SUCCEEDED
    assert "attribution_ranked_json" in result.artifacts
    assert "attribution_ranked_csv" in result.artifacts
    assert "attribution_summary" in result.artifacts
    assert "research_note" in result.artifacts

    # Validate JSON schema
    ranked = json.loads(
        Path(result.artifacts["attribution_ranked_json"]).read_text(encoding="utf-8")
    )
    assert len(ranked) == 3
    for row in ranked:
        assert "hook_site" in row
        assert "attribution_score" in row
        assert "abs_attribution_score" in row
        assert row["evidence_label"] == "attribution_approximation"

    # Validate CSV header
    csv_text = Path(result.artifacts["attribution_ranked_csv"]).read_text(encoding="utf-8")
    assert "rank,hook_site,attribution_score,abs_attribution_score,evidence_label" in csv_text

    # Summary schema
    summary = json.loads(
        Path(result.artifacts["attribution_summary"]).read_text(encoding="utf-8")
    )
    assert summary["prompt_pair_count"] == 1
    assert summary["hook_site_count"] == 3
    assert "top_k_sites" in summary


def test_attribution_report_contains_caveat(tmp_path: Path) -> None:
    backend = FakeAttributionBackend()
    spec = _make_spec(extra_params={"artifact_policy": {"write_report": True}})
    run = _make_run(tmp_path)
    result = AttributionPatchingExperiment(backend=backend).run(spec, run)
    report = Path(result.artifacts["research_note"]).read_text(encoding="utf-8")
    assert "approximation" in report.lower()
    assert "follow-up" in report.lower() or "Follow-up" in report


# ---------------------------------------------------------------------------
# Multiple prompt pairs averaging
# ---------------------------------------------------------------------------


def test_attribution_averages_across_pairs(tmp_path: Path) -> None:
    """Attribution scores are averaged across prompt pairs."""
    backend = FakeAttributionBackend()
    spec = ExperimentSpec(
        name="multi-pair",
        family="attribution_patching",
        backend="transformerlens",
        parameters={
            "model": "gpt2-small",
            "prompt_pairs": [
                {
                    "id": "p1",
                    "clean_prompt": "The capital of France is Paris",
                    "corrupted_prompt": "The capital of France is Rome",
                    "correct_token": " Paris",
                    "incorrect_token": " Rome",
                },
                {
                    "id": "p2",
                    "clean_prompt": "The Eiffel Tower is in Paris",
                    "corrupted_prompt": "The Eiffel Tower is in Rome",
                    "correct_token": " Paris",
                    "incorrect_token": " Rome",
                },
            ],
            "hook_sites": [_HOOK_A, _HOOK_B],
            "artifact_policy": {"write_report": False},
        },
    )
    run = _make_run(tmp_path)
    result = AttributionPatchingExperiment(backend=backend).run(spec, run)

    assert result.status == RunStatus.SUCCEEDED
    assert result.metrics["prompt_pair_count"] == 2.0


# ---------------------------------------------------------------------------
# Hook-site resolution
# ---------------------------------------------------------------------------


def test_expand_site_alias_with_layers() -> None:
    expanded = _expand_site_entry({"site": "resid_pre", "layers": [0, 1, 2]})
    assert expanded == [
        "blocks.0.hook_resid_pre",
        "blocks.1.hook_resid_pre",
        "blocks.2.hook_resid_pre",
    ]


def test_expand_site_literal_tl_name() -> None:
    name = "blocks.5.hook_resid_post"
    assert _expand_site_entry(name) == [name]


def test_resolve_hook_sites_deduplicates() -> None:
    sites = _resolve_hook_sites(
        [
            "blocks.0.hook_resid_pre",
            "blocks.0.hook_resid_pre",
            {"site": "resid_pre", "layers": [0]},
        ]
    )
    assert sites.count("blocks.0.hook_resid_pre") == 1


def test_resolve_hook_sites_empty_raises() -> None:
    with pytest.raises(ValueError, match="at least one hook_site"):
        _resolve_hook_sites([])


# ---------------------------------------------------------------------------
# No run_with_grad_cache raises
# ---------------------------------------------------------------------------


def test_missing_grad_cache_method_raises(tmp_path: Path) -> None:
    class BareBackend:
        name = "transformerlens"

        def load(self) -> None:
            pass

        def capture_activations(
            self, prompts: list[str], sites: list[str]
        ) -> dict[str, Any]:
            return {}

        def run_intervention(
            self, prompt: str, interventions: dict[str, Any]
        ) -> dict[str, Any]:
            raise NotImplementedError

        def run_activation_patching(
            self, request: ActivationPatchRequest
        ) -> list[ActivationPatchSiteResult]:
            raise NotImplementedError

        def run_cross_model_probe(
            self, request: CrossModelProbeRequest
        ) -> list[CrossModelProbeResult]:
            raise NotImplementedError

    spec = _make_spec()
    run = _make_run(tmp_path)
    with pytest.raises(
        (AttributeError, Exception),
    ):
        AttributionPatchingExperiment(backend=BareBackend()).run(spec, run)


# ---------------------------------------------------------------------------
# Proposal generator
# ---------------------------------------------------------------------------


def test_attribution_proposal_generator_emits_circuit_patching(tmp_path: Path) -> None:
    # Build minimal artifact files.
    summary = {
        "model": "gpt2-small",
        "prompt_pair_count": 3,
        "mean_abs_attribution": 0.05,
        "top_k": 5,
        "top_k_sites": ["blocks.0.hook_resid_pre", "blocks.1.hook_resid_pre"],
        "hook_site_count": 24,
    }
    ranked = [
        {
            "rank": 1,
            "hook_site": "blocks.0.hook_resid_pre",
            "attribution_score": 0.1,
            "abs_attribution_score": 0.1,
        },
        {
            "rank": 2,
            "hook_site": "blocks.1.hook_resid_pre",
            "attribution_score": -0.08,
            "abs_attribution_score": 0.08,
        },
    ]
    spec_data = {
        "parameters": {
            "model": "gpt2-small",
            "prompt_pairs": [
                {
                    "id": "capital-france",
                    "clean_prompt": "The capital of France is Paris",
                    "corrupted_prompt": "The capital of France is Rome",
                    "correct_token": " Paris",
                    "incorrect_token": " Rome",
                }
            ],
        }
    }
    (tmp_path / "attribution_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (tmp_path / "attribution_ranked.json").write_text(json.dumps(ranked), encoding="utf-8")
    (tmp_path / "spec.json").write_text(json.dumps(spec_data), encoding="utf-8")

    generator = AttributionPatchingProposalGenerator()
    proposals = generator.generate(tmp_path, limit=5)

    assert len(proposals) == 1
    proposal = proposals[0]
    assert proposal["family"] == "circuit_patching"
    assert "blocks.0.hook_resid_pre" in proposal["parameters"]["hook_sites"]
    assert proposal["parameters"]["model"] == "gpt2-small"
    assert len(proposal["parameters"]["prompt_pairs"]) == 1


def test_attribution_proposal_generator_missing_artifacts(tmp_path: Path) -> None:
    generator = AttributionPatchingProposalGenerator()
    proposals = generator.generate(tmp_path, limit=5)
    assert proposals == []


# ---------------------------------------------------------------------------
# Correlation with exact patching (Spearman rank ≥ 0.8)
# ---------------------------------------------------------------------------


def _spearman(x: list[float], y: list[float]) -> float:
    """Inline Spearman rank correlation (no scipy)."""
    n = len(x)
    if n < 2:
        return 0.0

    def _ranks(vals: list[float]) -> list[float]:
        sorted_idx = sorted(range(n), key=lambda i: vals[i])
        ranks = [0.0] * n
        for rank, idx in enumerate(sorted_idx):
            ranks[idx] = float(rank + 1)
        return ranks

    rx = _ranks(x)
    ry = _ranks(y)
    mean_rx = sum(rx) / n
    mean_ry = sum(ry) / n
    num = sum((rx[i] - mean_rx) * (ry[i] - mean_ry) for i in range(n))
    den_x = math.sqrt(sum((rx[i] - mean_rx) ** 2 for i in range(n)))
    den_y = math.sqrt(sum((ry[i] - mean_ry) ** 2 for i in range(n)))
    if den_x == 0 or den_y == 0:
        return 0.0
    return num / (den_x * den_y)


def test_attribution_correlates_with_exact_patching(tmp_path: Path) -> None:
    """Attribution ranking should correlate with exact patching recovery (Spearman ≥ 0.8).

    The fake backend's exact-patching recovery decreases monotonically with layer index,
    and the attribution scores also decrease monotonically (same synthetic activations),
    so the rank correlation should be ≥ 0.8.
    """
    from mech_interp.types import ActivationPatchPromptPair, ActivationPatchRequest

    # Probe sites at layers 0, 1, 2 — resid_pre only
    probe_sites = [
        "blocks.0.hook_resid_pre",
        "blocks.1.hook_resid_pre",
        "blocks.2.hook_resid_pre",
    ]

    # Synthetic backend with monotonically decreasing activation magnitude by layer
    clean_acts = {
        site: [[float(3 - int(site.split(".")[1])), 0.0]]
        for site in probe_sites
    }
    corrupt_acts = {site: [[0.0, 0.0]] for site in probe_sites}
    grads = {site: [[1.0, 0.0]] for site in probe_sites}

    backend = FakeAttributionBackend(
        clean_acts=clean_acts,
        corrupted_acts=corrupt_acts,
        corrupted_grads=grads,
    )

    # Run attribution patching
    spec = ExperimentSpec(
        name="correlation-test",
        family="attribution_patching",
        backend="transformerlens",
        parameters={
            "model": "gpt2-small",
            "prompt_pairs": [
                {
                    "id": "p1",
                    "clean_prompt": "The capital of France is Paris",
                    "corrupted_prompt": "The capital of France is Rome",
                    "correct_token": " Paris",
                    "incorrect_token": " Rome",
                },
            ],
            "hook_sites": probe_sites,
            "artifact_policy": {"write_report": False},
        },
    )
    run = _make_run(tmp_path)
    attr_result = AttributionPatchingExperiment(backend=backend).run(spec, run)
    attr_ranked = json.loads(
        Path(attr_result.artifacts["attribution_ranked_json"]).read_text(encoding="utf-8")
    )
    attr_scores = {row["hook_site"]: row["abs_attribution_score"] for row in attr_ranked}

    # Run exact patching via the fake backend's run_activation_patching
    pair = ActivationPatchPromptPair(
        id="p1",
        clean_prompt="The capital of France is Paris",
        corrupted_prompt="The capital of France is Rome",
        correct_token=" Paris",
        incorrect_token=" Rome",
    )
    patch_request = ActivationPatchRequest(
        model_name="gpt2-small",
        prompt_pairs=(pair,),
        hook_sites=tuple(probe_sites),
    )
    patch_results = backend.run_activation_patching(patch_request)
    exact_scores = {r.hook_site: r.recovery_fraction for r in patch_results}

    # Align scores by site
    attr_vals = [attr_scores[s] for s in probe_sites]
    exact_vals = [exact_scores[s] for s in probe_sites]

    rho = _spearman(attr_vals, exact_vals)
    assert rho >= 0.8, (
        f"Spearman correlation {rho:.3f} < 0.8 — attribution ranking diverged from exact patching"
    )
