"""End-to-end integration tests for attribution patching on gpt2-small.

Loads the real model once (session fixture from conftest.py).
Exercises the grad-cache path through TransformerLensBackend.run_with_grad_cache.

Assertions:
  (a) top-3 sites are non-zero and include early-layer residual sites.
  (b) ranking is deterministic across seeds.
  (c) at least one site has positive attribution and one negative (sign sensitivity).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from mech_interp.experiments.attribution_patching import AttributionPatchingExperiment
from mech_interp.types import ExperimentRun, ExperimentSpec, RunStatus, utc_now

pytestmark = pytest.mark.integration

_PROMPT_PAIRS = [
    {
        "id": "capital-france",
        "clean_prompt": "The capital of France is Paris",
        "corrupted_prompt": "The capital of France is Rome",
        "correct_token": " Paris",
        "incorrect_token": " Rome",
    },
    {
        "id": "capital-italy",
        "clean_prompt": "The capital of Italy is Rome",
        "corrupted_prompt": "The capital of Italy is Paris",
        "correct_token": " Rome",
        "incorrect_token": " Paris",
    },
    {
        "id": "tower-city",
        "clean_prompt": "The Eiffel Tower is in Paris",
        "corrupted_prompt": "The Eiffel Tower is in Rome",
        "correct_token": " Paris",
        "incorrect_token": " Rome",
    },
]

# Residual stream + MLP-out at layers 0-11
_HOOK_SITES = [
    {"site": "resid_pre", "layers": list(range(12))},
    {"site": "mlp_out", "layers": list(range(12))},
]


def _make_spec(seed: int = 42) -> ExperimentSpec:
    return ExperimentSpec(
        name="e2e-attribution-patching",
        family="attribution_patching",
        backend="transformerlens",
        description="",
        parameters={
            "model": "gpt2-small",
            "prompt_pairs": _PROMPT_PAIRS,
            "hook_sites": _HOOK_SITES,
            "seed": seed,
            "top_k": 10,
            "artifact_policy": {"write_report": True},
        },
    )


def _make_run(tmp_path: Path, run_id: int = 1) -> ExperimentRun:
    spec = _make_spec()
    return ExperimentRun(
        id=run_id,
        spec_name=spec.name,
        family=spec.family,
        backend=spec.backend,
        status=RunStatus.RUNNING,
        artifact_dir=tmp_path,
        created_at=utc_now(),
    )


def test_attribution_top3_sites_nonzero_and_include_early_layers(
    gpt2_backend: Any, tmp_path: Path
) -> None:
    """(a) Top-3 sites have non-zero attribution; at least one is an early-layer resid site."""
    spec = _make_spec()
    run = _make_run(tmp_path)
    result = AttributionPatchingExperiment(backend=gpt2_backend).run(spec, run)

    assert result.status == RunStatus.SUCCEEDED, f"Run failed: {result.notes}"

    ranked = json.loads(
        Path(result.artifacts["attribution_ranked_json"]).read_text(encoding="utf-8")
    )
    assert len(ranked) >= 3, "Expected at least 3 ranked sites"

    top3 = ranked[:3]
    for row in top3:
        assert row["abs_attribution_score"] > 0.0, (
            f"Top-3 site {row['hook_site']} has zero abs attribution"
        )

    # At least one of the top-3 should be an early-layer (layers 0-5) residual site
    early_resid = [
        r for r in top3
        if "hook_resid_pre" in r["hook_site"]
        and any(f"blocks.{i}." in r["hook_site"] for i in range(6))
    ]
    assert early_resid, (
        f"Expected at least one early-layer resid_pre site in top-3; got "
        f"{[r['hook_site'] for r in top3]}"
    )


def test_attribution_ranking_deterministic_across_seeds(
    gpt2_backend: Any, tmp_path: Path
) -> None:
    """(b) Ranking is deterministic — same order for seed=42 and seed=99."""

    def _run_attribution(seed: int, subdir: str) -> list[str]:
        spec = _make_spec(seed=seed)
        run = ExperimentRun(
            id=1,
            spec_name=spec.name,
            family=spec.family,
            backend=spec.backend,
            status=RunStatus.RUNNING,
            artifact_dir=tmp_path / subdir,
            created_at=utc_now(),
        )
        result = AttributionPatchingExperiment(backend=gpt2_backend).run(spec, run)
        assert result.status == RunStatus.SUCCEEDED
        ranked = json.loads(
            Path(result.artifacts["attribution_ranked_json"]).read_text(encoding="utf-8")
        )
        return [r["hook_site"] for r in ranked[:10]]

    order_42 = _run_attribution(42, "seed42")
    order_99 = _run_attribution(99, "seed99")

    assert order_42 == order_99, (
        f"Attribution ranking differed across seeds.\n"
        f"seed=42 top-10: {order_42}\n"
        f"seed=99 top-10: {order_99}"
    )


def test_attribution_sign_sensitivity(gpt2_backend: Any, tmp_path: Path) -> None:
    """(c) Attribution scores include both positive and negative values.

    This verifies the signed dot product is working correctly — if all scores
    have the same sign the gradient or difference term is degenerate.
    """
    spec = _make_spec()
    run = _make_run(tmp_path)
    result = AttributionPatchingExperiment(backend=gpt2_backend).run(spec, run)

    assert result.status == RunStatus.SUCCEEDED
    ranked = json.loads(
        Path(result.artifacts["attribution_ranked_json"]).read_text(encoding="utf-8")
    )
    attribution_scores = [r["attribution_score"] for r in ranked]

    positives = [s for s in attribution_scores if s > 0]
    negatives = [s for s in attribution_scores if s < 0]

    assert positives, "Expected at least one positive attribution score"
    assert negatives, "Expected at least one negative attribution score (sign sensitivity)"


def test_attribution_summary_artifact_schema(gpt2_backend: Any, tmp_path: Path) -> None:
    """Summary JSON has the expected top-level keys and sensible values."""
    spec = _make_spec()
    run = _make_run(tmp_path)
    result = AttributionPatchingExperiment(backend=gpt2_backend).run(spec, run)

    assert result.status == RunStatus.SUCCEEDED
    summary = json.loads(
        Path(result.artifacts["attribution_summary"]).read_text(encoding="utf-8")
    )
    assert summary["model"] == "gpt2-small"
    assert summary["prompt_pair_count"] == 3
    assert summary["hook_site_count"] == 24  # 12 resid_pre + 12 mlp_out
    assert summary["mean_abs_attribution"] > 0.0
    assert len(summary["top_k_sites"]) <= 10


def test_attribution_report_flags_top_sites(gpt2_backend: Any, tmp_path: Path) -> None:
    """Research note exists and flags top-K sites with YES in the table."""
    spec = _make_spec()
    run = _make_run(tmp_path)
    result = AttributionPatchingExperiment(backend=gpt2_backend).run(spec, run)

    assert result.status == RunStatus.SUCCEEDED
    assert "research_note" in result.artifacts

    report = Path(result.artifacts["research_note"]).read_text(encoding="utf-8")
    assert "YES" in report  # at least one site flagged for follow-up
    assert "approximation" in report.lower()
