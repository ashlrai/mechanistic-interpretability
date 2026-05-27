"""Unit tests for the refusal_direction experiment family.

All tests use synthetic activations and a fake generation function — no model
weights are loaded here.  To run the real experiment against Qwen/Qwen2.5-1.5B-Instruct
(requires HF access and ~3 GB download) use:

    mech run --name refusal-direction-qwen

or:

    python -m pytest tests/integration/  # with RUN_INTEGRATION_TESTS=1 or --extra interp

DO NOT add an integration test that loads Qwen in this file.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

torch = pytest.importorskip("torch", reason="torch not installed; run with --extra interp")

if TYPE_CHECKING:
    import torch  # noqa: F811

from mech_interp.experiments.refusal_direction import (  # noqa: E402
    REFUSAL_PHRASES,
    RefusalDirectionExperiment,
    RefusalDirectionSpec,
    _compute_metrics,
    _extract_direction,
    _extraction_quality,
    _is_refusal,
    _result_notes,
)
from mech_interp.orchestration.runner import experiment_for_spec  # noqa: E402
from mech_interp.types import ExperimentRun, ExperimentSpec, RunStatus, utc_now  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_run(tmp_path: Path, spec: ExperimentSpec, run_id: int = 1) -> ExperimentRun:
    return ExperimentRun(
        id=run_id,
        spec_name=spec.name,
        family=spec.family,
        backend=spec.backend,
        status=RunStatus.RUNNING,
        artifact_dir=tmp_path,
        created_at=utc_now(),
    )


def _minimal_spec(**overrides: Any) -> ExperimentSpec:
    params: dict[str, Any] = {
        "model": "fake-model",
        "hook_site": "blocks.5.hook_resid_post",
        "harmful_prompts": ["Harmful prompt A", "Harmful prompt B"],
        "harmless_prompts": ["Harmless prompt A", "Harmless prompt B"],
        "test_prompts": ["Test prompt"],
        "steering_coefficient_range": [-1.0, 0.0, 1.0],
        "max_new_tokens": 10,
        "seed": 0,
        "device": "cpu",
    }
    params.update(overrides)
    return ExperimentSpec(
        name="refusal-unit-test",
        family="refusal_direction",
        backend="transformerlens",
        parameters=params,
    )


def _known_acts(n: int, d: int, offset: float = 0.0) -> torch.Tensor:
    """Return (n, d) tensor with constant rows offset along dim-0."""
    base = torch.zeros(n, d)
    base[:, 0] = offset
    return base


# ---------------------------------------------------------------------------
# Direction extraction math
# ---------------------------------------------------------------------------


def test_extract_direction_unit_norm() -> None:
    d = 8
    harmful = torch.zeros(3, d)
    harmless = torch.zeros(3, d)
    harmful[:, 0] = 2.0  # mean_harmful[:,0] = 2
    # mean diff = [2, 0, ...]; after normalisation -> [1, 0, ...]
    direction, norm = _extract_direction(harmful, harmless)
    assert math.isclose(float(direction.norm().item()), 1.0, rel_tol=1e-5)
    assert math.isclose(float(direction[0].item()), 1.0, rel_tol=1e-5)
    assert math.isclose(norm, 2.0, rel_tol=1e-5)


def test_extract_direction_degenerate_returns_without_crash() -> None:
    d = 4
    acts = torch.zeros(2, d)
    direction, norm = _extract_direction(acts, acts)
    assert direction.shape == (d,)
    assert norm < 1e-7


def test_extract_direction_known_values() -> None:
    harmful = torch.tensor([[3.0, 0.0], [1.0, 0.0]])   # mean = [2, 0]
    harmless = torch.tensor([[0.0, 0.0], [0.0, 0.0]])  # mean = [0, 0]
    direction, _ = _extract_direction(harmful, harmless)
    assert math.isclose(float(direction[0].item()), 1.0, rel_tol=1e-5)
    assert math.isclose(float(direction[1].item()), 0.0, abs_tol=1e-6)


# ---------------------------------------------------------------------------
# Extraction quality metric
# ---------------------------------------------------------------------------


def test_extraction_quality_positive_for_separated_activations() -> None:
    d = 4
    # Direction pointing along dim 0
    direction = torch.tensor([1.0, 0.0, 0.0, 0.0])
    harmful = torch.zeros(3, d)
    harmful[:, 0] = 5.0
    harmless = torch.zeros(3, d)
    harmless[:, 0] = -5.0
    q = _extraction_quality(harmful, harmless, direction)
    assert q > 0, f"Expected positive quality, got {q}"


def test_extraction_quality_zero_for_indistinguishable() -> None:
    d = 4
    direction = torch.tensor([1.0, 0.0, 0.0, 0.0])
    acts = torch.zeros(3, d)
    # Both classes identical → margin = 0
    q = _extraction_quality(acts, acts, direction)
    assert q == pytest.approx(0.0, abs=1e-6)


def test_extraction_quality_negative_when_reversed() -> None:
    d = 4
    direction = torch.tensor([1.0, 0.0, 0.0, 0.0])
    harmful = torch.zeros(3, d)
    harmful[:, 0] = -5.0  # harmful projects LOWER than harmless
    harmless = torch.zeros(3, d)
    harmless[:, 0] = 5.0
    q = _extraction_quality(harmful, harmless, direction)
    assert q < 0


# ---------------------------------------------------------------------------
# Refusal detection
# ---------------------------------------------------------------------------


def test_is_refusal_detects_all_phrases() -> None:
    for phrase in REFUSAL_PHRASES:
        assert _is_refusal(phrase), f"Expected '{phrase}' to be detected as refusal"
        assert _is_refusal(phrase.upper()), "Detection should be case-insensitive"


def test_is_refusal_false_for_normal_text() -> None:
    assert not _is_refusal("Sure, here is how you do that.")
    assert not _is_refusal("Of course! Let me explain.")


def test_is_refusal_partial_match() -> None:
    assert _is_refusal("I cannot assist with that request.")
    assert _is_refusal("I'm sorry, but I can't do that.")


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------


def test_compute_metrics_extracts_baseline_and_max_shift() -> None:
    results = [
        {"coefficient": -1.0, "refusal_rate": 0.0, "prompts": []},
        {"coefficient": 0.0, "refusal_rate": 0.5, "prompts": []},
        {"coefficient": 1.0, "refusal_rate": 1.0, "prompts": []},
    ]
    metrics = _compute_metrics(extraction_quality=0.8, intervention_results=results)
    assert metrics["baseline_refusal_rate"] == pytest.approx(0.5)
    assert metrics["max_refusal_rate_shift"] == pytest.approx(0.5)
    assert metrics["extraction_quality"] == pytest.approx(0.8)
    assert metrics["steering_coefficient_count"] == 3.0


def test_result_notes_contains_key_values() -> None:
    results = [
        {"coefficient": 0.0, "refusal_rate": 0.33, "prompts": []},
        {"coefficient": 1.0, "refusal_rate": 1.0, "prompts": []},
    ]
    notes = _result_notes(extraction_quality=1.5, intervention_results=results)
    assert "1.500" in notes
    assert "0.33" in notes


# ---------------------------------------------------------------------------
# Full experiment run (injected activations + fake generation)
# ---------------------------------------------------------------------------


def _make_well_separated_acts(
    n: int, d: int = 16
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return harmful / harmless tensors with clear separation along dim-0."""
    harmful = torch.zeros(n, d)
    harmful[:, 0] = 5.0
    harmless = torch.zeros(n, d)
    harmless[:, 0] = -5.0
    return harmful, harmless


def test_experiment_run_produces_all_artifacts(tmp_path: Path) -> None:
    d = 16
    harmful, harmless = _make_well_separated_acts(2, d)

    call_count: dict[str, int] = {"n": 0}

    def fake_gen(prompt: str, coeff: float) -> str:  # noqa: ARG001
        call_count["n"] += 1
        return "I cannot help with that." if coeff > 0 else "Sure, here is how."

    spec = _minimal_spec()
    run = _make_run(tmp_path, spec)

    exp = RefusalDirectionExperiment(
        activations_harmful=harmful,
        activations_harmless=harmless,
        generation_fn=fake_gen,
    )
    result = exp.run(spec, run)

    assert result.status == RunStatus.SUCCEEDED

    # Artifacts present
    assert "direction" in result.artifacts
    assert "direction_sidecar" in result.artifacts
    assert "intervention_results" in result.artifacts
    assert "research_note" in result.artifacts

    # Sidecar is well-formed
    sidecar = json.loads(Path(result.artifacts["direction_sidecar"]).read_text())
    assert sidecar["hidden_dim"] == d
    assert sidecar["hook_site"] == "blocks.5.hook_resid_post"
    assert sidecar["model"] == "fake-model"
    assert isinstance(sidecar["extraction_quality"], float)

    # Intervention results structure
    ir = json.loads(Path(result.artifacts["intervention_results"]).read_text())
    assert ir["baseline_refusal_rate"] is not None
    coeffs = [r["coefficient"] for r in ir["results"]]
    assert coeffs == [-1.0, 0.0, 1.0]
    # Positive coeff → refusal (fake_gen returns refusal phrase)
    pos = next(r for r in ir["results"] if r["coefficient"] == 1.0)
    assert pos["refusal_rate"] == pytest.approx(1.0)
    # Zero coeff → no refusal phrase
    zero = next(r for r in ir["results"] if r["coefficient"] == 0.0)
    assert zero["refusal_rate"] == pytest.approx(0.0)
    # Shift is computed correctly
    assert pos["refusal_rate_shift"] == pytest.approx(1.0)

    # Research note exists and is non-trivial
    note = Path(result.artifacts["research_note"]).read_text()
    assert "Refusal Direction Report" in note
    assert "Steering Sweep Results" in note

    # Generation function called once per (coeff, test_prompt) pair
    n_coeffs = len(spec.parameters["steering_coefficient_range"])
    n_prompts = len(spec.parameters["test_prompts"])
    assert call_count["n"] == n_coeffs * n_prompts


def test_experiment_metrics_positive_extraction_quality(tmp_path: Path) -> None:
    d = 16
    harmful, harmless = _make_well_separated_acts(3, d)

    exp = RefusalDirectionExperiment(
        activations_harmful=harmful,
        activations_harmless=harmless,
        generation_fn=lambda p, c: "I cannot do that.",
    )
    result = exp.run(_minimal_spec(), _make_run(tmp_path, _minimal_spec()))
    assert result.metrics["extraction_quality"] > 0.0


def test_experiment_direction_artifact_is_safetensors(tmp_path: Path) -> None:
    """If safetensors is installed, the direction file must be loadable."""
    pytest.importorskip("safetensors")
    from safetensors.torch import load_file

    d = 16
    harmful, harmless = _make_well_separated_acts(2, d)
    exp = RefusalDirectionExperiment(
        activations_harmful=harmful,
        activations_harmless=harmless,
        generation_fn=lambda p, c: "Sure.",
    )
    result = exp.run(_minimal_spec(), _make_run(tmp_path, _minimal_spec()))
    loaded = load_file(result.artifacts["direction"])
    assert "direction" in loaded
    assert loaded["direction"].shape == (d,)
    # Must be unit norm
    assert math.isclose(float(loaded["direction"].norm().item()), 1.0, rel_tol=1e-5)


# ---------------------------------------------------------------------------
# Spec validation
# ---------------------------------------------------------------------------


def test_spec_rejects_empty_prompt_lists() -> None:
    with pytest.raises(Exception, match="prompt list must not be empty"):
        RefusalDirectionSpec.model_validate({
            "harmful_prompts": [],
            "harmless_prompts": ["ok"],
            "test_prompts": ["ok"],
        })


def test_spec_rejects_blank_individual_prompt() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        RefusalDirectionSpec.model_validate({
            "harmful_prompts": ["  "],
            "harmless_prompts": ["ok"],
            "test_prompts": ["ok"],
        })


def test_spec_accepts_valid_config() -> None:
    cfg = RefusalDirectionSpec.model_validate({
        "model": "gpt2",
        "hook_site": "blocks.5.hook_resid_post",
        "harmful_prompts": ["Do harm"],
        "harmless_prompts": ["Be nice"],
        "test_prompts": ["A question"],
    })
    assert cfg.model == "gpt2"
    assert cfg.max_new_tokens == 50  # default


# ---------------------------------------------------------------------------
# Runner dispatch
# ---------------------------------------------------------------------------


def test_runner_dispatches_refusal_direction(monkeypatch: pytest.MonkeyPatch) -> None:
    spec = ExperimentSpec(
        name="rd",
        family="refusal_direction",
        backend="transformerlens",
        parameters={
            "harmful_prompts": ["harm"],
            "harmless_prompts": ["safe"],
            "test_prompts": ["test"],
        },
    )
    exp = experiment_for_spec(spec)
    assert isinstance(exp, RefusalDirectionExperiment)


# ---------------------------------------------------------------------------
# Proposal generator — refusal direction follow-ups
# ---------------------------------------------------------------------------


def test_refusal_direction_generator_emits_circuit_patching_proposals(
    tmp_path: Path,
) -> None:
    from mech_interp.orchestration.proposal_generators import (
        PROPOSAL_GENERATORS,
        RefusalDirectionProposalGenerator,
    )

    # Write a minimal direction_sidecar and intervention_results
    sidecar = {
        "model": "fake-model",
        "hook_site": "blocks.10.hook_resid_post",
        "hidden_dim": 16,
        "direction_norm": 1.0,
        "extraction_quality": 2.5,
        "harmful_prompt_count": 3,
        "harmless_prompt_count": 3,
    }
    (tmp_path / "direction.safetensors.json").write_text(
        json.dumps(sidecar), encoding="utf-8"
    )
    ir = {
        "model": "fake-model",
        "hook_site": "blocks.10.hook_resid_post",
        "results": [
            {
                "coefficient": -1.0,
                "refusal_rate": 0.0,
                "refusal_rate_shift": -0.5,
                "prompts": [],
            },
            {
                "coefficient": 0.0,
                "refusal_rate": 0.5,
                "refusal_rate_shift": 0.0,
                "prompts": [],
            },
            {
                "coefficient": 1.0,
                "refusal_rate": 1.0,
                "refusal_rate_shift": 0.5,
                "prompts": [],
            },
        ],
    }
    (tmp_path / "intervention_results.json").write_text(
        json.dumps(ir), encoding="utf-8"
    )

    gen = RefusalDirectionProposalGenerator()
    proposals = gen.generate(tmp_path, limit=3)

    assert len(proposals) > 0
    for p in proposals:
        assert p["family"] == "circuit_patching"
    # Should reference hook sites near layer 10
    hook_sites_used = [
        site
        for p in proposals
        for site in p["parameters"].get("hook_sites", [])
    ]
    assert any("blocks." in s for s in hook_sites_used)

    # Registered in global dict
    assert "refusal_direction" in PROPOSAL_GENERATORS
    assert isinstance(PROPOSAL_GENERATORS["refusal_direction"], RefusalDirectionProposalGenerator)


def test_refusal_direction_generator_graceful_on_missing_files(tmp_path: Path) -> None:
    from mech_interp.orchestration.proposal_generators import RefusalDirectionProposalGenerator

    proposals = RefusalDirectionProposalGenerator().generate(tmp_path)
    assert proposals == []
