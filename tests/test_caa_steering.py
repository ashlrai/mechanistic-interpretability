"""Unit tests for the caa_steering experiment family.

All tests use synthetic activations and a fake generation function — no model
weights are loaded here. To run the real experiment against
Qwen/Qwen2.5-1.5B-Instruct (requires HF access and ~3 GB download), use:

    mech run --name caa-steering-qwen

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

from mech_interp.experiments.caa_steering import (  # noqa: E402
    REFUSAL_PHRASES,
    CAASteeringExperiment,
    CAASteeringSpec,
    ContrastivePairSpec,
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

D_MODEL = 16
LAYERS = [6, 8, 10]


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
        "hook_layers": LAYERS,
        "hook_site_template": "blocks.{L}.hook_resid_post",
        "contrastive_pairs": [
            {"a": "Harmful A", "b": "Harmless A", "label": "pair1"},
            {"a": "Harmful B", "b": "Harmless B", "label": "pair2"},
        ],
        "test_prompts": ["Test prompt one", "Test prompt two"],
        "steering_coefficient_range": [-1.0, 0.0, 1.0],
        "max_new_tokens": 10,
        "seed": 0,
        "device": "cpu",
    }
    params.update(overrides)
    return ExperimentSpec(
        name="caa-unit-test",
        family="caa_steering",
        backend="transformerlens",
        parameters=params,
    )


def _separated_acts(
    n: int, d: int = D_MODEL, offset: float = 5.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (acts_a, acts_b) well-separated along dim 0."""
    acts_a = torch.zeros(n, d)
    acts_a[:, 0] = offset
    acts_b = torch.zeros(n, d)
    acts_b[:, 0] = -offset
    return acts_a, acts_b


def _fake_activations(layers: list[int], n: int = 2, d: int = D_MODEL) -> tuple[
    dict[int, torch.Tensor], dict[int, torch.Tensor]
]:
    """Build per-layer activation dicts with known separation."""
    a: dict[int, torch.Tensor] = {}
    b: dict[int, torch.Tensor] = {}
    for L in layers:
        acts_a, acts_b = _separated_acts(n, d, offset=float(L))
        a[L] = acts_a
        b[L] = acts_b
    return a, b


def _fake_gen(refusal_at_positive: bool = True) -> Any:
    """Return a generation_fn(prompt, layer, coeff) -> str."""
    def fn(prompt: str, layer: int, coeff: float) -> str:  # noqa: ARG001
        if refusal_at_positive and coeff > 0:
            return "I cannot help with that."
        return "Sure, here is how."
    return fn


# ---------------------------------------------------------------------------
# Direction extraction math
# ---------------------------------------------------------------------------


def test_extract_direction_unit_norm() -> None:
    d = 8
    acts_a = torch.zeros(3, d)
    acts_a[:, 0] = 4.0
    acts_b = torch.zeros(3, d)
    direction, norm = _extract_direction(acts_a, acts_b)
    assert math.isclose(float(direction.norm().item()), 1.0, rel_tol=1e-5)
    assert math.isclose(float(direction[0].item()), 1.0, rel_tol=1e-5)
    assert math.isclose(norm, 4.0, rel_tol=1e-5)


def test_extract_direction_degenerate_no_crash() -> None:
    d = 4
    acts = torch.zeros(2, d)
    direction, norm = _extract_direction(acts, acts)
    assert direction.shape == (d,)
    assert norm < 1e-7


def test_extract_direction_known_values() -> None:
    acts_a = torch.tensor([[3.0, 0.0], [1.0, 0.0]])  # mean = [2, 0]
    acts_b = torch.zeros(2, 2)
    direction, _ = _extract_direction(acts_a, acts_b)
    assert math.isclose(float(direction[0].item()), 1.0, rel_tol=1e-5)
    assert math.isclose(float(direction[1].item()), 0.0, abs_tol=1e-6)


# ---------------------------------------------------------------------------
# Extraction quality
# ---------------------------------------------------------------------------


def test_extraction_quality_positive_for_separated() -> None:
    d = 4
    direction = torch.tensor([1.0, 0.0, 0.0, 0.0])
    acts_a = torch.zeros(3, d)
    acts_a[:, 0] = 5.0
    acts_b = torch.zeros(3, d)
    acts_b[:, 0] = -5.0
    q = _extraction_quality(acts_a, acts_b, direction)
    assert q > 0


def test_extraction_quality_zero_identical() -> None:
    d = 4
    direction = torch.tensor([1.0, 0.0, 0.0, 0.0])
    acts = torch.zeros(3, d)
    q = _extraction_quality(acts, acts, direction)
    assert q == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Refusal detection (shares REFUSAL_PHRASES constant)
# ---------------------------------------------------------------------------


def test_is_refusal_detects_all_phrases() -> None:
    for phrase in REFUSAL_PHRASES:
        assert _is_refusal(phrase)
        assert _is_refusal(phrase.upper())


def test_is_refusal_false_for_normal_text() -> None:
    assert not _is_refusal("Sure, here is how you do that.")
    assert not _is_refusal("Of course!")


# ---------------------------------------------------------------------------
# Direction extraction per layer
# ---------------------------------------------------------------------------


def test_directions_extracted_for_all_layers(tmp_path: Path) -> None:
    acts_a, acts_b = _fake_activations(LAYERS, n=2)
    exp = CAASteeringExperiment(
        activations_by_layer_a=acts_a,
        activations_by_layer_b=acts_b,
        generation_fn=_fake_gen(),
    )
    spec = _minimal_spec()
    run = _make_run(tmp_path, spec)
    result = exp.run(spec, run)

    assert result.status == RunStatus.SUCCEEDED

    # layer_effectiveness.json should have all layers
    eff_path = Path(result.artifacts["layer_effectiveness"])
    eff = json.loads(eff_path.read_text())
    for L in LAYERS:
        assert str(L) in eff["layers"], f"Layer {L} missing from layer_effectiveness"


def test_direction_normalisation_per_layer(tmp_path: Path) -> None:
    """Each per-layer direction must be unit-norm (saved to safetensors)."""
    pytest.importorskip("safetensors")
    from safetensors.torch import load_file

    acts_a, acts_b = _fake_activations(LAYERS, n=2)
    exp = CAASteeringExperiment(
        activations_by_layer_a=acts_a,
        activations_by_layer_b=acts_b,
        generation_fn=_fake_gen(),
    )
    result = exp.run(_minimal_spec(), _make_run(tmp_path, _minimal_spec()))
    loaded = load_file(result.artifacts["directions"])
    for L in LAYERS:
        key = f"layer_{L}"
        assert key in loaded, f"{key} not in safetensors"
        vec = loaded[key]
        norm = float(vec.norm().item())
        assert math.isclose(norm, 1.0, rel_tol=1e-4), f"Layer {L} direction norm={norm}"


# ---------------------------------------------------------------------------
# Intervention sweep completeness
# ---------------------------------------------------------------------------


def test_sweep_covers_all_layer_coeff_combinations(tmp_path: Path) -> None:
    """All (layer × coefficient) combos must appear in intervention_results.json."""
    acts_a, acts_b = _fake_activations(LAYERS, n=2)
    spec = _minimal_spec()
    exp = CAASteeringExperiment(
        activations_by_layer_a=acts_a,
        activations_by_layer_b=acts_b,
        generation_fn=_fake_gen(),
    )
    result = exp.run(spec, _make_run(tmp_path, spec))

    ir_path = Path(result.artifacts["intervention_results"])
    ir = json.loads(ir_path.read_text())

    layers_in_results = {entry["layer"] for entry in ir["results"]}
    coeffs_by_layer: dict[int, set[float]] = {}
    for entry in ir["results"]:
        layer = entry["layer"]
        coeffs_by_layer[layer] = {r["coefficient"] for r in entry["results"]}

    expected_coeffs = set(spec.parameters["steering_coefficient_range"])
    for L in LAYERS:
        assert L in layers_in_results, f"Layer {L} missing from results"
        assert coeffs_by_layer[L] == expected_coeffs, (
            f"Layer {L}: expected coeffs {expected_coeffs}, got {coeffs_by_layer[L]}"
        )


def test_sweep_generation_count(tmp_path: Path) -> None:
    """generation_fn is called exactly n_layers × n_coeffs × n_prompts times."""
    call_log: list[tuple[str, int, float]] = []

    def counting_gen(prompt: str, layer: int, coeff: float) -> str:
        call_log.append((prompt, layer, coeff))
        return "Sure."

    acts_a, acts_b = _fake_activations(LAYERS, n=2)
    spec = _minimal_spec()
    exp = CAASteeringExperiment(
        activations_by_layer_a=acts_a,
        activations_by_layer_b=acts_b,
        generation_fn=counting_gen,
    )
    exp.run(spec, _make_run(tmp_path, spec))

    n_layers = len(LAYERS)
    n_coeffs = len(spec.parameters["steering_coefficient_range"])
    n_prompts = len(spec.parameters["test_prompts"])
    assert len(call_log) == n_layers * n_coeffs * n_prompts


# ---------------------------------------------------------------------------
# Refusal-rate detection in sweep
# ---------------------------------------------------------------------------


def test_refusal_rate_increases_with_positive_coefficient(tmp_path: Path) -> None:
    acts_a, acts_b = _fake_activations([10], n=2)  # single layer
    spec = _minimal_spec(hook_layers=[10])
    exp = CAASteeringExperiment(
        activations_by_layer_a=acts_a,
        activations_by_layer_b=acts_b,
        generation_fn=_fake_gen(refusal_at_positive=True),
    )
    result = exp.run(spec, _make_run(tmp_path, spec))
    eff = json.loads(Path(result.artifacts["layer_effectiveness"]).read_text())
    sweep_10 = eff["layers"]["10"]["sweep"]
    rate_by_coeff = {s["coefficient"]: s["refusal_rate"] for s in sweep_10}
    # positive coeff → refusal phrase → rate == 1.0
    assert rate_by_coeff[1.0] == pytest.approx(1.0)
    # non-positive → no refusal phrase → rate == 0.0
    assert rate_by_coeff[0.0] == pytest.approx(0.0)
    assert rate_by_coeff[-1.0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------


def test_compute_metrics_picks_best_layer() -> None:
    layer_effectiveness = {
        6:  {"best_refusal_rate_shift": 0.2, "extraction_quality": 0.5, "best_coefficient": 1.0},
        10: {"best_refusal_rate_shift": 0.8, "extraction_quality": 1.5, "best_coefficient": 2.0},
    }
    metrics = _compute_metrics(layer_effectiveness)
    assert metrics["best_layer"] == pytest.approx(10.0)
    assert metrics["best_refusal_rate_shift"] == pytest.approx(0.8)
    assert metrics["best_extraction_quality"] == pytest.approx(1.5)
    assert metrics["n_layers_swept"] == pytest.approx(2.0)


def test_result_notes_contains_best_layer() -> None:
    layer_effectiveness = {
        6:  {"best_refusal_rate_shift": 0.1, "extraction_quality": 0.3, "best_coefficient": 1.0},
        10: {"best_refusal_rate_shift": 0.9, "extraction_quality": 2.0, "best_coefficient": 2.0},
    }
    notes = _result_notes(layer_effectiveness)
    assert "10" in notes
    assert "0.90" in notes


# ---------------------------------------------------------------------------
# Full experiment run — all artifacts present
# ---------------------------------------------------------------------------


def test_experiment_run_produces_all_artifacts(tmp_path: Path) -> None:
    acts_a, acts_b = _fake_activations(LAYERS, n=2)
    spec = _minimal_spec()
    exp = CAASteeringExperiment(
        activations_by_layer_a=acts_a,
        activations_by_layer_b=acts_b,
        generation_fn=_fake_gen(),
    )
    result = exp.run(spec, _make_run(tmp_path, spec))

    assert result.status == RunStatus.SUCCEEDED
    for key in ("directions", "layer_effectiveness", "intervention_results", "research_note"):
        assert key in result.artifacts, f"Missing artifact: {key}"
        assert Path(result.artifacts[key]).exists(), f"Artifact file missing: {key}"

    # research_note should contain the table
    note = Path(result.artifacts["research_note"]).read_text()
    assert "Layer-wise Effectiveness" in note
    assert "CAA Steering Report" in note


# ---------------------------------------------------------------------------
# Spec validation
# ---------------------------------------------------------------------------


def test_spec_rejects_empty_contrastive_pairs() -> None:
    with pytest.raises(Exception, match="contrastive_pairs must not be empty"):
        CAASteeringSpec.model_validate({
            "contrastive_pairs": [],
            "test_prompts": ["ok"],
        })


def test_spec_rejects_empty_test_prompts() -> None:
    with pytest.raises(Exception, match="test_prompts must not be empty"):
        CAASteeringSpec.model_validate({
            "contrastive_pairs": [{"a": "x", "b": "y"}],
            "test_prompts": [],
        })


def test_spec_rejects_empty_hook_layers() -> None:
    with pytest.raises(Exception, match="hook_layers must not be empty"):
        CAASteeringSpec.model_validate({
            "contrastive_pairs": [{"a": "x", "b": "y"}],
            "test_prompts": ["ok"],
            "hook_layers": [],
        })


def test_spec_accepts_valid_config() -> None:
    cfg = CAASteeringSpec.model_validate({
        "model": "gpt2",
        "hook_layers": [5, 10],
        "contrastive_pairs": [{"a": "harm", "b": "safe", "label": "test"}],
        "test_prompts": ["A question"],
    })
    assert cfg.model == "gpt2"
    assert cfg.hook_layers == [5, 10]
    assert cfg.max_new_tokens == 50  # default
    assert cfg.hook_site(5) == "blocks.5.hook_resid_post"
    assert cfg.hook_site(10) == "blocks.10.hook_resid_post"


def test_contrastive_pair_spec_fields() -> None:
    pair = ContrastivePairSpec(a="harmful prompt", b="harmless prompt", label="test")
    assert pair.a == "harmful prompt"
    assert pair.b == "harmless prompt"
    assert pair.label == "test"


# ---------------------------------------------------------------------------
# Runner dispatch
# ---------------------------------------------------------------------------


def test_runner_dispatches_caa_steering() -> None:
    spec = ExperimentSpec(
        name="caa",
        family="caa_steering",
        backend="transformerlens",
        parameters={
            "contrastive_pairs": [{"a": "harm", "b": "safe"}],
            "test_prompts": ["test"],
        },
    )
    exp = experiment_for_spec(spec)
    assert isinstance(exp, CAASteeringExperiment)


# ---------------------------------------------------------------------------
# Proposal generator
# ---------------------------------------------------------------------------


def test_caa_steering_proposal_generator(tmp_path: Path) -> None:
    from mech_interp.orchestration.proposal_generators import (
        PROPOSAL_GENERATORS,
        CAASteeringProposalGenerator,
    )

    eff: dict[str, Any] = {
        "model": "fake-model",
        "hook_site_template": "blocks.{L}.hook_resid_post",
        "hook_layers": [6, 8, 10, 12],
        "hidden_dim": 16,
        "contrastive_pair_count": 5,
        "layers": {
            "6":  {"extraction_quality": 0.5, "direction_norm": 1.0,
                   "baseline_refusal_rate": 0.0, "best_coefficient": 1.0,
                   "best_refusal_rate_shift": 0.3,
                   "sweep": [{"coefficient": 0.0, "refusal_rate": 0.0},
                              {"coefficient": 1.0, "refusal_rate": 0.3}]},
            "8":  {"extraction_quality": 0.9, "direction_norm": 1.0,
                   "baseline_refusal_rate": 0.0, "best_coefficient": 2.0,
                   "best_refusal_rate_shift": 0.6,
                   "sweep": [{"coefficient": 0.0, "refusal_rate": 0.0},
                              {"coefficient": 2.0, "refusal_rate": 0.6}]},
            "10": {"extraction_quality": 1.5, "direction_norm": 1.0,
                   "baseline_refusal_rate": 0.0, "best_coefficient": 2.0,
                   "best_refusal_rate_shift": 0.9,
                   "sweep": [{"coefficient": 0.0, "refusal_rate": 0.0},
                              {"coefficient": 2.0, "refusal_rate": 0.9}]},
            "12": {"extraction_quality": 0.7, "direction_norm": 1.0,
                   "baseline_refusal_rate": 0.0, "best_coefficient": 1.0,
                   "best_refusal_rate_shift": 0.4,
                   "sweep": [{"coefficient": 0.0, "refusal_rate": 0.0},
                              {"coefficient": 1.0, "refusal_rate": 0.4}]},
        },
    }
    (tmp_path / "layer_effectiveness.json").write_text(
        json.dumps(eff), encoding="utf-8"
    )
    (tmp_path / "spec.json").write_text(
        json.dumps({"parameters": {"model": "fake-model"}}), encoding="utf-8"
    )

    gen = CAASteeringProposalGenerator()
    proposals = gen.generate(tmp_path, limit=3)

    assert len(proposals) > 0
    for p in proposals:
        assert p["family"] == "circuit_patching"
        assert "blocks." in p["parameters"]["hook_sites"][0]

    # Best layer (10) should appear in at least one proposal
    all_sites = [s for p in proposals for s in p["parameters"].get("hook_sites", [])]
    assert any("10" in s for s in all_sites), "Expected layer 10 (best) in proposal sites"

    # Registered in global dict
    assert "caa_steering" in PROPOSAL_GENERATORS
    assert isinstance(PROPOSAL_GENERATORS["caa_steering"], CAASteeringProposalGenerator)


def test_caa_proposal_generator_graceful_on_missing_files(tmp_path: Path) -> None:
    from mech_interp.orchestration.proposal_generators import CAASteeringProposalGenerator
    proposals = CAASteeringProposalGenerator().generate(tmp_path)
    assert proposals == []
