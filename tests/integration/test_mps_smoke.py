"""MPS smoke test: verify SAE and ACDC-lite pipelines work on Apple Silicon.

Skipped automatically on Linux/Windows (or any machine without a working MPS
backend) so this never blocks CI. On the developer's MBP it validates that:

  - Activations come back in float32 (MPS can silently return fp16/bfloat16).
  - SAE training converges (MSE decreases) on the GPU.
  - ACDC-lite produces at least one node with non-zero importance on MPS.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.integration


def _mps_available() -> bool:
    try:
        import torch

        return bool(torch.backends.mps.is_available() and torch.backends.mps.is_built())
    except Exception:
        return False


def _have_interp_deps() -> bool:
    try:
        import transformer_lens  # noqa: F401
    except ImportError:
        return False
    return True


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mps_backend() -> object:
    """Load gpt2-small on MPS once per module."""
    if not _mps_available():
        pytest.skip("MPS not available on this machine (not Apple Silicon or MPS not built)")
    if not _have_interp_deps():
        pytest.skip("transformer-lens not installed; run `uv sync --extra interp`")

    from mech_interp.backends.instrumented import TransformerLensBackend

    backend = TransformerLensBackend(model_name="gpt2", device="mps")
    backend.load()
    return backend


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SAE_PROMPTS = [
    "The Eiffel Tower is in Paris.",
    "The Colosseum is in Rome.",
    "Big Ben is in London.",
    "The Great Wall is in China.",
    "The Statue of Liberty is in New York.",
    "Mount Fuji is in Japan.",
    "The Taj Mahal is in India.",
    "The Pyramids are in Egypt.",
    "The Louvre is in Paris.",
    "Niagara Falls is on the US-Canada border.",
]

_ACDC_PAIRS = [
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
        "id": "capital-uk",
        "clean_prompt": "The capital of England is London",
        "corrupted_prompt": "The capital of England is Paris",
        "correct_token": " London",
        "incorrect_token": " Paris",
    },
]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_mps_model_parameters_on_device(mps_backend: object) -> None:
    """Sanity-check: the loaded model's parameters must live on the MPS device."""
    import torch

    model = getattr(mps_backend, "model", None)
    assert model is not None, "Backend did not load a model"

    first_param = next(iter(model.parameters()))
    assert first_param.device.type == "mps", (
        f"Expected model parameters on 'mps', got '{first_param.device.type}'"
    )
    # Confirm float32 — TransformerLens default; MPS shouldn't silently downcast.
    assert first_param.dtype == torch.float32, (
        f"Expected model weights in float32, got {first_param.dtype}"
    )


def test_mps_activations_are_float32(mps_backend: object) -> None:
    """Captured activations must be float32 after the backend's defensive cast."""
    import torch

    from mech_interp.backends.instrumented import TransformerLensBackend

    assert isinstance(mps_backend, TransformerLensBackend)
    captured = mps_backend.capture_activations(
        ["Hello world", "Foo bar baz"],
        ["blocks.0.hook_resid_pre"],
    )
    assert "blocks.0.hook_resid_pre" in captured, "Hook site not in captured activations"
    tensor = captured["blocks.0.hook_resid_pre"]
    assert tensor.dtype == torch.float32, (
        f"Activation dtype should be float32 on MPS, got {tensor.dtype}"
    )
    assert tensor.device.type == "mps", (
        f"Activation should still live on MPS, got {tensor.device.type}"
    )


def test_mps_sae_training_converges(mps_backend: object, tmp_path: Path) -> None:
    """SAE training on MPS must complete with decreasing MSE."""
    from mech_interp.experiments.polysemanticity_sae import PolysemanticitySAEExperiment
    from mech_interp.types import ExperimentRun, ExperimentSpec, RunStatus, utc_now

    spec = ExperimentSpec(
        name="mps-sae-smoke",
        family="polysemanticity_sae",
        backend="transformerlens",
        description="MPS smoke test",
        parameters={
            "model": "gpt2-small",
            "hook_site": "blocks.0.hook_resid_pre",
            "n_features": 32,
            "k": 4,
            "epochs": 2,
            "batch_size": 64,
            "learning_rate": 1e-3,
            "seed": 42,
            "device": "mps",
            "prompts": _SAE_PROMPTS,
            "artifact_policy": {
                "retain_weights": True,
                "write_feature_analysis": True,
                "top_prompts_per_feature": 2,
            },
        },
    )
    run = ExperimentRun(
        id=1,
        spec_name=spec.name,
        family=spec.family,
        backend=spec.backend,
        status=RunStatus.RUNNING,
        artifact_dir=tmp_path,
        created_at=utc_now(),
    )

    from mech_interp.backends.instrumented import TransformerLensBackend
    assert isinstance(mps_backend, TransformerLensBackend)
    result = PolysemanticitySAEExperiment(backend=mps_backend).run(spec, run)

    assert result.status == RunStatus.SUCCEEDED, f"SAE run failed: {result.notes}"
    metrics = result.metrics

    # Reconstruction MSE must decrease — if MPS produces NaN/wrong dtypes this fails.
    assert metrics["final_loss"] < metrics["initial_loss"], (
        f"MSE did not decrease on MPS: initial={metrics['initial_loss']:.4f} "
        f"final={metrics['final_loss']:.4f}"
    )
    # Some features must be alive — zero live features means dtype/NaN corruption.
    assert metrics["live_features"] > 0, (
        "All SAE features are dead on MPS — likely a dtype or NaN corruption"
    )
    # mean_features_per_token should equal k=4.
    assert metrics["mean_features_per_token"] == pytest.approx(4.0, rel=0.01), (
        f"Unexpected mean features per token: {metrics['mean_features_per_token']}"
    )


def test_mps_acdc_lite_produces_nonzero_importance(
    mps_backend: object, tmp_path: Path
) -> None:
    """ACDC-lite on MPS must find at least one node with non-zero importance."""
    from mech_interp.experiments.acdc_lite import ACDCLiteExperiment
    from mech_interp.types import ExperimentRun, ExperimentSpec, RunStatus, utc_now

    spec = ExperimentSpec(
        name="mps-acdc-smoke",
        family="acdc_lite",
        backend="transformerlens",
        description="MPS smoke test",
        parameters={
            "model": "gpt2-small",
            "prompt_pairs": _ACDC_PAIRS,
            "layers": [0],
            "include_attention": True,
            "include_mlps": False,
            "tau": 0.001,
            "max_iterations": 2,
            "ablation_type": "mean",
            "seed": 42,
            "device": "mps",
        },
    )
    run = ExperimentRun(
        id=2,
        spec_name=spec.name,
        family=spec.family,
        backend=spec.backend,
        status=RunStatus.RUNNING,
        artifact_dir=tmp_path,
        created_at=utc_now(),
    )

    from mech_interp.backends.instrumented import TransformerLensBackend
    assert isinstance(mps_backend, TransformerLensBackend)
    result = ACDCLiteExperiment(backend=mps_backend).run(spec, run)

    assert result.status == RunStatus.SUCCEEDED, f"ACDC run failed: {result.notes}"
    metrics = result.metrics

    # Layer 0, attention-only → 12 heads = 12 candidate nodes.
    assert metrics["candidate_node_count"] == 12, (
        f"Expected 12 candidate nodes (L0 attention heads), got {metrics['candidate_node_count']}"
    )
    # At least one node must have non-zero importance — otherwise ablations had no
    # effect, which would indicate MPS is silently zeroing intermediate tensors.
    assert metrics["top_node_importance"] > 0.0, (
        "Top ACDC node has zero importance on MPS — ablation hooks may be broken"
    )
    # Faithfulness must be a valid probability.
    assert 0.0 <= metrics["faithfulness"] <= 1.0, (
        f"Faithfulness out of range: {metrics['faithfulness']}"
    )
