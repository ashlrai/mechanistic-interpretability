"""Integration test: Crosscoder on gpt2 + distilgpt2 (both 768-dim).

Requires RUN_INTEGRATION_TESTS=1.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

PROMPTS = [
    "The Eiffel Tower is located in",
    "Photosynthesis converts sunlight into",
    "The first president of the United States was",
    "Water boils at 100 degrees",
    "Romeo and Juliet was written by",
]

HOOK_SITE = "blocks.5.hook_resid_post"


@pytest.mark.skipif(
    os.environ.get("RUN_INTEGRATION_TESTS") != "1",
    reason="Set RUN_INTEGRATION_TESTS=1 to run integration tests",
)
class TestCrosscoderE2E:
    @pytest.fixture(scope="class")
    def trained_result(self) -> Any:
        """Train a small crosscoder on gpt2+distilgpt2 and return (cc, history, analysis)."""
        import torch

        from mech_interp.backends import create_instrumented_backend
        from mech_interp.sae.crosscoder_analysis import compute_crosscoder_analysis
        from mech_interp.sae.crosscoder_trainer import train_crosscoder

        device = "cpu"
        source_backend = create_instrumented_backend(
            "transformerlens", {"model_name": "gpt2", "device": device}
        )
        target_backend = create_instrumented_backend(
            "transformerlens", {"model_name": "distilgpt2", "device": device}
        )

        source_captured = source_backend.capture_activations(PROMPTS, [HOOK_SITE])
        target_captured = target_backend.capture_activations(PROMPTS, [HOOK_SITE])

        src_tensor = source_captured[HOOK_SITE]
        tgt_tensor = target_captured[HOOK_SITE]

        # Flatten (batch, seq, d_model) → (n_tokens, d_model)
        batch, seq, d_model = src_tensor.shape
        src_flat = src_tensor.reshape(batch * seq, d_model).to(torch.float32)
        tgt_flat = tgt_tensor.reshape(batch * seq, d_model).to(torch.float32)
        prompt_for_token = [PROMPTS[i] for i in range(batch) for _ in range(seq)]

        cc, history = train_crosscoder(
            (src_flat, tgt_flat),
            n_features=32,
            k=4,
            epochs=2,
            batch_size=64,
            device=device,
            seed=42,
        )

        analysis = compute_crosscoder_analysis(
            cc,
            (src_flat, tgt_flat),
            prompt_for_token,
            model_specific_threshold=0.5,
        )

        return cc, history, analysis, src_flat, tgt_flat

    def test_d_model_both_768(self, trained_result: Any) -> None:
        cc, _, _, src_flat, tgt_flat = trained_result
        assert cc.input_dim == 768
        assert src_flat.shape[1] == 768
        assert tgt_flat.shape[1] == 768

    def test_loss_decreases_source(self, trained_result: Any) -> None:
        _, history, _, _, _ = trained_result
        # Combined loss (sum of both models' MSE) should decrease
        assert history.final_loss < history.initial_loss, (
            f"Expected combined loss to decrease; "
            f"initial={history.initial_loss:.4f}, final={history.final_loss:.4f}"
        )

    def test_loss_decreases_over_epochs(self, trained_result: Any) -> None:
        _, history, _, _, _ = trained_result
        assert len(history.losses_per_epoch) == 2
        # Monotone decrease not guaranteed in 2 epochs, but final should be < initial
        assert history.losses_per_epoch[-1] < history.initial_loss

    def test_analysis_has_features(self, trained_result: Any) -> None:
        _, _, analysis, _, _ = trained_result
        assert analysis.n_features == 32
        assert analysis.live_count + analysis.dead_count == 32

    def test_analysis_has_conserved_and_specific(self, trained_result: Any) -> None:
        """Verify conserved + specific partitions live features.

        gpt2 and distilgpt2 share architecture up to layer 5, so scores cluster
        near 0 (conserved) with short training. We verify the partition is exact
        and that at least some live features exist.
        """
        _, _, analysis, _, _ = trained_result
        assert analysis.live_count > 0, "Expected at least some live features"
        assert analysis.conserved_count + analysis.model_specific_count == analysis.live_count, (
            f"conserved={analysis.conserved_count} + specific={analysis.model_specific_count} "
            f"!= live={analysis.live_count}"
        )

    def test_model_scores_in_range(self, trained_result: Any) -> None:
        _, _, analysis, _, _ = trained_result
        for record in analysis.features:
            assert -1.0 <= record.model_score <= 1.0, (
                f"Feature {record.feature_index} model_score={record.model_score} out of [-1,1]"
            )

    def test_decoder_norms_non_negative(self, trained_result: Any) -> None:
        _, _, analysis, _, _ = trained_result
        for record in analysis.features:
            for norm in record.decoder_norm_per_model:
                assert norm >= 0.0

    def test_feature_count_correct(self, trained_result: Any) -> None:
        _, _, analysis, _, _ = trained_result
        assert len(analysis.features) == 32

    def test_conserved_specific_sum_to_live(self, trained_result: Any) -> None:
        _, _, analysis, _, _ = trained_result
        assert analysis.conserved_count + analysis.model_specific_count == analysis.live_count


@pytest.mark.skipif(
    os.environ.get("RUN_INTEGRATION_TESTS") != "1",
    reason="Set RUN_INTEGRATION_TESTS=1 to run integration tests",
)
def test_crosscoder_d_model_mismatch_raises() -> None:
    """Crosscoder should raise ValueError when models have different d_model."""
    import torch

    from mech_interp.sae.crosscoder_trainer import train_crosscoder

    with pytest.raises(ValueError, match="same shape"):
        train_crosscoder(
            (torch.randn(50, 768), torch.randn(50, 512)),  # mismatched d_model
            n_features=16,
            k=2,
        )


@pytest.mark.skipif(
    os.environ.get("RUN_INTEGRATION_TESTS") != "1",
    reason="Set RUN_INTEGRATION_TESTS=1 to run integration tests",
)
def test_crosscoder_experiment_full_pipeline(tmp_path: Path) -> None:
    """Run the full CrosscoderExperiment pipeline end-to-end via the runner."""
    from unittest.mock import MagicMock

    import torch

    from mech_interp.experiments.crosscoder import CrosscoderExperiment
    from mech_interp.types import ExperimentRun, ExperimentSpec, RunStatus

    batch, seq, d_model = 5, 10, 768
    hook = "blocks.5.hook_resid_post"

    def _make_mock_backend(model_name: str) -> MagicMock:
        backend = MagicMock()
        backend.capture_activations.return_value = {
            hook: torch.randn(batch, seq, d_model)
        }
        return backend

    spec = ExperimentSpec(
        name="test-cc",
        family="crosscoder",
        backend="transformerlens",
        parameters={
            "source_model": "gpt2",
            "target_model": "distilgpt2",
            "hook_site": hook,
            "n_features": 16,
            "k": 2,
            "epochs": 1,
            "batch_size": 64,
            "prompts": PROMPTS,
        },
    )
    from mech_interp.types import utc_now

    run = ExperimentRun(
        id=999,
        spec_name="test-cc",
        family="crosscoder",
        backend="transformerlens",
        status=RunStatus.RUNNING,
        artifact_dir=tmp_path,
        created_at=utc_now(),
    )
    # Patch artifact dir
    import mech_interp.storage.artifacts as art_mod
    original_resolve = art_mod.resolve_run_artifact_dir

    from mech_interp.types import ExperimentRun as _ER

    def _mock_resolve(_run: _ER) -> Path:
        return tmp_path

    art_mod.resolve_run_artifact_dir = _mock_resolve  # type: ignore[assignment]
    try:
        experiment = CrosscoderExperiment(
            source_backend=_make_mock_backend("gpt2"),
            target_backend=_make_mock_backend("distilgpt2"),
        )
        result = experiment.run(spec, run)
    finally:
        art_mod.resolve_run_artifact_dir = original_resolve

    assert result.status == RunStatus.SUCCEEDED
    assert result.metrics["n_features"] == 16.0
    assert result.metrics["final_loss"] < result.metrics["initial_loss"]
    assert "crosscoder_weights" in result.artifacts
    assert "feature_analysis" in result.artifacts
    assert "divergent_features" in result.artifacts
