"""Unit tests for the Crosscoder model, trainer, and analysis."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import pytest
import torch

if TYPE_CHECKING:
    from torch import Tensor

    from mech_interp.sae.crosscoder import Crosscoder

# ---------------------------------------------------------------------------
# Crosscoder model
# ---------------------------------------------------------------------------


def _make_crosscoder(
    n_models: int = 2, input_dim: int = 16, n_features: int = 8, k: int = 2
) -> Crosscoder:
    from mech_interp.sae.crosscoder import Crosscoder

    return Crosscoder(n_models=n_models, input_dim=input_dim, n_features=n_features, k=k)


class TestCrosscoderInit:
    def test_basic_init(self) -> None:
        from mech_interp.sae.crosscoder import Crosscoder

        cc = Crosscoder(n_models=2, input_dim=16, n_features=8, k=2)
        assert cc.n_models == 2
        assert cc.input_dim == 16
        assert cc.n_features == 8
        assert cc.k == 2
        assert len(cc.decoders) == 2

    def test_encoder_input_dim(self) -> None:
        from mech_interp.sae.crosscoder import Crosscoder

        cc = Crosscoder(n_models=3, input_dim=10, n_features=4, k=1)
        assert cc.encoder.in_features == 30  # n_models * input_dim

    def test_decoder_output_dim(self) -> None:
        from mech_interp.sae.crosscoder import Crosscoder

        cc = Crosscoder(n_models=2, input_dim=16, n_features=8, k=2)
        for dec in cc.decoders:
            assert dec.out_features == 16

    def test_invalid_n_models(self) -> None:
        from mech_interp.sae.crosscoder import Crosscoder

        with pytest.raises(ValueError, match="n_models must be >= 2"):
            Crosscoder(n_models=1, input_dim=8, n_features=4, k=1)

    def test_invalid_k(self) -> None:
        from mech_interp.sae.crosscoder import Crosscoder

        with pytest.raises(ValueError, match="k must be"):
            Crosscoder(n_models=2, input_dim=8, n_features=4, k=0)

    def test_k_exceeds_features(self) -> None:
        from mech_interp.sae.crosscoder import Crosscoder

        with pytest.raises(ValueError, match="k must be"):
            Crosscoder(n_models=2, input_dim=8, n_features=4, k=5)

    def test_invalid_input_dim(self) -> None:
        from mech_interp.sae.crosscoder import Crosscoder

        with pytest.raises(ValueError, match="input_dim"):
            Crosscoder(n_models=2, input_dim=0, n_features=4, k=1)


class TestCrosscoderForward:
    def test_encode_returns_correct_shapes(self) -> None:
        cc = _make_crosscoder(n_models=2, input_dim=16, n_features=8, k=2)
        batch = 5
        acts = (torch.randn(batch, 16), torch.randn(batch, 16))
        codes, pre = cc.encode(acts)
        assert codes.shape == (batch, 8)
        assert pre.shape == (batch, 8)

    def test_encode_sparsity(self) -> None:
        """Top-K codes should have exactly k non-zero values per row."""
        cc = _make_crosscoder(n_models=2, input_dim=16, n_features=8, k=3)
        acts = (torch.randn(10, 16), torch.randn(10, 16))
        codes, _ = cc.encode(acts)
        nonzero_per_row = (codes != 0).sum(dim=-1)
        assert (nonzero_per_row == 3).all()

    def test_decode_returns_tuple(self) -> None:
        cc = _make_crosscoder(n_models=2, input_dim=16, n_features=8, k=2)
        codes = torch.randn(5, 8)
        recons = cc.decode(codes)
        assert len(recons) == 2
        for r in recons:
            assert r.shape == (5, 16)

    def test_forward_reconstruction_shapes(self) -> None:
        cc = _make_crosscoder(n_models=2, input_dim=16, n_features=8, k=2)
        acts = (torch.randn(7, 16), torch.randn(7, 16))
        recons, codes = cc(acts)
        assert len(recons) == 2
        assert codes.shape == (7, 8)
        for r in recons:
            assert r.shape == (7, 16)

    def test_wrong_number_of_activations(self) -> None:
        cc = _make_crosscoder(n_models=2, input_dim=16, n_features=8, k=2)
        with pytest.raises(ValueError, match="Expected 2 activation tensors"):
            cc.encode((torch.randn(5, 16),))

    def test_three_models(self) -> None:
        from mech_interp.sae.crosscoder import Crosscoder

        cc = Crosscoder(n_models=3, input_dim=8, n_features=4, k=1)
        acts = (torch.randn(3, 8), torch.randn(3, 8), torch.randn(3, 8))
        recons, codes = cc(acts)
        assert len(recons) == 3
        assert codes.shape == (3, 4)

    def test_state_dict_keys(self) -> None:
        cc = _make_crosscoder(n_models=2, input_dim=8, n_features=4, k=1)
        keys = set(cc.state_dict().keys())
        assert "encoder.weight" in keys
        assert "encoder.bias" in keys
        assert "decoder_0.weight" in keys
        assert "decoder_1.weight" in keys


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class TestCrosscoderTrainer:
    def test_train_reduces_loss(self) -> None:
        from mech_interp.sae.crosscoder_trainer import train_crosscoder

        torch.manual_seed(0)
        acts = (torch.randn(200, 16), torch.randn(200, 16))
        cc, history = train_crosscoder(
            acts, n_features=8, k=2, epochs=5, batch_size=50, seed=0
        )
        assert history.final_loss < history.initial_loss, (
            f"Expected loss to decrease; initial={history.initial_loss:.4f}, "
            f"final={history.final_loss:.4f}"
        )

    def test_history_epoch_count(self) -> None:
        from mech_interp.sae.crosscoder_trainer import train_crosscoder

        acts = (torch.randn(50, 8), torch.randn(50, 8))
        _, history = train_crosscoder(acts, n_features=4, k=1, epochs=3, batch_size=50)
        assert len(history.losses_per_epoch) == 3

    def test_mismatched_shapes_raises(self) -> None:
        from mech_interp.sae.crosscoder_trainer import train_crosscoder

        with pytest.raises(ValueError, match="same shape"):
            train_crosscoder(
                (torch.randn(50, 16), torch.randn(50, 8)),
                n_features=4,
                k=1,
            )

    def test_single_tensor_raises(self) -> None:
        from mech_interp.sae.crosscoder_trainer import train_crosscoder

        with pytest.raises(ValueError, match="at least 2"):
            train_crosscoder((torch.randn(50, 8),), n_features=4, k=1)

    def test_save_crosscoder_weights(self, tmp_path: object) -> None:
        from mech_interp.sae.crosscoder_trainer import save_crosscoder_weights, train_crosscoder

        acts = (torch.randn(50, 8), torch.randn(50, 8))
        cc, history = train_crosscoder(acts, n_features=4, k=1, epochs=1)
        path = tmp_path / "cc.safetensors"  # type: ignore[operator]
        save_crosscoder_weights(cc, path, history=history)
        assert path.is_file()
        config_path = path.with_suffix(".safetensors.json")
        assert config_path.is_file()
        import json
        config = json.loads(config_path.read_text())
        assert config["n_models"] == 2
        assert config["n_features"] == 4


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


class TestCrosscoderAnalysis:
    def _make_trained_cc(
        self,
        n_tokens: int = 100,
        input_dim: int = 16,
        n_features: int = 8,
        k: int = 2,
    ) -> tuple[Crosscoder, tuple[Tensor, Tensor], list[str]]:
        from mech_interp.sae.crosscoder_trainer import train_crosscoder

        torch.manual_seed(1)
        acts = (torch.randn(n_tokens, input_dim), torch.randn(n_tokens, input_dim))
        cc, _ = train_crosscoder(acts, n_features=n_features, k=k, epochs=2, batch_size=50, seed=1)
        prompts = [f"prompt_{i}" for i in range(n_tokens)]
        return cc, acts, prompts

    def test_analysis_runs(self) -> None:
        from mech_interp.sae.crosscoder_analysis import compute_crosscoder_analysis

        cc, acts, prompts = self._make_trained_cc()
        analysis = compute_crosscoder_analysis(cc, acts, prompts)
        assert analysis.n_features == 8
        assert analysis.live_count + analysis.dead_count == 8

    def test_decoder_norm_per_model_shape(self) -> None:
        from mech_interp.sae.crosscoder_analysis import compute_crosscoder_analysis

        cc, acts, prompts = self._make_trained_cc()
        analysis = compute_crosscoder_analysis(cc, acts, prompts)
        for record in analysis.features:
            assert len(record.decoder_norm_per_model) == 2
            for norm in record.decoder_norm_per_model:
                assert norm >= 0.0

    def test_model_score_range(self) -> None:
        from mech_interp.sae.crosscoder_analysis import compute_crosscoder_analysis

        cc, acts, prompts = self._make_trained_cc()
        analysis = compute_crosscoder_analysis(cc, acts, prompts)
        for record in analysis.features:
            assert -1.0 <= record.model_score <= 1.0, (
                f"Feature {record.feature_index} model_score={record.model_score} out of [-1, 1]"
            )

    def test_model_score_formula(self) -> None:
        """Verify (norm_a - norm_b) / (norm_a + norm_b) formula directly."""
        from mech_interp.sae.crosscoder_analysis import compute_crosscoder_analysis

        cc, acts, prompts = self._make_trained_cc(n_features=4, k=1)
        analysis = compute_crosscoder_analysis(cc, acts, prompts)
        for record in analysis.features:
            na, nb = record.decoder_norm_per_model[0], record.decoder_norm_per_model[1]
            denom = na + nb
            expected = (na - nb) / denom if denom > 1e-9 else 0.0
            assert math.isclose(record.model_score, expected, abs_tol=1e-5), (
                f"Feature {record.feature_index}: expected {expected}, got {record.model_score}"
            )

    def test_conserved_plus_specific_leq_live(self) -> None:
        from mech_interp.sae.crosscoder_analysis import compute_crosscoder_analysis

        cc, acts, prompts = self._make_trained_cc()
        analysis = compute_crosscoder_analysis(cc, acts, prompts, model_specific_threshold=0.5)
        assert analysis.conserved_count + analysis.model_specific_count == analysis.live_count

    def test_prompt_mismatch_raises(self) -> None:
        from mech_interp.sae.crosscoder_analysis import compute_crosscoder_analysis
        from mech_interp.sae.crosscoder_trainer import train_crosscoder

        acts = (torch.randn(50, 8), torch.randn(50, 8))
        cc, _ = train_crosscoder(acts, n_features=4, k=1, epochs=1)
        with pytest.raises(ValueError, match="prompt_for_token must have one entry"):
            compute_crosscoder_analysis(cc, acts, ["too_few"])

    def test_as_dict_serializable(self) -> None:
        import json

        from mech_interp.sae.crosscoder_analysis import compute_crosscoder_analysis

        cc, acts, prompts = self._make_trained_cc()
        analysis = compute_crosscoder_analysis(cc, acts, prompts)
        blob = analysis.as_dict()
        # Should round-trip through JSON without error
        json.dumps(blob)
        assert "features" in blob
        assert blob["n_features"] == 8
