"""Unit tests for sae_seed_stability analysis module.

Uses tiny synthetic SAEs (n_features=4, d_model=8) written to tmp_path.
No real model is loaded; no network calls are made.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

_TORCH_AVAILABLE = True
try:
    import torch  # noqa: F401
except ImportError:
    _TORCH_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _TORCH_AVAILABLE, reason="torch not installed"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_sae(tmp_path: Path, name: str, decoder_weight: torch.Tensor) -> Path:
    """Write a minimal SAE weights file + config JSON to tmp_path/<name>/."""
    import torch
    from safetensors.torch import save_file

    d_model, n_features = decoder_weight.shape
    run_dir = tmp_path / name
    run_dir.mkdir(parents=True, exist_ok=True)

    weights_path = run_dir / "sae_weights.safetensors"

    # Build minimal state dict matching TopKSAE encoder/decoder naming
    # (decoder is nn.Linear(n_features, d_model, bias=True))
    encoder_weight = torch.randn(n_features, d_model)
    encoder_bias = torch.zeros(n_features)
    decoder_bias = torch.zeros(d_model)
    state: dict[str, torch.Tensor] = {
        "encoder.weight": encoder_weight.contiguous(),
        "encoder.bias": encoder_bias.contiguous(),
        "decoder.weight": decoder_weight.contiguous(),
        "decoder.bias": decoder_bias.contiguous(),
    }
    save_file(state, str(weights_path))

    config = {"input_dim": d_model, "n_features": n_features, "k": 2, "training": None}
    (run_dir / "sae_weights.safetensors.json").write_text(
        json.dumps(config), encoding="utf-8"
    )
    return run_dir


# ---------------------------------------------------------------------------
# Tests: compute_sae_pair_alignment
# ---------------------------------------------------------------------------


class TestComputeSaePairAlignment:
    def test_identical_saes_return_cosine_one(self, tmp_path: Path) -> None:
        import torch

        from mech_interp.analysis.sae_seed_stability import compute_sae_pair_alignment

        W = torch.randn(8, 4)  # d_model=8, n_features=4
        dir_a = _write_sae(tmp_path, "run_a", W)
        dir_b = _write_sae(tmp_path, "run_b", W)

        result = compute_sae_pair_alignment(
            dir_a / "sae_weights.safetensors",
            dir_b / "sae_weights.safetensors",
            threshold=0.9,
        )

        assert result["n_matched_pairs"] == 4
        # Identical weights → all cosines should be ~1.0
        assert result["median_cosine"] == pytest.approx(1.0, abs=1e-4)
        assert result["mean_cosine"] == pytest.approx(1.0, abs=1e-4)
        assert result["matched_count_above_threshold"] == 4

    def test_orthogonal_saes_return_low_cosine(self, tmp_path: Path) -> None:
        import torch

        from mech_interp.analysis.sae_seed_stability import compute_sae_pair_alignment

        # Build a 4×4 orthogonal matrix; each column is a unit direction
        Q, _ = torch.linalg.qr(torch.randn(8, 4))
        W_a = Q.contiguous()  # (8, 4)
        W_b = (-Q).contiguous()  # negated → cosines should be -1 (worst case alignment)

        dir_a = _write_sae(tmp_path, "orth_a", W_a)
        dir_b = _write_sae(tmp_path, "orth_b", W_b)

        result = compute_sae_pair_alignment(
            dir_a / "sae_weights.safetensors",
            dir_b / "sae_weights.safetensors",
            threshold=0.9,
        )

        assert result["matched_count_above_threshold"] == 0
        # Negated directions → mean cosine should be negative (≤ 0)
        assert result["mean_cosine"] <= 0.0

    def test_top_k_truncation(self, tmp_path: Path) -> None:
        import torch

        from mech_interp.analysis.sae_seed_stability import compute_sae_pair_alignment

        W = torch.randn(8, 4)
        dir_a = _write_sae(tmp_path, "tk_a", W)
        dir_b = _write_sae(tmp_path, "tk_b", W)

        result = compute_sae_pair_alignment(
            dir_a / "sae_weights.safetensors",
            dir_b / "sae_weights.safetensors",
            top_k=2,
        )
        # top_k=2, but only 4 features → at most 2 entries in top_matches
        assert len(result["top_matches"]) == 2

    def test_d_model_mismatch_raises(self, tmp_path: Path) -> None:
        import torch

        from mech_interp.analysis.sae_seed_stability import compute_sae_pair_alignment

        dir_a = _write_sae(tmp_path, "mismatch_a", torch.randn(8, 4))
        dir_b = _write_sae(tmp_path, "mismatch_b", torch.randn(16, 4))

        with pytest.raises(ValueError, match="d_model mismatch"):
            compute_sae_pair_alignment(
                dir_a / "sae_weights.safetensors",
                dir_b / "sae_weights.safetensors",
            )

    def test_return_keys_present(self, tmp_path: Path) -> None:
        import torch

        from mech_interp.analysis.sae_seed_stability import compute_sae_pair_alignment

        W = torch.randn(8, 4)
        dir_a = _write_sae(tmp_path, "keys_a", W)
        dir_b = _write_sae(tmp_path, "keys_b", W)

        result = compute_sae_pair_alignment(
            dir_a / "sae_weights.safetensors",
            dir_b / "sae_weights.safetensors",
        )

        expected_keys = {
            "matched_count_above_threshold",
            "threshold",
            "n_matched_pairs",
            "median_cosine",
            "mean_cosine",
            "top_matches",
            "all_cosines",
        }
        assert expected_keys.issubset(result.keys())


# ---------------------------------------------------------------------------
# Tests: compute_stability_report
# ---------------------------------------------------------------------------


class TestComputeStabilityReport:
    def test_three_identical_runs(self, tmp_path: Path) -> None:
        import torch

        from mech_interp.analysis.sae_seed_stability import compute_stability_report

        W = torch.randn(8, 4)
        dirs = [_write_sae(tmp_path, f"run_{i}", W) for i in range(3)]

        report = compute_stability_report(dirs, threshold=0.9)

        assert report["summary"]["n_pairs"] == 3  # C(3,2)
        assert report["summary"]["median_of_medians"] == pytest.approx(1.0, abs=1e-3)
        assert report["summary"]["mean_stability_fraction"] == pytest.approx(1.0, abs=1e-3)
        assert len(report["pairwise"]) == 3

    def test_missing_weights_raises(self, tmp_path: Path) -> None:
        import torch

        from mech_interp.analysis.sae_seed_stability import compute_stability_report

        W = torch.randn(8, 4)
        dir_a = _write_sae(tmp_path, "good_run", W)
        missing = tmp_path / "empty_run"
        missing.mkdir()

        with pytest.raises(FileNotFoundError):
            compute_stability_report([dir_a, missing])

    def test_report_structure(self, tmp_path: Path) -> None:
        import torch

        from mech_interp.analysis.sae_seed_stability import compute_stability_report

        dirs = [_write_sae(tmp_path, f"struct_{i}", torch.randn(8, 4)) for i in range(2)]
        report = compute_stability_report(dirs, threshold=0.8)

        assert "runs" in report
        assert "pairwise" in report
        assert "summary" in report
        assert len(report["pairwise"]) == 1

        pair = report["pairwise"][0]
        assert "run_a_name" in pair
        assert "run_b_name" in pair
        assert "median_cosine" in pair
        assert "all_cosines" in pair


# ---------------------------------------------------------------------------
# Helpers for live-only tests
# ---------------------------------------------------------------------------


def _write_feature_analysis(run_dir: Path, live_indices: list[int], n_features: int) -> Path:
    """Write a minimal feature_analysis.json marking given indices as live."""
    features = []
    for i in range(n_features):
        features.append({"feature_index": i, "dead": i not in live_indices})
    data = {"n_features": n_features, "dead_count": n_features - len(live_indices),
            "live_count": len(live_indices), "mean_features_per_token": 0.5, "features": features}
    p = run_dir / "feature_analysis.json"
    import json as _json
    p.write_text(_json.dumps(data), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Tests: compute_live_only_alignment
# ---------------------------------------------------------------------------


class TestComputeLiveOnlyAlignment:
    def test_live_only_identical_saes_high_cosine(self, tmp_path: Path) -> None:
        import torch

        from mech_interp.analysis.sae_seed_stability import compute_live_only_alignment

        W = torch.randn(8, 4)
        dir_a = _write_sae(tmp_path, "lo_a", W)
        dir_b = _write_sae(tmp_path, "lo_b", W)
        ana_a = _write_feature_analysis(dir_a, live_indices=[0, 1, 2], n_features=4)
        ana_b = _write_feature_analysis(dir_b, live_indices=[0, 1, 2], n_features=4)

        result = compute_live_only_alignment(
            dir_a / "sae_weights.safetensors",
            dir_b / "sae_weights.safetensors",
            ana_a,
            ana_b,
            threshold=0.9,
        )

        assert result["mode"] == "live_only"
        assert result["live_features_a"] == 3
        assert result["live_features_b"] == 3
        assert result["n_matched_pairs"] == 3
        assert result["median_cosine"] == pytest.approx(1.0, abs=1e-4)
        assert result["matched_count_above_threshold"] == 3

    def test_live_only_restricts_to_live(self, tmp_path: Path) -> None:
        """Dead features should be excluded: n_matched_pairs == min(live_a, live_b)."""
        import torch

        from mech_interp.analysis.sae_seed_stability import compute_live_only_alignment

        W = torch.randn(8, 4)
        dir_a = _write_sae(tmp_path, "restrict_a", W)
        dir_b = _write_sae(tmp_path, "restrict_b", W)
        # Only 2 live features in A, 3 in B → matching uses 2
        ana_a = _write_feature_analysis(dir_a, live_indices=[0, 2], n_features=4)
        ana_b = _write_feature_analysis(dir_b, live_indices=[0, 1, 3], n_features=4)

        result = compute_live_only_alignment(
            dir_a / "sae_weights.safetensors",
            dir_b / "sae_weights.safetensors",
            ana_a,
            ana_b,
        )

        assert result["n_matched_pairs"] == 2

    def test_live_only_no_live_features_returns_zero(self, tmp_path: Path) -> None:
        import torch

        from mech_interp.analysis.sae_seed_stability import compute_live_only_alignment

        W = torch.randn(8, 4)
        dir_a = _write_sae(tmp_path, "dead_a", W)
        dir_b = _write_sae(tmp_path, "dead_b", W)
        ana_a = _write_feature_analysis(dir_a, live_indices=[], n_features=4)
        ana_b = _write_feature_analysis(dir_b, live_indices=[0, 1], n_features=4)

        result = compute_live_only_alignment(
            dir_a / "sae_weights.safetensors",
            dir_b / "sae_weights.safetensors",
            ana_a,
            ana_b,
        )

        assert result["n_matched_pairs"] == 0
        assert result["median_cosine"] == 0.0

    def test_live_only_return_keys(self, tmp_path: Path) -> None:
        import torch

        from mech_interp.analysis.sae_seed_stability import compute_live_only_alignment

        W = torch.randn(8, 4)
        dir_a = _write_sae(tmp_path, "keys2_a", W)
        dir_b = _write_sae(tmp_path, "keys2_b", W)
        ana_a = _write_feature_analysis(dir_a, live_indices=[0, 1], n_features=4)
        ana_b = _write_feature_analysis(dir_b, live_indices=[0, 1], n_features=4)

        result = compute_live_only_alignment(
            dir_a / "sae_weights.safetensors",
            dir_b / "sae_weights.safetensors",
            ana_a,
            ana_b,
        )

        expected = {"matched_count_above_threshold", "threshold", "n_matched_pairs",
                    "median_cosine", "mean_cosine", "top_matches", "all_cosines",
                    "live_features_a", "live_features_b", "mode"}
        assert expected.issubset(result.keys())


# ---------------------------------------------------------------------------
# Tests: compute_live_only_stability_report
# ---------------------------------------------------------------------------


class TestComputeLiveOnlyStabilityReport:
    def test_three_identical_runs_live_only(self, tmp_path: Path) -> None:
        import torch

        from mech_interp.analysis.sae_seed_stability import compute_live_only_stability_report

        W = torch.randn(8, 4)
        dirs = []
        for i in range(3):
            d = _write_sae(tmp_path, f"lo_rep_{i}", W)
            _write_feature_analysis(d, live_indices=[0, 1, 2, 3], n_features=4)
            dirs.append(d)

        report = compute_live_only_stability_report(dirs, threshold=0.9)

        assert report["summary"]["n_pairs"] == 3
        assert report["summary"]["mode"] == "live_only"
        assert report["summary"]["median_of_medians"] == pytest.approx(1.0, abs=1e-3)

    def test_missing_feature_analysis_raises(self, tmp_path: Path) -> None:
        import torch

        from mech_interp.analysis.sae_seed_stability import compute_live_only_stability_report

        W = torch.randn(8, 4)
        dir_a = _write_sae(tmp_path, "noa_a", W)
        dir_b = _write_sae(tmp_path, "noa_b", W)
        # dir_a has no feature_analysis.json

        with pytest.raises(FileNotFoundError, match="feature_analysis.json"):
            compute_live_only_stability_report([dir_a, dir_b])
