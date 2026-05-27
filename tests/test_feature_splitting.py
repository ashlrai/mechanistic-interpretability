"""Tests for mech_interp.analysis.feature_splitting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

from mech_interp.analysis.feature_splitting import (
    ChildRecord,
    FeatureSplitAnalysis,
    SplitRecord,
    _load_feature_analysis,
    compute_feature_split_analysis,
    compute_feature_splits,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _write_torch_weights(path: Path, decoder_weight: np.ndarray) -> None:
    """Save a minimal torch state dict so _load_decoder can read it without safetensors.

    decoder_weight is (d_model, n_features).  nn.Linear(n_features, d_model) stores
    its weight as (d_model, n_features), so we save it as-is.  _load_decoder then
    transposes to (n_features, d_model) for the cosine computation.
    """
    import torch

    # Store as (d_model, n_features) — matching nn.Linear(n_features, d_model).weight
    state = {"decoder.weight": torch.from_numpy(decoder_weight.astype(np.float32))}
    torch.save(state, path)


def _write_feature_analysis(path: Path, n_features: int, features: list[dict[str, Any]]) -> None:
    data = {
        "n_features": n_features,
        "dead_count": sum(1 for f in features if f.get("dead", False)),
        "live_count": sum(1 for f in features if not f.get("dead", False)),
        "mean_features_per_token": 8.0,
        "features": features,
    }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _make_feature_entry(
    idx: int, *, dead: bool = False, prompts: list[str] | None = None
) -> dict[str, Any]:
    return {
        "feature_index": idx,
        "dead": dead,
        "top_prompts": [
            {"activation": 1.0, "prompt": p, "rank": i + 1} for i, p in enumerate(prompts or [])
        ],
        "coherence_score": 0.5,
    }


@pytest.fixture()
def simple_pair(tmp_path: Path) -> dict[str, Any]:
    """Two SAEs: parent with 4 features (2 live), child with 8 features.

    We construct decoder directions so that:
      - parent feat 0 → perfectly matches child feat 0 (cosine 1.0)
      - parent feat 1 → strongly matches child feat 2 (cosine ~0.95) and
        moderately matches child feat 5 (cosine ~0.7)
    """
    rng = np.random.default_rng(0)
    d_model = 16

    # Parent decoders: (d_model, n_features=4), stored as nn.Linear weight
    parent_dec = rng.normal(size=(d_model, 4)).astype(np.float32)
    child_dec = rng.normal(size=(d_model, 8)).astype(np.float32)

    # Force known relationships
    v0 = rng.normal(size=d_model).astype(np.float32)
    v0 /= np.linalg.norm(v0)
    parent_dec[:, 0] = v0
    child_dec[:, 0] = v0  # cosine = 1.0

    v1 = rng.normal(size=d_model).astype(np.float32)
    v1 /= np.linalg.norm(v1)
    parent_dec[:, 1] = v1
    # child 2: very similar direction
    noise = rng.normal(scale=0.05, size=d_model).astype(np.float32)
    child_dec[:, 2] = v1 + noise
    # child 5: moderately similar
    noise2 = rng.normal(scale=0.3, size=d_model).astype(np.float32)
    child_dec[:, 5] = v1 + noise2

    parent_weights = tmp_path / "parent.pt"
    child_weights = tmp_path / "child.pt"
    _write_torch_weights(parent_weights, parent_dec)
    _write_torch_weights(child_weights, child_dec)

    parent_analysis = tmp_path / "parent_analysis.json"
    child_analysis = tmp_path / "child_analysis.json"
    _write_feature_analysis(
        parent_analysis,
        n_features=4,
        features=[
            _make_feature_entry(0, prompts=["geography text", "biology text"]),
            _make_feature_entry(1, prompts=["code text", "math text"]),
            _make_feature_entry(2, dead=True),
            _make_feature_entry(3, dead=True),
        ],
    )
    _write_feature_analysis(
        child_analysis,
        n_features=8,
        features=[
            _make_feature_entry(0, prompts=["geography A", "geography B"]),
            _make_feature_entry(1, dead=True),
            _make_feature_entry(2, prompts=["code A", "code B"]),
            _make_feature_entry(3, dead=True),
            _make_feature_entry(4, dead=True),
            _make_feature_entry(5, prompts=["math A"]),
            _make_feature_entry(6, dead=True),
            _make_feature_entry(7, dead=True),
        ],
    )

    return {
        "parent_sae_path": parent_weights,
        "child_sae_path": child_weights,
        "parent_analysis_path": parent_analysis,
        "child_analysis_path": child_analysis,
    }


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


class TestLoadFeatureAnalysis:
    def test_loads_valid_json(self, tmp_path: Path) -> None:
        p = tmp_path / "fa.json"
        _write_feature_analysis(
            p, n_features=4, features=[_make_feature_entry(0, prompts=["hello"])]
        )
        data = _load_feature_analysis(p)
        assert data["n_features"] == 4
        assert len(data["features"]) == 1

    def test_raises_on_missing(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            _load_feature_analysis(tmp_path / "nonexistent.json")


class TestSplitRecord:
    def test_best_cosine_empty(self) -> None:
        r = SplitRecord(parent_feature=0, parent_top_prompts=[])
        assert r.best_cosine == 0.0

    def test_best_cosine_populated(self) -> None:
        r = SplitRecord(
            parent_feature=0,
            parent_top_prompts=[],
            children=[
                ChildRecord(feature=1, cosine=0.9, top_prompts=[]),
                ChildRecord(feature=2, cosine=0.7, top_prompts=[]),
            ],
        )
        assert r.best_cosine == pytest.approx(0.9)

    def test_as_dict_roundtrip(self) -> None:
        r = SplitRecord(
            parent_feature=3,
            parent_top_prompts=["hello"],
            children=[ChildRecord(feature=7, cosine=0.85, top_prompts=["world"])],
        )
        d = r.as_dict()
        assert d["parent_feature"] == 3
        assert d["best_cosine"] == pytest.approx(0.85)
        assert d["children"][0]["feature"] == 7


class TestFeatureSplitAnalysis:
    def test_mean_fidelity_empty(self) -> None:
        fa = FeatureSplitAnalysis(
            parent_n_features=4, child_n_features=8, parent_live_count=0, split_records=[]
        )
        assert fa.mean_split_fidelity == 0.0

    def test_mean_fidelity_computed(self) -> None:
        records = [
            SplitRecord(
                0,
                [],
                [ChildRecord(1, 0.8, [])],
            ),
            SplitRecord(
                1,
                [],
                [ChildRecord(2, 0.6, [])],
            ),
        ]
        fa = FeatureSplitAnalysis(
            parent_n_features=4,
            child_n_features=8,
            parent_live_count=2,
            split_records=records,
        )
        assert fa.mean_split_fidelity == pytest.approx(0.7)

    def test_split_distribution(self) -> None:
        records = [
            SplitRecord(0, [], []),  # 0 children
            SplitRecord(1, [], [ChildRecord(2, 0.5, [])]),  # 1 child
            SplitRecord(2, [], [ChildRecord(3, 0.5, []), ChildRecord(4, 0.4, [])]),  # 2
            SplitRecord(
                3,
                [],
                [ChildRecord(5, 0.9, []), ChildRecord(6, 0.7, []), ChildRecord(7, 0.5, [])],
            ),  # 3+
        ]
        fa = FeatureSplitAnalysis(
            parent_n_features=8,
            child_n_features=16,
            parent_live_count=4,
            split_records=records,
        )
        dist = fa.split_distribution
        assert dist[0] == 1
        assert dist[1] == 1
        assert dist[2] == 1
        assert dist[3] == 1

    def test_as_dict_keys(self) -> None:
        fa = FeatureSplitAnalysis(
            parent_n_features=4,
            child_n_features=8,
            parent_live_count=0,
            split_records=[],
        )
        d = fa.as_dict()
        assert set(d.keys()) >= {
            "parent_n_features",
            "child_n_features",
            "parent_live_count",
            "mean_split_fidelity",
            "split_distribution",
            "split_records",
        }


class TestComputeFeatureSplits:
    def test_returns_only_live_parents(self, simple_pair: dict[str, Any]) -> None:
        records = compute_feature_splits(**simple_pair)
        # Only 2 live features in parent
        assert len(records) == 2
        parent_idxs = {r.parent_feature for r in records}
        assert parent_idxs == {0, 1}

    def test_top_child_for_perfect_match(self, simple_pair: dict[str, Any]) -> None:
        records = compute_feature_splits(**simple_pair, min_cosine=0.5)
        rec0 = next(r for r in records if r.parent_feature == 0)
        assert rec0.children, "Expected at least one child for parent feat 0"
        assert rec0.children[0].feature == 0
        assert rec0.children[0].cosine == pytest.approx(1.0, abs=1e-4)

    def test_min_cosine_filters(self, simple_pair: dict[str, Any]) -> None:
        records_strict = compute_feature_splits(**simple_pair, min_cosine=0.99)
        records_loose = compute_feature_splits(**simple_pair, min_cosine=0.0)
        total_strict = sum(len(r.children) for r in records_strict)
        total_loose = sum(len(r.children) for r in records_loose)
        assert total_strict <= total_loose

    def test_top_k_respected(self, simple_pair: dict[str, Any]) -> None:
        records = compute_feature_splits(**simple_pair, top_k_children=1, min_cosine=0.0)
        for r in records:
            assert len(r.children) <= 1

    def test_children_sorted_descending(self, simple_pair: dict[str, Any]) -> None:
        records = compute_feature_splits(**simple_pair, min_cosine=0.0)
        for r in records:
            cosines = [c.cosine for c in r.children]
            assert cosines == sorted(cosines, reverse=True)

    def test_missing_weights_raises(self, simple_pair: dict[str, Any], tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            compute_feature_splits(
                tmp_path / "missing.safetensors",
                simple_pair["child_sae_path"],
                simple_pair["parent_analysis_path"],
                simple_pair["child_analysis_path"],
            )


class TestComputeFeatureSplitAnalysis:
    def test_aggregate_stats(self, simple_pair: dict[str, Any]) -> None:
        result = compute_feature_split_analysis(**simple_pair)
        assert isinstance(result, FeatureSplitAnalysis)
        assert result.parent_n_features == 4
        assert result.child_n_features == 8
        assert result.parent_live_count == 2
        assert 0.0 <= result.mean_split_fidelity <= 1.0

    def test_as_dict_serialisable(self, simple_pair: dict[str, Any]) -> None:
        result = compute_feature_split_analysis(**simple_pair)
        d = result.as_dict()
        # Must be JSON-serialisable
        json.dumps(d)


# ---------------------------------------------------------------------------
# CLI integration test
# ---------------------------------------------------------------------------


class TestCLIAnalyzeFeatureSplits:
    def test_cli_smoke(self, simple_pair: dict[str, Any], tmp_path: Path) -> None:
        """CLI command runs without error on mock artifact dirs."""
        from unittest.mock import MagicMock

        from typer.testing import CliRunner

        from mech_interp import cli

        # Build fake artifact dirs that look like run dirs
        parent_dir = tmp_path / "artifacts" / "run-000001"
        child_dir = tmp_path / "artifacts" / "run-000002"
        parent_dir.mkdir(parents=True)
        child_dir.mkdir(parents=True)

        # Copy fixture files into fake run dirs; use .pt extension so the
        # torch path in _load_decoder fires (avoids safetensors dep in tests).
        import shutil

        shutil.copy(simple_pair["parent_sae_path"], parent_dir / "sae_weights.safetensors")
        shutil.copy(simple_pair["child_sae_path"], child_dir / "sae_weights.safetensors")
        shutil.copy(simple_pair["parent_analysis_path"], parent_dir / "feature_analysis.json")
        shutil.copy(simple_pair["child_analysis_path"], child_dir / "feature_analysis.json")

        # Patch _load_decoder so it uses the torch path regardless of extension.
        # The fixture .pt files store decoder.weight as (d_model, n_features);
        # _load_decoder transposes to (n_features, d_model).
        from mech_interp.analysis import feature_splitting as fs_mod

        def patched_load(path: Path) -> np.ndarray:
            import numpy as _np
            import torch as _torch

            state = _torch.load(path, map_location="cpu", weights_only=True)
            arr: _np.ndarray = state["decoder.weight"].numpy()
            if arr.ndim == 2:
                arr = arr.T
            return arr.astype(_np.float32)

        mock_config = MagicMock()
        mock_config.project.artifact_dir = tmp_path / "artifacts"

        with (
            patch.object(fs_mod, "_load_decoder", patched_load),
            patch("mech_interp.cli.load_config", return_value=mock_config),
        ):
            runner = CliRunner()
            result = runner.invoke(
                cli.app,
                [
                    "analyze-feature-splits",
                    "--parent-run", "1",
                    "--child-run", "2",
                ],
            )

        assert result.exit_code == 0, result.output
        assert "Mean split fidelity" in result.output
        output_file = child_dir / "feature_splits.json"
        assert output_file.exists()
        data = json.loads(output_file.read_text())
        assert "mean_split_fidelity" in data
