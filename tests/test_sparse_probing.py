"""Unit tests for SparseProbingExperiment.

Uses synthetic activations where the positive class has a clear direction in
dimension 0.  Verifies:
  - Probe recovers the direction (high accuracy)
  - L1 penalty produces a sparse weight vector (low nonzero count)
  - Artifact files written with correct structure
  - Coordinate-descent solver converges
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from mech_interp.experiments.sparse_probing import (
    SparseProbingExperiment,
    _accuracy,
    _train_sparse_probe,
    _write_safetensors,
)
from mech_interp.types import ExperimentRun, ExperimentSpec, RunStatus, utc_now

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

D_MODEL = 32
SEQ_LEN = 5


def _make_synthetic_activations(
    n_pos: int,
    n_neg: int,
    signal_dim: int = 0,
    signal_strength: float = 5.0,
    noise: float = 0.1,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, y) where positive examples have large value in signal_dim."""
    rng = np.random.default_rng(seed)
    n = n_pos + n_neg
    X = rng.normal(0, noise, size=(n, D_MODEL)).astype(np.float32)
    X[:n_pos, signal_dim] += signal_strength
    y = np.array([1.0] * n_pos + [0.0] * n_neg, dtype=np.float32)
    return X, y


class FakeProbingBackend:
    """Fake backend that returns pre-set activations from run_with_cache."""

    name = "transformerlens"

    def __init__(self, activations: np.ndarray) -> None:
        self._activations = activations
        self._model: Any = None
        self._make_model()

    def _make_model(self) -> None:
        import torch

        acts = torch.from_numpy(self._activations[:, np.newaxis, :])  # [n, 1, d]
        # Expand to [n, SEQ_LEN, d]
        acts = acts.expand(-1, SEQ_LEN, -1)

        cache_obj = {"blocks.6.hook_resid_pre": acts}

        model = MagicMock()
        model.cfg = MagicMock()
        model.cfg.n_layers = 12
        model.cfg.n_heads = 12

        def run_with_cache(
            prompts: Any, names_filter: Any = None
        ) -> tuple[Any, dict[str, Any]]:
            return MagicMock(), cache_obj

        model.run_with_cache = run_with_cache
        self._model = model

    def load(self) -> None:
        pass

    @property
    def model(self) -> Any:
        return self._model


def _make_spec(
    pos_prompts: list[str],
    neg_prompts: list[str],
) -> ExperimentSpec:
    return ExperimentSpec(
        name="test-sparse-probe",
        family="sparse_probing",
        backend="transformerlens",
        parameters={
            "model": "gpt2-small",
            "device": "cpu",
            "hook_site": "blocks.6.hook_resid_pre",
            "positive_prompts": pos_prompts,
            "negative_prompts": neg_prompts,
            "l1_alpha": 0.01,
            "epochs": 200,
            "train_fraction": 0.8,
            "seed": 42,
        },
    )


def _make_run(tmp_path: Path) -> ExperimentRun:
    return ExperimentRun(
        id=1,
        spec_name="test-sparse-probe",
        family="sparse_probing",
        backend="transformerlens",
        status=RunStatus.RUNNING,
        artifact_dir=tmp_path,
        created_at=utc_now(),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_probe_recovers_clear_direction(tmp_path: Path) -> None:
    """With a strong signal in dim-0, probe should achieve high accuracy."""
    pytest.importorskip("torch")

    n_pos, n_neg = 10, 10
    X, _y = _make_synthetic_activations(n_pos, n_neg, signal_dim=0, signal_strength=8.0)

    pos_prompts = [f"factual prompt {i}" for i in range(n_pos)]
    neg_prompts = [f"random prompt {i}" for i in range(n_neg)]

    backend = FakeProbingBackend(X)
    spec = _make_spec(pos_prompts, neg_prompts)
    run = _make_run(tmp_path)

    result = SparseProbingExperiment(backend=backend).run(spec, run)

    assert result.status == RunStatus.SUCCEEDED
    assert result.metrics["train_accuracy"] > 0.7
    assert result.metrics["eval_accuracy"] > 0.6


def test_probe_is_sparse(tmp_path: Path) -> None:
    """L1 regularisation should produce fewer than d_model/2 nonzero weights."""
    pytest.importorskip("torch")

    n_pos, n_neg = 12, 12
    X, _y = _make_synthetic_activations(n_pos, n_neg, signal_dim=0, signal_strength=10.0)

    pos_prompts = [f"fp{i}" for i in range(n_pos)]
    neg_prompts = [f"rp{i}" for i in range(n_neg)]

    backend = FakeProbingBackend(X)
    spec = _make_spec(pos_prompts, neg_prompts)
    run = _make_run(tmp_path)

    result = SparseProbingExperiment(backend=backend).run(spec, run)

    assert result.metrics["nonzero_count"] < D_MODEL // 2


def test_probe_artifacts_written(tmp_path: Path) -> None:
    """Both probe_weights.safetensors and probe_summary.json must exist."""
    pytest.importorskip("torch")

    n_pos, n_neg = 8, 8
    X, _y = _make_synthetic_activations(n_pos, n_neg)

    backend = FakeProbingBackend(X)
    spec = _make_spec([f"p{i}" for i in range(n_pos)], [f"n{i}" for i in range(n_neg)])
    run = _make_run(tmp_path)

    result = SparseProbingExperiment(backend=backend).run(spec, run)

    weights_path = Path(result.artifacts["probe_weights"])
    summary_path = Path(result.artifacts["probe_summary"])

    assert weights_path.is_file()
    assert weights_path.stat().st_size > 0

    summary = json.loads(summary_path.read_text())
    for key in ("l1_alpha", "nonzero_count", "total_count", "train_accuracy", "eval_accuracy",
                "positive_count", "negative_count"):
        assert key in summary, f"Missing key: {key}"
    assert summary["positive_count"] == n_pos
    assert summary["negative_count"] == n_neg
    assert summary["total_count"] == D_MODEL


def test_train_sparse_probe_basic() -> None:
    """Solver converges on trivially separable data."""
    rng = np.random.default_rng(0)
    n, d = 40, 16
    X = rng.normal(0, 1, (n, d)).astype(np.float32)
    X[:n // 2, 0] += 6.0  # strong signal in dim 0
    y = np.array([1.0] * (n // 2) + [0.0] * (n // 2), dtype=np.float32)

    w, b = _train_sparse_probe(X, y, l1_alpha=0.05, epochs=300)
    acc = _accuracy(X, y, w, b)
    assert acc > 0.85


def test_train_sparse_probe_sparsity() -> None:
    """High l1_alpha should zero out most irrelevant dims."""
    rng = np.random.default_rng(1)
    n, d = 60, 32
    X = rng.normal(0, 0.1, (n, d)).astype(np.float32)
    X[:n // 2, 2] += 10.0  # only dim 2 matters
    y = np.array([1.0] * (n // 2) + [0.0] * (n // 2), dtype=np.float32)

    w, _b = _train_sparse_probe(X, y, l1_alpha=0.1, epochs=300)
    nonzero = int(np.sum(np.abs(w) > 1e-8))
    assert nonzero < d // 2, f"Expected sparse weights, got {nonzero}/{d} nonzero"


def test_write_safetensors_parses(tmp_path: Path) -> None:
    """Verify the minimal safetensors file has valid 8-byte header + parseable JSON."""
    w = np.array([1.0, -2.0, 0.0, 3.5], dtype=np.float32)
    path = tmp_path / "w.safetensors"
    _write_safetensors(path, w)

    data = path.read_bytes()
    assert len(data) > 8

    header_size = struct.unpack("<Q", data[:8])[0]
    header_json = json.loads(data[8 : 8 + header_size].decode("utf-8").strip())
    assert "probe_direction" in header_json
    assert header_json["probe_direction"]["dtype"] == "F32"
    assert header_json["probe_direction"]["shape"] == [4]

    # Verify payload bytes round-trip
    start, end = header_json["probe_direction"]["data_offsets"]
    payload = data[8 + header_size + start : 8 + header_size + end]
    recovered = np.frombuffer(payload, dtype=np.float32)
    np.testing.assert_allclose(recovered, w)
