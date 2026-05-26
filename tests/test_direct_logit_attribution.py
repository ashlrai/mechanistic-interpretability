"""Unit tests for DirectLogitAttributionExperiment.

Uses a synthetic backend whose run_with_cache returns known activations and
whose W_U is a fixed synthetic matrix.  Verifies:
  - Per-component contribution math (vec @ W_U @ (e_correct - e_incorrect))
  - Sign convention (positive = toward correct)
  - Top-K ranking is descending by score
  - Artifact files are written with correct structure
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from mech_interp.experiments.direct_logit_attribution import (
    DirectLogitAttributionExperiment,
    _build_summary,
    _row,
    _write_csv,
)
from mech_interp.types import ExperimentRun, ExperimentSpec, RunStatus, utc_now

# ---------------------------------------------------------------------------
# Synthetic model / backend
# ---------------------------------------------------------------------------

D_MODEL = 8
D_VOCAB = 16
N_LAYERS = 2
N_HEADS = 2
SEQ_LEN = 5
CORRECT_TOKEN_ID = 3
INCORRECT_TOKEN_ID = 7

# We'll give attn head L1H0 a clearly positive contribution and MLP L0 a
# clearly negative contribution so we can check sign convention and ranking.

def _make_W_U() -> Any:
    """W_U shape [d_model, d_vocab].  Column 3 and 7 set to known values."""
    import torch
    W_U = torch.zeros(D_MODEL, D_VOCAB)
    # direction = W_U[:, 3] - W_U[:, 7]
    W_U[0, CORRECT_TOKEN_ID] = 1.0
    W_U[1, INCORRECT_TOKEN_ID] = 1.0
    return W_U


def _make_cache(torch: Any) -> dict[str, Any]:
    """Return synthetic cache tensors keyed by TransformerLens hook names."""
    cache: dict[str, Any] = {}

    # embed / pos_embed — neutral (all zeros)
    cache["hook_embed"] = torch.zeros(1, SEQ_LEN, D_MODEL)
    cache["hook_pos_embed"] = torch.zeros(1, SEQ_LEN, D_MODEL)

    # Attn heads: [batch, seq, n_heads, d_model]
    for layer in range(N_LAYERS):
        t = torch.zeros(1, SEQ_LEN, N_HEADS, D_MODEL)
        if layer == 1:
            # L1H0: vec[0] = +2 → score = +2 (toward correct)
            t[0, -1, 0, 0] = 2.0
            # L1H1: vec[1] = +1 → score = -1 (away from correct; hits incorrect dim)
            t[0, -1, 1, 1] = 1.0
        cache[f"blocks.{layer}.attn.hook_result"] = t

    # MLP out: [batch, seq, d_model]
    for layer in range(N_LAYERS):
        t = torch.zeros(1, SEQ_LEN, D_MODEL)
        if layer == 0:
            # L0_mlp: vec[1] = +3 → score = -3 (hits incorrect dim)
            t[0, -1, 1] = 3.0
        cache[f"blocks.{layer}.hook_mlp_out"] = t

    return cache


def _make_model(torch: Any) -> Any:
    """Create a mock TransformerLens HookedTransformer with known W_U."""
    cfg = MagicMock()
    cfg.n_layers = N_LAYERS
    cfg.n_heads = N_HEADS

    model = MagicMock()
    model.cfg = cfg
    model.W_U = _make_W_U()

    cache_obj = _make_cache(torch)

    def run_with_cache(prompt: Any, names_filter: Any = None) -> tuple[Any, dict[str, Any]]:
        return MagicMock(), cache_obj

    model.run_with_cache = run_with_cache

    def to_single_token(tok: str) -> int:
        if " Paris" in tok or tok.strip() == "correct":
            return CORRECT_TOKEN_ID
        return INCORRECT_TOKEN_ID

    model.to_single_token = to_single_token
    return model


class FakeDLABackend:
    name = "transformerlens"

    def __init__(self) -> None:
        self._model: Any = None

    def load(self) -> None:
        pass  # model set externally

    @property
    def model(self) -> Any:
        return self._model

    @model.setter
    def model(self, v: Any) -> None:
        self._model = v


def _make_spec() -> ExperimentSpec:
    return ExperimentSpec(
        name="test-dla",
        family="direct_logit_attribution",
        backend="transformerlens",
        parameters={
            "model": "gpt2-small",
            "device": "cpu",
            "seed": 42,
            "target_position": -1,
            "top_k": 5,
            "prompt_pairs": [
                {
                    "id": "capital-france",
                    "clean_prompt": "The capital of France is",
                    "correct_token": " Paris",
                    "incorrect_token": " London",
                }
            ],
        },
    )


def _make_run(tmp_path: Path) -> ExperimentRun:
    return ExperimentRun(
        id=1,
        spec_name="test-dla",
        family="direct_logit_attribution",
        backend="transformerlens",
        status=RunStatus.RUNNING,
        artifact_dir=tmp_path,
        created_at=utc_now(),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_dla_contribution_math(tmp_path: Path) -> None:
    """Verify the per-component scores match hand-computed values."""
    torch = pytest.importorskip("torch")

    backend = FakeDLABackend()
    backend.model = _make_model(torch)

    spec = _make_spec()
    run = _make_run(tmp_path)

    result = DirectLogitAttributionExperiment(backend=backend).run(spec, run)

    assert result.status == RunStatus.SUCCEEDED

    rows = json.loads(Path(result.artifacts["lda_ranked_json"]).read_text())
    by_comp = {r["component_id"]: r["score"] for r in rows}

    # W_U direction = W_U[:, 3] - W_U[:, 7] = e0 - e1 (first two dims)
    # L1H0: vec = [2, 0, ...] → dot with (e0 - e1) = +2
    assert abs(by_comp["L1H0"] - 2.0) < 1e-5, f"L1H0 score={by_comp['L1H0']}"
    # L1H1: vec = [0, 1, ...] → dot with (e0 - e1) = -1
    assert abs(by_comp["L1H1"] - (-1.0)) < 1e-5, f"L1H1 score={by_comp['L1H1']}"
    # L0_mlp: vec = [0, 3, ...] → dot with (e0 - e1) = -3
    assert abs(by_comp["L0_mlp"] - (-3.0)) < 1e-5, f"L0_mlp score={by_comp['L0_mlp']}"


def test_dla_sign_convention(tmp_path: Path) -> None:
    """Positive score → pushes toward correct token."""
    torch = pytest.importorskip("torch")

    backend = FakeDLABackend()
    backend.model = _make_model(torch)

    spec = _make_spec()
    run = _make_run(tmp_path)

    result = DirectLogitAttributionExperiment(backend=backend).run(spec, run)
    rows = json.loads(Path(result.artifacts["lda_ranked_json"]).read_text())
    by_comp = {r["component_id"]: r["score"] for r in rows}

    # L1H0 fires on correct dim → positive
    assert by_comp["L1H0"] > 0
    # L0_mlp fires on incorrect dim → negative
    assert by_comp["L0_mlp"] < 0


def test_dla_ranking_descending(tmp_path: Path) -> None:
    """Rows must be ranked descending by score."""
    torch = pytest.importorskip("torch")

    backend = FakeDLABackend()
    backend.model = _make_model(torch)

    spec = _make_spec()
    run = _make_run(tmp_path)

    result = DirectLogitAttributionExperiment(backend=backend).run(spec, run)
    rows = json.loads(Path(result.artifacts["lda_ranked_json"]).read_text())

    scores = [r["score"] for r in rows]
    assert scores == sorted(scores, reverse=True), "Rows not sorted descending by score"

    ranks = [r["rank"] for r in rows]
    assert ranks == list(range(1, len(ranks) + 1)), "Ranks not sequential"


def test_dla_artifacts_written(tmp_path: Path) -> None:
    """All three artifact files exist and have expected structure."""
    torch = pytest.importorskip("torch")

    backend = FakeDLABackend()
    backend.model = _make_model(torch)

    spec = _make_spec()
    run = _make_run(tmp_path)

    result = DirectLogitAttributionExperiment(backend=backend).run(spec, run)

    ranked = json.loads(Path(result.artifacts["lda_ranked_json"]).read_text())
    assert len(ranked) > 0
    assert all("component_id" in r for r in ranked)
    assert all("evidence_label" in r for r in ranked)
    assert all(r["evidence_label"] == "direct_logit_decomposition" for r in ranked)

    summary = json.loads(Path(result.artifacts["lda_summary"]).read_text())
    assert "top_positive" in summary
    assert "top_negative" in summary
    assert "total_components" in summary

    csv_text = Path(result.artifacts["lda_ranked_csv"]).read_text()
    assert "component_id" in csv_text
    assert "score" in csv_text


def test_dla_top_k_summary(tmp_path: Path) -> None:
    """summary top_positive contains components with positive scores."""
    torch = pytest.importorskip("torch")

    backend = FakeDLABackend()
    backend.model = _make_model(torch)

    spec = _make_spec()
    run = _make_run(tmp_path)

    result = DirectLogitAttributionExperiment(backend=backend).run(spec, run)
    summary = json.loads(Path(result.artifacts["lda_summary"]).read_text())

    for entry in summary["top_positive"]:
        assert entry["mean_score"] > 0
    for entry in summary["top_negative"]:
        assert entry["mean_score"] < 0


def test_build_summary_aggregates_across_prompts() -> None:
    """_build_summary averages scores per component across multiple prompts."""
    rows = [
        _row("p1", "attn_head", 1, 0, 4.0),
        _row("p2", "attn_head", 1, 0, 2.0),
        _row("p1", "mlp", 0, None, -1.0),
    ]
    # set ranks
    for i, r in enumerate(rows):
        r["rank"] = i + 1

    summary = _build_summary(rows, top_k=5)
    by_id = {e["component_id"]: e for e in summary["top_positive"] + summary["top_negative"]}
    assert abs(by_id["L1H0"]["mean_score"] - 3.0) < 1e-6
    assert by_id["L0_mlp"]["mean_score"] < 0


def test_write_csv_produces_valid_output(tmp_path: Path) -> None:
    """_write_csv writes a header + data row."""
    rows = [_row("p1", "attn_head", 2, 3, 1.5)]
    rows[0]["rank"] = 1
    out = tmp_path / "test.csv"
    _write_csv(out, rows)
    text = out.read_text()
    assert "component_id" in text
    assert "L2H3" in text


def test_write_csv_empty(tmp_path: Path) -> None:
    """_write_csv handles empty rows without crashing."""
    out = tmp_path / "empty.csv"
    _write_csv(out, [])
    assert "pair_id" in out.read_text()
