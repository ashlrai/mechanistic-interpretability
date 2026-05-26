"""Unit tests for ACDC-edge: pruning math, artifact shape, faithfulness clamp."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from mech_interp.experiments.acdc_edge import (
    CircuitEdge,
    EdgeCircuitArtifact,
    EdgeNode,
    EdgePruningStep,
    _build_edges,
    _build_nodes,
    _faithfulness,
    _write_circuit_dot,
    _write_edges_csv,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_nodes(layers: list[int], n_heads: int = 2, mlps: bool = True) -> list[EdgeNode]:
    return _build_nodes(layers, n_heads, include_attention=True, include_mlps=mlps)


def _make_edges(nodes: list[EdgeNode]) -> list[CircuitEdge]:
    return _build_edges(nodes)


def _scored_edges(
    nodes: list[EdgeNode],
    importances: dict[str, float],
) -> list[CircuitEdge]:
    edges = _make_edges(nodes)
    for e in edges:
        e.importance = importances.get(e.edge_id, 0.0)
    return edges


# ---------------------------------------------------------------------------
# Edge graph construction
# ---------------------------------------------------------------------------


def test_build_nodes_attention_only() -> None:
    nodes = _build_nodes([0, 1], n_heads=3, include_attention=True, include_mlps=False)
    assert len(nodes) == 6  # 2 layers × 3 heads
    assert all(n.component == "attn" for n in nodes)
    assert {n.node_id for n in nodes} == {
        "L0.H0", "L0.H1", "L0.H2",
        "L1.H0", "L1.H1", "L1.H2",
    }


def test_build_nodes_mlps_only() -> None:
    nodes = _build_nodes([0, 1, 2], n_heads=2, include_attention=False, include_mlps=True)
    assert len(nodes) == 3
    assert all(n.component == "mlp" for n in nodes)


def test_build_edges_direction() -> None:
    """Every edge must have src.layer < dst.layer."""
    nodes = _make_nodes([0, 1, 2], n_heads=2, mlps=True)
    edges = _make_edges(nodes)
    for e in edges:
        assert e.src_layer < e.dst_layer, f"Bad edge: {e.edge_id}"


def test_build_edges_count() -> None:
    """Two layers, 2 heads + 1 MLP each = 3 nodes/layer.
    Edges: every node in layer 0 → every node in layer 1.  That's 3×3 = 9."""
    nodes = _make_nodes([0, 1], n_heads=2, mlps=True)
    edges = _make_edges(nodes)
    # Layer 0 has 3 nodes; layer 1 has 3 nodes.  All L0→L1 pairs.
    assert len(edges) == 9


def test_build_edges_three_layers() -> None:
    """Three layers, 2 heads + 1 MLP = 3 nodes/layer.
    Pairs: L0→L1 (9) + L0→L2 (9) + L1→L2 (9) = 27."""
    nodes = _make_nodes([0, 1, 2], n_heads=2, mlps=True)
    edges = _make_edges(nodes)
    assert len(edges) == 27


def test_edge_ids_are_unique() -> None:
    nodes = _make_nodes([0, 1, 2], n_heads=4, mlps=True)
    edges = _make_edges(nodes)
    ids = [e.edge_id for e in edges]
    assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# Pruning math
# ---------------------------------------------------------------------------


def test_pruning_removes_below_tau() -> None:
    nodes = _make_nodes([0, 1], n_heads=2, mlps=False)
    tau = 0.05
    edges = _scored_edges(
        nodes,
        {
            "L0.H0->L1.H0": 0.10,
            "L0.H0->L1.H1": 0.03,  # below tau
            "L0.H1->L1.H0": 0.00,  # below tau
            "L0.H1->L1.H1": 0.20,
        },
    )
    surviving = [e for e in edges if e.importance >= tau]
    assert len(surviving) == 2
    assert all(e.importance >= tau for e in surviving)


def test_pruning_all_survive() -> None:
    nodes = _make_nodes([0, 1], n_heads=2, mlps=False)
    edges = _scored_edges(
        nodes,
        {e.edge_id: 1.0 for e in _make_edges(nodes)},
    )
    surviving = [e for e in edges if e.importance >= 0.01]
    assert len(surviving) == len(edges)


def test_pruning_none_survive() -> None:
    nodes = _make_nodes([0, 1], n_heads=2, mlps=False)
    edges = _scored_edges(nodes, {})  # all importances = 0.0
    surviving = [e for e in edges if e.importance >= 0.05]
    assert len(surviving) == 0


def test_iterative_pruning_convergence() -> None:
    """Simulate the iterative loop: should converge in ≤ max_iterations."""
    nodes = _make_nodes([0, 1], n_heads=2, mlps=False)
    edges = _scored_edges(
        nodes,
        {"L0.H0->L1.H0": 0.10, "L0.H0->L1.H1": 0.01, "L0.H1->L1.H0": 0.0, "L0.H1->L1.H1": 0.5},
    )
    tau = 0.05
    history: list[EdgePruningStep] = []
    surviving = list(edges)
    for i in range(10):
        before = len(surviving)
        surviving = [e for e in surviving if e.importance >= tau]
        removed = before - len(surviving)
        history.append(
            EdgePruningStep(iteration=i, survivors=len(surviving), removed=removed, tau=tau)
        )
        if removed == 0:
            break
    # Should converge in 2 iterations (first pass removes 2, second removes 0).
    assert len(history) <= 2
    assert history[-1].removed == 0


# ---------------------------------------------------------------------------
# Faithfulness
# ---------------------------------------------------------------------------


def test_faithfulness_perfect() -> None:
    assert _faithfulness(1.0, 1.0) == pytest.approx(1.0)


def test_faithfulness_zero() -> None:
    # pruned_diff = 0 when full_diff = 1 → error = 1 → faithfulness = 0.
    assert _faithfulness(1.0, 0.0) == pytest.approx(0.0)


def test_faithfulness_clamped_nonnegative() -> None:
    # pruned_diff wildly off — should never go below 0.
    assert _faithfulness(1.0, 100.0) >= 0.0


def test_faithfulness_partial() -> None:
    f = _faithfulness(2.0, 1.0)
    assert 0.0 <= f <= 1.0
    assert f == pytest.approx(0.5)


def test_faithfulness_zero_full_diff() -> None:
    # Edge case: full_diff = 0 → denom clips to 1e-6, should not raise.
    f = _faithfulness(0.0, 0.0)
    assert f == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# JSON artifact
# ---------------------------------------------------------------------------


def _make_artifact(n_edges: int = 4) -> EdgeCircuitArtifact:
    nodes = _make_nodes([0, 1], n_heads=2, mlps=False)
    edges = _make_edges(nodes)[:n_edges]
    for i, e in enumerate(edges):
        e.importance = float(i) * 0.1
        e.pruned = i % 2 == 0
    return EdgeCircuitArtifact(
        model="gpt2-small",
        nodes=nodes,
        edges=edges,
        pruning_history=[EdgePruningStep(0, 2, 2, 0.05)],
        faithfulness=0.75,
        full_logit_diff=1.5,
        pruned_logit_diff=1.1,
    )


def test_json_artifact_round_trip(tmp_path: Path) -> None:
    artifact = _make_artifact()
    path = tmp_path / "edges.json"
    path.write_text(json.dumps(artifact.to_dict(), indent=2) + "\n", encoding="utf-8")
    loaded = json.loads(path.read_text())
    assert loaded["model"] == "gpt2-small"
    assert isinstance(loaded["edges"], list)
    assert isinstance(loaded["nodes"], list)
    assert isinstance(loaded["pruning_history"], list)
    assert "faithfulness" in loaded
    assert "full_logit_diff" in loaded


def test_json_edges_have_required_keys(tmp_path: Path) -> None:
    artifact = _make_artifact()
    data = artifact.to_dict()
    required = {"edge_id", "src_id", "dst_id", "src_layer", "dst_layer", "importance", "pruned"}
    for edge in data["edges"]:
        assert required.issubset(edge.keys()), f"Missing keys in edge: {edge}"


# ---------------------------------------------------------------------------
# CSV artifact
# ---------------------------------------------------------------------------


def test_csv_artifact_well_formed(tmp_path: Path) -> None:
    artifact = _make_artifact(4)
    path = tmp_path / "edges.csv"
    _write_edges_csv(path, artifact.edges)
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert len(rows) == 4
    assert rows[0]["rank"] == "1"
    expected_cols = {
        "rank", "edge_id", "src_id", "dst_id", "src_layer", "dst_layer", "importance", "pruned"
    }
    assert expected_cols.issubset(set(rows[0].keys()))


def test_csv_ranks_are_sequential(tmp_path: Path) -> None:
    artifact = _make_artifact(3)
    path = tmp_path / "edges.csv"
    _write_edges_csv(path, artifact.edges)
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        ranks = [int(row["rank"]) for row in reader]
    assert ranks == [1, 2, 3]


# ---------------------------------------------------------------------------
# DOT artifact
# ---------------------------------------------------------------------------


def test_dot_artifact_well_formed(tmp_path: Path) -> None:
    artifact = _make_artifact()
    path = tmp_path / "circuit.dot"
    _write_circuit_dot(path, artifact)
    content = path.read_text()
    assert "digraph" in content
    assert "rankdir=TB" in content
    assert "->" in content


def test_dot_pruned_edges_dashed(tmp_path: Path) -> None:
    artifact = _make_artifact()
    # Ensure at least one pruned and one surviving edge.
    assert any(e.pruned for e in artifact.edges)
    assert any(not e.pruned for e in artifact.edges)
    path = tmp_path / "circuit.dot"
    _write_circuit_dot(path, artifact)
    content = path.read_text()
    assert 'style="dashed"' in content


def test_dot_surviving_edges_have_color(tmp_path: Path) -> None:
    artifact = _make_artifact()
    path = tmp_path / "circuit.dot"
    _write_circuit_dot(path, artifact)
    content = path.read_text()
    # Surviving edges should have a hex colour (green family).
    assert "#00" in content


def test_dot_empty_edges(tmp_path: Path) -> None:
    """Zero edges should still produce a valid digraph."""
    artifact = EdgeCircuitArtifact(model="gpt2-small")
    path = tmp_path / "circuit.dot"
    _write_circuit_dot(path, artifact)
    content = path.read_text()
    assert "digraph" in content
    assert "}" in content
