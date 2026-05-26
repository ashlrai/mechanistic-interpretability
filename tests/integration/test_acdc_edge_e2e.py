"""End-to-end ACDC-edge smoke test on gpt2-small.

Scoped to layers 0-1, max_edges=24, max_iterations=2, one prompt pair —
small enough to run in under 60 seconds.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from mech_interp.experiments.acdc_edge import ACDCEdgeExperiment
from mech_interp.types import ExperimentRun, ExperimentSpec, RunStatus, utc_now

pytestmark = pytest.mark.integration


def _spec() -> ExperimentSpec:
    return ExperimentSpec(
        name="e2e-acdc-edge",
        family="acdc_edge",
        backend="transformerlens",
        description="",
        parameters={
            "model": "gpt2-small",
            "prompt_pairs": [
                {
                    "id": "capital-france",
                    "clean_prompt": "The capital of France is Paris",
                    "corrupted_prompt": "The capital of France is Rome",
                    "correct_token": " Paris",
                    "incorrect_token": " Rome",
                },
            ],
            "layers": [0, 1],
            "include_attention": True,
            "include_mlps": False,  # 12 heads × 2 layers = 24 nodes → 144 edges
            "max_edges": 24,  # cap to keep test fast
            "tau": 0.001,
            "max_iterations": 2,
            "ablation_type": "mean",
            "seed": 42,
            "device": "cpu",
        },
    )


def _run(spec: ExperimentSpec, tmp_path: Path) -> ExperimentRun:
    return ExperimentRun(
        id=1,
        spec_name=spec.name,
        family=spec.family,
        backend=spec.backend,
        status=RunStatus.RUNNING,
        artifact_dir=tmp_path,
        created_at=utc_now(),
    )


def test_acdc_edge_discovers_and_ranks_edges(gpt2_backend: Any, tmp_path: Path) -> None:
    spec = _spec()
    result = ACDCEdgeExperiment(backend=gpt2_backend).run(spec, _run(spec, tmp_path))

    assert result.status == RunStatus.SUCCEEDED
    metrics = result.metrics

    # Candidate edge count is capped at max_edges=24.
    assert metrics["candidate_edge_count"] == 24
    assert metrics["surviving_edge_count"] >= 0
    assert metrics["surviving_edge_count"] <= 24

    # Pruning history must be non-empty.
    assert metrics["pruning_iterations"] >= 1

    # Faithfulness must be in [0, 1].
    assert 0.0 <= metrics["faithfulness"] <= 1.0

    # Top edge should have a nonzero importance score (the ablation is doing something).
    assert metrics["top_edge_importance"] > 0.0

    # Verify JSON artifact is well-formed.
    edges_data = json.loads(Path(result.artifacts["edges_json"]).read_text())
    assert edges_data["model"] == "gpt2-small"
    assert isinstance(edges_data["edges"], list)
    assert len(edges_data["edges"]) == 24  # capped

    # Edges should be ranked descending by importance.
    importances = [e["importance"] for e in edges_data["edges"]]
    assert importances == sorted(importances, reverse=True)

    # Pruning history non-empty.
    assert len(edges_data["pruning_history"]) >= 1

    # Each edge has required keys.
    required = {"edge_id", "src_id", "dst_id", "src_layer", "dst_layer", "importance", "pruned"}
    for edge in edges_data["edges"]:
        assert required.issubset(edge.keys())

    # DOT file is valid GraphViz.
    dot = Path(result.artifacts["circuit_dot"]).read_text()
    assert "digraph" in dot
    assert "->" in dot
