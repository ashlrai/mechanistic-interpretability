"""End-to-end ACDC-lite smoke test on gpt2-small.

Scoped to layers 0-1 with `include_mlps=False` so we evaluate ~24 candidate
nodes — small enough to run in under 30 seconds, large enough to verify the
pruning pipeline produces meaningful output.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from mech_interp.experiments.acdc_lite import ACDCLiteExperiment
from mech_interp.types import ExperimentRun, ExperimentSpec, RunStatus, utc_now

pytestmark = pytest.mark.integration


def _spec() -> ExperimentSpec:
    return ExperimentSpec(
        name="e2e-acdc",
        family="acdc_lite",
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
            "include_mlps": False,
            "tau": 0.001,
            "max_iterations": 3,
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


def test_acdc_lite_discovers_and_ranks_nodes(gpt2_backend: Any, tmp_path: Path) -> None:
    spec = _spec()
    result = ACDCLiteExperiment(backend=gpt2_backend).run(spec, _run(spec, tmp_path))

    assert result.status == RunStatus.SUCCEEDED
    metrics = result.metrics
    # gpt2-small layers 0-1, attention only → 2 layers × 12 heads = 24 nodes.
    assert metrics["candidate_node_count"] == 24
    assert metrics["surviving_node_count"] > 0
    assert metrics["surviving_node_count"] <= 24
    assert metrics["pruning_iterations"] >= 1
    # Faithfulness in [0, 1].
    assert 0.0 <= metrics["faithfulness"] <= 1.0

    circuit = json.loads(Path(result.artifacts["circuit_json"]).read_text())
    assert circuit["model"] == "gpt2-small"
    assert len(circuit["nodes"]) == 24
    # Ranked descending by importance.
    importances = [n["importance"] for n in circuit["nodes"]]
    assert importances == sorted(importances, reverse=True)
    # At least one node should have nonzero importance (otherwise the ablation
    # hook isn't doing anything).
    assert importances[0] > 0.0

    dot = Path(result.artifacts["circuit_dot"]).read_text()
    assert "digraph" in dot
    assert "L0.H0" in dot
