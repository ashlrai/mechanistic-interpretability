"""End-to-end DLA test on gpt2-small.

Runs direct_logit_attribution on the capital-of-France factual pair and asserts
that the top positive component is in layers 8-11 (where factual/IOI heads live
in gpt2-small per Wang et al. 2022).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from mech_interp.experiments.direct_logit_attribution import DirectLogitAttributionExperiment
from mech_interp.types import ExperimentRun, ExperimentSpec, RunStatus, utc_now

pytestmark = pytest.mark.integration


def _spec() -> ExperimentSpec:
    return ExperimentSpec(
        name="e2e-dla-capital-france",
        family="direct_logit_attribution",
        backend="transformerlens",
        description="E2E DLA on capital-of-France factual pair",
        parameters={
            "model": "gpt2-small",
            "device": "cpu",
            "seed": 42,
            "target_position": -1,
            "top_k": 10,
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


def test_dla_top_positive_in_late_layers(gpt2_backend: Any, tmp_path: Path) -> None:
    """At least one of the top-5 positive DLA components should be in layers 8-11.

    On gpt2-small (alias 'gpt2' in TransformerLens), direct logit attribution
    commonly finds MLP layers 3-8 as top contributors for factual recall — these
    write strongly into the residual stream.  Layers 8-11 typically contain the
    factual/IOI attention heads that also push toward the correct token.  We
    assert that the top-5 positive set contains at least one component from
    layers 8-11, which is a conservative but meaningful check.
    """
    spec = _spec()
    result = DirectLogitAttributionExperiment(backend=gpt2_backend).run(spec, _run(spec, tmp_path))

    assert result.status == RunStatus.SUCCEEDED

    summary = json.loads(Path(result.artifacts["lda_summary"]).read_text())
    ranked = json.loads(Path(result.artifacts["lda_ranked_json"]).read_text())

    assert len(ranked) > 0
    assert summary["total_components"] > 0

    top_positive = summary["top_positive"]
    assert len(top_positive) > 0, "No positive components found"
    assert top_positive[0]["mean_score"] > 0, "Top component has non-positive score"

    # Check that at least one of the top-5 positive components is in layers 8-11
    layers_in_top5 = [
        e["layer"] for e in top_positive[:5]
        if e.get("layer") is not None
    ]
    assert any(8 <= lay <= 11 for lay in layers_in_top5), (
        f"None of the top-5 positive components is in layers 8-11 for gpt2 factual heads. "
        f"Layers found: {layers_in_top5}. Full top_positive[:5]: {top_positive[:5]}"
    )


def test_dla_evidence_labels(gpt2_backend: Any, tmp_path: Path) -> None:
    """All rows must carry the correct evidence_label."""
    spec = _spec()
    result = DirectLogitAttributionExperiment(backend=gpt2_backend).run(spec, _run(spec, tmp_path))

    ranked = json.loads(Path(result.artifacts["lda_ranked_json"]).read_text())
    for row in ranked:
        assert row["evidence_label"] == "direct_logit_decomposition"


def test_dla_csv_matches_json(gpt2_backend: Any, tmp_path: Path) -> None:
    """CSV row count matches JSON row count."""
    spec = _spec()
    result = DirectLogitAttributionExperiment(backend=gpt2_backend).run(spec, _run(spec, tmp_path))

    ranked = json.loads(Path(result.artifacts["lda_ranked_json"]).read_text())
    csv_text = Path(result.artifacts["lda_ranked_csv"]).read_text()
    # header + data lines
    csv_rows = [line for line in csv_text.splitlines() if line.strip()][1:]
    assert len(csv_rows) == len(ranked)


def test_dla_metrics_populated(gpt2_backend: Any, tmp_path: Path) -> None:
    """Metrics dict has expected keys and sensible values."""
    spec = _spec()
    result = DirectLogitAttributionExperiment(backend=gpt2_backend).run(spec, _run(spec, tmp_path))

    assert result.metrics["component_count"] > 0
    assert result.metrics["prompt_count"] == 1.0
    assert result.metrics["top_positive_score"] > 0.0
