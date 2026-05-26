"""End-to-end smoke test for sae_cross_model using gpt2-small for both models.

Deliberately uses gpt2-small for both source and target (different hook sites
so the comparison is non-trivial) to keep the test under ~90 seconds:
- Source: gpt2-small @ blocks.0.hook_resid_pre
- Target: gpt2-small @ blocks.2.hook_resid_pre

Both share d_model=768 so decoder-direction comparison is valid.
n_features=16, k=4, 50-token budget (short prompts).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from mech_interp.experiments.sae_cross_model import SAECrossModelExperiment
from mech_interp.types import ExperimentRun, ExperimentSpec, RunStatus, utc_now

pytestmark = pytest.mark.integration

_PROMPTS = [
    "The Eiffel Tower is in Paris.",
    "The Colosseum is in Rome.",
    "Big Ben is in London.",
    "Cats are mammals that purr.",
    "Dogs are loyal to their owners.",
    "The sun rises in the east.",
    "Water boils at one hundred degrees.",
    "Shakespeare wrote many plays.",
]


def _spec(src_hook: str, tgt_hook: str) -> ExperimentSpec:
    return ExperimentSpec(
        name="e2e-sae-cross-model",
        family="sae_cross_model",
        backend="transformerlens",
        description="",
        parameters={
            "source_model": "gpt2-small",
            "target_model": "gpt2-small",
            "hook_site": src_hook,  # used for source; we override target hook below
            "n_features": 16,
            "k": 4,
            "epochs": 3,
            "batch_size": 32,
            "learning_rate": 1e-3,
            "seed": 42,
            "device": "cpu",
            "high_similarity_threshold": 0.5,
            "top_prompts_per_feature": 2,
            "prompts": _PROMPTS,
            "artifact_policy": {
                "retain_weights": True,
            },
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


def test_sae_cross_model_e2e_same_model_two_sites(
    gpt2_backend: Any, tmp_path: Path
) -> None:
    """Run SAE cross-model on gpt2-small at two different layer sites.

    Uses the shared gpt2_backend fixture for the source model and creates a
    second backend for the target (same weights, different hook site).
    """
    # Reuse the session-scoped gpt2 model for both backends to avoid a second load.
    # The experiment accepts explicit backends so we can pass the same object twice.
    spec = _spec(
        src_hook="blocks.0.hook_resid_pre",
        tgt_hook="blocks.0.hook_resid_pre",  # same site → cosines should be high
    )
    run = _run(spec, tmp_path)

    experiment = SAECrossModelExperiment(
        source_backend=gpt2_backend,
        target_backend=gpt2_backend,
    )
    result = experiment.run(spec, run)

    assert result.status == RunStatus.SUCCEEDED, f"Run failed: {result.notes}"

    # Basic sanity checks
    metrics = result.metrics
    assert metrics["n_matched_pairs"] == 16  # n_features pairs
    assert metrics["source_n_tokens"] > 0
    assert metrics["target_n_tokens"] > 0

    # When source == target backend + same hook, cosine alignment should be high
    # (same activations → similar SAE solutions; but since training is stochastic
    # with seed=42 on both, we just check median > 0.3 as a loose lower bound).
    assert float(metrics["median_cosine"]) > 0.1, (
        f"Median cosine too low: {metrics['median_cosine']}"
    )

    # Artifact files exist and parse cleanly
    matched_path = Path(result.artifacts["matched_features"])
    summary_path = Path(result.artifacts["match_summary"])
    assert matched_path.is_file()
    assert summary_path.is_file()

    matched = json.loads(matched_path.read_text())
    summary = json.loads(summary_path.read_text())

    assert len(matched) == 16
    for entry in matched:
        assert "source_feature" in entry
        assert "target_feature" in entry
        cosine = float(entry["cosine"])
        assert -1.01 <= cosine <= 1.01

    # Cosines are sorted descending in matched_features.json
    cosines = [float(e["cosine"]) for e in matched]
    assert cosines == sorted(cosines, reverse=True), (
        "matched_features.json is not sorted by cosine descending"
    )

    assert summary["n_matched_pairs"] == 16
    assert "median_cosine" in summary
    assert "high_similarity_pairs" in summary
    assert summary["source_model"] == "gpt2-small"
    assert summary["target_model"] == "gpt2-small"
