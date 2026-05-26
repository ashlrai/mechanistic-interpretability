"""End-to-end SAE training smoke test on gpt2-small layer-0 residual stream."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from mech_interp.experiments.polysemanticity_sae import PolysemanticitySAEExperiment
from mech_interp.types import ExperimentRun, ExperimentSpec, RunStatus, utc_now

pytestmark = pytest.mark.integration


def _spec(prompts: list[str]) -> ExperimentSpec:
    return ExperimentSpec(
        name="e2e-sae",
        family="polysemanticity_sae",
        backend="transformerlens",
        description="",
        parameters={
            "model": "gpt2-small",
            "hook_site": "blocks.0.hook_resid_pre",
            "n_features": 64,
            "k": 8,
            "epochs": 3,
            "batch_size": 32,
            "learning_rate": 1e-3,
            "seed": 42,
            "device": "cpu",
            "prompts": prompts,
            "artifact_policy": {
                "retain_weights": True,
                "write_feature_analysis": True,
                "top_prompts_per_feature": 3,
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


def test_sae_trains_and_writes_artifacts(gpt2_backend: Any, tmp_path: Path) -> None:
    prompts = [
        "The Eiffel Tower is in Paris.",
        "The Colosseum is in Rome.",
        "Big Ben is in London.",
        "Cats are mammals.",
        "Dogs are loyal.",
    ]
    spec = _spec(prompts)
    result = PolysemanticitySAEExperiment(backend=gpt2_backend).run(spec, _run(spec, tmp_path))

    assert result.status == RunStatus.SUCCEEDED
    metrics = result.metrics
    # Training must actually reduce the reconstruction loss.
    assert metrics["final_loss"] < metrics["initial_loss"], (
        f"loss did not decrease: {metrics['initial_loss']} -> {metrics['final_loss']}"
    )
    # Top-k = 8 → mean features per token should equal k.
    assert metrics["mean_features_per_token"] == pytest.approx(8.0, rel=0.01)
    # Some features must come alive (not all dead).
    assert metrics["live_features"] > 0
    # Sanity: artifact files exist and parse.
    weights = Path(result.artifacts["sae_weights"])
    analysis = Path(result.artifacts["feature_analysis"])
    history = Path(result.artifacts["training_history"])
    assert weights.is_file() and weights.stat().st_size > 0
    parsed_analysis = json.loads(analysis.read_text())
    assert parsed_analysis["n_features"] == 64
    parsed_history = json.loads(history.read_text())
    assert len(parsed_history["losses_per_epoch"]) == 3


def test_sae_run_is_deterministic_given_same_seed(gpt2_backend: Any, tmp_path: Path) -> None:
    prompts = ["The capital of France is Paris.", "The capital of Italy is Rome."]
    spec = _spec(prompts)

    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()

    first = PolysemanticitySAEExperiment(backend=gpt2_backend).run(spec, _run(spec, first_dir))
    second = PolysemanticitySAEExperiment(backend=gpt2_backend).run(spec, _run(spec, second_dir))

    assert first.metrics["final_loss"] == pytest.approx(second.metrics["final_loss"], abs=1e-6)
    assert first.metrics["dead_features"] == second.metrics["dead_features"]
