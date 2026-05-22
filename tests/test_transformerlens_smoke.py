from pathlib import Path
from typing import Any

import pytest

from mech_interp.experiments.transformerlens_smoke import TransformerLensSmokeExperiment
from mech_interp.orchestration import ExperimentRunner
from mech_interp.storage import ArtifactStore, SQLiteResultStore
from mech_interp.types import ExperimentResult, ExperimentRun, ExperimentSpec, RunStatus, utc_now


class FakeActivationBackend:
    name = "transformerlens"

    def load(self) -> None:
        raise AssertionError("Smoke experiment should use capture_activations directly.")

    def capture_activations(self, prompts: list[str], sites: list[str]) -> dict[str, Any]:
        assert prompts == ["A", "B"]
        assert sites == ["blocks.0.hook_resid_pre", "blocks.0.mlp.hook_post"]
        return {"blocks.0.hook_resid_pre": object()}

    def run_intervention(self, prompt: str, interventions: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError


def test_transformerlens_smoke_experiment_summarizes_capture(tmp_path: Path) -> None:
    spec = ExperimentSpec(
        name="tl-smoke",
        family="smoke",
        backend="transformerlens",
        parameters={
            "prompts": ["A", "B"],
            "sites": ["blocks.0.hook_resid_pre", "blocks.0.mlp.hook_post"],
        },
    )
    run = ExperimentRun(
        id=1,
        spec_name=spec.name,
        family=spec.family,
        backend=spec.backend,
        status=RunStatus.RUNNING,
        artifact_dir=tmp_path,
        created_at=utc_now(),
    )

    result = TransformerLensSmokeExperiment(backend=FakeActivationBackend()).run(spec, run)

    assert result.status == RunStatus.SUCCEEDED
    assert result.metrics == {
        "prompt_count": 2.0,
        "requested_site_count": 2.0,
        "captured_site_count": 1.0,
        "missing_site_count": 1.0,
    }
    assert "blocks.0.mlp.hook_post" in result.notes


def test_runner_uses_transformerlens_smoke_only_when_explicit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeSmokeExperiment:
        def run(self, spec: ExperimentSpec, run: ExperimentRun) -> ExperimentResult:
            return TransformerLensSmokeExperiment(backend=FakeActivationBackend()).run(spec, run)

    monkeypatch.setattr(
        "mech_interp.orchestration.runner.TransformerLensSmokeExperiment",
        FakeSmokeExperiment,
    )
    spec = ExperimentSpec(
        name="tl-smoke",
        family="smoke",
        backend="transformerlens",
        parameters={
            "runner": "transformerlens_smoke",
            "prompts": ["A", "B"],
            "sites": ["blocks.0.hook_resid_pre", "blocks.0.mlp.hook_post"],
        },
    )
    runner = ExperimentRunner(
        result_store=SQLiteResultStore(tmp_path / "runs.sqlite3", tmp_path / "artifacts"),
        artifact_store=ArtifactStore(tmp_path / "artifacts"),
    )

    result = runner.run(spec)

    assert result.status == RunStatus.SUCCEEDED
    assert result.metrics["captured_site_count"] == 1.0
    assert (tmp_path / "artifacts" / "run-000001" / "result.json").exists()
