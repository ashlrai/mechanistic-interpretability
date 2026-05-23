from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from mech_interp.experiments.activation_capture import (
    ActivationCaptureExperiment,
    summarize_activation,
)
from mech_interp.orchestration import ExperimentRunner
from mech_interp.storage import ArtifactStore, SQLiteResultStore
from mech_interp.types import (
    ActivationPatchRequest,
    ActivationPatchSiteResult,
    CrossModelProbeRequest,
    CrossModelProbeResult,
    ExperimentResult,
    ExperimentRun,
    ExperimentSpec,
    RunStatus,
    utc_now,
)


class FakeArray:
    shape = (2, 3)
    dtype = "float32"

    def numpy(self) -> list[list[float]]:
        return [[0.0, 1.0, 2.0], [0.0, 4.0, 5.0]]


class FakeTorchLikeArray:
    shape = (2,)
    dtype = "bfloat16"

    def detach(self) -> FakeTorchLikeArray:
        return self

    def cpu(self) -> FakeTorchLikeArray:
        return self

    def numpy(self) -> list[float]:
        return [1.0, 3.0]


class FakeActivationBackend:
    name = "fake"

    def load(self) -> None:
        raise AssertionError("Activation capture should call capture_activations directly.")

    def capture_activations(self, prompts: list[str], sites: list[str]) -> dict[str, Any]:
        assert prompts == ["A", "B"]
        assert sites == ["blocks.0.hook_resid_pre", "blocks.0.mlp.hook_post"]
        return {
            "blocks.0.hook_resid_pre": FakeArray(),
        }

    def run_intervention(self, prompt: str, interventions: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    def run_activation_patching(
        self,
        request: ActivationPatchRequest,
    ) -> list[ActivationPatchSiteResult]:
        raise NotImplementedError

    def run_cross_model_probe(
        self,
        request: CrossModelProbeRequest,
    ) -> list[CrossModelProbeResult]:
        raise NotImplementedError


def test_summarize_activation_uses_numpy_duck_typing() -> None:
    summary = summarize_activation(FakeArray())

    assert summary == {
        "shape": [2, 3],
        "dtype": "float32",
        "mean": 2.0,
        "std": pytest.approx(1.9148542155126762),
        "max": 5.0,
        "sparsity": pytest.approx(2 / 6),
    }


def test_summarize_activation_handles_torch_like_duck_type_without_torch() -> None:
    summary = summarize_activation(FakeTorchLikeArray())

    assert summary == {
        "shape": [2],
        "dtype": "bfloat16",
        "mean": 2.0,
        "std": 1.0,
        "max": 3.0,
        "sparsity": 0.0,
    }


def test_activation_capture_experiment_writes_summary_artifact(tmp_path: Path) -> None:
    spec = ExperimentSpec(
        name="activation-capture",
        family="polysemanticity",
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

    result = ActivationCaptureExperiment(backend=FakeActivationBackend()).run(spec, run)

    assert result.status == RunStatus.SUCCEEDED
    assert result.metrics == {
        "prompt_count": 2.0,
        "requested_site_count": 2.0,
        "captured_site_count": 1.0,
        "missing_site_count": 1.0,
    }
    artifact_path = Path(result.artifacts["activation_summary"])
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert payload["captured_sites"] == ["blocks.0.hook_resid_pre"]
    assert payload["missing_sites"] == ["blocks.0.mlp.hook_post"]
    assert payload["summaries"]["blocks.0.hook_resid_pre"]["shape"] == [2, 3]
    assert "blocks.0.mlp.hook_post" in result.notes


def test_runner_uses_activation_capture_when_explicit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeActivationCaptureExperiment:
        def run(self, spec: ExperimentSpec, run: ExperimentRun) -> ExperimentResult:
            return ActivationCaptureExperiment(backend=FakeActivationBackend()).run(spec, run)

    monkeypatch.setattr(
        "mech_interp.orchestration.runner.ActivationCaptureExperiment",
        FakeActivationCaptureExperiment,
    )
    spec = ExperimentSpec(
        name="activation-capture",
        family="polysemanticity",
        backend="transformerlens",
        parameters={
            "runner": "activation_capture",
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
    assert "activation_summary" in result.artifacts
    assert (tmp_path / "artifacts" / "run-000001" / "result.json").exists()
    manifest = runner.artifact_store.read_manifest(1)
    manifest_names = {artifact["name"] for artifact in manifest["artifacts"]}
    assert "activation_summary" in manifest_names
