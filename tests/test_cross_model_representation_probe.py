from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from mech_interp.experiments.cross_model_representation_probe import (
    CrossModelRepresentationProbeExperiment,
)
from mech_interp.orchestration import ExperimentRunner
from mech_interp.storage import ArtifactStore, SQLiteResultStore
from mech_interp.types import (
    ActivationPatchRequest,
    ActivationPatchSiteResult,
    CrossModelProbeRecord,
    CrossModelProbeRequest,
    CrossModelProbeResult,
    ExperimentResult,
    ExperimentRun,
    ExperimentSpec,
    RunStatus,
    utc_now,
)


class FakeCrossModelBackend:
    name = "transformerlens"

    def __init__(self) -> None:
        self.request: CrossModelProbeRequest | None = None
        self.last_probe_weights_path: str | None = None
        self.last_probe_weights: np.ndarray | None = None

    def load(self) -> None:
        raise AssertionError("Cross-model probe should call run_cross_model_probe directly.")

    def capture_activations(self, prompts: list[str], sites: list[str]) -> dict[str, Any]:
        raise NotImplementedError

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
        self.request = request
        return [
            CrossModelProbeResult(
                source_hook_site=request.source_hook_site,
                target_hook_site=request.target_hook_site,
                split="train",
                record_count=2,
                mean_cosine_similarity=0.99,
                normalized_mse=0.01,
                variance_explained=0.98,
            ),
            CrossModelProbeResult(
                source_hook_site=request.source_hook_site,
                target_hook_site=request.target_hook_site,
                split="eval",
                record_count=1,
                mean_cosine_similarity=0.8,
                normalized_mse=0.2,
                variance_explained=0.7,
            ),
        ]


def test_cross_model_probe_writes_artifacts_and_no_weights_by_default(tmp_path: Path) -> None:
    backend = FakeCrossModelBackend()
    spec = _spec()
    run = ExperimentRun(
        id=1,
        spec_name=spec.name,
        family=spec.family,
        backend=spec.backend,
        status=RunStatus.RUNNING,
        artifact_dir=tmp_path,
        created_at=utc_now(),
    )

    result = CrossModelRepresentationProbeExperiment(backend=backend).run(spec, run)

    assert result.status == RunStatus.SUCCEEDED
    assert result.metrics["eval_mean_cosine_similarity"] == 0.8
    assert "probe_weights" not in result.artifacts
    assert backend.request is not None
    assert backend.request.source_hook_site == "blocks.0.hook_resid_pre"
    summary = json.loads(
        Path(result.artifacts["cross_model_probe_summary"]).read_text(encoding="utf-8")
    )
    rows = json.loads(
        Path(result.artifacts["cross_model_probe_results_json"]).read_text(encoding="utf-8")
    )
    report = Path(result.artifacts["research_note"]).read_text(encoding="utf-8")
    assert summary["train_record_count"] == 2
    assert summary["evidence_label"] == "correlational alignment"
    assert rows[0]["rank"] == 1
    assert rows[0]["evidence_label"] == "correlational alignment"
    assert "does not establish causal interchangeability" in report


def test_cross_model_probe_labels_activation_verbalization_as_hypothesis(
    tmp_path: Path,
) -> None:
    backend = FakeCrossModelBackend()
    spec = _spec(
        {
            "artifact_policy": {
                "activation_verbalization": True,
                "max_verbalized_records": 2,
                "write_report": False,
            }
        }
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

    result = CrossModelRepresentationProbeExperiment(backend=backend).run(spec, run)

    assert backend.request is not None
    assert backend.request.max_verbalized_records == 2
    verbalizations = json.loads(
        Path(result.artifacts["activation_verbalization"]).read_text(encoding="utf-8")
    )
    assert [row["evidence_label"] for row in verbalizations] == [
        "hypothesis",
        "hypothesis",
    ]
    assert "research_note" not in result.artifacts


def test_cross_model_probe_writes_weights_only_when_requested(tmp_path: Path) -> None:
    backend = FakeCrossModelBackend()
    backend.last_probe_weights = np.array([[1.0, 2.0], [3.0, 4.0]])
    spec = _spec({"artifact_policy": {"retain_probe_weights": True, "write_report": False}})
    run = ExperimentRun(
        id=1,
        spec_name=spec.name,
        family=spec.family,
        backend=spec.backend,
        status=RunStatus.RUNNING,
        artifact_dir=tmp_path,
        created_at=utc_now(),
    )

    result = CrossModelRepresentationProbeExperiment(backend=backend).run(spec, run)

    assert "probe_weights" in result.artifacts
    weights_path = Path(result.artifacts["probe_weights"])
    assert weights_path.name == "probe_weights.npz"
    with np.load(weights_path, allow_pickle=False) as archive:
        assert archive["weights"].shape == (2, 2)


def test_cross_model_probe_dataset_hash_mismatch_fails(tmp_path: Path) -> None:
    dataset = tmp_path / "records.jsonl"
    dataset.write_text('{"split":"train","prompt":"a"}\n', encoding="utf-8")
    spec = _spec({"records": None, "dataset_path": str(dataset), "dataset_sha256": "wrong"})
    run = ExperimentRun(
        1,
        spec.name,
        spec.family,
        spec.backend,
        RunStatus.RUNNING,
        tmp_path,
        utc_now(),
    )

    with pytest.raises(ValueError, match="dataset hash mismatch"):
        CrossModelRepresentationProbeExperiment(backend=FakeCrossModelBackend()).run(spec, run)


def test_cross_model_probe_requires_train_and_eval(tmp_path: Path) -> None:
    spec = _spec({"records": [{"split": "train", "prompt": "only train"}]})
    run = ExperimentRun(
        1,
        spec.name,
        spec.family,
        spec.backend,
        RunStatus.RUNNING,
        tmp_path,
        utc_now(),
    )

    with pytest.raises(ValueError, match="eval record"):
        CrossModelRepresentationProbeExperiment(backend=FakeCrossModelBackend()).run(spec, run)


def test_runner_dispatches_cross_model_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeExperiment:
        def run(self, spec: ExperimentSpec, run: ExperimentRun) -> ExperimentResult:
            return CrossModelRepresentationProbeExperiment(backend=FakeCrossModelBackend()).run(
                spec,
                run,
            )

    monkeypatch.setattr(
        "mech_interp.orchestration.runner.CrossModelRepresentationProbeExperiment",
        FakeExperiment,
    )
    runner = ExperimentRunner(
        result_store=SQLiteResultStore(tmp_path / "runs.sqlite3", tmp_path / "artifacts"),
        artifact_store=ArtifactStore(tmp_path / "artifacts"),
    )

    result = runner.run(_spec())

    assert result.status == RunStatus.SUCCEEDED
    assert "cross_model_probe_summary" in result.artifacts


def test_probe_math_with_fake_activation_model() -> None:
    from mech_interp.backends.instrumented import _fit_and_score_probe

    request = CrossModelProbeRequest(
        source_model_name="source",
        target_model_name="target",
        records=(
            *[
                _record("train", f"train-{index}")
                for index in range(3)
            ],
            _record("eval", "eval-1"),
        ),
        source_hook_site="source.site",
        target_hook_site="target.site",
        ridge_alpha=0.01,
    )
    source = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0]])
    target = source @ np.array([[2.0, 0.0], [0.0, 3.0]])

    results = _fit_and_score_probe(request, source, target)

    assert [result.split for result in results] == ["train", "eval"]
    assert results[1].mean_cosine_similarity > 0.99


def _spec(overrides: dict[str, Any] | None = None) -> ExperimentSpec:
    parameters: dict[str, Any] = {
        "source_model": "source",
        "target_model": "target",
        "source_hook_site": "blocks.0.hook_resid_pre",
        "target_hook_site": "blocks.0.hook_resid_pre",
        "records": [
            {"id": "train-1", "split": "train", "prompt": "a"},
            {"id": "train-2", "split": "train", "prompt": "b"},
            {"id": "eval-1", "split": "eval", "prompt": "c"},
        ],
    }
    if overrides:
        parameters.update(overrides)
    return ExperimentSpec(
        name="cross",
        family="cross_model_representation_probe",
        backend="transformerlens",
        parameters=parameters,
    )


def _record(split: str, record_id: str) -> CrossModelProbeRecord:
    return CrossModelProbeRecord(id=record_id, split=split, prompt=record_id)
