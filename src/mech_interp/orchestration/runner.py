from __future__ import annotations

import hashlib
from dataclasses import asdict
from pathlib import Path

from mech_interp.experiments.activation_capture import ActivationCaptureExperiment
from mech_interp.experiments.base import Experiment
from mech_interp.experiments.circuit_patching import CircuitPatchingExperiment
from mech_interp.experiments.cross_model_representation_probe import (
    CrossModelRepresentationProbeExperiment,
)
from mech_interp.experiments.placeholder import SpecValidationExperiment
from mech_interp.experiments.transformerlens_smoke import TransformerLensSmokeExperiment
from mech_interp.storage import ArtifactStore, SQLiteResultStore
from mech_interp.types import ArtifactRecord, ExperimentResult, ExperimentSpec, RunStatus


class ExperimentRunner:
    def __init__(self, result_store: SQLiteResultStore, artifact_store: ArtifactStore) -> None:
        self.result_store = result_store
        self.artifact_store = artifact_store

    def run(self, spec: ExperimentSpec) -> ExperimentResult:
        run = self.result_store.create_run(spec)
        self.result_store.update_run_status(run.id, RunStatus.RUNNING)
        records = [
            self.artifact_store.write_json(
                run.id,
                "spec.json",
                {
                    "name": spec.name,
                    "family": spec.family,
                    "backend": spec.backend,
                    "description": spec.description,
                    "parameters": spec.parameters,
                },
            )
        ]

        try:
            experiment = experiment_for_spec(spec)
            result = experiment.run(spec, run)
            records.extend(_artifact_records_from_result(result))
        except Exception as exc:
            result = ExperimentResult(
                run_id=run.id,
                status=RunStatus.FAILED,
                notes=f"{type(exc).__name__}: {exc}",
            )

        records.append(
            self.artifact_store.write_json(
                run.id,
                "result.json",
                {
                    "run_id": result.run_id,
                    "status": result.status.value,
                    "metrics": result.metrics,
                    "artifacts": result.artifacts,
                    "notes": result.notes,
                },
            )
        )
        manifest = self.artifact_store.write_manifest(run.id, records)
        result = ExperimentResult(
            run_id=result.run_id,
            status=result.status,
            metrics=result.metrics,
            artifacts={**result.artifacts, "manifest": str(manifest.path)},
            notes=result.notes,
        )
        self.result_store.save_result(result)
        return result

    def run_many(self, specs: list[ExperimentSpec]) -> list[ExperimentResult]:
        return [self.run(spec) for spec in specs]


def result_to_row(result: ExperimentResult) -> dict[str, object]:
    row = asdict(result)
    row["status"] = result.status.value
    return row


def experiment_for_spec(spec: ExperimentSpec) -> Experiment:
    if spec.family == "circuit_patching" or spec.parameters.get("runner") == "circuit_patching":
        return CircuitPatchingExperiment()
    if spec.family == "cross_model_representation_probe":
        return CrossModelRepresentationProbeExperiment()
    if spec.parameters.get("runner") == "activation_capture":
        return ActivationCaptureExperiment()
    if spec.parameters.get("runner") == "transformerlens_smoke":
        return TransformerLensSmokeExperiment()
    return SpecValidationExperiment(spec.family)


def _artifact_records_from_result(result: ExperimentResult) -> list[ArtifactRecord]:
    records: list[ArtifactRecord] = []
    for name, value in result.artifacts.items():
        path = Path(value)
        if not path.is_file():
            continue
        content = path.read_bytes()
        records.append(
            ArtifactRecord(
                name=path.name if name == path.name else name,
                path=path,
                media_type=_media_type(path),
                sha256=hashlib.sha256(content).hexdigest(),
                size_bytes=len(content),
            )
        )
    return records


def _media_type(path: Path) -> str:
    if path.suffix == ".json":
        return "application/json"
    if path.suffix in {".txt", ".md"}:
        return "text/plain"
    return "application/octet-stream"
