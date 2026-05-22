from __future__ import annotations

from dataclasses import asdict

from mech_interp.experiments.base import Experiment
from mech_interp.experiments.placeholder import SpecValidationExperiment
from mech_interp.experiments.transformerlens_smoke import TransformerLensSmokeExperiment
from mech_interp.storage import ArtifactStore, SQLiteResultStore
from mech_interp.types import ExperimentResult, ExperimentSpec, RunStatus


class ExperimentRunner:
    def __init__(self, result_store: SQLiteResultStore, artifact_store: ArtifactStore) -> None:
        self.result_store = result_store
        self.artifact_store = artifact_store

    def run(self, spec: ExperimentSpec) -> ExperimentResult:
        run = self.result_store.create_run(spec)
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
    if spec.parameters.get("runner") == "transformerlens_smoke":
        return TransformerLensSmokeExperiment()
    return SpecValidationExperiment(spec.family)
