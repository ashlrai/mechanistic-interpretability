from __future__ import annotations

from mech_interp.experiments.base import Experiment
from mech_interp.types import ExperimentResult, ExperimentRun, ExperimentSpec, RunStatus


class SpecValidationExperiment(Experiment):
    """Metadata-only experiment used until each research family has a real implementation."""

    def __init__(self, family: str) -> None:
        self.family = family

    def run(self, spec: ExperimentSpec, run: ExperimentRun) -> ExperimentResult:
        parameter_count = float(len(spec.parameters))
        sequence_count = float(
            sum(1 for value in spec.parameters.values() if isinstance(value, list))
        )
        return ExperimentResult(
            run_id=run.id,
            status=RunStatus.SUCCEEDED,
            metrics={
                "parameter_count": parameter_count,
                "sequence_parameter_count": sequence_count,
            },
            notes=(
                "Spec validation run succeeded. This placeholder confirms orchestration, "
                "storage, and artifact persistence before model-backed execution is enabled."
            ),
        )
