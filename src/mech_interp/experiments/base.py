from __future__ import annotations

from abc import ABC, abstractmethod

from mech_interp.types import ExperimentResult, ExperimentRun, ExperimentSpec


class Experiment(ABC):
    family: str

    @abstractmethod
    def run(self, spec: ExperimentSpec, run: ExperimentRun) -> ExperimentResult:
        """Execute an experiment spec and return persisted metrics/artifact references."""
