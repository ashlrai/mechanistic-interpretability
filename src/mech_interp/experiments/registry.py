from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import yaml

from mech_interp.types import ExperimentSpec


class ExperimentRegistry:
    def __init__(self, specs: Iterable[ExperimentSpec] = ()) -> None:
        self._specs = {spec.name: spec for spec in specs}

    def register(self, spec: ExperimentSpec) -> None:
        self._specs[spec.name] = spec

    def get(self, name: str) -> ExperimentSpec:
        return self._specs[name]

    def list(self) -> list[ExperimentSpec]:
        return sorted(self._specs.values(), key=lambda spec: spec.name)


def load_experiment_specs(directory: str | Path = "experiments") -> ExperimentRegistry:
    root = Path(directory)
    registry = ExperimentRegistry()
    if not root.exists():
        return registry

    for path in sorted(root.glob("*.yaml")):
        with path.open("r", encoding="utf-8") as spec_file:
            raw = yaml.safe_load(spec_file) or {}
        registry.register(ExperimentSpec(**raw))
    return registry
