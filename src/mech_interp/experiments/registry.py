from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from mech_interp.experiments.base import ExperimentSpecSchema
from mech_interp.types import ExperimentSpec


class ExperimentSpecValidationError(ValueError):
    """Raised when an experiment YAML file cannot be loaded as a valid spec."""


class ExperimentRegistry:
    def __init__(self, specs: Iterable[ExperimentSpec] = ()) -> None:
        self._specs: dict[str, ExperimentSpec] = {}
        self._sources: dict[str, Path] = {}
        for spec in specs:
            self.register(spec)

    def register(self, spec: ExperimentSpec, source_path: str | Path | None = None) -> None:
        source = Path(source_path) if source_path is not None else None
        if spec.name in self._specs:
            existing = self._sources.get(spec.name)
            message = f"Duplicate experiment spec name '{spec.name}'"
            if source is not None:
                message += f" in {source}"
            if existing is not None:
                message += f"; first defined in {existing}"
            raise ExperimentSpecValidationError(message)
        self._specs[spec.name] = spec
        if source is not None:
            self._sources[spec.name] = source

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
        registry.register(load_experiment_spec(path), source_path=path)
    return registry


def load_experiment_spec(path: str | Path) -> ExperimentSpec:
    spec_path = Path(path)
    try:
        with spec_path.open("r", encoding="utf-8") as spec_file:
            raw: Any = yaml.safe_load(spec_file)
    except OSError as exc:
        raise ExperimentSpecValidationError(
            f"Unable to read experiment spec at {spec_path}: {exc}"
        ) from exc
    except yaml.YAMLError as exc:
        raise ExperimentSpecValidationError(
            f"Invalid experiment spec YAML at {spec_path}: {exc}"
        ) from exc

    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ExperimentSpecValidationError(
            f"Invalid experiment spec at {spec_path}: expected a YAML mapping at the top level"
        )

    try:
        return ExperimentSpecSchema.model_validate(raw).to_experiment_spec()
    except ValidationError as exc:
        raise ExperimentSpecValidationError(_format_validation_error(spec_path, exc)) from exc


def _format_validation_error(path: Path, error: ValidationError) -> str:
    details: list[str] = []
    for item in error.errors():
        location = ".".join(str(part) for part in item["loc"]) or "spec"
        details.append(f"{location}: {item['msg']}")
    return f"Invalid experiment spec at {path}: " + "; ".join(details)
