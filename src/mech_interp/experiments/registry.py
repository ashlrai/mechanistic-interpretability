from __future__ import annotations

import copy
import hashlib
import itertools
import json
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
        for spec in load_experiment_specs_from_file(path):
            registry.register(spec, source_path=path)
    return registry


def load_experiment_spec(path: str | Path) -> ExperimentSpec:
    specs = load_experiment_specs_from_file(path)
    if len(specs) != 1:
        raise ExperimentSpecValidationError(
            f"Experiment spec at {path} expands to {len(specs)} specs; use load_experiment_specs."
        )
    return specs[0]


def load_experiment_specs_from_file(path: str | Path) -> list[ExperimentSpec]:
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
    if "matrix" in raw:
        return _expand_matrix_spec(spec_path, raw)

    try:
        return [ExperimentSpecSchema.model_validate(raw).to_experiment_spec()]
    except ValidationError as exc:
        raise ExperimentSpecValidationError(_format_validation_error(spec_path, exc)) from exc


def _expand_matrix_spec(path: Path, raw: dict[str, Any]) -> list[ExperimentSpec]:
    matrix = raw.get("matrix")
    if not isinstance(matrix, dict) or not matrix:
        raise ExperimentSpecValidationError(f"Invalid experiment matrix at {path}: matrix is empty")

    base = {key: copy.deepcopy(value) for key, value in raw.items() if key != "matrix"}
    if "name" not in base:
        raise ExperimentSpecValidationError(
            f"Invalid experiment matrix at {path}: name is required"
        )

    axes: list[tuple[str, list[Any]]] = []
    for key, values in sorted(matrix.items()):
        if not isinstance(values, list) or not values:
            raise ExperimentSpecValidationError(
                f"Invalid experiment matrix at {path}: axis '{key}' must be a non-empty list"
            )
        _validate_unique_axis_values(path, key, values)
        axes.append((key, values))

    specs: list[ExperimentSpec] = []
    seen_names: set[str] = set()
    seen_hashes: set[str] = set()
    for values in itertools.product(*(axis_values for _, axis_values in axes)):
        materialized = copy.deepcopy(base)
        parameters = dict(materialized.get("parameters") or {})
        axis_payload = dict(zip((axis_name for axis_name, _ in axes), values, strict=True))
        for axis_name, value in axis_payload.items():
            _set_axis_value(materialized, parameters, axis_name, value)
        materialized["parameters"] = parameters
        digest = _matrix_payload_hash(axis_payload)
        suffix = digest[:12]
        materialized["name"] = f"{base['name']}-{suffix}"
        parameters["matrix"] = str(base["name"])
        parameters["matrix_axes"] = copy.deepcopy(axis_payload)
        parameters["generated_spec_hash"] = digest
        if materialized["name"] in seen_names or digest in seen_hashes:
            raise ExperimentSpecValidationError(
                f"Invalid experiment matrix at {path}: duplicate generated spec "
                f"for hash {digest}"
            )
        seen_names.add(str(materialized["name"]))
        seen_hashes.add(digest)
        try:
            specs.append(ExperimentSpecSchema.model_validate(materialized).to_experiment_spec())
        except ValidationError as exc:
            raise ExperimentSpecValidationError(_format_validation_error(path, exc)) from exc
    return specs


def _set_axis_value(
    materialized: dict[str, Any],
    parameters: dict[str, Any],
    axis_name: str,
    value: Any,
) -> None:
    top_level = {"family", "backend", "description", "seed", "model", "prompts", "artifact_policy"}
    if axis_name in top_level:
        materialized[axis_name] = value
        return
    if axis_name.startswith("parameters."):
        _set_dotted_value(parameters, axis_name.removeprefix("parameters."), value)
        return
    parameters[axis_name] = value


def _set_dotted_value(target: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = [part for part in dotted_key.split(".") if part]
    if not parts:
        return
    cursor = target
    for part in parts[:-1]:
        existing = cursor.get(part)
        if not isinstance(existing, dict):
            existing = {}
            cursor[part] = existing
        cursor = existing
    cursor[parts[-1]] = value


def _validate_unique_axis_values(path: Path, axis_name: str, values: list[Any]) -> None:
    seen: set[str] = set()
    duplicates: list[str] = []
    for value in values:
        canonical = _canonical_json(value)
        if canonical in seen:
            duplicates.append(canonical)
        seen.add(canonical)
    if duplicates:
        raise ExperimentSpecValidationError(
            f"Invalid experiment matrix at {path}: axis '{axis_name}' contains "
            "duplicate value(s)"
        )


def _matrix_payload_hash(axis_payload: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(axis_payload).encode("utf-8")).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(_canonicalize(value), sort_keys=True, separators=(",", ":"))


def _canonicalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _canonicalize(item)
            for key, item in sorted(value.items(), key=lambda entry: str(entry[0]))
        }
    if isinstance(value, list):
        return [_canonicalize(item) for item in value]
    if isinstance(value, tuple):
        return [_canonicalize(item) for item in value]
    return value


def _format_validation_error(path: Path, error: ValidationError) -> str:
    details: list[str] = []
    for item in error.errors():
        location = ".".join(str(part) for part in item["loc"]) or "spec"
        details.append(f"{location}: {item['msg']}")
    return f"Invalid experiment spec at {path}: " + "; ".join(details)
