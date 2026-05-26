from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import numpy as np

from mech_interp.backends import create_instrumented_backend
from mech_interp.experiments.base import Experiment
from mech_interp.storage.artifacts import resolve_run_artifact_dir
from mech_interp.types import (
    ExperimentResult,
    ExperimentRun,
    ExperimentSpec,
    InstrumentedModelBackend,
    RunStatus,
)


class ActivationCaptureExperiment(Experiment):
    family = "activation_capture"

    def __init__(self, backend: InstrumentedModelBackend | None = None) -> None:
        self.backend = backend

    def run(self, spec: ExperimentSpec, run: ExperimentRun) -> ExperimentResult:
        prompts = _string_list_parameter(spec, "prompts")
        sites = _string_list_parameter(spec, "sites")
        backend = self.backend or create_instrumented_backend(
            spec.backend,
            _backend_config(spec),
        )

        activations = backend.capture_activations(prompts, sites)
        captured_sites = [site for site in sites if site in activations]
        missing_sites = [site for site in sites if site not in activations]
        summaries = {
            site: summarize_activation(activations[site])
            for site in captured_sites
        }

        artifact_path = (resolve_run_artifact_dir(run) / "activation_summary.json").resolve()
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_text(
            json.dumps(
                {
                    "prompts": prompts,
                    "requested_sites": sites,
                    "captured_sites": captured_sites,
                    "missing_sites": missing_sites,
                    "summaries": summaries,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        return ExperimentResult(
            run_id=run.id,
            status=RunStatus.SUCCEEDED,
            metrics={
                "prompt_count": float(len(prompts)),
                "requested_site_count": float(len(sites)),
                "captured_site_count": float(len(captured_sites)),
                "missing_site_count": float(len(missing_sites)),
            },
            artifacts={
                "activation_summary": str(artifact_path),
                "captured_sites": ",".join(captured_sites),
                "missing_sites": ",".join(missing_sites),
            },
            notes=(
                f"Captured activation summaries for sites: {captured_sites}."
                if not missing_sites
                else f"Captured activation summaries with missing sites: {missing_sites}."
            ),
        )


def summarize_activation(value: Any) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "shape": _shape(value),
        "dtype": _dtype(value),
    }
    array = _as_numpy_array(value)
    if array is None or array.size == 0:
        return summary

    try:
        numeric = array.astype(float, copy=False)
    except (TypeError, ValueError):
        return summary

    summary.update(
        {
            "mean": float(np.mean(numeric)),
            "std": float(np.std(numeric)),
            "max": float(np.max(numeric)),
            "sparsity": float(np.mean(numeric == 0)),
        }
    )
    return summary


def _backend_config(spec: ExperimentSpec) -> dict[str, Any]:
    config = spec.parameters.get("backend_config", {})
    if config is not None and not isinstance(config, dict):
        raise ValueError("Activation capture parameter 'backend_config' must be a mapping.")

    backend_config = dict(config or {})
    if "model_name" in spec.parameters:
        backend_config["model_name"] = spec.parameters["model_name"]
    if "model" in spec.parameters:
        backend_config["model_name"] = spec.parameters["model"]
    if "device" in spec.parameters:
        backend_config["device"] = spec.parameters["device"]
    if "model_path" in spec.parameters:
        backend_config["model_path"] = spec.parameters["model_path"]
    return backend_config


def _string_list_parameter(spec: ExperimentSpec, name: str) -> list[str]:
    value = spec.parameters.get(name)
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and value and all(isinstance(item, str) for item in value):
        return value
    raise ValueError(
        f"Activation capture parameter '{name}' must be a string or non-empty list[str]."
    )


def _shape(value: Any) -> list[int | str] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    if isinstance(shape, int):
        return [shape]
    if isinstance(shape, Sequence):
        try:
            return [int(dim) for dim in shape]
        except (TypeError, ValueError):
            return [str(dim) for dim in shape]
    return None


def _dtype(value: Any) -> str | None:
    dtype = getattr(value, "dtype", None)
    return None if dtype is None else str(dtype)


def _as_numpy_array(value: Any) -> np.ndarray[Any, Any] | None:
    converted = value
    for method_name in ("detach", "cpu"):
        method = getattr(converted, method_name, None)
        if callable(method):
            converted = method()

    numpy_method = getattr(converted, "numpy", None)
    if callable(numpy_method):
        try:
            converted = numpy_method()
        except (TypeError, RuntimeError, ValueError):
            return None

    try:
        array = np.asarray(converted)
    except (TypeError, ValueError):
        return None
    return array
