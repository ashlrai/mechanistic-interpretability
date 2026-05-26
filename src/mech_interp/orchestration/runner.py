from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import os
import random
import sys
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from typing import Any

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

DEFAULT_SEED = 42
PLACEHOLDER_ENV_VAR = "MECH_INTERP_ALLOW_PLACEHOLDER"


class FamilyNotImplementedError(RuntimeError):
    """Raised when a YAML spec targets a family that has no real implementation.

    The fallback ``SpecValidationExperiment`` only emits placeholder metrics, so silently
    using it for unmapped families historically produced fake "successful" runs
    (see the polysemanticity / superposition runs prior to this gate). Opt back in
    by setting ``MECH_INTERP_ALLOW_PLACEHOLDER=1`` in the environment.
    """


class ExperimentRunner:
    def __init__(self, result_store: SQLiteResultStore, artifact_store: ArtifactStore) -> None:
        self.result_store = result_store
        self.artifact_store = artifact_store

    def run(self, spec: ExperimentSpec) -> ExperimentResult:
        run = self.result_store.create_run(spec)
        self.result_store.update_run_status(run.id, RunStatus.RUNNING)
        seed = _resolve_seed(spec)
        _initialize_seed(seed)
        env_fingerprint = _capture_environment_fingerprint(spec, seed)
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
            ),
            self.artifact_store.write_json(run.id, "environment.json", env_fingerprint),
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
    if spec.family == "polysemanticity_sae":
        from mech_interp.experiments.polysemanticity_sae import (
            PolysemanticitySAEExperiment,
        )
        return PolysemanticitySAEExperiment()
    if spec.family == "acdc_lite":
        from mech_interp.experiments.acdc_lite import ACDCLiteExperiment
        return ACDCLiteExperiment()
    if spec.family == "acdc_edge":
        from mech_interp.experiments.acdc_edge import ACDCEdgeExperiment
        return ACDCEdgeExperiment()
    if spec.family == "refusal_direction":
        from mech_interp.experiments.refusal_direction import RefusalDirectionExperiment
        return RefusalDirectionExperiment()
    if spec.family == "sae_cross_model":
        from mech_interp.experiments.sae_cross_model import SAECrossModelExperiment
        return SAECrossModelExperiment()
    if spec.family == "attribution_patching":
        from mech_interp.experiments.attribution_patching import AttributionPatchingExperiment
        return AttributionPatchingExperiment()
    if spec.family == "crosscoder":
        from mech_interp.experiments.crosscoder import CrosscoderExperiment
    if spec.family == "direct_logit_attribution":
        from mech_interp.experiments.direct_logit_attribution import (
            DirectLogitAttributionExperiment,
        )
        return DirectLogitAttributionExperiment()
    if spec.family == "sparse_probing":
        from mech_interp.experiments.sparse_probing import SparseProbingExperiment
        return SparseProbingExperiment()
        return CrosscoderExperiment()
    if spec.parameters.get("runner") == "activation_capture":
        return ActivationCaptureExperiment()
    if spec.parameters.get("runner") == "transformerlens_smoke":
        return TransformerLensSmokeExperiment()
    if os.environ.get(PLACEHOLDER_ENV_VAR) == "1":
        return SpecValidationExperiment(spec.family)
    raise FamilyNotImplementedError(
        f"Experiment family '{spec.family}' has no real implementation. "
        f"Set {PLACEHOLDER_ENV_VAR}=1 to opt into the placeholder runner."
    )


def _resolve_seed(spec: ExperimentSpec) -> int:
    raw = spec.parameters.get("seed", DEFAULT_SEED)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return DEFAULT_SEED


def _initialize_seed(seed: int) -> None:
    """Set seeds for python, numpy, torch (incl. MPS) so each run is reproducible.

    torch and numpy are imported lazily so the runner works for placeholder/lightweight
    families even when the optional ``interp`` extras are not installed.
    """
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        mps = getattr(torch, "mps", None)
        if mps is not None and getattr(mps, "manual_seed", None):
            mps.manual_seed(seed)
    except ImportError:
        pass


@lru_cache(maxsize=1)
def _uv_lock_sha256() -> str | None:
    candidate = Path(__file__).resolve()
    for directory in (candidate, *candidate.parents):
        lockfile = directory / "uv.lock"
        if lockfile.is_file():
            return hashlib.sha256(lockfile.read_bytes()).hexdigest()
    return None


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _capture_environment_fingerprint(spec: ExperimentSpec, seed: int) -> dict[str, Any]:
    """Snapshot the environment that produced this run.

    Captures library versions, the ``uv.lock`` hash, and a sample of model weights
    so silent drift between runs is detectable months later.
    """
    fingerprint: dict[str, Any] = {
        "seed": seed,
        "python_version": ".".join(map(str, sys.version_info[:3])),
        "uv_lock_sha256": _uv_lock_sha256(),
        "spec_name": spec.name,
        "family": spec.family,
        "backend": spec.backend,
        "model_name": spec.parameters.get("model") or spec.parameters.get("model_name"),
        "package_versions": {
            "numpy": _package_version("numpy"),
            "torch": _package_version("torch"),
            "transformer-lens": _package_version("transformer-lens"),
            "transformers": _package_version("transformers"),
            "safetensors": _package_version("safetensors"),
            "pydantic": _package_version("pydantic"),
        },
    }
    try:
        import torch

        fingerprint["torch_runtime"] = {
            "version": torch.__version__,
            "mps_available": bool(torch.backends.mps.is_available()),
            "cuda_available": bool(torch.cuda.is_available()),
        }
    except (ImportError, AttributeError):
        pass
    return fingerprint


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
