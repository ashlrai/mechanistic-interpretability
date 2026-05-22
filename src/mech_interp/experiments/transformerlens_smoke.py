from __future__ import annotations

from mech_interp.backends import create_instrumented_backend
from mech_interp.experiments.base import Experiment
from mech_interp.types import (
    ExperimentResult,
    ExperimentRun,
    ExperimentSpec,
    InstrumentedModelBackend,
    RunStatus,
)


class TransformerLensSmokeExperiment(Experiment):
    family = "transformerlens_smoke"

    def __init__(self, backend: InstrumentedModelBackend | None = None) -> None:
        self.backend = backend

    def run(self, spec: ExperimentSpec, run: ExperimentRun) -> ExperimentResult:
        prompts = _string_list_parameter(spec, "prompts", default=["The Eiffel Tower is in"])
        sites = _string_list_parameter(spec, "sites", default=["blocks.0.hook_resid_pre"])
        backend = self.backend or create_instrumented_backend(
            "transformerlens",
            {
                "model_name": spec.parameters.get("model_name", "gpt2-small"),
                "device": spec.parameters.get("device", "auto"),
            },
        )

        activations = backend.capture_activations(prompts, sites)
        captured_sites = [site for site in sites if site in activations]
        missing_sites = [site for site in sites if site not in activations]

        return ExperimentResult(
            run_id=run.id,
            status=RunStatus.SUCCEEDED,
            metrics={
                "prompt_count": float(len(prompts)),
                "requested_site_count": float(len(sites)),
                "captured_site_count": float(len(captured_sites)),
                "missing_site_count": float(len(missing_sites)),
            },
            notes=(
                f"TransformerLens smoke run captured selected activations: {captured_sites}."
                if not missing_sites
                else f"TransformerLens smoke run completed with missing sites: {missing_sites}."
            ),
        )


def _string_list_parameter(
    spec: ExperimentSpec,
    name: str,
    default: list[str],
) -> list[str]:
    value = spec.parameters.get(name, default)
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return value
    raise ValueError(f"TransformerLens smoke parameter '{name}' must be a string or list[str].")
