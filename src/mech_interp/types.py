from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol


class RunStatus(StrEnum):
    PLANNED = "planned"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass(frozen=True)
class GenerationRequest:
    model: str
    prompt: str
    temperature: float = 0.0
    max_tokens: int = 128


@dataclass(frozen=True)
class GenerationResponse:
    text: str
    provider: str
    model: str
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    family: str
    backend: str
    description: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExperimentRun:
    id: int
    spec_name: str
    family: str
    backend: str
    status: RunStatus
    artifact_dir: Path
    created_at: datetime


@dataclass(frozen=True)
class ExperimentResult:
    run_id: int
    status: RunStatus
    metrics: dict[str, float] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)
    notes: str = ""


@dataclass(frozen=True)
class ArtifactRecord:
    name: str
    path: Path
    media_type: str
    sha256: str
    size_bytes: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ActivationPatchPromptPair:
    id: str
    clean_prompt: str
    corrupted_prompt: str
    correct_token: str
    incorrect_token: str
    target_position: int = -1
    patch_position: int = -1
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ActivationPatchRequest:
    model_name: str
    prompt_pairs: tuple[ActivationPatchPromptPair, ...]
    hook_sites: tuple[str, ...]
    dtype: str = "float32"
    retain_activation_tensors: bool = False


@dataclass(frozen=True)
class ActivationPatchSiteResult:
    pair_id: str
    hook_site: str
    clean_logit_diff: float
    corrupted_logit_diff: float
    patched_logit_diff: float
    recovery_fraction: float
    activation_norm: float | None = None


@dataclass(frozen=True)
class CrossModelProbeRecord:
    id: str
    split: str
    prompt: str
    correct_token: str | None = None
    incorrect_token: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CrossModelProbeRequest:
    source_model_name: str
    target_model_name: str
    records: tuple[CrossModelProbeRecord, ...]
    source_hook_site: str
    target_hook_site: str
    ridge_alpha: float = 1.0
    dtype: str = "float32"
    retain_probe_weights: bool = False
    max_verbalized_records: int = 0


@dataclass(frozen=True)
class CrossModelProbeResult:
    source_hook_site: str
    target_hook_site: str
    split: str
    record_count: int
    mean_cosine_similarity: float
    normalized_mse: float
    variance_explained: float
    mean_logit_diff_error: float | None = None


class GenerationProvider(Protocol):
    name: str

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate text through a black-box local provider."""


class InstrumentedModelBackend(Protocol):
    name: str

    def load(self) -> None:
        """Load model weights and prepare activation instrumentation."""

    def capture_activations(
        self,
        prompts: list[str],
        sites: list[str],
    ) -> dict[str, Any]:
        """Capture activations for prompts at named hook sites."""

    def run_intervention(
        self,
        prompt: str,
        interventions: dict[str, Any],
    ) -> dict[str, Any]:
        """Run a prompt with activation-level interventions."""

    def run_activation_patching(
        self,
        request: ActivationPatchRequest,
    ) -> list[ActivationPatchSiteResult]:
        """Run clean-to-corrupted activation patching for requested hook sites."""

    def run_cross_model_probe(
        self,
        request: CrossModelProbeRequest,
    ) -> list[CrossModelProbeResult]:
        """Fit and evaluate a cross-model activation representation probe."""


def utc_now() -> datetime:
    return datetime.now(UTC)
