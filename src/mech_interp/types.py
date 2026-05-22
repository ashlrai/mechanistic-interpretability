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


def utc_now() -> datetime:
    return datetime.now(UTC)
