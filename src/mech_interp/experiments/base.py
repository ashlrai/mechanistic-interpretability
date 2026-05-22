from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic.types import StringConstraints

from mech_interp.experiments.families import (
    SUPPORTED_EXPERIMENT_BACKENDS,
    SUPPORTED_EXPERIMENT_FAMILIES,
)
from mech_interp.types import ExperimentResult, ExperimentRun, ExperimentSpec

NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class Experiment(ABC):
    family: str

    @abstractmethod
    def run(self, spec: ExperimentSpec, run: ExperimentRun) -> ExperimentResult:
        """Execute an experiment spec and return persisted metrics/artifact references."""


class ArtifactPolicySpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    retain_activation_tensors: bool = False
    write_manifest: bool = True
    max_artifact_bytes: int | None = Field(default=None, gt=0)


class ExperimentSpecSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: NonEmptyString
    family: NonEmptyString
    backend: NonEmptyString
    description: str = ""
    parameters: dict[str, Any] = Field(default_factory=dict)
    seed: int | None = Field(default=None, ge=0)
    model: NonEmptyString | None = None
    prompts: list[NonEmptyString] | None = Field(default=None, min_length=1)
    artifact_policy: ArtifactPolicySpec | None = None

    @field_validator("family")
    @classmethod
    def validate_family(cls, value: str) -> str:
        if value not in SUPPORTED_EXPERIMENT_FAMILIES:
            supported = ", ".join(SUPPORTED_EXPERIMENT_FAMILIES)
            raise ValueError(f"unsupported family '{value}'. Supported families: {supported}")
        return value

    @field_validator("backend")
    @classmethod
    def validate_backend(cls, value: str) -> str:
        if value not in SUPPORTED_EXPERIMENT_BACKENDS:
            supported = ", ".join(SUPPORTED_EXPERIMENT_BACKENDS)
            raise ValueError(f"unsupported backend '{value}'. Supported backends: {supported}")
        return value

    @model_validator(mode="after")
    def validate_parameter_collisions(self) -> ExperimentSpecSchema:
        top_level_parameter_keys = {
            key
            for key in ("seed", "model", "prompts", "artifact_policy")
            if getattr(self, key) is not None
        }
        collisions = sorted(top_level_parameter_keys.intersection(self.parameters))
        if collisions:
            names = ", ".join(collisions)
            raise ValueError(
                f"optional top-level field(s) also appear in parameters: {names}"
            )
        return self

    def to_experiment_spec(self) -> ExperimentSpec:
        parameters = dict(self.parameters)
        if self.seed is not None:
            parameters["seed"] = self.seed
        if self.model is not None:
            parameters["model"] = self.model
        if self.prompts is not None:
            parameters["prompts"] = list(self.prompts)
        if self.artifact_policy is not None:
            parameters["artifact_policy"] = self.artifact_policy.model_dump()

        return ExperimentSpec(
            name=self.name,
            family=self.family,
            backend=self.backend,
            description=self.description,
            parameters=parameters,
        )
