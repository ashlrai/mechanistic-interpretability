from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class ProjectConfig(BaseModel):
    name: str = "local-mech-interp"
    artifact_dir: Path = Path("artifacts")
    database_path: Path = Path("artifacts/runs.sqlite3")


class ProviderConfig(BaseModel):
    base_url: str
    default_model: str


class TransformerLensConfig(BaseModel):
    model_name: str = "gpt2-small"
    device: str = "auto"


class NNsightConfig(BaseModel):
    model_name: str = "gpt2"


class MLXConfig(BaseModel):
    model_path: str | None = None


class BackendConfig(BaseModel):
    default_instrumented: str = "transformerlens"
    transformerlens: TransformerLensConfig = Field(default_factory=TransformerLensConfig)
    nnsight: NNsightConfig = Field(default_factory=NNsightConfig)
    mlx: MLXConfig = Field(default_factory=MLXConfig)


class OrchestrationConfig(BaseModel):
    max_parallel_runs: int = 1
    max_prompts_per_batch: int = 32
    retain_activation_tensors: bool = False


class AppConfig(BaseModel):
    project: ProjectConfig = Field(default_factory=ProjectConfig)
    providers: dict[str, ProviderConfig] = Field(default_factory=dict)
    backends: BackendConfig = Field(default_factory=BackendConfig)
    orchestration: OrchestrationConfig = Field(default_factory=OrchestrationConfig)


def load_config(path: str | Path | None = None) -> AppConfig:
    path_value = (
        path if path is not None else os.getenv("MECH_INTERP_CONFIG", "configs/default.yaml")
    )
    config_path = Path(path_value)
    if not config_path.exists():
        return AppConfig()

    with config_path.open("r", encoding="utf-8") as config_file:
        raw: dict[str, Any] = yaml.safe_load(config_file) or {}
    return AppConfig.model_validate(raw)
