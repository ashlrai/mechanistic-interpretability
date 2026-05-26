"""Crosscoder experiment family: train a crosscoder on two (or more) models.

Pipeline:
  1. Load both models via the instrumented backend.
  2. Capture activations at ``hook_site`` for each model on the same prompts.
  3. Validate d_model match (ValueError on mismatch).
  4. Train a Crosscoder (Top-K, shared encoder, per-model decoders).
  5. Analyse per-feature model scores (conserved ≈ 0, model-specific ±1).
  6. Persist artifacts: weights, config, feature_analysis, divergent_features.

Reference: Lindsey et al., "Sparse Crosscoders for Cross-Layer Features in
Superposition" (Anthropic, 2024).
"""

from __future__ import annotations

import json
import logging
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from mech_interp.backends import create_instrumented_backend
from mech_interp.experiments.base import Experiment
from mech_interp.sae.crosscoder_analysis import compute_crosscoder_analysis
from mech_interp.sae.crosscoder_trainer import save_crosscoder_weights, train_crosscoder
from mech_interp.storage.artifacts import resolve_run_artifact_dir
from mech_interp.types import (
    ExperimentResult,
    ExperimentRun,
    ExperimentSpec,
    InstrumentedModelBackend,
    RunStatus,
)

logger = logging.getLogger(__name__)


class CrosscoderSpec(BaseModel):
    model_config = ConfigDict(extra="allow")

    source_model: str = "gpt2"
    target_model: str = "distilgpt2"
    hook_site: str
    n_features: int = Field(ge=2)
    k: int = Field(ge=1)
    epochs: int = Field(default=5, ge=1, le=10_000)
    batch_size: int = Field(default=512, ge=1)
    learning_rate: float = Field(default=1e-3, gt=0)
    seed: int = 42
    device: str = "cpu"
    prompts: list[str] | None = None
    corpus_path: str | None = None
    seq_len: int = Field(default=128, ge=1)
    max_tokens: int = Field(default=10_000, ge=1)
    # |model_score| > this → model-specific
    model_specific_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    top_divergent_features: int = Field(default=20, ge=1)

    @field_validator("hook_site")
    @classmethod
    def strip_hook_site(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("hook_site must not be empty")
        return value


class CrosscoderExperiment(Experiment):
    family = "crosscoder"

    def __init__(
        self,
        source_backend: InstrumentedModelBackend | None = None,
        target_backend: InstrumentedModelBackend | None = None,
    ) -> None:
        self.source_backend = source_backend
        self.target_backend = target_backend

    def run(self, spec: ExperimentSpec, run: ExperimentRun) -> ExperimentResult:
        import torch

        config = CrosscoderSpec.model_validate(spec.parameters)
        if config.k > config.n_features:
            raise ValueError(
                f"k={config.k} must be <= n_features={config.n_features}"
            )

        # Build or reuse backends — each model needs its own backend instance
        source_backend = self.source_backend or create_instrumented_backend(
            spec.backend,
            {"model_name": config.source_model, "device": config.device},
        )
        target_backend = self.target_backend or create_instrumented_backend(
            spec.backend,
            {"model_name": config.target_model, "device": config.device},
        )

        prompts = _resolve_prompts(config)

        # Capture activations from both models
        source_captured = source_backend.capture_activations(prompts, [config.hook_site])
        target_captured = target_backend.capture_activations(prompts, [config.hook_site])

        if config.hook_site not in source_captured:
            raise ValueError(
                f"Source backend did not return activations for '{config.hook_site}'."
            )
        if config.hook_site not in target_captured:
            raise ValueError(
                f"Target backend did not return activations for '{config.hook_site}'."
            )

        source_tensor: Any = source_captured[config.hook_site]
        target_tensor: Any = target_captured[config.hook_site]

        # Flatten (batch, seq, d_model) → (n_tokens, d_model)
        source_flat: Any
        target_flat: Any
        source_flat, prompt_for_token = _flatten_with_prompt_map(source_tensor, prompts)
        target_flat, _ = _flatten_with_prompt_map(target_tensor, prompts)

        # Validate d_model match
        if source_flat.shape[1] != target_flat.shape[1]:
            raise ValueError(
                f"source_model '{config.source_model}' has d_model={source_flat.shape[1]} "
                f"but target_model '{config.target_model}' "
                f"has d_model={target_flat.shape[1]}. "
                "Crosscoder requires both models to have the same d_model."
            )

        # Validate token count match
        if source_flat.shape[0] != target_flat.shape[0]:
            raise ValueError(
                f"Token count mismatch: source produced {source_flat.shape[0]} tokens "
                f"but target produced {target_flat.shape[0]} tokens."
            )

        # Pin to float32 for numerical stability
        source_flat = source_flat.detach().to(dtype=torch.float32)
        target_flat = target_flat.detach().to(dtype=torch.float32)

        torch.manual_seed(config.seed)
        activations_tuple = (source_flat, target_flat)

        crosscoder, history = train_crosscoder(
            activations_tuple,
            n_features=config.n_features,
            k=config.k,
            learning_rate=config.learning_rate,
            epochs=config.epochs,
            batch_size=config.batch_size,
            device=config.device,
            seed=config.seed,
        )

        # Final MSE metrics per model
        crosscoder.eval()
        with torch.no_grad():
            acts_device = tuple(a.to(config.device) for a in activations_tuple)
            recons, codes = crosscoder(acts_device)
            mse_source = float(
                torch.mean((acts_device[0] - recons[0]) ** 2).item()
            )
            mse_target = float(
                torch.mean((acts_device[1] - recons[1]) ** 2).item()
            )

        analysis = compute_crosscoder_analysis(
            crosscoder,
            acts_device,
            prompt_for_token,
            model_specific_threshold=config.model_specific_threshold,
        )

        # Build divergent features: model-specific, ranked by abs(model_score) × max_activation
        divergent = [
            r for r in analysis.features
            if not r.dead and abs(r.model_score) > config.model_specific_threshold
        ]
        divergent.sort(
            key=lambda r: abs(r.model_score) * r.max_activation, reverse=True
        )
        divergent_records = divergent[: config.top_divergent_features]

        # Persist artifacts
        artifact_dir = resolve_run_artifact_dir(run)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        artifacts: dict[str, str] = {}

        weights_path = artifact_dir / "crosscoder_weights.safetensors"
        save_crosscoder_weights(crosscoder, weights_path, history=history)
        artifacts["crosscoder_weights"] = str(weights_path.resolve())
        artifacts["crosscoder_config"] = str(
            (weights_path.with_suffix(".safetensors.json")).resolve()
        )

        feature_analysis_path = artifact_dir / "feature_analysis.json"
        feature_analysis_path.write_text(
            json.dumps(analysis.as_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        artifacts["feature_analysis"] = str(feature_analysis_path.resolve())

        divergent_path = artifact_dir / "divergent_features.json"
        divergent_path.write_text(
            json.dumps(
                {
                    "model_specific_threshold": config.model_specific_threshold,
                    "source_model": config.source_model,
                    "target_model": config.target_model,
                    "divergent_features": [
                        {
                            "feature_index": r.feature_index,
                            "model_score": r.model_score,
                            "max_activation": r.max_activation,
                            "decoder_norm_per_model": r.decoder_norm_per_model,
                            "top_prompts": r.top_prompts,
                        }
                        for r in divergent_records
                    ],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        artifacts["divergent_features"] = str(divergent_path.resolve())

        metrics: dict[str, float] = {
            "n_tokens": float(source_flat.shape[0]),
            "n_features": float(crosscoder.n_features),
            "k": float(crosscoder.k),
            "mse_source": mse_source,
            "mse_target": mse_target,
            "initial_loss": history.initial_loss,
            "final_loss": history.final_loss,
            "dead_features": float(analysis.dead_count),
            "dead_feature_ratio": float(analysis.dead_count / crosscoder.n_features),
            "live_features": float(analysis.live_count),
            "conserved_features": float(analysis.conserved_count),
            "model_specific_features": float(analysis.model_specific_count),
            "mean_features_per_token": analysis.mean_features_per_token,
        }
        return ExperimentResult(
            run_id=run.id,
            status=RunStatus.SUCCEEDED,
            metrics=metrics,
            artifacts=artifacts,
            notes=(
                f"Crosscoder: {config.source_model} vs {config.target_model} at "
                f"'{config.hook_site}', {config.n_features} features, k={config.k}, "
                f"{source_flat.shape[0]} tokens. "
                f"Conserved: {analysis.conserved_count}, "
                f"model-specific: {analysis.model_specific_count}."
            ),
        )


def _resolve_prompts(config: CrosscoderSpec) -> list[str]:
    if config.prompts:
        cleaned = [p.strip() for p in config.prompts if p and p.strip()]
        if not cleaned:
            raise ValueError("crosscoder prompts must contain at least one non-empty entry")
        return cleaned
    raise ValueError("crosscoder requires 'prompts' (or 'corpus_path' in a future version).")


def _flatten_with_prompt_map(
    activation_tensor: Any,
    prompts: list[str],
) -> tuple[Any, list[str]]:
    """Flatten ``(batch, seq, d_model)`` → ``(batch*seq, d_model)``."""
    import torch

    if not hasattr(activation_tensor, "shape"):
        raise ValueError(
            "Backend returned a non-tensor activation; crosscoder needs a torch tensor."
        )
    shape = tuple(activation_tensor.shape)
    if len(shape) == 2:
        batch, _d_model = shape
        flat = activation_tensor
    elif len(shape) == 3:
        batch, seq, _d_model = shape
        flat = activation_tensor.reshape(batch * seq, _d_model)
    else:
        raise ValueError(
            f"Unexpected activation shape {shape}; expected 2D or 3D tensor"
        )
    if batch != len(prompts):
        raise ValueError(
            f"Activation batch dim {batch} did not match prompt count {len(prompts)}"
        )
    seq_len = shape[1] if len(shape) == 3 else 1
    if not isinstance(flat, torch.Tensor):
        flat = torch.as_tensor(flat)
    prompt_for_token = [prompts[i] for i in range(batch) for _ in range(seq_len)]
    return flat, prompt_for_token
