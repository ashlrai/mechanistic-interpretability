"""Polysemanticity experiment family: train a Top-K SAE on residual-stream activations.

This is the real implementation that replaces the placeholder ``SpecValidationExperiment``
fallback for ``family: polysemanticity``. The pipeline is:

  1. Capture activations at a chosen hook site via the instrumented backend.
  2. Flatten ``(batch, seq, d_model)`` into ``(n_tokens, d_model)`` for SAE training.
  3. Train a Top-K SAE with deterministic seeding (Gao et al., 2024).
  4. Walk every learned feature; rank top-activating prompts; flag dead features.
  5. Persist SAE weights (safetensors), training history, and feature analysis JSON.
"""

from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from mech_interp.backends import create_instrumented_backend
from mech_interp.experiments.base import Experiment
from mech_interp.sae import (
    compute_feature_analysis,
    save_sae_weights,
    train_top_k_sae,
)
from mech_interp.storage.artifacts import resolve_run_artifact_dir
from mech_interp.types import (
    ExperimentResult,
    ExperimentRun,
    ExperimentSpec,
    InstrumentedModelBackend,
    RunStatus,
)

logger = logging.getLogger(__name__)


class _ArtifactPolicy(BaseModel):
    model_config = ConfigDict(extra="allow")

    retain_weights: bool = True
    write_feature_analysis: bool = True
    top_prompts_per_feature: int = Field(default=5, ge=1, le=50)


class PolysemanticitySAESpec(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str = "gpt2-small"
    model_name: str | None = None
    hook_site: str
    n_features: int = Field(ge=2)
    k: int = Field(ge=1)
    epochs: int = Field(default=5, ge=1, le=10_000)
    batch_size: int = Field(default=512, ge=1)
    learning_rate: float = Field(default=1e-3, gt=0)
    seed: int = 42
    device: str = "cpu"
    prompts: list[str] | None = None
    dataset_path: str | None = None
    dataset_sha256: str | None = None
    corpus_path: str | None = None
    seq_len: int = Field(default=128, ge=1)
    max_tokens: int = Field(default=10_000, ge=1)
    artifact_policy: _ArtifactPolicy = Field(default_factory=_ArtifactPolicy)

    @property
    def resolved_model_name(self) -> str:
        return self.model_name or self.model

    @field_validator("hook_site")
    @classmethod
    def strip_hook_site(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("hook_site must not be empty")
        return value


class PolysemanticitySAEExperiment(Experiment):
    family = "polysemanticity_sae"

    def __init__(self, backend: InstrumentedModelBackend | None = None) -> None:
        self.backend = backend

    def run(self, spec: ExperimentSpec, run: ExperimentRun) -> ExperimentResult:
        import torch

        config = PolysemanticitySAESpec.model_validate(spec.parameters)
        if config.k > config.n_features:
            raise ValueError(
                f"k={config.k} must be <= n_features={config.n_features} for a Top-K SAE."
            )

        backend = self.backend or create_instrumented_backend(
            spec.backend,
            {
                "model_name": config.resolved_model_name,
                "device": config.device,
            },
        )

        if config.corpus_path is not None:
            if config.prompts is not None:
                warnings.warn(
                    "Both 'prompts' and 'corpus_path' are set; ignoring 'prompts' and "
                    "using corpus_path instead.",
                    UserWarning,
                    stacklevel=2,
                )
            flat, prompt_for_token = _activations_from_corpus(config, backend)
            n_input_docs = len(set(prompt_for_token))
        else:
            prompts = _resolve_prompts(config)
            captured = backend.capture_activations(prompts, [config.hook_site])
            if config.hook_site not in captured:
                raise ValueError(
                    f"Backend did not return activations for hook site '{config.hook_site}'."
                )
            activation_tensor = captured[config.hook_site]
            flat, prompt_for_token = _flatten_with_prompt_map(activation_tensor, prompts)
            n_input_docs = len(prompts)

        # SAE training is pinned to fp32 — MPS sometimes degrades silently in fp16.
        flat = flat.detach().to(dtype=torch.float32)

        torch.manual_seed(config.seed)
        sae, history = train_top_k_sae(
            flat,
            n_features=config.n_features,
            k=config.k,
            learning_rate=config.learning_rate,
            epochs=config.epochs,
            batch_size=config.batch_size,
            device=config.device,
            seed=config.seed,
        )

        sae.eval()
        with torch.no_grad():
            recon, codes = sae(flat.to(config.device))
            mse = float(torch.mean((flat.to(config.device) - recon) ** 2).item())
            actual_variance = float(torch.var(flat.to(config.device), unbiased=False).item())
            explained_variance = (
                1.0 - mse / actual_variance if actual_variance > 0 else 0.0
            )

        analysis = compute_feature_analysis(
            sae,
            flat.to(config.device),
            prompt_for_token,
            top_prompts_per_feature=config.artifact_policy.top_prompts_per_feature,
        )

        artifact_dir = resolve_run_artifact_dir(run)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        artifacts: dict[str, str] = {}

        if config.artifact_policy.retain_weights:
            weights_path = artifact_dir / "sae_weights.safetensors"
            save_sae_weights(sae, weights_path, history=history)
            artifacts["sae_weights"] = str(weights_path.resolve())
            artifacts["sae_config"] = str((weights_path.with_suffix(".safetensors.json")).resolve())

        if config.artifact_policy.write_feature_analysis:
            analysis_path = artifact_dir / "feature_analysis.json"
            analysis_path.write_text(
                json.dumps(analysis.as_dict(), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            artifacts["feature_analysis"] = str(analysis_path.resolve())

        training_path = artifact_dir / "training_history.json"
        training_path.write_text(
            json.dumps(history.as_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        artifacts["training_history"] = str(training_path.resolve())

        metrics = {
            "n_tokens": float(flat.shape[0]),
            "n_features": float(sae.n_features),
            "k": float(sae.k),
            "reconstruction_mse": mse,
            "explained_variance": explained_variance,
            "initial_loss": history.initial_loss,
            "final_loss": history.final_loss,
            "dead_features": float(analysis.dead_count),
            "dead_feature_ratio": float(analysis.dead_count / sae.n_features),
            "live_features": float(analysis.live_count),
            "mean_features_per_token": analysis.mean_features_per_token,
        }
        return ExperimentResult(
            run_id=run.id,
            status=RunStatus.SUCCEEDED,
            metrics=metrics,
            artifacts=artifacts,
            notes=(
                f"Trained Top-{config.k} SAE with {config.n_features} features on "
                f"{flat.shape[0]} tokens from {n_input_docs} documents; "
                f"{analysis.live_count}/{sae.n_features} features active."
            ),
        )


def _resolve_prompts(config: PolysemanticitySAESpec) -> list[str]:
    if config.prompts:
        cleaned = [p.strip() for p in config.prompts if p and p.strip()]
        if not cleaned:
            raise ValueError(
                "polysemanticity_sae prompts must contain at least one non-empty entry"
            )
        return cleaned
    if config.dataset_path:
        from mech_interp.datasets import load_prompt_dataset

        dataset = load_prompt_dataset(config.dataset_path)
        if config.dataset_sha256 and dataset.sha256 != config.dataset_sha256:
            raise ValueError(
                "polysemanticity_sae dataset hash mismatch: "
                f"expected {config.dataset_sha256}, got {dataset.sha256}."
            )
        return [record.prompt for record in dataset.records]
    raise ValueError("polysemanticity_sae requires 'prompts' or 'dataset_path'.")


def _activations_from_corpus(
    config: PolysemanticitySAESpec,
    backend: InstrumentedModelBackend,
) -> tuple[Any, list[str]]:
    """Load corpus, tokenize it, run a forward pass, and return flat activations.

    The token→document map uses ``"doc_<n>"`` labels (with the first 60 chars of the
    document text appended) so the feature analysis JSON stays readable without
    storing full document text for every token position.
    """
    from mech_interp.datasets.corpus import load_text_corpus, tokenize_corpus

    corpus_path = Path(config.corpus_path)  # type: ignore[arg-type]
    documents = load_text_corpus(corpus_path, max_documents=None)
    if not documents:
        raise ValueError(f"corpus_path '{corpus_path}' produced no documents.")

    # Build short labels: "doc_0: The quick brown fox..." (≤60 chars of text)
    doc_labels = [
        f"doc_{i}: {doc[:60]}" for i, doc in enumerate(documents)
    ]

    # tokenize_corpus needs a HookedTransformer; access via backend
    model = getattr(backend, "model", None)
    if model is None:
        raise AttributeError(
            "Backend has no 'model' attribute; cannot tokenize corpus. "
            "Only the TransformerLens backend supports corpus_path."
        )

    token_tensor = tokenize_corpus(
        model,
        documents,
        seq_len=config.seq_len,
        max_tokens=config.max_tokens,
    )
    # token_tensor: (n_docs, seq_len)  int64
    n_docs_used = token_tensor.shape[0]
    labels_used = doc_labels[:n_docs_used]

    # Decode token ids back into text strings for capture_activations, which
    # expects a list[str]. Each row becomes a single string by decoding its tokens.
    tokenizer = model.tokenizer
    text_inputs = [
        tokenizer.decode(token_tensor[i].tolist(), skip_special_tokens=True)
        for i in range(n_docs_used)
    ]

    captured = backend.capture_activations(text_inputs, [config.hook_site])
    if config.hook_site not in captured:
        raise ValueError(
            f"Backend did not return activations for hook site '{config.hook_site}'."
        )
    activation_tensor = captured[config.hook_site]
    flat, _ = _flatten_with_prompt_map(activation_tensor, text_inputs)

    # Build token-level label list using doc labels (not raw text) for readability
    shape = tuple(activation_tensor.shape)
    seq = shape[1] if len(shape) == 3 else 1
    prompt_for_token = [labels_used[i] for i in range(n_docs_used) for _ in range(seq)]

    return flat, prompt_for_token


def _flatten_with_prompt_map(activation_tensor: Any, prompts: list[str]) -> tuple[Any, list[str]]:
    """Flatten ``(batch, seq, d_model)`` → ``(batch*seq, d_model)`` and build the
    token→prompt index so downstream analysis can attribute features back to prompts.
    """
    import torch

    if not hasattr(activation_tensor, "shape"):
        raise ValueError(
            "Backend returned a non-tensor activation; SAE training needs a torch tensor."
        )
    shape = tuple(activation_tensor.shape)
    if len(shape) == 2:
        batch, d_model = shape
        seq = 1
        flat = activation_tensor
    elif len(shape) == 3:
        batch, seq, d_model = shape
        flat = activation_tensor.reshape(batch * seq, d_model)
    else:
        raise ValueError(
            f"Unexpected activation shape {shape}; expected 2D or 3D tensor"
        )
    if batch != len(prompts):
        raise ValueError(
            f"Activation batch dim {batch} did not match prompt count {len(prompts)}"
        )
    if not isinstance(flat, torch.Tensor):
        flat = torch.as_tensor(flat)
    prompt_for_token = [prompts[i] for i in range(batch) for _ in range(seq)]
    return flat, prompt_for_token
