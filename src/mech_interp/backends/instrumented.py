from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any

import numpy as np

from mech_interp.analysis import logit_diff_recovery
from mech_interp.types import (
    ActivationPatchRequest,
    ActivationPatchSiteResult,
    CrossModelProbeRequest,
    CrossModelProbeResult,
    InstrumentedModelBackend,
)


class OptionalDependencyError(RuntimeError):
    def __init__(self, package: str, extra: str) -> None:
        super().__init__(
            f"Install optional dependency '{package}' with `uv sync --extra {extra}` "
            "before running this instrumented backend."
        )


class TransformerLensBackend:
    name = "transformerlens"

    def __init__(self, model_name: str = "gpt2-small", device: str = "auto") -> None:
        self.model_name = model_name
        self.device = device
        self.model: Any | None = None
        self.last_probe_weights: np.ndarray | None = None

    def load(self) -> None:
        try:
            transformer_lens = importlib.import_module("transformer_lens")
        except ImportError as exc:
            raise OptionalDependencyError("transformer-lens", "interp") from exc

        kwargs: dict[str, Any] = {}
        if self.device != "auto":
            kwargs["device"] = self.device
        self.model = transformer_lens.HookedTransformer.from_pretrained(self.model_name, **kwargs)

    def capture_activations(self, prompts: list[str], sites: list[str]) -> dict[str, Any]:
        if self.model is None:
            self.load()
        assert self.model is not None
        _, cache = self.model.run_with_cache(prompts, names_filter=lambda name: name in sites)
        result: dict[str, Any] = {}
        for site in sites:
            if site not in cache:
                continue
            tensor = cache[site]
            # MPS can silently produce float16/bfloat16 activations even when the model
            # is nominally float32. Downstream SAE training assumes float32, so we cast
            # here unconditionally when device is MPS rather than letting dtype
            # mismatches surface as cryptic GEMM errors later.
            if self.device == "mps" and hasattr(tensor, "to") and hasattr(tensor, "dtype"):
                import torch

                if tensor.dtype != torch.float32:
                    tensor = tensor.to(dtype=torch.float32)
            result[site] = tensor
        return result

    def run_intervention(self, prompt: str, interventions: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError(
            "TransformerLens interventions will be implemented in the circuit module."
        )

    def run_activation_patching(
        self,
        request: ActivationPatchRequest,
    ) -> list[ActivationPatchSiteResult]:
        if self.model is None:
            self.load()
        assert self.model is not None

        results: list[ActivationPatchSiteResult] = []
        for pair in request.prompt_pairs:
            correct_token_id = int(self.model.to_single_token(pair.correct_token))
            incorrect_token_id = int(self.model.to_single_token(pair.incorrect_token))
            clean_logits, clean_cache = self.model.run_with_cache(
                pair.clean_prompt,
                names_filter=lambda name: name in request.hook_sites,
            )
            corrupted_logits = self.model(pair.corrupted_prompt)
            for hook_site in request.hook_sites:
                if hook_site not in clean_cache:
                    continue
                clean_activation = clean_cache[hook_site]
                patch_position = pair.patch_position

                def patch_hook(
                    activation: Any,
                    _hook: Any = None,
                    clean_value: Any = clean_activation,
                    position: int = patch_position,
                    **_kwargs: Any,
                ) -> Any:
                    patched_activation = activation.clone()
                    patched_activation[:, position, ...] = clean_value[:, position, ...]
                    return patched_activation

                patched_logits = self.model.run_with_hooks(
                    pair.corrupted_prompt,
                    fwd_hooks=[(hook_site, patch_hook)],
                )
                recovery = logit_diff_recovery(
                    clean_logits=_logits_at_position(clean_logits, pair.target_position),
                    corrupted_logits=_logits_at_position(
                        corrupted_logits,
                        pair.target_position,
                    ),
                    patched_logits=_logits_at_position(patched_logits, pair.target_position),
                    correct_token_index=correct_token_id,
                    incorrect_token_index=incorrect_token_id,
                )
                results.append(
                    ActivationPatchSiteResult(
                        pair_id=pair.id,
                        hook_site=hook_site,
                        clean_logit_diff=recovery.clean_logit_diff,
                        corrupted_logit_diff=recovery.corrupted_logit_diff,
                        patched_logit_diff=recovery.patched_logit_diff,
                        recovery_fraction=recovery.recovery_fraction,
                        activation_norm=_tensor_norm(clean_activation, pair.patch_position),
                    )
                )
        return results

    def run_cross_model_probe(
        self,
        request: CrossModelProbeRequest,
    ) -> list[CrossModelProbeResult]:
        try:
            transformer_lens = importlib.import_module("transformer_lens")
        except ImportError as exc:
            raise OptionalDependencyError("transformer-lens", "interp") from exc

        kwargs: dict[str, Any] = {}
        if self.device != "auto":
            kwargs["device"] = self.device
        source_model = self.model
        if source_model is None or self.model_name != request.source_model_name:
            source_model = transformer_lens.HookedTransformer.from_pretrained(
                request.source_model_name,
                **kwargs,
            )
        target_model = transformer_lens.HookedTransformer.from_pretrained(
            request.target_model_name,
            **kwargs,
        )

        prompts = [record.prompt for record in request.records]
        _, source_cache = source_model.run_with_cache(
            prompts,
            names_filter=lambda name: name == request.source_hook_site,
        )
        _, target_cache = target_model.run_with_cache(
            prompts,
            names_filter=lambda name: name == request.target_hook_site,
        )
        source_matrix = _activation_matrix(source_cache[request.source_hook_site], request.dtype)
        target_matrix = _activation_matrix(target_cache[request.target_hook_site], request.dtype)
        results, weights = _fit_and_score_probe_with_weights(request, source_matrix, target_matrix)
        self.last_probe_weights = weights if request.retain_probe_weights else None
        return results


class NNsightBackend:
    name = "nnsight"

    def __init__(self, model_name: str = "gpt2") -> None:
        self.model_name = model_name
        self.model: Any | None = None

    def load(self) -> None:
        try:
            from nnsight import LanguageModel
        except ImportError as exc:
            raise OptionalDependencyError("nnsight", "interp") from exc
        self.model = LanguageModel(self.model_name)

    def capture_activations(self, prompts: list[str], sites: list[str]) -> dict[str, Any]:
        raise NotImplementedError("nnsight activation capture is reserved for a later module.")

    def run_intervention(self, prompt: str, interventions: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError("nnsight interventions are reserved for a later module.")

    def run_activation_patching(
        self,
        request: ActivationPatchRequest,
    ) -> list[ActivationPatchSiteResult]:
        raise NotImplementedError("nnsight activation patching is reserved for a later module.")

    def run_cross_model_probe(
        self,
        request: CrossModelProbeRequest,
    ) -> list[CrossModelProbeResult]:
        raise NotImplementedError("nnsight cross-model probing is reserved for a later module.")


class MLXInstrumentedBackend:
    name = "mlx"

    def __init__(self, model_path: str | None = None) -> None:
        self.model_path = model_path
        self.model: Any | None = None

    def load(self) -> None:
        try:
            from mlx_lm import load
        except ImportError as exc:
            raise OptionalDependencyError("mlx-lm", "apple") from exc
        if self.model_path is None:
            raise ValueError("MLX backend requires a local model_path.")
        self.model, _tokenizer = load(self.model_path)

    def capture_activations(self, prompts: list[str], sites: list[str]) -> dict[str, Any]:
        raise NotImplementedError("MLX-native activation capture requires custom hooks.")

    def run_intervention(self, prompt: str, interventions: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError("MLX-native interventions require custom hooks.")

    def run_activation_patching(
        self,
        request: ActivationPatchRequest,
    ) -> list[ActivationPatchSiteResult]:
        raise NotImplementedError("MLX-native activation patching requires custom hooks.")

    def run_cross_model_probe(
        self,
        request: CrossModelProbeRequest,
    ) -> list[CrossModelProbeResult]:
        raise NotImplementedError("MLX-native cross-model probing requires custom hooks.")


def create_instrumented_backend(
    backend: str,
    config: Mapping[str, Any] | None = None,
) -> InstrumentedModelBackend:
    config = config or {}
    normalized = backend.replace("-", "").replace("_", "").lower()

    if normalized in {"transformerlens", "tl"}:
        return TransformerLensBackend(
            model_name=str(config.get("model_name", "gpt2-small")),
            device=str(config.get("device", "auto")),
        )
    if normalized == "nnsight":
        return NNsightBackend(model_name=str(config.get("model_name", "gpt2")))
    if normalized == "mlx":
        model_path = config.get("model_path")
        return MLXInstrumentedBackend(
            model_path=str(model_path) if model_path is not None else None,
        )

    supported = "transformerlens, nnsight, mlx"
    raise ValueError(f"Unknown instrumented backend '{backend}'. Supported backends: {supported}.")


def _logits_at_position(logits: Any, position: int) -> list[float]:
    selected = logits
    if getattr(selected, "ndim", None) == 3:
        selected = selected[0, position, :]
    elif getattr(selected, "ndim", None) == 2:
        selected = selected[position, :]
    detach = getattr(selected, "detach", None)
    if callable(detach):
        selected = detach()
    cpu = getattr(selected, "cpu", None)
    if callable(cpu):
        selected = cpu()
    tolist = getattr(selected, "tolist", None)
    if callable(tolist):
        selected = tolist()
    return [float(value) for value in selected]


def _tensor_norm(tensor: Any, position: int) -> float | None:
    try:
        selected = tensor[:, position, ...]
        norm = selected.norm()
        item = getattr(norm, "item", None)
        return float(item() if callable(item) else norm)
    except (AttributeError, IndexError, TypeError, ValueError, RuntimeError):
        return None


def _activation_matrix(tensor: Any, dtype: str) -> np.ndarray:
    detach = getattr(tensor, "detach", None)
    if callable(detach):
        tensor = detach()
    cpu = getattr(tensor, "cpu", None)
    if callable(cpu):
        tensor = cpu()
    array = np.asarray(tensor, dtype=dtype)
    if array.ndim < 2:
        raise ValueError("Activation tensor must include record and hidden dimensions.")
    if array.ndim > 2:
        array = array.reshape(array.shape[0], -1, array.shape[-1])[:, -1, :]
    return array


def _fit_and_score_probe(
    request: CrossModelProbeRequest,
    source_matrix: np.ndarray,
    target_matrix: np.ndarray,
) -> list[CrossModelProbeResult]:
    results, _weights = _fit_and_score_probe_with_weights(request, source_matrix, target_matrix)
    return results


def _fit_and_score_probe_with_weights(
    request: CrossModelProbeRequest,
    source_matrix: np.ndarray,
    target_matrix: np.ndarray,
) -> tuple[list[CrossModelProbeResult], np.ndarray]:
    if source_matrix.shape[0] != len(request.records) or target_matrix.shape[0] != len(
        request.records
    ):
        raise ValueError("Activation record count did not match cross-model probe records.")

    train_indices = [
        index for index, record in enumerate(request.records) if record.split == "train"
    ]
    eval_indices = [index for index, record in enumerate(request.records) if record.split == "eval"]
    if not train_indices or not eval_indices:
        raise ValueError("Cross-model probe requires at least one train and one eval record.")

    source_train = source_matrix[train_indices]
    target_train = target_matrix[train_indices]
    weights = _ridge_weights(source_train, target_train, request.ridge_alpha)

    results: list[CrossModelProbeResult] = []
    for split, indices in (("train", train_indices), ("eval", eval_indices)):
        predicted = source_matrix[indices] @ weights
        actual = target_matrix[indices]
        results.append(
            CrossModelProbeResult(
                source_hook_site=request.source_hook_site,
                target_hook_site=request.target_hook_site,
                split=split,
                record_count=len(indices),
                mean_cosine_similarity=_mean_cosine(predicted, actual),
                normalized_mse=_normalized_mse(predicted, actual),
                variance_explained=_variance_explained(predicted, actual),
            )
        )
    return results, weights


def _ridge_weights(source: np.ndarray, target: np.ndarray, alpha: float) -> np.ndarray:
    regularizer = float(alpha) * np.eye(source.shape[1], dtype=source.dtype)
    return np.linalg.solve(source.T @ source + regularizer, source.T @ target)


def _mean_cosine(predicted: np.ndarray, actual: np.ndarray) -> float:
    numerator = np.sum(predicted * actual, axis=1)
    denominator = np.linalg.norm(predicted, axis=1) * np.linalg.norm(actual, axis=1)
    values = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0)
    return float(np.mean(values)) if values.size else 0.0


def _normalized_mse(predicted: np.ndarray, actual: np.ndarray) -> float:
    mse = np.mean((predicted - actual) ** 2)
    baseline = np.mean(actual**2)
    return float(mse / baseline) if baseline else float(mse)


def _variance_explained(predicted: np.ndarray, actual: np.ndarray) -> float:
    residual = np.sum((actual - predicted) ** 2)
    centered = actual - np.mean(actual, axis=0, keepdims=True)
    total = np.sum(centered**2)
    return float(1.0 - residual / total) if total else 0.0
