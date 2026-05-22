from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any

from mech_interp.types import InstrumentedModelBackend


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
        return {site: cache[site] for site in sites if site in cache}

    def run_intervention(self, prompt: str, interventions: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError(
            "TransformerLens interventions will be implemented in the circuit module."
        )


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
