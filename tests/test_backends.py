import importlib
from types import SimpleNamespace
from typing import Any

import pytest

from mech_interp.backends import (
    OptionalDependencyError,
    TransformerLensBackend,
    create_instrumented_backend,
)
from mech_interp.providers import LMStudioProvider, OllamaProvider


def test_generation_providers_construct() -> None:
    assert OllamaProvider().name == "ollama"
    assert LMStudioProvider().name == "lm_studio"


def test_transformerlens_backend_constructs_without_loading() -> None:
    backend = TransformerLensBackend(model_name="gpt2-small")

    assert backend.name == "transformerlens"
    assert backend.model_name == "gpt2-small"


def test_create_instrumented_backend_returns_transformerlens_backend() -> None:
    backend = create_instrumented_backend(
        "transformer-lens",
        {"model_name": "tiny-stories", "device": "cpu"},
    )

    assert isinstance(backend, TransformerLensBackend)
    assert backend.model_name == "tiny-stories"
    assert backend.device == "cpu"


def test_transformerlens_backend_missing_optional_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import_module = importlib.import_module

    def fake_import_module(name: str, package: str | None = None) -> Any:
        if name == "transformer_lens":
            raise ImportError("missing test dependency")
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)

    with pytest.raises(OptionalDependencyError, match="uv sync --extra interp"):
        TransformerLensBackend().load()


def test_transformerlens_backend_captures_selected_activations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeHookedTransformer:
        @classmethod
        def from_pretrained(cls, model_name: str, **kwargs: Any) -> "FakeHookedTransformer":
            assert model_name == "tiny"
            assert kwargs == {"device": "cpu"}
            return cls()

        def run_with_cache(
            self,
            prompts: list[str],
            names_filter: Any,
        ) -> tuple[None, dict[str, str]]:
            assert prompts == ["hello"]
            cache = {
                "blocks.0.hook_resid_pre": "captured-resid",
                "blocks.0.mlp.hook_post": "captured-mlp",
                "blocks.1.hook_resid_pre": "ignored",
            }
            return None, {name: value for name, value in cache.items() if names_filter(name)}

    def fake_import_module(name: str, package: str | None = None) -> Any:
        assert package is None
        if name == "transformer_lens":
            return SimpleNamespace(HookedTransformer=FakeHookedTransformer)
        raise AssertionError(f"Unexpected import: {name}")

    monkeypatch.setattr(importlib, "import_module", fake_import_module)
    backend = TransformerLensBackend(model_name="tiny", device="cpu")

    activations = backend.capture_activations(
        ["hello"],
        ["blocks.0.hook_resid_pre", "blocks.0.mlp.hook_post"],
    )

    assert activations == {
        "blocks.0.hook_resid_pre": "captured-resid",
        "blocks.0.mlp.hook_post": "captured-mlp",
    }
