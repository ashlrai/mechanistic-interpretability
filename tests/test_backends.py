import importlib
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from mech_interp.backends import (
    OptionalDependencyError,
    TransformerLensBackend,
    create_instrumented_backend,
)
from mech_interp.providers import LMStudioProvider, OllamaProvider
from mech_interp.types import ActivationPatchPromptPair, ActivationPatchRequest


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


def test_transformerlens_backend_runs_activation_patching_with_fake_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeScalar:
        def __init__(self, value: float) -> None:
            self.value = value

        def item(self) -> float:
            return self.value

    class FakeActivation:
        def __init__(self, value: float) -> None:
            self.value = value

        def clone(self) -> "FakeActivation":
            return FakeActivation(self.value)

        def __getitem__(self, key: object) -> "FakeActivation":
            return self

        def __setitem__(self, key: object, value: object) -> None:
            if isinstance(value, FakeActivation):
                self.value = value.value
                return
            if not isinstance(value, int | float):
                raise TypeError("fake activation value must be numeric")
            self.value = float(value)

        def norm(self) -> FakeScalar:
            return FakeScalar(abs(self.value))

    class FakeHookedTransformer:
        def to_single_token(self, token: str) -> int:
            return {" yes": 1, " no": 2}[token]

        def run_with_cache(
            self,
            prompt: str,
            names_filter: Any,
        ) -> tuple[np.ndarray[Any, Any], dict[str, FakeActivation]]:
            assert prompt == "clean"
            cache = {
                "blocks.0.hook_resid_pre": FakeActivation(12.0),
                "blocks.0.mlp.hook_post": FakeActivation(99.0),
            }
            return np.array([[[0.0, 5.0, 1.0]]]), {
                name: value for name, value in cache.items() if names_filter(name)
            }

        def __call__(self, prompt: str) -> np.ndarray[Any, Any]:
            assert prompt == "corrupted"
            return np.array([[[0.0, 2.0, 3.0]]])

        def run_with_hooks(
            self,
            prompt: str,
            fwd_hooks: list[tuple[str, Any]],
        ) -> np.ndarray[Any, Any]:
            assert prompt == "corrupted"
            hook_site, hook_fn = fwd_hooks[0]
            assert hook_site == "blocks.0.hook_resid_pre"
            patched = hook_fn(FakeActivation(0.0), None)
            assert patched.value == 12.0
            return np.array([[[0.0, 4.0, 2.0]]])

    backend = TransformerLensBackend(model_name="tiny")
    backend.model = FakeHookedTransformer()

    results = backend.run_activation_patching(
        ActivationPatchRequest(
            model_name="tiny",
            prompt_pairs=(
                ActivationPatchPromptPair(
                    id="pair",
                    clean_prompt="clean",
                    corrupted_prompt="corrupted",
                    correct_token=" yes",
                    incorrect_token=" no",
                ),
            ),
            hook_sites=("blocks.0.hook_resid_pre",),
        )
    )

    assert len(results) == 1
    assert results[0].hook_site == "blocks.0.hook_resid_pre"
    assert results[0].recovery_fraction == pytest.approx(0.6)
    assert results[0].activation_norm == 12.0
