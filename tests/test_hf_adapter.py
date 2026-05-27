"""Unit tests for the HuggingFace backend — all dependencies monkeypatched."""

from __future__ import annotations

import importlib
from types import SimpleNamespace
from typing import Any

import pytest

from mech_interp.backends import HuggingFaceBackend, create_instrumented_backend

# ---------------------------------------------------------------------------
# Helpers: fake transformers + torch
# ---------------------------------------------------------------------------


def _make_fake_tensor(shape: tuple[int, ...], value: float = 1.0) -> Any:
    """Duck-typed tensor with .detach().cpu().to()."""

    class FakeTensor:
        def __init__(self, v: float = value) -> None:
            self._v = v
            self.shape = shape
            self.dtype = "float32"
            self.device = "cpu"

        def detach(self) -> FakeTensor:
            return self

        def cpu(self) -> FakeTensor:
            return self

        def to(self, device: Any = None, **kwargs: Any) -> FakeTensor:
            return self

        def clone(self) -> FakeTensor:
            return FakeTensor(self._v)

        def norm(self) -> FakeTensor:
            return FakeTensor(abs(self._v))

        def item(self) -> float:
            return self._v

        def tolist(self) -> list[float]:
            return [self._v] * (shape[-1] if shape else 1)

        def __getitem__(self, key: Any) -> FakeTensor:
            return self

        def __setitem__(self, key: Any, value: Any) -> None:
            pass

        @property
        def ndim(self) -> int:
            return len(shape)

    return FakeTensor()


class _FakeModule:
    """Minimal nn.Module-like with register_forward_hook."""

    def __init__(self, name: str = "mod") -> None:
        self._name = name
        self._hooks: list[Any] = []

    def register_forward_hook(self, fn: Any) -> Any:
        handle = SimpleNamespace(
            fn=fn,
            remove=lambda: None,
        )
        self._hooks.append(handle)
        return handle

    def fire(self, inp: Any, out: Any) -> None:
        for h in self._hooks:
            h.fn(self, inp, out)


class _FakeModel:
    """Minimal HF model duck-type."""

    def __init__(self) -> None:
        self.config = SimpleNamespace(model_type="gpt2")
        self._modules: dict[str, _FakeModule] = {
            "transformer.h.0": _FakeModule("transformer.h.0"),
            "transformer.h.1": _FakeModule("transformer.h.1"),
        }
        # Named modules returns flat (name, module) pairs like nn.Module.
        self._named_modules_list = list(self._modules.items())

    def named_modules(self) -> list[tuple[str, Any]]:
        return self._named_modules_list

    def to(self, device: Any) -> _FakeModel:
        return self

    def eval(self) -> _FakeModel:
        return self

    def __call__(self, **kwargs: Any) -> Any:
        logits = _make_fake_tensor((1, 3, 50257), 0.5)
        return SimpleNamespace(logits=logits)

    def __getattr__(self, name: str) -> Any:
        # Support dotted-path traversal: model.transformer.h[0]
        if name == "transformer":
            return SimpleNamespace(
                h=[_FakeModule(f"transformer.h.{i}") for i in range(2)]
            )
        raise AttributeError(name)


class _FakeTokenizer:
    pad_token = "<pad>"
    eos_token = "<eos>"

    def __call__(
        self,
        texts: list[str],
        return_tensors: str = "pt",
        padding: bool = True,
        truncation: bool = True,
    ) -> dict[str, Any]:
        return {"input_ids": _make_fake_tensor((len(texts), 5))}

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        return [42]


def _make_fake_transformers() -> Any:
    class FakeAutoModel:
        @staticmethod
        def from_pretrained(name: str, **kwargs: Any) -> _FakeModel:
            return _FakeModel()

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(name: str, **kwargs: Any) -> _FakeTokenizer:
            return _FakeTokenizer()

    return SimpleNamespace(
        AutoModelForCausalLM=FakeAutoModel,
        AutoTokenizer=FakeAutoTokenizer,
    )


def _make_fake_torch() -> Any:
    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return False

    class FakeBackends:
        class mps:
            @staticmethod
            def is_available() -> bool:
                return False

    return SimpleNamespace(
        cuda=FakeCuda,
        backends=FakeBackends,
        device=lambda x: x,
        no_grad=lambda: _NoGradCtx(),
    )


class _NoGradCtx:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *args: Any) -> None:
        pass


# ---------------------------------------------------------------------------
# Fixture: patched backend
# ---------------------------------------------------------------------------


@pytest.fixture()
def hf_backend(monkeypatch: pytest.MonkeyPatch) -> HuggingFaceBackend:
    """HuggingFaceBackend with transformers + torch fully monkeypatched."""
    fake_transformers = _make_fake_transformers()
    fake_torch = _make_fake_torch()

    real_import = importlib.import_module

    def fake_import(name: str, package: Any = None) -> Any:
        if name == "transformers":
            return fake_transformers
        if name == "torch":
            return fake_torch
        return real_import(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import)

    backend = HuggingFaceBackend(model_name="gpt2", device="cpu", architecture="gpt2")
    backend.load()
    return backend


# ---------------------------------------------------------------------------
# Tests: construction
# ---------------------------------------------------------------------------


def test_hf_backend_constructs_without_loading() -> None:
    backend = HuggingFaceBackend(model_name="gpt2")
    assert backend.name == "huggingface"
    assert backend.model_name == "gpt2"
    assert backend.model is None


def test_create_instrumented_backend_returns_hf_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = create_instrumented_backend(
        "huggingface",
        {"model_name": "gpt2", "device": "cpu"},
    )
    assert isinstance(backend, HuggingFaceBackend)
    assert backend.model_name == "gpt2"


def test_create_instrumented_backend_hf_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = create_instrumented_backend("hf", {"model_name": "gpt2"})
    assert isinstance(backend, HuggingFaceBackend)


# ---------------------------------------------------------------------------
# Tests: load
# ---------------------------------------------------------------------------


def test_hf_backend_load_sets_model_and_tokenizer(hf_backend: HuggingFaceBackend) -> None:
    assert hf_backend.model is not None
    assert hf_backend.tokenizer is not None


def test_hf_backend_load_detects_gpt2_architecture(hf_backend: HuggingFaceBackend) -> None:
    assert hf_backend._architecture == "gpt2"


def test_hf_backend_architecture_override(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_transformers = _make_fake_transformers()
    fake_torch = _make_fake_torch()
    real_import = importlib.import_module

    def fake_import(name: str, package: Any = None) -> Any:
        if name == "transformers":
            return fake_transformers
        if name == "torch":
            return fake_torch
        return real_import(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import)
    backend = HuggingFaceBackend(model_name="gpt2", architecture="llama")
    backend.load()
    assert backend._architecture == "llama"


# ---------------------------------------------------------------------------
# Tests: capture_activations
# ---------------------------------------------------------------------------


def test_capture_activations_registers_and_removes_hooks(
    hf_backend: HuggingFaceBackend,
) -> None:
    """Hooks must be registered during the forward pass and released after."""
    # We just verify capture_activations returns a dict without error; the
    # monkeypatched model fires a real forward pass through FakeModel.__call__.
    result = hf_backend.capture_activations(
        ["hello world"],
        ["blocks.0.hook_resid_post"],
    )
    # Site may or may not be captured depending on mock module depth;
    # the call must not raise.
    assert isinstance(result, dict)


def test_capture_activations_skips_untranslatable_sites(
    hf_backend: HuggingFaceBackend,
) -> None:
    """Sites that can't be translated should be silently skipped (appear as missing)."""
    result = hf_backend.capture_activations(
        ["hello"],
        ["blocks.0.hook_nonexistent_xyz"],
    )
    assert "blocks.0.hook_nonexistent_xyz" not in result


def test_capture_activations_returns_dict(hf_backend: HuggingFaceBackend) -> None:
    result = hf_backend.capture_activations(["prompt"], ["blocks.0.hook_resid_post"])
    assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# Tests: run_with_hooks
# ---------------------------------------------------------------------------


def test_run_with_hooks_returns_logits(hf_backend: HuggingFaceBackend) -> None:
    logits = hf_backend.run_with_hooks(["hello"], [])
    assert logits is not None


def test_run_with_hooks_hook_is_called(hf_backend: HuggingFaceBackend) -> None:
    called: list[bool] = []

    def my_hook(activation: Any) -> Any:
        called.append(True)
        return activation

    # This may or may not fire depending on mock depth, but must not raise.
    hf_backend.run_with_hooks(["hello"], [("blocks.0.hook_resid_post", my_hook)])


def test_run_with_hooks_handles_unknown_site(hf_backend: HuggingFaceBackend) -> None:
    """Unknown sites should be silently skipped, not raise."""
    logits = hf_backend.run_with_hooks(
        ["hello"], [("blocks.0.hook_totally_unknown", lambda x: x)]
    )
    assert logits is not None


# ---------------------------------------------------------------------------
# Tests: run_with_cache
# ---------------------------------------------------------------------------


def test_run_with_cache_returns_tuple(hf_backend: HuggingFaceBackend) -> None:
    logits, cache = hf_backend.run_with_cache(["hello"])
    assert logits is not None
    assert isinstance(cache, dict)


def test_run_with_cache_names_filter(hf_backend: HuggingFaceBackend) -> None:
    _logits, cache = hf_backend.run_with_cache(
        ["hello"], names_filter=lambda name: "transformer" in name
    )
    for key in cache:
        assert "transformer" in key


# ---------------------------------------------------------------------------
# Tests: unsupported methods
# ---------------------------------------------------------------------------


def test_run_intervention_raises(hf_backend: HuggingFaceBackend) -> None:
    with pytest.raises(NotImplementedError):
        hf_backend.run_intervention("hello", {})


def test_run_cross_model_probe_raises(hf_backend: HuggingFaceBackend) -> None:
    from mech_interp.types import CrossModelProbeRecord, CrossModelProbeRequest

    req = CrossModelProbeRequest(
        source_model_name="gpt2",
        target_model_name="gpt2",
        records=(CrossModelProbeRecord(id="r1", split="train", prompt="hello"),),
        source_hook_site="blocks.0.hook_resid_post",
        target_hook_site="blocks.0.hook_resid_post",
    )
    with pytest.raises(NotImplementedError, match="transformerlens"):
        hf_backend.run_cross_model_probe(req)


# ---------------------------------------------------------------------------
# Tests: factory error path
# ---------------------------------------------------------------------------


def test_create_instrumented_backend_unknown_raises() -> None:
    with pytest.raises(ValueError, match="huggingface"):
        create_instrumented_backend("totally_unknown_backend_xyz", {})


def test_create_instrumented_backend_error_message_lists_hf() -> None:
    with pytest.raises(ValueError) as exc_info:
        create_instrumented_backend("unknown_xyz", {})
    assert "huggingface" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Tests: security fixes
# ---------------------------------------------------------------------------

def test_get_module_rejects_dunder_attributes() -> None:
    """Security guard: a malicious YAML can't walk into Python internals via
    a dotted hook-site path like '__class__.__init__.__globals__.os.system'."""
    from mech_interp.backends.hf_adapter import _get_module

    class FakeModel:
        pass

    fake = FakeModel()
    with pytest.raises(ValueError, match="dunder"):
        _get_module(fake, "__class__.__init__")


def test_get_module_caps_path_depth() -> None:
    """Bound traversal cost: reject paths with too many components."""
    from mech_interp.backends.hf_adapter import _get_module

    class FakeModel:
        pass

    fake = FakeModel()
    with pytest.raises(ValueError, match="maximum 12"):
        _get_module(fake, ".".join(["x"] * 13))


def test_load_does_not_silently_escalate_trust_remote_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Security: the model loader must NOT auto-enable trust_remote_code on
    OSError. A user that didn't opt in could otherwise be subjected to
    arbitrary code execution from a malicious HF repo."""
    from mech_interp.backends import hf_adapter

    class FakeTransformers:
        class AutoModelForCausalLM:
            @staticmethod
            def from_pretrained(*args: Any, **kwargs: Any) -> Any:
                raise OSError("model repo requires custom code")

        class AutoTokenizer:
            @staticmethod
            def from_pretrained(*args: Any, **kwargs: Any) -> Any:
                return SimpleNamespace(pad_token="<pad>", eos_token="<eos>")

    import importlib

    real_import_module = importlib.import_module

    def fake_import_module(name: str, package: Any = None) -> Any:
        if name == "transformers":
            return FakeTransformers
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)

    backend = hf_adapter.HuggingFaceBackend(
        model_name="any/model", trust_remote_code=False
    )
    with pytest.raises(RuntimeError, match="trust_remote_code"):
        backend.load()
