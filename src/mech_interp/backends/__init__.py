from mech_interp.backends.instrumented import (
    MLXInstrumentedBackend,
    NNsightBackend,
    OptionalDependencyError,
    TransformerLensBackend,
    create_instrumented_backend,
)

__all__ = [
    "HuggingFaceBackend",
    "MLXInstrumentedBackend",
    "NNsightBackend",
    "OptionalDependencyError",
    "TransformerLensBackend",
    "create_instrumented_backend",
]


def __getattr__(name: str) -> object:
    if name == "HuggingFaceBackend":
        from mech_interp.backends.hf_adapter import HuggingFaceBackend  # noqa: PLC0415

        return HuggingFaceBackend
    raise AttributeError(f"module 'mech_interp.backends' has no attribute {name!r}")
