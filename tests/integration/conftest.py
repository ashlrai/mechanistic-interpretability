"""Session-scoped fixtures for end-to-end tests against a real small model.

These tests load ``gpt2-small`` via TransformerLens once per pytest session. They
catch the class of regressions that mock-only unit tests miss (e.g., Run 15's
``patch_hook`` kwarg failure was invisible to mocks because TransformerLens's
real hook calling convention was never exercised).

Skip when TransformerLens / torch aren't installed so the suite stays runnable
in minimal environments.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from mech_interp.backends.instrumented import TransformerLensBackend


def _have_optional_interp_deps() -> bool:
    try:
        import torch  # noqa: F401
        import transformer_lens  # noqa: F401
    except ImportError:
        return False
    return True


@pytest.fixture(scope="session")
def gpt2_backend() -> TransformerLensBackend:
    """Load gpt2-small on CPU once per session.

    CPU avoids MPS-specific dtype flakiness in tests and keeps the fixture
    deterministic. The model is ~500 MB so the first load takes a few seconds;
    subsequent tests reuse the loaded weights.
    """
    if not _have_optional_interp_deps():
        pytest.skip("transformer-lens / torch not installed; run `uv sync --extra interp`")

    from mech_interp.backends.instrumented import TransformerLensBackend

    backend = TransformerLensBackend(model_name="gpt2", device="cpu")
    backend.load()
    return backend
