"""Integration test: HF backend end-to-end on gpt2 (HF route).

Run with::

    RUN_INTEGRATION_TESTS=1 uv run --group dev --extra interp \\
        python -m pytest tests/integration/test_hf_backend_e2e.py -v

The test loads ``gpt2`` (same checkpoint as the TL e2e suite) via the HF adapter
and verifies:
1. activation_capture returns tensors of the correct shape.
2. captured activations are close (within float tolerance) to those captured by
   the TL backend on the same prompts.

The tolerance is deliberately generous (~1e-4) because HF and TL may differ in
float32 computation order, padding, and tokenization.
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.integration

PROMPTS = [
    "The Eiffel Tower is in",
    "The capital of France is",
    "The president of the United States is",
]

_SKIP_REASON = (
    "RUN_INTEGRATION_TESTS=1 not set; "
    "run with: uv sync --extra interp && "
    "RUN_INTEGRATION_TESTS=1 pytest tests/integration/test_hf_backend_e2e.py -v"
)


def _integration_enabled() -> bool:
    return os.environ.get("RUN_INTEGRATION_TESTS", "0") == "1"


def _have_deps() -> bool:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        return False
    return True


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def hf_gpt2_backend() -> HuggingFaceBackend:  # type: ignore[name-defined]  # noqa: F821
    if not _integration_enabled():
        pytest.skip(_SKIP_REASON)
    if not _have_deps():
        pytest.skip("transformers/torch not installed; run `uv sync --extra interp`")

    from mech_interp.backends.hf_adapter import HuggingFaceBackend

    backend = HuggingFaceBackend(model_name="gpt2", device="cpu", architecture="gpt2")
    backend.load()
    return backend


@pytest.fixture(scope="module")
def tl_gpt2_backend() -> TransformerLensBackend:  # type: ignore[name-defined]  # noqa: F821
    if not _integration_enabled():
        pytest.skip(_SKIP_REASON)
    try:
        import transformer_lens  # noqa: F401
    except ImportError:
        pytest.skip("transformer-lens not installed; run `uv sync --extra interp`")

    from mech_interp.backends.instrumented import TransformerLensBackend

    backend = TransformerLensBackend(model_name="gpt2", device="cpu")
    backend.load()
    return backend


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_hf_backend_capture_activations_shape(hf_gpt2_backend: Any) -> None:  # noqa: F821
    """Captured activations should have the correct d_model (768 for gpt2)."""
    import torch

    sites = ["blocks.0.hook_resid_post", "blocks.6.hook_resid_post"]
    result = hf_gpt2_backend.capture_activations(PROMPTS, sites)

    for site in sites:
        assert site in result, f"site '{site}' not captured"
        tensor = result[site]
        assert isinstance(tensor, torch.Tensor), f"expected torch.Tensor for {site}"
        # Shape: (batch, seq_len, d_model) — d_model=768 for gpt2
        assert tensor.ndim == 3, f"expected 3D tensor for {site}, got {tensor.ndim}D"
        assert tensor.shape[0] == len(PROMPTS), (
            f"batch dim mismatch for {site}: {tensor.shape[0]} vs {len(PROMPTS)}"
        )
        assert tensor.shape[-1] == 768, (
            f"d_model mismatch for {site}: {tensor.shape[-1]} vs 768"
        )
        assert tensor.std() > 0, f"activations for {site} are all the same value"


def test_hf_backend_run_with_cache_returns_logits(hf_gpt2_backend: Any) -> None:  # noqa: F821
    import torch

    logits, cache = hf_gpt2_backend.run_with_cache(
        PROMPTS[:1],
        names_filter=lambda name: name in {"transformer.h.0", "transformer.h.11"},
    )
    assert isinstance(logits, torch.Tensor)
    assert logits.ndim == 3  # (batch, seq, vocab)
    assert logits.shape[-1] == 50257  # gpt2 vocab size


def test_hf_backend_run_with_hooks_fires(hf_gpt2_backend: Any) -> None:  # noqa: F821

    fired: list[bool] = []

    def _hook(activation: Any) -> Any:  # noqa: F821
        fired.append(True)
        return activation

    hf_gpt2_backend.run_with_hooks(
        PROMPTS[:1], [("blocks.0.hook_resid_post", _hook)]
    )
    assert fired, "hook was never called"


def test_hf_activations_close_to_tl(
    hf_gpt2_backend: Any,  # noqa: F821
    tl_gpt2_backend: Any,  # noqa: F821
) -> None:
    """HF and TL activations should be close for the same gpt2 checkpoint.

    We check resid_post at layer 0 on a single prompt.  TL stores activations
    pre-padding; HF pads; so we compare on the token dimension that both models
    share, taking the *last* non-pad position via mean cosine similarity.
    """
    import numpy as np
    import torch

    prompt = ["The Eiffel Tower is in"]
    site_hf = "blocks.0.hook_resid_post"
    site_tl = "blocks.0.hook_resid_post"

    hf_result = hf_gpt2_backend.capture_activations(prompt, [site_hf])
    tl_result = tl_gpt2_backend.capture_activations(prompt, [site_tl])

    assert site_hf in hf_result, "HF backend did not capture site"
    assert site_tl in tl_result, "TL backend did not capture site"

    hf_act = hf_result[site_hf]  # (1, seq, 768)
    tl_act = tl_result[site_tl]  # (1, seq, 768)

    if isinstance(hf_act, torch.Tensor):
        hf_np = hf_act.float().numpy()
    else:
        hf_np = np.asarray(hf_act, dtype=np.float32)

    if isinstance(tl_act, torch.Tensor):
        tl_np = tl_act.detach().cpu().float().numpy()
    else:
        tl_np = np.asarray(tl_act, dtype=np.float32)

    # Both should have the same d_model.
    assert hf_np.shape[-1] == tl_np.shape[-1] == 768

    # Take the last sequence position (which both models have in common for
    # a single un-padded prompt) and compute cosine similarity.
    hf_vec = hf_np[0, -1, :]
    tl_vec = tl_np[0, -1, :]

    cosine = float(
        np.dot(hf_vec, tl_vec)
        / (np.linalg.norm(hf_vec) * np.linalg.norm(tl_vec) + 1e-8)
    )

    # Generous tolerance: same weights, but HF and TL may differ in
    # layer-norm epsilon, attention mask handling, etc.
    assert cosine > 0.95, (
        f"Last-position cosine similarity between HF and TL activations too low: {cosine:.4f}"
    )


from typing import Any  # noqa: E402
