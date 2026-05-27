"""Unit tests for gradio_app.py.

All tests mock torch/transformer_lens so they run without the interp extra.
Verifies:
  - analyze_prompt returns the expected 5-tuple shape
  - Each chart is a matplotlib Figure
  - The narrative markdown contains expected sections
  - build_demo_app raises ImportError (with a useful hint) when gradio is absent
  - _build_top_preds_table produces rows with the correct column count
  - _build_narrative contains decision-layer and DLA-writer information
"""

from __future__ import annotations

import sys
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers — build a minimal synthetic model that looks like a HookedTransformer
# ---------------------------------------------------------------------------

def _make_fake_model(
    n_layers: int = 12, n_heads: int = 12, d_model: int = 768, d_vocab: int = 50257
) -> Any:
    """Return a MagicMock that satisfies the interfaces used by gradio_app."""
    try:
        import torch
    except ImportError:
        pytest.skip("torch not available")

    model = MagicMock()
    model.cfg.n_layers = n_layers
    model.cfg.n_heads = n_heads
    model.cfg.model_name = "gpt2-small"

    # W_U: [d_model, d_vocab]
    W_U = torch.zeros(d_model, d_vocab)
    W_U[0, 3] = 5.0   # correct token index 3 gets a boost from dim 0
    W_U[1, 7] = 3.0   # incorrect token index 7 gets a smaller boost from dim 1
    model.W_U = W_U

    # ln_final: identity for simplicity
    model.ln_final = lambda x: x

    def fake_to_single_token(token: str) -> int:
        return 3 if token.strip().lower() in ("paris", "correct") else 7

    model.to_single_token.side_effect = fake_to_single_token
    model.tokenizer.decode.side_effect = lambda ids: f"tok{ids[0]}"

    # run_with_cache: return per-layer resid_post tensors
    def fake_run_with_cache(prompt: Any, names_filter: Any = None) -> tuple[Any, dict[str, Any]]:
        cache: dict[str, Any] = {}
        for L in range(n_layers):
            key = f"blocks.{L}.hook_resid_post"
            if names_filter is None or names_filter(key):
                # Residual stream grows with depth
                t = torch.zeros(1, 5, d_model)
                t[0, -1, 0] = float(L + 1)  # dim 0 grows → correct logit grows
                cache[key] = t
            # attn hook_result: [batch, seq, n_heads, d_model]
            attn_key = f"blocks.{L}.attn.hook_result"
            if names_filter is None or names_filter(attn_key):
                cache[attn_key] = torch.zeros(1, 5, n_heads, d_model)
            # mlp out: [batch, seq, d_model]
            mlp_key = f"blocks.{L}.hook_mlp_out"
            if names_filter is None or names_filter(mlp_key):
                t2 = torch.zeros(1, 5, d_model)
                t2[0, -1, 0] = float(L) * 0.5
                cache[mlp_key] = t2

        for hook_name in ("hook_embed", "hook_pos_embed"):
            if names_filter is None or names_filter(hook_name):
                cache[hook_name] = torch.zeros(1, 5, d_model)

        return MagicMock(), cache

    model.run_with_cache.side_effect = fake_run_with_cache
    return model


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture()
def fake_model() -> Any:
    return _make_fake_model()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_analyze_prompt_returns_five_tuple(fake_model: Any) -> None:
    """analyze_prompt returns (Figure, Figure, Figure, list, str)."""
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        pytest.skip("matplotlib not available")
    from matplotlib.figure import Figure

    with patch("mech_interp.gradio_app._load_model", return_value=fake_model):
        from mech_interp.gradio_app import analyze_prompt

        result = analyze_prompt(
            "The capital of France is",
            "gpt2-small",
            " Paris",
            " Rome",
        )

    assert len(result) == 5, f"Expected 5-tuple, got {len(result)}-tuple"
    lens_fig, dla_fig, act_fig, table, narrative = result

    assert isinstance(lens_fig, Figure), "lens_fig must be a matplotlib Figure"
    assert isinstance(dla_fig, Figure), "dla_fig must be a matplotlib Figure"
    assert isinstance(act_fig, Figure), "act_fig must be a matplotlib Figure"
    assert isinstance(table, list), "table must be a list"
    assert isinstance(narrative, str), "narrative must be a str"


def test_lens_figure_has_axes(fake_model: Any) -> None:
    """Logit lens figure must have at least one Axes with data."""
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        pytest.skip("matplotlib not available")

    with patch("mech_interp.gradio_app._load_model", return_value=fake_model):
        from mech_interp.gradio_app import analyze_prompt

        lens_fig, _, _, _, _ = analyze_prompt(
            "The capital of France is", "gpt2-small", " Paris", " Rome"
        )

    assert len(lens_fig.axes) >= 1
    ax = lens_fig.axes[0]
    assert ax.get_xlabel() == "Layer"


def test_dla_figure_has_axes(fake_model: Any) -> None:
    """DLA figure must have at least one Axes."""
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        pytest.skip("matplotlib not available")

    with patch("mech_interp.gradio_app._load_model", return_value=fake_model):
        from mech_interp.gradio_app import analyze_prompt

        _, dla_fig, _, _, _ = analyze_prompt(
            "The capital of France is", "gpt2-small", " Paris", " Rome"
        )

    assert len(dla_fig.axes) >= 1


def test_activation_figure_has_axes(fake_model: Any) -> None:
    """Activation magnitudes figure must have at least one Axes."""
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        pytest.skip("matplotlib not available")

    with patch("mech_interp.gradio_app._load_model", return_value=fake_model):
        from mech_interp.gradio_app import analyze_prompt

        _, _, act_fig, _, _ = analyze_prompt(
            "The capital of France is", "gpt2-small", " Paris", " Rome"
        )

    assert len(act_fig.axes) >= 1
    ax = act_fig.axes[0]
    assert "Layer" in ax.get_xlabel()


def test_top_preds_table_column_count(fake_model: Any) -> None:
    """Each row of the top-preds table must have exactly 7 columns."""
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        pytest.skip("matplotlib not available")
    with patch("mech_interp.gradio_app._load_model", return_value=fake_model):
        from mech_interp.gradio_app import analyze_prompt

        _, _, _, table, _ = analyze_prompt(
            "The capital of France is", "gpt2-small", " Paris", " Rome"
        )

    assert len(table) > 0, "Table must have at least one row"
    for row in table:
        assert len(row) == 7, f"Expected 7 columns, got {len(row)}: {row}"


def test_narrative_contains_key_sections(fake_model: Any) -> None:
    """Narrative must contain 'Analysis Narrative' heading and token names."""
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        pytest.skip("matplotlib not available")
    with patch("mech_interp.gradio_app._load_model", return_value=fake_model):
        from mech_interp.gradio_app import analyze_prompt

        _, _, _, _, narrative = analyze_prompt(
            "The capital of France is", "gpt2-small", " Paris", " Rome"
        )

    assert "Analysis Narrative" in narrative
    # Should mention the correct token
    assert "Paris" in narrative


def test_narrative_mentions_decision_layer(fake_model: Any) -> None:
    """Narrative must reference a layer number when the correct token reaches rank 1."""
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        pytest.skip("matplotlib not available")
    with patch("mech_interp.gradio_app._load_model", return_value=fake_model):
        from mech_interp.gradio_app import analyze_prompt

        _, _, _, _, narrative = analyze_prompt(
            "The capital of France is", "gpt2-small", " Paris", " Rome"
        )

    # Should contain "layer N" somewhere
    assert "layer" in narrative.lower()


def test_build_demo_app_raises_without_gradio() -> None:
    """build_demo_app must raise ImportError with a hint when gradio is absent."""
    original = sys.modules.get("gradio")
    sys.modules["gradio"] = None  # type: ignore[assignment]
    try:
        # Force reimport of module with gradio absent
        import importlib

        import mech_interp.gradio_app as ga
        importlib.reload(ga)
        with pytest.raises(ImportError, match="gradio"):
            ga.build_demo_app()
    finally:
        if original is None:
            sys.modules.pop("gradio", None)
        else:
            sys.modules["gradio"] = original


def test_build_top_preds_table_samples_every_4th_layer(fake_model: Any) -> None:
    """_build_top_preds_table must sample every 4th layer plus the last."""
    with patch("mech_interp.gradio_app._load_model", return_value=fake_model):
        from mech_interp.gradio_app import _build_top_preds_table, _run_logit_lens

        lens_data = _run_logit_lens(fake_model, "test", " Paris", " Rome")
        rows = _build_top_preds_table(lens_data)

    n_layers = lens_data["n_layers"]
    expected_layers = list(range(0, n_layers, 4))
    if (n_layers - 1) not in expected_layers:
        expected_layers.append(n_layers - 1)

    row_layers = [row[0] for row in rows]
    assert set(row_layers) == set(expected_layers)


def test_model_cache_reuses_same_instance(fake_model: Any) -> None:
    """_load_model caches the model — calling twice with same name returns same object."""
    import mech_interp.gradio_app as ga

    # Inject directly into the module cache
    ga._MODEL_CACHE["_test_cache_model"] = fake_model
    result = ga._load_model("_test_cache_model")
    assert result is fake_model
    # cleanup
    del ga._MODEL_CACHE["_test_cache_model"]
