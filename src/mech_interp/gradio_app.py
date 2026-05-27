"""Gradio interactive demo for the Mechanistic Interpretability Platform.

Provides a live web UI where a user types a prompt and sees four analyses
rendered in real time:
  - Logit lens: rank-of-correct vs layer (line chart)
  - Direct logit attribution: top-10 components by contribution (bar chart)
  - Activation magnitudes: per-layer residual-stream norms (line chart)
  - Top predictions per layer: table of top-3 tokens at every 4th layer

A short narrative ties the panels together after analysis completes.

Usage:
    mech gradio            # launches on http://localhost:7860
    mech gradio --port 7861 --share
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from matplotlib.figure import Figure

# ---------------------------------------------------------------------------
# Module-level model cache — keyed by model name so first load is one-time.
# ---------------------------------------------------------------------------
_MODEL_CACHE: dict[str, Any] = {}

_ACCENT = "#176b87"
_GRID_ALPHA = 0.25
_SUPPORTED_MODELS = ["gpt2-small", "gpt2-medium", "qwen2.5-1.5b-instruct"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_demo_app() -> Any:
    """Construct the Gradio Blocks app."""
    try:
        import gradio as gr
    except ImportError as exc:
        raise ImportError(
            "gradio is not installed. Run: uv sync --extra gradio"
        ) from exc

    with gr.Blocks(title="Mechanistic Interpretability Platform — Live Demo") as demo:
        gr.Markdown(
            "## Mechanistic Interpretability Platform — Live Demo\n"
            "Type a prompt and click **Analyze** to run logit lens, direct logit attribution, "
            "activation capture, and top-predictions-per-layer in real time."
        )

        with gr.Row():
            prompt_box = gr.Textbox(
                label="Prompt",
                value="The capital of France is",
                placeholder="Enter a prompt to analyze…",
                scale=3,
            )
            analyze_btn = gr.Button("Analyze", variant="primary", scale=1)

        with gr.Row():
            model_dd = gr.Dropdown(
                choices=_SUPPORTED_MODELS,
                value="gpt2-small",
                label="Model",
                scale=1,
            )
            correct_box = gr.Textbox(
                label='Correct token (e.g. " Paris")',
                value=" Paris",
                scale=1,
            )
            incorrect_box = gr.Textbox(
                label='Incorrect token (e.g. " Rome")',
                value=" Rome",
                scale=1,
            )

        with gr.Row():
            lens_plot = gr.Plot(label="Logit Lens — rank of correct token per layer")
            dla_plot = gr.Plot(label="Direct Logit Attribution — top 10 components")

        with gr.Row():
            act_plot = gr.Plot(label="Activation Magnitudes — residual-stream norm per layer")
            top_preds_table = gr.Dataframe(
                label="Top-3 predictions at every 4th layer",
                headers=["Layer", "Rank 1", "P1", "Rank 2", "P2", "Rank 3", "P3"],
                datatype=["number", "str", "number", "str", "number", "str", "number"],
            )

        narrative_md = gr.Markdown(label="Narrative")

        analyze_btn.click(
            fn=analyze_prompt,
            inputs=[prompt_box, model_dd, correct_box, incorrect_box],
            outputs=[lens_plot, dla_plot, act_plot, top_preds_table, narrative_md],
        )

    return demo


def analyze_prompt(
    prompt: str,
    model_name: str,
    correct_token: str,
    incorrect_token: str,
) -> tuple[Figure, Figure, Figure, list[list[Any]], str]:
    """Run all 4 analyses; return (lens_fig, dla_fig, act_fig, table_rows, narrative)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    prompt = prompt.strip()
    # Ensure tokens start with a space (transformer tokenizers almost always
    # have a leading-space variant for mid-sentence tokens).
    correct_token = _normalize_token(correct_token, default=" Paris")
    incorrect_token = _normalize_token(incorrect_token, default=" Rome")

    model = _load_model(model_name)

    # ------------------------------------------------------------------
    # 1. Logit lens — forward pass with resid_post cache
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    lens_data = _run_logit_lens(model, prompt, correct_token, incorrect_token)
    lens_fig = _plot_lens(lens_data)
    lens_time = time.perf_counter() - t0

    # ------------------------------------------------------------------
    # 2. Direct logit attribution — same forward pass reused internally
    # ------------------------------------------------------------------
    t1 = time.perf_counter()
    dla_data = _run_dla(model, prompt, correct_token, incorrect_token)
    dla_fig = _plot_dla(dla_data)
    dla_time = time.perf_counter() - t1

    # ------------------------------------------------------------------
    # 3. Activation magnitudes — lightweight resid_post norm per layer
    # ------------------------------------------------------------------
    t2 = time.perf_counter()
    act_data = _run_activation_magnitudes(model, prompt)
    act_fig = _plot_activations(act_data)
    act_time = time.perf_counter() - t2

    # ------------------------------------------------------------------
    # 4. Top-predictions table — reuse lens_data (already has top_k)
    # ------------------------------------------------------------------
    table_rows = _build_top_preds_table(lens_data)

    # ------------------------------------------------------------------
    # Narrative
    # ------------------------------------------------------------------
    narrative = _build_narrative(
        lens_data=lens_data,
        dla_data=dla_data,
        correct_token=correct_token,
        incorrect_token=incorrect_token,
        timings={"lens": lens_time, "dla": dla_time, "act": act_time},
    )

    plt.close("all")
    return lens_fig, dla_fig, act_fig, table_rows, narrative


# ---------------------------------------------------------------------------
# Model loading (cached)
# ---------------------------------------------------------------------------

def _normalize_token(raw: str, *, default: str) -> str:
    """Strip, fall back to *default* if empty, and ensure a leading space."""
    stripped = raw.strip()
    if not stripped:
        return default
    return stripped if stripped.startswith(" ") else " " + stripped


def _load_model(model_name: str) -> Any:
    if model_name in _MODEL_CACHE:
        return _MODEL_CACHE[model_name]
    try:
        import transformer_lens
    except ImportError as exc:
        raise RuntimeError(
            "transformer-lens is not installed. Run: uv sync --extra interp"
        ) from exc
    model = transformer_lens.HookedTransformer.from_pretrained(model_name)
    model.eval()
    _MODEL_CACHE[model_name] = model
    return model


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def _run_logit_lens(
    model: Any,
    prompt: str,
    correct_token: str,
    incorrect_token: str,
) -> dict[str, Any]:
    """Single forward pass; return per-layer lens data reusing experiment internals."""
    import torch

    from mech_interp.experiments.logit_lens import _LensParams, _PromptConfig, _run_lens_for_prompt

    prompt_cfg = _PromptConfig(
        id="demo",
        prompt=prompt,
        correct_token=correct_token,
        incorrect_token=incorrect_token,
    )
    params = _LensParams(
        model=model.cfg.model_name if hasattr(model.cfg, "model_name") else "gpt2-small",
        prompts=[prompt_cfg],
        top_k=3,
        mode="logit",
    )
    per_layer = _run_lens_for_prompt(
        model=model,
        prompt_cfg=prompt_cfg,
        params=params,
        torch=torch,
        tuned_transforms=None,
    )
    n_layers = model.cfg.n_layers
    ranks = [rec["rank_correct"] for rec in per_layer]
    top_k_by_layer = [rec["top_k"] for rec in per_layer]

    # Find decision layer — first layer where correct token is rank 1
    decision_layer: int | None = None
    for rec in per_layer:
        if rec["rank_correct"] == 1:
            decision_layer = rec["layer"]
            break

    return {
        "n_layers": n_layers,
        "layers": list(range(len(per_layer))),
        "ranks": ranks,
        "top_k_by_layer": top_k_by_layer,
        "decision_layer": decision_layer,
        "per_layer": per_layer,
    }


def _run_dla(
    model: Any,
    prompt: str,
    correct_token: str,
    incorrect_token: str,
) -> dict[str, Any]:
    """Single forward pass for DLA; return top-10 components."""
    import torch

    from mech_interp.experiments.direct_logit_attribution import (
        _attribute_prompt,
        _build_summary,
        _DLAParams,
        _PromptPair,
    )

    class _FakeBackend:
        def __init__(self, m: Any) -> None:
            self.model = m

    pair = _PromptPair(
        id="demo",
        clean_prompt=prompt,
        correct_token=correct_token,
        incorrect_token=incorrect_token,
    )
    params = _DLAParams(
        model="gpt2-small",
        prompt_pairs=[pair],
        top_k=10,
    )
    rows = _attribute_prompt(_FakeBackend(model), pair, params, torch)
    summary = _build_summary(rows, top_k=10)

    top_pos = summary["top_positive"][:10]
    top_neg = summary["top_negative"][:10]

    # Merge: top positive and top negative, sorted by abs score
    combined = sorted(top_pos + top_neg, key=lambda x: abs(x["mean_score"]), reverse=True)[:10]

    return {
        "components": combined,
        "top_positive": top_pos,
        "top_negative": top_neg,
        "summary": summary,
    }


def _run_activation_magnitudes(model: Any, prompt: str) -> dict[str, Any]:
    """Forward pass capturing resid_post per layer; return L2 norms."""
    import torch

    n_layers = model.cfg.n_layers
    hook_names = {f"blocks.{L}.hook_resid_post" for L in range(n_layers)}

    _, cache = model.run_with_cache(
        prompt,
        names_filter=lambda name: name in hook_names,
    )

    layers = []
    norms = []
    for L in range(n_layers):
        key = f"blocks.{L}.hook_resid_post"
        if key not in cache:
            continue
        resid = cache[key]  # [batch, seq, d_model]
        # Last token position, L2 norm across d_model
        vec = resid[0, -1, :]
        norm = float(torch.norm(vec).item())
        layers.append(L)
        norms.append(norm)

    return {"layers": layers, "norms": norms}


# ---------------------------------------------------------------------------
# Plot builders
# ---------------------------------------------------------------------------

def _plot_lens(data: dict[str, Any]) -> Figure:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.plot(data["layers"], data["ranks"], color=_ACCENT, linewidth=2, marker="o", markersize=4)
    if data["decision_layer"] is not None:
        ax.axvline(data["decision_layer"], color="#e07b39", linestyle="--", linewidth=1.2,
                   label=f"Rank 1 @ L{data['decision_layer']}")
        ax.legend(fontsize=8)
    ax.set_xlabel("Layer", fontsize=9)
    ax.set_ylabel("Rank of correct token", fontsize=9)
    ax.set_title("Logit Lens", fontsize=10, fontweight="bold")
    ax.grid(True, alpha=_GRID_ALPHA)
    ax.invert_yaxis()  # rank 1 at top
    fig.tight_layout()
    return fig


def _plot_dla(data: dict[str, Any]) -> Figure:
    import matplotlib.pyplot as plt

    components = data["components"]
    if not components:
        fig, ax = plt.subplots(figsize=(5.5, 3.5))
        ax.text(0.5, 0.5, "No DLA data", ha="center", va="center")
        return fig

    labels = [c["component_id"] for c in components]
    scores = [c["mean_score"] for c in components]
    colors = [_ACCENT if s > 0 else "#c0392b" for s in scores]

    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.barh(labels, scores, color=colors, edgecolor="white", linewidth=0.4)
    ax.axvline(0, color="#555", linewidth=0.8)
    ax.set_xlabel("DLA score (positive = toward correct token)", fontsize=8)
    ax.set_title("Direct Logit Attribution — top 10", fontsize=10, fontweight="bold")
    ax.grid(True, axis="x", alpha=_GRID_ALPHA)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    return fig


def _plot_activations(data: dict[str, Any]) -> Figure:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.plot(data["layers"], data["norms"], color=_ACCENT, linewidth=2, marker="o", markersize=4)
    ax.set_xlabel("Layer", fontsize=9)
    ax.set_ylabel("Residual-stream L2 norm (last token)", fontsize=9)
    ax.set_title("Activation Magnitudes", fontsize=10, fontweight="bold")
    ax.grid(True, alpha=_GRID_ALPHA)
    fig.tight_layout()
    return fig


def _build_top_preds_table(lens_data: dict[str, Any]) -> list[list[Any]]:
    """Return rows for display: every 4th layer (plus last), top-3 tokens + probs."""
    import math

    n = lens_data["n_layers"]
    sampled_layers = list(range(0, n, 4))
    if (n - 1) not in sampled_layers:
        sampled_layers.append(n - 1)

    per_layer = lens_data["per_layer"]
    layer_map = {rec["layer"]: rec for rec in per_layer}

    rows = []
    for L in sampled_layers:
        rec = layer_map.get(L)
        if rec is None:
            continue
        top_k = rec.get("top_k", [])
        # Convert logits to softmax probabilities
        logits = [entry["logit"] for entry in top_k]
        if logits:
            max_l = max(logits)
            exps = [math.exp(lg - max_l) for lg in logits]
            total = sum(exps)
            probs = [e / total for e in exps]
        else:
            probs = []

        row: list[Any] = [L]
        for i in range(3):
            if i < len(top_k):
                row.append(top_k[i]["token"])
                row.append(round(probs[i], 4) if i < len(probs) else 0.0)
            else:
                row.extend(["", 0.0])
        rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# Narrative builder
# ---------------------------------------------------------------------------

def _build_narrative(
    *,
    lens_data: dict[str, Any],
    dla_data: dict[str, Any],
    correct_token: str,
    incorrect_token: str,
    timings: dict[str, float],
) -> str:
    n_layers = lens_data["n_layers"]
    decision_layer = lens_data["decision_layer"]
    ranks = lens_data["ranks"]
    final_rank = ranks[-1] if ranks else None

    top_pos = dla_data["top_positive"]
    top_writer = top_pos[0] if top_pos else None

    # Sentence 1: when the model commits
    if decision_layer is not None:
        commit_sentence = (
            f"The model first predicts **{correct_token.strip()}** as rank 1 at "
            f"**layer {decision_layer}** (out of {n_layers} total layers)."
        )
    elif final_rank is not None and final_rank <= 5:
        commit_sentence = (
            f"**{correct_token.strip()}** reaches rank {final_rank} by the final layer "
            f"but never hits rank 1 in intermediate layers."
        )
    else:
        commit_sentence = (
            f"**{correct_token.strip()}** does not reach rank 1 in any layer "
            f"(final rank: {final_rank})."
        )

    # Sentence 2: strongest DLA writer
    if top_writer:
        writer_sentence = (
            f"The strongest writer toward {correct_token.strip()} is "
            f"**{top_writer['component_id']}** "
            f"(DLA score: {top_writer['mean_score']:+.2f})."
        )
    else:
        writer_sentence = "No positive DLA contributors found."

    # Sentence 3: early layers + timing
    early_rank = ranks[0] if ranks else None
    mid_layer = n_layers // 2
    mid_rank = ranks[mid_layer] if len(ranks) > mid_layer else None

    early_part = (
        f"Early layers (L0) show rank {early_rank}" if early_rank else "Early layers unavailable"
    )
    mid_part = f"mid-network (L{mid_layer}) rank {mid_rank}" if mid_rank else ""
    context_sentence = (
        f"{early_part}; {mid_part} — "
        f"analysis completed in {sum(timings.values()):.1f}s "
        f"(lens {timings['lens']:.1f}s, DLA {timings['dla']:.1f}s, "
        f"activations {timings['act']:.1f}s)."
    )

    return f"""### Analysis Narrative

{commit_sentence}

{writer_sentence}

{context_sentence}
"""
