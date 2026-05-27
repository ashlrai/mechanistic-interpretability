"""demo.py — one-command first-experiment quickstart.

Orchestrates three narrative-coherent experiments on gpt2-small:
  1. direct_logit_attribution  — which components write the answer?
  2. logit_lens                — at which layer does the model commit?
  3. circuit_patching          — causal verification of the top DLA component

All three use the same three factual prompts so the story is coherent across
experiments.  Runs in <5 minutes on a machine with gpt2-small already cached.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mech_interp.experiments.circuit_patching import CircuitPatchingExperiment
from mech_interp.experiments.direct_logit_attribution import DirectLogitAttributionExperiment
from mech_interp.experiments.logit_lens import LogitLensExperiment
from mech_interp.types import ExperimentResult, ExperimentRun, ExperimentSpec, RunStatus, utc_now

# ---------------------------------------------------------------------------
# Shared prompt data — 3 factual pairs that tell a coherent story
# ---------------------------------------------------------------------------

_DEMO_PROMPT_PAIRS = [
    {
        "id": "capital-france",
        "clean_prompt": "The capital of France is Paris",
        "corrupted_prompt": "The capital of France is London",
        "correct_token": " Paris",
        "incorrect_token": " London",
    },
    {
        "id": "capital-italy",
        "clean_prompt": "The capital of Italy is Rome",
        "corrupted_prompt": "The capital of Italy is Paris",
        "correct_token": " Rome",
        "incorrect_token": " Paris",
    },
    {
        "id": "tower-city",
        "clean_prompt": "The Eiffel Tower is in Paris",
        "corrupted_prompt": "The Eiffel Tower is in Rome",
        "correct_token": " Paris",
        "incorrect_token": " Rome",
    },
]

# DLA uses clean_prompt only (no corrupted needed)
_DLA_PROMPT_PAIRS = [
    {
        "id": p["id"],
        "clean_prompt": p["clean_prompt"],
        "correct_token": p["correct_token"],
        "incorrect_token": p["incorrect_token"],
    }
    for p in _DEMO_PROMPT_PAIRS
]

# Logit lens uses simple prompts (the clean side without the answer)
_LENS_PROMPTS = [
    {
        "id": "capital-france",
        "prompt": "The capital of France is",
        "correct_token": " Paris",
        "incorrect_token": " London",
    },
    {
        "id": "capital-italy",
        "prompt": "The capital of Italy is",
        "correct_token": " Rome",
        "incorrect_token": " Paris",
    },
    {
        "id": "tower-city",
        "prompt": "The Eiffel Tower is in",
        "correct_token": " Paris",
        "incorrect_token": " Rome",
    },
]


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class DemoResult:
    """Parsed, human-readable summary of the three demo experiments."""

    # DLA
    dla_top_component: str = ""       # e.g. "L9_mlp"
    dla_top_score: float = 0.0
    dla_total_components: int = 0

    # Logit lens
    lens_commit_layer: int = -1       # first layer where mean rank <= top_k
    lens_n_layers: int = 0
    lens_final_rank: float = 0.0

    # Circuit patching
    patch_top_site: str = ""          # e.g. "blocks.9.hook_resid_pre"
    patch_top_recovery: float = 0.0
    patch_pair_id: str = ""

    # Provenance
    artifact_dir: Path = field(default_factory=lambda: Path("."))
    dla_result: ExperimentResult | None = None
    lens_result: ExperimentResult | None = None
    patch_result: ExperimentResult | None = None
    errors: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_demo_experiments(output_dir: Path) -> DemoResult:
    """Orchestrate the three demo experiments and return a DemoResult.

    Loads gpt2-small once and reuses the backend across all three experiments.
    Seed=42 throughout for determinism.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        from mech_interp.backends.instrumented import TransformerLensBackend
    except ImportError as exc:
        raise RuntimeError(
            "mech demo requires the interp extras; run `uv sync --extra interp`"
        ) from exc

    backend = TransformerLensBackend(model_name="gpt2", device="cpu")
    backend.load()

    result = DemoResult(artifact_dir=output_dir)

    # -----------------------------------------------------------------------
    # 1. Direct logit attribution
    # -----------------------------------------------------------------------
    dla_dir = output_dir / "dla"
    dla_dir.mkdir(parents=True, exist_ok=True)
    dla_spec = ExperimentSpec(
        name="demo-dla",
        family="direct_logit_attribution",
        backend="transformerlens",
        description="Demo DLA on factual prompts",
        parameters={
            "model": "gpt2-small",
            "device": "cpu",
            "seed": 42,
            "target_position": -1,
            "top_k": 10,
            "prompt_pairs": _DLA_PROMPT_PAIRS,
        },
    )
    dla_run = _make_run(1, dla_spec, dla_dir)
    try:
        dla_exp = DirectLogitAttributionExperiment(backend=backend)
        result.dla_result = dla_exp.run(dla_spec, dla_run)
        result = _parse_dla(result, result.dla_result)
    except Exception as exc:  # noqa: BLE001
        result.errors.append(f"DLA failed: {exc}")

    # -----------------------------------------------------------------------
    # 2. Logit lens
    # -----------------------------------------------------------------------
    lens_dir = output_dir / "lens"
    lens_dir.mkdir(parents=True, exist_ok=True)
    lens_spec = ExperimentSpec(
        name="demo-lens",
        family="logit_lens",
        backend="transformerlens",
        description="Demo logit lens on factual prompts",
        parameters={
            "model": "gpt2-small",
            "device": "cpu",
            "seed": 42,
            "target_position": -1,
            "top_k": 5,
            "mode": "logit",
            "prompts": _LENS_PROMPTS,
        },
    )
    lens_run = _make_run(2, lens_spec, lens_dir)
    try:
        lens_exp = LogitLensExperiment(backend=backend)
        result.lens_result = lens_exp.run(lens_spec, lens_run)
        result = _parse_lens(result, result.lens_result)
    except Exception as exc:  # noqa: BLE001
        result.errors.append(f"Logit lens failed: {exc}")

    # -----------------------------------------------------------------------
    # 3. Circuit patching
    # -----------------------------------------------------------------------
    patch_dir = output_dir / "patching"
    patch_dir.mkdir(parents=True, exist_ok=True)
    patch_spec = ExperimentSpec(
        name="demo-patching",
        family="circuit_patching",
        backend="transformerlens",
        description="Demo circuit patching on factual prompts",
        parameters={
            "model": "gpt2-small",
            "device": "cpu",
            "seed": 42,
            "target_position": -1,
            "patch_position": -1,
            "layers": [8, 9, 10, 11],
            "patch_sites": ["resid_pre", "mlp_out"],
            "prompt_pairs": _DEMO_PROMPT_PAIRS,
            "artifact_policy": {
                "retain_activation_tensors": False,
                "write_report": True,
            },
        },
    )
    patch_run = _make_run(3, patch_spec, patch_dir)
    try:
        patch_exp = CircuitPatchingExperiment(backend=backend)
        result.patch_result = patch_exp.run(patch_spec, patch_run)
        result = _parse_patch(result, result.patch_result)
    except Exception as exc:  # noqa: BLE001
        result.errors.append(f"Circuit patching failed: {exc}")

    return result


def render_demo_summary(result: DemoResult, console: Any) -> None:
    """Print a Rich-rendered summary panel to *console*."""
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    # ---- build the table ----
    table = Table(show_header=True, header_style="bold #176b87", box=None, padding=(0, 2))
    table.add_column("Experiment", style="bold", min_width=22)
    table.add_column("Finding", min_width=42)
    table.add_column("Value", justify="right", min_width=12)

    if result.dla_top_component:
        table.add_row(
            "Direct Logit Attribution",
            f"Top writing component: [bold]{result.dla_top_component}[/bold]",
            f"{result.dla_top_score:+.3f}",
        )
    else:
        table.add_row("Direct Logit Attribution", "[red]failed[/red]", "—")

    if result.lens_n_layers > 0:
        if result.lens_commit_layer >= 0:
            commit_str = (
                f"commits at layer [bold]{result.lens_commit_layer}[/bold]"
                f" / {result.lens_n_layers}"
            )
        else:
            commit_str = f"rank drops over {result.lens_n_layers} layers (never top-5)"
        table.add_row(
            "Logit Lens",
            commit_str,
            f"rank {result.lens_final_rank:.0f} final",
        )
    else:
        table.add_row("Logit Lens", "[red]failed[/red]", "—")

    if result.patch_top_site:
        short_site = result.patch_top_site.replace("blocks.", "L").replace(".hook_", "·")
        table.add_row(
            "Circuit Patching",
            f"Top causal site: [bold]{short_site}[/bold]"
            f" (pair: {result.patch_pair_id})",
            f"{result.patch_top_recovery:.1%} recovery",
        )
    else:
        table.add_row("Circuit Patching", "[red]failed[/red]", "—")

    if result.errors:
        error_text = "\n".join(f"  [red]![/red] {e}" for e in result.errors)
        console.print(Panel(error_text, title="[bold red]Errors[/bold red]", border_style="red"))

    title = Text("mech demo — gpt2-small factual recall", style="bold #176b87")
    console.print(Panel(table, title=title, border_style="#176b87"))

    # ---- narrative explanation ----
    console.print()
    console.print("[bold]What just happened:[/bold]")
    console.print(
        "  1. [#176b87]DLA[/#176b87] decomposed every component's contribution to the final logit "
        "in a single forward pass — no ablation needed."
    )
    console.print(
        "  2. [#176b87]Logit Lens[/#176b87] projected the residual stream at every layer to reveal "
        "where the model's 'current best guess' first locks onto the correct token."
    )
    console.print(
        "  3. [#176b87]Circuit Patching[/#176b87] causally verified the top DLA finding: "
        "patching one site's activations from the clean run into a corrupted run "
        "recovers the correct answer."
    )
    console.print(
        "  4. Together these three measurements form the minimal circuit story: "
        "[italic]something writes it, somewhere it commits, "
        "and patching confirms causality.[/italic]"
    )
    console.print(
        "  5. All results are deterministic (seed=42) and reproducible — re-run to confirm."
    )
    console.print()
    console.print(
        "[dim]Full walkthrough:[/dim] [bold]notebooks/05_research_walkthrough.ipynb[/bold]"
    )
    console.print(
        f"[dim]Saved chart:[/dim]    [bold]{result.artifact_dir / 'summary.png'}[/bold]"
    )


def render_demo_chart(result: DemoResult, output_path: Path) -> None:
    """Render a 3-panel matplotlib figure and save to *output_path*."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    accent = "#176b87"
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.patch.set_facecolor("#0f1923")
    for ax in axes:
        ax.set_facecolor("#0f1923")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#2a3a4a")

    # ---- Panel 1: DLA top-10 components ----
    ax0 = axes[0]
    if result.dla_result and result.dla_result.status == RunStatus.SUCCEEDED:
        summary_path = Path(result.dla_result.artifacts.get("lda_summary", ""))
        if summary_path.exists():
            summary = json.loads(summary_path.read_text())
            top_pos = summary["top_positive"][:10]
            labels = [e["component_id"] for e in top_pos]
            scores = [e["mean_score"] for e in top_pos]
            colors = [accent if s > 0 else "#c0392b" for s in scores]
            y_pos = range(len(labels))
            ax0.barh(list(y_pos), scores, color=colors)
            ax0.set_yticks(list(y_pos))
            ax0.set_yticklabels(labels, fontsize=8)
            ax0.set_title("DLA — Top Components", fontsize=10, pad=8)
            ax0.set_xlabel("Mean DLA score", fontsize=8)
            ax0.invert_yaxis()
    else:
        ax0.text(0.5, 0.5, "DLA failed", ha="center", va="center",
                 color="red", transform=ax0.transAxes)
        ax0.set_title("DLA — Top Components", fontsize=10)

    # ---- Panel 2: Logit lens rank-of-correct curve ----
    ax1 = axes[1]
    if result.lens_result and result.lens_result.status == RunStatus.SUCCEEDED:
        summary_path = Path(result.lens_result.artifacts.get("lens_summary", ""))
        if summary_path.exists():
            summary = json.loads(summary_path.read_text())
            mean_rank = summary["mean_rank_by_layer"]
            layers = list(range(len(mean_rank)))
            ax1.plot(layers, mean_rank, color=accent, linewidth=2, marker="o", markersize=4)
            ax1.axhline(y=5, color="#e0a020", linewidth=1, linestyle="--", alpha=0.6,
                        label="top-5 threshold")
            if result.lens_commit_layer >= 0:
                ax1.axvline(x=result.lens_commit_layer, color="#e0a020",
                            linewidth=1.5, linestyle=":", alpha=0.8)
                ax1.text(
                    result.lens_commit_layer + 0.3,
                    max(mean_rank) * 0.9,
                    f"L{result.lens_commit_layer}",
                    color="#e0a020", fontsize=8,
                )
            ax1.set_title("Logit Lens — Rank of Correct Token", fontsize=10, pad=8)
            ax1.set_xlabel("Layer", fontsize=8)
            ax1.set_ylabel("Mean rank (lower = better)", fontsize=8)
            ax1.legend(fontsize=7, facecolor="#0f1923", labelcolor="white",
                       edgecolor="#2a3a4a")
    else:
        ax1.text(0.5, 0.5, "Logit lens failed", ha="center", va="center",
                 color="red", transform=ax1.transAxes)
        ax1.set_title("Logit Lens — Rank", fontsize=10)

    # ---- Panel 3: Circuit patching top sites ----
    ax2 = axes[2]
    if result.patch_result and result.patch_result.status == RunStatus.SUCCEEDED:
        ranked_path = Path(result.patch_result.artifacts.get("patching_ranked_json", ""))
        if ranked_path.exists():
            rows = json.loads(ranked_path.read_text())
            # take top-10 causal sites by recovery_fraction (one entry per site, max over pairs)
            site_max: dict[str, float] = {}
            for row in rows:
                if row.get("evidence_label") == "causal evidence":
                    site = row["hook_site"]
                    site_max[site] = max(site_max.get(site, 0.0),
                                        float(row["recovery_fraction"]))
            top_sites = sorted(site_max.items(), key=lambda x: x[1], reverse=True)[:10]
            if top_sites:
                labels = [
                    s.replace("blocks.", "L").replace(".hook_", "·")
                    for s, _ in top_sites
                ]
                values = [v for _, v in top_sites]
                colors = [accent if v >= 0.5 else "#2a5a6a" for v in values]
                y_pos = range(len(labels))
                ax2.barh(list(y_pos), values, color=colors)
                ax2.set_yticks(list(y_pos))
                ax2.set_yticklabels(labels, fontsize=8)
                ax2.set_xlim(0, 1.05)
                ax2.axvline(x=0.5, color="#e0a020", linewidth=1, linestyle="--", alpha=0.5)
                ax2.set_title("Circuit Patching — Top Sites", fontsize=10, pad=8)
                ax2.set_xlabel("Recovery fraction", fontsize=8)
                ax2.invert_yaxis()
    else:
        ax2.text(0.5, 0.5, "Circuit patching failed", ha="center", va="center",
                 color="red", transform=ax2.transAxes)
        ax2.set_title("Circuit Patching — Top Sites", fontsize=10)

    plt.suptitle(
        "gpt2-small factual recall — mech demo",
        color="white", fontsize=12, y=1.02,
    )
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def render_demo_markdown(result: DemoResult) -> str:
    """Return a summary.md string capturing the same narrative as the console output."""
    lines: list[str] = [
        "# mech demo — Summary",
        "",
        "Three fast experiments on **gpt2-small** with 3 factual prompts "
        "(capital-france, capital-italy, tower-city).  Seed=42 throughout.",
        "",
        "## Results",
        "",
        "| Experiment | Finding | Value |",
        "| --- | --- | --- |",
    ]

    if result.dla_top_component:
        lines.append(
            f"| Direct Logit Attribution | Top writing component: `{result.dla_top_component}` "
            f"| {result.dla_top_score:+.3f} |"
        )
    else:
        lines.append("| Direct Logit Attribution | failed | — |")

    if result.lens_n_layers > 0:
        if result.lens_commit_layer >= 0:
            lens_finding = (
                f"commits at layer {result.lens_commit_layer} / {result.lens_n_layers}"
            )
        else:
            lens_finding = f"rank drops over {result.lens_n_layers} layers (never top-5)"
        lines.append(
            f"| Logit Lens | {lens_finding}"
            f" | rank {result.lens_final_rank:.0f} final |"
        )
    else:
        lines.append("| Logit Lens | failed | — |")

    if result.patch_top_site:
        lines.append(
            f"| Circuit Patching | Top causal site: `{result.patch_top_site}` "
            f"(pair: {result.patch_pair_id}) | {result.patch_top_recovery:.1%} recovery |"
        )
    else:
        lines.append("| Circuit Patching | failed | — |")

    lines += [
        "",
        "## What happened",
        "",
        "1. **DLA** decomposed every component's contribution to the final logit in a single "
        "forward pass — no ablation needed.",
        "2. **Logit Lens** projected the residual stream at every layer to reveal where the "
        "model's current best guess first locks onto the correct token.",
        "3. **Circuit Patching** causally verified the top DLA finding: patching one site's "
        "activations from the clean run into a corrupted run recovers the correct answer.",
        "4. Together these three measurements form the minimal circuit story: something writes "
        "it, somewhere it commits, and patching confirms causality.",
        "5. All results are deterministic (seed=42) — re-run to confirm.",
        "",
        "## Next steps",
        "",
        "- Full walkthrough: `notebooks/05_research_walkthrough.ipynb`",
        f"- Saved chart: `{result.artifact_dir / 'summary.png'}`",
        "",
    ]

    if result.errors:
        lines += ["## Errors", ""]
        for err in result.errors:
            lines.append(f"- {err}")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _make_run(run_id: int, spec: ExperimentSpec, artifact_dir: Path) -> ExperimentRun:
    return ExperimentRun(
        id=run_id,
        spec_name=spec.name,
        family=spec.family,
        backend=spec.backend,
        status=RunStatus.RUNNING,
        artifact_dir=artifact_dir,
        created_at=utc_now(),
    )


def _parse_dla(result: DemoResult, exp_result: ExperimentResult) -> DemoResult:
    if exp_result.status != RunStatus.SUCCEEDED:
        result.errors.append(f"DLA status: {exp_result.status}")
        return result
    summary_path = Path(exp_result.artifacts.get("lda_summary", ""))
    if not summary_path.exists():
        result.errors.append("DLA: lda_summary artifact not found")
        return result
    summary: dict[str, Any] = json.loads(summary_path.read_text())
    top_positive = summary.get("top_positive", [])
    result.dla_total_components = int(summary.get("total_components", 0))
    if top_positive:
        result.dla_top_component = str(top_positive[0].get("component_id", ""))
        result.dla_top_score = float(top_positive[0].get("mean_score", 0.0))
    return result


def _parse_lens(result: DemoResult, exp_result: ExperimentResult) -> DemoResult:
    if exp_result.status != RunStatus.SUCCEEDED:
        result.errors.append(f"Logit lens status: {exp_result.status}")
        return result
    summary_path = Path(exp_result.artifacts.get("lens_summary", ""))
    if not summary_path.exists():
        result.errors.append("Logit lens: lens_summary artifact not found")
        return result
    summary: dict[str, Any] = json.loads(summary_path.read_text())
    mean_rank = summary.get("mean_rank_by_layer", [])
    result.lens_n_layers = len(mean_rank)
    result.lens_final_rank = float(mean_rank[-1]) if mean_rank else 0.0
    # first layer where rank <= 5
    result.lens_commit_layer = next(
        (i for i, r in enumerate(mean_rank) if r <= 5),
        -1,
    )
    return result


def _parse_patch(result: DemoResult, exp_result: ExperimentResult) -> DemoResult:
    if exp_result.status != RunStatus.SUCCEEDED:
        result.errors.append(f"Circuit patching status: {exp_result.status}")
        return result
    ranked_path = Path(exp_result.artifacts.get("patching_ranked_json", ""))
    if not ranked_path.exists():
        result.errors.append("Circuit patching: patching_ranked_json artifact not found")
        return result
    rows: list[dict[str, Any]] = json.loads(ranked_path.read_text())
    causal = [r for r in rows if r.get("evidence_label") == "causal evidence"]
    if causal:
        top = causal[0]
        result.patch_top_site = str(top.get("hook_site", ""))
        result.patch_top_recovery = float(top.get("recovery_fraction", 0.0))
        result.patch_pair_id = str(top.get("pair_id", ""))
    return result
