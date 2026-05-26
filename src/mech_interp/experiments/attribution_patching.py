"""Attribution Patching experiment — first-order Taylor approximation of activation patching.

Reference: Syed, Conmy, Nanda "Attribution Patching: Activation Patching at Industrial Scale"
(2023).  One forward + one backward pass produces approximate effect scores for ALL hook sites
simultaneously, vs O(N) forward passes for exact patching.

Math:
    attribution(h) ≈ (clean[h] - corrupted[h]) · ∇_h L

where L = logit(correct) − logit(incorrect) evaluated on the corrupted run.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from mech_interp.backends import create_instrumented_backend
from mech_interp.experiments.base import Experiment
from mech_interp.storage.artifacts import resolve_run_artifact_dir
from mech_interp.types import (
    ExperimentResult,
    ExperimentRun,
    ExperimentSpec,
    InstrumentedModelBackend,
    RunStatus,
)

ATTRIBUTION_EVIDENCE_LABEL = "attribution_approximation"
_TOP_K_FLAG = 10  # sites whose abs(attribution) appears in top-10 → flag for follow-up


# ---------------------------------------------------------------------------
# Pydantic spec
# ---------------------------------------------------------------------------


class AttributionPromptPairSpec(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str | None = None
    clean_prompt: str
    corrupted_prompt: str
    correct_token: str
    incorrect_token: str
    target_position: int = -1
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("clean_prompt", "corrupted_prompt", "correct_token", "incorrect_token")
    @classmethod
    def strip_non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be empty")
        return value


class HookSiteSpec(BaseModel):
    """Expanded hook site specification supporting aliases + layer ranges."""

    model_config = ConfigDict(extra="allow")

    site: str  # alias or fully-qualified TL name
    layers: list[int] | None = None


class AttributionArtifactPolicy(BaseModel):
    model_config = ConfigDict(extra="allow")

    write_report: bool = True


class AttributionPatchingSpec(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str = "gpt2-small"
    prompt_pairs: list[AttributionPromptPairSpec]
    # hook_sites: list of TL names or alias-spec dicts
    hook_sites: list[str | dict[str, Any]] = Field(default_factory=list)
    target_position: int = -1
    seed: int = 42
    device: str = "auto"
    artifact_policy: AttributionArtifactPolicy = Field(
        default_factory=AttributionArtifactPolicy
    )
    top_k: int = 10


# ---------------------------------------------------------------------------
# Result dataclass (kept plain dict for JSON serialisability)
# ---------------------------------------------------------------------------


def _site_row(
    hook_site: str,
    attribution_score: float,
    abs_attribution_score: float,
) -> dict[str, Any]:
    return {
        "hook_site": hook_site,
        "attribution_score": attribution_score,
        "abs_attribution_score": abs_attribution_score,
        "evidence_label": ATTRIBUTION_EVIDENCE_LABEL,
    }


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------


class AttributionPatchingExperiment(Experiment):
    family = "attribution_patching"

    def __init__(self, backend: InstrumentedModelBackend | None = None) -> None:
        self.backend = backend

    def run(self, spec: ExperimentSpec, run: ExperimentRun) -> ExperimentResult:
        config = AttributionPatchingSpec.model_validate(spec.parameters)
        hook_sites = _resolve_hook_sites(config.hook_sites)

        backend = self.backend or create_instrumented_backend(
            spec.backend,
            {
                "model_name": config.model,
                "device": config.device,
            },
        )

        # Accumulate per-site attribution scores across prompt pairs.
        # site → list of attribution values (one per pair)
        site_attributions: dict[str, list[float]] = {site: [] for site in hook_sites}

        for idx, pair_spec in enumerate(config.prompt_pairs):
            pair_id = pair_spec.id or f"pair-{idx + 1:04d}"
            if pair_spec.target_position != 0:
                target_pos = pair_spec.target_position
            else:
                target_pos = config.target_position
            scores = _attribution_scores_for_pair(
                backend=backend,
                pair_id=pair_id,
                clean_prompt=pair_spec.clean_prompt,
                corrupted_prompt=pair_spec.corrupted_prompt,
                correct_token=pair_spec.correct_token,
                incorrect_token=pair_spec.incorrect_token,
                hook_sites=hook_sites,
                target_position=target_pos,
            )
            for site, score in scores.items():
                site_attributions[site].append(score)

        # Aggregate: mean attribution and mean abs attribution across pairs.
        aggregated: list[dict[str, Any]] = []
        for site in hook_sites:
            values = site_attributions[site]
            if not values:
                continue
            mean_attr = sum(values) / len(values)
            mean_abs = sum(abs(v) for v in values) / len(values)
            aggregated.append(_site_row(site, mean_attr, mean_abs))

        # Rank by descending abs attribution.
        ranked = sorted(aggregated, key=lambda r: r["abs_attribution_score"], reverse=True)
        for rank, row in enumerate(ranked, start=1):
            row["rank"] = rank

        artifact_dir = resolve_run_artifact_dir(run)
        artifact_dir.mkdir(parents=True, exist_ok=True)

        ranked_json_path = artifact_dir / "attribution_ranked.json"
        ranked_csv_path = artifact_dir / "attribution_ranked.csv"
        summary_path = artifact_dir / "attribution_summary.json"
        report_path = artifact_dir / "research_note.md"

        ranked_json_path.write_text(
            json.dumps(ranked, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        _write_csv(ranked_csv_path, ranked)

        top_k_sites = [r["hook_site"] for r in ranked[: config.top_k]]
        mean_abs = (
            sum(r["abs_attribution_score"] for r in ranked) / len(ranked) if ranked else 0.0
        )
        summary = {
            "model": config.model,
            "prompt_pair_count": len(config.prompt_pairs),
            "hook_site_count": len(hook_sites),
            "top_k": config.top_k,
            "top_k_sites": top_k_sites,
            "mean_abs_attribution": mean_abs,
        }
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        artifacts: dict[str, str] = {
            "attribution_ranked_json": str(ranked_json_path.resolve()),
            "attribution_ranked_csv": str(ranked_csv_path.resolve()),
            "attribution_summary": str(summary_path.resolve()),
        }

        if config.artifact_policy.write_report:
            report_path.write_text(_render_report(spec, summary, ranked), encoding="utf-8")
            artifacts["research_note"] = str(report_path.resolve())

        metrics = {
            "prompt_pair_count": float(len(config.prompt_pairs)),
            "hook_site_count": float(len(hook_sites)),
            "top_abs_attribution": float(ranked[0]["abs_attribution_score"]) if ranked else 0.0,
            "mean_abs_attribution": mean_abs,
        }

        notes = _result_notes(ranked)
        return ExperimentResult(
            run_id=run.id,
            status=RunStatus.SUCCEEDED,
            metrics=metrics,
            artifacts=artifacts,
            notes=notes,
        )


# ---------------------------------------------------------------------------
# Core attribution computation
# ---------------------------------------------------------------------------


def _attribution_scores_for_pair(
    *,
    backend: InstrumentedModelBackend,
    pair_id: str,
    clean_prompt: str,
    corrupted_prompt: str,
    correct_token: str,
    incorrect_token: str,
    hook_sites: list[str],
    target_position: int,
) -> dict[str, float]:
    """Run clean cache + corrupted gradient pass, return per-site attribution scalars."""
    # We need the TransformerLensBackend's extended grad-cache method.
    # Dispatch via duck-type so the fake backend in tests can implement the same
    # interface without inheriting from TransformerLensBackend.
    run_with_grad = getattr(backend, "run_with_grad_cache", None)
    if run_with_grad is None:
        raise AttributeError(
            f"Backend {type(backend).__name__} does not implement run_with_grad_cache. "
            "Add a TransformerLensBackend or a compatible fake."
        )

    clean_cache = backend.capture_activations([clean_prompt], hook_sites)
    result: dict[str, float] = run_with_grad(
        prompt=corrupted_prompt,
        hook_sites=hook_sites,
        correct_token=correct_token,
        incorrect_token=incorrect_token,
        target_position=target_position,
        clean_cache=clean_cache,
    )
    return result


# ---------------------------------------------------------------------------
# Hook-site resolution (same alias table as circuit_patching, extended)
# ---------------------------------------------------------------------------

_ALIAS_TEMPLATES = {
    "resid_pre": "blocks.{layer}.hook_resid_pre",
    "resid_post": "blocks.{layer}.hook_resid_post",
    "mlp_out": "blocks.{layer}.hook_mlp_out",
    "mlp_post": "blocks.{layer}.mlp.hook_post",
    "attn_z": "blocks.{layer}.attn.hook_z",
    "attn_out": "blocks.{layer}.attn.hook_result",
}

# gpt2-small has 12 layers (0-11)
_DEFAULT_LAYERS = list(range(12))


def _expand_site_entry(entry: str | dict[str, Any]) -> list[str]:
    """Expand one hook_sites element (string alias or HookSiteSpec dict) to TL names."""
    if isinstance(entry, dict):
        validated = HookSiteSpec.model_validate(entry)
        alias = validated.site
        layers = validated.layers if validated.layers is not None else _DEFAULT_LAYERS
    else:
        alias = entry
        # If it already looks like a fully-qualified TL name, return as-is.
        if "blocks." in alias or "hook_" in alias:
            return [alias]
        layers = _DEFAULT_LAYERS

    template = _ALIAS_TEMPLATES.get(alias)
    if template is None:
        # Treat as literal fully-qualified name.
        return [alias]
    return [template.format(layer=layer) for layer in layers]


def _resolve_hook_sites(raw: list[str | dict[str, Any]]) -> list[str]:
    seen: dict[str, None] = {}
    for entry in raw:
        for site in _expand_site_entry(entry):
            seen[site] = None
    if not seen:
        raise ValueError("attribution_patching requires at least one hook_site.")
    return list(seen)


# ---------------------------------------------------------------------------
# Artifact helpers
# ---------------------------------------------------------------------------


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "rank", "hook_site", "attribution_score", "abs_attribution_score", "evidence_label",
    ]
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _render_report(
    spec: ExperimentSpec,
    summary: dict[str, Any],
    ranked: list[dict[str, Any]],
) -> str:
    top_k = int(summary.get("top_k", _TOP_K_FLAG))
    top_sites_set = {r["hook_site"] for r in ranked[:top_k]}
    lines = [
        f"# Attribution Patching Report: {spec.name}",
        "",
        f"- Model: {summary['model']}",
        f"- Prompt pairs: {summary['prompt_pair_count']}",
        f"- Hook sites evaluated: {summary['hook_site_count']}",
        f"- Mean abs attribution: {summary['mean_abs_attribution']:.6f}",
        "",
        "## Attribution Scores (ranked by |attribution|)",
        "",
        "**Note:** Attribution patching is a first-order Taylor approximation "
        "of exact activation patching. Scores indicate *approximate* causal "
        "importance. Sites with large |attribution| are *candidates* for "
        "follow-up exact patching — not definitive causal evidence.",
        "",
    ]
    if not ranked:
        lines.append("No attribution results were produced.")
    else:
        lines.append("| Rank | Hook site | Attribution | |Attribution| | Follow-up? |")
        lines.append("| ---: | --- | ---: | ---: | :---: |")
        for row in ranked[:20]:
            flag = "YES" if row["hook_site"] in top_sites_set else ""
            lines.append(
                f"| {row['rank']} | `{row['hook_site']}` | "
                f"{row['attribution_score']:.6f} | {row['abs_attribution_score']:.6f} | {flag} |"
            )
    lines.extend([
        "",
        "## Follow-up Recommendation",
        "",
        f"The top-{top_k} sites by |attribution| are flagged above. "
        "Run exact `circuit_patching` on these sites to confirm causal weight. "
        "Attribution approximation quality degrades when activations are far from linear.",
        "",
    ])
    return "\n".join(lines)


def _result_notes(ranked: list[dict[str, Any]]) -> str:
    if not ranked:
        return "Attribution patching completed with no results."
    top = ranked[0]
    return (
        f"Attribution patching completed. "
        f"Top site {top['hook_site']} has mean |attribution| = "
        f"{top['abs_attribution_score']:.6f} (attribution_approximation evidence)."
    )
