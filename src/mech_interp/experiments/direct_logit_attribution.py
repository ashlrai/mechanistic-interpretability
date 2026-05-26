"""Direct Logit Attribution (DLA) experiment.

Decomposes the final logit at a chosen token position as a sum of contributions
from every model component (per-head attention output + per-layer MLP output +
embedding).  For each component c:

    DLA(c) = (cached_output_of_c[target_pos]) @ W_U @ (e_correct - e_incorrect)

where e_i is the one-hot for token i.  Positive = pushes toward correct token;
negative = pushes toward incorrect token.  Single forward pass, no ablation.

Reference: Elhage et al. (2021) §3.3; Wang et al. (2022) IOI §3.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from mech_interp.experiments.base import Experiment
from mech_interp.experiments.families import ExperimentFamily
from mech_interp.storage.artifacts import resolve_run_artifact_dir
from mech_interp.types import (
    ExperimentResult,
    ExperimentRun,
    ExperimentSpec,
    RunStatus,
)

_TOP_K_DEFAULT = 10


class DirectLogitAttributionExperiment(Experiment):
    family = ExperimentFamily.DIRECT_LOGIT_ATTRIBUTION

    def __init__(self, backend: Any = None) -> None:
        self._backend = backend

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self, spec: ExperimentSpec, run: ExperimentRun) -> ExperimentResult:
        params = _DLAParams.model_validate(spec.parameters)

        # Lazy import so the experiment works without torch in unit tests.
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "direct_logit_attribution requires torch; run `uv sync --extra interp`"
            ) from exc

        backend = self._backend or _build_backend(spec, params)

        # --- single forward pass with cache ---------------------------
        all_rows: list[dict[str, Any]] = []
        for pair in params.prompt_pairs:
            rows = _attribute_prompt(backend, pair, params, torch)
            all_rows.extend(rows)

        # --- artifacts ------------------------------------------------
        artifact_dir = resolve_run_artifact_dir(run)
        artifact_dir.mkdir(parents=True, exist_ok=True)

        ranked_json_path = (artifact_dir / "lda_ranked.json").resolve()
        ranked_csv_path = (artifact_dir / "lda_ranked.csv").resolve()
        summary_json_path = (artifact_dir / "lda_summary.json").resolve()

        ranked_json_path.write_text(
            json.dumps(all_rows, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        _write_csv(ranked_csv_path, all_rows)

        top_k = int(params.top_k)
        summary = _build_summary(all_rows, top_k=top_k)
        summary_json_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        # --- metrics --------------------------------------------------
        top_pos = summary["top_positive"][:1]
        top_neg = summary["top_negative"][:1]
        metrics: dict[str, float] = {
            "component_count": float(len({r["component_id"] for r in all_rows})),
            "prompt_count": float(len(params.prompt_pairs)),
            "top_positive_score": float(top_pos[0]["mean_score"]) if top_pos else 0.0,
            "top_negative_score": float(top_neg[0]["mean_score"]) if top_neg else 0.0,
        }

        notes = _build_notes(summary)

        return ExperimentResult(
            run_id=run.id,
            status=RunStatus.SUCCEEDED,
            metrics=metrics,
            artifacts={
                "lda_ranked_json": str(ranked_json_path),
                "lda_ranked_csv": str(ranked_csv_path),
                "lda_summary": str(summary_json_path),
            },
            notes=notes,
        )


# ------------------------------------------------------------------
# Pydantic schema
# ------------------------------------------------------------------

class _PromptPair(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str = ""
    clean_prompt: str
    correct_token: str
    incorrect_token: str
    # corrupted_prompt accepted but ignored (API consistency)
    corrupted_prompt: str = ""


class _DLAParams(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str = "gpt2-small"
    prompt_pairs: list[_PromptPair]
    target_position: int = -1  # -1 = last token
    top_k: int = Field(default=_TOP_K_DEFAULT, ge=1)
    seed: int = 42
    device: str = "cpu"

    @field_validator("prompt_pairs", mode="before")
    @classmethod
    def _require_pairs(cls, v: Any) -> Any:
        if not isinstance(v, list) or len(v) == 0:
            raise ValueError("prompt_pairs must be a non-empty list")
        return v


# ------------------------------------------------------------------
# Core math
# ------------------------------------------------------------------

def _attribute_prompt(
    backend: Any,
    pair: _PromptPair,
    params: _DLAParams,
    torch: Any,
) -> list[dict[str, Any]]:
    """Run one prompt through the model, return per-component DLA rows."""
    model = backend.model
    if model is None:
        backend.load()
        model = backend.model
    assert model is not None

    n_layers: int = model.cfg.n_layers
    n_heads: int = model.cfg.n_heads

    # Hook names for per-head attention result and per-layer MLP out.
    attn_hook = "blocks.{}.attn.hook_result"   # shape [batch, seq, n_heads, d_model]
    mlp_hook = "blocks.{}.hook_mlp_out"        # shape [batch, seq, d_model]
    embed_hook = "hook_embed"                   # shape [batch, seq, d_model]
    pos_embed_hook = "hook_pos_embed"           # shape [batch, seq, d_model]

    sites = (
        [embed_hook, pos_embed_hook]
        + [attn_hook.format(layer) for layer in range(n_layers)]
        + [mlp_hook.format(layer) for layer in range(n_layers)]
    )

    _, cache = model.run_with_cache(
        pair.clean_prompt,
        names_filter=lambda name: name in set(sites),
    )

    # W_U: [d_model, d_vocab]
    W_U = model.W_U  # noqa: N806

    correct_id = int(model.to_single_token(pair.correct_token))
    incorrect_id = int(model.to_single_token(pair.incorrect_token))

    # direction in vocab space: e_correct - e_incorrect
    direction = W_U[:, correct_id] - W_U[:, incorrect_id]  # [d_model]

    target_pos: int = params.target_position

    rows: list[dict[str, Any]] = []
    pair_id = pair.id or pair.clean_prompt[:40]

    # Embedding components
    for hook_name, ctype in ((embed_hook, "embed"), (pos_embed_hook, "pos_embed")):
        if hook_name not in cache:
            continue
        act = cache[hook_name]  # [batch, seq, d_model]
        vec = act[0, target_pos, :]  # [d_model]
        score = float((vec @ direction).item())
        rows.append(_row(pair_id, ctype, None, None, score))

    # Attention heads: hook_result shape [batch, seq, n_heads, d_model]
    for layer in range(n_layers):
        key = attn_hook.format(layer)
        if key not in cache:
            continue
        act = cache[key]  # [batch, seq, n_heads, d_model]
        for head in range(n_heads):
            vec = act[0, target_pos, head, :]  # [d_model]
            score = float((vec @ direction).item())
            rows.append(_row(pair_id, "attn_head", layer, head, score))

    # MLP layers: hook_mlp_out shape [batch, seq, d_model]
    for layer in range(n_layers):
        key = mlp_hook.format(layer)
        if key not in cache:
            continue
        act = cache[key]  # [batch, seq, d_model]
        vec = act[0, target_pos, :]  # [d_model]
        score = float((vec @ direction).item())
        rows.append(_row(pair_id, "mlp", layer, None, score))

    # Sort descending by score
    rows.sort(key=lambda r: r["score"], reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank

    return rows


def _row(
    pair_id: str,
    component_type: str,
    layer: int | None,
    head: int | None,
    score: float,
) -> dict[str, Any]:
    if component_type == "attn_head":
        assert layer is not None and head is not None
        component_id = f"L{layer}H{head}"
    elif component_type == "mlp":
        assert layer is not None
        component_id = f"L{layer}_mlp"
    else:
        component_id = component_type

    return {
        "pair_id": pair_id,
        "component_type": component_type,
        "component_id": component_id,
        "layer": layer,
        "head": head,
        "score": score,
        "evidence_label": "direct_logit_decomposition",
        "rank": 0,  # filled in after sorting
    }


# ------------------------------------------------------------------
# Summary / CSV helpers
# ------------------------------------------------------------------

def _build_summary(
    rows: list[dict[str, Any]],
    *,
    top_k: int,
) -> dict[str, Any]:
    """Aggregate per-prompt rows into mean scores per component."""
    from collections import defaultdict

    score_by_comp: dict[str, list[float]] = defaultdict(list)
    meta_by_comp: dict[str, dict[str, Any]] = {}

    for row in rows:
        cid = row["component_id"]
        score_by_comp[cid].append(row["score"])
        if cid not in meta_by_comp:
            meta_by_comp[cid] = {
                "component_id": cid,
                "component_type": row["component_type"],
                "layer": row["layer"],
                "head": row["head"],
            }

    aggregated = []
    for cid, scores in score_by_comp.items():
        mean_score = sum(scores) / len(scores)
        entry = {**meta_by_comp[cid], "mean_score": mean_score, "n_prompts": len(scores)}
        aggregated.append(entry)

    aggregated.sort(key=lambda x: x["mean_score"], reverse=True)

    top_positive = [e for e in aggregated if e["mean_score"] > 0][:top_k]
    top_negative = sorted(
        [e for e in aggregated if e["mean_score"] < 0],
        key=lambda x: x["mean_score"],
    )[:top_k]

    return {
        "top_positive": top_positive,
        "top_negative": top_negative,
        "total_components": len(aggregated),
    }


_CSV_FIELDNAMES = [
    "pair_id", "component_id", "component_type", "layer", "head", "score", "rank",
    "evidence_label",
]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text(",".join(_CSV_FIELDNAMES) + "\n", encoding="utf-8")
        return
    fieldnames = _CSV_FIELDNAMES
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _build_notes(summary: dict[str, Any]) -> str:
    top_pos = summary["top_positive"][:3]
    top_neg = summary["top_negative"][:3]
    pos_str = ", ".join(f"{e['component_id']} ({e['mean_score']:.3f})" for e in top_pos)
    neg_str = ", ".join(f"{e['component_id']} ({e['mean_score']:.3f})" for e in top_neg)
    return (
        f"DLA complete. {summary['total_components']} components scored. "
        f"Top positive: [{pos_str}]. Top negative: [{neg_str}]."
    )


# ------------------------------------------------------------------
# Backend construction
# ------------------------------------------------------------------

def _build_backend(spec: ExperimentSpec, params: _DLAParams) -> Any:
    from mech_interp.backends import create_instrumented_backend

    config: dict[str, Any] = {"model_name": params.model, "device": params.device}
    config.update(spec.parameters.get("backend_config", {}) or {})
    backend = create_instrumented_backend(spec.backend, config)
    backend.load()
    return backend
