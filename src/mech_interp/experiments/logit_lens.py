"""Logit Lens and Tuned Lens experiment.

Projects the residual stream at every layer through the unembedding matrix to see
"what the model would predict if it stopped here."

Logit lens (nostalgebraist, 2020):
    logits_L = ln_final(resid_post_L) @ W_U

Tuned lens (Belrose et al., 2023) replaces the raw unembed with a per-layer
learned affine transform applied before unembedding:
    logits_L = ln_final(A_L @ resid_post_L + b_L) @ W_U

Reference artifacts:
  - lens_results.json  — per-prompt per-layer details
  - lens_summary.json  — layer-by-layer mean rank and CE-loss curves
  - research_note.md   — ASCII chart of rank-of-correct across layers
"""

from __future__ import annotations

import json
import math
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

_TOP_K_DEFAULT = 5


class LogitLensExperiment(Experiment):
    family = ExperimentFamily.LOGIT_LENS

    def __init__(self, backend: Any = None) -> None:
        self._backend = backend

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self, spec: ExperimentSpec, run: ExperimentRun) -> ExperimentResult:
        params = _LensParams.model_validate(spec.parameters)

        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "logit_lens requires torch; run `uv sync --extra interp`"
            ) from exc

        backend = self._backend or _build_backend(spec, params)

        model = backend.model
        if model is None:
            backend.load()
            model = backend.model
        assert model is not None

        # Load tuned-lens transforms if requested
        tuned_transforms: dict[int, dict[str, Any]] | None = None
        if params.mode == "tuned" and params.tuned_lens_path:
            tuned_transforms = _load_tuned_lens(params.tuned_lens_path, torch)

        # Run lens for all prompts
        all_prompt_results: list[dict[str, Any]] = []
        for prompt_cfg in params.prompts:
            per_layer = _run_lens_for_prompt(
                model=model,
                prompt_cfg=prompt_cfg,
                params=params,
                torch=torch,
                tuned_transforms=tuned_transforms,
            )
            all_prompt_results.append(
                {
                    "id": prompt_cfg.id,
                    "prompt": prompt_cfg.prompt,
                    "correct_token": prompt_cfg.correct_token,
                    "incorrect_token": prompt_cfg.incorrect_token,
                    "layers": per_layer,
                }
            )

        # Build summary: per-layer mean rank-of-correct + mean CE-loss
        summary = _build_summary(all_prompt_results)

        # Write artifacts
        artifact_dir = resolve_run_artifact_dir(run)
        artifact_dir.mkdir(parents=True, exist_ok=True)

        lens_results_path = (artifact_dir / "lens_results.json").resolve()
        lens_summary_path = (artifact_dir / "lens_summary.json").resolve()
        research_note_path = (artifact_dir / "research_note.md").resolve()

        lens_results_path.write_text(
            json.dumps(all_prompt_results, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        lens_summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        research_note_path.write_text(
            _build_research_note(all_prompt_results, summary, params),
            encoding="utf-8",
        )

        # Scalar metrics
        n_layers = len(summary["mean_rank_by_layer"])
        final_mean_rank = summary["mean_rank_by_layer"][-1] if n_layers else 0.0
        final_mean_ce = summary["mean_ce_by_layer"][-1] if n_layers else 0.0
        first_top_k_layer = _first_top_k_layer(summary["mean_rank_by_layer"], params.top_k)

        metrics: dict[str, float] = {
            "n_layers": float(n_layers),
            "n_prompts": float(len(params.prompts)),
            "final_mean_rank": float(final_mean_rank),
            "final_mean_ce_loss": float(final_mean_ce),
            "first_top_k_layer": float(first_top_k_layer),
        }

        notes = _build_notes(summary, params)

        return ExperimentResult(
            run_id=run.id,
            status=RunStatus.SUCCEEDED,
            metrics=metrics,
            artifacts={
                "lens_results": str(lens_results_path),
                "lens_summary": str(lens_summary_path),
                "research_note": str(research_note_path),
            },
            notes=notes,
        )


# ------------------------------------------------------------------
# Pydantic schema
# ------------------------------------------------------------------

class _PromptConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str = ""
    prompt: str
    correct_token: str
    incorrect_token: str = ""

    @field_validator("id", mode="before")
    @classmethod
    def _default_id(cls, v: Any) -> Any:
        return v if v else ""


class _LensParams(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str = "gpt2-small"
    prompts: list[_PromptConfig]
    target_position: int = -1
    top_k: int = Field(default=_TOP_K_DEFAULT, ge=1)
    mode: str = "logit"          # "logit" | "tuned"
    tuned_lens_path: str | None = None
    seed: int = 42
    device: str = "cpu"

    @field_validator("prompts", mode="before")
    @classmethod
    def _require_prompts(cls, v: Any) -> Any:
        if not isinstance(v, list) or len(v) == 0:
            raise ValueError("prompts must be a non-empty list")
        return v

    @field_validator("mode", mode="before")
    @classmethod
    def _validate_mode(cls, v: Any) -> Any:
        if str(v) not in {"logit", "tuned"}:
            raise ValueError("mode must be 'logit' or 'tuned'")
        return str(v)


# ------------------------------------------------------------------
# Core math
# ------------------------------------------------------------------

def _run_lens_for_prompt(
    *,
    model: Any,
    prompt_cfg: _PromptConfig,
    params: _LensParams,
    torch: Any,
    tuned_transforms: dict[int, dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Run logit lens (or tuned lens) on a single prompt; return per-layer dicts."""
    n_layers: int = model.cfg.n_layers

    # Collect hook names for every layer's resid_post
    hook_names = [f"blocks.{L}.hook_resid_post" for L in range(n_layers)]
    hook_set = set(hook_names)

    _, cache = model.run_with_cache(
        prompt_cfg.prompt,
        names_filter=lambda name: name in hook_set,
    )

    W_U: Any = model.W_U  # [d_model, d_vocab]  # noqa: N806

    correct_id = int(model.to_single_token(prompt_cfg.correct_token))
    has_incorrect = bool(prompt_cfg.incorrect_token)
    incorrect_id = int(model.to_single_token(prompt_cfg.incorrect_token)) if has_incorrect else -1

    target_pos: int = params.target_position

    per_layer_results: list[dict[str, Any]] = []
    for L in range(n_layers):
        hook_name = f"blocks.{L}.hook_resid_post"
        if hook_name not in cache:
            continue
        resid: Any = cache[hook_name]  # [batch, seq, d_model]
        vec: Any = resid[0, target_pos, :]   # [d_model]

        # Apply tuned-lens affine transform if in tuned mode
        if params.mode == "tuned" and tuned_transforms and L in tuned_transforms:
            W: Any = tuned_transforms[L]["weight"]  # noqa: N806
            b: Any = tuned_transforms[L]["bias"]
            vec = vec @ W.T + b  # noqa: N806

        # Apply final layer norm then project through W_U
        normed: Any = model.ln_final(vec.unsqueeze(0).unsqueeze(0))[0, 0, :]  # [d_model]
        logits_L: Any = normed @ W_U  # [d_vocab]

        # Softmax probabilities for CE loss computation
        log_probs: Any = torch.log_softmax(logits_L, dim=-1)
        ce_loss: float = float(-log_probs[correct_id].item())

        # Rank of correct token (1 = top)
        sorted_ids: Any = torch.argsort(logits_L, descending=True)
        rank_correct: int = int((sorted_ids == correct_id).nonzero(as_tuple=True)[0].item()) + 1

        # Top-K predictions
        top_k_ids = sorted_ids[: params.top_k].tolist()
        try:
            top_k_tokens = [model.tokenizer.decode([tid]).strip() for tid in top_k_ids]
        except Exception:
            top_k_tokens = [str(tid) for tid in top_k_ids]
        top_k_logits = logits_L[top_k_ids].tolist()

        layer_record: dict[str, Any] = {
            "layer": L,
            "rank_correct": rank_correct,
            "ce_loss": ce_loss,
            "top_k": [
                {"token": tok, "logit": float(lg)}
                for tok, lg in zip(top_k_tokens, top_k_logits, strict=True)
            ],
        }

        if has_incorrect:
            rank_incorrect: int = (
                int((sorted_ids == incorrect_id).nonzero(as_tuple=True)[0].item()) + 1
            )
            layer_record["rank_incorrect"] = rank_incorrect

        per_layer_results.append(layer_record)

    return per_layer_results


def _build_summary(all_prompt_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate across prompts: mean rank-of-correct and mean CE-loss per layer."""
    if not all_prompt_results:
        return {"mean_rank_by_layer": [], "mean_ce_by_layer": [], "n_layers": 0}

    # Collect per-layer lists across prompts
    n_layers = max(len(p["layers"]) for p in all_prompt_results)
    rank_by_layer: list[list[float]] = [[] for _ in range(n_layers)]
    ce_by_layer: list[list[float]] = [[] for _ in range(n_layers)]

    for prompt_result in all_prompt_results:
        for layer_record in prompt_result["layers"]:
            L: int = layer_record["layer"]
            rank_by_layer[L].append(float(layer_record["rank_correct"]))
            ce_by_layer[L].append(float(layer_record["ce_loss"]))

    mean_rank = [
        float(sum(ranks) / len(ranks)) if ranks else float("nan")
        for ranks in rank_by_layer
    ]
    mean_ce = [
        float(sum(ces) / len(ces)) if ces else float("nan")
        for ces in ce_by_layer
    ]

    return {
        "n_layers": n_layers,
        "mean_rank_by_layer": mean_rank,
        "mean_ce_by_layer": mean_ce,
    }


def _first_top_k_layer(mean_rank_by_layer: list[float], top_k: int) -> int:
    """Return first layer where mean rank-of-correct <= top_k, else n_layers."""
    for L, rank in enumerate(mean_rank_by_layer):
        if not math.isnan(rank) and rank <= top_k:
            return L
    return len(mean_rank_by_layer)


# ------------------------------------------------------------------
# ASCII chart + research note
# ------------------------------------------------------------------

_SPARKLINE_CHARS = " ▁▂▃▄▅▆▇█"


def _sparkline_from_values(values: list[float], high_is_bad: bool = True) -> str:
    """Build a single-line sparkline string from a list of floats."""
    if not values:
        return ""
    finite = [v for v in values if not math.isnan(v)]
    if not finite:
        return "?" * len(values)
    vmin, vmax = min(finite), max(finite)
    span = vmax - vmin if vmax > vmin else 1.0
    chars: list[str] = []
    for v in values:
        if math.isnan(v):
            chars.append("?")
            continue
        # Normalise 0..1; if high_is_bad, high values → high char index (bad)
        norm = (v - vmin) / span
        if not high_is_bad:
            norm = 1.0 - norm
        idx = min(int(norm * (len(_SPARKLINE_CHARS) - 1)), len(_SPARKLINE_CHARS) - 1)
        chars.append(_SPARKLINE_CHARS[idx])
    return "".join(chars)


def _ascii_rank_chart(mean_rank: list[float], top_k: int) -> str:
    """Return a multi-line ASCII chart showing rank-of-correct per layer."""
    lines: list[str] = []
    lines.append("Layer-by-layer rank-of-correct (logit lens):")
    lines.append(f"  top_k={top_k}  (first col=L0, last=final layer)")
    lines.append("")

    # Sparkline row (inverted: low rank = high char = good)
    spark = _sparkline_from_values(mean_rank, high_is_bad=True)
    lines.append(f"  rank:  {spark}")
    lines.append("")
    lines.append("  Layer | Mean rank")
    lines.append("  ------+----------")
    for L, rank in enumerate(mean_rank):
        in_top_k = "  <-- FIRST top-K" if int(round(rank)) <= top_k else ""
        lines.append(f"  {L:5d} | {rank:9.1f}{in_top_k}")
    return "\n".join(lines)


def _build_research_note(
    all_prompt_results: list[dict[str, Any]],
    summary: dict[str, Any],
    params: _LensParams,
) -> str:
    mean_rank: list[float] = summary["mean_rank_by_layer"]
    mean_ce: list[float] = summary["mean_ce_by_layer"]
    first_topk = _first_top_k_layer(mean_rank, params.top_k)
    n_layers = summary["n_layers"]

    lines: list[str] = [
        "# Logit Lens Research Note",
        "",
        f"- Model: `{params.model}`",
        f"- Mode: `{params.mode}`",
        f"- Prompts: {len(all_prompt_results)}",
        f"- top_k: {params.top_k}",
        "",
        "## Layer-by-layer rank-of-correct (mean across prompts)",
        "",
        _ascii_rank_chart(mean_rank, params.top_k),
        "",
        "## CE-loss sparkline (lower = better)",
        "",
        f"  ce:    {_sparkline_from_values(mean_ce, high_is_bad=True)}",
        "",
        "## Key finding",
        "",
    ]

    if first_topk < n_layers:
        lines.append(
            f"Correct token first enters mean top-{params.top_k} at **layer {first_topk}** "
            f"(out of {n_layers} layers)."
        )
    else:
        lines.append(
            f"Correct token never reached mean top-{params.top_k} before the final layer."
        )

    lines.append("")
    lines.append("## Per-prompt summary")
    lines.append("")

    for pr in all_prompt_results:
        prompt_id = pr.get("id") or pr.get("prompt", "")[:40]
        layer_records: list[dict[str, Any]] = pr["layers"]
        if not layer_records:
            continue
        final = layer_records[-1]
        first_topk_prompt = next(
            (
                rec["layer"]
                for rec in layer_records
                if rec["rank_correct"] <= params.top_k
            ),
            None,
        )
        topk_note = (
            f"enters top-{params.top_k} at L{first_topk_prompt}"
            if first_topk_prompt is not None
            else f"never in top-{params.top_k}"
        )
        lines.append(
            f"- **{prompt_id}**: final rank={final['rank_correct']}, "
            f"final CE={final['ce_loss']:.3f}, {topk_note}"
        )

    lines.append("")
    return "\n".join(lines)


def _build_notes(summary: dict[str, Any], params: _LensParams) -> str:
    mean_rank = summary["mean_rank_by_layer"]
    n = len(mean_rank)
    if n == 0:
        return "No layers processed."
    final_rank = mean_rank[-1]
    first_topk = _first_top_k_layer(mean_rank, params.top_k)
    topk_note = (
        f"enters top-{params.top_k} at L{first_topk}"
        if first_topk < n
        else f"never in top-{params.top_k}"
    )
    return (
        f"Logit lens complete. {n} layers, {len(params.prompts)} prompt(s). "
        f"Final mean rank={final_rank:.1f}. Correct token {topk_note}."
    )


# ------------------------------------------------------------------
# Tuned lens loader
# ------------------------------------------------------------------

def _load_tuned_lens(path: str, torch: Any) -> dict[int, dict[str, Any]]:
    """Load per-layer affine transforms from a safetensors file.

    Expected keys: ``layer_{L}.weight`` [d_model, d_model] and
    ``layer_{L}.bias`` [d_model].
    """
    from safetensors.torch import load_file

    tensors = load_file(path)
    transforms: dict[int, dict[str, Any]] = {}
    for key, tensor in tensors.items():
        if key.startswith("layer_") and key.endswith(".weight"):
            layer_idx = int(key[len("layer_") : -len(".weight")])
            transforms.setdefault(layer_idx, {})["weight"] = tensor
        elif key.startswith("layer_") and key.endswith(".bias"):
            layer_idx = int(key[len("layer_") : -len(".bias")])
            transforms.setdefault(layer_idx, {})["bias"] = tensor
    return transforms


# ------------------------------------------------------------------
# Backend construction
# ------------------------------------------------------------------

def _build_backend(spec: ExperimentSpec, params: _LensParams) -> Any:
    from mech_interp.backends import create_instrumented_backend

    config: dict[str, Any] = {"model_name": params.model, "device": params.device}
    config.update(spec.parameters.get("backend_config", {}) or {})
    backend = create_instrumented_backend(spec.backend, config)
    backend.load()
    return backend
