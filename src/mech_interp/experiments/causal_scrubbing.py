"""Causal Scrubbing experiment family.

Implementation of Chan et al. "Causal Scrubbing: A method for rigorously testing
interpretability hypotheses" (Redwood Research, 2022).

A circuit hypothesis specifies which hook sites are *inside* the circuit
(``protected_sites``) and which should be resampled from a distribution of
"equivalent" inputs (``scrubbed_sites``).  For every scrubbed site, each
forward pass swaps the activation with one drawn from a prompt sharing the
same ``equivalence_label``.  If the hypothesis is correct, the scrubbed model
should behave indistinguishably from the full model — measured by
KL(p_full || p_scrubbed) near zero.

Faithfulness = exp(-mean_KL), so 1.0 is perfect and near-0 means the circuit
is insufficient to explain the behaviour.
"""

from __future__ import annotations

import json
import math
import random
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from mech_interp.backends import create_instrumented_backend
from mech_interp.experiments.base import Experiment
from mech_interp.storage.artifacts import resolve_run_artifact_dir
from mech_interp.types import (
    ExperimentResult,
    ExperimentRun,
    ExperimentSpec,
    RunStatus,
)

# ---------------------------------------------------------------------------
# Pydantic schema
# ---------------------------------------------------------------------------


class ArtifactPolicy(BaseModel):
    model_config = ConfigDict(extra="allow")

    retain_activation_tensors: bool = False
    write_report: bool = True


class ScrubPromptSpec(BaseModel):
    """A single prompt in the scrubbing dataset."""

    model_config = ConfigDict(extra="allow")

    id: str
    prompt: str
    equivalence_label: str
    target_position: int = -1

    @field_validator("id", "prompt", "equivalence_label")
    @classmethod
    def non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be empty")
        return value


class CausalScrubbingSpec(BaseModel):
    """Validated parameter block for a causal_scrubbing experiment."""

    model_config = ConfigDict(extra="allow")

    model: str = "gpt2-small"
    prompts: list[ScrubPromptSpec]
    protected_sites: list[str] = Field(default_factory=list)
    scrubbed_sites: list[str]
    seed: int = 42
    device: str = "auto"
    artifact_policy: ArtifactPolicy = Field(default_factory=ArtifactPolicy)

    @field_validator("prompts")
    @classmethod
    def at_least_two_prompts(cls, value: list[ScrubPromptSpec]) -> list[ScrubPromptSpec]:
        if len(value) < 2:
            raise ValueError("causal_scrubbing requires at least 2 prompts")
        return value

    @field_validator("scrubbed_sites")
    @classmethod
    def scrubbed_non_empty(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("scrubbed_sites must contain at least one site")
        return value


# ---------------------------------------------------------------------------
# Experiment implementation
# ---------------------------------------------------------------------------


class CausalScrubbingExperiment(Experiment):
    """Run the causal scrubbing protocol on a circuit hypothesis.

    Pipeline:
    1. Capture activations at every requested hook site for every prompt
       (one run_with_cache per prompt).
    2. Group prompts by equivalence_label.
    3. For each prompt, run forward with hooks that REPLACE activations at
       scrubbed_sites with activations from a randomly-chosen *other* prompt
       in the same equivalence class.
    4. Compute KL(p_full || p_scrubbed) over the answer-token distribution at
       target_position.
    5. Report per-prompt KL, mean/max KL, faithfulness = exp(-mean_KL).
    """

    family = "causal_scrubbing"

    def __init__(self, backend: Any | None = None) -> None:
        self._backend = backend

    def run(self, spec: ExperimentSpec, run: ExperimentRun) -> ExperimentResult:
        config = CausalScrubbingSpec.model_validate(spec.parameters)

        backend = self._backend or create_instrumented_backend(
            spec.backend,
            {
                "model_name": config.model,
                "device": config.device,
            },
        )

        # Ensure model is loaded.
        # Access `.model` directly — TransformerLensBackend exposes this attribute
        # even though the InstrumentedModelBackend Protocol does not declare it.
        _backend: Any = backend
        if not hasattr(_backend, "model") or _backend.model is None:
            _backend.load()

        rng = random.Random(config.seed)
        all_sites = list(dict.fromkeys([*config.protected_sites, *config.scrubbed_sites]))

        # Step 1 — capture activations + full-model logits for every prompt.
        activation_cache: dict[str, dict[str, Any]] = {}  # prompt_id → {site → tensor}
        full_logits_cache: dict[str, list[float]] = {}    # prompt_id → logits at target_pos

        for prompt_spec in config.prompts:
            logits, cache = _backend.model.run_with_cache(
                prompt_spec.prompt,
                names_filter=lambda name, _s=all_sites: name in _s,
            )
            activation_cache[prompt_spec.id] = {
                site: cache[site] for site in all_sites if site in cache
            }
            full_logits_cache[prompt_spec.id] = _logits_at(logits, prompt_spec.target_position)

        # Step 2 — group by equivalence label.
        label_to_ids: dict[str, list[str]] = {}
        for p in config.prompts:
            label_to_ids.setdefault(p.equivalence_label, []).append(p.id)

        try:
            _validate_equivalence_classes(config.prompts, label_to_ids)
        except ValueError as exc:
            return ExperimentResult(
                run_id=run.id,
                status=RunStatus.FAILED,
                notes=str(exc),
            )

        # Step 3 + 4 — scrub and compute KL per prompt.
        per_prompt_results: list[dict[str, Any]] = []

        for prompt_spec in config.prompts:
            peer_ids = [
                pid
                for pid in label_to_ids[prompt_spec.equivalence_label]
                if pid != prompt_spec.id
            ]
            source_id = rng.choice(peer_ids)
            source_acts = activation_cache[source_id]
            scrubbed_logits = _run_scrubbed(
                backend=_backend,
                prompt=prompt_spec.prompt,
                scrubbed_sites=config.scrubbed_sites,
                source_activations=source_acts,
            )
            scrubbed_logits_at_pos = _logits_at_raw(scrubbed_logits, prompt_spec.target_position)
            full_logits_at_pos = full_logits_cache[prompt_spec.id]
            kl = _kl_divergence(full_logits_at_pos, scrubbed_logits_at_pos)
            per_prompt_results.append(
                {
                    "prompt_id": prompt_spec.id,
                    "equivalence_label": prompt_spec.equivalence_label,
                    "scrub_source_id": source_id,
                    "kl_divergence": kl,
                }
            )

        # Step 5 — summary statistics.
        kl_values = [r["kl_divergence"] for r in per_prompt_results]
        mean_kl = sum(kl_values) / len(kl_values) if kl_values else 0.0
        max_kl = max(kl_values, default=0.0)
        faithfulness = math.exp(-mean_kl)

        equiv_class_sizes = {label: len(ids) for label, ids in label_to_ids.items()}

        # Write artifacts.
        artifact_dir = resolve_run_artifact_dir(run)
        artifact_dir.mkdir(parents=True, exist_ok=True)

        results_path = artifact_dir / "scrubbing_results.json"
        summary_path = artifact_dir / "scrubbing_summary.json"
        report_path = artifact_dir / "research_note.md"

        results_path.write_text(
            json.dumps(per_prompt_results, indent=2) + "\n",
            encoding="utf-8",
        )

        summary = {
            "model": config.model,
            "prompt_count": len(config.prompts),
            "protected_sites": config.protected_sites,
            "scrubbed_sites": config.scrubbed_sites,
            "equivalence_class_sizes": equiv_class_sizes,
            "mean_kl": mean_kl,
            "max_kl": max_kl,
            "scrubbed_faithfulness": faithfulness,
            "seed": config.seed,
        }
        summary_path.write_text(
            json.dumps(summary, indent=2) + "\n",
            encoding="utf-8",
        )

        artifacts: dict[str, str] = {
            "scrubbing_results": str(results_path.resolve()),
            "scrubbing_summary": str(summary_path.resolve()),
        }

        if config.artifact_policy.write_report:
            report_path.write_text(
                _render_report(spec, summary, per_prompt_results),
                encoding="utf-8",
            )
            artifacts["research_note"] = str(report_path.resolve())

        metrics: dict[str, float] = {
            "prompt_count": float(len(config.prompts)),
            "mean_kl": mean_kl,
            "max_kl": max_kl,
            "scrubbed_faithfulness": faithfulness,
            "protected_site_count": float(len(config.protected_sites)),
            "scrubbed_site_count": float(len(config.scrubbed_sites)),
        }

        notes = (
            f"Causal scrubbing completed. "
            f"Faithfulness={faithfulness:.4f} (mean KL={mean_kl:.4f}). "
            f"Protected sites: {len(config.protected_sites)}. "
            f"Scrubbed sites: {len(config.scrubbed_sites)}."
        )

        return ExperimentResult(
            run_id=run.id,
            status=RunStatus.SUCCEEDED,
            metrics=metrics,
            artifacts=artifacts,
            notes=notes,
        )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _run_scrubbed(
    *,
    backend: Any,
    prompt: str,
    scrubbed_sites: list[str],
    source_activations: dict[str, Any],
) -> Any:
    """Run forward pass replacing scrubbed_sites with source activations."""
    fwd_hooks = []
    for site in scrubbed_sites:
        if site not in source_activations:
            continue
        source = source_activations[site]

        def _make_hook(src: Any) -> Any:
            def hook(activation: Any, _hook: Any = None, **_kw: Any) -> Any:
                # Replace the full activation with the source — same shape required.
                replacement = src.clone().to(activation.device)
                if replacement.shape == activation.shape:
                    return replacement
                # If shapes differ (e.g. different seq length), broadcast last dim.
                out = activation.clone()
                seq = min(replacement.shape[1], activation.shape[1])
                out[:, :seq, ...] = replacement[:, :seq, ...]
                return out
            return hook

        fwd_hooks.append((site, _make_hook(source)))

    raw_backend: Any = backend
    if not fwd_hooks:
        # No sites to scrub — just run normally.
        return raw_backend.model(prompt)

    return raw_backend.model.run_with_hooks(prompt, fwd_hooks=fwd_hooks)


def _logits_at(logits: Any, position: int) -> list[float]:
    """Extract vocab logits at a sequence position from a (1, seq, vocab) tensor."""
    import torch
    with torch.no_grad():
        if logits.ndim == 3:
            vec = logits[0, position, :]
        elif logits.ndim == 2:
            vec = logits[position, :]
        else:
            vec = logits
        return vec.detach().cpu().tolist()  # type: ignore[no-any-return]


def _logits_at_raw(logits: Any, position: int) -> list[float]:
    """Same as _logits_at but handles no-grad context around raw tensor."""
    return _logits_at(logits, position)


def _kl_divergence(p_logits: list[float], q_logits: list[float]) -> float:
    """KL(P || Q) where P and Q are given as logits (unnormalized log probs).

    Uses the log-sum-exp trick for numerical stability.
    KL(P||Q) = Σ p_i * (log p_i - log q_i) = Σ p_i * (log_p_i - log_q_i)
    """
    import math

    def log_softmax(logits: list[float]) -> list[float]:
        m = max(logits)
        exps = [math.exp(x - m) for x in logits]
        z = sum(exps)
        return [math.log(e / z) for e in exps]

    log_p = log_softmax(p_logits)
    log_q = log_softmax(q_logits)

    kl = 0.0
    for lp, lq in zip(log_p, log_q, strict=True):
        p = math.exp(lp)
        if p > 0:
            kl += p * (lp - lq)
    return max(0.0, kl)  # clamp numerical negatives


def _validate_equivalence_classes(
    prompts: list[ScrubPromptSpec],
    label_to_ids: dict[str, list[str]],
) -> None:
    """Raise if any prompt has no peer in its equivalence class."""
    singletons = [
        label
        for label, ids in label_to_ids.items()
        if len(ids) < 2
    ]
    if singletons:
        labels_str = ", ".join(repr(s) for s in singletons[:5])
        raise ValueError(
            f"Causal scrubbing requires each equivalence class to have ≥2 prompts. "
            f"Singleton classes: {labels_str}."
        )


def _render_report(
    spec: ExperimentSpec,
    summary: dict[str, Any],
    per_prompt: list[dict[str, Any]],
) -> str:
    faithfulness = summary["scrubbed_faithfulness"]
    mean_kl = summary["mean_kl"]
    max_kl = summary["max_kl"]
    if faithfulness >= 0.8:
        verdict = "SUPPORTED"
    elif faithfulness >= 0.5:
        verdict = "PARTIAL"
    else:
        verdict = "REJECTED"
    lines = [
        f"# Causal Scrubbing Report: {spec.name}",
        "",
        f"**Hypothesis:** {spec.description or 'See spec parameters.'}",
        "",
        "## Circuit Specification",
        "",
        f"- Model: `{summary['model']}`",
        f"- Protected sites (inside circuit): {summary['protected_sites']}",
        f"- Scrubbed sites (resampled): {summary['scrubbed_sites']}",
        f"- Prompts: {summary['prompt_count']}",
        f"- Equivalence classes: {list(summary['equivalence_class_sizes'].items())}",
        "",
        "## Results",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Mean KL(full \\|\\| scrubbed) | {mean_kl:.6f} |",
        f"| Max KL | {max_kl:.6f} |",
        f"| Scrubbed faithfulness = exp(-mean KL) | **{faithfulness:.4f}** |",
        f"| Verdict | **{verdict}** |",
        "",
        "## Interpretation",
        "",
        "- Faithfulness ≥ 0.8 → hypothesis strongly supported: the circuit explains the behaviour.",
        "- Faithfulness 0.5–0.8 → partial support: circuit captures some but not all computation.",
        "- Faithfulness < 0.5 → hypothesis rejected: protected sites are insufficient.",
        "",
        "## Per-Prompt KL",
        "",
        "| Prompt ID | Equiv. Label | Scrub Source | KL |",
        "| --- | --- | --- | ---: |",
    ]
    for row in per_prompt:
        lines.append(
            f"| `{row['prompt_id']}` | `{row['equivalence_label']}` "
            f"| `{row['scrub_source_id']}` | {row['kl_divergence']:.6f} |"
        )
    lines.append("")
    return "\n".join(lines)
