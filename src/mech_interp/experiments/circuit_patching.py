from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from mech_interp.backends import create_instrumented_backend
from mech_interp.datasets import PromptDataset, load_prompt_dataset
from mech_interp.experiments.base import Experiment
from mech_interp.orchestration.resource_policy import ActivationEstimate, ResourcePolicy
from mech_interp.types import (
    ActivationPatchPromptPair,
    ActivationPatchRequest,
    ActivationPatchSiteResult,
    ExperimentResult,
    ExperimentRun,
    ExperimentSpec,
    InstrumentedModelBackend,
    RunStatus,
)

CAUSAL_EVIDENCE_LABEL = "causal evidence"
CONTROL_EVIDENCE_LABEL = "control"


class PromptPairSpec(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str | None = None
    clean_prompt: str
    corrupted_prompt: str
    correct_token: str | None = None
    incorrect_token: str | None = None
    target_position: int | None = None
    patch_position: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("clean_prompt", "corrupted_prompt", "correct_token", "incorrect_token")
    @classmethod
    def strip_non_empty(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not value.strip():
            raise ValueError("must not be empty")
        return value


class AnswerTokenSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    correct: str
    incorrect: str

    @field_validator("correct", "incorrect")
    @classmethod
    def strip_non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("answer token must not be empty")
        return value


class ArtifactPolicy(BaseModel):
    model_config = ConfigDict(extra="allow")

    retain_activation_tensors: bool = False
    write_report: bool = True


class CircuitPatchingSpec(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str = "gpt2-small"
    model_name: str | None = None
    prompt_pairs: list[PromptPairSpec] | None = None
    source_prompt: str | None = None
    target_prompt: str | None = None
    clean_prompt: str | None = None
    corrupted_prompt: str | None = None
    dataset_path: str | None = None
    dataset_sha256: str | None = None
    answer_tokens: AnswerTokenSpec | None = None
    hook_sites: list[str] | None = None
    sites: list[str] | None = None
    patch_sites: list[str] | None = None
    control_hook_sites: list[str] | None = None
    control_sites: list[str] | None = None
    control_patch_sites: list[str] | None = None
    layers: list[int] | None = None
    target_position: int = -1
    patch_position: int = -1
    sequence_length: int | None = None
    hidden_size: int = 768
    dtype: str = "float32"
    resource_policy: dict[str, Any] = Field(default_factory=dict)
    artifact_policy: ArtifactPolicy = Field(default_factory=ArtifactPolicy)

    @property
    def resolved_model_name(self) -> str:
        return self.model_name or self.model


class CircuitPatchingExperiment(Experiment):
    family = "circuit_patching"

    def __init__(self, backend: InstrumentedModelBackend | None = None) -> None:
        self.backend = backend

    def run(self, spec: ExperimentSpec, run: ExperimentRun) -> ExperimentResult:
        config = CircuitPatchingSpec.model_validate(spec.parameters)
        prompt_pairs, dataset_metadata = _resolve_prompt_pairs(config)
        hook_sites = _resolve_hook_sites(config)
        raw_control_hook_sites = _resolve_control_hook_sites(config)
        experimental_hook_site_set = set(hook_sites)
        control_hook_sites = [
            site for site in raw_control_hook_sites
            if site not in experimental_hook_site_set
        ]
        ignored_control_hook_sites = [
            site for site in raw_control_hook_sites
            if site in experimental_hook_site_set
        ]
        requested_hook_sites = _dedupe_hook_sites([*hook_sites, *control_hook_sites])
        _validate_resource_policy(
            config,
            prompt_pairs=prompt_pairs,
            hook_sites=requested_hook_sites,
        )

        backend = self.backend or create_instrumented_backend(
            spec.backend,
            {
                "model_name": config.resolved_model_name,
                "device": spec.parameters.get("device", "auto"),
            },
        )
        request = ActivationPatchRequest(
            model_name=config.resolved_model_name,
            prompt_pairs=tuple(prompt_pairs),
            hook_sites=tuple(requested_hook_sites),
            dtype=config.dtype,
            retain_activation_tensors=config.artifact_policy.retain_activation_tensors,
        )
        raw_results = backend.run_activation_patching(request)
        ranked_results = sorted(raw_results, key=lambda item: item.recovery_fraction, reverse=True)
        missing_sites = _missing_sites(prompt_pairs, requested_hook_sites, raw_results)
        missing_experimental_sites = _missing_sites(prompt_pairs, hook_sites, raw_results)
        missing_control_sites = _missing_sites(prompt_pairs, control_hook_sites, raw_results)

        artifact_dir = _run_artifact_dir(run)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        ranked_json = artifact_dir / "patching_ranked_results.json"
        ranked_csv = artifact_dir / "patching_ranked_results.csv"
        summary_path = artifact_dir / "patching_summary.json"
        report_path = artifact_dir / "research_note.md"

        control_hook_site_set = set(control_hook_sites)
        result_rows = [
            {
                "rank": rank,
                **_result_row(result, control_hook_sites=control_hook_site_set),
            }
            for rank, result in enumerate(ranked_results, start=1)
        ]
        causal_rows = [
            row for row in result_rows
            if row["evidence_label"] == CAUSAL_EVIDENCE_LABEL
        ]
        control_rows = [
            row for row in result_rows
            if row["evidence_label"] == CONTROL_EVIDENCE_LABEL
        ]
        ranked_json.write_text(
            json.dumps(result_rows, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _write_ranked_csv(ranked_csv, result_rows)

        summary = {
            "model": config.resolved_model_name,
            "prompt_pair_count": len(prompt_pairs),
            "hook_site_count": len(hook_sites),
            "requested_hook_sites": requested_hook_sites,
            "experimental_hook_sites": hook_sites,
            "control_hook_sites": control_hook_sites,
            "ignored_control_hook_sites": ignored_control_hook_sites,
            "control_site_count": len(control_hook_sites),
            "result_count": len(raw_results),
            "missing_sites": missing_sites,
            "missing_experimental_sites": missing_experimental_sites,
            "missing_control_sites": missing_control_sites,
            "top_results": result_rows[:10],
            "top_causal_results": causal_rows[:10],
            "control_results": control_rows[:10],
            "control_summary": _control_summary(control_rows, missing_control_sites),
            "dataset": dataset_metadata,
            "artifact_policy": config.artifact_policy.model_dump(),
        }
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if config.artifact_policy.write_report:
            report_path.write_text(_render_report(spec, summary), encoding="utf-8")

        metrics = _metrics(
            prompt_pairs,
            requested_hook_sites,
            ranked_results,
            missing_sites,
            control_hook_sites=control_hook_sites,
        )
        artifacts = {
            "patching_summary": str(summary_path.resolve()),
            "patching_ranked_json": str(ranked_json.resolve()),
            "patching_ranked_csv": str(ranked_csv.resolve()),
        }
        if config.artifact_policy.write_report:
            artifacts["research_note"] = str(report_path.resolve())

        return ExperimentResult(
            run_id=run.id,
            status=RunStatus.SUCCEEDED,
            metrics=metrics,
            artifacts=artifacts,
            notes=_result_notes(ranked_results, missing_sites, control_hook_sites),
        )


def _resolve_prompt_pairs(
    config: CircuitPatchingSpec,
) -> tuple[list[ActivationPatchPromptPair], dict[str, Any] | None]:
    if config.prompt_pairs:
        pairs = [
            _pair_from_spec(index, pair, config)
            for index, pair in enumerate(config.prompt_pairs)
        ]
        return pairs, None

    if config.dataset_path:
        dataset = load_prompt_dataset(config.dataset_path)
        if config.dataset_sha256 and dataset.sha256 != config.dataset_sha256:
            raise ValueError(
                "Circuit patching dataset hash mismatch: "
                f"expected {config.dataset_sha256}, got {dataset.sha256}."
            )
        return _pairs_from_dataset(dataset, config), {
            "path": str(dataset.source_path) if dataset.source_path is not None else None,
            "name": dataset.name,
            "sha256": dataset.sha256,
            "record_hashes": dataset.record_hashes(),
        }

    clean_prompt = config.clean_prompt or config.source_prompt
    corrupted_prompt = config.corrupted_prompt or config.target_prompt
    if clean_prompt and corrupted_prompt:
        pair = PromptPairSpec(
            id="pair-0001",
            clean_prompt=clean_prompt,
            corrupted_prompt=corrupted_prompt,
        )
        return [_pair_from_spec(0, pair, config)], None

    raise ValueError(
        "Circuit patching requires prompt_pairs, dataset_path, or clean/corrupted prompts."
    )


def _pair_from_spec(
    index: int,
    pair: PromptPairSpec,
    config: CircuitPatchingSpec,
) -> ActivationPatchPromptPair:
    correct_token = pair.correct_token or (
        config.answer_tokens.correct if config.answer_tokens is not None else None
    )
    incorrect_token = pair.incorrect_token or (
        config.answer_tokens.incorrect if config.answer_tokens is not None else None
    )
    if correct_token is None or incorrect_token is None:
        raise ValueError(
            "Circuit patching prompt pairs require correct and incorrect answer tokens."
        )
    return ActivationPatchPromptPair(
        id=pair.id or f"pair-{index + 1:04d}",
        clean_prompt=pair.clean_prompt,
        corrupted_prompt=pair.corrupted_prompt,
        correct_token=correct_token,
        incorrect_token=incorrect_token,
        target_position=pair.target_position
        if pair.target_position is not None
        else config.target_position,
        patch_position=(
            pair.patch_position if pair.patch_position is not None else config.patch_position
        ),
        metadata=dict(pair.metadata),
    )


def _pairs_from_dataset(
    dataset: PromptDataset,
    config: CircuitPatchingSpec,
) -> list[ActivationPatchPromptPair]:
    grouped: dict[str, dict[str, Any]] = defaultdict(dict)
    for record in dataset.records:
        pair_id = record.metadata.get("pair_id")
        kind = record.metadata.get("kind")
        if not isinstance(pair_id, str) or kind not in {"clean", "corrupted"}:
            continue
        grouped[pair_id][str(kind)] = record

    pairs: list[ActivationPatchPromptPair] = []
    for pair_id in sorted(grouped):
        clean = grouped[pair_id].get("clean")
        corrupted = grouped[pair_id].get("corrupted")
        if clean is None or corrupted is None:
            continue
        correct = clean.metadata.get("correct_token") or clean.metadata.get("answer")
        incorrect = corrupted.metadata.get("incorrect_token") or corrupted.metadata.get("answer")
        pair_spec = PromptPairSpec(
            id=pair_id,
            clean_prompt=clean.prompt,
            corrupted_prompt=corrupted.prompt,
            correct_token=str(correct) if correct is not None else None,
            incorrect_token=str(incorrect) if incorrect is not None else None,
            metadata={
                "clean_record_id": clean.id,
                "corrupted_record_id": corrupted.id,
            },
        )
        pairs.append(_pair_from_spec(len(pairs), pair_spec, config))

    if not pairs:
        raise ValueError(
            "Circuit patching dataset did not contain clean/corrupted records grouped by pair_id."
        )
    return pairs


def _resolve_hook_sites(config: CircuitPatchingSpec) -> list[str]:
    explicit_sites = config.hook_sites or config.sites
    if explicit_sites:
        return _non_empty_strings(explicit_sites, "hook_sites")

    patch_sites = _non_empty_strings(config.patch_sites or [], "patch_sites")
    if not patch_sites:
        raise ValueError("Circuit patching requires hook_sites, sites, or patch_sites.")
    layers = config.layers or [0]
    resolved: list[str] = []
    for layer in layers:
        if layer < 0:
            raise ValueError("Circuit patching layers must be non-negative integers.")
        for site in patch_sites:
            resolved.append(_expand_site(layer, site))
    return resolved


def _resolve_control_hook_sites(config: CircuitPatchingSpec) -> list[str]:
    explicit_sites = config.control_hook_sites or config.control_sites
    if explicit_sites:
        return _non_empty_strings(explicit_sites, "control_hook_sites")

    control_patch_sites = _non_empty_strings(
        config.control_patch_sites or [],
        "control_patch_sites",
    )
    if not control_patch_sites:
        return []

    layers = config.layers or [0]
    resolved: list[str] = []
    for layer in layers:
        if layer < 0:
            raise ValueError("Circuit patching layers must be non-negative integers.")
        for site in control_patch_sites:
            resolved.append(_expand_site(layer, site))
    return resolved


def _dedupe_hook_sites(hook_sites: list[str]) -> list[str]:
    return list(dict.fromkeys(hook_sites))


def _expand_site(layer: int, site: str) -> str:
    aliases = {
        "resid_pre": f"blocks.{layer}.hook_resid_pre",
        "resid_post": f"blocks.{layer}.hook_resid_post",
        "mlp_post": f"blocks.{layer}.mlp.hook_post",
        "attn_out": f"blocks.{layer}.attn.hook_result",
    }
    return aliases.get(site, site)


def _non_empty_strings(values: list[str], name: str) -> list[str]:
    cleaned = [value.strip() for value in values if isinstance(value, str) and value.strip()]
    if len(cleaned) != len(values):
        raise ValueError(
            f"Circuit patching parameter '{name}' must contain only non-empty strings."
        )
    return cleaned


def _validate_resource_policy(
    config: CircuitPatchingSpec,
    *,
    prompt_pairs: list[ActivationPatchPromptPair],
    hook_sites: list[str],
) -> None:
    sequence_length = config.sequence_length
    if sequence_length is None:
        sequence_length = max(
            len(pair.clean_prompt.split()) + 1
            for pair in prompt_pairs
        )
    estimate = ActivationEstimate(
        batch_size=len(prompt_pairs) * 2,
        sequence_length=sequence_length,
        hidden_size=config.hidden_size,
        hook_count=len(hook_sites),
        dtype=config.dtype,
    )
    policy = ResourcePolicy(**config.resource_policy)
    policy.validate_activation_estimate(estimate)


def _missing_sites(
    pairs: list[ActivationPatchPromptPair],
    hook_sites: list[str],
    results: list[ActivationPatchSiteResult],
) -> list[dict[str, str]]:
    seen = {(result.pair_id, result.hook_site) for result in results}
    return [
        {"pair_id": pair.id, "hook_site": hook_site}
        for pair in pairs
        for hook_site in hook_sites
        if (pair.id, hook_site) not in seen
    ]


def _metrics(
    prompt_pairs: list[ActivationPatchPromptPair],
    hook_sites: list[str],
    results: list[ActivationPatchSiteResult],
    missing_sites: list[dict[str, str]],
    *,
    control_hook_sites: list[str],
) -> dict[str, float]:
    recoveries = [result.recovery_fraction for result in results]
    positive = [value for value in recoveries if value > 0]
    control_site_set = set(control_hook_sites)
    control_results = [
        result for result in results
        if result.hook_site in control_site_set
    ]
    return {
        "prompt_pair_count": float(len(prompt_pairs)),
        "requested_site_count": float(len(hook_sites)),
        "control_site_count": float(len(control_hook_sites)),
        "patch_result_count": float(len(results)),
        "missing_site_count": float(len(missing_sites)),
        "top_recovery_fraction": max(recoveries, default=0.0),
        "mean_recovery_fraction": sum(recoveries) / len(recoveries) if recoveries else 0.0,
        "positive_recovery_fraction": len(positive) / len(recoveries) if recoveries else 0.0,
        "control_result_count": float(len(control_results)),
        "top_control_recovery_fraction": max(
            (result.recovery_fraction for result in control_results),
            default=0.0,
        ),
    }


def _result_row(
    result: ActivationPatchSiteResult,
    *,
    control_hook_sites: set[str],
) -> dict[str, Any]:
    evidence_label = (
        CONTROL_EVIDENCE_LABEL
        if result.hook_site in control_hook_sites
        else CAUSAL_EVIDENCE_LABEL
    )
    return {**asdict(result), "evidence_label": evidence_label}


def _control_summary(
    control_rows: list[dict[str, Any]],
    missing_control_sites: list[dict[str, str]],
) -> dict[str, Any]:
    recoveries = [float(row["recovery_fraction"]) for row in control_rows]
    return {
        "evidence_label": CONTROL_EVIDENCE_LABEL,
        "configured": bool(control_rows or missing_control_sites),
        "result_count": len(control_rows),
        "missing_result_count": len(missing_control_sites),
        "max_recovery_fraction": max(recoveries, default=0.0),
        "mean_recovery_fraction": (
            sum(recoveries) / len(recoveries) if recoveries else 0.0
        ),
    }


def _write_ranked_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "rank",
        "pair_id",
        "hook_site",
        "clean_logit_diff",
        "corrupted_logit_diff",
        "patched_logit_diff",
        "recovery_fraction",
        "activation_norm",
        "evidence_label",
    ]
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _render_report(spec: ExperimentSpec, summary: dict[str, Any]) -> str:
    top_results = summary.get("top_causal_results") or summary["top_results"]
    control_results = summary.get("control_results", [])
    control_summary = summary.get("control_summary", {})
    lines = [
        f"# Circuit Patching Report: {spec.name}",
        "",
        f"- Model: {summary['model']}",
        f"- Prompt pairs: {summary['prompt_pair_count']}",
        f"- Experimental hook sites: {summary['hook_site_count']}",
        f"- Control hook sites: {summary.get('control_site_count', 0)}",
        f"- Patch results: {summary['result_count']}",
        f"- Missing pair/site results: {len(summary['missing_sites'])}",
        "- Evidence labels: causal evidence for configured experimental interventions; "
        "control for configured control hook sites.",
        "- Interpretation: patching is intervention evidence, not proof of a complete "
        "circuit. Compare against controls before promoting a site.",
        "",
        "## Top Patch Sites",
        "",
    ]
    if not top_results:
        lines.append("No patch results were produced.")
    else:
        lines.append("| Rank | Pair | Hook site | Recovery | Patched logit diff |")
        lines.append("| ---: | --- | --- | ---: | ---: |")
        for rank, row in enumerate(top_results[:10], start=1):
            lines.append(
                "| "
                f"{rank} | {row['pair_id']} | `{row['hook_site']}` | "
                f"{row['recovery_fraction']:.4f} | {row['patched_logit_diff']:.4f} |"
            )
    lines.extend(["", "## Controls", ""])
    if not control_summary.get("configured"):
        lines.append(
            "No circuit patch controls were configured for this run; treat labels as "
            "uncontrolled intervention evidence."
        )
    elif not control_results:
        lines.append("Circuit patch controls were configured but produced no results.")
    else:
        lines.append("| Rank | Pair | Hook site | Recovery | Patched logit diff |")
        lines.append("| ---: | --- | --- | ---: | ---: |")
        for row in control_results:
            lines.append(
                "| "
                f"{row['rank']} | {row['pair_id']} | `{row['hook_site']}` | "
                f"{row['recovery_fraction']:.4f} | {row['patched_logit_diff']:.4f} |"
            )
    lines.append("")
    return "\n".join(lines)


def _result_notes(
    results: list[ActivationPatchSiteResult],
    missing_sites: list[dict[str, str]],
    control_hook_sites: list[str],
) -> str:
    if not results:
        return "Circuit patching completed without any patch-site results."
    control_hook_site_set = set(control_hook_sites)
    top = next(
        (
            result for result in results
            if result.hook_site not in control_hook_site_set
        ),
        results[0],
    )
    suffix = "" if not missing_sites else f" Missing pair/site results: {len(missing_sites)}."
    control_suffix = (
        " No configured controls."
        if not control_hook_sites
        else f" Control hook sites configured: {len(control_hook_sites)}."
    )
    return (
        "Circuit patching completed. "
        f"Top site {top.hook_site} on {top.pair_id} recovered "
        f"{top.recovery_fraction:.3f} of the clean-corrupted logit diff."
        f"{suffix}"
        f"{control_suffix}"
    )


def _run_artifact_dir(run: ExperimentRun) -> Path:
    expected_name = f"run-{run.id:06d}"
    if run.artifact_dir.name == expected_name:
        return run.artifact_dir
    return run.artifact_dir / expected_name
