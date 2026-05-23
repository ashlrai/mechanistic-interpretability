from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mech_interp.storage import SQLiteResultStore


@dataclass(frozen=True)
class AggregateReportArtifacts:
    output_dir: Path
    summary_json: Path
    research_note: Path
    top_sites_csv: Path


def summarize_recent_runs(store: SQLiteResultStore, limit: int = 100) -> dict[str, object]:
    runs = store.list_runs(limit=limit)
    return {
        "run_count": len(runs),
        "statuses": dict(sorted(Counter(run.status.value for run in runs).items())),
        "families": dict(sorted(Counter(run.family for run in runs).items())),
        "backends": dict(sorted(Counter(run.backend for run in runs).items())),
    }


def write_aggregate_reports(
    store: SQLiteResultStore,
    output_dir: str | Path,
    *,
    limit: int = 100,
) -> AggregateReportArtifacts:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    runs = store.list_runs(limit=limit)
    run_summaries: list[dict[str, Any]] = []
    top_sites: list[dict[str, Any]] = []
    control_sites: list[dict[str, Any]] = []
    representation_probes: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for run in runs:
        result = store.get_result(run.id)
        spec = store.get_run_spec(run.id) or {}
        run_summary: dict[str, Any] = {
            "run_id": run.id,
            "spec_name": run.spec_name,
            "family": run.family,
            "backend": run.backend,
            "status": run.status.value,
            "metrics": result.metrics if result is not None else {},
            "notes": result.notes if result is not None else "",
            "artifacts": result.artifacts if result is not None else {},
            "spec": spec,
        }
        if result is not None:
            artifact_payloads = _load_known_artifacts(result.artifacts)
            run_summary["loaded_artifacts"] = sorted(artifact_payloads)
            summary = artifact_payloads.get("patching_summary")
            if isinstance(summary, dict):
                run_summary["patching_summary"] = summary
            probe_summary = artifact_payloads.get("cross_model_probe_summary")
            if isinstance(probe_summary, dict):
                run_summary["cross_model_probe_summary"] = probe_summary
                representation_probes.append(
                    {
                        "run_id": run.id,
                        "spec_name": run.spec_name,
                        "source_model": probe_summary.get("source_model"),
                        "target_model": probe_summary.get("target_model"),
                        "source_hook_site": probe_summary.get("source_hook_site"),
                        "target_hook_site": probe_summary.get("target_hook_site"),
                        "eval_mean_cosine_similarity": result.metrics.get(
                            "eval_mean_cosine_similarity",
                            0.0,
                        ),
                        "eval_variance_explained": result.metrics.get(
                            "eval_variance_explained",
                            0.0,
                        ),
                        "evidence_label": _evidence_label(
                            probe_summary,
                            "correlational alignment",
                        ),
                    }
                )
            ranked = artifact_payloads.get("patching_ranked_json")
            if isinstance(ranked, list):
                for source_rank, row in enumerate(ranked[:20], start=1):
                    if isinstance(row, dict):
                        evidence_label = _evidence_label(row, "causal evidence")
                        indexed_row = {
                            "run_id": run.id,
                            "spec_name": run.spec_name,
                            "source_rank": row.get("rank", source_rank),
                            **row,
                            "evidence_label": evidence_label,
                        }
                        if evidence_label == "control":
                            control_sites.append(indexed_row)
                        else:
                            top_sites.append(indexed_row)
            if run.status.value == "failed":
                failures.append(
                    {
                        "run_id": run.id,
                        "spec_name": run.spec_name,
                        "notes": result.notes,
                        "artifacts": result.artifacts,
                    }
                )
        run_summaries.append(run_summary)

    ranked_top_sites = _rank_rows(
        sorted(
            top_sites,
            key=lambda row: float(row.get("recovery_fraction", 0.0) or 0.0),
            reverse=True,
        )[:50]
    )
    ranked_control_sites = _rank_rows(
        sorted(
            control_sites,
            key=lambda row: float(row.get("recovery_fraction", 0.0) or 0.0),
            reverse=True,
        )[:50]
    )
    summary = {
        **summarize_recent_runs(store, limit=limit),
        "runs": run_summaries,
        "failed_runs": failures,
        "evidence_labels": {
            "causal evidence": "Activation patch intervention result.",
            "control": "Configured circuit patch control result.",
            "correlational alignment": "Cross-model representation probe statistic.",
            "hypothesis": "Generated interpretation requiring follow-up tests.",
        },
        "top_circuit_patching_sites": ranked_top_sites,
        "circuit_patching_control_sites": ranked_control_sites,
        "cross_model_representation_probes": sorted(
            representation_probes,
            key=lambda row: float(row.get("eval_mean_cosine_similarity", 0.0) or 0.0),
            reverse=True,
        ),
    }
    summary_json = output / "latest_summary.json"
    research_note = output / "latest_research_note.md"
    top_sites_csv = output / "circuit_patching_top_sites.csv"
    summary_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    research_note.write_text(_render_aggregate_note(summary), encoding="utf-8")
    _write_top_sites_csv(top_sites_csv, summary["top_circuit_patching_sites"])
    return AggregateReportArtifacts(
        output_dir=output,
        summary_json=summary_json,
        research_note=research_note,
        top_sites_csv=top_sites_csv,
    )


def _load_known_artifacts(artifacts: dict[str, str]) -> dict[str, Any]:
    loaded: dict[str, Any] = {}
    for key in (
        "manifest",
        "patching_summary",
        "patching_ranked_json",
        "cross_model_probe_summary",
        "cross_model_probe_results_json",
    ):
        value = artifacts.get(key)
        if not value:
            continue
        path = Path(value)
        try:
            loaded[key] = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            loaded[key] = {"missing_or_invalid": str(path)}
    return loaded


def _evidence_label(row: dict[str, Any], default: str) -> str:
    value = row.get("evidence_label")
    return value if isinstance(value, str) and value.strip() else default


def _rank_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {**row, "rank": rank}
        for rank, row in enumerate(rows, start=1)
    ]


def _render_aggregate_note(summary: dict[str, Any]) -> str:
    lines = [
        "# Latest Research Summary",
        "",
        f"- Runs summarized: {summary['run_count']}",
        f"- Statuses: {summary['statuses']}",
        f"- Families: {summary['families']}",
        "- Evidence labels: causal evidence = patch intervention result; control = "
        "configured negative/control patch site; correlational alignment = probe "
        "statistic; hypothesis = generated interpretation for follow-up.",
        "",
        "## Top Circuit Patching Sites",
        "",
    ]
    top_sites = summary["top_circuit_patching_sites"]
    if not top_sites:
        lines.append("No circuit patching ranked results were available.")
    else:
        lines.append("| Run | Spec | Label | Pair | Hook site | Recovery |")
        lines.append("| ---: | --- | --- | --- | --- | ---: |")
        for row in top_sites[:10]:
            lines.append(
                "| "
                f"{row['run_id']} | {row['spec_name']} | "
                f"{row.get('evidence_label', 'causal evidence')} | "
                f"{row.get('pair_id', '-')} | "
                f"`{row.get('hook_site', '-')}` | "
                f"{float(row.get('recovery_fraction', 0.0) or 0.0):.4f} |"
            )
    control_sites = summary.get("circuit_patching_control_sites", [])
    lines.extend(["", "## Circuit Patching Controls", ""])
    if not control_sites:
        lines.append("No circuit patch control results were available.")
    else:
        lines.append("| Run | Spec | Pair | Hook site | Recovery |")
        lines.append("| ---: | --- | --- | --- | ---: |")
        for row in control_sites[:10]:
            lines.append(
                "| "
                f"{row['run_id']} | {row['spec_name']} | "
                f"{row.get('pair_id', '-')} | "
                f"`{row.get('hook_site', '-')}` | "
                f"{float(row.get('recovery_fraction', 0.0) or 0.0):.4f} |"
            )
    probe_rows = summary["cross_model_representation_probes"]
    lines.extend(["", "## Cross-Model Representation Probes", ""])
    if not probe_rows:
        lines.append("No cross-model representation probe summaries were available.")
    else:
        lines.append(
            "| Run | Spec | Label | Source -> Target | Sites | Eval cosine | Eval variance |"
        )
        lines.append("| ---: | --- | --- | --- | --- | ---: | ---: |")
        for row in probe_rows[:10]:
            lines.append(
                "| "
                f"{row['run_id']} | {row['spec_name']} | "
                f"{row.get('evidence_label', 'correlational alignment')} | "
                f"{row.get('source_model', '-')} -> {row.get('target_model', '-')} | "
                f"`{row.get('source_hook_site', '-')}` -> "
                f"`{row.get('target_hook_site', '-')}` | "
                f"{float(row.get('eval_mean_cosine_similarity', 0.0) or 0.0):.4f} | "
                f"{float(row.get('eval_variance_explained', 0.0) or 0.0):.4f} |"
            )
    lines.extend(["", "## Failed Runs", ""])
    failures = summary["failed_runs"]
    if not failures:
        lines.append("No failed runs in this report window.")
    else:
        for failure in failures:
            lines.append(
                f"- Run {failure['run_id']} ({failure['spec_name']}): {failure['notes'] or '-'}"
            )
    lines.append("")
    return "\n".join(lines)


def _write_top_sites_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "run_id",
        "spec_name",
        "rank",
        "source_rank",
        "evidence_label",
        "pair_id",
        "hook_site",
        "clean_logit_diff",
        "corrupted_logit_diff",
        "patched_logit_diff",
        "recovery_fraction",
        "activation_norm",
    ]
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
