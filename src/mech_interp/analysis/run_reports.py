from __future__ import annotations

import csv
import json
import warnings
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


def inspect_run_family(
    family: str,
    artifact_dir: Path,
    *,
    top_n: int = 5,
) -> dict[str, Any]:
    """Return a family-specific summary block for ``mech inspect-run``.

    Uses ``INSPECTOR_BY_FAMILY`` registry; warns (does not crash) on unknown
    families so Wave 2 additions (refusal_direction, acdc_edge, …) degrade
    gracefully.
    """
    inspector = INSPECTOR_BY_FAMILY.get(family)
    if inspector is None:
        warnings.warn(
            f"No inspector registered for family '{family}'; returning raw artifact list.",
            stacklevel=2,
        )
        return _fallback_inspect(artifact_dir)
    result: dict[str, Any] = inspector(artifact_dir, top_n=top_n)
    return result


def _fallback_inspect(artifact_dir: Path) -> dict[str, Any]:
    if artifact_dir.exists():
        files = sorted(p.name for p in artifact_dir.iterdir() if p.is_file())
    else:
        files = []
    return {"artifacts": files}


def _inspect_circuit_patching(artifact_dir: Path, *, top_n: int = 5) -> dict[str, Any]:
    ranked_path = artifact_dir / "patching_ranked_results.json"
    try:
        ranked: list[dict[str, Any]] = json.loads(ranked_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        ranked = []
    top = [
        {
            "hook_site": row.get("hook_site"),
            "recovery_fraction": row.get("recovery_fraction"),
            "evidence_label": row.get("evidence_label"),
        }
        for row in ranked[:top_n]
    ]
    return {"family": "circuit_patching", "top_sites": top, "total_ranked": len(ranked)}


def _inspect_polysemanticity_sae(artifact_dir: Path, *, top_n: int = 5) -> dict[str, Any]:
    analysis_path = artifact_dir / "feature_analysis.json"
    try:
        analysis: dict[str, Any] = json.loads(analysis_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        analysis = {}
    features = analysis.get("features", [])
    live = [f for f in features if not f.get("dead", True)]
    top_live = sorted(live, key=lambda f: f.get("max_activation", 0.0), reverse=True)[:top_n]
    top_features = [
        {
            "feature_index": f.get("feature_index"),
            "max_activation": f.get("max_activation"),
            "coherence_score": f.get("coherence_score"),
        }
        for f in top_live
    ]
    reconstruction_mse = analysis.get("reconstruction_mse")
    return {
        "family": "polysemanticity_sae",
        "live_feature_count": len(live),
        "total_features": len(features),
        "reconstruction_mse": reconstruction_mse,
        "top_features": top_features,
    }


def _inspect_acdc_lite(artifact_dir: Path, *, top_n: int = 5) -> dict[str, Any]:
    circuit_path = artifact_dir / "circuit.json"
    edges_path = artifact_dir / "edges.json"
    try:
        circuit: dict[str, Any] = json.loads(circuit_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        circuit = {}
    try:
        edges: list[Any] = json.loads(edges_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        edges = []
    nodes = circuit.get("nodes", [])
    survivors = [n for n in nodes if not n.get("pruned", True)]
    top_edges = [
        {"src": e.get("src"), "dst": e.get("dst"), "weight": e.get("weight")}
        for e in (edges if isinstance(edges, list) else [])[:top_n]
    ]
    return {
        "family": "acdc_lite",
        "survivor_count": len(survivors),
        "total_nodes": len(nodes),
        "faithfulness": circuit.get("faithfulness"),
        "top_edges": top_edges,
    }


def _inspect_direction(artifact_dir: Path, *, top_n: int = 5) -> dict[str, Any]:
    """Defensive inspector for refusal_direction family (Wave 2)."""
    meta_path = artifact_dir / "direction.safetensors.json"
    try:
        meta: dict[str, Any] = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        meta = {}
    return {"family": "refusal_direction", "meta": meta}


# Registry mapping family -> inspector callable
INSPECTOR_BY_FAMILY: dict[str, Any] = {
    "circuit_patching": _inspect_circuit_patching,
    "polysemanticity_sae": _inspect_polysemanticity_sae,
    "acdc_lite": _inspect_acdc_lite,
    "refusal_direction": _inspect_direction,
}


def environment_provenance(artifact_dir: Path) -> dict[str, Any] | None:
    """Read environment.json from a run artifact dir; return None if absent."""
    env_path = artifact_dir / "environment.json"
    try:
        payload = json.loads(env_path.read_text(encoding="utf-8"))
        return {
            "torch_version": (payload.get("package_versions") or {}).get("torch"),
            "seed": payload.get("seed"),
            "uv_lock_sha": (payload.get("uv_lock_sha256") or "")[:12] or None,
            "python_version": payload.get("python_version"),
        }
    except (OSError, json.JSONDecodeError):
        return None


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

    # ---- SAE per-run summaries ------------------------------------------------
    sae_summaries: list[dict[str, Any]] = []
    acdc_summaries: list[dict[str, Any]] = []
    for run_summary in run_summaries:
        if run_summary.get("family") == "polysemanticity_sae":
            # Re-read feature_analysis from artifact path if present
            result_artifacts = run_summary.get("artifacts", {})
            fa_path_str = result_artifacts.get("feature_analysis")
            fa_payload: dict[str, Any] = {}
            if fa_path_str:
                try:
                    fa_payload = json.loads(Path(fa_path_str).read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    fa_payload = {}
            features = fa_payload.get("features", []) if isinstance(fa_payload, dict) else []
            live_count = sum(1 for f in features if not f.get("dead", True))
            sae_summaries.append(
                {
                    "run_id": run_summary["run_id"],
                    "spec_name": run_summary["spec_name"],
                    "live_feature_count": live_count,
                    "total_features": len(features),
                    "reconstruction_mse": fa_payload.get("reconstruction_mse")
                    if isinstance(fa_payload, dict)
                    else None,
                }
            )
        elif run_summary.get("family") == "acdc_lite":
            result_artifacts = run_summary.get("artifacts", {})
            circuit_path_str = result_artifacts.get("circuit")
            circuit_payload: dict[str, Any] = {}
            if circuit_path_str:
                try:
                    circuit_payload = json.loads(
                        Path(circuit_path_str).read_text(encoding="utf-8")
                    )
                except (OSError, json.JSONDecodeError):
                    pass
            nodes = circuit_payload.get("nodes", [])
            survivors = [n for n in nodes if not n.get("pruned", True)]
            acdc_summaries.append(
                {
                    "run_id": run_summary["run_id"],
                    "spec_name": run_summary["spec_name"],
                    "survivor_count": len(survivors),
                    "total_nodes": len(nodes),
                    "faithfulness": circuit_payload.get("faithfulness"),
                }
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
        "sae_run_summaries": sae_summaries,
        "acdc_run_summaries": acdc_summaries,
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
        "feature_analysis",
        "circuit",
        "edges",
        "environment",
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


def _load_run_artifacts_from_dir(run_artifact_dir: Path) -> dict[str, Any]:
    """Walk a run dir and load all recognised JSON artifact files.

    This is a best-effort scan — used by export-run to include every file
    regardless of whether the runner explicitly registered it.
    """
    known_names = {
        "feature_analysis.json": "feature_analysis",
        "circuit.json": "circuit",
        "edges.json": "edges",
        "environment.json": "environment",
        "spec.json": "spec",
        "result.json": "result",
        "manifest.json": "manifest",
        "patching_summary.json": "patching_summary",
        "patching_ranked_results.json": "patching_ranked_json",
        "cross_model_probe_summary.json": "cross_model_probe_summary",
        "direction.safetensors.json": "direction_meta",
    }
    loaded: dict[str, Any] = {}
    if not run_artifact_dir.is_dir():
        return loaded
    for path in sorted(run_artifact_dir.iterdir()):
        if not path.is_file():
            continue
        key = known_names.get(path.name, path.name)
        if path.suffix == ".json":
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
    sae_rows = summary.get("sae_run_summaries", [])
    lines.extend(["", "## SAE Run Summaries", ""])
    if not sae_rows:
        lines.append("No polysemanticity_sae runs in this report window.")
    else:
        lines.append("| Run | Spec | Live Features | Total | Reconstruction MSE |")
        lines.append("| ---: | --- | ---: | ---: | ---: |")
        for row in sae_rows:
            mse = row.get("reconstruction_mse")
            mse_str = f"{float(mse):.4f}" if mse is not None else "-"
            lines.append(
                f"| {row['run_id']} | {row['spec_name']} | "
                f"{row['live_feature_count']} | {row['total_features']} | {mse_str} |"
            )

    acdc_rows = summary.get("acdc_run_summaries", [])
    lines.extend(["", "## ACDC-Lite Run Summaries", ""])
    if not acdc_rows:
        lines.append("No acdc_lite runs in this report window.")
    else:
        lines.append("| Run | Spec | Survivors | Total Nodes | Faithfulness |")
        lines.append("| ---: | --- | ---: | ---: | ---: |")
        for row in acdc_rows:
            faith = row.get("faithfulness")
            faith_str = f"{float(faith):.4f}" if faith is not None else "-"
            lines.append(
                f"| {row['run_id']} | {row['spec_name']} | "
                f"{row['survivor_count']} | {row['total_nodes']} | {faith_str} |"
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
