from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from mech_interp.config import load_config
from mech_interp.experiments.registry import load_experiment_spec


@dataclass(frozen=True)
class ProposalResult:
    output_dir: Path
    manifest_path: Path
    spec_paths: list[Path]


def propose_followups(
    family: str,
    output_dir: str | Path,
    *,
    limit: int = 20,
    reports_dir: str | Path | None = None,
) -> ProposalResult:
    if family != "circuit_patching":
        raise ValueError("Only circuit_patching follow-up proposals are supported in V1.")
    config = load_config()
    report_dir = (
        Path(reports_dir) if reports_dir is not None else config.project.artifact_dir / "reports"
    )
    summary_path = report_dir / "latest_summary.json"
    summary = _read_summary(summary_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    proposals: list[dict[str, Any]] = []
    for index, site in enumerate(summary.get("top_circuit_patching_sites", [])[:limit], start=1):
        if not isinstance(site, dict):
            continue
        spec = _proposal_from_site(index, site)
        spec_path = output / f"{spec['name']}.yaml"
        spec_path.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
        load_experiment_spec(spec_path)
        proposals.append(
            {
                "path": str(spec_path),
                "name": spec["name"],
                "source_run_ids": [site.get("run_id")],
                "rationale": (
                    "Top recovered patch site from aggregate report; proposed as a focused "
                    "causal retest with the same hook site."
                ),
                "validated": True,
            }
        )

    for failure in summary.get("failed_runs", [])[: max(limit - len(proposals), 0)]:
        if not isinstance(failure, dict):
            continue
        proposals.append(
            {
                "path": None,
                "name": f"diagnose-run-{failure.get('run_id')}",
                "source_run_ids": [failure.get("run_id")],
                "rationale": f"Failed run needs manual diagnosis: {failure.get('notes') or '-'}",
                "validated": False,
            }
        )

    manifest = {
        "family": family,
        "source_report": str(summary_path),
        "proposal_count": len(proposals),
        "proposals": proposals,
        "guardrail": "Generated specs are not executed automatically.",
    }
    manifest_path = output / "proposal_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return ProposalResult(
        output_dir=output,
        manifest_path=manifest_path,
        spec_paths=[Path(item["path"]) for item in proposals if item.get("path")],
    )


def _read_summary(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError:
        return {"top_circuit_patching_sites": [], "failed_runs": []}
    if not isinstance(payload, dict):
        raise ValueError(f"Aggregate report {path} did not contain a JSON object.")
    return payload


def _proposal_from_site(index: int, site: dict[str, Any]) -> dict[str, Any]:
    hook_site = str(site.get("hook_site") or "blocks.0.hook_resid_pre")
    source_name = str(site.get("spec_name") or "unknown").replace("_", "-")
    return {
        "name": f"proposed-circuit-followup-{index:04d}",
        "family": "circuit_patching",
        "backend": "transformerlens",
        "description": f"Follow-up causal retest for {source_name} at {hook_site}.",
        "parameters": {
            "model": "gpt2-small",
            "source_prompt": "Replace with the clean prompt from the source run.",
            "target_prompt": "Replace with the corrupted prompt from the source run.",
            "answer_tokens": {"correct": " yes", "incorrect": " no"},
            "hook_sites": [hook_site],
            "sequence_length": 32,
            "resource_policy": {"max_activation_fraction": 0.25},
            "artifact_policy": {
                "retain_activation_tensors": False,
                "write_report": True,
            },
        },
    }
