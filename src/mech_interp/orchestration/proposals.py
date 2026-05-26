from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from mech_interp.config import load_config
from mech_interp.experiments.registry import load_experiment_spec
from mech_interp.orchestration.proposal_generators import PROPOSAL_GENERATORS


@dataclass(frozen=True)
class ProposalResult:
    output_dir: Path
    manifest_path: Path
    spec_paths: list[Path]


def propose_from_run(
    family: str,
    artifact_dir: str | Path,
    output_dir: str | Path,
    *,
    limit: int = 5,
) -> ProposalResult:
    """Per-run follow-up generator for the agentic loop.

    Routes to the appropriate ``ProposalGenerator`` from ``PROPOSAL_GENERATORS``;
    raises ``ValueError`` if the family has no generator. Each emitted spec is
    YAML-written next to a manifest, then validated through the registry so
    malformed proposals fail loudly before they're queued.
    """
    if family not in PROPOSAL_GENERATORS:
        supported = ", ".join(sorted(PROPOSAL_GENERATORS))
        raise ValueError(
            f"No per-run proposal generator for family '{family}'. Supported: {supported}."
        )
    generator = PROPOSAL_GENERATORS[family]
    artifact_dir = Path(artifact_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    specs = generator.generate(artifact_dir, limit=limit)
    proposals: list[dict[str, Any]] = []
    for spec in specs:
        spec_path = output / f"{spec['name']}.yaml"
        spec_path.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
        load_experiment_spec(spec_path)
        proposals.append(
            {
                "path": str(spec_path),
                "name": spec["name"],
                "source_artifact_dir": str(artifact_dir),
                "rationale": spec.get("description", ""),
                "validated": True,
            }
        )

    manifest = {
        "family": family,
        "source_artifact_dir": str(artifact_dir),
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


def propose_followups(
    family: str,
    output_dir: str | Path,
    *,
    limit: int = 20,
    reports_dir: str | Path | None = None,
) -> ProposalResult:
    if family != "circuit_patching":
        raise ValueError(
            f"Aggregate-report follow-ups are only supported for circuit_patching; "
            f"for '{family}' use propose_from_run(...) against the run artifact dir."
        )
    config = load_config()
    report_dir = (
        Path(reports_dir) if reports_dir is not None else config.project.artifact_dir / "reports"
    )
    summary_path = report_dir / "latest_summary.json"
    summary = _read_summary(summary_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    proposals: list[dict[str, Any]] = []
    specs_by_run_id = _specs_by_run_id(summary)
    for index, site in enumerate(summary.get("top_circuit_patching_sites", [])[:limit], start=1):
        if not isinstance(site, dict):
            continue
        source_spec = specs_by_run_id.get(int(site["run_id"])) if "run_id" in site else None
        spec = _proposal_from_site(index, site, source_spec)
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


def _specs_by_run_id(summary: dict[str, Any]) -> dict[int, dict[str, Any]]:
    runs = summary.get("runs", [])
    specs: dict[int, dict[str, Any]] = {}
    if not isinstance(runs, list):
        return specs
    for run in runs:
        if not isinstance(run, dict):
            continue
        run_id = run.get("run_id")
        spec = run.get("spec")
        if isinstance(run_id, int) and isinstance(spec, dict):
            specs[run_id] = spec
    return specs


def _proposal_from_site(
    index: int,
    site: dict[str, Any],
    source_spec: dict[str, Any] | None,
) -> dict[str, Any]:
    hook_site = str(site.get("hook_site") or "blocks.0.hook_resid_pre")
    source_name = str(site.get("spec_name") or "unknown").replace("_", "-")
    source_parameters = source_spec.get("parameters", {}) if source_spec is not None else {}
    if not isinstance(source_parameters, dict):
        source_parameters = {}
    prompts = _source_prompts(source_parameters)
    answer_tokens = _answer_tokens(source_parameters)
    return {
        "name": f"proposed-circuit-followup-{index:04d}",
        "family": "circuit_patching",
        "backend": str(source_spec.get("backend", "transformerlens"))
        if source_spec is not None
        else "transformerlens",
        "description": f"Follow-up causal retest for {source_name} at {hook_site}.",
        "parameters": {
            "model": str(source_parameters.get("model", "gpt2-small")),
            "source_prompt": prompts["source_prompt"],
            "target_prompt": prompts["target_prompt"],
            "answer_tokens": answer_tokens,
            "hook_sites": [hook_site],
            "target_position": int(source_parameters.get("target_position", -1)),
            "patch_position": int(source_parameters.get("patch_position", -1)),
            "sequence_length": int(source_parameters.get("sequence_length", 32)),
            "resource_policy": {"max_activation_fraction": 0.25},
            "artifact_policy": {
                "retain_activation_tensors": False,
                "write_report": True,
            },
        },
    }


def _source_prompts(parameters: dict[str, Any]) -> dict[str, str]:
    prompt_pairs = parameters.get("prompt_pairs")
    if isinstance(prompt_pairs, list) and prompt_pairs and isinstance(prompt_pairs[0], dict):
        first_pair = prompt_pairs[0]
        clean = first_pair.get("clean_prompt")
        corrupted = first_pair.get("corrupted_prompt")
        if isinstance(clean, str) and isinstance(corrupted, str):
            return {"source_prompt": clean, "target_prompt": corrupted}
    clean = parameters.get("source_prompt") or parameters.get("clean_prompt")
    corrupted = parameters.get("target_prompt") or parameters.get("corrupted_prompt")
    return {
        "source_prompt": str(clean or "The Eiffel Tower is in Paris"),
        "target_prompt": str(corrupted or "The Eiffel Tower is in Rome"),
    }


def _answer_tokens(parameters: dict[str, Any]) -> dict[str, str]:
    prompt_pairs = parameters.get("prompt_pairs")
    if isinstance(prompt_pairs, list) and prompt_pairs and isinstance(prompt_pairs[0], dict):
        first_pair = prompt_pairs[0]
        correct = first_pair.get("correct_token")
        incorrect = first_pair.get("incorrect_token")
        if isinstance(correct, str) and isinstance(incorrect, str):
            return {"correct": correct, "incorrect": incorrect}
    answer_tokens = parameters.get("answer_tokens")
    if isinstance(answer_tokens, dict):
        correct = answer_tokens.get("correct")
        incorrect = answer_tokens.get("incorrect")
        if isinstance(correct, str) and isinstance(incorrect, str):
            return {"correct": correct, "incorrect": incorrect}
    return {"correct": " Paris", "incorrect": " Rome"}
