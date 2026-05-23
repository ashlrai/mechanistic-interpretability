from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from mech_interp.orchestration.preflight import preflight_spec
from mech_interp.orchestration.proposals import propose_followups
from mech_interp.storage.sqlite_store import SQLiteResultStore
from mech_interp.types import ExperimentSpec


@dataclass(frozen=True)
class IterationCaps:
    max_generated_specs: int = 50
    max_queued_per_iteration: int = 10
    max_failed_retry_count: int = 2
    allow_tensor_retention: bool = False


@dataclass(frozen=True)
class IterationResult:
    manifest_path: Path
    generated: int
    queued: int


def run_bounded_iteration(
    store: SQLiteResultStore,
    output_dir: Path,
    candidate_specs: list[ExperimentSpec],
    caps: IterationCaps,
) -> IterationResult:
    accepted: list[ExperimentSpec] = []
    proposals: list[dict[str, Any]] = []
    max_generated = max(caps.max_generated_specs, 0)
    max_queued = max(caps.max_queued_per_iteration, 0)
    queue_items = {item.spec_name: item for item in store.list_queue_items()}
    seen_names: set[str] = set()
    seen_hashes: set[str] = set()
    bounded_candidates = candidate_specs[:max_generated]
    for spec in bounded_candidates:
        report = preflight_spec(spec)
        retains_tensors = _retains_tensors(spec)
        spec_hash = _spec_hash(spec)
        rejection_reasons = _rejection_reasons(
            spec,
            spec_hash,
            report_ok=report.ok,
            retains_tensors=retains_tensors,
            caps=caps,
            queue_items=queue_items,
            seen_names=seen_names,
            seen_hashes=seen_hashes,
        )
        validation_status = "valid" if not rejection_reasons else "rejected"
        proposals.append(
            {
                "spec_name": spec.name,
                "spec_sha256": spec_hash,
                "source_run_ids": spec.parameters.get("source_run_ids", []),
                "rationale": spec.parameters.get("rationale", "Generated from local run evidence."),
                "falsification_condition": spec.parameters.get(
                    "falsification_condition",
                    (
                        "Follow-up fails to improve the priority metric or control "
                        "recovery remains high."
                    ),
                ),
                "resource_estimate": spec.parameters.get("resource_estimate", {}),
                "validation_status": validation_status,
                "rejection_reasons": rejection_reasons,
                "preflight": [check.__dict__ for check in report.checks],
            }
        )
        if validation_status == "valid":
            seen_names.add(spec.name)
            seen_hashes.add(spec_hash)
        if validation_status == "valid" and len(accepted) < max_queued:
            accepted.append(spec)

    queued = store.enqueue_experiment_specs(accepted)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "iteration_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "caps": caps.__dict__,
                "generated": len(bounded_candidates),
                "queued": queued,
                "proposals": proposals,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return IterationResult(
        manifest_path=manifest_path,
        generated=len(bounded_candidates),
        queued=queued,
    )


def propose_and_enqueue_iteration(
    store: SQLiteResultStore,
    family: str,
    proposal_dir: Path,
    caps: IterationCaps,
) -> IterationResult:
    proposed = propose_followups(family, proposal_dir, limit=caps.max_generated_specs)
    specs = [_spec_from_yaml(path) for path in proposed.spec_paths]
    return run_bounded_iteration(store, proposal_dir, specs, caps)


def _spec_from_yaml(path: Path) -> ExperimentSpec:
    from mech_interp.experiments.registry import load_experiment_spec

    return load_experiment_spec(path)


def _retains_tensors(spec: ExperimentSpec) -> bool:
    policy = spec.parameters.get("artifact_policy", {})
    return bool(
        (isinstance(policy, dict) and policy.get("retain_activation_tensors"))
        or (isinstance(policy, dict) and policy.get("retain_probe_weights"))
        or spec.parameters.get("retain_probe_weights")
    )


def _rejection_reasons(
    spec: ExperimentSpec,
    spec_hash: str,
    *,
    report_ok: bool,
    retains_tensors: bool,
    caps: IterationCaps,
    queue_items: dict[str, Any],
    seen_names: set[str],
    seen_hashes: set[str],
) -> list[str]:
    reasons: list[str] = []
    if not report_ok:
        reasons.append("preflight_error")
    if retains_tensors and not caps.allow_tensor_retention:
        reasons.append("tensor_retention_blocked")
    if spec.name in seen_names or spec_hash in seen_hashes:
        reasons.append("duplicate_candidate")
    queue_item = queue_items.get(spec.name)
    if queue_item is not None:
        if (
            str(queue_item.status) == "failed"
            and queue_item.retry_count >= caps.max_failed_retry_count
        ):
            reasons.append("failed_retry_cap")
        else:
            reasons.append("duplicate_queue")
    return reasons


def _spec_hash(spec: ExperimentSpec) -> str:
    import hashlib

    payload = json.dumps(asdict(spec), default=str, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
