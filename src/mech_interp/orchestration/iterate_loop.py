"""Closed-loop iterate for per-run proposal families.

Chains propose_from_run → execute → (optionally recurse) so agentic research
loops require no manual intervention after a seed run completes.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mech_interp.experiments.registry import load_experiment_spec
from mech_interp.orchestration.proposal_generators import PROPOSAL_GENERATORS
from mech_interp.orchestration.proposals import propose_from_run
from mech_interp.orchestration.runner import ExperimentRunner
from mech_interp.types import RunStatus


@dataclass
class ProposalRecord:
    """Outcome for a single generated proposal within an iterate run."""

    path: str
    name: str
    status: str  # "dry_run" | "succeeded" | "failed" | "skipped"
    child_run_id: int | None = None
    notes: str = ""


@dataclass
class IterateResult:
    source_artifact_dir: Path
    proposals: list[ProposalRecord] = field(default_factory=list)
    total_runs: int = 0
    max_depth_reached: int = 0


def iterate_from_run(
    family: str,
    artifact_dir: Path,
    output_dir: Path,
    *,
    limit: int = 5,
    max_depth: int = 1,
    execute: bool = True,
    runner: ExperimentRunner | None = None,
    _current_depth: int = 1,
) -> IterateResult:
    """Source-run → generate proposals → execute proposals → recurse.

    Parameters
    ----------
    family:
        Experiment family of the *source* run (must be in PROPOSAL_GENERATORS).
    artifact_dir:
        Artifact directory of the source run.
    output_dir:
        Root directory for all generated specs and manifests.  Depth-specific
        subdirectories (``depth-1/``, ``depth-2/``, …) are created automatically.
    limit:
        Maximum number of follow-up specs to generate per depth level.
    max_depth:
        Maximum recursion depth.  ``max_depth=1`` means generate-and-execute the
        first tier of proposals only; ``max_depth=2`` also recurses into
        successful child runs.
    execute:
        If ``False`` behave like ``propose_from_run`` — write specs but do not
        run them (equivalent to ``--dry-run``).
    runner:
        ``ExperimentRunner`` instance.  Required when ``execute=True``; ignored
        when ``execute=False``.

    Raises
    ------
    ValueError
        If ``family`` has no registered ``ProposalGenerator``.
    """
    if family not in PROPOSAL_GENERATORS:
        supported = ", ".join(sorted(PROPOSAL_GENERATORS))
        raise ValueError(
            f"No per-run proposal generator for family '{family}'. Supported: {supported}."
        )
    if execute and runner is None:
        raise ValueError("runner must be provided when execute=True.")

    depth_dir = output_dir / f"depth-{_current_depth}"
    proposal_result = propose_from_run(family, artifact_dir, depth_dir, limit=limit)

    result = IterateResult(
        source_artifact_dir=artifact_dir,
        max_depth_reached=_current_depth,
    )

    for spec_path in proposal_result.spec_paths:
        record = ProposalRecord(path=str(spec_path), name=spec_path.stem, status="dry_run")

        if not execute:
            result.proposals.append(record)
            continue

        # Execute the spec.
        try:
            spec = load_experiment_spec(spec_path)
            run_result = runner.run(spec)  # type: ignore[union-attr]
            record.child_run_id = run_result.run_id
            record.status = run_result.status.value
            if run_result.status == RunStatus.SUCCEEDED:
                result.total_runs += 1
        except Exception as exc:  # noqa: BLE001
            record.status = "failed"
            record.notes = f"{type(exc).__name__}: {exc}"

        result.proposals.append(record)

    # Recurse into successful child runs that have a proposal generator.
    if execute and _current_depth < max_depth:
        for record in list(result.proposals):
            if record.status != RunStatus.SUCCEEDED.value:
                continue
            if record.child_run_id is None:
                continue
            # Determine the child run's family to see if it supports proposal generation.
            child_artifact_dir = _artifact_dir_for_run_id(artifact_dir, record.child_run_id)
            child_family = _family_from_spec_json(child_artifact_dir)
            if child_family is None or child_family not in PROPOSAL_GENERATORS:
                # Child family doesn't support proposal generation; skip silently.
                continue
            try:
                child_result = iterate_from_run(
                    child_family,
                    child_artifact_dir,
                    output_dir,
                    limit=limit,
                    max_depth=max_depth,
                    execute=execute,
                    runner=runner,
                    _current_depth=_current_depth + 1,
                )
                result.proposals.extend(child_result.proposals)
                result.total_runs += child_result.total_runs
                result.max_depth_reached = max(
                    result.max_depth_reached, child_result.max_depth_reached
                )
            except ValueError as exc:
                warnings.warn(
                    f"Skipping recursion for child run {record.child_run_id}: {exc}",
                    stacklevel=2,
                )

    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _artifact_dir_for_run_id(sibling_dir: Path, run_id: int) -> Path:
    """Infer the artifact directory for a child run.

    The runner writes artifacts under ``<artifact_root>/run-<run_id:06d>/``.
    We walk up from the source artifact dir to find the root.
    """
    # sibling_dir is e.g. .../artifacts/run-000001
    # try parent then parent/run-XXXXXX
    parent = sibling_dir.parent
    candidate = parent / f"run-{run_id:06d}"
    return candidate


def _family_from_spec_json(artifact_dir: Path) -> str | None:
    """Read the family recorded in the spec.json artifact, if present."""
    spec_path = artifact_dir / "spec.json"
    try:
        payload: dict[str, Any] = json.loads(spec_path.read_text(encoding="utf-8"))
        family = payload.get("family")
        return str(family) if family is not None else None
    except (OSError, ValueError):
        return None


def _gather_proposal_paths(result: IterateResult) -> list[dict[str, Any]]:
    """Flatten result proposals to a serialisable list (for CLI display)."""
    return [
        {
            "path": rec.path,
            "name": rec.name,
            "status": rec.status,
            "child_run_id": rec.child_run_id,
            "notes": rec.notes,
        }
        for rec in result.proposals
    ]
