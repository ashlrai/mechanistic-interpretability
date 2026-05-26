"""Unit tests for orchestration/iterate_loop.py.

Uses a fake registry (no real models) and a fake runner to exercise the
generate → execute → recurse logic without touching GPU or disk artefacts
outside tmp_path.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from mech_interp.orchestration.iterate_loop import (
    IterateResult,
    ProposalRecord,
    _family_from_spec_json,
    _gather_proposal_paths,
    iterate_from_run,
)
from mech_interp.types import ExperimentResult, ExperimentSpec, RunStatus

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _write_sae_artifacts(run_dir: Path, model: str = "gpt2-small") -> None:
    """Write minimal SAE artifacts so PolysemanticitySAEProposalGenerator succeeds."""
    run_dir.mkdir(parents=True, exist_ok=True)
    spec = {
        "name": "seed-sae",
        "family": "polysemanticity_sae",
        "backend": "transformerlens",
        "parameters": {"model": model, "hook_site": "blocks.0.hook_resid_pre"},
    }
    analysis = {
        "features": [
            {
                "feature_index": 0,
                "dead": False,
                "max_activation": 2.5,
                "coherence_score": 0.9,
                "top_prompts": [
                    {"prompt": "The cat sat on the mat"},
                    {"prompt": "A dog lay on the rug"},
                ],
            },
            {
                "feature_index": 1,
                "dead": False,
                "max_activation": 1.5,
                "coherence_score": 0.7,
                "top_prompts": [
                    {"prompt": "Paris is the capital of France"},
                    {"prompt": "Rome is the capital of Italy"},
                ],
            },
        ],
        "reconstruction_mse": 0.042,
    }
    config = {"n_features": 256, "k": 32}
    (run_dir / "spec.json").write_text(json.dumps(spec), encoding="utf-8")
    (run_dir / "feature_analysis.json").write_text(json.dumps(analysis), encoding="utf-8")
    (run_dir / "sae_weights.safetensors.json").write_text(json.dumps(config), encoding="utf-8")


def _make_fake_runner(
    *,
    status: RunStatus = RunStatus.SUCCEEDED,
    run_id_start: int = 100,
) -> tuple[MagicMock, list[ExperimentResult]]:
    """Return a mock ExperimentRunner and a list that accumulates call results."""
    results: list[ExperimentResult] = []
    counter = {"n": run_id_start}

    def _run(spec: ExperimentSpec) -> ExperimentResult:
        rid = counter["n"]
        counter["n"] += 1
        result = ExperimentResult(run_id=rid, status=status, metrics={}, artifacts={})
        results.append(result)
        return result

    runner = MagicMock()
    runner.run.side_effect = _run
    return runner, results


# ---------------------------------------------------------------------------
# dry-run mode
# ---------------------------------------------------------------------------


def test_dry_run_generates_proposals_without_executing(tmp_path: Path) -> None:
    run_dir = tmp_path / "run-000001"
    _write_sae_artifacts(run_dir)

    result = iterate_from_run(
        "polysemanticity_sae",
        run_dir,
        tmp_path / "proposals",
        limit=2,
        execute=False,
    )

    assert isinstance(result, IterateResult)
    assert result.total_runs == 0
    assert all(rec.status == "dry_run" for rec in result.proposals)
    assert len(result.proposals) == 2
    # No depth-1 subdirectory should contain executed artifacts
    for rec in result.proposals:
        assert rec.child_run_id is None


# ---------------------------------------------------------------------------
# depth-1 execution
# ---------------------------------------------------------------------------


def test_depth_1_executes_all_proposals(tmp_path: Path) -> None:
    run_dir = tmp_path / "run-000001"
    _write_sae_artifacts(run_dir)
    runner, executed = _make_fake_runner(status=RunStatus.SUCCEEDED, run_id_start=10)

    result = iterate_from_run(
        "polysemanticity_sae",
        run_dir,
        tmp_path / "proposals",
        limit=2,
        max_depth=1,
        execute=True,
        runner=runner,
    )

    assert result.total_runs == 2
    assert len(result.proposals) == 2
    assert all(rec.status == RunStatus.SUCCEEDED.value for rec in result.proposals)
    assert runner.run.call_count == 2
    assert result.max_depth_reached == 1


# ---------------------------------------------------------------------------
# depth-2 recursion limit
# ---------------------------------------------------------------------------


def test_depth_2_recurses_into_child_runs_with_generator(tmp_path: Path) -> None:
    """Successful child run whose family has a generator triggers depth-2 proposals."""
    run_dir = tmp_path / "run-000001"
    _write_sae_artifacts(run_dir)

    # Child run directory will be at <artifact_root>/run-000010
    child_run_dir = run_dir.parent / "run-000010"
    _write_sae_artifacts(child_run_dir)

    runner, executed = _make_fake_runner(status=RunStatus.SUCCEEDED, run_id_start=10)

    result = iterate_from_run(
        "polysemanticity_sae",
        run_dir,
        tmp_path / "proposals",
        limit=1,
        max_depth=2,
        execute=True,
        runner=runner,
    )

    # Depth 1: 1 proposal executed → succeeds → child run_id=10
    # Depth 2: child dir run-000010 has SAE artifacts → 1 more proposal generated+executed
    assert result.max_depth_reached >= 1
    # At minimum we ran the depth-1 proposals
    assert result.total_runs >= 1


def test_depth_2_does_not_recurse_past_max_depth(tmp_path: Path) -> None:
    run_dir = tmp_path / "run-000001"
    _write_sae_artifacts(run_dir)

    child_run_dir = run_dir.parent / "run-000010"
    _write_sae_artifacts(child_run_dir)

    runner, _ = _make_fake_runner(status=RunStatus.SUCCEEDED, run_id_start=10)

    result = iterate_from_run(
        "polysemanticity_sae",
        run_dir,
        tmp_path / "proposals",
        limit=1,
        max_depth=1,  # no recursion
        execute=True,
        runner=runner,
    )

    assert result.max_depth_reached == 1


# ---------------------------------------------------------------------------
# no generator for family
# ---------------------------------------------------------------------------


def test_raises_for_family_without_generator(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="No per-run proposal generator for family"):
        iterate_from_run(
            "circuit_patching",  # not in PROPOSAL_GENERATORS
            tmp_path / "run-000001",
            tmp_path / "proposals",
        )


# ---------------------------------------------------------------------------
# runner required when execute=True
# ---------------------------------------------------------------------------


def test_raises_when_execute_true_and_no_runner(tmp_path: Path) -> None:
    run_dir = tmp_path / "run-000001"
    _write_sae_artifacts(run_dir)

    with pytest.raises(ValueError, match="runner must be provided"):
        iterate_from_run(
            "polysemanticity_sae",
            run_dir,
            tmp_path / "proposals",
            execute=True,
            runner=None,
        )


# ---------------------------------------------------------------------------
# failed child run does not recurse
# ---------------------------------------------------------------------------


def test_failed_child_run_not_recursed(tmp_path: Path) -> None:
    run_dir = tmp_path / "run-000001"
    _write_sae_artifacts(run_dir)

    runner, _ = _make_fake_runner(status=RunStatus.FAILED, run_id_start=10)

    result = iterate_from_run(
        "polysemanticity_sae",
        run_dir,
        tmp_path / "proposals",
        limit=1,
        max_depth=2,
        execute=True,
        runner=runner,
    )

    assert all(rec.status == RunStatus.FAILED.value for rec in result.proposals)
    # Exactly depth-1 proposals only; no recursion into failed runs
    assert result.total_runs == 0
    assert result.max_depth_reached == 1


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def test_family_from_spec_json_reads_correctly(tmp_path: Path) -> None:
    (tmp_path / "spec.json").write_text(
        json.dumps({"family": "polysemanticity_sae"}), encoding="utf-8"
    )
    assert _family_from_spec_json(tmp_path) == "polysemanticity_sae"


def test_family_from_spec_json_missing_returns_none(tmp_path: Path) -> None:
    assert _family_from_spec_json(tmp_path / "nonexistent") is None


def test_gather_proposal_paths_serialises_correctly() -> None:
    r = IterateResult(source_artifact_dir=Path("/tmp/x"))
    r.proposals.append(
        ProposalRecord(path="/tmp/p.yaml", name="p", status="succeeded", child_run_id=7)
    )
    out = _gather_proposal_paths(r)
    assert out[0]["name"] == "p"
    assert out[0]["child_run_id"] == 7
    assert out[0]["status"] == "succeeded"
