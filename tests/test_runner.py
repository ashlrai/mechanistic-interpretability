from pathlib import Path

import pytest

from mech_interp.orchestration import ExperimentRunner
from mech_interp.orchestration.runner import (
    PLACEHOLDER_ENV_VAR,
    FamilyNotImplementedError,
)
from mech_interp.storage import ArtifactStore, SQLiteResultStore
from mech_interp.types import ExperimentSpec, RunStatus


def _make_runner(tmp_path: Path) -> ExperimentRunner:
    return ExperimentRunner(
        result_store=SQLiteResultStore(tmp_path / "runs.sqlite3", tmp_path / "artifacts"),
        artifact_store=ArtifactStore(tmp_path / "artifacts"),
    )


def _spec(family: str = "an_unmapped_family") -> ExperimentSpec:
    return ExperimentSpec(
        name="placeholder-smoke",
        family=family,
        backend="transformerlens",
        description="placeholder smoke",
        parameters={"seed": 7},
    )


def test_runner_persists_placeholder_result_when_opted_in(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(PLACEHOLDER_ENV_VAR, "1")
    result = _make_runner(tmp_path).run(_spec())

    assert result.status == RunStatus.SUCCEEDED
    assert "manifest" in result.artifacts
    assert (tmp_path / "artifacts" / "run-000001" / "spec.json").exists()
    assert (tmp_path / "artifacts" / "run-000001" / "result.json").exists()
    assert (tmp_path / "artifacts" / "run-000001" / "environment.json").exists()


def test_runner_blocks_unmapped_family_without_opt_in(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(PLACEHOLDER_ENV_VAR, raising=False)
    result = _make_runner(tmp_path).run(_spec())

    assert result.status == RunStatus.FAILED
    assert "FamilyNotImplementedError" in (result.notes or "")
    # Environment fingerprint is still written even when the experiment is blocked,
    # so the audit trail captures what was attempted.
    assert (tmp_path / "artifacts" / "run-000001" / "environment.json").exists()


def test_environment_fingerprint_captures_seed_and_versions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(PLACEHOLDER_ENV_VAR, "1")
    _make_runner(tmp_path).run(_spec())

    import json

    env = json.loads((tmp_path / "artifacts" / "run-000001" / "environment.json").read_text())
    assert env["seed"] == 7
    assert "python_version" in env
    assert "package_versions" in env
    assert "numpy" in env["package_versions"]


def test_family_not_implemented_error_is_actionable() -> None:
    # Direct check so the error message tells users exactly what to do.
    msg = str(FamilyNotImplementedError("Experiment family 'foo' has no real implementation."))
    assert "foo" in msg
