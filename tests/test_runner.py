from pathlib import Path

from mech_interp.experiments import load_experiment_specs
from mech_interp.orchestration import ExperimentRunner
from mech_interp.storage import ArtifactStore, SQLiteResultStore
from mech_interp.types import RunStatus


def test_runner_persists_placeholder_result(tmp_path: Path) -> None:
    spec = load_experiment_specs("experiments").get("polysemanticity-smoke")
    runner = ExperimentRunner(
        result_store=SQLiteResultStore(tmp_path / "runs.sqlite3", tmp_path / "artifacts"),
        artifact_store=ArtifactStore(tmp_path / "artifacts"),
    )

    result = runner.run(spec)

    assert result.status == RunStatus.SUCCEEDED
    assert "manifest" in result.artifacts
    assert (tmp_path / "artifacts" / "run-000001" / "spec.json").exists()
    assert (tmp_path / "artifacts" / "run-000001" / "result.json").exists()
