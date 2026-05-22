from pathlib import Path

from mech_interp.storage import SQLiteResultStore
from mech_interp.types import ExperimentResult, ExperimentSpec, RunStatus


def test_sqlite_store_creates_run_and_result(tmp_path: Path) -> None:
    store = SQLiteResultStore(tmp_path / "runs.sqlite3", tmp_path / "artifacts")
    spec = ExperimentSpec(name="test", family="polysemanticity", backend="transformerlens")

    run = store.create_run(spec)
    store.save_result(
        ExperimentResult(
            run_id=run.id,
            status=RunStatus.SUCCEEDED,
            metrics={"score": 1.0},
            artifacts={"report": "report.md"},
        )
    )

    assert run.id == 1
    assert (tmp_path / "runs.sqlite3").exists()
    assert (tmp_path / "artifacts").exists()
