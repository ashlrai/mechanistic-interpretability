from pathlib import Path

from mech_interp.analysis import summarize_recent_runs
from mech_interp.storage import SQLiteResultStore
from mech_interp.types import ExperimentResult, ExperimentSpec, RunStatus


def test_summarize_recent_runs_counts_statuses_and_families(tmp_path: Path) -> None:
    store = SQLiteResultStore(tmp_path / "runs.sqlite3", tmp_path / "artifacts")
    first = store.create_run(
        ExperimentSpec(name="a", family="polysemanticity", backend="transformerlens")
    )
    second = store.create_run(
        ExperimentSpec(name="b", family="superposition", backend="transformerlens")
    )
    store.save_result(ExperimentResult(run_id=first.id, status=RunStatus.SUCCEEDED))
    store.save_result(ExperimentResult(run_id=second.id, status=RunStatus.FAILED))

    summary = summarize_recent_runs(store)

    assert summary["run_count"] == 2
    assert summary["statuses"] == {"failed": 1, "succeeded": 1}
    assert summary["families"] == {"polysemanticity": 1, "superposition": 1}
    assert summary["backends"] == {"transformerlens": 2}
