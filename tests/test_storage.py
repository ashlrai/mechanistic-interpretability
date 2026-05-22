import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from mech_interp.storage import SQLiteResultStore
from mech_interp.types import ExperimentResult, ExperimentSpec, RunStatus


def test_sqlite_store_creates_run_and_result(tmp_path: Path) -> None:
    store = SQLiteResultStore(
        tmp_path / "runs.sqlite3",
        tmp_path / "artifacts",
        resolved_config={"project": {"artifact_dir": tmp_path / "artifacts"}},
    )
    spec = ExperimentSpec(
        name="test",
        family="polysemanticity",
        backend="transformerlens",
        parameters={"layers": [0, 1]},
    )

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
    assert store.get_run_spec(run.id) == {
        "backend": "transformerlens",
        "description": "",
        "family": "polysemanticity",
        "name": "test",
        "parameters": {"layers": [0, 1]},
    }
    assert store.get_run_config(run.id) == {
        "project": {"artifact_dir": str(tmp_path / "artifacts")}
    }
    assert store.get_result(run.id) == ExperimentResult(
        run_id=run.id,
        status=RunStatus.SUCCEEDED,
        metrics={"score": 1.0},
        artifacts={"report": "report.md"},
    )


def test_sqlite_store_reads_created_at_from_database(tmp_path: Path) -> None:
    store = SQLiteResultStore(tmp_path / "runs.sqlite3", tmp_path / "artifacts")
    run = store.create_run(ExperimentSpec(name="test", family="family", backend="backend"))
    expected = datetime(2024, 1, 2, 3, 4, 5, tzinfo=UTC)

    with sqlite3.connect(tmp_path / "runs.sqlite3") as connection:
        connection.execute(
            "UPDATE runs SET created_at = ? WHERE id = ?",
            (expected.isoformat(), run.id),
        )

    assert store.list_runs()[0].created_at == expected


def test_sqlite_store_updates_run_status_with_transition_rules(tmp_path: Path) -> None:
    store = SQLiteResultStore(tmp_path / "runs.sqlite3", tmp_path / "artifacts")
    run = store.create_run(ExperimentSpec(name="test", family="family", backend="backend"))

    running = store.update_run_status(run.id, RunStatus.RUNNING)
    assert running.status == RunStatus.RUNNING

    succeeded = store.update_run_status(run.id, RunStatus.SUCCEEDED)
    assert succeeded.status == RunStatus.SUCCEEDED

    with pytest.raises(ValueError, match="Cannot transition"):
        store.update_run_status(run.id, RunStatus.FAILED)


def test_sqlite_store_migrates_existing_database(tmp_path: Path) -> None:
    database_path = tmp_path / "runs.sqlite3"
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            CREATE TABLE runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                spec_name TEXT NOT NULL,
                family TEXT NOT NULL,
                backend TEXT NOT NULL,
                status TEXT NOT NULL,
                artifact_dir TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE results (
                run_id INTEGER PRIMARY KEY,
                status TEXT NOT NULL,
                metrics_json TEXT NOT NULL,
                artifacts_json TEXT NOT NULL,
                notes TEXT NOT NULL,
                FOREIGN KEY(run_id) REFERENCES runs(id)
            )
            """
        )

    store = SQLiteResultStore(database_path, tmp_path / "artifacts")
    store.initialize()

    with sqlite3.connect(database_path) as connection:
        columns = {
            str(row[1])
            for row in connection.execute("PRAGMA table_info(runs)").fetchall()
        }
    assert {"spec_json", "config_json"} <= columns
