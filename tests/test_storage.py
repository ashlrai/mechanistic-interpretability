import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from mech_interp.storage import SQLiteResultStore, resolve_run_artifact_dir
from mech_interp.types import ExperimentResult, ExperimentRun, ExperimentSpec, RunStatus


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


def test_sqlite_store_migrates_partial_v2_queue_tables(tmp_path: Path) -> None:
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
                spec_json TEXT NOT NULL DEFAULT '{}',
                config_json TEXT NOT NULL DEFAULT '{}',
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
        connection.execute(
            """
            CREATE TABLE experiment_queue (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                spec_name TEXT NOT NULL UNIQUE,
                status TEXT NOT NULL,
                retry_count INTEGER NOT NULL DEFAULT 0,
                error TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            "CREATE TABLE queue_attempts (id INTEGER PRIMARY KEY AUTOINCREMENT, queue_id INTEGER)"
        )
        connection.execute(
            "CREATE TABLE run_events (id INTEGER PRIMARY KEY AUTOINCREMENT, event_type TEXT)"
        )

    store = SQLiteResultStore(database_path, tmp_path / "artifacts")
    store.initialize()

    with sqlite3.connect(database_path) as connection:
        queue_attempt_columns = {
            str(row[1])
            for row in connection.execute("PRAGMA table_info(queue_attempts)").fetchall()
        }
        event_columns = {
            str(row[1])
            for row in connection.execute("PRAGMA table_info(run_events)").fetchall()
        }
        user_version = connection.execute("PRAGMA user_version").fetchone()[0]
    assert {
        "lease_token",
        "worker_id",
        "status",
        "claimed_at",
        "heartbeat_at",
        "started_at",
        "finished_at",
        "error",
        "run_id",
    } <= queue_attempt_columns
    assert {"run_id", "queue_id", "attempt_id", "message", "payload_json", "created_at"} <= (
        event_columns
    )
    assert user_version == 2


def test_resolve_run_artifact_dir_returns_artifact_dir_as_is(tmp_path: Path) -> None:
    """resolve_run_artifact_dir must never nest a run-NNNNNN subdir."""
    run = ExperimentRun(
        id=1,
        spec_name="test",
        family="polysemanticity",
        backend="transformerlens",
        status=RunStatus.PLANNED,
        artifact_dir=tmp_path,
        created_at=datetime(2024, 1, 1, tzinfo=UTC),
    )
    result = resolve_run_artifact_dir(run)
    # Always trusts the caller — no sub-dir appended.
    assert result == tmp_path


def test_resolve_run_artifact_dir_flat_path_does_not_nest(tmp_path: Path) -> None:
    """Passing a flat tmp dir must not produce parent/run-000001/run-000001."""
    run = ExperimentRun(
        id=1,
        spec_name="test",
        family="circuit_patching",
        backend="transformerlens",
        status=RunStatus.PLANNED,
        artifact_dir=tmp_path,
        created_at=datetime(2024, 1, 1, tzinfo=UTC),
    )
    result = resolve_run_artifact_dir(run)
    # Must not equal tmp_path / "run-000001"
    assert result == tmp_path
    assert result != tmp_path / "run-000001"


def test_resolve_run_artifact_dir_named_dir(tmp_path: Path) -> None:
    """Passing the already-correct run-NNNNNN dir returns it unchanged."""
    named = tmp_path / "run-000026"
    run = ExperimentRun(
        id=26,
        spec_name="test",
        family="acdc_lite",
        backend="transformerlens",
        status=RunStatus.SUCCEEDED,
        artifact_dir=named,
        created_at=datetime(2024, 1, 1, tzinfo=UTC),
    )
    result = resolve_run_artifact_dir(run)
    assert result == named


def test_sqlite_store_archive_runs(tmp_path: Path) -> None:
    store = SQLiteResultStore(tmp_path / "runs.sqlite3", tmp_path / "artifacts")
    spec = ExperimentSpec(name="test", family="polysemanticity", backend="transformerlens")
    run = store.create_run(spec)

    # Create the artifact directory so rename can be tested.
    run_dir = tmp_path / "artifacts" / f"run-{run.id:06d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "result.json").write_text("{}", encoding="utf-8")

    archived = store.archive_runs([run.id], tmp_path / "artifacts")
    assert archived == [run.id]

    # Artifact dir moved.
    dest = tmp_path / "artifacts" / "archived" / f"run-{run.id:06d}"
    assert dest.is_dir()
    assert not run_dir.exists()

    # Archived run is excluded from default list_runs.
    active_runs = store.list_runs(limit=100)
    assert not any(r.id == run.id for r in active_runs)

    # But visible with include_archived=True.
    all_runs = store.list_runs(limit=100, include_archived=True)
    assert any(r.id == run.id for r in all_runs)


def test_sqlite_store_archive_runs_missing_dir(tmp_path: Path) -> None:
    """archive_runs stamps the DB even if the artifact directory is absent."""
    store = SQLiteResultStore(tmp_path / "runs.sqlite3", tmp_path / "artifacts")
    spec = ExperimentSpec(name="test", family="superposition", backend="transformerlens")
    run = store.create_run(spec)

    # Do NOT create the directory — it's already missing.
    archived = store.archive_runs([run.id], tmp_path / "artifacts")
    assert archived == [run.id]

    active_runs = store.list_runs(limit=100)
    assert not any(r.id == run.id for r in active_runs)


def test_sqlite_store_list_placeholder_runs_before(tmp_path: Path) -> None:
    store = SQLiteResultStore(tmp_path / "runs.sqlite3", tmp_path / "artifacts")

    poly_run = store.create_run(
        ExperimentSpec(name="poly", family="polysemanticity", backend="b")
    )
    super_run = store.create_run(
        ExperimentSpec(name="super", family="superposition", backend="b")
    )
    real_run = store.create_run(
        ExperimentSpec(name="real", family="circuit_patching", backend="b")
    )

    # Only poly and super should appear, and only when id < threshold.
    before_all = store.list_placeholder_runs_before(real_run.id + 1)
    ids = {r.id for r in before_all}
    assert poly_run.id in ids
    assert super_run.id in ids
    assert real_run.id not in ids

    # Threshold cuts off runs at or above it.
    before_first = store.list_placeholder_runs_before(poly_run.id)
    assert before_first == []
