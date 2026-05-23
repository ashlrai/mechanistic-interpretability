import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from typer.testing import CliRunner

from mech_interp import cli
from mech_interp.orchestration.queue import ExperimentRunQueue
from mech_interp.storage.sqlite_store import QueueStatus, SQLiteResultStore
from mech_interp.types import ExperimentResult, ExperimentSpec, RunStatus


def test_queue_plan_is_idempotent(tmp_path: Path) -> None:
    queue = ExperimentRunQueue(SQLiteResultStore(tmp_path / "runs.sqlite3", tmp_path / "artifacts"))
    specs = [
        ExperimentSpec(name="first", family="polysemanticity", backend="transformerlens"),
        ExperimentSpec(name="second", family="superposition", backend="transformerlens"),
    ]

    first_plan = queue.plan(specs)
    second_plan = queue.plan(specs)

    assert first_plan.enqueued == 2
    assert first_plan.total == 2
    assert second_plan.enqueued == 0
    assert [item.spec_name for item in queue.list()] == ["first", "second"]
    assert {item.status for item in queue.list()} == {QueueStatus.PLANNED}


def test_queue_claim_and_mark_statuses(tmp_path: Path) -> None:
    queue = ExperimentRunQueue(SQLiteResultStore(tmp_path / "runs.sqlite3", tmp_path / "artifacts"))
    queue.plan(
        [ExperimentSpec(name="retry-me", family="polysemanticity", backend="transformerlens")]
    )

    claimed = queue.claim_next()
    assert claimed is not None
    assert claimed.spec_name == "retry-me"
    assert claimed.status == QueueStatus.RUNNING
    assert claimed.lease_token is not None

    store = queue.store
    failed = store.mark_queue_item_failed_by_lease(
        claimed.id,
        claimed.lease_token,
        "provider unavailable",
    )
    assert failed.status == QueueStatus.FAILED
    assert failed.retry_count == 1
    assert failed.error == "provider unavailable"

    retried = queue.claim_next()
    assert retried is not None
    assert retried.status == QueueStatus.RUNNING
    assert retried.retry_count == 1
    assert retried.error is None
    assert retried.lease_token is not None

    succeeded = queue.store.mark_queue_item_succeeded_by_lease(retried.id, retried.lease_token)
    assert succeeded.status == QueueStatus.SUCCEEDED
    assert succeeded.retry_count == 1
    assert succeeded.error is None
    assert queue.claim_next() is None


def test_queue_cli_plan_next_and_list(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = tmp_path / "config.yaml"
    database_path = tmp_path / "queue.sqlite3"
    artifact_dir = tmp_path / "artifacts"
    spec_dir = tmp_path / "experiments"
    spec_dir.mkdir()
    config_path.write_text(
        f"""
project:
  artifact_dir: {artifact_dir}
  database_path: {database_path}
""",
        encoding="utf-8",
    )
    (spec_dir / "queued.yaml").write_text(
        """
name: queued-cli
family: circuit_patching
backend: transformerlens
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("MECH_INTERP_CONFIG", str(config_path))

    runner = CliRunner()
    plan_result = runner.invoke(cli.app, ["queue", "plan", "--directory", str(spec_dir)])
    next_result = runner.invoke(cli.app, ["queue", "next"])
    list_result = runner.invoke(cli.app, ["queue", "list"])

    assert plan_result.exit_code == 0
    assert "Queued 1 new experiment spec" in plan_result.output
    assert next_result.exit_code == 0
    assert "queued-cli" in next_result.output
    assert list_result.exit_code == 0
    assert "queued-cli" in list_result.output
    assert "running" in list_result.output


def test_queue_run_once_marks_success_and_links_run_id(tmp_path: Path) -> None:
    store = SQLiteResultStore(tmp_path / "runs.sqlite3", tmp_path / "artifacts")
    queue = ExperimentRunQueue(store)
    spec = ExperimentSpec(name="queued", family="polysemanticity", backend="transformerlens")
    queue.plan([spec])

    class FakeRunner:
        def run(self, item: ExperimentSpec) -> ExperimentResult:
            run = store.create_run(item)
            return ExperimentResult(run_id=run.id, status=RunStatus.SUCCEEDED)

    result = queue.run_once({"queued": spec}, FakeRunner())

    assert result is not None
    [item] = queue.list()
    assert item.status == QueueStatus.SUCCEEDED
    assert item.run_id == result.run_id


def test_queue_run_once_marks_failed_result(tmp_path: Path) -> None:
    store = SQLiteResultStore(tmp_path / "runs.sqlite3", tmp_path / "artifacts")
    queue = ExperimentRunQueue(store)
    spec = ExperimentSpec(name="queued", family="polysemanticity", backend="transformerlens")
    queue.plan([spec])

    class FakeRunner:
        def run(self, item: ExperimentSpec) -> ExperimentResult:
            run = store.create_run(item)
            return ExperimentResult(run_id=run.id, status=RunStatus.FAILED, notes="boom")

    result = queue.run_once({"queued": spec}, FakeRunner())

    assert result is not None
    [item] = queue.list()
    assert item.status == QueueStatus.FAILED
    assert item.retry_count == 1
    assert item.run_id == result.run_id
    assert item.error == "boom"


def test_requeue_stale_running_item(tmp_path: Path) -> None:
    database_path = tmp_path / "runs.sqlite3"
    store = SQLiteResultStore(database_path, tmp_path / "artifacts")
    queue = ExperimentRunQueue(store)
    queue.plan([ExperimentSpec(name="queued", family="polysemanticity", backend="transformerlens")])
    queue.claim_next()
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            "UPDATE experiment_queue SET heartbeat_at = ? WHERE spec_name = ?",
            ("2000-01-01T00:00:00+00:00", "queued"),
        )

    requeued = queue.requeue_stale(1)

    assert len(requeued) == 1
    assert requeued[0].status == QueueStatus.PLANNED


def test_recent_heartbeat_prevents_stale_requeue(tmp_path: Path) -> None:
    database_path = tmp_path / "runs.sqlite3"
    store = SQLiteResultStore(database_path, tmp_path / "artifacts")
    queue = ExperimentRunQueue(store)
    queue.plan([ExperimentSpec(name="queued", family="polysemanticity", backend="transformerlens")])
    claimed = queue.claim_next()
    assert claimed is not None
    assert claimed.lease_token is not None
    store.heartbeat_queue_item(claimed.id, claimed.lease_token)
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            "UPDATE experiment_queue SET updated_at = ? WHERE id = ?",
            ("2000-01-01T00:00:00+00:00", claimed.id),
        )

    assert queue.requeue_stale(60) == []
    [item] = queue.list()
    assert item.status == QueueStatus.RUNNING


def test_stale_requeue_closes_attempt_and_records_event(tmp_path: Path) -> None:
    database_path = tmp_path / "runs.sqlite3"
    store = SQLiteResultStore(database_path, tmp_path / "artifacts")
    queue = ExperimentRunQueue(store)
    queue.plan([ExperimentSpec(name="queued", family="polysemanticity", backend="transformerlens")])
    claimed = queue.claim_next()
    assert claimed is not None
    assert claimed.attempt_id is not None
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            "UPDATE experiment_queue SET heartbeat_at = ? WHERE id = ?",
            ("2000-01-01T00:00:00+00:00", claimed.id),
        )

    queue.requeue_stale(1)

    with sqlite3.connect(database_path) as connection:
        attempt = connection.execute(
            "SELECT status, error, finished_at FROM queue_attempts WHERE id = ?",
            (claimed.attempt_id,),
        ).fetchone()
    assert attempt[0] == "stale"
    assert attempt[1] == "Requeued stale running item."
    assert attempt[2] is not None
    [event] = [
        event
        for event in store.list_run_events(limit=10)
        if event.event_type == "stale_requeued"
    ]
    assert event.queue_id == claimed.id
    assert event.attempt_id == claimed.attempt_id


def test_queue_rejects_late_completion_after_stale_requeue(tmp_path: Path) -> None:
    database_path = tmp_path / "runs.sqlite3"
    store = SQLiteResultStore(database_path, tmp_path / "artifacts")
    queue = ExperimentRunQueue(store)
    queue.plan([ExperimentSpec(name="queued", family="polysemanticity", backend="transformerlens")])
    claimed = queue.claim_next()
    assert claimed is not None
    assert claimed.lease_token is not None
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            "UPDATE experiment_queue SET heartbeat_at = ? WHERE id = ?",
            ("2000-01-01T00:00:00+00:00", claimed.id),
        )

    queue.requeue_stale(1)

    with pytest.raises(RuntimeError, match="active lease"):
        store.mark_queue_item_succeeded_by_lease(claimed.id, claimed.lease_token)


def test_queue_rejects_late_failure_after_new_lease(tmp_path: Path) -> None:
    database_path = tmp_path / "runs.sqlite3"
    store = SQLiteResultStore(database_path, tmp_path / "artifacts")
    queue = ExperimentRunQueue(store)
    queue.plan([ExperimentSpec(name="queued", family="polysemanticity", backend="transformerlens")])
    first = queue.claim_next()
    assert first is not None
    assert first.lease_token is not None
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            "UPDATE experiment_queue SET heartbeat_at = ? WHERE id = ?",
            ("2000-01-01T00:00:00+00:00", first.id),
        )

    queue.requeue_stale(1)
    second = queue.claim_next()
    assert second is not None
    assert second.lease_token is not None
    assert second.lease_token != first.lease_token

    with pytest.raises(RuntimeError, match="active lease"):
        store.mark_queue_item_failed_by_lease(first.id, first.lease_token, "late failure")
    succeeded = store.mark_queue_item_succeeded_by_lease(second.id, second.lease_token)
    assert succeeded.status == QueueStatus.SUCCEEDED


def test_queue_attempts_and_events_follow_lease_lifecycle(tmp_path: Path) -> None:
    database_path = tmp_path / "runs.sqlite3"
    store = SQLiteResultStore(database_path, tmp_path / "artifacts")
    queue = ExperimentRunQueue(store)
    spec = ExperimentSpec(name="queued", family="polysemanticity", backend="transformerlens")
    queue.plan([spec])
    claimed = queue.claim_next(worker_id="worker-a")
    assert claimed is not None
    assert claimed.lease_token is not None
    assert claimed.attempt_id is not None
    run = store.create_run(spec)

    store.start_queue_attempt(claimed.id, claimed.lease_token, run.id)
    store.heartbeat_queue_item(claimed.id, claimed.lease_token)
    store.mark_queue_item_succeeded_by_lease(claimed.id, claimed.lease_token, run.id)

    with sqlite3.connect(database_path) as connection:
        attempt = connection.execute(
            """
            SELECT status, run_id, started_at, heartbeat_at, finished_at
            FROM queue_attempts
            WHERE id = ?
            """,
            (claimed.attempt_id,),
        ).fetchone()
    assert attempt[0] == QueueStatus.SUCCEEDED.value
    assert attempt[1] == run.id
    assert attempt[2] is not None
    assert attempt[3] is not None
    assert attempt[4] is not None
    events = [
        event.event_type
        for event in reversed(store.list_run_events(limit=10))
        if event.attempt_id == claimed.attempt_id
    ]
    assert events == ["claimed", "started", "heartbeat", "succeeded"]


def test_concurrent_claims_are_unique(tmp_path: Path) -> None:
    store = SQLiteResultStore(tmp_path / "runs.sqlite3", tmp_path / "artifacts")
    queue = ExperimentRunQueue(store)
    specs = [
        ExperimentSpec(name=f"queued-{index}", family="polysemanticity", backend="transformerlens")
        for index in range(8)
    ]
    queue.plan(specs)

    def claim(index: int) -> str | None:
        item = ExperimentRunQueue(store).claim_next(worker_id=f"worker-{index}")
        return item.spec_name if item is not None else None

    with ThreadPoolExecutor(max_workers=12) as executor:
        claimed = list(executor.map(claim, range(12)))

    claimed_names = [name for name in claimed if name is not None]
    assert len(claimed_names) == len(specs)
    assert sorted(claimed_names) == [spec.name for spec in specs]
    assert queue.claim_next() is None


def test_queue_pause_resume_cancel_and_requeue_by_id(tmp_path: Path) -> None:
    store = SQLiteResultStore(tmp_path / "runs.sqlite3", tmp_path / "artifacts")
    queue = ExperimentRunQueue(store)
    queue.plan([ExperimentSpec(name="queued", family="polysemanticity", backend="transformerlens")])
    [item] = queue.list()

    paused = queue.pause(item.id)
    assert paused.status == QueueStatus.PAUSED
    assert queue.claim_next() is None

    resumed = queue.resume(item.id)
    assert resumed.status == QueueStatus.PLANNED

    cancelled = queue.cancel(item.id)
    assert cancelled.status == QueueStatus.CANCELLED
    assert cancelled.cancelled_at is not None

    requeued = queue.requeue(item.id)
    assert requeued.status == QueueStatus.PLANNED


def test_queue_enforces_max_retries(tmp_path: Path) -> None:
    store = SQLiteResultStore(tmp_path / "runs.sqlite3", tmp_path / "artifacts")
    queue = ExperimentRunQueue(store)
    spec = ExperimentSpec(
        name="queued",
        family="polysemanticity",
        backend="transformerlens",
        parameters={"max_retries": 1},
    )
    queue.plan([spec])
    claimed = queue.claim_next()
    assert claimed is not None
    assert claimed.lease_token is not None

    failed = store.mark_queue_item_failed_by_lease(
        claimed.id,
        claimed.lease_token,
        "failed once",
    )

    assert failed.status == QueueStatus.FAILED
    assert failed.retry_count == 1
    assert queue.claim_next() is None
