from pathlib import Path

import pytest
from typer.testing import CliRunner

from mech_interp import cli
from mech_interp.orchestration.queue import ExperimentRunQueue
from mech_interp.storage.sqlite_store import QueueStatus, SQLiteResultStore
from mech_interp.types import ExperimentSpec


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

    failed = queue.mark_failed("retry-me", "provider unavailable")
    assert failed.status == QueueStatus.FAILED
    assert failed.retry_count == 1
    assert failed.error == "provider unavailable"

    retried = queue.claim_next()
    assert retried is not None
    assert retried.status == QueueStatus.RUNNING
    assert retried.retry_count == 1
    assert retried.error is None

    succeeded = queue.mark_succeeded("retry-me")
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
