from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, cast

from mech_interp.types import ExperimentResult, ExperimentRun, ExperimentSpec, RunStatus, utc_now


class QueueStatus(StrEnum):
    PLANNED = "planned"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass(frozen=True)
class ExperimentQueueItem:
    id: int
    spec_name: str
    status: QueueStatus
    retry_count: int
    error: str | None
    created_at: datetime
    updated_at: datetime


class SQLiteResultStore:
    def __init__(
        self,
        database_path: str | Path,
        artifact_dir: str | Path,
        resolved_config: Mapping[str, Any] | None = None,
    ) -> None:
        self.database_path = Path(database_path)
        self.artifact_dir = Path(artifact_dir)
        self.resolved_config = dict(resolved_config or {})

    def initialize(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS runs (
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
                CREATE TABLE IF NOT EXISTS results (
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
                CREATE TABLE IF NOT EXISTS experiment_queue (
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
            self._migrate(connection)

    def create_run(self, spec: ExperimentSpec) -> ExperimentRun:
        self.initialize()
        created_at = utc_now()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO runs (
                    spec_name,
                    family,
                    backend,
                    status,
                    artifact_dir,
                    spec_json,
                    config_json,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    spec.name,
                    spec.family,
                    spec.backend,
                    RunStatus.PLANNED.value,
                    str(self.artifact_dir),
                    self._json_dumps(asdict(spec)),
                    self._json_dumps(self.resolved_config),
                    created_at.isoformat(),
                ),
            )
            if cursor.lastrowid is None:
                raise RuntimeError("SQLite did not return a run id.")
            run_id = cursor.lastrowid
        return ExperimentRun(
            id=run_id,
            spec_name=spec.name,
            family=spec.family,
            backend=spec.backend,
            status=RunStatus.PLANNED,
            artifact_dir=self.artifact_dir,
            created_at=created_at,
        )

    def save_result(self, result: ExperimentResult) -> None:
        self.initialize()
        with self._connect() as connection:
            self._transition_status(connection, result.run_id, result.status)
            connection.execute(
                """
                INSERT OR REPLACE INTO results
                    (run_id, status, metrics_json, artifacts_json, notes)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    result.run_id,
                    result.status.value,
                    json.dumps(result.metrics, sort_keys=True),
                    json.dumps(result.artifacts, sort_keys=True),
                    result.notes,
                ),
            )

    def list_runs(self, limit: int = 20) -> list[ExperimentRun]:
        self.initialize()
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, spec_name, family, backend, status, artifact_dir, created_at
                FROM runs
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [
            self._run_from_row(row)
            for row in rows
        ]

    def get_result(self, run_id: int) -> ExperimentResult | None:
        self.initialize()
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT run_id, status, metrics_json, artifacts_json, notes
                FROM results
                WHERE run_id = ?
                """,
                (run_id,),
            ).fetchone()
        if row is None:
            return None
        return ExperimentResult(
            run_id=int(row[0]),
            status=RunStatus(str(row[1])),
            metrics=cast(dict[str, float], self._json_loads_object(str(row[2]))),
            artifacts=cast(dict[str, str], self._json_loads_object(str(row[3]))),
            notes=str(row[4]),
        )

    def get_run_spec(self, run_id: int) -> dict[str, Any] | None:
        return self._get_run_json_column(run_id, "spec_json")

    def get_run_config(self, run_id: int) -> dict[str, Any] | None:
        return self._get_run_json_column(run_id, "config_json")

    def update_run_status(self, run_id: int, status: RunStatus) -> ExperimentRun:
        self.initialize()
        with self._connect() as connection:
            self._transition_status(connection, run_id, status)
            row = self._get_run_row(connection, run_id)
        return self._run_from_row(row)

    def enqueue_experiment_specs(self, specs: list[ExperimentSpec]) -> int:
        self.initialize()
        now = utc_now().isoformat()
        with self._connect() as connection:
            before = connection.total_changes
            connection.executemany(
                """
                INSERT OR IGNORE INTO experiment_queue (
                    spec_name,
                    status,
                    retry_count,
                    error,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, 0, NULL, ?, ?)
                """,
                [
                    (spec.name, QueueStatus.PLANNED.value, now, now)
                    for spec in specs
                ],
            )
            return connection.total_changes - before

    def claim_next_queue_item(self) -> ExperimentQueueItem | None:
        self.initialize()
        now = utc_now().isoformat()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT id, spec_name, status, retry_count, error, created_at, updated_at
                FROM experiment_queue
                WHERE status IN (?, ?)
                ORDER BY retry_count ASC, id ASC
                LIMIT 1
                """,
                (QueueStatus.PLANNED.value, QueueStatus.FAILED.value),
            ).fetchone()
            if row is None:
                return None
            queue_id = int(row[0])
            connection.execute(
                """
                UPDATE experiment_queue
                SET status = ?, error = NULL, updated_at = ?
                WHERE id = ?
                """,
                (QueueStatus.RUNNING.value, now, queue_id),
            )
            updated = connection.execute(
                """
                SELECT id, spec_name, status, retry_count, error, created_at, updated_at
                FROM experiment_queue
                WHERE id = ?
                """,
                (queue_id,),
            ).fetchone()
        return self._queue_item_from_row(cast(tuple[Any, ...], updated))

    def mark_queue_item_succeeded(self, spec_name: str) -> ExperimentQueueItem:
        return self._mark_queue_item(spec_name, QueueStatus.SUCCEEDED, error=None)

    def mark_queue_item_failed(self, spec_name: str, error: str) -> ExperimentQueueItem:
        return self._mark_queue_item(spec_name, QueueStatus.FAILED, error=error)

    def list_queue_items(self) -> list[ExperimentQueueItem]:
        self.initialize()
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, spec_name, status, retry_count, error, created_at, updated_at
                FROM experiment_queue
                ORDER BY id ASC
                """
            ).fetchall()
        return [self._queue_item_from_row(cast(tuple[Any, ...], row)) for row in rows]

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _migrate(self, connection: sqlite3.Connection) -> None:
        columns = {
            str(row[1])
            for row in connection.execute("PRAGMA table_info(runs)").fetchall()
        }
        if "spec_json" not in columns:
            connection.execute("ALTER TABLE runs ADD COLUMN spec_json TEXT NOT NULL DEFAULT '{}'")
        if "config_json" not in columns:
            connection.execute("ALTER TABLE runs ADD COLUMN config_json TEXT NOT NULL DEFAULT '{}'")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS experiment_queue (
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

    def _transition_status(
        self,
        connection: sqlite3.Connection,
        run_id: int,
        next_status: RunStatus,
    ) -> None:
        row = connection.execute("SELECT status FROM runs WHERE id = ?", (run_id,)).fetchone()
        if row is None:
            raise KeyError(f"Run {run_id} does not exist.")
        current_status = RunStatus(str(row[0]))
        allowed_transitions: dict[RunStatus, set[RunStatus]] = {
            RunStatus.PLANNED: {
                RunStatus.PLANNED,
                RunStatus.RUNNING,
                RunStatus.SUCCEEDED,
                RunStatus.FAILED,
            },
            RunStatus.RUNNING: {RunStatus.RUNNING, RunStatus.SUCCEEDED, RunStatus.FAILED},
            RunStatus.SUCCEEDED: {RunStatus.SUCCEEDED},
            RunStatus.FAILED: {RunStatus.FAILED},
        }
        if next_status not in allowed_transitions[current_status]:
            raise ValueError(
                f"Cannot transition run {run_id} from "
                f"{current_status.value} to {next_status.value}."
            )
        if next_status != current_status:
            connection.execute(
                "UPDATE runs SET status = ? WHERE id = ?",
                (next_status.value, run_id),
            )

    def _get_run_json_column(self, run_id: int, column: str) -> dict[str, Any] | None:
        if column not in {"spec_json", "config_json"}:
            raise ValueError(f"Unsupported run JSON column: {column}")
        self.initialize()
        with self._connect() as connection:
            row = connection.execute(
                f"SELECT {column} FROM runs WHERE id = ?",
                (run_id,),
            ).fetchone()
        if row is None:
            return None
        return self._json_loads_object(str(row[0]))

    def _get_run_row(self, connection: sqlite3.Connection, run_id: int) -> tuple[Any, ...]:
        row = connection.execute(
            """
            SELECT id, spec_name, family, backend, status, artifact_dir, created_at
            FROM runs
            WHERE id = ?
            """,
            (run_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"Run {run_id} does not exist.")
        return cast(tuple[Any, ...], row)

    def _mark_queue_item(
        self,
        spec_name: str,
        status: QueueStatus,
        error: str | None,
    ) -> ExperimentQueueItem:
        self.initialize()
        now = utc_now().isoformat()
        with self._connect() as connection:
            row = connection.execute(
                "SELECT id, status FROM experiment_queue WHERE spec_name = ?",
                (spec_name,),
            ).fetchone()
            if row is None:
                raise KeyError(f"Queue item for spec '{spec_name}' does not exist.")
            if status == QueueStatus.FAILED:
                connection.execute(
                    """
                    UPDATE experiment_queue
                    SET status = ?, retry_count = retry_count + 1, error = ?, updated_at = ?
                    WHERE spec_name = ?
                    """,
                    (status.value, error, now, spec_name),
                )
            else:
                connection.execute(
                    """
                    UPDATE experiment_queue
                    SET status = ?, error = ?, updated_at = ?
                    WHERE spec_name = ?
                    """,
                    (status.value, error, now, spec_name),
                )
            updated = connection.execute(
                """
                SELECT id, spec_name, status, retry_count, error, created_at, updated_at
                FROM experiment_queue
                WHERE spec_name = ?
                """,
                (spec_name,),
            ).fetchone()
        return self._queue_item_from_row(cast(tuple[Any, ...], updated))

    def _run_from_row(self, row: tuple[Any, ...]) -> ExperimentRun:
        return ExperimentRun(
            id=int(row[0]),
            spec_name=str(row[1]),
            family=str(row[2]),
            backend=str(row[3]),
            status=RunStatus(str(row[4])),
            artifact_dir=Path(str(row[5])),
            created_at=self._parse_datetime(str(row[6])),
        )

    def _queue_item_from_row(self, row: tuple[Any, ...]) -> ExperimentQueueItem:
        return ExperimentQueueItem(
            id=int(row[0]),
            spec_name=str(row[1]),
            status=QueueStatus(str(row[2])),
            retry_count=int(row[3]),
            error=None if row[4] is None else str(row[4]),
            created_at=self._parse_datetime(str(row[5])),
            updated_at=self._parse_datetime(str(row[6])),
        )

    def _json_dumps(self, payload: Mapping[str, Any]) -> str:
        return json.dumps(payload, default=str, sort_keys=True)

    def _json_loads_object(self, payload: str) -> dict[str, Any]:
        decoded = json.loads(payload)
        if not isinstance(decoded, dict):
            raise ValueError("Expected a JSON object.")
        return cast(dict[str, Any], decoded)

    def _parse_datetime(self, payload: str) -> datetime:
        parsed = datetime.fromisoformat(payload)
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=UTC)
        return parsed
