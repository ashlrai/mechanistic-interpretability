from __future__ import annotations

import json
import sqlite3
import uuid
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, cast

from mech_interp.types import ExperimentResult, ExperimentRun, ExperimentSpec, RunStatus, utc_now


class QueueStatus(StrEnum):
    PLANNED = "planned"
    PAUSED = "paused"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class ExperimentQueueItem:
    id: int
    spec_name: str
    status: QueueStatus
    retry_count: int
    error: str | None
    run_id: int | None
    created_at: datetime
    updated_at: datetime
    lease_token: str | None = None
    worker_id: str | None = None
    attempt_id: int | None = None
    claimed_at: datetime | None = None
    heartbeat_at: datetime | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    max_retries: int = 2
    priority: int = 0
    cancelled_at: datetime | None = None
    current_phase: str | None = None
    spec_json: str = "{}"
    spec_sha256: str | None = None
    source_path: str | None = None
    dataset_hashes: str = "{}"
    tags: str = "[]"
    hypothesis: str | None = None
    matrix_id: int | None = None


@dataclass(frozen=True)
class RunEvent:
    id: int
    run_id: int | None
    queue_id: int | None
    attempt_id: int | None
    event_type: str
    message: str
    payload: dict[str, Any]
    created_at: datetime


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
                    spec_sha256 TEXT,
                    source_path TEXT,
                    dataset_hashes TEXT NOT NULL DEFAULT '{}',
                    tags TEXT NOT NULL DEFAULT '[]',
                    hypothesis TEXT,
                    matrix_id INTEGER,
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
                    status TEXT NOT NULL CHECK(
                        status IN (
                            'planned',
                            'paused',
                            'running',
                            'succeeded',
                            'failed',
                            'cancelled'
                        )
                    ),
                    retry_count INTEGER NOT NULL DEFAULT 0,
                    error TEXT,
                    run_id INTEGER,
                    lease_token TEXT,
                    worker_id TEXT,
                    attempt_id INTEGER,
                    claimed_at TEXT,
                    heartbeat_at TEXT,
                    started_at TEXT,
                    finished_at TEXT,
                    max_retries INTEGER NOT NULL DEFAULT 2,
                    priority INTEGER NOT NULL DEFAULT 0,
                    cancelled_at TEXT,
                    current_phase TEXT,
                    spec_json TEXT NOT NULL DEFAULT '{}',
                    spec_sha256 TEXT,
                    source_path TEXT,
                    dataset_hashes TEXT NOT NULL DEFAULT '{}',
                    tags TEXT NOT NULL DEFAULT '[]',
                    hypothesis TEXT,
                    matrix_id INTEGER,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(run_id) REFERENCES runs(id)
                )
                """
            )
            self._migrate(connection)
            self._create_indexes(connection)
            connection.execute("PRAGMA user_version = 2")

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
                    spec_sha256,
                    source_path,
                    dataset_hashes,
                    tags,
                    hypothesis,
                    matrix_id,
                    config_json,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    spec.name,
                    spec.family,
                    spec.backend,
                    RunStatus.PLANNED.value,
                    str(self.artifact_dir),
                    self._spec_json(spec),
                    self._spec_sha256(spec),
                    None,
                    self._json_dumps(_dataset_hashes(spec)),
                    self._json_dumps(_tags(spec)),
                    _hypothesis(spec),
                    None,
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
            connection.execute("DELETE FROM run_metrics WHERE run_id = ?", (result.run_id,))
            connection.executemany(
                """
                INSERT INTO run_metrics (run_id, key, value)
                VALUES (?, ?, ?)
                """,
                [
                    (result.run_id, key, float(value))
                    for key, value in result.metrics.items()
                    if isinstance(value, int | float)
                ],
            )
            connection.execute("DELETE FROM run_artifacts WHERE run_id = ?", (result.run_id,))
            connection.executemany(
                """
                INSERT INTO run_artifacts (run_id, name, path, media_type, sha256, size_bytes)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        result.run_id,
                        name,
                        path,
                        _media_type_from_path(path),
                        None,
                        _path_size(path),
                    )
                    for name, path in result.artifacts.items()
                ],
            )
            self._index_science_artifacts(connection, result.run_id, result.artifacts)
            self.append_event(
                connection,
                "succeeded" if result.status == RunStatus.SUCCEEDED else "failed",
                run_id=result.run_id,
                message=result.notes,
                payload={"metrics": result.metrics, "artifacts": result.artifacts},
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
                    run_id,
                    max_retries,
                    priority,
                    spec_json,
                    spec_sha256,
                    dataset_hashes,
                    tags,
                    hypothesis,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, 0, NULL, NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        spec.name,
                        QueueStatus.PLANNED.value,
                        int(spec.parameters.get("max_retries", 2)),
                        int(spec.parameters.get("priority", 0)),
                        self._spec_json(spec),
                        self._spec_sha256(spec),
                        self._json_dumps(_dataset_hashes(spec)),
                        self._json_dumps(_tags(spec)),
                        _hypothesis(spec),
                        now,
                        now,
                    )
                    for spec in specs
                ],
            )
            return connection.total_changes - before

    def claim_next_queue_item(self, worker_id: str | None = None) -> ExperimentQueueItem | None:
        self.initialize()
        now = utc_now().isoformat()
        token = uuid.uuid4().hex
        worker = worker_id or f"worker-{uuid.uuid4().hex[:8]}"
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT id
                FROM experiment_queue
                WHERE status = ?
                   OR (status = ? AND retry_count < max_retries)
                ORDER BY priority DESC, retry_count ASC, id ASC
                LIMIT 1
                """,
                (QueueStatus.PLANNED.value, QueueStatus.FAILED.value),
            ).fetchone()
            if row is None:
                return None
            queue_id = int(row[0])
            attempt_cursor = connection.execute(
                """
                INSERT INTO queue_attempts (
                    queue_id, lease_token, worker_id, status, claimed_at, heartbeat_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (queue_id, token, worker, QueueStatus.RUNNING.value, now, now),
            )
            if attempt_cursor.lastrowid is None:
                raise RuntimeError("SQLite did not return a queue attempt id.")
            attempt_id = int(attempt_cursor.lastrowid)
            connection.execute(
                """
                UPDATE experiment_queue
                SET status = ?,
                    error = NULL,
                    lease_token = ?,
                    worker_id = ?,
                    attempt_id = ?,
                    claimed_at = ?,
                    heartbeat_at = ?,
                    started_at = NULL,
                    finished_at = NULL,
                    current_phase = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (
                    QueueStatus.RUNNING.value,
                    token,
                    worker,
                    attempt_id,
                    now,
                    now,
                    "claimed",
                    now,
                    queue_id,
                ),
            )
            self.append_event(
                connection,
                "claimed",
                queue_id=queue_id,
                attempt_id=attempt_id,
                message=f"Queue item {queue_id} claimed by {worker}.",
                payload={"worker_id": worker},
            )
            updated = connection.execute(
                self._queue_select_sql("WHERE id = ?"),
                (queue_id,),
            ).fetchone()
        return self._queue_item_from_row(cast(tuple[Any, ...], updated))

    def mark_queue_item_succeeded(
        self,
        spec_name: str,
        run_id: int | None = None,
    ) -> ExperimentQueueItem:
        return self._mark_queue_item(
            spec_name,
            QueueStatus.SUCCEEDED,
            error=None,
            run_id=run_id,
        )

    def mark_queue_item_succeeded_by_lease(
        self,
        queue_id: int,
        lease_token: str,
        run_id: int | None = None,
    ) -> ExperimentQueueItem:
        return self._mark_queue_item_by_lease(
            queue_id,
            lease_token,
            QueueStatus.SUCCEEDED,
            error=None,
            run_id=run_id,
        )

    def mark_queue_item_failed(
        self,
        spec_name: str,
        error: str,
        run_id: int | None = None,
    ) -> ExperimentQueueItem:
        return self._mark_queue_item(
            spec_name,
            QueueStatus.FAILED,
            error=error,
            run_id=run_id,
        )

    def mark_queue_item_failed_by_lease(
        self,
        queue_id: int,
        lease_token: str,
        error: str,
        run_id: int | None = None,
    ) -> ExperimentQueueItem:
        return self._mark_queue_item_by_lease(
            queue_id,
            lease_token,
            QueueStatus.FAILED,
            error=error,
            run_id=run_id,
        )

    def start_queue_attempt(
        self,
        queue_id: int,
        lease_token: str,
        run_id: int,
    ) -> ExperimentQueueItem:
        return self.update_queue_phase(
            queue_id,
            lease_token,
            "started",
            run_id=run_id,
            event_type="started",
        )

    def update_queue_phase(
        self,
        queue_id: int,
        lease_token: str,
        phase: str,
        run_id: int | None = None,
        event_type: str = "phase_changed",
    ) -> ExperimentQueueItem:
        self.initialize()
        now = utc_now().isoformat()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._assert_active_lease(connection, queue_id, lease_token)
            connection.execute(
                """
                UPDATE experiment_queue
                SET current_phase = ?,
                    heartbeat_at = ?,
                    started_at = COALESCE(started_at, ?),
                    run_id = COALESCE(?, run_id),
                    updated_at = ?
                WHERE id = ?
                """,
                (phase, now, now if phase == "started" else None, run_id, now, queue_id),
            )
            row = connection.execute(
                "SELECT attempt_id, run_id FROM experiment_queue WHERE id = ?",
                (queue_id,),
            ).fetchone()
            attempt_id = int(row[0]) if row and row[0] is not None else None
            effective_run_id = int(row[1]) if row and row[1] is not None else None
            connection.execute(
                """
                UPDATE queue_attempts
                SET heartbeat_at = ?,
                    started_at = COALESCE(started_at, ?),
                    run_id = COALESCE(?, run_id)
                WHERE id = ?
                """,
                (
                    now,
                    now if phase == "started" else None,
                    effective_run_id,
                    attempt_id,
                ),
            )
            self.append_event(
                connection,
                event_type,
                run_id=effective_run_id,
                queue_id=queue_id,
                attempt_id=attempt_id,
                message=f"Queue item {queue_id} phase: {phase}.",
                payload={"phase": phase},
            )
            updated = connection.execute(
                self._queue_select_sql("WHERE id = ?"),
                (queue_id,),
            ).fetchone()
        return self._queue_item_from_row(cast(tuple[Any, ...], updated))

    def heartbeat_queue_item(self, queue_id: int, lease_token: str) -> ExperimentQueueItem:
        self.initialize()
        now = utc_now().isoformat()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._assert_active_lease(connection, queue_id, lease_token)
            connection.execute(
                "UPDATE experiment_queue SET heartbeat_at = ?, updated_at = ? WHERE id = ?",
                (now, now, queue_id),
            )
            row = connection.execute(
                "SELECT attempt_id, run_id FROM experiment_queue WHERE id = ?",
                (queue_id,),
            ).fetchone()
            attempt_id = int(row[0]) if row and row[0] is not None else None
            connection.execute(
                "UPDATE queue_attempts SET heartbeat_at = ? WHERE id = ?",
                (now, attempt_id),
            )
            self.append_event(
                connection,
                "heartbeat",
                run_id=int(row[1]) if row and row[1] is not None else None,
                queue_id=queue_id,
                attempt_id=attempt_id,
                message=f"Queue item {queue_id} heartbeat.",
            )
            updated = connection.execute(
                self._queue_select_sql("WHERE id = ?"),
                (queue_id,),
            ).fetchone()
        return self._queue_item_from_row(cast(tuple[Any, ...], updated))

    def list_queue_items(self) -> list[ExperimentQueueItem]:
        self.initialize()
        with self._connect() as connection:
            rows = connection.execute(self._queue_select_sql("ORDER BY id ASC")).fetchall()
        return [self._queue_item_from_row(cast(tuple[Any, ...], row)) for row in rows]

    def requeue_stale_items(self, stale_after_seconds: int) -> list[ExperimentQueueItem]:
        self.initialize()
        now_dt = utc_now()
        cutoff = now_dt.timestamp() - stale_after_seconds
        now = now_dt.isoformat()
        stale_rows: list[tuple[int, int | None, int | None]] = []
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute(
                """
                SELECT id, attempt_id, run_id, COALESCE(heartbeat_at, updated_at)
                FROM experiment_queue
                WHERE status = ?
                """,
                (QueueStatus.RUNNING.value,),
            ).fetchall()
            for row in rows:
                if self._parse_datetime(str(row[3])).timestamp() <= cutoff:
                    stale_rows.append(
                        (
                            int(row[0]),
                            int(row[1]) if row[1] is not None else None,
                            int(row[2]) if row[2] is not None else None,
                        )
                    )
            requeued_ids = [row[0] for row in stale_rows]
            if not requeued_ids:
                return []
            placeholders = ", ".join("?" for _ in requeued_ids)
            stale_attempt_ids = [row[1] for row in stale_rows if row[1] is not None]
            if stale_attempt_ids:
                attempt_placeholders = ", ".join("?" for _ in stale_attempt_ids)
                connection.execute(
                    f"""
                    UPDATE queue_attempts
                    SET status = ?,
                        error = ?,
                        heartbeat_at = ?,
                        finished_at = ?
                    WHERE id IN ({attempt_placeholders})
                    """,
                    (
                        "stale",
                        "Requeued stale running item.",
                        now,
                        now,
                        *stale_attempt_ids,
                    ),
                )
            for queue_id, attempt_id, run_id in stale_rows:
                self.append_event(
                    connection,
                    "stale_requeued",
                    run_id=run_id,
                    queue_id=queue_id,
                    attempt_id=attempt_id,
                    message=f"Queue item {queue_id} requeued after stale heartbeat.",
                    payload={"stale_after_seconds": stale_after_seconds},
                )
            connection.execute(
                f"""
                UPDATE experiment_queue
                SET status = ?,
                    error = ?,
                    lease_token = NULL,
                    worker_id = NULL,
                    attempt_id = NULL,
                    claimed_at = NULL,
                    heartbeat_at = NULL,
                    started_at = NULL,
                    finished_at = NULL,
                    current_phase = NULL,
                    updated_at = ?
                WHERE id IN ({placeholders})
                """,
                (
                    QueueStatus.PLANNED.value,
                    "Requeued stale running item.",
                    now,
                    *requeued_ids,
                ),
            )
            updated = connection.execute(
                self._queue_select_sql(f"WHERE id IN ({placeholders}) ORDER BY id ASC"),
                requeued_ids,
            ).fetchall()
        return [self._queue_item_from_row(cast(tuple[Any, ...], row)) for row in updated]

    def pause_queue_item(self, queue_id: int) -> ExperimentQueueItem:
        return self._set_queue_status(queue_id, QueueStatus.PAUSED, "Queue item paused.")

    def resume_queue_item(self, queue_id: int) -> ExperimentQueueItem:
        return self._set_queue_status(queue_id, QueueStatus.PLANNED, "Queue item resumed.")

    def cancel_queue_item(self, queue_id: int) -> ExperimentQueueItem:
        return self._set_queue_status(queue_id, QueueStatus.CANCELLED, "Queue item cancelled.")

    def requeue_item(self, queue_id: int) -> ExperimentQueueItem:
        return self._set_queue_status(queue_id, QueueStatus.PLANNED, "Queue item requeued.")

    def queue_counts_by_status(self) -> dict[str, int]:
        self.initialize()
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT status, COUNT(*) FROM experiment_queue GROUP BY status"
            ).fetchall()
        return {str(status): int(count) for status, count in rows}

    def list_run_events(self, limit: int = 50) -> list[RunEvent]:
        self.initialize()
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, run_id, queue_id, attempt_id,
                       event_type, message, payload_json, created_at
                FROM run_events
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [self._event_from_row(cast(tuple[Any, ...], row)) for row in rows]

    def query_runs(
        self,
        *,
        family: str | None = None,
        status: str | None = None,
        backend: str | None = None,
        model: str | None = None,
        dataset_hash: str | None = None,
        tag: str | None = None,
        metric: str | None = None,
        metric_min: float | None = None,
        hook_site: str | None = None,
        layer: int | None = None,
        matrix_id: int | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        self.initialize()
        clauses: list[str] = []
        params: list[Any] = []
        if family:
            clauses.append("r.family = ?")
            params.append(family)
        if status:
            clauses.append("r.status = ?")
            params.append(status)
        if backend:
            clauses.append("r.backend = ?")
            params.append(backend)
        if matrix_id is not None:
            clauses.append("r.matrix_id = ?")
            params.append(matrix_id)
        if model:
            clauses.append("r.spec_json LIKE ?")
            params.append(f'%"{model}"%')
        if dataset_hash:
            clauses.append("r.dataset_hashes LIKE ?")
            params.append(f"%{dataset_hash}%")
        if tag:
            clauses.append("r.tags LIKE ?")
            params.append(f'%"{tag}"%')
        if hook_site:
            clauses.append(
                """
                (
                    r.spec_json LIKE ?
                    OR EXISTS (
                        SELECT 1 FROM circuit_patch_rows c
                        WHERE c.run_id = r.id AND c.hook_site = ?
                    )
                    OR EXISTS (
                        SELECT 1 FROM probe_rows p
                        WHERE p.run_id = r.id
                        AND (p.source_hook_site = ? OR p.target_hook_site = ?)
                    )
                )
                """
            )
            params.extend([f"%{hook_site}%", hook_site, hook_site, hook_site])
        if layer is not None:
            clauses.append(
                """
                (
                    r.spec_json LIKE ?
                    OR EXISTS (
                        SELECT 1 FROM circuit_patch_rows c
                        WHERE c.run_id = r.id AND c.layer = ?
                    )
                )
                """
            )
            params.extend([f"%{layer}%", layer])
        if metric:
            clauses.append(
                """
                EXISTS (
                    SELECT 1 FROM run_metrics m
                    WHERE m.run_id = r.id AND m.key = ?
                    AND (? IS NULL OR m.value >= ?)
                )
                """
            )
            params.extend([metric, metric_min, metric_min])
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(limit)
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT r.id, r.spec_name, r.family, r.backend, r.status, r.created_at,
                       r.spec_sha256, r.tags, r.hypothesis, r.matrix_id,
                       res.metrics_json
                FROM runs r
                LEFT JOIN results res ON res.run_id = r.id
                {where}
                ORDER BY r.id DESC
                LIMIT ?
                """,
                params,
            ).fetchall()
        return [
            {
                "run_id": int(row[0]),
                "spec_name": str(row[1]),
                "family": str(row[2]),
                "backend": str(row[3]),
                "status": str(row[4]),
                "created_at": str(row[5]),
                "spec_sha256": row[6],
                "tags": self._json_loads_any(str(row[7])),
                "hypothesis": row[8],
                "matrix_id": row[9],
                "metrics": self._json_loads_any(str(row[10])) if row[10] else {},
            }
            for row in rows
        ]

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA busy_timeout = 30000")
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
        for name, definition in {
            "spec_sha256": "TEXT",
            "source_path": "TEXT",
            "dataset_hashes": "TEXT NOT NULL DEFAULT '{}'",
            "tags": "TEXT NOT NULL DEFAULT '[]'",
            "hypothesis": "TEXT",
            "matrix_id": "INTEGER",
        }.items():
            if name not in columns:
                connection.execute(f"ALTER TABLE runs ADD COLUMN {name} {definition}")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS experiment_queue (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                spec_name TEXT NOT NULL UNIQUE,
                status TEXT NOT NULL,
                retry_count INTEGER NOT NULL DEFAULT 0,
                error TEXT,
                run_id INTEGER,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(run_id) REFERENCES runs(id)
            )
            """
        )
        queue_columns = {
            str(row[1])
            for row in connection.execute("PRAGMA table_info(experiment_queue)").fetchall()
        }
        if "run_id" not in queue_columns:
            connection.execute("ALTER TABLE experiment_queue ADD COLUMN run_id INTEGER")
        for name, definition in {
            "lease_token": "TEXT",
            "worker_id": "TEXT",
            "attempt_id": "INTEGER",
            "claimed_at": "TEXT",
            "heartbeat_at": "TEXT",
            "started_at": "TEXT",
            "finished_at": "TEXT",
            "max_retries": "INTEGER NOT NULL DEFAULT 2",
            "priority": "INTEGER NOT NULL DEFAULT 0",
            "cancelled_at": "TEXT",
            "current_phase": "TEXT",
            "spec_json": "TEXT NOT NULL DEFAULT '{}'",
            "spec_sha256": "TEXT",
            "source_path": "TEXT",
            "dataset_hashes": "TEXT NOT NULL DEFAULT '{}'",
            "tags": "TEXT NOT NULL DEFAULT '[]'",
            "hypothesis": "TEXT",
            "matrix_id": "INTEGER",
        }.items():
            if name not in queue_columns:
                connection.execute(f"ALTER TABLE experiment_queue ADD COLUMN {name} {definition}")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS queue_attempts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                queue_id INTEGER NOT NULL,
                lease_token TEXT NOT NULL,
                worker_id TEXT NOT NULL,
                status TEXT NOT NULL,
                claimed_at TEXT NOT NULL,
                heartbeat_at TEXT,
                started_at TEXT,
                finished_at TEXT,
                error TEXT,
                run_id INTEGER,
                FOREIGN KEY(queue_id) REFERENCES experiment_queue(id),
                FOREIGN KEY(run_id) REFERENCES runs(id)
            )
            """
        )
        self._ensure_columns(
            connection,
            "queue_attempts",
            {
                "queue_id": "INTEGER",
                "lease_token": "TEXT",
                "worker_id": "TEXT",
                "status": "TEXT",
                "claimed_at": "TEXT",
                "heartbeat_at": "TEXT",
                "started_at": "TEXT",
                "finished_at": "TEXT",
                "error": "TEXT",
                "run_id": "INTEGER",
            },
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS run_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER,
                queue_id INTEGER,
                attempt_id INTEGER,
                event_type TEXT NOT NULL,
                message TEXT NOT NULL DEFAULT '',
                payload_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                FOREIGN KEY(run_id) REFERENCES runs(id),
                FOREIGN KEY(queue_id) REFERENCES experiment_queue(id),
                FOREIGN KEY(attempt_id) REFERENCES queue_attempts(id)
            )
            """
        )
        self._ensure_columns(
            connection,
            "run_events",
            {
                "run_id": "INTEGER",
                "queue_id": "INTEGER",
                "attempt_id": "INTEGER",
                "event_type": "TEXT",
                "message": "TEXT NOT NULL DEFAULT ''",
                "payload_json": "TEXT NOT NULL DEFAULT '{}'",
                "created_at": "TEXT",
            },
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS run_metrics (
                run_id INTEGER NOT NULL,
                key TEXT NOT NULL,
                value REAL NOT NULL,
                PRIMARY KEY(run_id, key),
                FOREIGN KEY(run_id) REFERENCES runs(id)
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS run_artifacts (
                run_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                path TEXT NOT NULL,
                media_type TEXT NOT NULL,
                sha256 TEXT,
                size_bytes INTEGER,
                PRIMARY KEY(run_id, name),
                FOREIGN KEY(run_id) REFERENCES runs(id)
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS experiment_matrices (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                hypothesis TEXT,
                tags TEXT NOT NULL DEFAULT '[]',
                source_path TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS matrix_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                matrix_id INTEGER NOT NULL,
                queue_id INTEGER,
                spec_name TEXT NOT NULL,
                spec_sha256 TEXT NOT NULL,
                status TEXT NOT NULL,
                FOREIGN KEY(matrix_id) REFERENCES experiment_matrices(id),
                FOREIGN KEY(queue_id) REFERENCES experiment_queue(id),
                UNIQUE(matrix_id, spec_sha256)
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS circuit_patch_rows (
                run_id INTEGER NOT NULL,
                pair_id TEXT NOT NULL,
                hook_site TEXT NOT NULL,
                layer INTEGER,
                position INTEGER,
                recovery REAL NOT NULL,
                evidence_label TEXT NOT NULL,
                FOREIGN KEY(run_id) REFERENCES runs(id)
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS probe_rows (
                run_id INTEGER NOT NULL,
                source_hook_site TEXT NOT NULL,
                target_hook_site TEXT NOT NULL,
                split TEXT NOT NULL,
                mean_cosine_similarity REAL NOT NULL,
                normalized_mse REAL NOT NULL,
                variance_explained REAL NOT NULL,
                evidence_label TEXT NOT NULL,
                FOREIGN KEY(run_id) REFERENCES runs(id)
            )
            """
        )

    def _create_indexes(self, connection: sqlite3.Connection) -> None:
        statements = [
            (
                "CREATE INDEX IF NOT EXISTS idx_queue_status_priority "
                "ON experiment_queue(status, priority DESC, retry_count, id)"
            ),
            "CREATE INDEX IF NOT EXISTS idx_queue_lease ON experiment_queue(id, lease_token)",
            (
                "CREATE INDEX IF NOT EXISTS idx_queue_heartbeat "
                "ON experiment_queue(status, heartbeat_at, updated_at)"
            ),
            "CREATE INDEX IF NOT EXISTS idx_runs_family_status ON runs(family, status)",
            "CREATE INDEX IF NOT EXISTS idx_runs_backend ON runs(backend)",
            "CREATE INDEX IF NOT EXISTS idx_runs_matrix ON runs(matrix_id)",
            "CREATE INDEX IF NOT EXISTS idx_run_metrics_key_value ON run_metrics(key, value)",
            "CREATE INDEX IF NOT EXISTS idx_run_artifacts_name ON run_artifacts(name)",
            "CREATE INDEX IF NOT EXISTS idx_events_queue ON run_events(queue_id, id)",
            "CREATE INDEX IF NOT EXISTS idx_attempts_queue ON queue_attempts(queue_id, id)",
            "CREATE INDEX IF NOT EXISTS idx_attempts_lease ON queue_attempts(lease_token)",
            (
                "CREATE INDEX IF NOT EXISTS idx_circuit_patch_lookup "
                "ON circuit_patch_rows(run_id, pair_id, hook_site, layer, position, recovery)"
            ),
            (
                "CREATE INDEX IF NOT EXISTS idx_probe_lookup "
                "ON probe_rows(run_id, source_hook_site, target_hook_site, split)"
            ),
        ]
        for statement in statements:
            connection.execute(statement)

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
        run_id: int | None,
    ) -> ExperimentQueueItem:
        self.initialize()
        now = utc_now().isoformat()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT id, status, lease_token, attempt_id, run_id
                FROM experiment_queue
                WHERE spec_name = ?
                """,
                (spec_name,),
            ).fetchone()
            if row is None:
                raise KeyError(f"Queue item for spec '{spec_name}' does not exist.")
            queue_id = int(row[0])
            current_status = QueueStatus(str(row[1]))
            lease_token = None if row[2] is None else str(row[2])
            attempt_id = int(row[3]) if row[3] is not None else None
            existing_run_id = int(row[4]) if row[4] is not None else None
            if current_status == QueueStatus.RUNNING and lease_token is not None:
                raise RuntimeError(
                    f"Queue item {queue_id} has an active lease; use the lease-scoped API."
                )
            event_run_id = run_id if run_id is not None else existing_run_id
            if status == QueueStatus.FAILED:
                connection.execute(
                    """
                    UPDATE experiment_queue
                    SET status = ?,
                        retry_count = retry_count + 1,
                        error = ?,
                        run_id = COALESCE(?, run_id),
                        lease_token = NULL,
                        worker_id = NULL,
                        finished_at = ?,
                        heartbeat_at = ?,
                        current_phase = ?,
                        updated_at = ?
                    WHERE spec_name = ?
                    """,
                    (status.value, error, run_id, now, now, status.value, now, spec_name),
                )
            else:
                connection.execute(
                    """
                    UPDATE experiment_queue
                    SET status = ?,
                        error = ?,
                        run_id = COALESCE(?, run_id),
                        lease_token = NULL,
                        worker_id = NULL,
                        finished_at = ?,
                        heartbeat_at = ?,
                        current_phase = ?,
                        updated_at = ?
                    WHERE spec_name = ?
                    """,
                    (status.value, error, run_id, now, now, status.value, now, spec_name),
                )
            if attempt_id is not None:
                connection.execute(
                    """
                    UPDATE queue_attempts
                    SET status = ?, error = ?, run_id = COALESCE(?, run_id), finished_at = ?
                    WHERE id = ?
                    """,
                    (status.value, error, run_id, now, attempt_id),
                )
            self.append_event(
                connection,
                "succeeded" if status == QueueStatus.SUCCEEDED else "failed",
                run_id=event_run_id,
                queue_id=queue_id,
                attempt_id=attempt_id,
                message=error or f"Queue item {queue_id} {status.value}.",
            )
            updated = connection.execute(
                self._queue_select_sql("WHERE spec_name = ?"),
                (spec_name,),
            ).fetchone()
        return self._queue_item_from_row(cast(tuple[Any, ...], updated))

    def _mark_queue_item_by_lease(
        self,
        queue_id: int,
        lease_token: str,
        status: QueueStatus,
        error: str | None,
        run_id: int | None,
    ) -> ExperimentQueueItem:
        self.initialize()
        now = utc_now().isoformat()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._assert_active_lease(connection, queue_id, lease_token)
            row = connection.execute(
                """
                SELECT attempt_id, run_id, retry_count, max_retries
                FROM experiment_queue
                WHERE id = ?
                """,
                (queue_id,),
            ).fetchone()
            attempt_id = int(row[0]) if row and row[0] is not None else None
            existing_run_id = int(row[1]) if row and row[1] is not None else None
            retry_count = int(row[2]) if row and row[2] is not None else 0
            max_retries = int(row[3]) if row and row[3] is not None else 0
            event_run_id = run_id if run_id is not None else existing_run_id
            if status == QueueStatus.FAILED:
                retry_clause = "retry_count = retry_count + 1,"
                retry_exhausted = retry_count + 1 >= max_retries
            else:
                retry_clause = ""
                retry_exhausted = False
            connection.execute(
                f"""
                UPDATE experiment_queue
                SET status = ?,
                    {retry_clause}
                    error = ?,
                    run_id = COALESCE(?, run_id),
                    lease_token = NULL,
                    worker_id = NULL,
                    finished_at = ?,
                    heartbeat_at = ?,
                    current_phase = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (status.value, error, run_id, now, now, status.value, now, queue_id),
            )
            connection.execute(
                """
                UPDATE queue_attempts
                SET status = ?, error = ?, run_id = COALESCE(?, run_id), finished_at = ?
                WHERE id = ?
                """,
                (status.value, error, run_id, now, attempt_id),
            )
            self.append_event(
                connection,
                "succeeded" if status == QueueStatus.SUCCEEDED else "failed",
                run_id=event_run_id,
                queue_id=queue_id,
                attempt_id=attempt_id,
                message=error or f"Queue item {queue_id} {status.value}.",
                payload=(
                    {"max_retries_exhausted": retry_exhausted}
                    if status == QueueStatus.FAILED
                    else None
                ),
            )
            updated = connection.execute(
                self._queue_select_sql("WHERE id = ?"),
                (queue_id,),
            ).fetchone()
        return self._queue_item_from_row(cast(tuple[Any, ...], updated))

    def _assert_active_lease(
        self,
        connection: sqlite3.Connection,
        queue_id: int,
        lease_token: str,
    ) -> None:
        row = connection.execute(
            "SELECT status, lease_token FROM experiment_queue WHERE id = ?",
            (queue_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"Queue item {queue_id} does not exist.")
        if str(row[0]) != QueueStatus.RUNNING.value or str(row[1]) != lease_token:
            raise RuntimeError(f"Queue item {queue_id} no longer has the active lease.")

    def _set_queue_status(
        self,
        queue_id: int,
        status: QueueStatus,
        message: str,
    ) -> ExperimentQueueItem:
        self.initialize()
        now = utc_now().isoformat()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            current = connection.execute(
                "SELECT status, attempt_id, run_id FROM experiment_queue WHERE id = ?",
                (queue_id,),
            ).fetchone()
            if current is None:
                raise KeyError(f"Queue item {queue_id} does not exist.")
            active_attempt_id = (
                int(current[1])
                if str(current[0]) == QueueStatus.RUNNING.value and current[1] is not None
                else None
            )
            event_run_id = int(current[2]) if current[2] is not None else None
            if status in {QueueStatus.PLANNED, QueueStatus.PAUSED, QueueStatus.CANCELLED}:
                connection.execute(
                    """
                    UPDATE experiment_queue
                    SET status = ?,
                        lease_token = NULL,
                        worker_id = NULL,
                        attempt_id = NULL,
                        claimed_at = NULL,
                        heartbeat_at = NULL,
                        started_at = NULL,
                        finished_at = CASE WHEN ? = ? THEN ? ELSE NULL END,
                        current_phase = NULL,
                        cancelled_at = CASE WHEN ? = ? THEN ? ELSE NULL END,
                        updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        status.value,
                        status.value,
                        QueueStatus.CANCELLED.value,
                        now,
                        status.value,
                        QueueStatus.CANCELLED.value,
                        now,
                        now,
                        queue_id,
                    ),
                )
            else:
                connection.execute(
                    "UPDATE experiment_queue SET status = ?, updated_at = ? WHERE id = ?",
                    (status.value, now, queue_id),
                )
            if active_attempt_id is not None:
                connection.execute(
                    """
                    UPDATE queue_attempts
                    SET status = ?, error = ?, heartbeat_at = ?, finished_at = ?
                    WHERE id = ?
                    """,
                    (status.value, message, now, now, active_attempt_id),
                )
            self.append_event(
                connection,
                status.value,
                run_id=event_run_id,
                queue_id=queue_id,
                attempt_id=active_attempt_id,
                message=message,
            )
            row = connection.execute(self._queue_select_sql("WHERE id = ?"), (queue_id,)).fetchone()
        return self._queue_item_from_row(cast(tuple[Any, ...], row))

    def append_event(
        self,
        connection: sqlite3.Connection,
        event_type: str,
        *,
        run_id: int | None = None,
        queue_id: int | None = None,
        attempt_id: int | None = None,
        message: str = "",
        payload: Mapping[str, Any] | None = None,
    ) -> None:
        connection.execute(
            """
            INSERT INTO run_events (
                run_id, queue_id, attempt_id, event_type, message, payload_json, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                queue_id,
                attempt_id,
                event_type,
                message,
                self._json_dumps(payload or {}),
                utc_now().isoformat(),
            ),
        )

    def _ensure_columns(
        self,
        connection: sqlite3.Connection,
        table: str,
        definitions: Mapping[str, str],
    ) -> None:
        columns = {
            str(row[1])
            for row in connection.execute(f"PRAGMA table_info({table})").fetchall()
        }
        for name, definition in definitions.items():
            if name not in columns:
                connection.execute(f"ALTER TABLE {table} ADD COLUMN {name} {definition}")

    def _index_science_artifacts(
        self,
        connection: sqlite3.Connection,
        run_id: int,
        artifacts: Mapping[str, str],
    ) -> None:
        connection.execute("DELETE FROM circuit_patch_rows WHERE run_id = ?", (run_id,))
        connection.execute("DELETE FROM probe_rows WHERE run_id = ?", (run_id,))
        patch_rows = _read_json_file(artifacts.get("patching_ranked_json"))
        if isinstance(patch_rows, list):
            connection.executemany(
                """
                INSERT INTO circuit_patch_rows (
                    run_id, pair_id, hook_site, layer, position, recovery, evidence_label
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        run_id,
                        str(row.get("pair_id", "")),
                        str(row.get("hook_site", "")),
                        _layer_from_hook_site(str(row.get("hook_site", ""))),
                        _int_or_none(row.get("patch_position")),
                        float(row.get("recovery_fraction", 0.0) or 0.0),
                        str(row.get("evidence_label", "causal evidence")),
                    )
                    for row in patch_rows
                    if isinstance(row, dict)
                ],
            )
        probe_rows = _read_json_file(artifacts.get("cross_model_probe_results_json"))
        if isinstance(probe_rows, list):
            connection.executemany(
                """
                INSERT INTO probe_rows (
                    run_id, source_hook_site, target_hook_site, split,
                    mean_cosine_similarity, normalized_mse,
                    variance_explained, evidence_label
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        run_id,
                        str(row.get("source_hook_site", "")),
                        str(row.get("target_hook_site", "")),
                        str(row.get("split", "")),
                        float(row.get("mean_cosine_similarity", 0.0) or 0.0),
                        float(row.get("normalized_mse", 0.0) or 0.0),
                        float(row.get("variance_explained", 0.0) or 0.0),
                        str(row.get("evidence_label", "correlational alignment")),
                    )
                    for row in probe_rows
                    if isinstance(row, dict)
                ],
            )

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
            run_id=int(row[5]) if row[5] is not None else None,
            created_at=self._parse_datetime(str(row[6])),
            updated_at=self._parse_datetime(str(row[7])),
            lease_token=None if row[8] is None else str(row[8]),
            worker_id=None if row[9] is None else str(row[9]),
            attempt_id=int(row[10]) if row[10] is not None else None,
            claimed_at=self._optional_datetime(row[11]),
            heartbeat_at=self._optional_datetime(row[12]),
            started_at=self._optional_datetime(row[13]),
            finished_at=self._optional_datetime(row[14]),
            max_retries=int(row[15]),
            priority=int(row[16]),
            cancelled_at=self._optional_datetime(row[17]),
            current_phase=None if row[18] is None else str(row[18]),
            spec_json=str(row[19]),
            spec_sha256=None if row[20] is None else str(row[20]),
            source_path=None if row[21] is None else str(row[21]),
            dataset_hashes=str(row[22]),
            tags=str(row[23]),
            hypothesis=None if row[24] is None else str(row[24]),
            matrix_id=int(row[25]) if row[25] is not None else None,
        )

    def _event_from_row(self, row: tuple[Any, ...]) -> RunEvent:
        return RunEvent(
            id=int(row[0]),
            run_id=int(row[1]) if row[1] is not None else None,
            queue_id=int(row[2]) if row[2] is not None else None,
            attempt_id=int(row[3]) if row[3] is not None else None,
            event_type=str(row[4]),
            message=str(row[5]),
            payload=self._json_loads_object(str(row[6])),
            created_at=self._parse_datetime(str(row[7])),
        )

    def _queue_select_sql(self, suffix: str) -> str:
        return f"""
            SELECT id, spec_name, status, retry_count, error, run_id, created_at, updated_at,
                   lease_token, worker_id, attempt_id, claimed_at, heartbeat_at, started_at,
                   finished_at, max_retries, priority, cancelled_at, current_phase, spec_json,
                   spec_sha256, source_path, dataset_hashes, tags, hypothesis, matrix_id
            FROM experiment_queue
            {suffix}
        """

    def _spec_json(self, spec: ExperimentSpec) -> str:
        return self._json_dumps(asdict(spec))

    def _spec_sha256(self, spec: ExperimentSpec) -> str:
        import hashlib

        return hashlib.sha256(self._spec_json(spec).encode("utf-8")).hexdigest()

    def _json_dumps(self, payload: Any) -> str:
        return json.dumps(payload, default=str, sort_keys=True)

    def _json_loads_object(self, payload: str) -> dict[str, Any]:
        decoded = json.loads(payload)
        if not isinstance(decoded, dict):
            raise ValueError("Expected a JSON object.")
        return cast(dict[str, Any], decoded)

    def _json_loads_any(self, payload: str) -> Any:
        return json.loads(payload)

    def _parse_datetime(self, payload: str) -> datetime:
        parsed = datetime.fromisoformat(payload)
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=UTC)
        return parsed

    def _optional_datetime(self, payload: Any) -> datetime | None:
        if payload is None:
            return None
        return self._parse_datetime(str(payload))


def _dataset_hashes(spec: ExperimentSpec) -> dict[str, str]:
    value = spec.parameters.get("dataset_hashes", {})
    return cast(dict[str, str], value) if isinstance(value, dict) else {}


def _tags(spec: ExperimentSpec) -> list[str]:
    value = spec.parameters.get("tags", [])
    return [str(item) for item in value] if isinstance(value, list) else []


def _hypothesis(spec: ExperimentSpec) -> str | None:
    value = spec.parameters.get("hypothesis")
    return str(value) if value is not None else None


def _media_type_from_path(path_value: str) -> str:
    suffix = Path(path_value).suffix
    if suffix == ".json":
        return "application/json"
    if suffix == ".csv":
        return "text/csv"
    if suffix in {".md", ".txt"}:
        return "text/plain"
    if suffix == ".npz":
        return "application/x-numpy-npz"
    return "application/octet-stream"


def _path_size(path_value: str) -> int | None:
    try:
        return Path(path_value).stat().st_size
    except OSError:
        return None


def _read_json_file(path_value: str | None) -> Any:
    if path_value is None:
        return None
    try:
        return json.loads(Path(path_value).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _layer_from_hook_site(hook_site: str) -> int | None:
    parts = hook_site.split(".")
    for index, part in enumerate(parts[:-1]):
        if part == "blocks":
            return _int_or_none(parts[index + 1])
    return None


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
