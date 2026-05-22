from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from mech_interp.types import ExperimentResult, ExperimentRun, ExperimentSpec, RunStatus, utc_now


class SQLiteResultStore:
    def __init__(self, database_path: str | Path, artifact_dir: str | Path) -> None:
        self.database_path = Path(database_path)
        self.artifact_dir = Path(artifact_dir)

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

    def create_run(self, spec: ExperimentSpec) -> ExperimentRun:
        self.initialize()
        created_at = utc_now()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO runs (spec_name, family, backend, status, artifact_dir, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    spec.name,
                    spec.family,
                    spec.backend,
                    RunStatus.PLANNED.value,
                    str(self.artifact_dir),
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
            connection.execute(
                "UPDATE runs SET status = ? WHERE id = ?",
                (result.status.value, result.run_id),
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
            ExperimentRun(
                id=int(row[0]),
                spec_name=str(row[1]),
                family=str(row[2]),
                backend=str(row[3]),
                status=RunStatus(str(row[4])),
                artifact_dir=Path(str(row[5])),
                created_at=utc_now(),
            )
            for row in rows
        ]

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.database_path)
