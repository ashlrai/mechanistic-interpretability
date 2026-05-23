from __future__ import annotations

import time
import uuid
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Protocol

from mech_interp.storage.sqlite_store import ExperimentQueueItem, SQLiteResultStore
from mech_interp.types import ExperimentResult, ExperimentSpec, RunStatus


@dataclass(frozen=True)
class QueuePlan:
    enqueued: int
    total: int


class QueueExperimentRunner(Protocol):
    def run(self, spec: ExperimentSpec) -> ExperimentResult:
        """Run an experiment spec and return a persisted result."""


class ExperimentRunQueue:
    def __init__(self, store: SQLiteResultStore) -> None:
        self.store = store

    def plan(self, specs: Iterable[ExperimentSpec]) -> QueuePlan:
        spec_list = list(specs)
        enqueued = self.store.enqueue_experiment_specs(spec_list)
        return QueuePlan(enqueued=enqueued, total=len(spec_list))

    def claim_next(self, worker_id: str | None = None) -> ExperimentQueueItem | None:
        return self.store.claim_next_queue_item(worker_id=worker_id)

    def mark_succeeded(self, spec_name: str, run_id: int | None = None) -> ExperimentQueueItem:
        return self.store.mark_queue_item_succeeded(spec_name, run_id=run_id)

    def mark_failed(
        self,
        spec_name: str,
        error: str,
        run_id: int | None = None,
    ) -> ExperimentQueueItem:
        return self.store.mark_queue_item_failed(spec_name, error, run_id=run_id)

    def list(self) -> Sequence[ExperimentQueueItem]:
        return self.store.list_queue_items()

    def requeue_stale(self, stale_after_seconds: int) -> Sequence[ExperimentQueueItem]:
        return self.store.requeue_stale_items(stale_after_seconds)

    def pause(self, queue_id: int) -> ExperimentQueueItem:
        return self.store.pause_queue_item(queue_id)

    def resume(self, queue_id: int) -> ExperimentQueueItem:
        return self.store.resume_queue_item(queue_id)

    def cancel(self, queue_id: int) -> ExperimentQueueItem:
        return self.store.cancel_queue_item(queue_id)

    def requeue(self, queue_id: int) -> ExperimentQueueItem:
        return self.store.requeue_item(queue_id)

    def run_once(
        self,
        specs_by_name: dict[str, ExperimentSpec],
        runner: QueueExperimentRunner,
    ) -> ExperimentResult | None:
        item = self.claim_next(worker_id=f"cli-{uuid.uuid4().hex[:8]}")
        if item is None:
            return None
        spec = specs_by_name.get(item.spec_name)
        if spec is None:
            if item.lease_token is None:
                self.mark_failed(item.spec_name, f"Spec '{item.spec_name}' was not found.")
            else:
                self.store.mark_queue_item_failed_by_lease(
                    item.id,
                    item.lease_token,
                    f"Spec '{item.spec_name}' was not found.",
                )
            return None
        try:
            if item.lease_token is not None:
                self.store.update_queue_phase(item.id, item.lease_token, "running_experiment")
            result = runner.run(spec)
        except Exception as exc:
            if item.lease_token is None:
                self.mark_failed(item.spec_name, f"{type(exc).__name__}: {exc}")
            else:
                self.store.mark_queue_item_failed_by_lease(
                    item.id,
                    item.lease_token,
                    f"{type(exc).__name__}: {exc}",
                )
            raise
        if result.status == RunStatus.SUCCEEDED:
            if item.lease_token is None:
                self.mark_succeeded(item.spec_name, run_id=result.run_id)
            else:
                self.store.mark_queue_item_succeeded_by_lease(
                    item.id,
                    item.lease_token,
                    run_id=result.run_id,
                )
        else:
            if item.lease_token is None:
                self.mark_failed(item.spec_name, result.notes or result.status.value, result.run_id)
            else:
                self.store.mark_queue_item_failed_by_lease(
                    item.id,
                    item.lease_token,
                    result.notes or result.status.value,
                    result.run_id,
                )
        return result

    def run_loop(
        self,
        specs_by_name: dict[str, ExperimentSpec],
        runner: QueueExperimentRunner,
        poll_interval: float,
    ) -> None:
        while True:
            result = self.run_once(specs_by_name, runner)
            if result is None:
                time.sleep(poll_interval)
