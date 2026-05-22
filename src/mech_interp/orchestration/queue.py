from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from mech_interp.storage.sqlite_store import ExperimentQueueItem, SQLiteResultStore
from mech_interp.types import ExperimentSpec


@dataclass(frozen=True)
class QueuePlan:
    enqueued: int
    total: int


class ExperimentRunQueue:
    def __init__(self, store: SQLiteResultStore) -> None:
        self.store = store

    def plan(self, specs: Iterable[ExperimentSpec]) -> QueuePlan:
        spec_list = list(specs)
        enqueued = self.store.enqueue_experiment_specs(spec_list)
        return QueuePlan(enqueued=enqueued, total=len(spec_list))

    def claim_next(self) -> ExperimentQueueItem | None:
        return self.store.claim_next_queue_item()

    def mark_succeeded(self, spec_name: str) -> ExperimentQueueItem:
        return self.store.mark_queue_item_succeeded(spec_name)

    def mark_failed(self, spec_name: str, error: str) -> ExperimentQueueItem:
        return self.store.mark_queue_item_failed(spec_name, error)

    def list(self) -> list[ExperimentQueueItem]:
        return self.store.list_queue_items()
