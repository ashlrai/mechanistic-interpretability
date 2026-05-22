from __future__ import annotations

from collections import Counter

from mech_interp.storage import SQLiteResultStore


def summarize_recent_runs(store: SQLiteResultStore, limit: int = 100) -> dict[str, object]:
    runs = store.list_runs(limit=limit)
    return {
        "run_count": len(runs),
        "statuses": dict(sorted(Counter(run.status.value for run in runs).items())),
        "families": dict(sorted(Counter(run.family for run in runs).items())),
        "backends": dict(sorted(Counter(run.backend for run in runs).items())),
    }
