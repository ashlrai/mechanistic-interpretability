from typing import Any

from mech_interp.orchestration.planner import RunPlan, plan_runs
from mech_interp.orchestration.queue import ExperimentRunQueue, QueuePlan
from mech_interp.orchestration.resource_policy import ActivationEstimate, ResourcePolicy

__all__ = [
    "ActivationEstimate",
    "ExperimentRunQueue",
    "ExperimentRunner",
    "QueuePlan",
    "ResourcePolicy",
    "RunPlan",
    "plan_runs",
]


def __getattr__(name: str) -> Any:
    if name == "ExperimentRunner":
        from mech_interp.orchestration.runner import ExperimentRunner

        return ExperimentRunner
    raise AttributeError(f"module 'mech_interp.orchestration' has no attribute {name!r}")
