from __future__ import annotations

from dataclasses import dataclass

from mech_interp.types import ExperimentSpec


@dataclass(frozen=True)
class RunPlan:
    specs: list[ExperimentSpec]
    max_parallel_runs: int
    max_prompts_per_batch: int


def plan_runs(
    specs: list[ExperimentSpec],
    max_parallel_runs: int = 1,
    max_prompts_per_batch: int = 32,
) -> RunPlan:
    if max_parallel_runs < 1:
        raise ValueError("max_parallel_runs must be at least 1.")
    if max_prompts_per_batch < 1:
        raise ValueError("max_prompts_per_batch must be at least 1.")
    return RunPlan(
        specs=specs,
        max_parallel_runs=max_parallel_runs,
        max_prompts_per_batch=max_prompts_per_batch,
    )
