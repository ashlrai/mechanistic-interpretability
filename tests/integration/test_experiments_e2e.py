"""End-to-end smoke tests that exercise real TransformerLens execution on gpt2-small.

These complement the unit tests (which all use fakes) by catching bugs that only
appear when TransformerLens's real hook calling convention is exercised — like
the Run 15 ``patch_hook`` keyword failure.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from mech_interp.experiments.activation_capture import ActivationCaptureExperiment
from mech_interp.experiments.circuit_patching import CircuitPatchingExperiment
from mech_interp.types import (
    ActivationPatchPromptPair,
    ActivationPatchRequest,
    ExperimentRun,
    ExperimentSpec,
    RunStatus,
    utc_now,
)

pytestmark = pytest.mark.integration


def _run(spec: ExperimentSpec, tmp_path: Path) -> ExperimentRun:
    return ExperimentRun(
        id=1,
        spec_name=spec.name,
        family=spec.family,
        backend=spec.backend,
        status=RunStatus.RUNNING,
        artifact_dir=tmp_path,
        created_at=utc_now(),
    )


def test_activation_capture_produces_real_tensor_summaries(
    gpt2_backend: Any, tmp_path: Path
) -> None:
    spec = ExperimentSpec(
        name="e2e-activation-capture",
        family="activation_capture",
        backend="transformerlens",
        description="",
        parameters={
            "model": "gpt2-small",
            "prompts": [
                "The Eiffel Tower is in",
                "The capital of France is",
            ],
            "sites": ["blocks.0.hook_resid_pre", "blocks.1.hook_resid_pre"],
            "seed": 42,
        },
    )
    result = ActivationCaptureExperiment(backend=gpt2_backend).run(spec, _run(spec, tmp_path))

    assert result.status == RunStatus.SUCCEEDED
    summary = json.loads(Path(result.artifacts["activation_summary"]).read_text())
    assert summary["missing_sites"] == []
    for site in spec.parameters["sites"]:
        stats = summary["summaries"][site]
        assert stats["shape"][-1] == 768  # gpt2-small d_model
        assert stats["std"] > 0
        assert "torch.float" in stats["dtype"]


def test_circuit_patching_recovery_matches_existing_evidence(
    gpt2_backend: Any, tmp_path: Path
) -> None:
    """gpt2-small residual-stream patching at layer 2 should recover near 1.0,
    while the mlp.hook_post control should recover near zero. Mirrors run-000024.
    """
    spec = ExperimentSpec(
        name="e2e-circuit-patching",
        family="circuit_patching",
        backend="transformerlens",
        description="",
        parameters={
            "model": "gpt2-small",
            "layers": [2],
            "patch_sites": ["resid_pre"],
            "control_patch_sites": ["mlp_post"],
            "prompt_pairs": [
                {
                    "id": "capital-france",
                    "clean_prompt": "The capital of France is Paris",
                    "corrupted_prompt": "The capital of France is Rome",
                    "correct_token": " Paris",
                    "incorrect_token": " Rome",
                },
            ],
            "seed": 42,
            "artifact_policy": {
                "retain_activation_tensors": False,
                "write_report": False,
            },
        },
    )
    result = CircuitPatchingExperiment(backend=gpt2_backend).run(spec, _run(spec, tmp_path))

    assert result.status == RunStatus.SUCCEEDED
    ranked = json.loads(Path(result.artifacts["patching_ranked_json"]).read_text())
    assert ranked

    experimental = [row for row in ranked if "hook_resid_pre" in row["hook_site"]]
    control = [row for row in ranked if "mlp.hook_post" in row["hook_site"]]
    assert experimental, "expected experimental site row"
    assert control, "expected control site row"
    # Real evidence from run-000024: residual patching cleanly recovers the answer.
    assert experimental[0]["recovery_fraction"] > 0.9
    # Control should recover noticeably less.
    assert control[0]["recovery_fraction"] < experimental[0]["recovery_fraction"] - 0.3


def test_run_activation_patching_accepts_real_tl_hook_kwargs(gpt2_backend: Any) -> None:
    """Direct regression for Run 15: patch_hook must accept TL's ``hook=`` kwarg
    when invoked through the real run_with_hooks pipeline (not just fakes)."""
    request = ActivationPatchRequest(
        model_name="gpt2-small",
        prompt_pairs=(
            ActivationPatchPromptPair(
                id="pair",
                clean_prompt="The capital of France is Paris",
                corrupted_prompt="The capital of France is Rome",
                correct_token=" Paris",
                incorrect_token=" Rome",
            ),
        ),
        hook_sites=("blocks.0.hook_resid_pre",),
    )
    results = gpt2_backend.run_activation_patching(request)
    assert len(results) == 1
    assert results[0].hook_site == "blocks.0.hook_resid_pre"
    assert results[0].patched_logit_diff is not None
