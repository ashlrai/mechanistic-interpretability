"""End-to-end integration test for causal scrubbing on gpt2-small.

4 IOI-style prompts in 2 equivalence classes.  All attention hook-z sites
except L9 and L9 are scrubbed.  The scrubbed faithfulness should be a
real number in (0, 1) and KL should be finite.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from mech_interp.experiments.causal_scrubbing import CausalScrubbingExperiment
from mech_interp.types import ExperimentRun, ExperimentSpec, RunStatus, utc_now

pytestmark = pytest.mark.integration


def _spec() -> ExperimentSpec:
    return ExperimentSpec(
        name="e2e-causal-scrubbing-ioi",
        family="causal_scrubbing",
        backend="transformerlens",
        description=(
            "Hypothesis: heads L9.H6 and L9.H9 (name-mover heads) account for "
            "name copying in IOI. Scrub all other attention layers."
        ),
        parameters={
            "model": "gpt2",
            "device": "cpu",
            "seed": 42,
            "prompts": [
                {
                    "id": "ioi-jm-1",
                    "prompt": "When Mary and John went to the store, John gave a book to",
                    "equivalence_label": "John-Mary",
                    "target_position": -1,
                },
                {
                    "id": "ioi-jm-2",
                    "prompt": "When Mary and John went to the park, John handed the ball to",
                    "equivalence_label": "John-Mary",
                    "target_position": -1,
                },
                {
                    "id": "ioi-te-1",
                    "prompt": "When Emma and Tom went to the store, Tom gave a gift to",
                    "equivalence_label": "Tom-Emma",
                    "target_position": -1,
                },
                {
                    "id": "ioi-te-2",
                    "prompt": "When Emma and Tom went to the cafe, Tom passed the menu to",
                    "equivalence_label": "Tom-Emma",
                    "target_position": -1,
                },
            ],
            # Keep L9 attention (name-mover heads live here) in the circuit.
            "protected_sites": [
                "blocks.9.attn.hook_z",
            ],
            # Scrub all other attention layers.
            "scrubbed_sites": [
                "blocks.0.attn.hook_z",
                "blocks.1.attn.hook_z",
                "blocks.2.attn.hook_z",
                "blocks.3.attn.hook_z",
                "blocks.4.attn.hook_z",
                "blocks.5.attn.hook_z",
                "blocks.6.attn.hook_z",
                "blocks.7.attn.hook_z",
                "blocks.8.attn.hook_z",
                "blocks.10.attn.hook_z",
                "blocks.11.attn.hook_z",
            ],
            "artifact_policy": {
                "retain_activation_tensors": False,
                "write_report": True,
            },
        },
    )


def _make_run(tmp_path: Path) -> ExperimentRun:
    return ExperimentRun(
        id=1,
        spec_name="e2e-causal-scrubbing-ioi",
        family="causal_scrubbing",
        backend="transformerlens",
        status=RunStatus.RUNNING,
        artifact_dir=tmp_path,
        created_at=utc_now(),
    )


def test_causal_scrubbing_e2e_faithfulness_in_range(
    gpt2_backend: Any, tmp_path: Path
) -> None:
    """Scrubbed faithfulness must be a real number strictly in (0, 1)."""
    spec = _spec()
    run = _make_run(tmp_path)

    result = CausalScrubbingExperiment(backend=gpt2_backend).run(spec, run)

    assert result.status == RunStatus.SUCCEEDED, f"Experiment failed: {result.notes}"

    faithfulness = result.metrics["scrubbed_faithfulness"]
    mean_kl = result.metrics["mean_kl"]

    assert math.isfinite(faithfulness), "Faithfulness is not finite"
    assert math.isfinite(mean_kl), "Mean KL is not finite"
    assert 0.0 < faithfulness < 1.0, (
        f"Expected faithfulness in (0,1), got {faithfulness:.4f}"
    )
    assert mean_kl >= 0.0, f"Mean KL must be non-negative, got {mean_kl}"


def test_causal_scrubbing_e2e_artifacts_exist(gpt2_backend: Any, tmp_path: Path) -> None:
    """All three artifact files must be present and valid JSON/text."""
    spec = _spec()
    run = _make_run(tmp_path)

    result = CausalScrubbingExperiment(backend=gpt2_backend).run(spec, run)
    assert result.status == RunStatus.SUCCEEDED

    # scrubbing_results.json — per-prompt rows.
    results_path = Path(result.artifacts["scrubbing_results"])
    assert results_path.is_file()
    rows = json.loads(results_path.read_text())
    assert len(rows) == 4  # one per prompt
    required_keys = {"prompt_id", "equivalence_label", "scrub_source_id", "kl_divergence"}
    for row in rows:
        assert required_keys.issubset(row.keys())
        assert math.isfinite(row["kl_divergence"])
        assert row["kl_divergence"] >= 0.0

    # scrubbing_summary.json
    summary_path = Path(result.artifacts["scrubbing_summary"])
    assert summary_path.is_file()
    summary = json.loads(summary_path.read_text())
    assert summary["prompt_count"] == 4
    assert len(summary["equivalence_class_sizes"]) == 2
    assert summary["equivalence_class_sizes"]["John-Mary"] == 2
    assert summary["equivalence_class_sizes"]["Tom-Emma"] == 2

    # research_note.md
    report_path = Path(result.artifacts["research_note"])
    assert report_path.is_file()
    text = report_path.read_text(encoding="utf-8")
    assert "Causal Scrubbing Report" in text
    assert "faithfulness" in text.lower()


def test_causal_scrubbing_e2e_scrub_sources_from_same_class(
    gpt2_backend: Any, tmp_path: Path
) -> None:
    """Each prompt's scrub source must belong to the same equivalence class."""
    spec = _spec()
    run = _make_run(tmp_path)

    result = CausalScrubbingExperiment(backend=gpt2_backend).run(spec, run)
    assert result.status == RunStatus.SUCCEEDED

    rows = json.loads(Path(result.artifacts["scrubbing_results"]).read_text())
    prompts_spec = spec.parameters["prompts"]
    id_to_label = {p["id"]: p["equivalence_label"] for p in prompts_spec}

    for row in rows:
        prompt_label = id_to_label[row["prompt_id"]]
        source_label = id_to_label[row["scrub_source_id"]]
        assert prompt_label == source_label, (
            f"Prompt {row['prompt_id']} (label={prompt_label}) was scrubbed from "
            f"{row['scrub_source_id']} (label={source_label}) — cross-class scrub!"
        )
