"""End-to-end integration test for `mech demo`.

Loads gpt2-small once (via the session-scoped gpt2_backend fixture) and exercises
the full demo pipeline:
  - run_demo_experiments produces a DemoResult with all three experiments succeeding
  - summary.png is created (or gracefully skipped when matplotlib is absent)
  - summary.md is written and contains the expected sections

Uses the same session-scoped backend as other integration tests to avoid
loading the model twice in a single pytest session.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from mech_interp.demo import (
    DemoResult,
    render_demo_chart,
    render_demo_markdown,
)
from mech_interp.types import RunStatus

pytestmark = pytest.mark.integration


def _have_matplotlib() -> bool:
    try:
        import matplotlib  # noqa: F401
        return True
    except ImportError:
        return False


def test_demo_all_three_experiments_succeed(
    gpt2_backend: Any, tmp_path: Path
) -> None:
    """All three sub-experiments should succeed and populate the DemoResult."""
    # Patch the backend into the experiments by monkey-patching demo module
    import mech_interp.demo as demo_mod

    def _patched_run(output_dir: Path) -> DemoResult:
        """Run experiments with the pre-loaded fixture backend."""
        from mech_interp.experiments.circuit_patching import CircuitPatchingExperiment
        from mech_interp.experiments.direct_logit_attribution import (
            DirectLogitAttributionExperiment,
        )
        from mech_interp.experiments.logit_lens import LogitLensExperiment
        from mech_interp.types import ExperimentSpec

        output_dir.mkdir(parents=True, exist_ok=True)
        result = DemoResult(artifact_dir=output_dir)

        # DLA
        dla_dir = output_dir / "dla"
        dla_dir.mkdir(parents=True, exist_ok=True)
        dla_spec = ExperimentSpec(
            name="demo-dla",
            family="direct_logit_attribution",
            backend="transformerlens",
            parameters={
                "model": "gpt2-small",
                "device": "cpu",
                "seed": 42,
                "target_position": -1,
                "top_k": 10,
                "prompt_pairs": demo_mod._DLA_PROMPT_PAIRS,
            },
        )
        dla_run = demo_mod._make_run(1, dla_spec, dla_dir)
        result.dla_result = DirectLogitAttributionExperiment(backend=gpt2_backend).run(
            dla_spec, dla_run
        )
        result = demo_mod._parse_dla(result, result.dla_result)

        # Logit lens
        lens_dir = output_dir / "lens"
        lens_dir.mkdir(parents=True, exist_ok=True)
        lens_spec = ExperimentSpec(
            name="demo-lens",
            family="logit_lens",
            backend="transformerlens",
            parameters={
                "model": "gpt2-small",
                "device": "cpu",
                "seed": 42,
                "target_position": -1,
                "top_k": 5,
                "mode": "logit",
                "prompts": demo_mod._LENS_PROMPTS,
            },
        )
        lens_run = demo_mod._make_run(2, lens_spec, lens_dir)
        result.lens_result = LogitLensExperiment(backend=gpt2_backend).run(
            lens_spec, lens_run
        )
        result = demo_mod._parse_lens(result, result.lens_result)

        # Circuit patching
        patch_dir = output_dir / "patching"
        patch_dir.mkdir(parents=True, exist_ok=True)
        patch_spec = ExperimentSpec(
            name="demo-patching",
            family="circuit_patching",
            backend="transformerlens",
            parameters={
                "model": "gpt2-small",
                "device": "cpu",
                "seed": 42,
                "target_position": -1,
                "patch_position": -1,
                "layers": [8, 9, 10, 11],
                "patch_sites": ["resid_pre", "mlp_out"],
                "prompt_pairs": demo_mod._DEMO_PROMPT_PAIRS,
                "artifact_policy": {
                    "retain_activation_tensors": False,
                    "write_report": True,
                },
            },
        )
        patch_run = demo_mod._make_run(3, patch_spec, patch_dir)
        result.patch_result = CircuitPatchingExperiment(backend=gpt2_backend).run(
            patch_spec, patch_run
        )
        result = demo_mod._parse_patch(result, result.patch_result)
        return result

    result = _patched_run(tmp_path)

    assert not result.errors, f"Demo produced errors: {result.errors}"

    # DLA
    assert result.dla_result is not None
    assert result.dla_result.status == RunStatus.SUCCEEDED
    assert result.dla_top_component != ""
    assert result.dla_top_score > 0

    # Logit lens
    assert result.lens_result is not None
    assert result.lens_result.status == RunStatus.SUCCEEDED
    assert result.lens_n_layers > 0
    assert result.lens_final_rank > 0

    # Circuit patching
    assert result.patch_result is not None
    assert result.patch_result.status == RunStatus.SUCCEEDED
    assert result.patch_top_site != ""
    assert 0.0 <= result.patch_top_recovery <= 1.0


def test_demo_summary_md_sections(gpt2_backend: Any, tmp_path: Path) -> None:
    """summary.md must contain the expected sections."""
    import mech_interp.demo as demo_mod

    # Build a minimal result using a quick DLA run
    dla_dir = tmp_path / "dla"
    dla_dir.mkdir(parents=True, exist_ok=True)
    from mech_interp.experiments.direct_logit_attribution import (
        DirectLogitAttributionExperiment,
    )
    from mech_interp.types import ExperimentSpec

    spec = ExperimentSpec(
        name="demo-dla-md-test",
        family="direct_logit_attribution",
        backend="transformerlens",
        parameters={
            "model": "gpt2-small",
            "device": "cpu",
            "seed": 42,
            "target_position": -1,
            "top_k": 10,
            "prompt_pairs": demo_mod._DLA_PROMPT_PAIRS,
        },
    )
    run = demo_mod._make_run(1, spec, dla_dir)
    dla_result = DirectLogitAttributionExperiment(backend=gpt2_backend).run(spec, run)

    result = DemoResult(artifact_dir=tmp_path)
    result.dla_result = dla_result
    result = demo_mod._parse_dla(result, dla_result)

    md = render_demo_markdown(result)
    md_path = tmp_path / "summary.md"
    md_path.write_text(md, encoding="utf-8")

    assert md_path.exists()
    content = md_path.read_text()
    assert "## Results" in content
    assert "## What happened" in content
    assert "## Next steps" in content
    assert "05_research_walkthrough" in content


@pytest.mark.skipif(not _have_matplotlib(), reason="matplotlib not installed")
def test_demo_chart_saved(gpt2_backend: Any, tmp_path: Path) -> None:
    """summary.png must be created when matplotlib is available."""
    import mech_interp.demo as demo_mod

    dla_dir = tmp_path / "dla"
    dla_dir.mkdir(parents=True, exist_ok=True)
    from mech_interp.experiments.direct_logit_attribution import (
        DirectLogitAttributionExperiment,
    )
    from mech_interp.types import ExperimentSpec

    spec = ExperimentSpec(
        name="demo-chart-test",
        family="direct_logit_attribution",
        backend="transformerlens",
        parameters={
            "model": "gpt2-small",
            "device": "cpu",
            "seed": 42,
            "target_position": -1,
            "top_k": 10,
            "prompt_pairs": demo_mod._DLA_PROMPT_PAIRS,
        },
    )
    run = demo_mod._make_run(1, spec, dla_dir)
    dla_result = DirectLogitAttributionExperiment(backend=gpt2_backend).run(spec, run)

    result = DemoResult(artifact_dir=tmp_path)
    result.dla_result = dla_result
    result = demo_mod._parse_dla(result, dla_result)

    chart_path = tmp_path / "summary.png"
    render_demo_chart(result, chart_path)
    assert chart_path.exists(), "summary.png was not created"
    assert chart_path.stat().st_size > 0
