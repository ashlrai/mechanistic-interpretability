"""Unit tests for mech_interp.demo — render functions with fake DemoResult.

These tests do NOT load any model; they verify that the Rich rendering and
markdown generation produce the expected fields and structure.
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest

from mech_interp.demo import DemoResult, render_demo_markdown, render_demo_summary


def _fake_result(tmp_path: Path) -> DemoResult:
    return DemoResult(
        dla_top_component="L9_mlp",
        dla_top_score=3.142,
        dla_total_components=192,
        lens_commit_layer=7,
        lens_n_layers=12,
        lens_final_rank=1.0,
        patch_top_site="blocks.9.hook_resid_pre",
        patch_top_recovery=0.93,
        patch_pair_id="capital-france",
        artifact_dir=tmp_path,
    )


def test_render_demo_summary_contains_top_component(tmp_path: Path) -> None:
    """Rich panel must mention the top DLA component."""
    from rich.console import Console

    buf = io.StringIO()
    con = Console(file=buf, highlight=False, markup=True, width=120)
    render_demo_summary(_fake_result(tmp_path), con)
    output = buf.getvalue()
    assert "L9_mlp" in output


def test_render_demo_summary_contains_commit_layer(tmp_path: Path) -> None:
    """Rich panel must state which layer the model commits at."""
    from rich.console import Console

    buf = io.StringIO()
    con = Console(file=buf, highlight=False, markup=True, width=120)
    render_demo_summary(_fake_result(tmp_path), con)
    output = buf.getvalue()
    assert "7" in output  # commit layer


def test_render_demo_summary_contains_patch_recovery(tmp_path: Path) -> None:
    """Rich panel must include the circuit patching recovery percentage."""
    from rich.console import Console

    buf = io.StringIO()
    con = Console(file=buf, highlight=False, markup=True, width=120)
    render_demo_summary(_fake_result(tmp_path), con)
    output = buf.getvalue()
    assert "93%" in output or "0.93" in output or "93" in output


def test_render_demo_summary_contains_notebook_link(tmp_path: Path) -> None:
    """Rich panel must point to the research walkthrough notebook."""
    from rich.console import Console

    buf = io.StringIO()
    con = Console(file=buf, highlight=False, markup=True, width=120)
    render_demo_summary(_fake_result(tmp_path), con)
    output = buf.getvalue()
    assert "05_research_walkthrough" in output


def test_render_demo_summary_contains_five_explanation_lines(tmp_path: Path) -> None:
    """The five-line explanation must be present (look for numbered items 1-5)."""
    from rich.console import Console

    buf = io.StringIO()
    con = Console(file=buf, highlight=False, markup=True, width=120)
    render_demo_summary(_fake_result(tmp_path), con)
    output = buf.getvalue()
    for n in ("1.", "2.", "3.", "4.", "5."):
        assert n in output, f"Missing numbered item {n!r} in summary output"


def test_render_demo_markdown_has_results_table(tmp_path: Path) -> None:
    """Markdown output must include a Results table with expected rows."""
    md = render_demo_markdown(_fake_result(tmp_path))
    assert "## Results" in md
    assert "Direct Logit Attribution" in md
    assert "Logit Lens" in md
    assert "Circuit Patching" in md


def test_render_demo_markdown_has_what_happened_section(tmp_path: Path) -> None:
    md = render_demo_markdown(_fake_result(tmp_path))
    assert "## What happened" in md


def test_render_demo_markdown_has_next_steps(tmp_path: Path) -> None:
    md = render_demo_markdown(_fake_result(tmp_path))
    assert "## Next steps" in md
    assert "05_research_walkthrough" in md


def test_render_demo_markdown_values(tmp_path: Path) -> None:
    """Numeric values and component names must appear in the markdown."""
    md = render_demo_markdown(_fake_result(tmp_path))
    assert "L9_mlp" in md
    assert "blocks.9.hook_resid_pre" in md
    assert "capital-france" in md


def test_render_demo_summary_with_errors(tmp_path: Path) -> None:
    """Errors list must produce an error section in the Rich output."""
    from rich.console import Console

    result = _fake_result(tmp_path)
    result.errors.append("DLA failed: some reason")

    buf = io.StringIO()
    con = Console(file=buf, highlight=False, markup=True, width=120)
    render_demo_summary(result, con)
    output = buf.getvalue()
    assert "DLA failed" in output


def test_render_demo_markdown_errors_section(tmp_path: Path) -> None:
    result = _fake_result(tmp_path)
    result.errors.append("Circuit patching failed: timeout")
    md = render_demo_markdown(result)
    assert "## Errors" in md
    assert "Circuit patching failed" in md


def test_demo_result_default_no_errors() -> None:
    r = DemoResult()
    assert r.errors == []
    assert r.dla_top_component == ""
    assert r.lens_commit_layer == -1


@pytest.mark.parametrize("commit_layer,n_layers", [(0, 12), (7, 12), (11, 12)])
def test_render_demo_markdown_commit_layer_variants(
    tmp_path: Path, commit_layer: int, n_layers: int
) -> None:
    r = _fake_result(tmp_path)
    r.lens_commit_layer = commit_layer
    r.lens_n_layers = n_layers
    md = render_demo_markdown(r)
    assert str(commit_layer) in md
    assert str(n_layers) in md
