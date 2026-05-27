"""Smoke tests for notebooks: well-formed JSON, ≥5 cells, kernel spec present."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

NOTEBOOKS_DIR = Path(__file__).parent.parent / "notebooks"

EXPECTED_NOTEBOOKS = [
    "01_circuit_patching.ipynb",
    "02_polysemanticity_sae.ipynb",
    "03_acdc_lite.ipynb",
    "04_agentic_loop.ipynb",
    "05_research_walkthrough.ipynb",
]

# Section markers expected in the research-walkthrough notebook.
WALKTHROUGH_SECTIONS = [
    "Section 1",
    "Section 2",
    "Section 3",
    "Section 4",
    "Section 5",
    "Section 6",
    "Section 7",
    "Section 8",
    "Section 9",
]


@pytest.mark.parametrize("nb_name", EXPECTED_NOTEBOOKS)
def test_notebook_is_valid_json(nb_name: str) -> None:
    path = NOTEBOOKS_DIR / nb_name
    assert path.exists(), f"Notebook not found: {path}"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(data, dict), f"{nb_name}: top-level must be a JSON object"


@pytest.mark.parametrize("nb_name", EXPECTED_NOTEBOOKS)
def test_notebook_has_at_least_five_cells(nb_name: str) -> None:
    path = NOTEBOOKS_DIR / nb_name
    data = json.loads(path.read_text(encoding="utf-8"))
    cells = data.get("cells", [])
    assert len(cells) >= 5, (
        f"{nb_name}: expected ≥5 cells, got {len(cells)}"
    )


@pytest.mark.parametrize("nb_name", EXPECTED_NOTEBOOKS)
def test_notebook_has_kernel_spec(nb_name: str) -> None:
    path = NOTEBOOKS_DIR / nb_name
    data = json.loads(path.read_text(encoding="utf-8"))
    metadata = data.get("metadata", {})
    kernelspec = metadata.get("kernelspec")
    assert kernelspec is not None, (
        f"{nb_name}: missing kernelspec in metadata"
    )
    assert "name" in kernelspec, (
        f"{nb_name}: kernelspec missing 'name' key"
    )


@pytest.mark.parametrize("nb_name", EXPECTED_NOTEBOOKS)
def test_notebook_has_nbformat(nb_name: str) -> None:
    path = NOTEBOOKS_DIR / nb_name
    data = json.loads(path.read_text(encoding="utf-8"))
    assert "nbformat" in data, f"{nb_name}: missing 'nbformat' key"
    assert data["nbformat"] >= 4, f"{nb_name}: nbformat must be ≥ 4"


@pytest.mark.parametrize("nb_name", EXPECTED_NOTEBOOKS)
def test_notebook_cells_have_source(nb_name: str) -> None:
    path = NOTEBOOKS_DIR / nb_name
    data = json.loads(path.read_text(encoding="utf-8"))
    for i, cell in enumerate(data.get("cells", [])):
        assert "source" in cell, (
            f"{nb_name} cell {i}: missing 'source' key"
        )
        assert "cell_type" in cell, (
            f"{nb_name} cell {i}: missing 'cell_type' key"
        )


def test_all_expected_notebooks_exist() -> None:
    missing = [
        nb for nb in EXPECTED_NOTEBOOKS
        if not (NOTEBOOKS_DIR / nb).exists()
    ]
    assert not missing, f"Missing notebooks: {missing}"


def test_walkthrough_notebook_has_all_sections() -> None:
    """05_research_walkthrough.ipynb must contain all nine section markers."""
    nb_name = "05_research_walkthrough.ipynb"
    path = NOTEBOOKS_DIR / nb_name
    assert path.exists(), f"Notebook not found: {path}"
    data = json.loads(path.read_text(encoding="utf-8"))

    # Collect all source text from markdown cells.
    all_source = ""
    for cell in data.get("cells", []):
        if cell.get("cell_type") == "markdown":
            src = cell.get("source", "")
            if isinstance(src, list):
                all_source += "".join(src)
            else:
                all_source += src

    missing_sections = [s for s in WALKTHROUGH_SECTIONS if s not in all_source]
    assert not missing_sections, (
        f"{nb_name}: missing section markers: {missing_sections}"
    )


def test_walkthrough_notebook_has_nine_or_more_code_cells() -> None:
    """05_research_walkthrough.ipynb must have at least 9 code cells (one per section)."""
    nb_name = "05_research_walkthrough.ipynb"
    path = NOTEBOOKS_DIR / nb_name
    data = json.loads(path.read_text(encoding="utf-8"))
    code_cells = [c for c in data.get("cells", []) if c.get("cell_type") == "code"]
    assert len(code_cells) >= 9, (
        f"{nb_name}: expected ≥9 code cells, got {len(code_cells)}"
    )
