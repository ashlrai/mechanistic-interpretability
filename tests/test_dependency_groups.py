"""Verify pyproject.toml extras and dependency groups are well-formed.

Loads pyproject.toml via tomllib (stdlib >=3.11) — no network required.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

PYPROJECT = Path(__file__).parent.parent / "pyproject.toml"

# sae-lens is intentionally NOT in extras: all published versions pin
# typer<0.13 which conflicts with our CLI (typer>=0.13). Install sae-lens
# manually. The compat shim (sae/compat.py) uses dynamic import.
EXPECTED_EXTRAS = {"interp", "nnsight", "apple", "gradio"}
EXPECTED_DEV_DEPS = {"pytest", "ruff", "mypy", "mkdocs", "mkdocs-material"}
EXPECTED_INTERP_DEPS = {"torch", "transformer-lens", "transformers", "safetensors", "einops"}


def _toml() -> dict:  # type: ignore[type-arg]
    return tomllib.loads(PYPROJECT.read_text())


def test_pyproject_exists() -> None:
    assert PYPROJECT.exists()


def test_expected_extras_present() -> None:
    data = _toml()
    extras = set(data["project"]["optional-dependencies"].keys())
    missing = EXPECTED_EXTRAS - extras
    assert not missing, f"Missing extras: {missing}"


def test_interp_extra_deps() -> None:
    data = _toml()
    interp = data["project"]["optional-dependencies"]["interp"]
    names = {dep.split(">=")[0].split(",")[0].strip() for dep in interp}
    missing = EXPECTED_INTERP_DEPS - names
    assert not missing, f"Missing from interp extra: {missing}"


def test_interp_deps_have_upper_bounds() -> None:
    """All interp deps should have an upper-bound to prevent silent breakage."""
    data = _toml()
    interp = data["project"]["optional-dependencies"]["interp"]
    for dep in interp:
        assert "<" in dep, f"interp dep missing upper bound: {dep!r}"


def test_core_deps_have_upper_bounds() -> None:
    data = _toml()
    for dep in data["project"]["dependencies"]:
        assert "<" in dep, f"core dep missing upper bound: {dep!r}"


def test_dev_group_deps_present() -> None:
    data = _toml()
    dev = data["dependency-groups"]["dev"]
    # dev group entries are strings like "pytest>=8.2.0"
    names = {dep.split(">=")[0].split(",")[0].strip() for dep in dev}
    missing = EXPECTED_DEV_DEPS - names
    assert not missing, f"Missing from dev group: {missing}"


def test_requires_python() -> None:
    data = _toml()
    assert data["project"]["requires-python"] == ">=3.11"
