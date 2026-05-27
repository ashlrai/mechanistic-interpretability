"""Verify check.sh contains the expected commands and structural elements.

This test parses scripts/check.sh as text — no subprocess execution.
"""

from __future__ import annotations

from pathlib import Path

SCRIPT = Path(__file__).parent.parent / "scripts" / "check.sh"


def _lines() -> list[str]:
    return SCRIPT.read_text().splitlines()


def test_check_script_exists() -> None:
    assert SCRIPT.exists(), "scripts/check.sh not found"


def test_check_script_has_strict_mode() -> None:
    text = SCRIPT.read_text()
    assert "set -euo pipefail" in text, "check.sh must use set -euo pipefail"


def test_check_script_has_fast_and_full_modes() -> None:
    text = SCRIPT.read_text()
    assert "--fast" in text, "check.sh must document --fast mode"
    assert "--full" in text, "check.sh must document --full mode"


def test_check_script_runs_ruff() -> None:
    text = SCRIPT.read_text()
    assert "ruff check src tests" in text


def test_check_script_runs_mypy() -> None:
    text = SCRIPT.read_text()
    assert "mypy src tests" in text


def test_check_script_runs_pytest_ignoring_integration() -> None:
    text = SCRIPT.read_text()
    assert "--ignore=tests/integration" in text, "fast mode must ignore integration tests"


def test_check_script_runs_mech_validate() -> None:
    text = SCRIPT.read_text()
    assert "mech validate" in text


def test_check_script_runs_mkdocs_build_strict() -> None:
    text = SCRIPT.read_text()
    assert "mkdocs build --strict" in text, "check.sh must build docs in strict mode"


def test_check_script_integration_opt_in() -> None:
    text = SCRIPT.read_text()
    assert "tests/integration" in text, "integration test path must appear"
    assert "RUN_INTEGRATION_TESTS" in text or "--full" in text
