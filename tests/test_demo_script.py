"""Smoke tests for scripts/demo.sh.

Validates:
1. The script is well-formed bash (parses with ``bash -n``).
2. Every spec name referenced in the script is registered in the experiment registry.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
DEMO_SCRIPT = REPO_ROOT / "scripts" / "demo.sh"
EXPERIMENTS_DIR = REPO_ROOT / "experiments"

# Spec names used in demo.sh (keep in sync with the script's run_spec calls).
DEMO_SPEC_NAMES = [
    "direct-logit-attribution-factual",
    "circuit-patching-smoke",
    "attribution-patching-factual-recall",
    "acdc-lite-gpt2-factual",
    "polysemanticity-sae-smoke",
    "sparse-probing-factual-vs-random",
]


def _registered_spec_names() -> set[str]:
    """Read spec names from all YAML files in experiments/."""
    try:
        import yaml
    except ImportError:
        pytest.skip("PyYAML not available")

    names: set[str] = set()
    for yaml_file in EXPERIMENTS_DIR.glob("*.yaml"):
        try:
            data = yaml.safe_load(yaml_file.read_text(encoding="utf-8"))
            if isinstance(data, dict) and "name" in data:
                names.add(str(data["name"]))
        except Exception:  # noqa: BLE001
            pass
    return names


def test_demo_script_exists() -> None:
    assert DEMO_SCRIPT.exists(), f"demo.sh not found at {DEMO_SCRIPT}"


def test_demo_script_is_executable() -> None:
    assert DEMO_SCRIPT.exists()
    assert DEMO_SCRIPT.stat().st_mode & 0o111, "demo.sh is not executable"


def test_demo_script_parses_as_valid_bash() -> None:
    """bash -n performs syntax check without execution."""
    result = subprocess.run(
        ["bash", "-n", str(DEMO_SCRIPT)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"demo.sh failed bash -n syntax check:\n{result.stderr}"
    )


def test_demo_script_references_known_spec_names() -> None:
    """Every spec name in DEMO_SPEC_NAMES must be registered in experiments/."""
    registered = _registered_spec_names()
    missing = [name for name in DEMO_SPEC_NAMES if name not in registered]
    assert not missing, (
        f"demo.sh references spec names not found in experiments/: {missing}"
    )


def test_demo_spec_names_appear_in_script() -> None:
    """Each expected spec name must appear literally in demo.sh."""
    content = DEMO_SCRIPT.read_text(encoding="utf-8")
    missing = [name for name in DEMO_SPEC_NAMES if name not in content]
    assert not missing, (
        f"Expected spec names not found in demo.sh: {missing}"
    )


def test_demo_script_has_mech_runs_call() -> None:
    content = DEMO_SCRIPT.read_text(encoding="utf-8")
    assert "mech runs" in content, "demo.sh must call 'mech runs'"


def test_demo_script_has_report_runs_call() -> None:
    content = DEMO_SCRIPT.read_text(encoding="utf-8")
    assert "mech report-runs" in content, "demo.sh must call 'mech report-runs'"


def test_demo_script_skips_hf_token_families() -> None:
    """Families requiring HF_TOKEN must not be in run_spec calls."""
    content = DEMO_SCRIPT.read_text(encoding="utf-8")
    # Extract lines that call run_spec.
    run_lines = [ln for ln in content.splitlines() if ln.strip().startswith("run_spec")]
    run_content = "\n".join(run_lines)
    # These families need HF_TOKEN and must not appear as active run_spec calls.
    excluded_specs = [
        "refusal-direction",
        "sae-cross-model",
        "caa-steering",
        "causal-scrubbing",
    ]
    for excluded in excluded_specs:
        assert excluded not in run_content, (
            f"demo.sh run_spec calls include '{excluded}' which requires HF_TOKEN"
        )


def test_demo_script_shebang() -> None:
    content = DEMO_SCRIPT.read_text(encoding="utf-8")
    first_line = content.splitlines()[0]
    assert first_line.startswith("#!/usr/bin/env bash"), (
        f"demo.sh must start with '#!/usr/bin/env bash', got: {first_line!r}"
    )


def test_demo_script_has_set_euo_pipefail() -> None:
    content = DEMO_SCRIPT.read_text(encoding="utf-8")
    assert "set -euo pipefail" in content, (
        "demo.sh must contain 'set -euo pipefail'"
    )
