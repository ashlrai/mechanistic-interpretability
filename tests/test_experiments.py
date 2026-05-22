from pathlib import Path

import pytest

from mech_interp.experiments import load_experiment_specs
from mech_interp.experiments.registry import ExperimentSpecValidationError, load_experiment_spec


def test_load_experiment_specs() -> None:
    registry = load_experiment_specs("experiments")
    names = {spec.name for spec in registry.list()}

    assert "polysemanticity-smoke" in names
    assert "superposition-smoke" in names
    assert "circuit-patching-smoke" in names


def test_load_experiment_spec_supports_optional_typed_fields(tmp_path: Path) -> None:
    spec_path = tmp_path / "optional.yaml"
    spec_path.write_text(
        """
name: optional-smoke
family: polysemanticity
backend: transformerlens
description: Optional field coverage.
seed: 123
model: gpt2-small
prompts:
  - "A clean prompt"
artifact_policy:
  retain_activation_tensors: true
  write_manifest: false
  max_artifact_bytes: 1024
parameters:
  activation_site: resid_post
""",
        encoding="utf-8",
    )

    spec = load_experiment_spec(spec_path)

    assert spec.name == "optional-smoke"
    assert spec.parameters["seed"] == 123
    assert spec.parameters["model"] == "gpt2-small"
    assert spec.parameters["prompts"] == ["A clean prompt"]
    assert spec.parameters["artifact_policy"] == {
        "retain_activation_tensors": True,
        "write_manifest": False,
        "max_artifact_bytes": 1024,
    }


def test_load_experiment_spec_rejects_missing_required_fields(tmp_path: Path) -> None:
    spec_path = tmp_path / "missing.yaml"
    spec_path.write_text(
        """
name: missing-backend
family: polysemanticity
""",
        encoding="utf-8",
    )

    with pytest.raises(ExperimentSpecValidationError, match=str(spec_path)) as exc_info:
        load_experiment_spec(spec_path)

    assert "backend" in str(exc_info.value)
    assert "Field required" in str(exc_info.value)


def test_load_experiment_spec_rejects_unknown_family(tmp_path: Path) -> None:
    spec_path = tmp_path / "unknown-family.yaml"
    spec_path.write_text(
        """
name: unknown-family
family: attention_archaeology
backend: transformerlens
""",
        encoding="utf-8",
    )

    with pytest.raises(ExperimentSpecValidationError, match=str(spec_path)) as exc_info:
        load_experiment_spec(spec_path)

    assert "unsupported family 'attention_archaeology'" in str(exc_info.value)
    assert "polysemanticity" in str(exc_info.value)


def test_load_experiment_spec_rejects_unknown_backend(tmp_path: Path) -> None:
    spec_path = tmp_path / "unknown-backend.yaml"
    spec_path.write_text(
        """
name: unknown-backend
family: superposition
backend: mystery_backend
""",
        encoding="utf-8",
    )

    with pytest.raises(ExperimentSpecValidationError, match=str(spec_path)) as exc_info:
        load_experiment_spec(spec_path)

    assert "unsupported backend 'mystery_backend'" in str(exc_info.value)
    assert "transformerlens" in str(exc_info.value)


def test_load_experiment_spec_rejects_invalid_optional_fields(tmp_path: Path) -> None:
    spec_path = tmp_path / "invalid-optionals.yaml"
    spec_path.write_text(
        """
name: invalid-optionals
family: superposition
backend: transformerlens
seed: -1
prompts:
  - ""
artifact_policy:
  max_artifact_bytes: 0
""",
        encoding="utf-8",
    )

    with pytest.raises(ExperimentSpecValidationError, match=str(spec_path)) as exc_info:
        load_experiment_spec(spec_path)

    message = str(exc_info.value)
    assert "seed" in message
    assert "prompts.0" in message
    assert "artifact_policy.max_artifact_bytes" in message


def test_load_experiment_spec_rejects_non_mapping_yaml(tmp_path: Path) -> None:
    spec_path = tmp_path / "list.yaml"
    spec_path.write_text("- not\n- a\n- mapping\n", encoding="utf-8")

    with pytest.raises(ExperimentSpecValidationError, match=str(spec_path)) as exc_info:
        load_experiment_spec(spec_path)

    assert "expected a YAML mapping" in str(exc_info.value)


def test_load_experiment_specs_rejects_duplicate_names(tmp_path: Path) -> None:
    first_path = tmp_path / "a.yaml"
    second_path = tmp_path / "b.yaml"
    spec_yaml = """
name: duplicate
family: circuit_patching
backend: transformerlens
"""
    first_path.write_text(spec_yaml, encoding="utf-8")
    second_path.write_text(spec_yaml, encoding="utf-8")

    with pytest.raises(ExperimentSpecValidationError, match=str(second_path)) as exc_info:
        load_experiment_specs(tmp_path)

    message = str(exc_info.value)
    assert "Duplicate experiment spec name 'duplicate'" in message
    assert str(first_path) in message
