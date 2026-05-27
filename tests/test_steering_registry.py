"""Tests for the steering-vector registry."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest


def test_registry_has_known_entries() -> None:
    from mech_interp.steering.registry import STEERING_REGISTRY

    assert "refusal-qwen-2.5-1.5b-l10" in STEERING_REGISTRY
    assert "sentiment-gpt2-medium-l8" in STEERING_REGISTRY
    assert "helpfulness-qwen-2.5-0.5b-l8" in STEERING_REGISTRY

    for descriptor in STEERING_REGISTRY.values():
        assert descriptor.name
        assert descriptor.model_name
        assert descriptor.hook_site
        assert descriptor.license
        assert descriptor.description
        assert descriptor.local_path is not None


def test_registry_entries_are_frozen() -> None:
    from mech_interp.steering.registry import STEERING_REGISTRY

    descriptor = STEERING_REGISTRY["refusal-qwen-2.5-1.5b-l10"]
    with pytest.raises((AttributeError, TypeError)):
        descriptor.name = "modified"  # type: ignore[misc]


def test_load_steering_vector_unknown_raises() -> None:
    from mech_interp.steering.registry import load_steering_vector

    with pytest.raises(KeyError, match="Unknown steering vector"):
        load_steering_vector("nonexistent-vector-xyz")


def test_load_steering_vector_missing_file_raises(tmp_path: Path) -> None:
    from mech_interp.steering.registry import load_steering_vector

    with pytest.raises(FileNotFoundError, match="not found"):
        load_steering_vector("refusal-qwen-2.5-1.5b-l10", base_dir=tmp_path)


def test_load_steering_vector_returns_direction_and_metadata(tmp_path: Path) -> None:
    import torch
    from safetensors.torch import save_file

    from mech_interp.steering.registry import STEERING_REGISTRY, load_steering_vector

    name = "refusal-qwen-2.5-1.5b-l10"
    descriptor = STEERING_REGISTRY[name]
    assert descriptor.local_path is not None

    # Build the expected file path under tmp_path
    dest = tmp_path / descriptor.local_path
    dest.parent.mkdir(parents=True, exist_ok=True)
    d_model = 1536
    direction_tensor = torch.randn(d_model)
    direction_tensor = direction_tensor / direction_tensor.norm()
    save_file({"direction": direction_tensor}, str(dest))

    direction, metadata = load_steering_vector(name, device="cpu", base_dir=tmp_path)

    assert direction.shape == (d_model,)
    assert metadata["name"] == name
    assert metadata["model_name"] == descriptor.model_name
    assert metadata["hook_site"] == descriptor.hook_site
    assert metadata["license"] == descriptor.license


def test_load_steering_vector_reads_sidecar_json(tmp_path: Path) -> None:
    import torch
    from safetensors.torch import save_file

    from mech_interp.steering.registry import STEERING_REGISTRY, load_steering_vector

    name = "refusal-qwen-2.5-1.5b-l10"
    descriptor = STEERING_REGISTRY[name]
    assert descriptor.local_path is not None

    dest = tmp_path / descriptor.local_path
    dest.parent.mkdir(parents=True, exist_ok=True)
    direction_tensor = torch.randn(32)
    direction_tensor = direction_tensor / direction_tensor.norm()
    save_file({"direction": direction_tensor}, str(dest))

    # Write a sidecar JSON
    sidecar: dict[str, Any] = {
        "extraction_quality": 4.105,
        "hidden_dim": 32,
        "harmful_prompt_count": 5,
    }
    sidecar_path = dest.with_suffix(".safetensors.json")
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")

    _, metadata = load_steering_vector(name, device="cpu", base_dir=tmp_path)
    assert metadata["extraction_quality"] == pytest.approx(4.105)
    assert metadata["hidden_dim"] == 32


def test_load_steering_vector_bad_keys_raises(tmp_path: Path) -> None:
    import torch
    from safetensors.torch import save_file

    from mech_interp.steering.registry import STEERING_REGISTRY, load_steering_vector

    name = "refusal-qwen-2.5-1.5b-l10"
    descriptor = STEERING_REGISTRY[name]
    assert descriptor.local_path is not None

    dest = tmp_path / descriptor.local_path
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Save with a wrong key name
    save_file({"wrong_key": torch.randn(32)}, str(dest))

    with pytest.raises(ValueError, match="direction"):
        load_steering_vector(name, device="cpu", base_dir=tmp_path)


def test_load_steering_vector_no_local_path_raises() -> None:
    """A descriptor with no local_path and no HF repo should raise ValueError."""
    from mech_interp.steering.registry import SteeringVectorDescriptor, load_steering_vector

    bare = SteeringVectorDescriptor(
        name="bare-test",
        model_name="gpt2",
        hook_site="blocks.0.hook_resid_pre",
        direction_norm=1.0,
        description="test",
        license="MIT",
        local_path=None,
    )
    with patch(
        "mech_interp.steering.registry.STEERING_REGISTRY",
        {"bare-test": bare},
    ):
        with pytest.raises(ValueError, match="no local_path"):
            load_steering_vector("bare-test")
