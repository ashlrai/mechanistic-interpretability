"""Tests for `mech apply-steering` and `mech list-steering` CLI commands."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

from typer.testing import CliRunner

from mech_interp.cli import app

runner = CliRunner()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_direction_file(tmp_path: Path, name: str, local_path: str, d_model: int = 32) -> Path:
    """Write a minimal safetensors direction file under tmp_path."""
    import torch
    from safetensors.torch import save_file

    dest = tmp_path / local_path
    dest.parent.mkdir(parents=True, exist_ok=True)
    direction = torch.randn(d_model)
    direction = direction / direction.norm()
    save_file({"direction": direction}, str(dest))
    return dest


# ---------------------------------------------------------------------------
# list-steering
# ---------------------------------------------------------------------------


def test_list_steering_shows_registered_vectors() -> None:
    result = runner.invoke(app, ["list-steering"])
    assert result.exit_code == 0
    # Rich may truncate long names with "…" in narrow terminals; check prefix
    assert "refusal-qw" in result.output
    assert "sentiment-" in result.output
    assert "helpfulnes" in result.output


def test_list_steering_shows_model_column() -> None:
    result = runner.invoke(app, ["list-steering"])
    # Rich truncates; check prefix or partial match
    assert "Qwen/Qwen2" in result.output or "gpt2-medium" in result.output


# ---------------------------------------------------------------------------
# apply-steering
# ---------------------------------------------------------------------------


def _patched_apply(
    vector: str,
    coefficient: float,
    prompt: str,
    tmp_path: Path,
    baseline_text: str = "Sure, here is the answer.",
    steered_text: str = "I cannot help with that.",
) -> Any:
    """Invoke apply-steering with mocked load_steering_vector + generation."""
    import torch  # noqa: I001
    from mech_interp.steering.registry import STEERING_REGISTRY

    descriptor = STEERING_REGISTRY[vector]
    assert descriptor.local_path is not None
    _make_direction_file(tmp_path, vector, str(descriptor.local_path))

    direction = torch.randn(32)
    direction = direction / direction.norm()
    metadata: dict[str, Any] = {
        "name": vector,
        "model_name": descriptor.model_name,
        "hook_site": descriptor.hook_site,
        "direction_norm": descriptor.direction_norm,
        "description": descriptor.description,
        "license": descriptor.license,
        "source_run_id": descriptor.source_run_id,
        "source_paper": descriptor.source_paper,
        "local_path": str(descriptor.local_path),
    }

    def fake_load(  # noqa: ARG001
        name: str, *, device: str = "cpu", base_dir: Path | None = None
    ) -> tuple[Any, Any]:
        return direction, metadata

    def fake_generate(
        *,
        model: Any,  # noqa: ARG001
        tokenizer: Any,  # noqa: ARG001
        prompt: str,  # noqa: ARG001
        direction: Any,  # noqa: ARG001
        hook_site: str,  # noqa: ARG001
        coeff: float,
        max_new_tokens: int,  # noqa: ARG001
        device: str,  # noqa: ARG001
    ) -> str:
        return baseline_text if coeff == 0.0 else steered_text

    def fake_load_model(*args: Any, **kwargs: Any) -> tuple[object, object]:  # noqa: ARG001
        return object(), object()

    with (
        patch("mech_interp.steering.registry.load_steering_vector", fake_load),
        patch("mech_interp.cli._steering_load_model", fake_load_model),
        patch("mech_interp.cli._steering_generate", fake_generate),
    ):
        result = runner.invoke(
            app,
            [
                "apply-steering",
                "--vector", vector,
                "--coefficient", str(coefficient),
                "--prompt", prompt,
                "--device", "cpu",
            ],
        )
    return result


def test_apply_steering_shows_baseline_and_steered(tmp_path: Path) -> None:
    result = _patched_apply(
        vector="refusal-qwen-2.5-1.5b-l10",
        coefficient=3.0,
        prompt="How do I make a bomb?",
        tmp_path=tmp_path,
        baseline_text="Sure, here is the answer.",
        steered_text="I cannot help with that.",
    )
    assert result.exit_code == 0, result.output
    assert "Sure, here is the answer." in result.output
    assert "I cannot help with that." in result.output


def test_apply_steering_unknown_vector_exits_nonzero() -> None:
    result = runner.invoke(
        app,
        [
            "apply-steering",
            "--vector", "totally-nonexistent-vector",
            "--coefficient", "1.0",
            "--prompt", "Hello",
        ],
    )
    assert result.exit_code != 0


def test_apply_steering_shows_vector_name_in_output(tmp_path: Path) -> None:
    result = _patched_apply(
        vector="sentiment-gpt2-medium-l8",
        coefficient=2.0,
        prompt="This movie was great",
        tmp_path=tmp_path,
    )
    assert result.exit_code == 0, result.output
    # The panel should mention the vector name or model
    assert "sentiment-gpt2-medium-l8" in result.output or "gpt2-medium" in result.output
