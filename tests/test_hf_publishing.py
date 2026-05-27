"""Unit tests for the HuggingFace Hub publishing pipeline.

All tests use mocked huggingface_hub calls — nothing is actually uploaded.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_sae_run_dir(tmp_path: Path, run_id: int = 51) -> Path:
    """Create a minimal fake SAE run artifact directory."""
    run_dir = tmp_path / f"run-{run_id:06d}"
    run_dir.mkdir(parents=True)

    # sae_weights.safetensors (fake binary)
    (run_dir / "sae_weights.safetensors").write_bytes(b"\x00" * 16)

    # sae_weights.safetensors.json (config sidecar)
    (run_dir / "sae_weights.safetensors.json").write_text(
        json.dumps(
            {
                "dtype": "torch.float32",
                "input_dim": 1024,
                "k": 32,
                "n_features": 2048,
            }
        ),
        encoding="utf-8",
    )

    # environment.json
    (run_dir / "environment.json").write_text(
        json.dumps(
            {
                "backend": "transformerlens",
                "family": "polysemanticity_sae",
                "model_name": "gpt2-medium",
                "python_version": "3.12.2",
                "package_versions": {"torch": "2.12.0"},
            }
        ),
        encoding="utf-8",
    )

    # spec.json
    (run_dir / "spec.json").write_text(
        json.dumps(
            {
                "description": "Top-K SAE on gpt2-medium.",
                "family": "polysemanticity_sae",
                "name": "polysemanticity-sae-gpt2-medium",
                "parameters": {
                    "hook_site": "blocks.12.hook_resid_pre",
                    "model": "gpt2-medium",
                    "n_features": 2048,
                    "k": 32,
                },
            }
        ),
        encoding="utf-8",
    )

    # result.json
    (run_dir / "result.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "status": "succeeded",
                "metrics": {
                    "explained_variance": 0.9748,
                    "live_features": 473,
                    "n_tokens": 8125,
                    "n_features": 2048,
                    "k": 32,
                },
            }
        ),
        encoding="utf-8",
    )

    # feature_analysis.json (small fake)
    (run_dir / "feature_analysis.json").write_text(
        json.dumps({"features": []}),
        encoding="utf-8",
    )

    return tmp_path


def _make_steering_files(tmp_path: Path, name: str = "sentiment-gpt2-medium-l8") -> Path:
    """Create fake steering vector files under tmp_path/data/steering/."""
    steering_dir = tmp_path / "data" / "steering"
    steering_dir.mkdir(parents=True)

    fname = name.replace("-", "_") + ".safetensors"
    (steering_dir / fname).write_bytes(b"\x00" * 8)
    (steering_dir / (fname + ".json")).write_text(
        json.dumps(
            {
                "name": name,
                "model": "gpt2-medium",
                "hook_site": "blocks.8.hook_resid_pre",
                "direction_norm": 15.36,
                "extraction_quality": 1.68,
                "license": "research-only",
                "source_paper": "Zou et al. 2023",
            }
        ),
        encoding="utf-8",
    )
    return tmp_path


def _make_investigation_dir(tmp_path: Path, slug: str = "sae_replication_crisis") -> Path:
    """Create minimal investigation docs structure."""
    inv_dir = tmp_path / "investigations"
    inv_dir.mkdir(parents=True)
    (inv_dir / f"{slug}.md").write_text(
        "# SAE Replication Crisis\n\nThis is the body of the investigation.\n",
        encoding="utf-8",
    )
    return tmp_path


# ---------------------------------------------------------------------------
# SAE bundle tests
# ---------------------------------------------------------------------------


def test_build_sae_bundle_includes_all_artifacts(tmp_path: Path) -> None:
    from mech_interp.publishing.hf_upload import build_sae_bundle

    _make_sae_run_dir(tmp_path, run_id=51)
    bundle = build_sae_bundle(51, artifact_root=tmp_path)

    assert bundle.kind == "sae"
    assert bundle.name.startswith("sae-run51")
    file_names = {p.name for p in bundle.local_paths}
    assert "sae_weights.safetensors" in file_names
    assert "sae_weights.safetensors.json" in file_names
    assert "environment.json" in file_names
    assert "feature_analysis.json" in file_names


def test_build_sae_bundle_metadata_contains_run_id(tmp_path: Path) -> None:
    from mech_interp.publishing.hf_upload import build_sae_bundle

    _make_sae_run_dir(tmp_path, run_id=51)
    bundle = build_sae_bundle(51, artifact_root=tmp_path)

    assert bundle.metadata["run_id"] == 51
    assert "spec" in bundle.metadata
    assert "environment" in bundle.metadata


def test_build_sae_bundle_missing_run_raises(tmp_path: Path) -> None:
    from mech_interp.publishing.hf_upload import build_sae_bundle

    with pytest.raises(FileNotFoundError, match="not found"):
        build_sae_bundle(999, artifact_root=tmp_path)


# ---------------------------------------------------------------------------
# Steering bundle tests
# ---------------------------------------------------------------------------


def test_build_steering_bundle_includes_safetensors_and_sidecar(tmp_path: Path) -> None:
    from mech_interp.publishing.hf_upload import build_steering_bundle
    from mech_interp.steering.registry import STEERING_REGISTRY, SteeringVectorDescriptor

    name = "test-sentiment-vector"
    # Inject a test descriptor into the registry temporarily
    steering_dir = tmp_path / "data" / "steering"
    steering_dir.mkdir(parents=True)
    st_file = steering_dir / "test_sentiment_vector.safetensors"
    st_file.write_bytes(b"\x00" * 8)
    sidecar = steering_dir / "test_sentiment_vector.safetensors.json"
    sidecar.write_text(json.dumps({"extraction_quality": 1.5}), encoding="utf-8")

    descriptor = SteeringVectorDescriptor(
        name=name,
        model_name="gpt2-medium",
        hook_site="blocks.8.hook_resid_pre",
        direction_norm=15.36,
        description="Test steering vector.",
        license="research-only",
        local_path=Path("data/steering/test_sentiment_vector.safetensors"),
    )

    patched_registry = {**STEERING_REGISTRY, name: descriptor}
    with patch("mech_interp.steering.registry.STEERING_REGISTRY", patched_registry):
        bundle = build_steering_bundle(name, base_dir=tmp_path)

    assert bundle.kind == "steering"
    assert bundle.name == name
    file_names = {p.name for p in bundle.local_paths}
    assert "test_sentiment_vector.safetensors" in file_names
    assert "test_sentiment_vector.safetensors.json" in file_names


def test_build_steering_bundle_unknown_raises() -> None:
    from mech_interp.publishing.hf_upload import build_steering_bundle

    with pytest.raises(KeyError, match="Unknown steering vector"):
        build_steering_bundle("nonexistent-vector-xyz")


def test_build_steering_bundle_metadata_has_model_fields(tmp_path: Path) -> None:
    from mech_interp.publishing.hf_upload import build_steering_bundle
    from mech_interp.steering.registry import STEERING_REGISTRY, SteeringVectorDescriptor

    name = "test-metadata-vector"
    steering_dir = tmp_path / "data" / "steering"
    steering_dir.mkdir(parents=True)
    st_file = steering_dir / "test_metadata_vector.safetensors"
    st_file.write_bytes(b"\x00" * 8)

    descriptor = SteeringVectorDescriptor(
        name=name,
        model_name="gpt2-test",
        hook_site="blocks.5.hook_resid_pre",
        direction_norm=7.0,
        description="Metadata test.",
        license="research-only",
        local_path=Path("data/steering/test_metadata_vector.safetensors"),
    )
    patched_registry = {**STEERING_REGISTRY, name: descriptor}
    with patch("mech_interp.steering.registry.STEERING_REGISTRY", patched_registry):
        bundle = build_steering_bundle(name, base_dir=tmp_path)

    assert bundle.metadata["model_name"] == "gpt2-test"
    assert bundle.metadata["hook_site"] == "blocks.5.hook_resid_pre"
    assert bundle.metadata["kind"] == "steering"


# ---------------------------------------------------------------------------
# Investigation bundle tests
# ---------------------------------------------------------------------------


def test_build_investigation_bundle_handles_missing_publication_dir(tmp_path: Path) -> None:
    from mech_interp.publishing.hf_upload import build_investigation_bundle

    slug = "sae_replication_crisis"
    docs_dir = _make_investigation_dir(tmp_path, slug)
    # No publications dir — should not raise
    bundle = build_investigation_bundle(slug, docs_dir=docs_dir)

    assert bundle.kind == "investigation"
    assert bundle.name == slug
    assert bundle.metadata["has_publication_artifacts"] is False
    assert any(p.name == "sae_replication_crisis.md" for p in bundle.local_paths)


def test_build_investigation_bundle_includes_publication_artifacts(tmp_path: Path) -> None:
    from mech_interp.publishing.hf_upload import build_investigation_bundle

    slug = "sae_replication_crisis"
    docs_dir = _make_investigation_dir(tmp_path, slug)

    # Create a publications artifacts directory
    pub_dir = docs_dir / "publications" / "sae_replication_crisis_artifacts"
    pub_dir.mkdir(parents=True)
    (pub_dir / "metrics.json").write_text(json.dumps({"f1": 0.9}), encoding="utf-8")
    (pub_dir / "paper.md").write_text("# Paper\n", encoding="utf-8")

    bundle = build_investigation_bundle(slug, docs_dir=docs_dir)

    assert bundle.metadata["has_publication_artifacts"] is True
    file_names = {p.name for p in bundle.local_paths}
    assert "metrics.json" in file_names
    assert "paper.md" in file_names


def test_build_investigation_bundle_missing_slug_raises(tmp_path: Path) -> None:
    from mech_interp.publishing.hf_upload import build_investigation_bundle

    docs_dir = tmp_path
    (docs_dir / "investigations").mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="Investigation not found"):
        build_investigation_bundle("nonexistent_slug_xyz", docs_dir=docs_dir)


def test_build_investigation_bundle_extracts_title(tmp_path: Path) -> None:
    from mech_interp.publishing.hf_upload import build_investigation_bundle

    docs_dir = tmp_path
    inv_dir = docs_dir / "investigations"
    inv_dir.mkdir(parents=True)
    (inv_dir / "my_investigation.md").write_text(
        "# My Great Investigation\n\nSome content here.\n",
        encoding="utf-8",
    )

    bundle = build_investigation_bundle("my_investigation", docs_dir=docs_dir)
    assert bundle.metadata["title"] == "My Great Investigation"


# ---------------------------------------------------------------------------
# upload_bundle dry-run / real-upload tests
# ---------------------------------------------------------------------------


def test_upload_bundle_dry_run_does_not_call_hub(tmp_path: Path, capsys: Any) -> None:
    from mech_interp.publishing.hf_upload import HubArtifactBundle, upload_bundle

    bundle = HubArtifactBundle(
        name="test-bundle",
        kind="steering",
        local_paths=[],
        metadata={"kind": "steering", "name": "test-bundle", "model_name": "gpt2", "hook_site": "blocks.5.hook_resid_pre", "direction_norm": 5.0, "description": "test", "license": "research-only"},  # noqa: E501
        license="research-only",
    )

    with patch("huggingface_hub.HfApi") as mock_api, \
         patch("huggingface_hub.create_repo") as mock_create:
        url = upload_bundle(bundle, repo_id="testuser/test-bundle", dry_run=True)

    mock_api.assert_not_called()
    mock_create.assert_not_called()
    assert url == "https://huggingface.co/testuser/test-bundle"


def test_upload_bundle_calls_create_repo_when_requested(tmp_path: Path) -> None:
    from mech_interp.publishing.hf_upload import HubArtifactBundle, upload_bundle

    bundle = HubArtifactBundle(
        name="test-sae",
        kind="sae",
        local_paths=[],
        metadata={"kind": "sae", "run_id": 1, "license": "research-only"},
        license="research-only",
    )

    mock_api_instance = MagicMock()
    mock_api_class = MagicMock(return_value=mock_api_instance)

    with patch("mech_interp.publishing.hf_upload.HfApi", mock_api_class), \
         patch("mech_interp.publishing.hf_upload.hf_create_repo") as mock_create, \
         patch("mech_interp.publishing.hf_upload._stage_bundle"):
        url = upload_bundle(
            bundle,
            repo_id="testuser/test-sae",
            create_repo=True,
            dry_run=False,
        )

    mock_create.assert_called_once()
    call_kwargs = mock_create.call_args
    assert call_kwargs.kwargs.get("repo_id") == "testuser/test-sae" or \
           call_kwargs.args[0] == "testuser/test-sae"
    assert url == "https://huggingface.co/testuser/test-sae"


def test_upload_bundle_skips_create_repo_when_not_requested(tmp_path: Path) -> None:
    from mech_interp.publishing.hf_upload import HubArtifactBundle, upload_bundle

    bundle = HubArtifactBundle(
        name="test-sae",
        kind="sae",
        local_paths=[],
        metadata={"kind": "sae", "run_id": 1, "license": "research-only"},
        license="research-only",
    )

    mock_api_instance = MagicMock()
    mock_api_class = MagicMock(return_value=mock_api_instance)

    with patch("mech_interp.publishing.hf_upload.HfApi", mock_api_class), \
         patch("mech_interp.publishing.hf_upload.hf_create_repo") as mock_create, \
         patch("mech_interp.publishing.hf_upload._stage_bundle"):
        upload_bundle(
            bundle,
            repo_id="testuser/test-sae",
            create_repo=False,
            dry_run=False,
        )

    mock_create.assert_not_called()


# ---------------------------------------------------------------------------
# README frontmatter tests
# ---------------------------------------------------------------------------


def test_generated_readme_has_yaml_frontmatter(tmp_path: Path) -> None:
    from mech_interp.publishing.hf_upload import (
        HubArtifactBundle,
        _generate_readme,
    )

    # Test all three kinds
    for kind, metadata in [
        (
            "sae",
            {
                "run_id": 51,
                "kind": "sae",
                "license": "research-only",
                "spec": {"description": "Test SAE.", "parameters": {"model": "gpt2-medium", "hook_site": "blocks.12.hook_resid_pre", "n_features": 2048, "k": 32}},  # noqa: E501
                "result": {"run_id": 51, "metrics": {"explained_variance": 0.97, "live_features": 473, "n_tokens": 8125}},  # noqa: E501
                "environment": {"model_name": "gpt2-medium", "python_version": "3.12.2", "package_versions": {"torch": "2.12.0"}},  # noqa: E501
            },
        ),
        (
            "steering",
            {
                "name": "sentiment-gpt2-medium-l8",
                "kind": "steering",
                "model_name": "gpt2-medium",
                "hook_site": "blocks.8.hook_resid_pre",
                "direction_norm": 15.36,
                "description": "Test steering vector.",
                "license": "research-only",
                "source_paper": "Zou et al. 2023",
                "source_run_id": None,
            },
        ),
        (
            "investigation",
            {
                "slug": "sae_replication_crisis",
                "kind": "investigation",
                "title": "SAE Replication Crisis",
                "description": "A test investigation.",
                "license": "CC-BY-4.0",
                "has_publication_artifacts": False,
            },
        ),
    ]:
        bundle = HubArtifactBundle(
            name="test",
            kind=kind,
            local_paths=[],
            metadata=metadata,
            license=str(metadata.get("license", "research-only")),
        )
        readme = _generate_readme(bundle, "testuser/test-repo")

        assert readme.startswith("---\n"), f"README for kind={kind} missing YAML frontmatter"
        assert "license:" in readme
        assert "tags:" in readme
        assert "library_name: mech-interpretability" in readme
        # Check frontmatter closes
        lines = readme.splitlines()
        frontmatter_closes = sum(1 for line in lines if line.strip() == "---")
        assert frontmatter_closes >= 2, f"YAML frontmatter not closed for kind={kind}"
