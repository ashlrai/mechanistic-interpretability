"""Tests for the pretrained SAE registry (HF download + load)."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest


def test_registry_has_known_entries() -> None:
    from mech_interp.sae.registry import SAE_REGISTRY

    assert "neelnanda-gpt2-small-res-jb-l8" in SAE_REGISTRY
    for descriptor in SAE_REGISTRY.values():
        assert descriptor.config["input_dim"] > 0
        assert descriptor.config["n_features"] > 0
        assert descriptor.hf_repo
        assert descriptor.weights_file
        assert descriptor.license


def test_download_sae_invokes_huggingface_hub(tmp_path: Path) -> None:
    from mech_interp.sae.registry import SAE_REGISTRY, download_sae

    descriptor = SAE_REGISTRY["neelnanda-gpt2-small-res-jb-l8"]
    fake_weights = tmp_path / "fake_weights.safetensors"
    fake_weights.write_bytes(b"fake")

    with patch("huggingface_hub.hf_hub_download", return_value=str(fake_weights)) as mock_dl:
        local = download_sae("neelnanda-gpt2-small-res-jb-l8", dest_dir=tmp_path)
        mock_dl.assert_called_once()
        kwargs = mock_dl.call_args.kwargs
        assert kwargs["repo_id"] == descriptor.hf_repo
        assert kwargs["filename"] == descriptor.weights_file
    assert local == fake_weights


def test_download_sae_unknown_raises() -> None:
    from mech_interp.sae.registry import download_sae

    with pytest.raises(ValueError, match="Unknown SAE"):
        download_sae("nonexistent-sae-xyz", dest_dir=Path("/tmp"))


def test_load_pretrained_sae_shapes_match_descriptor(tmp_path: Path) -> None:
    import torch
    from safetensors.torch import save_file

    from mech_interp.sae.registry import SAE_REGISTRY, load_pretrained_sae

    name = "neelnanda-gpt2-small-res-jb-l8"
    descriptor = SAE_REGISTRY[name]
    n_features = int(descriptor.config["n_features"])
    input_dim = int(descriptor.config["input_dim"])

    state: dict[str, Any] = {
        "W_enc": torch.randn(input_dim, n_features),  # sae_lens layout (input_dim, n_features)
        "b_enc": torch.zeros(n_features),
        "W_dec": torch.randn(n_features, input_dim),
        "b_dec": torch.zeros(input_dim),
    }
    weights_path = tmp_path / name / "fake.safetensors"
    weights_path.parent.mkdir(parents=True)
    save_file(state, str(weights_path))

    with patch(
        "mech_interp.sae.registry.download_sae", return_value=weights_path
    ):
        sae, config = load_pretrained_sae(name, device="cpu", cache_dir=tmp_path)

    assert sae.n_features == n_features
    assert sae.input_dim == input_dim
    assert config["source"] == "pretrained_sae"
    assert config["name"] == name

    x = torch.randn(2, input_dim)
    recon, codes = sae(x)
    assert recon.shape == (2, input_dim)
    assert codes.shape == (2, n_features)


def test_load_pretrained_sae_handles_topksae_key_layout(tmp_path: Path) -> None:
    """Some local SAEs are saved in TopKSAE format (encoder.weight / decoder.weight)."""
    import torch
    from safetensors.torch import save_file

    from mech_interp.sae.registry import SAE_REGISTRY, load_pretrained_sae

    name = "neelnanda-gpt2-small-res-jb-l8"
    descriptor = SAE_REGISTRY[name]
    n_features = int(descriptor.config["n_features"])
    input_dim = int(descriptor.config["input_dim"])

    state: dict[str, Any] = {
        "encoder.weight": torch.randn(n_features, input_dim),
        "encoder.bias": torch.zeros(n_features),
        "decoder.weight": torch.randn(input_dim, n_features),
        "decoder.bias": torch.zeros(input_dim),
    }
    weights_path = tmp_path / name / "fake.safetensors"
    weights_path.parent.mkdir(parents=True)
    save_file(state, str(weights_path))

    with patch("mech_interp.sae.registry.download_sae", return_value=weights_path):
        sae, _ = load_pretrained_sae(name, device="cpu", cache_dir=tmp_path)
    assert sae.n_features == n_features
