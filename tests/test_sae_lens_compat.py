"""Tests for the sae_lens compatibility shim (src/mech_interp/sae/compat.py).

All tests use mocked sae_lens so the suite passes without sae_lens installed.
"""

from __future__ import annotations

import sys
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

torch = pytest.importorskip("torch", reason="torch not installed; run with --extra interp")
from torch import nn  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers — build a minimal fake sae_lens module
# ---------------------------------------------------------------------------


def _make_fake_sae_lens() -> ModuleType:
    """Return a lightweight mock of sae_lens sufficient for compat.py."""
    mod = ModuleType("sae_lens")

    class FakeSAEConfig:
        def __init__(self, d: dict[str, Any]) -> None:
            for k, v in d.items():
                setattr(self, k, v)
            self.d_in = d["d_in"]
            self.d_sae = d["d_sae"]
            self.model_name = d.get("model_name", "")
            self.hook_name = d.get("hook_name", "")
            self.hook_layer = d.get("hook_layer", 0)
            self.architecture = d.get("architecture", "standard")
            self.activation_fn_name = d.get("activation_fn_name", "relu")

        @classmethod
        def from_dict(cls, d: dict[str, Any]) -> FakeSAEConfig:
            return cls(d)

    class FakeSAE(nn.Module):
        def __init__(self, cfg: FakeSAEConfig) -> None:
            super().__init__()
            self.cfg = cfg
            d_in = cfg.d_in
            d_sae = cfg.d_sae
            # sae_lens layout
            self.W_enc = nn.Parameter(torch.zeros(d_in, d_sae))
            self.b_enc = nn.Parameter(torch.zeros(d_sae))
            self.W_dec = nn.Parameter(torch.zeros(d_sae, d_in))
            self.b_dec = nn.Parameter(torch.zeros(d_in))

        @classmethod
        def from_pretrained(
            cls,
            release: str,
            sae_id: str,
            device: str = "cpu",
        ) -> tuple[FakeSAE, dict[str, Any], None]:
            cfg = FakeSAEConfig(
                {
                    "d_in": 64,
                    "d_sae": 128,
                    "model_name": "gpt2",
                    "hook_name": sae_id,
                    "hook_layer": 0,
                    "architecture": "standard",
                    "activation_fn_name": "relu",
                }
            )
            sae = cls(cfg)
            torch.nn.init.normal_(sae.W_enc)
            torch.nn.init.normal_(sae.W_dec)
            return sae, {"release": release, "sae_id": sae_id}, None

    mod.SAE = FakeSAE  # type: ignore[attr-defined]
    mod.SAEConfig = FakeSAEConfig  # type: ignore[attr-defined]

    # pretrained_saes submodule
    pretrained_saes = ModuleType("sae_lens.pretrained_saes")

    def get_pretrained_saes_directory() -> dict[str, Any]:
        rel = MagicMock()
        rel.model = "gpt2"
        rel.saes_map = {"blocks.0.hook_resid_pre": {}, "blocks.8.hook_resid_pre": {}}
        return {"gpt2-small-res-jb": rel}

    pretrained_saes.get_pretrained_saes_directory = get_pretrained_saes_directory  # type: ignore[attr-defined]
    mod.pretrained_saes = pretrained_saes  # type: ignore[attr-defined]
    sys.modules["sae_lens"] = mod
    sys.modules["sae_lens.pretrained_saes"] = pretrained_saes
    return mod


def _make_loaded_sae(input_dim: int = 64, n_features: int = 128) -> Any:
    """Return a LoadedSAE with random weights."""
    from mech_interp.sae.registry import LoadedSAE

    enc_w = torch.randn(n_features, input_dim)
    enc_b = torch.zeros(n_features)
    dec_w = torch.randn(input_dim, n_features)
    dec_b = torch.zeros(input_dim)
    return LoadedSAE(
        encoder_weight=enc_w,
        encoder_bias=enc_b,
        decoder_weight=dec_w,
        decoder_bias=dec_b,
        input_dim=input_dim,
        n_features=n_features,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestToSaeLens:
    def test_to_sae_lens_returns_compatible_shape(self) -> None:
        """to_sae_lens produces a sae_lens.SAE with the correct transposed weight shapes."""
        _make_fake_sae_lens()

        from mech_interp.sae.compat import to_sae_lens

        input_dim, n_features = 64, 128
        sae = _make_loaded_sae(input_dim, n_features)
        sl_sae, cfg = to_sae_lens(
            sae, hook_name="blocks.8.hook_resid_pre", model_name="gpt2"
        )

        # Weight shapes in sae_lens layout
        assert sl_sae.W_enc.shape == (input_dim, n_features)
        assert sl_sae.b_enc.shape == (n_features,)
        assert sl_sae.W_dec.shape == (n_features, input_dim)
        assert sl_sae.b_dec.shape == (input_dim,)

        assert cfg["d_in"] == input_dim
        assert cfg["d_sae"] == n_features
        assert cfg["hook_name"] == "blocks.8.hook_resid_pre"
        assert cfg["model_name"] == "gpt2"

    def test_to_sae_lens_weight_values_round_trip(self) -> None:
        """Encoder weights survive to_sae_lens (transposed correctly)."""
        _make_fake_sae_lens()
        from mech_interp.sae.compat import to_sae_lens

        sae = _make_loaded_sae(64, 128)
        orig_enc = sae.encoder.weight.detach().clone()  # (n_features, input_dim)

        sl_sae, _ = to_sae_lens(sae, hook_name="blocks.0.hook_resid_pre", model_name="gpt2")
        # sl_sae.W_enc should be orig_enc.t()
        assert torch.allclose(sl_sae.W_enc, orig_enc.t(), atol=1e-6)


class TestFromSaeLens:
    def test_from_sae_lens_round_trips(self) -> None:
        """from_sae_lens ∘ to_sae_lens is an identity on weight values."""
        _make_fake_sae_lens()
        from mech_interp.sae.compat import from_sae_lens, to_sae_lens

        orig = _make_loaded_sae(64, 128)
        sl_sae, _ = to_sae_lens(orig, hook_name="blocks.8.hook_resid_pre", model_name="gpt2")
        loaded, meta = from_sae_lens(sl_sae)

        assert loaded.input_dim == orig.input_dim
        assert loaded.n_features == orig.n_features

        # Encoder weight round-trip
        assert torch.allclose(
            loaded.encoder.weight, orig.encoder.weight, atol=1e-6
        ), "encoder.weight not preserved through round-trip"
        # Decoder weight round-trip
        assert torch.allclose(
            loaded.decoder.weight, orig.decoder.weight, atol=1e-6
        ), "decoder.weight not preserved through round-trip"

        assert meta["source"] == "sae_lens"

    def test_from_sae_lens_metadata_fields(self) -> None:
        """config_dict from from_sae_lens contains expected provenance keys."""
        _make_fake_sae_lens()
        from mech_interp.sae.compat import from_sae_lens, to_sae_lens

        sae = _make_loaded_sae(64, 128)
        sl_sae, _ = to_sae_lens(
            sae, hook_name="blocks.8.hook_resid_pre", model_name="gpt2"
        )
        _, meta = from_sae_lens(sl_sae)
        assert "input_dim" in meta
        assert "n_features" in meta
        assert meta["input_dim"] == 64
        assert meta["n_features"] == 128


class TestLoadFromSaeLensRelease:
    def test_load_from_sae_lens_release_mocked(self) -> None:
        """load_from_sae_lens_release calls SAE.from_pretrained and returns a LoadedSAE."""
        _make_fake_sae_lens()
        from mech_interp.sae.compat import load_from_sae_lens_release
        from mech_interp.sae.registry import LoadedSAE

        loaded, cfg = load_from_sae_lens_release(
            "gpt2-small-res-jb", "blocks.8.hook_resid_pre", device="cpu"
        )

        assert isinstance(loaded, LoadedSAE)
        assert loaded.input_dim == 64
        assert loaded.n_features == 128
        assert cfg["source"] == "sae_lens_release"
        assert cfg["release"] == "gpt2-small-res-jb"
        assert cfg["sae_id"] == "blocks.8.hook_resid_pre"


class TestMissingSaeLens:
    def test_compat_module_handles_missing_sae_lens(self) -> None:
        """All public functions raise OptionalDependencyError when sae_lens is absent."""
        # Temporarily hide sae_lens from the import system
        saved = sys.modules.pop("sae_lens", None)
        saved_pretrained = sys.modules.pop("sae_lens.pretrained_saes", None)
        try:
            # Force re-import of compat so lazy imports fire fresh
            if "mech_interp.sae.compat" in sys.modules:
                del sys.modules["mech_interp.sae.compat"]

            with patch.dict(sys.modules, {"sae_lens": None}):
                from mech_interp.sae.compat import (
                    OptionalDependencyError,
                    from_sae_lens,
                    load_from_sae_lens_release,
                    to_sae_lens,
                )

                sae = _make_loaded_sae()

                with pytest.raises(OptionalDependencyError):
                    to_sae_lens(sae, hook_name="blocks.0.hook_resid_pre", model_name="gpt2")

                with pytest.raises(OptionalDependencyError):
                    from_sae_lens(MagicMock())

                with pytest.raises(OptionalDependencyError):
                    load_from_sae_lens_release("any-release", "any-id")
        finally:
            # Restore whatever was there
            if saved is not None:
                sys.modules["sae_lens"] = saved
            elif "sae_lens" in sys.modules:
                del sys.modules["sae_lens"]
            if saved_pretrained is not None:
                sys.modules["sae_lens.pretrained_saes"] = saved_pretrained

    def test_discover_releases_empty_without_sae_lens(self) -> None:
        """discover_sae_lens_releases returns [] rather than raising when sae_lens absent."""
        saved = sys.modules.pop("sae_lens", None)
        saved_pretrained = sys.modules.pop("sae_lens.pretrained_saes", None)
        try:
            with patch.dict(sys.modules, {"sae_lens": None}):
                if "mech_interp.sae.compat" in sys.modules:
                    del sys.modules["mech_interp.sae.compat"]
                from mech_interp.sae.compat import discover_sae_lens_releases

                result = discover_sae_lens_releases()
                assert result == []
        finally:
            if saved is not None:
                sys.modules["sae_lens"] = saved
            elif "sae_lens" in sys.modules:
                del sys.modules["sae_lens"]
            if saved_pretrained is not None:
                sys.modules["sae_lens.pretrained_saes"] = saved_pretrained


class TestDiscoverReleases:
    def test_discover_releases_with_fake_sae_lens(self) -> None:
        """discover_sae_lens_releases returns a list of dicts with expected keys."""
        _make_fake_sae_lens()
        # Re-import to pick up the freshly installed mock
        if "mech_interp.sae.compat" in sys.modules:
            del sys.modules["mech_interp.sae.compat"]

        from mech_interp.sae.compat import discover_sae_lens_releases

        results = discover_sae_lens_releases()
        assert isinstance(results, list)
        assert len(results) > 0
        for r in results:
            assert "release" in r
            assert "sae_id" in r
            assert "model" in r
