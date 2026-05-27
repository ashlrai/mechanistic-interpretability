"""Unit tests for hf_site_translator — translation table coverage."""

from __future__ import annotations

import pytest

from mech_interp.backends.hf_site_translator import (
    ARCHITECTURE_SITE_MAPS,
    SUPPORTED_ARCHITECTURES,
    _looks_like_hf_path,
    resolve_architecture,
    translate_hook_site,
)

# ---------------------------------------------------------------------------
# Basic coverage
# ---------------------------------------------------------------------------


class TestGPT2Map:
    def test_resid_pre_translates(self) -> None:
        path, io = translate_hook_site("blocks.0.hook_resid_pre", "gpt2")
        assert path == "transformer.h.0"
        assert io == "input"

    def test_resid_post_translates(self) -> None:
        path, io = translate_hook_site("blocks.5.hook_resid_post", "gpt2")
        assert path == "transformer.h.5"
        assert io == "output"

    def test_mlp_out_translates(self) -> None:
        path, io = translate_hook_site("blocks.3.hook_mlp_out", "gpt2")
        assert path == "transformer.h.3.mlp"
        assert io == "output"

    def test_attn_out_translates(self) -> None:
        path, io = translate_hook_site("blocks.11.attn.hook_attn_out", "gpt2")
        assert path == "transformer.h.11.attn.c_proj"
        assert io == "output"

    def test_ln1_translates(self) -> None:
        path, io = translate_hook_site("blocks.0.ln1.hook_normalized", "gpt2")
        assert path == "transformer.h.0.ln_1"
        assert io == "output"

    def test_ln2_translates(self) -> None:
        path, io = translate_hook_site("blocks.0.ln2.hook_normalized", "gpt2")
        assert path == "transformer.h.0.ln_2"
        assert io == "output"

    def test_mlp_pre_translates(self) -> None:
        path, io = translate_hook_site("blocks.2.mlp.hook_pre", "gpt2")
        assert path == "transformer.h.2.mlp.c_fc"
        assert io == "output"

    def test_mlp_post_translates(self) -> None:
        path, io = translate_hook_site("blocks.2.mlp.hook_post", "gpt2")
        assert path == "transformer.h.2.mlp.act"
        assert io == "output"


class TestLlamaMap:
    def test_resid_pre_translates(self) -> None:
        path, io = translate_hook_site("blocks.10.hook_resid_pre", "llama")
        assert path == "model.layers.10"
        assert io == "input"

    def test_resid_post_translates(self) -> None:
        path, io = translate_hook_site("blocks.0.hook_resid_post", "llama")
        assert path == "model.layers.0"
        assert io == "output"

    def test_mlp_out_translates(self) -> None:
        path, io = translate_hook_site("blocks.7.hook_mlp_out", "llama")
        assert path == "model.layers.7.mlp"
        assert io == "output"

    def test_attn_q_translates(self) -> None:
        path, io = translate_hook_site("blocks.4.attn.hook_q", "llama")
        assert path == "model.layers.4.self_attn.q_proj"
        assert io == "output"

    def test_ln1_rms_translates(self) -> None:
        path, io = translate_hook_site("blocks.1.ln1.hook_normalized", "llama")
        assert path == "model.layers.1.input_layernorm"
        assert io == "output"

    def test_ln2_rms_translates(self) -> None:
        path, io = translate_hook_site("blocks.1.ln2.hook_normalized", "llama")
        assert path == "model.layers.1.post_attention_layernorm"
        assert io == "output"


class TestQwen2Map:
    def test_resid_post_translates(self) -> None:
        path, io = translate_hook_site("blocks.0.hook_resid_post", "qwen2")
        assert path == "model.layers.0"
        assert io == "output"

    def test_mlp_out_translates(self) -> None:
        path, io = translate_hook_site("blocks.15.hook_mlp_out", "qwen2")
        assert path == "model.layers.15.mlp"
        assert io == "output"

    def test_attn_v_translates(self) -> None:
        path, io = translate_hook_site("blocks.3.attn.hook_v", "qwen2")
        assert path == "model.layers.3.self_attn.v_proj"
        assert io == "output"


# ---------------------------------------------------------------------------
# Architecture aliases
# ---------------------------------------------------------------------------


def test_llama3_alias_resolves_to_llama() -> None:
    assert resolve_architecture("llama3") == "llama"


def test_phi3_alias_resolves_to_phi() -> None:
    assert resolve_architecture("phi3") == "phi"


def test_gemma_alias_resolves_to_gemma2() -> None:
    assert resolve_architecture("gemma") == "gemma2"


def test_unknown_alias_returns_lowercased() -> None:
    assert resolve_architecture("MyCustomArch") == "mycustomarch"


def test_llama_alias_via_translate() -> None:
    path, io = translate_hook_site("blocks.0.hook_resid_post", "llama3")
    assert path == "model.layers.0"
    assert io == "output"


# ---------------------------------------------------------------------------
# Raw HF path passthrough
# ---------------------------------------------------------------------------


def test_raw_hf_path_passthrough() -> None:
    path, io = translate_hook_site("model.layers.10.mlp", "llama")
    assert path == "model.layers.10.mlp"
    assert io == "output"


def test_raw_hf_path_without_dot_raises() -> None:
    """A site name with no dot and starting with 'blocks.' should try translation."""
    # This is not a raw HF path — it starts with 'blocks.' — so it goes through
    # the normal flow and raises KeyError for an unknown pattern.
    with pytest.raises(KeyError):
        translate_hook_site("blocks.0.hook_nonexistent", "gpt2")


def test_looks_like_hf_path_true() -> None:
    assert _looks_like_hf_path("model.layers.0.mlp") is True


def test_looks_like_hf_path_false_for_tl_site() -> None:
    assert _looks_like_hf_path("blocks.0.hook_resid_post") is False


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_unsupported_architecture_raises_value_error() -> None:
    with pytest.raises(ValueError, match="Unsupported architecture"):
        translate_hook_site("blocks.0.hook_resid_post", "totally_unknown_arch_xyz")


def test_unsupported_site_raises_key_error() -> None:
    with pytest.raises(KeyError, match="not in translation map"):
        translate_hook_site("blocks.0.hook_nonexistent_site", "gpt2")


def test_error_message_mentions_supported_architectures() -> None:
    with pytest.raises(ValueError) as exc_info:
        translate_hook_site("blocks.0.hook_resid_post", "unknown_arch_xyz")
    msg = str(exc_info.value)
    for arch in SUPPORTED_ARCHITECTURES:
        assert arch in msg


# ---------------------------------------------------------------------------
# Completeness: every architecture has the core sites
# ---------------------------------------------------------------------------

CORE_PATTERNS = [
    "blocks.{L}.hook_resid_pre",
    "blocks.{L}.hook_resid_post",
    "blocks.{L}.hook_mlp_out",
    "blocks.{L}.attn.hook_z",
    "blocks.{L}.ln1.hook_normalized",
    "blocks.{L}.ln2.hook_normalized",
]


@pytest.mark.parametrize("arch", list(ARCHITECTURE_SITE_MAPS.keys()))
@pytest.mark.parametrize("pattern", CORE_PATTERNS)
def test_core_patterns_present_for_all_architectures(arch: str, pattern: str) -> None:
    assert pattern in ARCHITECTURE_SITE_MAPS[arch], (
        f"Pattern '{pattern}' missing from architecture '{arch}'"
    )


@pytest.mark.parametrize("arch", list(ARCHITECTURE_SITE_MAPS.keys()))
def test_all_hf_paths_contain_layer_placeholder_or_no_l(arch: str) -> None:
    """Every pattern with {{L}} in the key must have {{L}} in the HF path."""
    for tl_pattern, (hf_pattern, _io) in ARCHITECTURE_SITE_MAPS[arch].items():
        if "{L}" in tl_pattern:
            assert "{L}" in hf_pattern, (
                f"arch={arch}: TL pattern '{tl_pattern}' has {{L}} but HF path "
                f"'{hf_pattern}' does not"
            )
