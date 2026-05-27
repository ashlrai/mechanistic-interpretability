"""Translation layer: TransformerLens hook site names → HuggingFace module paths.

Each architecture family maps TL-style site patterns (with ``{L}`` as a layer
placeholder) to the corresponding dotted HF module path.  The second element of
each value tuple is ``"output"`` or ``"input"`` and tells the hook registrar
whether to capture the module's forward *output* (most TL sites) or *input*.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Architecture maps
# ---------------------------------------------------------------------------
# Format:
#   tl_pattern (with {L}) -> (hf_module_pattern (with {L}), "output" | "input")
#
# When a TL site doesn't map to a meaningful HF module boundary, we omit it
# rather than silently capturing something wrong.

_GPT2_MAP: dict[str, tuple[str, str]] = {
    # Residual stream — captured at the transformer block level.
    # GPT-2 blocks don't expose a dedicated pre/post residual hook, so we hook
    # the block itself and capture its output as the post-residual state.
    "blocks.{L}.hook_resid_pre": ("transformer.h.{L}", "input"),
    "blocks.{L}.hook_resid_post": ("transformer.h.{L}", "output"),
    # Attention
    "blocks.{L}.attn.hook_z": ("transformer.h.{L}.attn", "output"),
    "blocks.{L}.attn.hook_q": ("transformer.h.{L}.attn", "output"),
    "blocks.{L}.attn.hook_k": ("transformer.h.{L}.attn", "output"),
    "blocks.{L}.attn.hook_v": ("transformer.h.{L}.attn", "output"),
    "blocks.{L}.attn.hook_attn_out": ("transformer.h.{L}.attn.c_proj", "output"),
    # MLP
    "blocks.{L}.hook_mlp_out": ("transformer.h.{L}.mlp", "output"),
    "blocks.{L}.mlp.hook_pre": ("transformer.h.{L}.mlp.c_fc", "output"),
    "blocks.{L}.mlp.hook_post": ("transformer.h.{L}.mlp.act", "output"),
    # Layer norms
    "blocks.{L}.ln1.hook_normalized": ("transformer.h.{L}.ln_1", "output"),
    "blocks.{L}.ln2.hook_normalized": ("transformer.h.{L}.ln_2", "output"),
}

_LLAMA_MAP: dict[str, tuple[str, str]] = {
    "blocks.{L}.hook_resid_pre": ("model.layers.{L}", "input"),
    "blocks.{L}.hook_resid_post": ("model.layers.{L}", "output"),
    # Attention
    "blocks.{L}.attn.hook_z": ("model.layers.{L}.self_attn", "output"),
    "blocks.{L}.attn.hook_q": ("model.layers.{L}.self_attn.q_proj", "output"),
    "blocks.{L}.attn.hook_k": ("model.layers.{L}.self_attn.k_proj", "output"),
    "blocks.{L}.attn.hook_v": ("model.layers.{L}.self_attn.v_proj", "output"),
    "blocks.{L}.attn.hook_attn_out": ("model.layers.{L}.self_attn.o_proj", "output"),
    # MLP
    "blocks.{L}.hook_mlp_out": ("model.layers.{L}.mlp", "output"),
    "blocks.{L}.mlp.hook_pre": ("model.layers.{L}.mlp.gate_proj", "output"),
    "blocks.{L}.mlp.hook_post": ("model.layers.{L}.mlp.act_fn", "output"),
    # Layer norms (RMSNorm in Llama)
    "blocks.{L}.ln1.hook_normalized": ("model.layers.{L}.input_layernorm", "output"),
    "blocks.{L}.ln2.hook_normalized": ("model.layers.{L}.post_attention_layernorm", "output"),
}

# Qwen2 is Llama-style; module paths differ slightly.
_QWEN2_MAP: dict[str, tuple[str, str]] = {
    "blocks.{L}.hook_resid_pre": ("model.layers.{L}", "input"),
    "blocks.{L}.hook_resid_post": ("model.layers.{L}", "output"),
    "blocks.{L}.attn.hook_z": ("model.layers.{L}.self_attn", "output"),
    "blocks.{L}.attn.hook_q": ("model.layers.{L}.self_attn.q_proj", "output"),
    "blocks.{L}.attn.hook_k": ("model.layers.{L}.self_attn.k_proj", "output"),
    "blocks.{L}.attn.hook_v": ("model.layers.{L}.self_attn.v_proj", "output"),
    "blocks.{L}.attn.hook_attn_out": ("model.layers.{L}.self_attn.o_proj", "output"),
    "blocks.{L}.hook_mlp_out": ("model.layers.{L}.mlp", "output"),
    "blocks.{L}.mlp.hook_pre": ("model.layers.{L}.mlp.gate_proj", "output"),
    "blocks.{L}.mlp.hook_post": ("model.layers.{L}.mlp.act_fn", "output"),
    "blocks.{L}.ln1.hook_normalized": ("model.layers.{L}.input_layernorm", "output"),
    "blocks.{L}.ln2.hook_normalized": ("model.layers.{L}.post_attention_layernorm", "output"),
}

# Phi-3 / Phi-2 style.
_PHI_MAP: dict[str, tuple[str, str]] = {
    "blocks.{L}.hook_resid_pre": ("model.layers.{L}", "input"),
    "blocks.{L}.hook_resid_post": ("model.layers.{L}", "output"),
    "blocks.{L}.attn.hook_z": ("model.layers.{L}.self_attn", "output"),
    "blocks.{L}.attn.hook_q": ("model.layers.{L}.self_attn.q_proj", "output"),
    "blocks.{L}.attn.hook_k": ("model.layers.{L}.self_attn.k_proj", "output"),
    "blocks.{L}.attn.hook_v": ("model.layers.{L}.self_attn.v_proj", "output"),
    "blocks.{L}.attn.hook_attn_out": ("model.layers.{L}.self_attn.dense", "output"),
    "blocks.{L}.hook_mlp_out": ("model.layers.{L}.mlp", "output"),
    "blocks.{L}.mlp.hook_pre": ("model.layers.{L}.mlp.fc1", "output"),
    "blocks.{L}.mlp.hook_post": ("model.layers.{L}.mlp.fc2", "input"),
    "blocks.{L}.ln1.hook_normalized": ("model.layers.{L}.input_layernorm", "output"),
    "blocks.{L}.ln2.hook_normalized": ("model.layers.{L}.post_attention_layernorm", "output"),
}

# Gemma2 is very close to Llama; keep a separate entry so users can target it
# by name and for future divergence.
_GEMMA2_MAP: dict[str, tuple[str, str]] = {
    "blocks.{L}.hook_resid_pre": ("model.layers.{L}", "input"),
    "blocks.{L}.hook_resid_post": ("model.layers.{L}", "output"),
    "blocks.{L}.attn.hook_z": ("model.layers.{L}.self_attn", "output"),
    "blocks.{L}.attn.hook_q": ("model.layers.{L}.self_attn.q_proj", "output"),
    "blocks.{L}.attn.hook_k": ("model.layers.{L}.self_attn.k_proj", "output"),
    "blocks.{L}.attn.hook_v": ("model.layers.{L}.self_attn.v_proj", "output"),
    "blocks.{L}.attn.hook_attn_out": ("model.layers.{L}.self_attn.o_proj", "output"),
    "blocks.{L}.hook_mlp_out": ("model.layers.{L}.mlp", "output"),
    "blocks.{L}.mlp.hook_pre": ("model.layers.{L}.mlp.gate_proj", "output"),
    "blocks.{L}.mlp.hook_post": ("model.layers.{L}.mlp.act_fn", "output"),
    "blocks.{L}.ln1.hook_normalized": ("model.layers.{L}.input_layernorm", "output"),
    "blocks.{L}.ln2.hook_normalized": ("model.layers.{L}.post_attention_layernorm", "output"),
}

# Mistral uses the same structure as Llama.
_MISTRAL_MAP: dict[str, tuple[str, str]] = dict(_LLAMA_MAP)

ARCHITECTURE_SITE_MAPS: dict[str, dict[str, tuple[str, str]]] = {
    "gpt2": _GPT2_MAP,
    "llama": _LLAMA_MAP,
    "qwen2": _QWEN2_MAP,
    "phi": _PHI_MAP,
    "gemma2": _GEMMA2_MAP,
    "mistral": _MISTRAL_MAP,
}

# Architectures whose model_type strings map to a canonical family key.
_MODEL_TYPE_ALIASES: dict[str, str] = {
    "gpt2": "gpt2",
    "llama": "llama",
    "llama2": "llama",
    "llama3": "llama",
    "qwen2": "qwen2",
    "phi": "phi",
    "phi3": "phi",
    "gemma": "gemma2",
    "gemma2": "gemma2",
    "mistral": "mistral",
}

SUPPORTED_ARCHITECTURES = sorted(ARCHITECTURE_SITE_MAPS.keys())

# Detect "raw" HF paths: contain a dot but don't start with "blocks."
# These are passed through unchanged.
_TL_PREFIX = "blocks."


def _looks_like_hf_path(site: str) -> bool:
    return "." in site and not site.startswith(_TL_PREFIX)


def _layer_from_tl_site(site: str) -> tuple[str, int]:
    """Return (pattern_with_{L}, layer_index) by parsing the layer number."""
    parts = site.split(".")
    for i, part in enumerate(parts):
        if part.isdigit():
            layer = int(part)
            pattern = ".".join(parts[:i] + ["{L}"] + parts[i + 1 :])
            return pattern, layer
    return site, -1


def translate_hook_site(tl_site: str, architecture: str) -> tuple[str, str]:
    """Translate a TL hook-site name to an (hf_module_path, io) pair.

    Parameters
    ----------
    tl_site:
        Either a TransformerLens-style site (``blocks.10.hook_resid_post``)
        or a raw HF dotted module path (``model.layers.10.mlp``).
    architecture:
        Architecture family key from ``ARCHITECTURE_SITE_MAPS`` or a
        ``model_type`` alias (e.g., ``"llama3"`` → ``"llama"``).

    Returns
    -------
    (hf_module_path, io) where *io* is ``"output"`` or ``"input"``.

    Raises
    ------
    KeyError
        If the TL site is not in the architecture's translation map.
    ValueError
        If the architecture is not supported.
    """
    # Raw HF paths — pass through directly.
    if _looks_like_hf_path(tl_site):
        return tl_site, "output"

    # Resolve architecture alias.
    arch_key = _MODEL_TYPE_ALIASES.get(architecture.lower(), architecture.lower())
    if arch_key not in ARCHITECTURE_SITE_MAPS:
        supported = ", ".join(SUPPORTED_ARCHITECTURES)
        raise ValueError(
            f"Unsupported architecture '{architecture}'. "
            f"Supported: {supported}. "
            "Pass a raw HF dotted module path to bypass translation."
        )

    arch_map = ARCHITECTURE_SITE_MAPS[arch_key]
    pattern, layer = _layer_from_tl_site(tl_site)

    if pattern not in arch_map:
        supported_sites = sorted(arch_map.keys())
        raise KeyError(
            f"TL site '{tl_site}' (pattern '{pattern}') not in translation map for "
            f"'{arch_key}'. Supported patterns:\n"
            + "\n".join(f"  {s}" for s in supported_sites)
            + "\nOr pass a raw HF dotted module path (e.g. 'model.layers.10.mlp')."
        )

    hf_pattern, io = arch_map[pattern]
    if layer >= 0:
        hf_path = hf_pattern.replace("{L}", str(layer))
    else:
        hf_path = hf_pattern

    return hf_path, io


def resolve_architecture(model_type: str) -> str:
    """Map a HF ``config.model_type`` string to a canonical architecture key.

    Returns the canonical key, or the original string lower-cased if unknown
    (callers can still attempt translation and get a clear error).
    """
    return _MODEL_TYPE_ALIASES.get(model_type.lower(), model_type.lower())
