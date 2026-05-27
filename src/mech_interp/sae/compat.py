"""sae_lens compatibility shim.

Bidirectional bridge between our SAE types and the sae_lens ecosystem
(https://github.com/jbloomAus/SAELens).  The module imports safely with no
sae_lens installed — all sae_lens symbols are guarded by lazy imports.

Weight layout conventions
-------------------------
Our layout (TopKSAE / LoadedSAE):
  encoder.weight : (n_features, input_dim)   — nn.Linear convention
  decoder.weight : (input_dim, n_features)   — nn.Linear convention

sae_lens layout (SAE object attributes):
  W_enc : (input_dim, n_features)
  W_dec : (n_features, input_dim)
  b_enc : (n_features,)
  b_dec : (input_dim,)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mech_interp.sae.registry import LoadedSAE


class OptionalDependencyError(ImportError):
    """Raised when sae_lens is not installed."""

    def __init__(self) -> None:
        super().__init__(
            "sae_lens is not installed. "
            "Install it with: pip install 'mech-interpretability[sae-lens]' "
            "or: pip install sae-lens>=3.0.0"
        )


def _require_sae_lens() -> Any:
    """Import and return the sae_lens module, or raise OptionalDependencyError."""
    try:
        import sae_lens

        return sae_lens
    except ImportError as exc:
        raise OptionalDependencyError() from exc


def to_sae_lens(
    sae: Any,
    *,
    hook_name: str,
    model_name: str,
) -> tuple[Any, dict[str, Any]]:
    """Convert one of our SAEs into a ``sae_lens.SAE`` instance.

    Parameters
    ----------
    sae:
        A ``TopKSAE`` or ``LoadedSAE`` instance.
    hook_name:
        TransformerLens hook name, e.g. ``"blocks.8.hook_resid_pre"``.
    model_name:
        TransformerLens model name, e.g. ``"gpt2"``.

    Returns
    -------
    (sae_lens_sae, config_dict)
        ``sae_lens_sae`` is a fully-constructed ``sae_lens.SAE``;
        ``config_dict`` is the ``SAEConfig``-compatible dict used to build it.

    Raises
    ------
    OptionalDependencyError
        If sae_lens is not installed.
    """
    import torch

    sl = _require_sae_lens()

    input_dim: int = sae.input_dim
    n_features: int = sae.n_features

    # Build the SAEConfig dict.  sae_lens >= 3 requires these fields.
    cfg_dict: dict[str, Any] = {
        "architecture": "standard",
        "d_in": input_dim,
        "d_sae": n_features,
        "dtype": "float32",
        "model_name": model_name,
        "hook_name": hook_name,
        "hook_layer": _layer_from_hook(hook_name),
        "hook_head_index": None,
        "activation_fn_name": "relu",
        "apply_b_dec_to_input": False,
        "finetuning_scaling_factor": False,
        "context_size": 128,
        "dataset_path": "",
        "normalize_activations": "none",
        "prepend_bos": True,
        "model_from_pretrained_kwargs": {},
    }

    cfg = sl.SAEConfig.from_dict(cfg_dict)
    sae_lens_sae = sl.SAE(cfg)

    # Copy weights: our encoder.weight is (n_features, input_dim); sae_lens W_enc is transposed.
    with torch.no_grad():
        sae_lens_sae.W_enc.copy_(sae.encoder.weight.t())  # → (input_dim, n_features)
        sae_lens_sae.b_enc.copy_(sae.encoder.bias)
        sae_lens_sae.W_dec.copy_(sae.decoder.weight.t())  # → (n_features, input_dim)
        sae_lens_sae.b_dec.copy_(sae.decoder.bias)

    return sae_lens_sae, cfg_dict


def from_sae_lens(sae_lens_sae: Any) -> tuple[LoadedSAE, dict[str, Any]]:
    """Convert a ``sae_lens.SAE`` into our ``LoadedSAE``.

    Parameters
    ----------
    sae_lens_sae:
        A ``sae_lens.SAE`` instance.

    Returns
    -------
    (loaded_sae, config_dict)
        ``loaded_sae`` is a ``LoadedSAE`` with weights copied from the sae_lens SAE;
        ``config_dict`` carries provenance metadata.

    Raises
    ------
    OptionalDependencyError
        If sae_lens is not installed (fail-fast on import).
    """
    import torch

    _require_sae_lens()  # ensure sae_lens is present before we touch its attributes

    from mech_interp.sae.registry import LoadedSAE

    cfg = sae_lens_sae.cfg

    input_dim: int = int(cfg.d_in)
    n_features: int = int(cfg.d_sae)

    # sae_lens W_enc is (input_dim, n_features); transpose to our (n_features, input_dim)
    encoder_weight = sae_lens_sae.W_enc.detach().t().contiguous()
    encoder_bias = sae_lens_sae.b_enc.detach().clone()
    # sae_lens W_dec is (n_features, input_dim); transpose to our (input_dim, n_features)
    decoder_weight = sae_lens_sae.W_dec.detach().t().contiguous()
    decoder_bias = sae_lens_sae.b_dec.detach().clone()

    loaded = LoadedSAE(
        encoder_weight=encoder_weight,
        encoder_bias=encoder_bias,
        decoder_weight=decoder_weight,
        decoder_bias=decoder_bias,
        input_dim=input_dim,
        n_features=n_features,
    )

    config_dict: dict[str, Any] = {
        "source": "sae_lens",
        "model_name": getattr(cfg, "model_name", None),
        "hook_name": getattr(cfg, "hook_name", None),
        "hook_layer": getattr(cfg, "hook_layer", None),
        "input_dim": input_dim,
        "n_features": n_features,
        "architecture": getattr(cfg, "architecture", "standard"),
    }

    # Detect if this is actually a TopK-style SAE so downstream callers know
    activation_fn = getattr(cfg, "activation_fn_name", "relu")
    if activation_fn in ("topk",):
        k_val = getattr(cfg, "activation_fn_kwargs", {})
        if isinstance(k_val, dict):
            config_dict["k"] = k_val.get("k")

    _ = torch  # silence unused-import checker; torch used indirectly via tensor ops
    return loaded, config_dict


def load_from_sae_lens_release(
    release: str,
    sae_id: str,
    *,
    device: str = "cpu",
) -> tuple[LoadedSAE, dict[str, Any]]:
    """Use sae_lens's own ``SAE.from_pretrained`` to load any registered SAE.

    Parameters
    ----------
    release:
        sae_lens release string, e.g. ``"gpt2-small-res-jb"``.
    sae_id:
        SAE id within that release, e.g. ``"blocks.8.hook_resid_pre"``.
    device:
        Torch device string (default ``"cpu"``).

    Returns
    -------
    (loaded_sae, config_dict)

    Raises
    ------
    OptionalDependencyError
        If sae_lens is not installed.
    """
    sl = _require_sae_lens()

    sae_lens_sae, cfg_dict, _sparsity = sl.SAE.from_pretrained(
        release=release,
        sae_id=sae_id,
        device=device,
    )

    loaded, meta = from_sae_lens(sae_lens_sae)
    config_dict: dict[str, Any] = {
        **meta,
        **cfg_dict,
        "release": release,
        "sae_id": sae_id,
        "source": "sae_lens_release",
    }
    return loaded, config_dict


def discover_sae_lens_releases() -> list[dict[str, Any]]:
    """Return all sae_lens-known releases as a list of dicts.

    Returns an empty list (rather than raising) when sae_lens is not installed,
    so callers can safely call this unconditionally and check the result.
    """
    try:
        from sae_lens import pretrained_saes
    except ImportError:
        return []

    try:
        directory = pretrained_saes.get_pretrained_saes_directory()
    except Exception:  # noqa: BLE001 — sae_lens internals can raise unpredictably
        return []

    results: list[dict[str, Any]] = []
    for release_name, release_obj in directory.items():
        saes_map = getattr(release_obj, "saes_map", {})
        model = getattr(release_obj, "model", "")
        for sae_id in saes_map:
            results.append(
                {
                    "release": release_name,
                    "sae_id": sae_id,
                    "model": model,
                }
            )
    return results


def _layer_from_hook(hook_name: str) -> int:
    """Extract a layer index from a TransformerLens hook name.

    Examples: ``"blocks.8.hook_resid_pre"`` → 8, ``"hook_embed"`` → 0.
    """
    parts = hook_name.split(".")
    for part in parts:
        if part.isdigit():
            return int(part)
    return 0
