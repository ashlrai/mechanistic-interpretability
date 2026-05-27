"""Pretrained SAE registry — download and load published SAEs from HuggingFace.

Most public SAEs (jbloom's GPT2 collection, gemma-scope, sae_lens checkpoints) are
plain L1 SAEs with an encoder + decoder. We load them into a thin wrapper that
mirrors the ``TopKSAE`` interface so the existing feature-analysis code reuses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch import Tensor


@dataclass(frozen=True)
class SAEDescriptor:
    name: str
    hf_repo: str
    weights_file: str
    config: dict[str, Any]
    license: str
    hf_revision: str | None = None
    description: str = ""


SAE_REGISTRY: dict[str, SAEDescriptor] = {
    "neelnanda-gpt2-small-res-jb-l8": SAEDescriptor(
        name="neelnanda-gpt2-small-res-jb-l8",
        hf_repo="jbloom/GPT2-Small-SAEs-Reformatted",
        weights_file="blocks.8.hook_resid_pre/sae_weights.safetensors",
        config={
            "n_features": 24576,
            "k": None,
            "input_dim": 768,
            "hook_site": "blocks.8.hook_resid_pre",
            "model_name": "gpt2-small",
        },
        license="MIT",
        description="Joseph Bloom's SAE on gpt2-small layer 8 residual stream (24K features).",
    ),
    "neelnanda-gpt2-small-res-jb-l6": SAEDescriptor(
        name="neelnanda-gpt2-small-res-jb-l6",
        hf_repo="jbloom/GPT2-Small-SAEs-Reformatted",
        weights_file="blocks.6.hook_resid_pre/sae_weights.safetensors",
        config={
            "n_features": 24576,
            "k": None,
            "input_dim": 768,
            "hook_site": "blocks.6.hook_resid_pre",
            "model_name": "gpt2-small",
        },
        license="MIT",
        description="Joseph Bloom's SAE on gpt2-small layer 6 residual stream (24K features).",
    ),
}


class LoadedSAE:
    """Compatibility shim for plain L1 SAEs loaded from HuggingFace.

    Mirrors ``TopKSAE``'s public surface (``encode``, ``decode``, ``forward``)
    without enforcing top-k masking — most published SAEs use L1 regularization
    instead. ``compute_feature_analysis`` works on both.
    """

    def __init__(
        self,
        encoder_weight: Tensor,
        encoder_bias: Tensor,
        decoder_weight: Tensor,
        decoder_bias: Tensor,
        *,
        input_dim: int,
        n_features: int,
        k: int | None = None,
    ) -> None:
        import torch
        from torch import nn

        self.input_dim = input_dim
        self.n_features = n_features
        self.k = k

        self.encoder = nn.Linear(input_dim, n_features, bias=True)
        self.decoder = nn.Linear(n_features, input_dim, bias=True)
        with torch.no_grad():
            self.encoder.weight.copy_(encoder_weight)
            self.encoder.bias.copy_(encoder_bias)
            self.decoder.weight.copy_(decoder_weight)
            self.decoder.bias.copy_(decoder_bias)
        self._module = nn.ModuleList([self.encoder, self.decoder])

    def to(self, device: str) -> LoadedSAE:
        self._module.to(device)
        return self

    def eval(self) -> LoadedSAE:
        self._module.eval()
        return self

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        import torch

        pre = self.encoder(x)
        codes = torch.relu(pre)
        if self.k is not None and self.k < self.n_features:
            topk_vals, topk_idx = torch.topk(codes, k=self.k, dim=-1)
            mask = torch.zeros_like(codes)
            mask.scatter_(-1, topk_idx, topk_vals)
            codes = mask
        return codes, pre

    def decode(self, codes: Tensor) -> Tensor:
        result: Tensor = self.decoder(codes)
        return result

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        codes, _ = self.encode(x)
        return self.decode(codes), codes

    __call__ = forward


def download_sae(name: str, *, dest_dir: Path) -> Path:
    """Download an SAE's weights file via huggingface_hub. Cache under ``dest_dir/<name>/``."""
    if name not in SAE_REGISTRY:
        supported = ", ".join(sorted(SAE_REGISTRY))
        raise ValueError(f"Unknown SAE '{name}'. Available: {supported}.")
    descriptor = SAE_REGISTRY[name]
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required to download SAEs. "
            "Run `uv sync --extra interp` to install."
        ) from exc

    cache_dir = dest_dir / name
    cache_dir.mkdir(parents=True, exist_ok=True)
    local_path = hf_hub_download(
        repo_id=descriptor.hf_repo,
        filename=descriptor.weights_file,
        revision=descriptor.hf_revision,
        local_dir=str(cache_dir),
    )
    return Path(local_path)


def load_pretrained_sae(
    name: str,
    *,
    device: str = "cpu",
    cache_dir: Path | None = None,
) -> tuple[LoadedSAE, dict[str, Any]]:
    """Download (if needed) and load weights into a ``LoadedSAE``.

    Returns ``(sae, config_dict)`` where config_dict contains the registry
    config plus a ``source`` key indicating provenance.
    """
    if name not in SAE_REGISTRY:
        supported = ", ".join(sorted(SAE_REGISTRY))
        raise ValueError(f"Unknown SAE '{name}'. Available: {supported}.")
    descriptor = SAE_REGISTRY[name]

    resolved_cache = cache_dir or Path("data/saes/cache")
    weights_path = download_sae(name, dest_dir=resolved_cache)

    try:
        from safetensors.torch import load_file
    except ImportError as exc:
        raise RuntimeError(
            "safetensors is required to load SAEs. Run `uv sync --extra interp`."
        ) from exc

    state = load_file(str(weights_path))

    encoder_weight, encoder_bias, decoder_weight, decoder_bias = _extract_sae_tensors(state)

    config = dict(descriptor.config)
    input_dim = int(config["input_dim"])
    n_features = int(config["n_features"])
    if encoder_weight.shape != (n_features, input_dim):
        raise ValueError(
            f"SAE '{name}' encoder weight shape {tuple(encoder_weight.shape)} "
            f"does not match expected ({n_features}, {input_dim})"
        )

    sae = LoadedSAE(
        encoder_weight=encoder_weight,
        encoder_bias=encoder_bias,
        decoder_weight=decoder_weight,
        decoder_bias=decoder_bias,
        input_dim=input_dim,
        n_features=n_features,
        k=config.get("k"),
    ).to(device).eval()

    out_config = {
        **config,
        "name": name,
        "hf_repo": descriptor.hf_repo,
        "weights_path": str(weights_path),
        "license": descriptor.license,
        "source": "pretrained_sae",
    }
    return sae, out_config


def _extract_sae_tensors(
    state: dict[str, Tensor],
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Locate encoder/decoder weights+biases in a state dict.

    Different SAE checkpoints use different key names (``W_enc``/``b_enc`` for
    sae_lens vs ``encoder.weight``/``encoder.bias`` for TopKSAE). Probe both.
    """
    candidates = (
        ("W_enc", "b_enc", "W_dec", "b_dec"),
        ("encoder.weight", "encoder.bias", "decoder.weight", "decoder.bias"),
    )
    for enc_w_key, enc_b_key, dec_w_key, dec_b_key in candidates:
        if all(k in state for k in (enc_w_key, enc_b_key, dec_w_key, dec_b_key)):
            enc_w = state[enc_w_key]
            # sae_lens stores W_enc as (input_dim, n_features);
            # TopKSAE uses (n_features, input_dim).  Transpose so callers see one layout.
            if enc_w_key == "W_enc":
                enc_w = enc_w.T
            dec_w = state[dec_w_key]
            if dec_w_key == "W_dec":
                dec_w = dec_w.T
            return enc_w, state[enc_b_key], dec_w, state[dec_b_key]
    raise ValueError(
        f"Could not locate encoder/decoder weights in state dict. "
        f"Available keys: {sorted(state.keys())[:10]}..."
    )


@dataclass
class AnalysisRecord:
    name: str
    config: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)
