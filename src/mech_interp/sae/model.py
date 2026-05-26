"""Top-K Sparse Autoencoder (Gao et al., 2024).

Picks the k largest pre-activations per token and zeroes the rest, which sidesteps
the L1 sparsity-coefficient tuning problem and produces interpretable, sparse
feature codes. Designed to be a thin nn.Module so the trainer / analysis layers
can stay readable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor


class TopKSAE:
    """Top-K SAE with a tied-init decoder.

    The encoder maps d_model → n_features, top-k selects the k largest pre-activations
    per token, and the decoder maps the resulting sparse code back to d_model. We
    initialise the decoder to the transpose of the encoder (a small but well-known
    bootstrapping trick) so training starts at a sensible reconstruction.
    """

    def __init__(
        self,
        input_dim: int,
        n_features: int,
        k: int,
        *,
        dtype: object | None = None,
        device: str | None = None,
    ) -> None:
        import torch
        from torch import nn

        if k <= 0 or k > n_features:
            raise ValueError(f"k must be in (0, n_features]; got k={k}, n_features={n_features}")
        if input_dim <= 0 or n_features <= 0:
            raise ValueError("input_dim and n_features must be positive")

        self.input_dim = input_dim
        self.n_features = n_features
        self.k = k
        resolved_dtype = dtype or torch.float32

        self.encoder = nn.Linear(input_dim, n_features, bias=True, dtype=resolved_dtype)
        self.decoder = nn.Linear(n_features, input_dim, bias=True, dtype=resolved_dtype)
        with torch.no_grad():
            # Tied init: decoder = encoder.T (standard SAE warm-start).
            self.decoder.weight.copy_(self.encoder.weight.t())
            self.decoder.bias.zero_()
        if device is not None:
            self.encoder.to(device)
            self.decoder.to(device)
        self._module = nn.ModuleList([self.encoder, self.decoder])

    def parameters(self) -> object:  # noqa: D401 -- thin proxy
        return self._module.parameters()

    def to(self, device: str) -> TopKSAE:
        self._module.to(device)
        return self

    def train(self, mode: bool = True) -> TopKSAE:
        self._module.train(mode)
        return self

    def eval(self) -> TopKSAE:
        return self.train(False)

    def state_dict(self) -> dict[str, Tensor]:
        state: dict[str, Tensor] = {}
        for name, module in (("encoder", self.encoder), ("decoder", self.decoder)):
            for param_name, tensor in module.state_dict().items():
                state[f"{name}.{param_name}"] = tensor
        return state

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Return (sparse_codes, dense_pre_activations)."""
        import torch

        pre = self.encoder(x)
        topk_vals, topk_idx = torch.topk(pre, k=self.k, dim=-1)
        codes = torch.zeros_like(pre)
        codes.scatter_(-1, topk_idx, topk_vals)
        return codes, pre

    def decode(self, codes: Tensor) -> Tensor:
        result: Tensor = self.decoder(codes)
        return result

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        codes, _ = self.encode(x)
        return self.decode(codes), codes

    __call__ = forward
