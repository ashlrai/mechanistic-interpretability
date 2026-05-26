"""Top-K Crosscoder over activations from n_models (Lindsey et al., 2024).

A crosscoder is an SAE trained jointly on two (or more) models at the SAME hook
site. Each feature has one decoder direction *per model*; a shared Top-K gate
keeps the feature index space unified. Features with similar decoder norms across
models are conserved; features with one near-zero decoder are model-specific.

Reference: Lindsey et al., "Sparse Crosscoders for Cross-Layer Features in
Superposition" (Anthropic, 2024).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor


class Crosscoder:
    """Top-K crosscoder over activations from n_models.

    Encoder: concat(act_0, act_1, ..., act_{n-1}) → n_features  (shared)
    Decoder: n_features → act_i  (one nn.Linear per model)
    Top-K mask is applied once and shared across all per-model decoders so
    the feature index space is unified.

    All models must have the same d_model (raises ValueError otherwise).
    """

    def __init__(
        self,
        n_models: int,
        input_dim: int,
        n_features: int,
        k: int,
        *,
        dtype: object | None = None,
        device: str | None = None,
    ) -> None:
        import torch
        from torch import nn

        if n_models < 2:
            raise ValueError(f"n_models must be >= 2; got {n_models}")
        if k <= 0 or k > n_features:
            raise ValueError(
                f"k must be in (0, n_features]; got k={k}, n_features={n_features}"
            )
        if input_dim <= 0 or n_features <= 0:
            raise ValueError("input_dim and n_features must be positive")

        self.n_models = n_models
        self.input_dim = input_dim
        self.n_features = n_features
        self.k = k
        resolved_dtype = dtype or torch.float32

        # Shared encoder: concatenated activations → features
        self.encoder = nn.Linear(
            n_models * input_dim, n_features, bias=True, dtype=resolved_dtype
        )
        # Per-model decoders
        self.decoders: list[nn.Linear] = [
            nn.Linear(n_features, input_dim, bias=True, dtype=resolved_dtype)
            for _ in range(n_models)
        ]

        # Tied init: each decoder's weight starts as encoder.weight[:, i*d:(i+1)*d].T
        with torch.no_grad():
            for i, dec in enumerate(self.decoders):
                start = i * input_dim
                end = start + input_dim
                dec.weight.copy_(self.encoder.weight[:, start:end].t())
                dec.bias.zero_()

        # Single ModuleList so .parameters() / .to() / .train() work uniformly
        self._module = nn.ModuleList([self.encoder, *self.decoders])

        if device is not None:
            self._module.to(device)

    # ------------------------------------------------------------------
    # nn.Module proxy helpers
    # ------------------------------------------------------------------

    def parameters(self) -> object:  # noqa: D401
        return self._module.parameters()

    def to(self, device: str) -> Crosscoder:
        self._module.to(device)
        return self

    def train(self, mode: bool = True) -> Crosscoder:
        self._module.train(mode)
        return self

    def eval(self) -> Crosscoder:
        return self.train(False)

    def state_dict(self) -> dict[str, Tensor]:
        state: dict[str, Tensor] = {}
        for param_name, tensor in self.encoder.state_dict().items():
            state[f"encoder.{param_name}"] = tensor
        for i, dec in enumerate(self.decoders):
            for param_name, tensor in dec.state_dict().items():
                state[f"decoder_{i}.{param_name}"] = tensor
        return state

    # ------------------------------------------------------------------
    # Forward API
    # ------------------------------------------------------------------

    def encode(self, activations: tuple[Tensor, ...]) -> tuple[Tensor, Tensor]:
        """Encode a tuple of per-model activations into sparse feature codes.

        Args:
            activations: Tuple of n_models tensors, each ``(batch, input_dim)``.

        Returns:
            ``(sparse_codes, pre_activations)`` — both ``(batch, n_features)``.
        """
        import torch

        if len(activations) != self.n_models:
            raise ValueError(
                f"Expected {self.n_models} activation tensors; got {len(activations)}"
            )
        concat = torch.cat(activations, dim=-1)  # (batch, n_models * input_dim)
        pre = self.encoder(concat)  # (batch, n_features)
        topk_vals, topk_idx = torch.topk(pre, k=self.k, dim=-1)
        codes = torch.zeros_like(pre)
        codes.scatter_(-1, topk_idx, topk_vals)
        return codes, pre

    def decode(self, codes: Tensor) -> tuple[Tensor, ...]:
        """Run each per-model decoder and return a tuple of reconstructions.

        Args:
            codes: Sparse feature codes ``(batch, n_features)``.

        Returns:
            Tuple of n_models tensors, each ``(batch, input_dim)``.
        """
        return tuple(dec(codes) for dec in self.decoders)

    def forward(
        self, activations: tuple[Tensor, ...]
    ) -> tuple[tuple[Tensor, ...], Tensor]:
        """Full forward pass.

        Returns:
            ``(reconstructions, sparse_codes)`` where reconstructions is a
            tuple of per-model tensors and sparse_codes is ``(batch, n_features)``.
        """
        codes, _ = self.encode(activations)
        return self.decode(codes), codes

    __call__ = forward
