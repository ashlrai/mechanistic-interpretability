"""Adam reconstruction-MSE training loop for the Crosscoder.

Like ``train_top_k_sae`` but takes a tuple of activation tensors (one per model).
Reconstruction loss = sum of per-model MSE losses.

Persistence via ``save_crosscoder_weights`` writes one safetensors file with
prefixed parameter names (``encoder.*``, ``decoder_0.*``, ``decoder_1.*``, ...).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from mech_interp.sae.crosscoder import Crosscoder
from mech_interp.sae.trainer import TrainingHistory

if TYPE_CHECKING:
    from torch import Tensor


def train_crosscoder(
    activations_per_model: tuple[Tensor, ...],
    *,
    n_features: int,
    k: int,
    learning_rate: float = 1e-3,
    epochs: int = 10,
    batch_size: int = 512,
    device: str = "cpu",
    seed: int = 42,
) -> tuple[Crosscoder, TrainingHistory]:
    """Train a Crosscoder on a tuple of flattened activation tensors.

    Args:
        activations_per_model: Tuple of ``(n_tokens, d_model)`` tensors, one per
            model.  All tensors must have the same shape.
        n_features: Number of crosscoder features.
        k: Number of active features per token (Top-K).
        learning_rate: Adam learning rate.
        epochs: Number of full passes over the data.
        batch_size: Mini-batch size.
        device: Target device (``"cpu"``, ``"cuda"``, ``"mps"``).
        seed: RNG seed for reproducibility.

    Returns:
        ``(crosscoder, history)``
    """
    import torch

    if len(activations_per_model) < 2:
        raise ValueError("train_crosscoder requires at least 2 activation tensors")

    # Validate shapes
    ref_shape = tuple(activations_per_model[0].shape)
    if len(ref_shape) != 2:
        raise ValueError(
            f"Each activation tensor must be (n_tokens, d_model); got shape {ref_shape}"
        )
    n_tokens, d_model = ref_shape
    for i, act in enumerate(activations_per_model[1:], start=1):
        if tuple(act.shape) != ref_shape:
            raise ValueError(
                f"All activation tensors must have the same shape {ref_shape}; "
                f"model {i} has shape {tuple(act.shape)}"
            )

    n_models = len(activations_per_model)

    # MPS has known instability in fp16/bfloat16 — pin to float32.
    if device == "mps":
        activations_per_model = tuple(a.float() for a in activations_per_model)

    torch.manual_seed(seed)
    crosscoder = Crosscoder(
        n_models=n_models,
        input_dim=d_model,
        n_features=n_features,
        k=k,
        dtype=activations_per_model[0].dtype,
    )
    crosscoder.to(device)
    optimizer = torch.optim.Adam(crosscoder.parameters(), lr=learning_rate)  # type: ignore[arg-type]

    acts_detached = tuple(a.detach() for a in activations_per_model)
    history = TrainingHistory(
        epochs=epochs, batch_size=batch_size, learning_rate=learning_rate
    )

    crosscoder.train()
    with torch.no_grad():
        first_slice = tuple(
            a[: min(batch_size, n_tokens)].to(device) for a in acts_detached
        )
        recons, _ = crosscoder(first_slice)
        init_losses = [
            float(torch.mean((first_slice[i] - recons[i]) ** 2).item())
            for i in range(n_models)
        ]
        history.initial_loss = float(sum(init_losses))

    for _epoch in range(epochs):
        permutation = torch.randperm(n_tokens)
        epoch_loss = 0.0
        batches = 0
        for start in range(0, n_tokens, batch_size):
            indices = permutation[start : start + batch_size]
            batch = tuple(a[indices].to(device) for a in acts_detached)
            recons, _ = crosscoder(batch)
            loss = torch.mean((batch[0] - recons[0]) ** 2)
            for _mi in range(1, n_models):
                loss = loss + torch.mean((batch[_mi] - recons[_mi]) ** 2)
            optimizer.zero_grad()
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()
            epoch_loss += float(loss.item())
            batches += 1
        history.losses_per_epoch.append(epoch_loss / max(batches, 1))

    crosscoder.eval()
    with torch.no_grad():
        final_slice = tuple(
            a[: min(batch_size, n_tokens)].to(device) for a in acts_detached
        )
        recons, _ = crosscoder(final_slice)
        final_losses = [
            float(torch.mean((final_slice[i] - recons[i]) ** 2).item())
            for i in range(n_models)
        ]
        history.final_loss = float(sum(final_losses))

    return crosscoder, history


def save_crosscoder_weights(
    crosscoder: Crosscoder,
    path: Path,
    history: TrainingHistory | None = None,
) -> Path:
    """Persist the Crosscoder to a single safetensors file.

    Parameter names are prefixed: ``encoder.*``, ``decoder_0.*``,
    ``decoder_1.*``, etc.  A sibling ``<path>.json`` config is also written.

    Falls back to ``torch.save`` when safetensors is not installed (tests only).
    """
    import torch

    path = Path(path)
    state = {
        name: tensor.detach().cpu().contiguous()
        for name, tensor in crosscoder.state_dict().items()
    }
    try:
        from safetensors.torch import save_file

        save_file(state, str(path))
    except ImportError:
        torch.save(state, path)

    config_path = path.with_suffix(path.suffix + ".json")
    config_path.write_text(
        json.dumps(
            {
                "n_models": crosscoder.n_models,
                "input_dim": crosscoder.input_dim,
                "n_features": crosscoder.n_features,
                "k": crosscoder.k,
                "dtype": str(next(iter(state.values())).dtype),
                "training": history.as_dict() if history is not None else None,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return path
