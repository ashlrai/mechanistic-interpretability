"""Adam reconstruction-MSE training loop for the Top-K SAE.

The trainer is intentionally minimal: it expects a pre-flattened ``(n_tokens, d_model)``
activation tensor (the experiment family handles capture + flatten), and emits a
``TrainingHistory`` snapshot per epoch so callers can persist the loss trajectory.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from mech_interp.sae.model import TopKSAE

if TYPE_CHECKING:
    from torch import Tensor


@dataclass
class TrainingHistory:
    epochs: int
    batch_size: int
    learning_rate: float
    initial_loss: float = 0.0
    final_loss: float = 0.0
    losses_per_epoch: list[float] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "initial_loss": self.initial_loss,
            "final_loss": self.final_loss,
            "losses_per_epoch": self.losses_per_epoch,
        }


def train_top_k_sae(
    activations: Tensor,
    *,
    n_features: int,
    k: int,
    learning_rate: float = 1e-3,
    epochs: int = 10,
    batch_size: int = 512,
    device: str = "cpu",
    seed: int = 42,
) -> tuple[TopKSAE, TrainingHistory]:
    """Train a Top-K SAE on flattened activations and return (sae, history).

    ``activations`` should be shape ``(n_tokens, d_model)`` and live on CPU; we move
    it to ``device`` in mini-batches so we never have to copy the full activation
    matrix to GPU/MPS at once.
    """
    import torch

    if activations.ndim != 2:
        raise ValueError(
            f"train_top_k_sae expects (n_tokens, d_model); got shape {tuple(activations.shape)}"
        )

    # MPS topk and scatter ops have known instability in fp16/bfloat16. Pin to
    # float32 before touching the SAE so training is numerically stable regardless
    # of what dtype the backend returned.
    if device == "mps":
        activations = activations.float()

    torch.manual_seed(seed)
    n_tokens, d_model = activations.shape
    sae = TopKSAE(input_dim=d_model, n_features=n_features, k=k, dtype=activations.dtype)
    sae.to(device)
    # ``sae.parameters()`` returns an Iterator[Parameter] at runtime; the typed
    # proxy on TopKSAE returns ``object`` so we cast through Any here.
    optimizer = torch.optim.Adam(sae.parameters(), lr=learning_rate)  # type: ignore[arg-type]

    activations = activations.detach()
    history = TrainingHistory(
        epochs=epochs, batch_size=batch_size, learning_rate=learning_rate
    )

    sae.train()
    with torch.no_grad():
        first_batch = activations[: min(batch_size, n_tokens)].to(device)
        recon, _ = sae(first_batch)
        history.initial_loss = float(torch.mean((first_batch - recon) ** 2).item())

    for _epoch in range(epochs):
        permutation = torch.randperm(n_tokens)
        epoch_loss = 0.0
        batches = 0
        for start in range(0, n_tokens, batch_size):
            indices = permutation[start : start + batch_size]
            batch = activations[indices].to(device)
            recon, _ = sae(batch)
            loss = torch.mean((batch - recon) ** 2)
            optimizer.zero_grad()
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()
            epoch_loss += float(loss.item())
            batches += 1
        history.losses_per_epoch.append(epoch_loss / max(batches, 1))

    sae.eval()
    with torch.no_grad():
        final_batch = activations[: min(batch_size, n_tokens)].to(device)
        recon, _ = sae(final_batch)
        history.final_loss = float(torch.mean((final_batch - recon) ** 2).item())

    return sae, history


def save_sae_weights(sae: TopKSAE, path: Path, history: TrainingHistory | None = None) -> Path:
    """Persist the SAE to safetensors with a sibling ``<path>.json`` config.

    Falls back to ``torch.save`` if safetensors isn't installed; the experiment
    family asserts safetensors is available at runtime, so this fallback only
    fires in unit tests.
    """
    import torch

    path = Path(path)
    state = {name: tensor.detach().cpu().contiguous() for name, tensor in sae.state_dict().items()}
    try:
        from safetensors.torch import save_file

        save_file(state, str(path))
    except ImportError:
        torch.save(state, path)

    config_path = path.with_suffix(path.suffix + ".json")
    config_path.write_text(
        json.dumps(
            {
                "input_dim": sae.input_dim,
                "n_features": sae.n_features,
                "k": sae.k,
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
