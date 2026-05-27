"""Tuned-lens calibration helper.

Trains per-layer affine transforms to minimise the KL divergence between
the projected logits at each layer and the final-layer logits — the standard
tuned-lens training objective (Belrose et al., 2023).

Usage::

    from mech_interp.analysis.tuned_lens_calibration import train_tuned_lens
    transforms = train_tuned_lens(model, prompts, epochs=50, seed=42)
    save_tuned_lens(transforms, Path("data/tuned-lens/gpt2-small.safetensors"))

CLI::

    mech calibrate-tuned-lens \\
        --model gpt2-small \\
        --prompts data/prompts/factual.jsonl \\
        --epochs 50 \\
        --output data/tuned-lens/gpt2-small.safetensors
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any


def train_tuned_lens(
    model: Any,
    prompts: list[str],
    *,
    epochs: int = 50,
    lr: float = 1e-3,
    seed: int = 42,
    device: str | None = None,
) -> dict[int, dict[str, Any]]:
    """Train per-layer affine transforms (W, b) for the tuned lens.

    Each transform A_L is initialised to the identity matrix.  Training
    minimises the KL divergence between the tuned-lens projected distribution
    at layer L and the model's final-layer distribution (soft labels):

        loss_L = KL( softmax(ln_final(A_L @ resid_L + b_L) @ W_U)
                  || softmax(ln_final(resid_final) @ W_U) )

    Args:
        model: A loaded TransformerLens HookedTransformer (or compatible API).
        prompts: List of text prompts to train on.
        epochs: Number of full-dataset epochs.
        lr: Adam learning rate.
        seed: Random seed for reproducibility.
        device: Optional torch device string; inferred from model params if None.

    Returns:
        Mapping from layer index to {"weight": Tensor, "bias": Tensor}.
    """
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:
        raise RuntimeError(
            "tuned_lens_calibration requires torch; run `uv sync --extra interp`"
        ) from exc

    random.seed(seed)
    torch.manual_seed(seed)

    n_layers: int = model.cfg.n_layers
    d_model: int = model.cfg.d_model

    inferred_device: str = device or str(next(model.parameters()).device)

    # Per-layer trainable affine transforms (init = identity + zero bias)
    class _AffineTransform(nn.Module):
        def __init__(self, d: int) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.eye(d))
            self.bias = nn.Parameter(torch.zeros(d))

        def forward(self, x: Any) -> Any:
            return x @ self.weight.T + self.bias

    transforms = nn.ModuleList(
        [_AffineTransform(d_model).to(inferred_device) for _ in range(n_layers)]
    )

    optimizer = torch.optim.Adam(transforms.parameters(), lr=lr)
    kl_loss = nn.KLDivLoss(reduction="batchmean", log_target=False)

    hook_names = [f"blocks.{L}.hook_resid_post" for L in range(n_layers)]
    hook_set = set(hook_names)

    model.eval()

    for _epoch in range(epochs):
        epoch_loss = 0.0
        for prompt in prompts:
            with torch.no_grad():
                _, cache = model.run_with_cache(
                    prompt,
                    names_filter=lambda name: name in hook_set,
                )
                # Final layer residual stream → target distribution
                final_hook = f"blocks.{n_layers - 1}.hook_resid_post"
                resid_final: Any = cache[final_hook][0, -1, :]  # [d_model]
                final_normed: Any = model.ln_final(
                    resid_final.unsqueeze(0).unsqueeze(0)
                )[0, 0, :]
                final_logits: Any = final_normed @ model.W_U
                target_probs: Any = torch.softmax(final_logits, dim=-1)

            optimizer.zero_grad()
            loss = torch.tensor(0.0, device=inferred_device)

            for L in range(n_layers - 1):  # don't train transform for final layer
                hook_name = f"blocks.{L}.hook_resid_post"
                if hook_name not in cache:
                    continue
                resid_L: Any = cache[hook_name][0, -1, :]  # [d_model]
                transformed: Any = transforms[L](resid_L)
                normed_L: Any = model.ln_final(
                    transformed.unsqueeze(0).unsqueeze(0)
                )[0, 0, :]
                logits_L: Any = normed_L @ model.W_U
                log_probs_L: Any = torch.log_softmax(logits_L, dim=-1)
                loss = loss + kl_loss(log_probs_L.unsqueeze(0), target_probs.unsqueeze(0))

            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()
            epoch_loss += float(loss.item())

        _ = epoch_loss  # available for debugging; we don't log by default

    # Return detached tensors
    result: dict[int, dict[str, Any]] = {}
    for L in range(n_layers - 1):
        transform_L: _AffineTransform = transforms[L]  # type: ignore[assignment]
        result[L] = {
            "weight": transform_L.weight.detach().cpu(),
            "bias": transform_L.bias.detach().cpu(),
        }
    return result


def save_tuned_lens(
    transforms: dict[int, dict[str, Any]],
    output_path: Path | str,
) -> Path:
    """Save per-layer affine transforms to a safetensors file.

    Key format: ``layer_{L}.weight`` and ``layer_{L}.bias``.
    """
    from safetensors.torch import save_file

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    flat: dict[str, Any] = {}
    for L, tensors in transforms.items():
        flat[f"layer_{L}.weight"] = tensors["weight"]
        flat[f"layer_{L}.bias"] = tensors["bias"]

    save_file(flat, str(output_path))
    return output_path


def load_prompts_from_jsonl(path: Path | str) -> list[str]:
    """Read a JSONL file and return the ``prompt`` field of each record."""
    prompts: list[str] = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                prompt = record.get("prompt") or record.get("clean_prompt")
                if isinstance(prompt, str) and prompt:
                    prompts.append(prompt)
            except json.JSONDecodeError:
                continue
    return prompts
