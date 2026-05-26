"""Sparse Probing experiment.

Trains an L1-regularised logistic-regression probe on residual-stream (or any
hook-site) activations to distinguish two prompt classes.  The L1 penalty drives
most weights to zero, leaving a sparse direction that corresponds to a learned
concept feature.

Training: coordinate-descent on the L1-penalised logistic loss (pure numpy).
No sklearn/scipy dependency.

Reference: Gurnee et al. (2023) "Finding Neurons in a Haystack" §3.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator

from mech_interp.experiments.base import Experiment
from mech_interp.experiments.families import ExperimentFamily
from mech_interp.storage.artifacts import resolve_run_artifact_dir
from mech_interp.types import (
    ExperimentResult,
    ExperimentRun,
    ExperimentSpec,
    RunStatus,
)


class SparseProbingExperiment(Experiment):
    family = ExperimentFamily.SPARSE_PROBING

    def __init__(self, backend: Any = None) -> None:
        self._backend = backend

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self, spec: ExperimentSpec, run: ExperimentRun) -> ExperimentResult:
        params = _SparseProbingParams.model_validate(spec.parameters)

        rng = np.random.default_rng(params.seed)

        backend = self._backend or _build_backend(spec, params)

        # Collect activations
        pos_prompts = list(params.positive_prompts)
        neg_prompts = list(params.negative_prompts)
        all_prompts = pos_prompts + neg_prompts
        labels = np.array(
            [1.0] * len(pos_prompts) + [0.0] * len(neg_prompts), dtype=np.float32
        )

        acts = _capture_activations(backend, all_prompts, params.hook_site)
        # acts: [n_prompts, d_model]

        n = len(all_prompts)
        indices = rng.permutation(n)
        n_train = max(1, int(n * params.train_fraction))
        train_idx = indices[:n_train]
        eval_idx = indices[n_train:]
        if len(eval_idx) == 0:
            eval_idx = train_idx  # degenerate fallback for tiny datasets

        X_train = acts[train_idx]
        y_train = labels[train_idx]
        X_eval = acts[eval_idx]
        y_eval = labels[eval_idx]

        # Train sparse probe
        weights, bias = _train_sparse_probe(
            X_train, y_train,
            l1_alpha=params.l1_alpha,
            epochs=params.epochs,
        )

        nonzero_count = int(np.sum(np.abs(weights) > 1e-8))
        train_acc = _accuracy(X_train, y_train, weights, bias)
        eval_acc = _accuracy(X_eval, y_eval, weights, bias)

        # Artifacts
        artifact_dir = resolve_run_artifact_dir(run)
        artifact_dir.mkdir(parents=True, exist_ok=True)

        weights_path = (artifact_dir / "probe_weights.safetensors").resolve()
        summary_path = (artifact_dir / "probe_summary.json").resolve()

        _write_safetensors(weights_path, weights)

        summary: dict[str, Any] = {
            "l1_alpha": params.l1_alpha,
            "nonzero_count": nonzero_count,
            "total_count": int(weights.shape[0]),
            "train_accuracy": float(train_acc),
            "eval_accuracy": float(eval_acc),
            "positive_count": len(pos_prompts),
            "negative_count": len(neg_prompts),
            "hook_site": params.hook_site,
            "model": params.model,
            "epochs": params.epochs,
        }
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        metrics: dict[str, float] = {
            "train_accuracy": float(train_acc),
            "eval_accuracy": float(eval_acc),
            "nonzero_count": float(nonzero_count),
            "total_count": float(weights.shape[0]),
            "sparsity": 1.0 - float(nonzero_count) / float(max(1, weights.shape[0])),
        }

        notes = (
            f"Sparse probe at {params.hook_site}: "
            f"eval_acc={eval_acc:.3f}, "
            f"nonzero={nonzero_count}/{weights.shape[0]} dims."
        )

        return ExperimentResult(
            run_id=run.id,
            status=RunStatus.SUCCEEDED,
            metrics=metrics,
            artifacts={
                "probe_weights": str(weights_path),
                "probe_summary": str(summary_path),
            },
            notes=notes,
        )


# ------------------------------------------------------------------
# Pydantic schema
# ------------------------------------------------------------------

class _SparseProbingParams(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str = "gpt2-small"
    hook_site: str = "blocks.6.hook_resid_pre"
    positive_prompts: list[str]
    negative_prompts: list[str]
    l1_alpha: float = Field(default=0.01, gt=0.0)
    epochs: int = Field(default=100, ge=1)
    train_fraction: float = Field(default=0.8, gt=0.0, lt=1.0)
    seed: int = 42
    device: str = "cpu"

    @field_validator("positive_prompts", "negative_prompts", mode="before")
    @classmethod
    def _require_nonempty(cls, v: Any) -> Any:
        if not isinstance(v, list) or len(v) == 0:
            raise ValueError("positive_prompts and negative_prompts must be non-empty lists")
        return v


# ------------------------------------------------------------------
# Activation capture
# ------------------------------------------------------------------

def _capture_activations(backend: Any, prompts: list[str], hook_site: str) -> np.ndarray:
    """Return [n_prompts, d_model] float32 array of last-token activations."""
    model = getattr(backend, "model", None)
    if model is None:
        backend.load()
        model = backend.model
    assert model is not None

    _, cache = model.run_with_cache(
        prompts,
        names_filter=lambda name: name == hook_site,
    )
    if hook_site not in cache:
        raise ValueError(f"hook_site '{hook_site}' not found in model cache")

    tensor = cache[hook_site]  # [batch, seq, d_model]
    # Take last-token position
    detach = getattr(tensor, "detach", None)
    if callable(detach):
        tensor = detach()
    cpu = getattr(tensor, "cpu", None)
    if callable(cpu):
        tensor = cpu()
    arr = np.asarray(tensor, dtype=np.float32)  # [batch, seq, d_model]
    if arr.ndim == 3:
        arr = arr[:, -1, :]  # last token → [batch, d_model]
    elif arr.ndim == 2:
        pass  # already [batch, d_model] if seq was squeezed
    return arr


# ------------------------------------------------------------------
# Sparse probe solver (coordinate descent on L1-logistic)
# ------------------------------------------------------------------

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return np.asarray(1.0 / (1.0 + np.exp(-np.clip(x, -30, 30))))


def _logistic_loss(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float) -> float:
    logits = X @ w + b
    p = _sigmoid(logits)
    eps = 1e-9
    return float(-np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)))


def _train_sparse_probe(
    X: np.ndarray,
    y: np.ndarray,
    *,
    l1_alpha: float,
    epochs: int,
    lr: float = 0.1,
) -> tuple[np.ndarray, float]:
    """Coordinate-descent L1-regularised logistic regression.

    Uses a proximal gradient step: gradient descent on the smooth logistic part,
    then soft-thresholding for the L1 part.  Fast enough for d_model ≤ 4096 and
    epochs ≤ 500.
    """
    n, d = X.shape

    # Normalise features for stable convergence
    col_std = np.std(X, axis=0) + 1e-8
    X_n = X / col_std

    # Adaptive learning rate via Lipschitz constant of logistic gradient
    # L ≤ 0.25 * max_eigenvalue(X^T X / n)
    # We use a cheap upper bound: 0.25 * max_col_norm^2 / n
    L_lip = 0.25 * float(np.max(np.sum(X_n ** 2, axis=0))) / n + 1e-8
    step = 1.0 / L_lip

    w_n = np.zeros(d, dtype=np.float32)
    b_n = 0.0

    for _ in range(epochs):
        logits = X_n @ w_n + b_n
        p = _sigmoid(logits)
        residual = p - y          # [n]
        grad_w = X_n.T @ residual / n   # [d]
        grad_b = float(np.mean(residual))

        # Gradient step
        w_n = w_n - step * grad_w
        b_n = b_n - step * grad_b

        # Proximal (soft-threshold) for L1 on weights only
        threshold = l1_alpha * step
        w_n = np.sign(w_n) * np.maximum(np.abs(w_n) - threshold, 0.0)

    # Map weights back to original (un-normalised) scale
    w_original = (w_n / col_std).astype(np.float32)
    return w_original, float(b_n)


def _accuracy(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float) -> float:
    logits = X @ w + b
    preds = (logits >= 0.0).astype(np.float32)
    return float(np.mean(preds == y))


# ------------------------------------------------------------------
# Safetensors writer (minimal, no dependency beyond struct)
# ------------------------------------------------------------------

def _write_safetensors(path: Path, weights: np.ndarray) -> None:
    """Write a single float32 weight tensor as a valid safetensors file.

    Format: 8-byte LE header_size + JSON header + tensor data.
    https://huggingface.co/docs/safetensors/index
    """
    import struct

    # Ensure contiguous float32
    w = np.ascontiguousarray(weights, dtype=np.float32)
    data_bytes = w.tobytes()
    data_len = len(data_bytes)

    metadata = {
        "probe_direction": {
            "dtype": "F32",
            "shape": list(w.shape),
            "data_offsets": [0, data_len],
        }
    }
    header_json = json.dumps(metadata, separators=(",", ":")).encode("utf-8")
    # Pad header to 8-byte boundary
    pad = (8 - len(header_json) % 8) % 8
    header_json = header_json + b" " * pad

    header_size = struct.pack("<Q", len(header_json))
    path.write_bytes(header_size + header_json + data_bytes)


# ------------------------------------------------------------------
# Backend construction
# ------------------------------------------------------------------

def _build_backend(spec: ExperimentSpec, params: _SparseProbingParams) -> Any:
    from mech_interp.backends import create_instrumented_backend

    config: dict[str, Any] = {"model_name": params.model, "device": params.device}
    config.update(spec.parameters.get("backend_config", {}) or {})
    backend = create_instrumented_backend(spec.backend, config)
    backend.load()
    return backend
