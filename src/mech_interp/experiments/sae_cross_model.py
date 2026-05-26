"""SAE cross-model feature comparison (family = ``sae_cross_model``).

Pipeline
--------
1. Train (or load) a Top-K SAE on **source_model** at ``hook_site``.
2. Train (or load) a Top-K SAE on **target_model** at the same ``hook_site``.
3. Compute cosine similarity between every pair of (source_decoder_direction,
   target_decoder_direction) — each direction is a ``d_model``-vector from the
   SAE decoder weight matrix.
4. Run greedy bipartite matching via ``scipy.optimize.linear_sum_assignment``
   (sklearn closure; already in the transitive dep set of transformer-lens).
5. Persist ``matched_features.json`` and ``match_summary.json``.

References
----------
- Gao et al. (2024) Top-K SAE
- Conmy et al. (2023) ACDC (for context on cross-model circuit analysis)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from mech_interp.backends import create_instrumented_backend
from mech_interp.experiments.base import Experiment
from mech_interp.sae import (
    TopKSAE,
    compute_feature_analysis,
    save_sae_weights,
    train_top_k_sae,
)
from mech_interp.storage.artifacts import resolve_run_artifact_dir
from mech_interp.types import (
    ExperimentResult,
    ExperimentRun,
    ExperimentSpec,
    InstrumentedModelBackend,
    RunStatus,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Spec
# ---------------------------------------------------------------------------


class SAECrossModelSpec(BaseModel):
    model_config = ConfigDict(extra="allow")

    source_model: str = "gpt2"
    source_model_name: str | None = None
    target_model: str = "gpt2-medium"
    target_model_name: str | None = None
    hook_site: str

    # SAE hyperparameters (shared between source and target)
    n_features: int = Field(default=256, ge=2)
    k: int = Field(default=32, ge=1)
    epochs: int = Field(default=5, ge=1, le=10_000)
    batch_size: int = Field(default=512, ge=1)
    learning_rate: float = Field(default=1e-3, gt=0)
    seed: int = 42
    device: str = "cpu"

    # Optional pre-trained SAE paths (skip training if provided)
    source_sae_path: str | None = None
    target_sae_path: str | None = None

    # Prompts / corpus for activation capture
    prompts: list[str] | None = None
    dataset_path: str | None = None
    seq_len: int = Field(default=128, ge=1)
    max_tokens: int = Field(default=10_000, ge=1)

    # Matching
    top_prompts_per_feature: int = Field(default=3, ge=1, le=20)
    high_similarity_threshold: float = Field(default=0.8, ge=0.0, le=1.0)

    @property
    def resolved_source_model_name(self) -> str:
        return self.source_model_name or self.source_model

    @property
    def resolved_target_model_name(self) -> str:
        return self.target_model_name or self.target_model

    @field_validator("hook_site")
    @classmethod
    def strip_hook_site(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("hook_site must not be empty")
        return value


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------


class SAECrossModelExperiment(Experiment):
    family = "sae_cross_model"

    def __init__(
        self,
        source_backend: InstrumentedModelBackend | None = None,
        target_backend: InstrumentedModelBackend | None = None,
    ) -> None:
        self.source_backend = source_backend
        self.target_backend = target_backend

    def run(self, spec: ExperimentSpec, run: ExperimentRun) -> ExperimentResult:
        import torch

        config = SAECrossModelSpec.model_validate(spec.parameters)
        if config.k > config.n_features:
            raise ValueError(
                f"k={config.k} must be <= n_features={config.n_features} for Top-K SAE."
            )

        # Backends
        src_backend = self.source_backend or create_instrumented_backend(
            spec.backend,
            {"model_name": config.resolved_source_model_name, "device": config.device},
        )
        tgt_backend = self.target_backend or create_instrumented_backend(
            spec.backend,
            {"model_name": config.resolved_target_model_name, "device": config.device},
        )

        prompts = _resolve_prompts(config)

        # Activations
        src_flat, src_prompt_map = _capture_flat_activations(
            src_backend, prompts, config.hook_site, config
        )
        tgt_flat, tgt_prompt_map = _capture_flat_activations(
            tgt_backend, prompts, config.hook_site, config
        )

        # SAEs
        torch.manual_seed(config.seed)
        src_sae = _get_or_train_sae(
            src_flat, config, path_str=config.source_sae_path, label="source"
        )
        torch.manual_seed(config.seed)
        tgt_sae = _get_or_train_sae(
            tgt_flat, config, path_str=config.target_sae_path, label="target"
        )

        # Feature analysis for top-prompts
        src_analysis = compute_feature_analysis(
            src_sae,
            src_flat.to(config.device),
            src_prompt_map,
            top_prompts_per_feature=config.top_prompts_per_feature,
        )
        tgt_analysis = compute_feature_analysis(
            tgt_sae,
            tgt_flat.to(config.device),
            tgt_prompt_map,
            top_prompts_per_feature=config.top_prompts_per_feature,
        )

        # Cosine similarity matrix between decoder directions
        src_dirs = _decoder_directions(src_sae)  # (n_src, d_model)
        tgt_dirs = _decoder_directions(tgt_sae)  # (n_tgt, d_model)

        # If d_model differs (e.g. gpt2 vs gpt2-medium), we cannot compare
        # directions directly. Raise a clear error.
        if src_dirs.shape[1] != tgt_dirs.shape[1]:
            raise ValueError(
                f"Source SAE d_model={src_dirs.shape[1]} != "
                f"target SAE d_model={tgt_dirs.shape[1]}. "
                "Direct decoder-direction comparison requires identical d_model. "
                "Use the same model family or project to a shared subspace first."
            )

        sim_matrix = _cosine_similarity_matrix(src_dirs, tgt_dirs)  # (n_src, n_tgt)

        # Greedy bipartite matching (maximise total similarity)
        matched = _greedy_bipartite_match(sim_matrix)

        # Build top-prompt lookup dicts
        src_top_prompts = _top_prompts_by_feature(src_analysis)
        tgt_top_prompts = _top_prompts_by_feature(tgt_analysis)

        matched_features = _build_matched_features(
            matched, sim_matrix, src_top_prompts, tgt_top_prompts
        )

        # Summary
        cosines = [m["cosine"] for m in matched_features]
        high_sim_count = sum(1 for c in cosines if c > config.high_similarity_threshold)
        median_cosine = float(_median(cosines)) if cosines else 0.0
        match_summary: dict[str, object] = {
            "source_model": config.resolved_source_model_name,
            "target_model": config.resolved_target_model_name,
            "hook_site": config.hook_site,
            "n_source_features": src_sae.n_features,
            "n_target_features": tgt_sae.n_features,
            "n_matched_pairs": len(matched_features),
            "high_similarity_pairs": high_sim_count,
            "high_similarity_threshold": config.high_similarity_threshold,
            "median_cosine": median_cosine,
            "source_dead_features": src_analysis.dead_count,
            "target_dead_features": tgt_analysis.dead_count,
        }

        # Persist
        artifact_dir = resolve_run_artifact_dir(run)
        artifact_dir.mkdir(parents=True, exist_ok=True)

        matched_path = artifact_dir / "matched_features.json"
        summary_path = artifact_dir / "match_summary.json"

        matched_path.write_text(
            json.dumps(matched_features, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        summary_path.write_text(
            json.dumps(match_summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        artifacts: dict[str, str] = {
            "matched_features": str(matched_path.resolve()),
            "match_summary": str(summary_path.resolve()),
        }

        # Optionally save SAE weights
        src_weights = artifact_dir / "source_sae_weights.safetensors"
        tgt_weights = artifact_dir / "target_sae_weights.safetensors"
        save_sae_weights(src_sae, src_weights)
        save_sae_weights(tgt_sae, tgt_weights)
        artifacts["source_sae_weights"] = str(src_weights.resolve())
        artifacts["target_sae_weights"] = str(tgt_weights.resolve())

        metrics: dict[str, float] = {
            "n_matched_pairs": float(len(matched_features)),
            "high_similarity_pairs": float(high_sim_count),
            "median_cosine": median_cosine,
            "source_dead_features": float(src_analysis.dead_count),
            "target_dead_features": float(tgt_analysis.dead_count),
            "source_n_tokens": float(src_flat.shape[0]),
            "target_n_tokens": float(tgt_flat.shape[0]),
        }

        return ExperimentResult(
            run_id=run.id,
            status=RunStatus.SUCCEEDED,
            metrics=metrics,
            artifacts=artifacts,
            notes=(
                f"SAE cross-model comparison: {config.resolved_source_model_name} vs "
                f"{config.resolved_target_model_name} at {config.hook_site}; "
                f"{high_sim_count}/{len(matched_features)} pairs with "
                f"cos>{config.high_similarity_threshold:.2f}; "
                f"median cosine={median_cosine:.3f}."
            ),
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_prompts(config: SAECrossModelSpec) -> list[str]:
    if config.prompts:
        cleaned = [p.strip() for p in config.prompts if p and p.strip()]
        if not cleaned:
            raise ValueError("sae_cross_model prompts must contain at least one non-empty entry")
        return cleaned
    if config.dataset_path:
        from mech_interp.datasets import load_prompt_dataset
        dataset = load_prompt_dataset(config.dataset_path)
        return [record.prompt for record in dataset.records]
    raise ValueError("sae_cross_model requires 'prompts' or 'dataset_path'.")


def _capture_flat_activations(
    backend: InstrumentedModelBackend,
    prompts: list[str],
    hook_site: str,
    config: SAECrossModelSpec,
) -> tuple[Any, list[str]]:
    import torch

    captured = backend.capture_activations(prompts, [hook_site])
    if hook_site not in captured:
        raise ValueError(
            f"Backend did not return activations for hook site '{hook_site}'."
        )
    tensor = captured[hook_site]
    shape = tuple(tensor.shape)
    if len(shape) == 2:
        flat = tensor
        prompt_map = list(prompts)
    elif len(shape) == 3:
        batch, seq, _ = shape
        flat = tensor.reshape(batch * seq, shape[2])
        prompt_map = [prompts[i] for i in range(batch) for _ in range(seq)]
    else:
        raise ValueError(f"Unexpected activation shape {shape}")

    if not isinstance(flat, torch.Tensor):
        flat = torch.as_tensor(flat)
    flat = flat.detach().to(dtype=torch.float32)
    return flat, prompt_map


def _get_or_train_sae(
    flat: Any,
    config: SAECrossModelSpec,
    *,
    path_str: str | None,
    label: str,
) -> TopKSAE:
    """Load SAE from *path_str* if given, otherwise train from *flat*."""
    import torch

    if path_str is not None:
        return _load_sae(Path(path_str))

    logger.info("Training %s SAE (%d features, k=%d)…", label, config.n_features, config.k)
    sae, _ = train_top_k_sae(
        flat,
        n_features=config.n_features,
        k=config.k,
        learning_rate=config.learning_rate,
        epochs=config.epochs,
        batch_size=config.batch_size,
        device=config.device,
        seed=config.seed,
    )
    _ = torch  # keep import
    return sae


def _load_sae(path: Path) -> TopKSAE:
    """Load a Top-K SAE from a safetensors file + sibling JSON config."""
    config_path = path.with_suffix(path.suffix + ".json")
    if not config_path.exists():
        raise FileNotFoundError(
            f"SAE config not found at {config_path}; "
            "expected a sibling .json file alongside the weights."
        )
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    sae = TopKSAE(
        input_dim=int(cfg["input_dim"]),
        n_features=int(cfg["n_features"]),
        k=int(cfg["k"]),
    )
    try:
        import torch
        from safetensors.torch import load_file

        state = load_file(str(path))
        sae.encoder.load_state_dict(
            {k.removeprefix("encoder."): v for k, v in state.items() if k.startswith("encoder.")},
            strict=True,
        )
        sae.decoder.load_state_dict(
            {k.removeprefix("decoder."): v for k, v in state.items() if k.startswith("decoder.")},
            strict=True,
        )
        _ = torch  # keep import
    except ImportError:
        import torch
        state = torch.load(str(path), map_location="cpu")
        sae.encoder.load_state_dict(
            {k.removeprefix("encoder."): v for k, v in state.items() if k.startswith("encoder.")},
            strict=True,
        )
        sae.decoder.load_state_dict(
            {k.removeprefix("decoder."): v for k, v in state.items() if k.startswith("decoder.")},
            strict=True,
        )
    sae.eval()
    return sae


def _decoder_directions(sae: TopKSAE) -> Any:
    """Return decoder weight matrix (n_features, d_model), L2-normalised."""
    import torch

    W = sae.decoder.weight.detach().cpu()  # shape (d_model, n_features)
    W = W.t()  # (n_features, d_model)
    norms = W.norm(dim=1, keepdim=True).clamp(min=1e-8)
    _ = torch  # keep import
    return W / norms


def _cosine_similarity_matrix(src: Any, tgt: Any) -> Any:
    """Compute (n_src, n_tgt) cosine similarity matrix.

    Both inputs are already L2-normalised, so cosine = dot product.
    """
    return (src @ tgt.t()).cpu()


def _greedy_bipartite_match(sim_matrix: Any) -> list[tuple[int, int]]:
    """Return a list of (src_idx, tgt_idx) pairs that maximise total similarity.

    Uses ``scipy.optimize.linear_sum_assignment`` when available (the full
    Hungarian algorithm, O(n³)). Falls back to a simple greedy rank-sort when
    scipy is not installed.

    The matrix may be non-square; we match min(n_src, n_tgt) pairs.
    """
    try:
        import numpy as np
        from scipy.optimize import linear_sum_assignment  # type: ignore[import-untyped]

        cost = -np.asarray(sim_matrix.numpy(), dtype=float)
        row_ind, col_ind = linear_sum_assignment(cost)
        return list(zip(row_ind.tolist(), col_ind.tolist(), strict=True))
    except ImportError:
        pass

    # Greedy fallback: sort all (i,j) pairs by similarity descending; pick
    # each pair if neither index has been matched yet.
    import torch

    n_src, n_tgt = sim_matrix.shape
    pairs_with_scores = []
    for i in range(n_src):
        for j in range(n_tgt):
            pairs_with_scores.append((float(sim_matrix[i, j].item()), i, j))
    pairs_with_scores.sort(reverse=True)
    used_src: set[int] = set()
    used_tgt: set[int] = set()
    result: list[tuple[int, int]] = []
    for _, i, j in pairs_with_scores:
        if i not in used_src and j not in used_tgt:
            result.append((i, j))
            used_src.add(i)
            used_tgt.add(j)
        if len(result) >= min(n_src, n_tgt):
            break
    _ = torch  # keep import
    return result


def _top_prompts_by_feature(analysis: Any) -> dict[int, list[str]]:
    """Return {feature_idx: [prompt, ...]} from a FeatureAnalysis."""
    result: dict[int, list[str]] = {}
    for record in analysis.features:
        prompts = [str(entry["prompt"]) for entry in record.top_prompts]
        result[record.feature_index] = prompts
    return result


def _build_matched_features(
    matched: list[tuple[int, int]],
    sim_matrix: Any,
    src_top_prompts: dict[int, list[str]],
    tgt_top_prompts: dict[int, list[str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for src_idx, tgt_idx in matched:
        cosine = float(sim_matrix[src_idx, tgt_idx].item())
        rows.append(
            {
                "source_feature": src_idx,
                "target_feature": tgt_idx,
                "cosine": round(cosine, 6),
                "source_top_prompts": src_top_prompts.get(src_idx, [])[:3],
                "target_top_prompts": tgt_top_prompts.get(tgt_idx, [])[:3],
            }
        )
    rows.sort(key=lambda r: r["cosine"], reverse=True)
    return rows


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    mid = n // 2
    if n % 2 == 0:
        return (sorted_vals[mid - 1] + sorted_vals[mid]) / 2.0
    return sorted_vals[mid]
