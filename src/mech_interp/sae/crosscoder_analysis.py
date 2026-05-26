"""Per-feature analysis pass for a trained Crosscoder.

Extends the SAE analysis with model-diffing statistics:
- ``decoder_norm_per_model``: L2 norm of each per-model decoder direction.
- ``model_score``: (norm_a - norm_b) / (norm_a + norm_b) ∈ [-1, 1].
  ≈ 0 → conserved across models; ±1 → model-specific.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import TYPE_CHECKING

from mech_interp.sae.crosscoder import Crosscoder

if TYPE_CHECKING:
    from torch import Tensor


@dataclass
class CrosscoderFeatureRecord:
    feature_index: int
    dead: bool
    activation_count: int = 0
    max_activation: float = 0.0
    mean_activation: float = 0.0
    top_prompts: list[dict[str, object]] = field(default_factory=list)
    coherence_score: float = 0.0
    # Crosscoder-specific fields
    decoder_norm_per_model: list[float] = field(default_factory=list)
    model_score: float = 0.0  # (norm_0 - norm_1) / (norm_0 + norm_1) for 2-model case


@dataclass
class CrosscoderAnalysis:
    n_features: int
    dead_count: int
    live_count: int
    mean_features_per_token: float
    conserved_count: int
    model_specific_count: int
    model_specific_threshold: float
    features: list[CrosscoderFeatureRecord]

    def as_dict(self) -> dict[str, object]:
        return {
            "n_features": self.n_features,
            "dead_count": self.dead_count,
            "live_count": self.live_count,
            "mean_features_per_token": self.mean_features_per_token,
            "conserved_count": self.conserved_count,
            "model_specific_count": self.model_specific_count,
            "model_specific_threshold": self.model_specific_threshold,
            "features": [
                {
                    "feature_index": r.feature_index,
                    "dead": r.dead,
                    "activation_count": r.activation_count,
                    "max_activation": r.max_activation,
                    "mean_activation": r.mean_activation,
                    "top_prompts": r.top_prompts,
                    "coherence_score": r.coherence_score,
                    "decoder_norm_per_model": r.decoder_norm_per_model,
                    "model_score": r.model_score,
                }
                for r in self.features
            ],
        }


def compute_crosscoder_analysis(
    crosscoder: Crosscoder,
    activations_per_model: tuple[Tensor, ...],
    prompt_for_token: list[str],
    *,
    top_prompts_per_feature: int = 5,
    model_specific_threshold: float = 0.5,
) -> CrosscoderAnalysis:
    """Analyse a trained Crosscoder and surface per-feature diffing statistics.

    Args:
        crosscoder: Trained Crosscoder.
        activations_per_model: Tuple of ``(n_tokens, d_model)`` tensors, one per
            model.
        prompt_for_token: List of length ``n_tokens`` mapping each token back to
            its source prompt string.
        top_prompts_per_feature: How many top-activating prompts to record.
        model_specific_threshold: |model_score| > this → model-specific.

    Returns:
        ``CrosscoderAnalysis`` with per-feature records.
    """
    import torch

    n_tokens = activations_per_model[0].shape[0]
    if n_tokens != len(prompt_for_token):
        raise ValueError(
            "prompt_for_token must have one entry per activation row "
            f"({n_tokens}); got {len(prompt_for_token)}"
        )

    crosscoder.eval()
    with torch.no_grad():
        codes, _ = crosscoder.encode(activations_per_model)  # (n_tokens, n_features)

        # Pre-compute decoder norms per feature per model.
        # decoder weight: (input_dim, n_features) → norm over input_dim axis
        decoder_norms: list[list[float]] = []
        for dec in crosscoder.decoders:
            # weight shape: (input_dim, n_features)
            norms = dec.weight.norm(dim=0)  # (n_features,)
            decoder_norms.append(norms.tolist())

    active_mask = codes != 0
    counts = active_mask.sum(dim=0)  # (n_features,)
    mean_per_token = float(active_mask.sum(dim=1).float().mean().item())
    abs_codes = codes.abs()
    max_activations = abs_codes.max(dim=0).values  # (n_features,)

    records: list[CrosscoderFeatureRecord] = []
    dead_count = 0
    conserved_count = 0
    model_specific_count = 0

    for feature_idx in range(crosscoder.n_features):
        # Decoder norms for this feature across models
        norms_for_feature = [decoder_norms[m][feature_idx] for m in range(crosscoder.n_models)]

        # model_score: (norm_0 - norm_1) / (norm_0 + norm_1) for 2-model case.
        # For n>2, use first two models (the primary comparison pair).
        norm_a = norms_for_feature[0]
        norm_b = norms_for_feature[1]
        denom = norm_a + norm_b
        model_score = (norm_a - norm_b) / denom if denom > 1e-9 else 0.0

        active = int(counts[feature_idx].item())
        if active == 0:
            dead_count += 1
            records.append(
                CrosscoderFeatureRecord(
                    feature_index=feature_idx,
                    dead=True,
                    decoder_norm_per_model=norms_for_feature,
                    model_score=model_score,
                )
            )
            continue

        nonzero_vals = abs_codes[:, feature_idx][active_mask[:, feature_idx]]
        mean_nonzero = float(nonzero_vals.mean().item()) if nonzero_vals.numel() else 0.0
        top_n = min(top_prompts_per_feature, active)
        top_vals, top_idx = torch.topk(abs_codes[:, feature_idx], k=top_n)
        top_prompts: list[dict[str, object]] = []
        seen_prompts: set[str] = set()
        for rank, (val, idx) in enumerate(
            zip(top_vals.tolist(), top_idx.tolist(), strict=True), start=1
        ):
            prompt = prompt_for_token[idx]
            if prompt in seen_prompts:
                continue
            seen_prompts.add(prompt)
            top_prompts.append({"rank": rank, "activation": float(val), "prompt": prompt})

        # Conserved vs model-specific classification
        if abs(model_score) <= model_specific_threshold:
            conserved_count += 1
        else:
            model_specific_count += 1

        records.append(
            CrosscoderFeatureRecord(
                feature_index=feature_idx,
                dead=False,
                activation_count=active,
                max_activation=float(max_activations[feature_idx].item()),
                mean_activation=mean_nonzero,
                top_prompts=top_prompts,
                coherence_score=_token_overlap_coherence(
                    [str(entry["prompt"]) for entry in top_prompts]
                ),
                decoder_norm_per_model=norms_for_feature,
                model_score=model_score,
            )
        )

    live_count = crosscoder.n_features - dead_count
    return CrosscoderAnalysis(
        n_features=crosscoder.n_features,
        dead_count=dead_count,
        live_count=live_count,
        mean_features_per_token=mean_per_token,
        conserved_count=conserved_count,
        model_specific_count=model_specific_count,
        model_specific_threshold=model_specific_threshold,
        features=records,
    )


def _token_overlap_coherence(prompts: list[str]) -> float:
    """Average pairwise Jaccard overlap of word sets."""
    if len(prompts) < 2:
        return 1.0 if prompts else 0.0
    token_sets = [set(p.lower().split()) for p in prompts]
    pair_scores = [
        len(a & b) / len(a | b) if (a | b) else 0.0
        for a, b in combinations(token_sets, 2)
    ]
    return float(sum(pair_scores) / len(pair_scores))
