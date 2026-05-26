"""Per-feature analysis pass for a trained Top-K SAE.

Identifies dead features, ranks the top-activating prompts per feature, and
computes a simple token-overlap coherence heuristic. The coherence score is
intentionally lightweight — it's a triage signal, not a publication-grade metric.
Swap in sentence-transformers later if you need semantic similarity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import TYPE_CHECKING

from mech_interp.sae.model import TopKSAE

if TYPE_CHECKING:
    from torch import Tensor


@dataclass
class FeatureRecord:
    feature_index: int
    dead: bool
    activation_count: int = 0
    max_activation: float = 0.0
    mean_activation: float = 0.0
    top_prompts: list[dict[str, object]] = field(default_factory=list)
    coherence_score: float = 0.0


@dataclass
class FeatureAnalysis:
    n_features: int
    dead_count: int
    live_count: int
    mean_features_per_token: float
    features: list[FeatureRecord]

    def as_dict(self) -> dict[str, object]:
        return {
            "n_features": self.n_features,
            "dead_count": self.dead_count,
            "live_count": self.live_count,
            "mean_features_per_token": self.mean_features_per_token,
            "features": [
                {
                    "feature_index": record.feature_index,
                    "dead": record.dead,
                    "activation_count": record.activation_count,
                    "max_activation": record.max_activation,
                    "mean_activation": record.mean_activation,
                    "top_prompts": record.top_prompts,
                    "coherence_score": record.coherence_score,
                }
                for record in self.features
            ],
        }


def compute_feature_analysis(
    sae: TopKSAE,
    activations: Tensor,
    prompt_for_token: list[str],
    *,
    top_prompts_per_feature: int = 5,
) -> FeatureAnalysis:
    """Run the SAE over ``activations`` and surface per-feature statistics.

    ``prompt_for_token`` must have length ``activations.shape[0]`` and map each
    token row back to the source prompt string so the top-prompt list is humanly
    interpretable.
    """
    import torch

    if activations.shape[0] != len(prompt_for_token):
        raise ValueError(
            "prompt_for_token must have one entry per activation row "
            f"({activations.shape[0]}); got {len(prompt_for_token)}"
        )

    sae.eval()
    with torch.no_grad():
        _, codes = sae(activations)

    active_mask = codes != 0
    counts = active_mask.sum(dim=0)
    mean_per_token = float(active_mask.sum(dim=1).float().mean().item())
    abs_codes = codes.abs()
    max_activations = abs_codes.max(dim=0).values

    records: list[FeatureRecord] = []
    dead_count = 0
    for feature_idx in range(sae.n_features):
        active = int(counts[feature_idx].item())
        if active == 0:
            dead_count += 1
            records.append(FeatureRecord(feature_index=feature_idx, dead=True))
            continue
        nonzero_vals = abs_codes[:, feature_idx][active_mask[:, feature_idx]]
        mean_nonzero = float(nonzero_vals.mean().item()) if nonzero_vals.numel() else 0.0
        top_n = min(top_prompts_per_feature, active)
        top_vals, top_idx = torch.topk(abs_codes[:, feature_idx], k=top_n)
        top_prompts: list[dict[str, object]] = []
        seen_prompts: set[str] = set()
        pairs = list(zip(top_vals.tolist(), top_idx.tolist(), strict=True))
        for rank, (val, idx) in enumerate(pairs, start=1):
            prompt = prompt_for_token[idx]
            if prompt in seen_prompts:
                continue
            seen_prompts.add(prompt)
            top_prompts.append({"rank": rank, "activation": float(val), "prompt": prompt})
        records.append(
            FeatureRecord(
                feature_index=feature_idx,
                dead=False,
                activation_count=active,
                max_activation=float(max_activations[feature_idx].item()),
                mean_activation=mean_nonzero,
                top_prompts=top_prompts,
                coherence_score=_token_overlap_coherence(
                    [str(entry["prompt"]) for entry in top_prompts]
                ),
            )
        )

    live_count = sae.n_features - dead_count
    return FeatureAnalysis(
        n_features=sae.n_features,
        dead_count=dead_count,
        live_count=live_count,
        mean_features_per_token=mean_per_token,
        features=records,
    )


def _token_overlap_coherence(prompts: list[str]) -> float:
    """Average pairwise Jaccard overlap of word sets.

    TODO: swap in sentence-transformer cosine similarity once we wire up an
    embedding dependency. For now this is a fast, dependency-free triage signal.
    """
    if len(prompts) < 2:
        return 1.0 if prompts else 0.0
    token_sets = [set(p.lower().split()) for p in prompts]
    pair_scores: list[float] = []
    for a, b in combinations(token_sets, 2):
        union = len(a | b)
        pair_scores.append(len(a & b) / union if union else 0.0)
    return float(sum(pair_scores) / len(pair_scores)) if pair_scores else 0.0
