"""Feature-splitting analysis across SAE sizes.

For each *live* feature in a parent SAE, find the top-K features in a child
(larger) SAE by decoder cosine similarity, then pair their top-activating
prompts side by side.  This lets us test the "clean splitting" claim from
Anthropic's Towards Monosemanticity paper: when you double the dictionary, do
features split into more-specific specialisations, or does the dictionary
reshuffle?

Primary entry-point::

    records = compute_feature_splits(
        parent_sae_path, child_sae_path,
        parent_analysis_path, child_analysis_path,
    )
    # records is list[SplitRecord], one per live parent feature.

The scalar ``mean_split_fidelity`` (mean of the best-child cosine across all
live parent features) is the headline metric:

    * > 0.8 → clean splitting (Anthropic's claim holds)
    * 0.5–0.8 → partial specialisation
    * < 0.5 → dictionary reshuffle
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ChildRecord:
    feature: int
    cosine: float
    top_prompts: list[str]


@dataclass
class SplitRecord:
    """Splitting relationship between one parent feature and its best children."""

    parent_feature: int
    parent_top_prompts: list[str]
    children: list[ChildRecord] = field(default_factory=list)

    @property
    def best_cosine(self) -> float:
        """Cosine of the top-1 child, or 0.0 if no child clears min_cosine."""
        return self.children[0].cosine if self.children else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "parent_feature": self.parent_feature,
            "parent_top_prompts": self.parent_top_prompts,
            "best_cosine": self.best_cosine,
            "children": [
                {
                    "feature": c.feature,
                    "cosine": c.cosine,
                    "top_prompts": c.top_prompts,
                }
                for c in self.children
            ],
        }


@dataclass
class FeatureSplitAnalysis:
    """Aggregate results for one (parent, child) SAE pair."""

    parent_n_features: int
    child_n_features: int
    parent_live_count: int
    split_records: list[SplitRecord]

    @property
    def mean_split_fidelity(self) -> float:
        """Mean best-child cosine across all live parent features."""
        if not self.split_records:
            return 0.0
        return sum(r.best_cosine for r in self.split_records) / len(self.split_records)

    @property
    def split_distribution(self) -> dict[int, int]:
        """Map from child-count bucket (0,1,2,3+) to number of parent features."""
        dist: dict[int, int] = {0: 0, 1: 0, 2: 0, 3: 0}
        for r in self.split_records:
            bucket = min(len(r.children), 3)
            dist[bucket] = dist.get(bucket, 0) + 1
        return dist

    def as_dict(self) -> dict[str, Any]:
        return {
            "parent_n_features": self.parent_n_features,
            "child_n_features": self.child_n_features,
            "parent_live_count": self.parent_live_count,
            "mean_split_fidelity": self.mean_split_fidelity,
            "split_distribution": {str(k): v for k, v in self.split_distribution.items()},
            "split_records": [r.as_dict() for r in self.split_records],
        }


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------


def compute_feature_splits(
    parent_sae_path: Path,
    child_sae_path: Path,
    parent_analysis_path: Path,
    child_analysis_path: Path,
    *,
    top_k_children: int = 3,
    min_cosine: float = 0.3,
) -> list[SplitRecord]:
    """For each live feature in the parent SAE, find the top-K children in the
    child SAE by decoder cosine similarity above *min_cosine*.

    Parameters
    ----------
    parent_sae_path:
        Path to the parent SAE weights file (.safetensors or .pt).
    child_sae_path:
        Path to the child SAE weights file.
    parent_analysis_path:
        Path to the parent ``feature_analysis.json`` produced by
        :func:`mech_interp.sae.compute_feature_analysis`.
    child_analysis_path:
        Path to the child ``feature_analysis.json``.
    top_k_children:
        How many child features to report per parent (default 3).
    min_cosine:
        Minimum decoder cosine to include a child (default 0.3).

    Returns
    -------
    list[SplitRecord]
        One record per *live* parent feature, sorted by parent feature index.
    """
    import numpy as np

    parent_decoder = _load_decoder(parent_sae_path)  # (n_parent, d_model)
    child_decoder = _load_decoder(child_sae_path)  # (n_child, d_model)

    parent_analysis = _load_feature_analysis(parent_analysis_path)
    child_analysis = _load_feature_analysis(child_analysis_path)

    # L2-normalise decoders for cosine similarity
    parent_norms = np.linalg.norm(parent_decoder, axis=1, keepdims=True)
    child_norms = np.linalg.norm(child_decoder, axis=1, keepdims=True)
    parent_norms = np.where(parent_norms == 0, 1.0, parent_norms)
    child_norms = np.where(child_norms == 0, 1.0, child_norms)
    parent_normed = parent_decoder / parent_norms
    child_normed = child_decoder / child_norms

    # cosine_matrix[i, j] = cos(parent_i, child_j)
    cosine_matrix = parent_normed @ child_normed.T  # (n_parent, n_child)

    # Build top-prompt lookup for child features: feature_idx → list[str]
    child_prompts: dict[int, list[str]] = {}
    for feat in child_analysis.get("features", []):
        if not feat.get("dead", True):
            child_prompts[feat["feature_index"]] = feat.get("top_prompts", [])

    records: list[SplitRecord] = []
    for feat in parent_analysis.get("features", []):
        if feat.get("dead", True):
            continue
        p_idx = feat["feature_index"]
        p_top_prompts = feat.get("top_prompts", [])

        row = cosine_matrix[p_idx]  # (n_child,)
        # Sort child features descending by cosine
        sorted_child_idxs = np.argsort(row)[::-1]

        children: list[ChildRecord] = []
        for c_idx in sorted_child_idxs:
            cos = float(row[c_idx])
            if cos < min_cosine:
                break
            if len(children) >= top_k_children:
                break
            children.append(
                ChildRecord(
                    feature=int(c_idx),
                    cosine=round(cos, 4),
                    top_prompts=child_prompts.get(int(c_idx), []),
                )
            )

        records.append(
            SplitRecord(
                parent_feature=p_idx,
                parent_top_prompts=p_top_prompts,
                children=children,
            )
        )

    return records


def compute_feature_split_analysis(
    parent_sae_path: Path,
    child_sae_path: Path,
    parent_analysis_path: Path,
    child_analysis_path: Path,
    *,
    top_k_children: int = 3,
    min_cosine: float = 0.3,
) -> FeatureSplitAnalysis:
    """Wrapper that returns a :class:`FeatureSplitAnalysis` with aggregate stats."""

    records = compute_feature_splits(
        parent_sae_path,
        child_sae_path,
        parent_analysis_path,
        child_analysis_path,
        top_k_children=top_k_children,
        min_cosine=min_cosine,
    )

    parent_analysis = _load_feature_analysis(parent_analysis_path)
    child_analysis = _load_feature_analysis(child_analysis_path)

    return FeatureSplitAnalysis(
        parent_n_features=parent_analysis.get("n_features", 0),
        child_n_features=child_analysis.get("n_features", 0),
        parent_live_count=len(records),
        split_records=records,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_decoder(path: Path) -> Any:
    """Load the decoder weight matrix from a safetensors or torch checkpoint.

    Returns a numpy array of shape ``(n_features, d_model)``.
    """
    import numpy as np

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"SAE weights not found: {path}")

    if path.suffix == ".safetensors":
        try:
            from safetensors import safe_open

            with safe_open(str(path), framework="pt", device="cpu") as f:  # type: ignore[no-untyped-call]
                # Key is "decoder.weight" with shape (d_model, n_features) — transpose
                keys = list(f.keys())
                decoder_key = next((k for k in keys if "decoder" in k and "weight" in k), None)
                if decoder_key is None:
                    raise KeyError(f"No decoder.weight key in {path}. Keys: {keys}")
                tensor = f.get_tensor(decoder_key)
                arr = tensor.numpy()
        except ImportError:
            # Fall through to torch.load path
            import torch

            state = torch.load(path, map_location="cpu", weights_only=True)
            arr = state["decoder.weight"].numpy()
    else:
        import torch

        state = torch.load(path, map_location="cpu", weights_only=True)
        arr = state["decoder.weight"].numpy()

    # arr shape is (d_model, n_features) from nn.Linear → transpose to (n_features, d_model)
    if arr.ndim == 2:
        arr = arr.T
    return arr.astype(np.float32)


def _load_feature_analysis(path: Path) -> dict[str, Any]:
    """Load feature_analysis.json produced by compute_feature_analysis."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Feature analysis not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)  # type: ignore[no-any-return]
