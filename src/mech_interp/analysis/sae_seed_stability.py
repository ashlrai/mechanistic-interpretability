"""SAE seed-stability analysis — bipartite matching across same-architecture SAEs.

Compares two Top-K SAEs trained with different random seeds but identical
hyperparameters by matching their decoder directions via the Hungarian algorithm
(scipy.optimize.linear_sum_assignment) or a greedy fallback.

Public API
----------
compute_sae_pair_alignment(sae_path_a, sae_path_b, *, top_k=20, threshold=0.9) -> dict
    Load two SAE safetensors, compute pairwise decoder-cosine matrix, run
    bipartite matching, return summary statistics + top-K matched pairs.

compute_stability_report(run_dirs, *, top_k=20, threshold=0.9) -> dict
    Given N run artifact directories, compute all N*(N-1)/2 pairwise alignments
    and return the full matrix plus aggregate stats.

compute_live_only_alignment(sae_path_a, sae_path_b, analysis_path_a, analysis_path_b,
                            *, top_k=20, threshold=0.9) -> dict
    Like compute_sae_pair_alignment but restricts Hungarian matching to live
    (non-dead) features identified from feature_analysis.json files. Returns the
    same dict shape plus live_features_a/b counts.

compute_live_only_stability_report(run_dirs, *, top_k=20, threshold=0.9) -> dict
    Full N-run stability report using live-only matching for every pair.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal: SAE loader (replicates _load_sae from sae_cross_model without import)
# ---------------------------------------------------------------------------


def _load_decoder_directions(weights_path: Path) -> Any:
    """Load SAE weights and return L2-normalised decoder directions (n_features, d_model)."""
    import torch

    config_path = weights_path.with_suffix(weights_path.suffix + ".json")
    if not config_path.exists():
        raise FileNotFoundError(
            f"SAE config not found at {config_path}. "
            "Expected a sibling .json alongside the safetensors weights."
        )
    cfg = json.loads(config_path.read_text(encoding="utf-8"))

    from mech_interp.sae.model import TopKSAE

    sae = TopKSAE(
        input_dim=int(cfg["input_dim"]),
        n_features=int(cfg["n_features"]),
        k=int(cfg["k"]),
    )

    try:
        from safetensors.torch import load_file

        state = load_file(str(weights_path))
    except ImportError:
        state = torch.load(str(weights_path), map_location="cpu")

    sae.encoder.load_state_dict(
        {k.removeprefix("encoder."): v for k, v in state.items() if k.startswith("encoder.")},
        strict=True,
    )
    sae.decoder.load_state_dict(
        {k.removeprefix("decoder."): v for k, v in state.items() if k.startswith("decoder.")},
        strict=True,
    )
    sae.eval()

    # decoder.weight is (d_model, n_features); transpose → (n_features, d_model)
    W: Any = sae.decoder.weight.detach().cpu().t()
    norms = W.norm(dim=1, keepdim=True).clamp(min=1e-8)
    return W / norms  # (n_features, d_model), L2-normalised


# ---------------------------------------------------------------------------
# Internal: bipartite matching (reuse scipy if available, greedy fallback)
# ---------------------------------------------------------------------------


def _bipartite_match(sim_matrix: Any) -> list[tuple[int, int]]:
    """Return (row, col) pairs that maximise total cosine similarity."""
    import numpy as np

    try:
        from scipy.optimize import linear_sum_assignment  # type: ignore[import-untyped]

        cost = -np.asarray(sim_matrix, dtype=float)
        row_ind, col_ind = linear_sum_assignment(cost)
        return list(zip(row_ind.tolist(), col_ind.tolist(), strict=True))
    except ImportError:
        pass

    # Greedy fallback
    n_rows, n_cols = sim_matrix.shape
    flat = [
        (float(sim_matrix[i, j]), i, j) for i in range(n_rows) for j in range(n_cols)
    ]
    flat.sort(reverse=True)
    used_rows: set[int] = set()
    used_cols: set[int] = set()
    pairs: list[tuple[int, int]] = []
    for _, i, j in flat:
        if i not in used_rows and j not in used_cols:
            pairs.append((i, j))
            used_rows.add(i)
            used_cols.add(j)
        if len(pairs) >= min(n_rows, n_cols):
            break
    return pairs


# ---------------------------------------------------------------------------
# Public: single pair alignment
# ---------------------------------------------------------------------------


def compute_sae_pair_alignment(
    sae_path_a: Path | str,
    sae_path_b: Path | str,
    *,
    top_k: int = 20,
    threshold: float = 0.9,
) -> dict[str, Any]:
    """Compute bipartite alignment between two SAEs trained on the same architecture.

    Parameters
    ----------
    sae_path_a, sae_path_b:
        Paths to ``sae_weights.safetensors`` (or ``.pt``) files.
        Each must have a sibling ``<name>.json`` with ``input_dim``,
        ``n_features``, and ``k`` keys.
    top_k:
        Number of top matched pairs to include in the returned list.
    threshold:
        Cosine threshold above which a match is considered "the same feature".

    Returns
    -------
    dict with keys:
        matched_count_above_threshold  int
        median_cosine                  float
        mean_cosine                    float
        top_matches                    list[dict] — up to top_k entries each with
                                       a_idx, b_idx, cosine
        all_cosines                    list[float] — cosine for every matched pair
    """
    import numpy as np

    path_a = Path(sae_path_a)
    path_b = Path(sae_path_b)

    dirs_a = _load_decoder_directions(path_a)  # (n_a, d_model)
    dirs_b = _load_decoder_directions(path_b)  # (n_b, d_model)

    if dirs_a.shape[1] != dirs_b.shape[1]:
        raise ValueError(
            f"d_model mismatch: {dirs_a.shape[1]} vs {dirs_b.shape[1]}. "
            "Both SAEs must have been trained on the same model architecture."
        )

    # Cosine similarity matrix — both matrices already L2-normalised
    import torch

    sim: Any = (dirs_a @ dirs_b.t()).numpy()  # (n_a, n_b), numpy array

    pairs = _bipartite_match(sim)

    cosines = [float(sim[i, j]) for i, j in pairs]
    above = sum(1 for c in cosines if c >= threshold)

    sorted_pairs = sorted(pairs, key=lambda p: sim[p[0], p[1]], reverse=True)
    top_matches = [
        {"a_idx": int(i), "b_idx": int(j), "cosine": round(float(sim[i, j]), 6)}
        for i, j in sorted_pairs[:top_k]
    ]

    arr = np.array(cosines, dtype=float)
    median_cosine = float(np.median(arr)) if len(arr) else 0.0
    mean_cosine = float(np.mean(arr)) if len(arr) else 0.0

    _ = torch  # imported for type checker; actual computation via numpy

    return {
        "matched_count_above_threshold": above,
        "threshold": threshold,
        "n_matched_pairs": len(pairs),
        "median_cosine": round(median_cosine, 6),
        "mean_cosine": round(mean_cosine, 6),
        "top_matches": top_matches,
        "all_cosines": [round(c, 6) for c in cosines],
    }


# ---------------------------------------------------------------------------
# Public: full N-run stability report
# ---------------------------------------------------------------------------


def compute_stability_report(
    run_dirs: Sequence[Path | str],
    *,
    top_k: int = 20,
    threshold: float = 0.9,
) -> dict[str, Any]:
    """Compute pairwise alignment for every pair of run directories.

    Each directory must contain ``sae_weights.safetensors`` (or the
    polysemanticity experiment's canonical name) alongside its config JSON.

    Returns a dict suitable for JSON serialisation:
        runs            list[str] — resolved run directory names
        pairwise        list[dict] — one entry per (i, j) pair with i < j,
                        containing run_a, run_b, and all keys from
                        compute_sae_pair_alignment
        summary         dict — aggregate stats across all pairs:
                            n_pairs, median_of_medians, mean_of_means,
                            stability_fraction (fraction of pairs where
                            matched_count_above_threshold / n_matched_pairs >= 0.5)
    """
    import numpy as np

    resolved: list[Path] = [Path(d) for d in run_dirs]

    # Candidate weight file names in order of preference
    _WEIGHT_NAMES = ["sae_weights.safetensors", "sae_weights.pt"]

    def _find_weights(run_dir: Path) -> Path:
        for name in _WEIGHT_NAMES:
            p = run_dir / name
            if p.exists():
                return p
        raise FileNotFoundError(
            f"No SAE weights found in {run_dir}. "
            f"Tried: {_WEIGHT_NAMES}"
        )

    weight_paths = [_find_weights(d) for d in resolved]

    pairwise: list[dict[str, Any]] = []
    n = len(resolved)
    for i in range(n):
        for j in range(i + 1, n):
            logger.info(
                "Aligning %s ↔ %s", resolved[i].name, resolved[j].name
            )
            result = compute_sae_pair_alignment(
                weight_paths[i], weight_paths[j], top_k=top_k, threshold=threshold
            )
            pairwise.append(
                {
                    "run_a": str(resolved[i]),
                    "run_b": str(resolved[j]),
                    "run_a_name": resolved[i].name,
                    "run_b_name": resolved[j].name,
                    **result,
                }
            )

    # Aggregate
    all_medians = [p["median_cosine"] for p in pairwise]
    all_means = [p["mean_cosine"] for p in pairwise]
    stability_fracs = [
        p["matched_count_above_threshold"] / p["n_matched_pairs"]
        if p["n_matched_pairs"] > 0
        else 0.0
        for p in pairwise
    ]

    summary: dict[str, Any] = {
        "n_runs": n,
        "n_pairs": len(pairwise),
        "threshold": threshold,
        "median_of_medians": round(float(np.median(all_medians)), 6) if all_medians else 0.0,
        "mean_of_means": round(float(np.mean(all_means)), 6) if all_means else 0.0,
        "stability_fraction_per_pair": [round(f, 4) for f in stability_fracs],
        "mean_stability_fraction": round(float(np.mean(stability_fracs)), 4)
        if stability_fracs
        else 0.0,
    }

    return {
        "runs": [str(d) for d in resolved],
        "pairwise": pairwise,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# Public: live-feature-only single pair alignment
# ---------------------------------------------------------------------------


def _load_live_feature_indices(analysis_path: Path) -> set[int]:
    """Read feature_analysis.json and return indices of non-dead features."""
    data = json.loads(analysis_path.read_text(encoding="utf-8"))
    return {
        int(f["feature_index"])
        for f in data.get("features", [])
        if not f.get("dead", True)
    }


def compute_live_only_alignment(
    sae_path_a: Path | str,
    sae_path_b: Path | str,
    analysis_path_a: Path | str,
    analysis_path_b: Path | str,
    *,
    top_k: int = 20,
    threshold: float = 0.9,
) -> dict[str, Any]:
    """Bipartite alignment restricted to live (non-dead) features.

    Reads ``feature_analysis.json`` for each SAE to identify live features,
    then runs Hungarian matching only on the live × live submatrix.

    Parameters
    ----------
    sae_path_a, sae_path_b:
        Paths to ``sae_weights.safetensors`` (or ``.pt``) files.
    analysis_path_a, analysis_path_b:
        Paths to the corresponding ``feature_analysis.json`` files.
    top_k:
        Number of top matched pairs to include.
    threshold:
        Cosine threshold for "same feature".

    Returns
    -------
    Same dict shape as ``compute_sae_pair_alignment`` plus:
        live_features_a   int — number of live features in SAE A
        live_features_b   int — number of live features in SAE B
        mode              str — "live_only"
    """
    import numpy as np

    path_a = Path(sae_path_a)
    path_b = Path(sae_path_b)

    dirs_a = _load_decoder_directions(path_a)  # (n_a, d_model)
    dirs_b = _load_decoder_directions(path_b)  # (n_b, d_model)

    if dirs_a.shape[1] != dirs_b.shape[1]:
        raise ValueError(
            f"d_model mismatch: {dirs_a.shape[1]} vs {dirs_b.shape[1]}. "
            "Both SAEs must have been trained on the same model architecture."
        )

    live_a = sorted(_load_live_feature_indices(Path(analysis_path_a)))
    live_b = sorted(_load_live_feature_indices(Path(analysis_path_b)))

    if not live_a or not live_b:
        # Degenerate case: one SAE has no live features
        return {
            "matched_count_above_threshold": 0,
            "threshold": threshold,
            "n_matched_pairs": 0,
            "median_cosine": 0.0,
            "mean_cosine": 0.0,
            "top_matches": [],
            "all_cosines": [],
            "live_features_a": len(live_a),
            "live_features_b": len(live_b),
            "mode": "live_only",
        }

    import torch

    # Submatrix: (n_live_a, n_live_b)
    sub_a = dirs_a[live_a]  # (n_live_a, d_model)
    sub_b = dirs_b[live_b]  # (n_live_b, d_model)
    sim: Any = (sub_a @ sub_b.t()).numpy()  # numpy array

    pairs = _bipartite_match(sim)

    cosines = [float(sim[i, j]) for i, j in pairs]
    above = sum(1 for c in cosines if c >= threshold)

    sorted_pairs = sorted(pairs, key=lambda p: sim[p[0], p[1]], reverse=True)
    top_matches = [
        {
            "a_idx": int(live_a[i]),
            "b_idx": int(live_b[j]),
            "cosine": round(float(sim[i, j]), 6),
        }
        for i, j in sorted_pairs[:top_k]
    ]

    arr = np.array(cosines, dtype=float)
    median_cosine = float(np.median(arr)) if len(arr) else 0.0
    mean_cosine = float(np.mean(arr)) if len(arr) else 0.0

    _ = torch  # imported for matmul; computation via numpy

    return {
        "matched_count_above_threshold": above,
        "threshold": threshold,
        "n_matched_pairs": len(pairs),
        "median_cosine": round(median_cosine, 6),
        "mean_cosine": round(mean_cosine, 6),
        "top_matches": top_matches,
        "all_cosines": [round(c, 6) for c in cosines],
        "live_features_a": len(live_a),
        "live_features_b": len(live_b),
        "mode": "live_only",
    }


# ---------------------------------------------------------------------------
# Public: full N-run live-only stability report
# ---------------------------------------------------------------------------


def compute_live_only_stability_report(
    run_dirs: Sequence[Path | str],
    *,
    top_k: int = 20,
    threshold: float = 0.9,
) -> dict[str, Any]:
    """Compute pairwise live-only alignment for every pair of run directories.

    Each directory must contain ``sae_weights.safetensors`` (or ``.pt``) and
    ``feature_analysis.json``.

    Returns the same structure as ``compute_stability_report`` with an extra
    ``mode: "live_only"`` key in the summary and per-pair live feature counts.
    """
    import numpy as np

    resolved: list[Path] = [Path(d) for d in run_dirs]

    _WEIGHT_NAMES = ["sae_weights.safetensors", "sae_weights.pt"]

    def _find_weights(run_dir: Path) -> Path:
        for name in _WEIGHT_NAMES:
            p = run_dir / name
            if p.exists():
                return p
        raise FileNotFoundError(
            f"No SAE weights found in {run_dir}. Tried: {_WEIGHT_NAMES}"
        )

    def _find_analysis(run_dir: Path) -> Path:
        p = run_dir / "feature_analysis.json"
        if not p.exists():
            raise FileNotFoundError(
                f"feature_analysis.json not found in {run_dir}. "
                "Re-run the experiment with artifact_policy.write_feature_analysis=true."
            )
        return p

    weight_paths = [_find_weights(d) for d in resolved]
    analysis_paths = [_find_analysis(d) for d in resolved]

    pairwise: list[dict[str, Any]] = []
    n = len(resolved)
    for i in range(n):
        for j in range(i + 1, n):
            logger.info(
                "Live-only aligning %s ↔ %s", resolved[i].name, resolved[j].name
            )
            result = compute_live_only_alignment(
                weight_paths[i],
                weight_paths[j],
                analysis_paths[i],
                analysis_paths[j],
                top_k=top_k,
                threshold=threshold,
            )
            pairwise.append(
                {
                    "run_a": str(resolved[i]),
                    "run_b": str(resolved[j]),
                    "run_a_name": resolved[i].name,
                    "run_b_name": resolved[j].name,
                    **result,
                }
            )

    all_medians = [p["median_cosine"] for p in pairwise]
    all_means = [p["mean_cosine"] for p in pairwise]
    stability_fracs = [
        p["matched_count_above_threshold"] / p["n_matched_pairs"]
        if p["n_matched_pairs"] > 0
        else 0.0
        for p in pairwise
    ]

    summary: dict[str, Any] = {
        "n_runs": n,
        "n_pairs": len(pairwise),
        "threshold": threshold,
        "median_of_medians": round(float(np.median(all_medians)), 6) if all_medians else 0.0,
        "mean_of_means": round(float(np.mean(all_means)), 6) if all_means else 0.0,
        "stability_fraction_per_pair": [round(f, 4) for f in stability_fracs],
        "mean_stability_fraction": round(float(np.mean(stability_fracs)), 4)
        if stability_fracs
        else 0.0,
        "mode": "live_only",
    }

    return {
        "runs": [str(d) for d in resolved],
        "pairwise": pairwise,
        "summary": summary,
    }
