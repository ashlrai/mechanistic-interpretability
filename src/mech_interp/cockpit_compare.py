"""Helpers for A/B run comparison (cockpit + CLI).

Kept separate from cockpit.py so the CLI can import without pulling FastAPI.
"""
from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Metric-diff heuristics
# Lower is better: mse, loss, dead_fraction, normalized_mse
# Higher is better: recovery_fraction, faithfulness, variance_explained,
#                   coherence_score, live_fraction, extraction_quality
# ---------------------------------------------------------------------------
_LOWER_IS_BETTER: frozenset[str] = frozenset(
    {
        "mse",
        "loss",
        "train_loss",
        "val_loss",
        "dead_fraction",
        "normalized_mse",
        "mean_logit_diff_error",
        "perplexity",
    }
)
_HIGHER_IS_BETTER: frozenset[str] = frozenset(
    {
        "recovery_fraction",
        "faithfulness",
        "variance_explained",
        "coherence_score",
        "live_fraction",
        "extraction_quality",
        "mean_cosine_similarity",
        "accuracy",
        "f1",
    }
)


def metric_direction(key: str) -> str:
    """Return 'lower' | 'higher' | 'unknown' for a metric name."""
    k = key.lower()
    for stem in _LOWER_IS_BETTER:
        if stem in k:
            return "lower"
    for stem in _HIGHER_IS_BETTER:
        if stem in k:
            return "higher"
    return "unknown"


def _pct_diff(a: float, b: float) -> float | None:
    """Relative difference (b - a) / |a|; None when denominator is zero."""
    if a == 0.0:
        return None
    return (b - a) / abs(a)


def build_metric_rows(
    metrics_a: dict[str, float],
    metrics_b: dict[str, float],
) -> list[dict[str, Any]]:
    """Return one row per metric present in either run.

    Each row: {key, val_a, val_b, pct_diff, direction, highlight, badge_class}
    highlight is True when |pct_diff| > 5 %.
    badge_class is 'better' | 'worse' | 'neutral' (only set when highlight=True).
    """
    all_keys = sorted(set(metrics_a) | set(metrics_b))
    rows: list[dict[str, Any]] = []
    for key in all_keys:
        val_a = metrics_a.get(key)
        val_b = metrics_b.get(key)
        pct: float | None = None
        if val_a is not None and val_b is not None:
            pct = _pct_diff(float(val_a), float(val_b))

        highlight = pct is not None and abs(pct) > 0.05
        badge: str = "neutral"
        if highlight:
            direction = metric_direction(key)
            if direction == "lower":
                badge = "better" if (pct or 0) < 0 else "worse"
            elif direction == "higher":
                badge = "better" if (pct or 0) > 0 else "worse"
            else:
                badge = "neutral"

        rows.append(
            {
                "key": key,
                "val_a": val_a,
                "val_b": val_b,
                "pct_diff": pct,
                "highlight": highlight,
                "badge_class": badge,
            }
        )
    return rows


def build_param_diff_rows(
    params_a: dict[str, Any],
    params_b: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return rows for parameters that differ between the two specs."""
    all_keys = sorted(set(params_a) | set(params_b))
    rows: list[dict[str, Any]] = []
    for key in all_keys:
        va = params_a.get(key, _MISSING)
        vb = params_b.get(key, _MISSING)
        if va == vb:
            continue
        rows.append(
            {
                "key": key,
                "val_a": None if va is _MISSING else va,
                "val_b": None if vb is _MISSING else vb,
                "only_a": vb is _MISSING,
                "only_b": va is _MISSING,
            }
        )
    return rows


def build_env_diff_rows(
    env_a: dict[str, Any] | None,
    env_b: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Diff environment.json dicts on the canonical provenance keys + any extras."""
    _PRIORITY_KEYS = (
        "seed",
        "torch_version",
        "uv_lock_sha256",
        "model_name",
        "python_version",
        "platform",
        "transformer_lens_version",
        "numpy_version",
    )
    ea = env_a or {}
    eb = env_b or {}
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []

    def _add(key: str) -> None:
        if key in seen:
            return
        seen.add(key)
        va = ea.get(key, _MISSING)
        vb = eb.get(key, _MISSING)
        differs = va != vb
        rows.append(
            {
                "key": key,
                "val_a": None if va is _MISSING else va,
                "val_b": None if vb is _MISSING else vb,
                "differs": differs,
            }
        )

    for k in _PRIORITY_KEYS:
        if k in ea or k in eb:
            _add(k)
    for k in sorted(set(ea) | set(eb)):
        _add(k)
    return rows


class _MissingSentinel:
    """Sentinel distinct from None for missing dict keys."""

    def __repr__(self) -> str:
        return "<missing>"


_MISSING = _MissingSentinel()
