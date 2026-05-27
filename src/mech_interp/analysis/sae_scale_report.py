"""Scale report for a trained Top-K SAE run.

``generate_scale_report`` produces a one-shot summary suitable for printing as
JSON or including in a research note.  It reads from the SQLite store plus the
artifact directory written by ``PolysemanticitySAEExperiment``.

CLI entry-point: ``mech sae-scale-report --run-id N``
"""

from __future__ import annotations

import json
import statistics
from datetime import UTC, datetime
from typing import Any

from mech_interp.storage import SQLiteResultStore


def generate_scale_report(run_id: int, store: SQLiteResultStore) -> dict[str, Any]:
    """Return a structured summary for SAE run *run_id*.

    Keys
    ----
    run_id, spec_name, family, model, hook_site, n_features, k, epochs
        From the spec / result.
    wall_clock_seconds
        Derived from ``training_history.json`` if it has an ``elapsed_seconds``
        field; otherwise estimated from the manifest ``created_at`` vs the run
        ``created_at`` timestamps.  ``null`` when neither is available.
    initial_loss, final_loss, reconstruction_mse
        From training history / metrics.
    live_features, dead_features, dead_ratio
        Feature health from ``feature_analysis.json``.
    median_coherence_score
        Median Jaccard coherence across all live features.
    top_10_features
        List of up to 10 features sorted by ``max_activation`` descending,
        each with ``feature_index``, ``max_activation``, ``coherence_score``,
        and ``first_top_prompt`` (the rank-1 prompt string).
    memory_footprint_bytes
        Encoder + decoder weight size: ``n_features * d_model * 2 * 4`` bytes
        (float32).  ``d_model`` is inferred from ``sae_weights.safetensors.json``
        when present, else from the first activation shape in the analysis.
    """
    store.initialize()

    # --- run row ----------------------------------------------------------
    runs = [r for r in store.list_runs(limit=500, include_archived=True) if r.id == run_id]
    if not runs:
        raise ValueError(f"Run {run_id} not found in store.")
    run = runs[0]

    # --- result row -------------------------------------------------------
    result = store.get_result(run_id)
    metrics: dict[str, Any] = dict(result.metrics) if result else {}

    # --- spec -------------------------------------------------------------
    spec_data = store.get_run_spec(run_id) or {}
    params: dict[str, Any] = spec_data.get("parameters", {})

    model_name: str = params.get("model", params.get("model_name", "unknown"))
    hook_site: str = params.get("hook_site", "unknown")
    n_features: int = int(params.get("n_features", metrics.get("n_features", 0)))
    k_val: int = int(params.get("k", metrics.get("k", 0)))
    epochs: int = int(params.get("epochs", 0))

    # --- artifact dir -----------------------------------------------------
    artifact_dir = store.artifact_dir / f"run-{run_id:06d}"

    # --- training history -------------------------------------------------
    wall_clock_seconds: float | None = None
    initial_loss: float | None = None
    final_loss: float | None = None

    history_path = artifact_dir / "training_history.json"
    if history_path.exists():
        history = json.loads(history_path.read_text(encoding="utf-8"))
        initial_loss = history.get("initial_loss")
        final_loss = history.get("final_loss")
        # Some runs write elapsed_seconds explicitly; fall back to manifest diff
        wall_clock_seconds = history.get("elapsed_seconds")

    # Try manifest timestamps if not in history
    if wall_clock_seconds is None:
        manifest_path = artifact_dir / "manifest.json"
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                ts = manifest.get("finished_at") or manifest.get("created_at")
                if ts and run.created_at:
                    finished = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    started = run.created_at
                    if started.tzinfo is None:
                        started = started.replace(tzinfo=UTC)
                    wall_clock_seconds = (finished - started).total_seconds()
            except Exception:  # noqa: BLE001
                pass

    # --- feature analysis -------------------------------------------------
    live_features = int(metrics.get("live_features", 0))
    dead_features = int(metrics.get("dead_features", 0))
    dead_ratio = float(metrics.get("dead_feature_ratio", 0.0))
    reconstruction_mse = float(metrics.get("reconstruction_mse", 0.0))

    median_coherence: float | None = None
    top_10_features: list[dict[str, Any]] = []

    analysis_path = artifact_dir / "feature_analysis.json"
    if analysis_path.exists():
        analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
        features: list[dict[str, Any]] = analysis.get("features", [])

        live = [f for f in features if not f.get("dead", True)]
        if live:
            coherences = [f.get("coherence_score", 0.0) for f in live]
            median_coherence = statistics.median(coherences)

            sorted_live = sorted(live, key=lambda f: f.get("max_activation", 0.0), reverse=True)
            for feat in sorted_live[:10]:
                top_prompts = feat.get("top_prompts", [])
                first_prompt = top_prompts[0]["prompt"] if top_prompts else None
                top_10_features.append(
                    {
                        "feature_index": feat["feature_index"],
                        "max_activation": feat.get("max_activation", 0.0),
                        "coherence_score": feat.get("coherence_score", 0.0),
                        "first_top_prompt": first_prompt,
                    }
                )

    # --- memory footprint -------------------------------------------------
    # Encoder: n_features * d_model + n_features (bias)
    # Decoder: d_model * n_features + d_model (bias)
    # ≈ n_features * d_model * 2 * 4 bytes (float32, ignoring small biases)
    memory_bytes: int | None = None
    d_model: int | None = None

    sae_config_path = artifact_dir / "sae_weights.safetensors.json"
    if sae_config_path.exists():
        try:
            sae_cfg = json.loads(sae_config_path.read_text(encoding="utf-8"))
            d_model = sae_cfg.get("input_dim")
        except Exception:  # noqa: BLE001
            pass

    if d_model is not None and n_features > 0:
        memory_bytes = n_features * d_model * 2 * 4  # encoder + decoder, float32

    return {
        "run_id": run_id,
        "spec_name": run.spec_name,
        "family": run.family,
        "model": model_name,
        "hook_site": hook_site,
        "n_features": n_features,
        "k": k_val,
        "epochs": epochs,
        "wall_clock_seconds": wall_clock_seconds,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "reconstruction_mse": reconstruction_mse,
        "live_features": live_features,
        "dead_features": dead_features,
        "dead_ratio": dead_ratio,
        "median_coherence_score": median_coherence,
        "top_10_features": top_10_features,
        "memory_footprint_bytes": memory_bytes,
    }
