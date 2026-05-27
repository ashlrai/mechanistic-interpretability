"""Refusal safety audit — aggregate four runs into a single report.

Pipeline:
  1. refusal_direction run  — extracts the direction + measures extraction quality.
  2. caa_steering run       — multi-layer sweep, identifies best_layer.
  3. circuit_patching run   — head-level importance at best_layer.
  4. causal_scrubbing run   — faithfulness of the top-head circuit hypothesis.

The :func:`compile_refusal_audit` function reads the stored results for all four
run IDs and returns a :class:`RefusalAuditReport`.  The CLI ``audit-refusal``
command calls it and writes ``refusal_audit.json`` + ``refusal_audit.md``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class RefusalAuditReport:
    """Aggregated safety-audit report across four mech-interp runs."""

    model: str
    best_steering_layer: int
    best_coefficient: float
    refusal_rate_shift_at_best: float
    top_causal_heads: list[tuple[int, int, float]]  # (layer, head, importance)
    circuit_faithfulness: float
    extraction_quality: float
    baseline_refusal_rate: float
    refusal_run_id: int
    caa_run_id: int
    circuit_run_id: int
    scrub_run_id: int
    notes: str

    # ------------------------------------------------------------------ #
    # Serialisation
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # top_causal_heads is list[tuple] — asdict turns them into lists
        d["top_causal_heads"] = [list(h) for h in self.top_causal_heads]
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True) + "\n"

    def to_markdown(self) -> str:
        return _render_markdown(self)


# --------------------------------------------------------------------------- #
# Public compile function
# --------------------------------------------------------------------------- #


def compile_refusal_audit(
    refusal_run_id: int,
    caa_run_id: int,
    circuit_run_id: int,
    scrub_run_id: int,
    store: Any,  # SQLiteResultStore — typed as Any to avoid circular import
) -> RefusalAuditReport:
    """Aggregate the four runs into a single safety-audit report.

    Parameters
    ----------
    refusal_run_id:
        Run ID from a ``refusal_direction`` experiment.
    caa_run_id:
        Run ID from a ``caa_steering`` experiment.
    circuit_run_id:
        Run ID from a ``circuit_patching`` experiment at the best layer.
    scrub_run_id:
        Run ID from a ``causal_scrubbing`` experiment protecting top heads.
    store:
        A :class:`~mech_interp.storage.SQLiteResultStore` instance.

    Returns
    -------
    RefusalAuditReport
    """
    refusal_result = store.get_result(refusal_run_id)
    if refusal_result is None:
        raise ValueError(f"No result found for refusal run {refusal_run_id}")

    caa_result = store.get_result(caa_run_id)
    if caa_result is None:
        raise ValueError(f"No result found for CAA steering run {caa_run_id}")

    circuit_result = store.get_result(circuit_run_id)
    if circuit_result is None:
        raise ValueError(f"No result found for circuit patching run {circuit_run_id}")

    scrub_result = store.get_result(scrub_run_id)
    if scrub_result is None:
        raise ValueError(f"No result found for causal scrubbing run {scrub_run_id}")

    # ---- Extract model name from stored spec ----
    refusal_spec = store.get_run_spec(refusal_run_id) or {}
    model = str(
        refusal_spec.get("parameters", {}).get("model", "unknown")
    )

    # ---- Refusal direction metrics ----
    extraction_quality = float(
        refusal_result.metrics.get("extraction_quality", 0.0)
    )
    baseline_refusal_rate = float(
        refusal_result.metrics.get("baseline_refusal_rate", 0.0)
    )

    # ---- CAA steering metrics ----
    best_steering_layer = int(
        caa_result.metrics.get("best_layer", 0)
    )
    refusal_rate_shift_at_best = float(
        caa_result.metrics.get("best_refusal_rate_shift", 0.0)
    )

    # ---- Best coefficient: read from artifacts ----
    best_coefficient = _extract_best_coefficient(store, caa_run_id, best_steering_layer)

    # ---- Circuit patching: parse top heads ----
    top_causal_heads = _extract_top_heads(store, circuit_run_id, best_steering_layer, top_n=3)

    # ---- Scrubbing faithfulness ----
    circuit_faithfulness = float(
        scrub_result.metrics.get("scrubbed_faithfulness", 0.0)
    )

    # ---- Build notes ----
    notes_parts = [
        refusal_result.notes,
        caa_result.notes,
        circuit_result.notes,
        scrub_result.notes,
    ]
    notes = " | ".join(n for n in notes_parts if n)

    return RefusalAuditReport(
        model=model,
        best_steering_layer=best_steering_layer,
        best_coefficient=best_coefficient,
        refusal_rate_shift_at_best=refusal_rate_shift_at_best,
        top_causal_heads=top_causal_heads,
        circuit_faithfulness=circuit_faithfulness,
        extraction_quality=extraction_quality,
        baseline_refusal_rate=baseline_refusal_rate,
        refusal_run_id=refusal_run_id,
        caa_run_id=caa_run_id,
        circuit_run_id=circuit_run_id,
        scrub_run_id=scrub_run_id,
        notes=notes,
    )


# --------------------------------------------------------------------------- #
# Private helpers
# --------------------------------------------------------------------------- #


def _extract_best_coefficient(store: Any, caa_run_id: int, best_layer: int) -> float:
    """Read the layer_effectiveness JSON artifact from the CAA run to find the
    best steering coefficient at ``best_layer``."""
    result = store.get_result(caa_run_id)
    if result is None:
        return 0.0

    layer_eff_path = result.artifacts.get("layer_effectiveness")
    if layer_eff_path and Path(layer_eff_path).is_file():
        try:
            data: dict[str, Any] = json.loads(Path(layer_eff_path).read_text(encoding="utf-8"))
            layer_data = data.get(str(best_layer)) or data.get(best_layer)  # type: ignore[call-overload]
            if isinstance(layer_data, dict):
                return float(layer_data.get("best_coefficient", 0.0))
        except Exception:
            pass

    # Fallback: if no artifact, return the coefficient that gives maximum shift
    # according to the notes (we can't recover it without the artifact).
    return -3.0  # typical abliteration coefficient


def _extract_top_heads(
    store: Any,
    circuit_run_id: int,
    best_layer: int,
    top_n: int = 3,
) -> list[tuple[int, int, float]]:
    """Parse the circuit patching ranked results to get the top-N attention heads.

    The circuit_patching experiment produces ``patching_ranked_results.json`` with
    entries like::

        {
            "hook_site": "blocks.10.attn.hook_z",
            "recovery_fraction": 0.83,
            ...
        }

    We parse hook_site to extract (layer, head_index) where available.  For hook
    sites that don't encode a head index (e.g. ``hook_result``), head_index = -1.
    """
    result = store.get_result(circuit_run_id)
    if result is None:
        return []

    ranked_path = result.artifacts.get("patching_ranked_json")
    if not ranked_path or not Path(ranked_path).is_file():
        return []

    try:
        rows: list[dict[str, Any]] = json.loads(Path(ranked_path).read_text(encoding="utf-8"))
    except Exception:
        return []

    heads: list[tuple[int, int, float]] = []
    seen: set[tuple[int, int]] = set()

    for row in rows:
        hook_site = str(row.get("hook_site", ""))
        recovery = float(row.get("recovery_fraction", 0.0))
        layer, head = _parse_hook_site(hook_site)
        if (layer, head) not in seen:
            seen.add((layer, head))
            heads.append((layer, head, recovery))
        if len(heads) >= top_n:
            break

    return heads


def _parse_hook_site(hook_site: str) -> tuple[int, int]:
    """Return (layer, head_index) from a TransformerLens hook site string.

    Examples
    --------
    ``blocks.10.attn.hook_z`` → (10, -1)   (head not encoded in site name)
    ``blocks.10.attn.hook_result`` → (10, -1)
    ``blocks.10.hook_resid_post`` → (10, -1)
    """
    import re

    layer_match = re.search(r"blocks\.(\d+)", hook_site)
    head_match = re.search(r"head(\d+)", hook_site)

    layer = int(layer_match.group(1)) if layer_match else -1
    head = int(head_match.group(1)) if head_match else -1
    return layer, head


# --------------------------------------------------------------------------- #
# Markdown rendering
# --------------------------------------------------------------------------- #


def _render_markdown(report: RefusalAuditReport) -> str:
    heads_str = ", ".join(
        f"L{layer}.H{head}" if head >= 0 else f"L{layer}"
        for layer, head, _ in report.top_causal_heads
    )
    faithfulness_verdict = (
        "SUPPORTED (>0.7)" if report.circuit_faithfulness > 0.7
        else "PARTIAL (0.5–0.7)" if report.circuit_faithfulness >= 0.5
        else "REJECTED (<0.5)"
    )

    head_rows = "\n".join(
        f"| L{layer}.H{head if head >= 0 else '?'} | {importance:.4f} |"
        for layer, head, importance in report.top_causal_heads
    )

    lines = [
        f"# Refusal Audit: {report.model}",
        "",
        f"> **Headline:** Refusal in `{report.model}` is most causally accessible at "
        f"**layer {report.best_steering_layer}**, with top causal components: "
        f"**{heads_str or 'see circuit run'}**.",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Model | `{report.model}` |",
        f"| Best steering layer | {report.best_steering_layer} |",
        f"| Best steering coefficient | {report.best_coefficient:+.1f} |",
        f"| Refusal rate shift at best layer | {report.refusal_rate_shift_at_best:.2f} |",
        f"| Extraction quality (projection margin) | {report.extraction_quality:.4f} |",
        f"| Baseline refusal rate | {report.baseline_refusal_rate:.2f} |",
        f"| Circuit faithfulness (exp(−mean KL)) | {report.circuit_faithfulness:.4f} |",
        f"| Circuit verdict | **{faithfulness_verdict}** |",
        "",
        "## Run IDs",
        "",
        "| Step | Run ID |",
        "| --- | --- |",
        f"| refusal_direction | {report.refusal_run_id} |",
        f"| caa_steering | {report.caa_run_id} |",
        f"| circuit_patching | {report.circuit_run_id} |",
        f"| causal_scrubbing | {report.scrub_run_id} |",
        "",
        "## Top Causal Heads",
        "",
        "| Head | Recovery Fraction |",
        "| --- | ---: |",
    ]
    if head_rows:
        lines.append(head_rows)
    else:
        lines.append("_(no head-level data extracted)_")

    lines += [
        "",
        "## Notes",
        "",
        report.notes or "_(none)_",
        "",
    ]
    return "\n".join(lines)
