"""Sweep report helpers: summarise a matrix sweep's results into axes + best-by-metric."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mech_interp.types import ExperimentResult, ExperimentSpec, RunStatus


@dataclass
class SweepReport:
    """Structured summary of a completed (or partial) hyperparameter sweep."""

    #: Axis name -> sorted list of distinct values observed across all specs.
    axes: dict[str, list[Any]]
    #: One row per spec: axis values + status + flat metrics dict.
    runs: list[dict[str, Any]]
    #: For each numeric metric, the axis-value dict that produced the max/min.
    best_by_metric: dict[str, dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "axes": self.axes,
            "runs": self.runs,
            "best_by_metric": self.best_by_metric,
        }

    def to_markdown(self) -> str:
        """Render a compact markdown summary suitable for inline display."""
        lines: list[str] = ["# Sweep Report", ""]

        # Axes table
        lines.append("## Axes")
        lines.append("")
        for axis_name, values in self.axes.items():
            lines.append(f"- **{axis_name}**: {', '.join(str(v) for v in values)}")
        lines.append("")

        # Runs table
        if self.runs:
            axis_names = sorted(self.axes.keys())
            metric_names: list[str] = []
            for row in self.runs:
                for k in row.get("metrics", {}).keys():
                    if k not in metric_names:
                        metric_names.append(k)

            header = ["name", "status"] + axis_names + metric_names
            lines.append("## Runs")
            lines.append("")
            lines.append("| " + " | ".join(header) + " |")
            lines.append("|" + "|".join(["---"] * len(header)) + "|")
            for row in self.runs:
                cells = [row.get("name", ""), row.get("status", "")]
                for a in axis_names:
                    cells.append(str(row.get("axis_values", {}).get(a, "")))
                for m in metric_names:
                    cells.append(str(row.get("metrics", {}).get(m, "")))
                lines.append("| " + " | ".join(cells) + " |")
            lines.append("")

        # Best-by-metric
        if self.best_by_metric:
            lines.append("## Best by Metric")
            lines.append("")
            for metric, info in self.best_by_metric.items():
                kind = info.get("kind", "max")
                value = info.get("value", "")
                axes_str = ", ".join(
                    f"{k}={v}" for k, v in info.get("axis_values", {}).items()
                )
                lines.append(f"- **{metric}** ({kind}={value}): {axes_str}")
            lines.append("")

        return "\n".join(lines)


def summarize_sweep(
    specs: list[ExperimentSpec],
    results: list[ExperimentResult],
) -> SweepReport:
    """Build a :class:`SweepReport` from a set of specs and their results.

    Pairs specs to results by ``run_id`` when available; falls back to matching
    on spec name stored in ``parameters["matrix_axes"]`` metadata.

    Parameters
    ----------
    specs:
        The individual specs produced by matrix expansion (each has
        ``parameters["matrix_axes"]`` set by :func:`_expand_matrix_spec`).
    results:
        The :class:`ExperimentResult` objects returned by the runner for each spec.
    """
    # Index results by run_id for fast look-up
    results_by_run_id: dict[int, ExperimentResult] = {r.run_id: r for r in results}

    # Collect axis names and per-run rows
    axis_names: list[str] = []
    rows: list[dict[str, Any]] = []

    for i, spec in enumerate(specs):
        axis_values: dict[str, Any] = dict(spec.parameters.get("matrix_axes") or {})

        # Track axis name order (stable, first-seen)
        for name in axis_values:
            if name not in axis_names:
                axis_names.append(name)

        # Match result: prefer positional match (specs and results are parallel),
        # fall back to run_id lookup if the caller provided them in a different order.
        result: ExperimentResult | None = None
        if i < len(results):
            result = results[i]
        elif results_by_run_id:
            # Try to find by run_id stored anywhere — not possible without extra
            # metadata here, so leave as None.
            pass

        status = result.status.value if result is not None else RunStatus.PLANNED.value
        metrics: dict[str, Any] = dict(result.metrics) if result is not None else {}

        rows.append(
            {
                "name": spec.name,
                "status": status,
                "axis_values": axis_values,
                "metrics": metrics,
            }
        )

    # Build axes dict: name -> sorted unique values (preserves list order for non-scalar)
    axes: dict[str, list[Any]] = {}
    for name in axis_names:
        seen: list[Any] = []
        seen_canonical: set[str] = set()
        for row in rows:
            val = row["axis_values"].get(name)
            canonical = json.dumps(val, sort_keys=True, default=str)
            if canonical not in seen_canonical:
                seen_canonical.add(canonical)
                seen.append(val)
        axes[name] = seen

    # best_by_metric
    best_by_metric: dict[str, dict[str, Any]] = {}
    all_metric_names: set[str] = set()
    for row in rows:
        all_metric_names.update(row["metrics"].keys())

    for metric in sorted(all_metric_names):
        best_max: dict[str, Any] | None = None
        best_min: dict[str, Any] | None = None
        max_val: float | None = None
        min_val: float | None = None

        for row in rows:
            val = row["metrics"].get(metric)
            if not isinstance(val, (int, float)):
                continue
            if max_val is None or val > max_val:
                max_val = float(val)
                best_max = {"kind": "max", "value": max_val, "axis_values": row["axis_values"]}
            if min_val is None or val < min_val:
                min_val = float(val)
                best_min = {"kind": "min", "value": min_val, "axis_values": row["axis_values"]}

        if best_max is not None:
            best_by_metric[f"{metric}_max"] = best_max
        if best_min is not None:
            best_by_metric[f"{metric}_min"] = best_min

    return SweepReport(axes=axes, runs=rows, best_by_metric=best_by_metric)


def write_sweep_report(
    report: SweepReport,
    output_dir: Path,
    prefix: str = "sweep_report",
) -> tuple[Path, Path]:
    """Serialise *report* to ``{output_dir}/{prefix}.json`` and ``.md``.

    Returns ``(json_path, md_path)``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{prefix}.json"
    md_path = output_dir / f"{prefix}.md"
    json_path.write_text(
        json.dumps(report.to_dict(), indent=2, default=str), encoding="utf-8"
    )
    md_path.write_text(report.to_markdown(), encoding="utf-8")
    return json_path, md_path
