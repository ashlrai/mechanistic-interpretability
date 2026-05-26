"""SVG timeline renderer for experiment runs.

Produces an inline SVG (no JS required) with one swimlane per family,
dots coloured by run status, and native browser <title> tooltips.
"""
from __future__ import annotations

import html
from datetime import UTC, datetime, timedelta
from typing import Any

from mech_interp.types import ExperimentRun, RunStatus

# SVG canvas constants
SVG_WIDTH = 620
SVG_HEIGHT = 420
MARGIN_LEFT = 140  # room for family labels
MARGIN_RIGHT = 20
MARGIN_TOP = 30
MARGIN_BOTTOM = 40
PLOT_W = SVG_WIDTH - MARGIN_LEFT - MARGIN_RIGHT
DOT_R = 6

# Status colours
_STATUS_COLOUR: dict[str, str] = {
    RunStatus.SUCCEEDED: "#22c55e",   # green
    RunStatus.FAILED: "#ef4444",       # red
    RunStatus.RUNNING: "#3b82f6",      # blue
    RunStatus.PLANNED: "#94a3b8",      # grey
}
_DEFAULT_COLOUR = "#94a3b8"


def _status_colour(status: RunStatus) -> str:
    return _STATUS_COLOUR.get(status, _DEFAULT_COLOUR)


def _window_start(window: str, now: datetime) -> datetime:
    if window == "24h":
        return now - timedelta(hours=24)
    if window == "7d":
        return now - timedelta(days=7)
    if window == "30d":
        return now - timedelta(days=30)
    # "all" — go back far enough
    return datetime.min.replace(tzinfo=now.tzinfo)


def build_timeline_svg(
    runs: list[ExperimentRun],
    window: str = "all",
    now: datetime | None = None,
) -> tuple[str, list[str]]:
    """Render run history as an SVG string.

    Returns:
        (svg_string, family_labels) — family_labels is the ordered swimlane list.
    """
    _now = now or datetime.now(tz=UTC)
    cutoff = _window_start(window, _now)

    # Filter by time window — runs without timezone info are treated as UTC
    def _ts(run: ExperimentRun) -> datetime:
        ts = run.created_at
        if ts.tzinfo is None:
            return ts.replace(tzinfo=UTC)
        return ts

    visible = [r for r in runs if _ts(r) >= cutoff]

    # Collect ordered families (by first appearance)
    families: list[str] = []
    seen_fam: set[str] = set()
    for r in sorted(visible, key=_ts):
        if r.family not in seen_fam:
            families.append(r.family)
            seen_fam.add(r.family)

    if not visible or not families:
        return _empty_svg(), []

    n_lanes = len(families)
    fam_index = {f: i for i, f in enumerate(families)}

    # Time axis
    times = [_ts(r) for r in visible]
    t_min = min(times)
    t_max = max(times)
    t_span = (t_max - t_min).total_seconds() or 1.0  # avoid divide-by-zero

    # Swimlane layout
    usable_h = SVG_HEIGHT - MARGIN_TOP - MARGIN_BOTTOM
    lane_h = usable_h / max(n_lanes, 1)

    def _x(ts: datetime) -> float:
        frac = (ts - t_min).total_seconds() / t_span
        return MARGIN_LEFT + frac * PLOT_W

    def _y(family: str) -> float:
        idx = fam_index[family]
        return MARGIN_TOP + (idx + 0.5) * lane_h

    parts: list[str] = []

    # SVG header
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{SVG_WIDTH}" height="{SVG_HEIGHT}" '
        f'style="font-family:ui-sans-serif,system-ui,sans-serif;font-size:12px;">'
    )

    # Background
    parts.append(f'<rect width="{SVG_WIDTH}" height="{SVG_HEIGHT}" fill="#f7f8fa"/>')

    # Swimlane bands
    for i, fam in enumerate(families):
        lane_y = MARGIN_TOP + i * lane_h
        fill = "#ffffff" if i % 2 == 0 else "#eef1f4"
        parts.append(
            f'<rect x="{MARGIN_LEFT}" y="{lane_y:.1f}" '
            f'width="{PLOT_W}" height="{lane_h:.1f}" fill="{fill}"/>'
        )
        # Family label (truncated at 18 chars)
        label = fam if len(fam) <= 18 else fam[:17] + "…"
        label_y = lane_y + lane_h / 2 + 4
        parts.append(
            f'<text x="{MARGIN_LEFT - 6}" y="{label_y:.1f}" '
            f'text-anchor="end" fill="#172026">{html.escape(label)}</text>'
        )

    # Time axis line
    axis_y = SVG_HEIGHT - MARGIN_BOTTOM
    parts.append(
        f'<line x1="{MARGIN_LEFT}" y1="{axis_y}" '
        f'x2="{MARGIN_LEFT + PLOT_W}" y2="{axis_y}" '
        f'stroke="#d8dee4" stroke-width="1"/>'
    )

    # Axis tick labels (start and end)
    fmt = "%Y-%m-%d %H:%M" if t_span > 3600 * 24 else "%H:%M"
    parts.append(
        f'<text x="{MARGIN_LEFT}" y="{axis_y + 14}" '
        f'fill="#66717a">{html.escape(t_min.strftime(fmt))}</text>'
    )
    parts.append(
        f'<text x="{MARGIN_LEFT + PLOT_W}" y="{axis_y + 14}" '
        f'text-anchor="end" fill="#66717a">{html.escape(t_max.strftime(fmt))}</text>'
    )

    # Dots — one per run
    # Build top metric string for tooltip
    for run in visible:
        cx = _x(_ts(run))
        cy = _y(run.family)
        colour = _status_colour(run.status)
        # Tooltip: run id + spec + status (no metrics on ExperimentRun; shown from result)
        tooltip = f"Run {run.id}: {run.spec_name} [{run.status.value}]"
        # clickable: navigate to /runs/<id>
        parts.append(
            f'<a href="/runs/{run.id}">'
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{DOT_R}" '
            f'fill="{colour}" stroke="white" stroke-width="1.5" '
            f'style="cursor:pointer">'
            f"<title>{html.escape(tooltip)}</title>"
            f"</circle>"
            f"</a>"
        )

    parts.append("</svg>")
    return "\n".join(parts), families


def _empty_svg() -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{SVG_WIDTH}" height="120" '
        f'style="font-family:ui-sans-serif,system-ui,sans-serif;font-size:14px;">'
        f'<rect width="{SVG_WIDTH}" height="120" fill="#f7f8fa"/>'
        f'<text x="{SVG_WIDTH // 2}" y="65" text-anchor="middle" fill="#66717a">'
        f"No runs in the selected time window."
        f"</text>"
        f"</svg>"
    )


# Expose the filter options for the template
TIME_WINDOWS: list[dict[str, Any]] = [
    {"value": "24h", "label": "Last 24 h"},
    {"value": "7d", "label": "Last 7 days"},
    {"value": "30d", "label": "Last 30 days"},
    {"value": "all", "label": "All time"},
]
