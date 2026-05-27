#!/usr/bin/env python3
"""
Generate docs/images/social_preview.png (1280x640) for the v0.1.0-preview release.

Run with:
    uv run python scripts/generate_social_preview.py

Requires matplotlib (available in the `notebook` dependency group):
    uv run --group notebook python scripts/generate_social_preview.py
"""

from __future__ import annotations

import pathlib
import sys


# ── Layout constants (all in figure-fraction units 0..1) ────────────────────

WIDTH_PX = 1280
HEIGHT_PX = 640
DPI = 160  # 1280/160=8 in, 640/160=4 in

BG_COLOR = "#0f1923"
ACCENT = "#176b87"
ACCENT_LIGHT = "#54c5d0"
TEXT_PRIMARY = "#e8f4f8"
TEXT_MUTED = "#7fb3c8"
BORDER_COLOR = "#1e3a4a"


def _border_box(ax: "matplotlib.axes.Axes") -> None:  # type: ignore[name-defined]
    """Draw a rounded border rectangle on the axes."""
    from matplotlib.patches import FancyBboxPatch

    box = FancyBboxPatch(
        (0.018, 0.03),
        0.964,
        0.94,
        boxstyle="round,pad=0.01",
        linewidth=2,
        edgecolor=ACCENT,
        facecolor=BORDER_COLOR,
        alpha=0.55,
        transform=ax.transAxes,
        zorder=0,
    )
    ax.add_patch(box)


def _hline(
    ax: "matplotlib.axes.Axes",  # type: ignore[name-defined]
    y: float,
    x0: float = 0.03,
    x1: float = 0.97,
    linewidth: float = 1.5,
    color: str = ACCENT,
    alpha: float = 0.6,
) -> None:
    """Horizontal line in axes-fraction coordinates."""
    ax.plot(
        [x0, x1],
        [y, y],
        linewidth=linewidth,
        color=color,
        alpha=alpha,
        transform=ax.transAxes,
        solid_capstyle="butt",
    )


def _accent_bar(ax: "matplotlib.axes.Axes") -> None:  # type: ignore[name-defined]
    """Thin horizontal accent stripe under the title."""
    _hline(ax, y=0.735, linewidth=1.5, color=ACCENT, alpha=0.6)


def _dot(ax: "matplotlib.axes.Axes", x: float, y: float) -> None:  # type: ignore[name-defined]
    """Small filled square bullet."""
    ax.plot(
        x,
        y,
        marker="s",
        markersize=5,
        color=ACCENT_LIGHT,
        transform=ax.transAxes,
        zorder=3,
        linestyle="None",
    )


def build_figure() -> "matplotlib.figure.Figure":  # type: ignore[name-defined]
    """Build and return the social preview figure."""
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.rcParams["font.family"] = "monospace"

    fig = plt.figure(figsize=(WIDTH_PX / DPI, HEIGHT_PX / DPI), dpi=DPI)
    fig.patch.set_facecolor(BG_COLOR)

    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(BG_COLOR)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    _border_box(ax)
    _accent_bar(ax)

    # ── Title block ──────────────────────────────────────────────────────────
    ax.text(
        0.5,
        0.865,
        "Mechanistic Interpretability Platform",
        ha="center",
        va="center",
        fontsize=22,
        fontweight="bold",
        color=TEXT_PRIMARY,
        transform=ax.transAxes,
    )
    ax.text(
        0.5,
        0.795,
        "Local mech-interp research at the speed of curiosity",
        ha="center",
        va="center",
        fontsize=12,
        color=ACCENT_LIGHT,
        transform=ax.transAxes,
    )

    # ── Stats grid (two columns, three rows) ─────────────────────────────────
    stats = [
        ("14 experiment families", "527 tests passing"),
        ("55 CLI commands", "19-second quickstart"),
        ("2 publishable findings", "Apache 2 / MIT licensed"),
    ]

    y_start = 0.660
    row_gap = 0.088
    col_left = 0.08
    col_right = 0.55
    bullet_offset = 0.022

    for i, (left, right) in enumerate(stats):
        y = y_start - i * row_gap
        _dot(ax, col_left - bullet_offset, y)
        ax.text(
            col_left,
            y,
            left,
            ha="left",
            va="center",
            fontsize=11,
            color=TEXT_PRIMARY,
            transform=ax.transAxes,
        )
        _dot(ax, col_right - bullet_offset, y)
        ax.text(
            col_right,
            y,
            right,
            ha="left",
            va="center",
            fontsize=11,
            color=TEXT_PRIMARY,
            transform=ax.transAxes,
        )

    # ── Divider ──────────────────────────────────────────────────────────────
    _hline(ax, y=0.385, x0=0.05, x1=0.95, linewidth=0.8, alpha=0.35)

    # ── Headline findings ────────────────────────────────────────────────────
    headlines = [
        "  Headline: SAE features are NOT seed-stable  (live cosine 0.50 L0 · 0.32 L6 · 0 cross 0.9 threshold)",
        "  Headline: standard abliteration recipe fails on Qwen  (faithfulness 0.041, hypothesis rejected)",
    ]
    y_hl = 0.320
    for headline in headlines:
        ax.text(
            0.5,
            y_hl,
            headline,
            ha="center",
            va="center",
            fontsize=9.5,
            color=ACCENT_LIGHT,
            fontstyle="italic",
            transform=ax.transAxes,
        )
        y_hl -= 0.072

    # ── Footer ───────────────────────────────────────────────────────────────
    ax.text(
        0.05,
        0.095,
        "github.com/ashlrai/mechanistic-interpretability",
        ha="left",
        va="center",
        fontsize=9,
        color=TEXT_MUTED,
        transform=ax.transAxes,
    )
    ax.text(
        0.95,
        0.095,
        "v0.1.0-preview",
        ha="right",
        va="center",
        fontsize=9,
        fontweight="bold",
        color=ACCENT_LIGHT,
        transform=ax.transAxes,
    )
    _hline(ax, y=0.118, linewidth=0.8, alpha=0.35)

    return fig


def main() -> None:
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        print(
            "matplotlib is required. Install with:\n"
            "  uv run --group notebook python scripts/generate_social_preview.py",
            file=sys.stderr,
        )
        sys.exit(1)

    import matplotlib.pyplot as plt

    repo_root = pathlib.Path(__file__).parent.parent
    out_path = repo_root / "docs" / "images" / "social_preview.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig = build_figure()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)

    size_kb = out_path.stat().st_size // 1024
    print(f"Written: {out_path}  ({size_kb} KB, {WIDTH_PX}x{HEIGHT_PX} px)")


if __name__ == "__main__":
    main()
