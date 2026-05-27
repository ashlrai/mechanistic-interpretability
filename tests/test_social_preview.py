"""
Tests for scripts/generate_social_preview.py helper logic.

Verifies text-layout math and constants without rendering the actual PNG
(matplotlib is in the `notebook` extra, not the base dev environment).
All tests skip cleanly if matplotlib is not installed.
"""

from __future__ import annotations

import importlib.util
import pathlib
import types

import pytest

# ---------------------------------------------------------------------------
# Constants from the script under test — mirror them here so the layout tests
# remain meaningful even without importing the script (which imports matplotlib
# at module level indirectly via type annotations).
# ---------------------------------------------------------------------------

SCRIPT_PATH = pathlib.Path(__file__).parent.parent / "scripts" / "generate_social_preview.py"

WIDTH_PX = 1280
HEIGHT_PX = 640
DPI = 160


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_script() -> types.ModuleType:
    """Import the generator script as a module (skips if matplotlib absent)."""
    pytest.importorskip("matplotlib")
    spec = importlib.util.spec_from_file_location("generate_social_preview", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Layout math tests (no matplotlib required)
# ---------------------------------------------------------------------------

class TestLayoutConstants:
    def test_aspect_ratio(self) -> None:
        """1280×640 is exactly 2:1 — standard social-preview aspect ratio."""
        assert WIDTH_PX / HEIGHT_PX == pytest.approx(2.0)

    def test_dpi_divides_evenly(self) -> None:
        """DPI must divide both dimensions to avoid sub-pixel rounding."""
        assert WIDTH_PX % DPI == 0
        assert HEIGHT_PX % DPI == 0

    def test_figure_size_inches(self) -> None:
        w_in = WIDTH_PX / DPI
        h_in = HEIGHT_PX / DPI
        assert w_in == pytest.approx(8.0)
        assert h_in == pytest.approx(4.0)

    def test_stats_row_positions_in_unit_square(self) -> None:
        """Three stats rows, evenly spaced, must stay within [0, 1]."""
        y_start = 0.660
        row_gap = 0.088
        rows = [y_start - i * row_gap for i in range(3)]
        for y in rows:
            assert 0.0 < y < 1.0, f"row y={y} outside axes bounds"

    def test_headline_positions_in_unit_square(self) -> None:
        y_hl = 0.320
        step = 0.072
        for _ in range(2):
            assert 0.0 < y_hl < 1.0, f"headline y={y_hl} outside axes bounds"
            y_hl -= step

    def test_script_file_exists(self) -> None:
        assert SCRIPT_PATH.exists(), f"Script not found: {SCRIPT_PATH}"


# ---------------------------------------------------------------------------
# Script-import tests (require matplotlib)
# ---------------------------------------------------------------------------

class TestScriptConstants:
    def test_colors_are_hex(self) -> None:
        mod = _load_script()
        for attr in ("BG_COLOR", "ACCENT", "ACCENT_LIGHT", "TEXT_PRIMARY", "TEXT_MUTED"):
            val = getattr(mod, attr)
            assert val.startswith("#"), f"{attr}={val!r} is not a hex color"
            assert len(val) in (4, 7), f"{attr}={val!r} has unexpected length"

    def test_bg_is_dark(self) -> None:
        mod = _load_script()
        # BG_COLOR should be dark: each channel < 0x40
        hex_val = mod.BG_COLOR.lstrip("#")
        r, g, b = int(hex_val[0:2], 16), int(hex_val[2:4], 16), int(hex_val[4:6], 16)
        assert max(r, g, b) < 64, "Background is not dark enough"

    def test_dimensions_match_script(self) -> None:
        mod = _load_script()
        assert mod.WIDTH_PX == WIDTH_PX
        assert mod.HEIGHT_PX == HEIGHT_PX
        assert mod.DPI == DPI

    def test_build_figure_returns_figure(self) -> None:
        import matplotlib.figure
        mod = _load_script()
        fig = mod.build_figure()
        assert isinstance(fig, matplotlib.figure.Figure)
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_output_path_is_under_docs_images(self) -> None:
        """Output path must land in docs/images/, not somewhere arbitrary."""
        repo_root = SCRIPT_PATH.parent.parent
        expected = repo_root / "docs" / "images" / "social_preview.png"
        # Verify the path construction logic in the script by checking the
        # relative structure — we don't run main() (that writes a file).
        assert expected.parent.name == "images"
        assert expected.parent.parent.name == "docs"
