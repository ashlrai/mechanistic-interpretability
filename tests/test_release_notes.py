"""
Verify that docs/release-notes/v0.1.0-preview.md has the expected structure.

These are fast, dependency-free tests — they only read the markdown file.
"""

from __future__ import annotations

import pathlib
import re

import pytest

NOTES_PATH = (
    pathlib.Path(__file__).parent.parent
    / "docs"
    / "release-notes"
    / "v0.1.0-preview.md"
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def notes_text() -> str:
    assert NOTES_PATH.exists(), f"Release notes not found: {NOTES_PATH}"
    return NOTES_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def headings(notes_text: str) -> list[str]:
    """All markdown heading lines (## level or higher)."""
    return [
        line.strip()
        for line in notes_text.splitlines()
        if re.match(r"^#{1,3} ", line)
    ]


# ---------------------------------------------------------------------------
# Section presence
# ---------------------------------------------------------------------------

class TestRequiredSections:
    def test_has_summary(self, notes_text: str) -> None:
        assert "## Summary" in notes_text

    def test_has_headline_findings(self, notes_text: str) -> None:
        assert "## Headline findings" in notes_text

    def test_has_capabilities(self, notes_text: str) -> None:
        assert "## Capabilities" in notes_text

    def test_has_known_gaps(self, notes_text: str) -> None:
        assert "## Known gaps" in notes_text

    def test_has_reproducibility(self, notes_text: str) -> None:
        assert "## Reproducibility" in notes_text

    def test_has_acknowledgments(self, notes_text: str) -> None:
        assert "## Acknowledgments" in notes_text

    def test_has_notable_changes(self, notes_text: str) -> None:
        assert "## Notable changes" in notes_text


# ---------------------------------------------------------------------------
# Headline numbers — every cited number must appear verbatim
# ---------------------------------------------------------------------------

class TestHeadlineNumbers:
    """Guard against accidentally editing out the key statistics."""

    def test_sae_layer0_cosine(self, notes_text: str) -> None:
        assert "0.50" in notes_text, "Layer-0 live cosine 0.50 missing"

    def test_sae_layer6_cosine(self, notes_text: str) -> None:
        assert "0.32" in notes_text, "Layer-6 live cosine 0.32 missing"

    def test_sae_layer6_512_cosine(self, notes_text: str) -> None:
        assert "0.26" in notes_text, "Layer-6 512-feature cosine 0.26 missing"

    def test_sae_threshold_zero_crossings(self, notes_text: str) -> None:
        # The notes must say no condition crosses 0.9
        assert "0.9" in notes_text

    def test_qwen_extraction_quality(self, notes_text: str) -> None:
        assert "4.105" in notes_text, "Qwen extraction quality 4.105 missing"

    def test_qwen_faithfulness(self, notes_text: str) -> None:
        assert "0.041" in notes_text, "Qwen faithfulness 0.041 missing"


# ---------------------------------------------------------------------------
# Cross-references to source docs
# ---------------------------------------------------------------------------

class TestCitations:
    def test_cites_sae_publication(self, notes_text: str) -> None:
        assert "sae_replication_crisis.md" in notes_text

    def test_cites_refusal_audit(self, notes_text: str) -> None:
        assert "refusal_audit.md" in notes_text


# ---------------------------------------------------------------------------
# Known-gaps honesty checks
# ---------------------------------------------------------------------------

class TestKnownGaps:
    def test_mentions_multi_model_gap(self, notes_text: str) -> None:
        # Must acknowledge that only Qwen has been audited
        assert "Qwen2.5-1.5B" in notes_text

    def test_mentions_cuda_gap(self, notes_text: str) -> None:
        assert "CUDA" in notes_text or "GPU" in notes_text

    def test_mentions_ioi_gap(self, notes_text: str) -> None:
        assert "IOI" in notes_text or "indirect object" in notes_text.lower()


# ---------------------------------------------------------------------------
# Reproducibility block must have shell code
# ---------------------------------------------------------------------------

class TestReproducibility:
    def test_has_mech_command(self, notes_text: str) -> None:
        assert "uv run mech" in notes_text

    def test_has_reproduce_sh_reference(self, notes_text: str) -> None:
        assert "reproduce.sh" in notes_text
