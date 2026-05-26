"""Unit tests for the corpus downloader helper.

No real network calls are made. ``huggingface_hub`` is monkeypatched so the
tests run fast in CI/offline environments.  If huggingface_hub is not installed
the tests xfail (not silently skipped) so the dependency gap is visible.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Availability gate
# ---------------------------------------------------------------------------

_HF_HUB_AVAILABLE = True
try:
    import huggingface_hub  # noqa: F401
except ImportError:
    _HF_HUB_AVAILABLE = False

_hf_required = pytest.mark.xfail(
    not _HF_HUB_AVAILABLE,
    reason="huggingface_hub not installed",
    strict=False,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_jsonl_download(
    documents: list[str],
    *,
    text_field: str = "text",
) -> Any:
    """Return a callable that mimics hf_hub_download by writing a temp JSONL."""
    import tempfile

    # Write the fake JSONL once at creation time; return its path on every call.
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
    )
    for doc in documents:
        tmp.write(json.dumps({text_field: doc}) + "\n")
    tmp.close()
    fake_path = tmp.name

    def _download(*args: Any, **kwargs: Any) -> str:
        return fake_path

    return _download


# ---------------------------------------------------------------------------
# Tests: CORPORA registry
# ---------------------------------------------------------------------------


@_hf_required
def test_corpora_registry_has_expected_keys() -> None:
    from mech_interp.datasets.downloader import CORPORA

    assert "pile-1k" in CORPORA
    assert "owt-1k" in CORPORA
    for descriptor in CORPORA.values():
        assert descriptor.hf_repo
        assert descriptor.hf_split
        assert descriptor.text_field
        assert descriptor.license


@_hf_required
def test_corpus_descriptor_is_frozen() -> None:
    from mech_interp.datasets.downloader import CORPORA

    descriptor = CORPORA["pile-1k"]
    with pytest.raises((AttributeError, TypeError)):
        descriptor.name = "should-fail"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Tests: --list equivalent
# ---------------------------------------------------------------------------


@_hf_required
def test_corpora_list_contains_all_names(capsys: pytest.CaptureFixture[str]) -> None:
    """All CORPORA keys must be printable without error."""
    from mech_interp.datasets.downloader import CORPORA

    names = sorted(CORPORA.keys())
    assert len(names) >= 2
    # Print them exactly as the CLI would
    output = "\n".join(names)
    print(output)
    captured = capsys.readouterr()
    for name in names:
        assert name in captured.out


# ---------------------------------------------------------------------------
# Tests: download_corpus with mocked fetch
# ---------------------------------------------------------------------------


@_hf_required
def test_download_corpus_writes_jsonl_with_text_field(tmp_path: Path) -> None:
    """Mock the HF hub download; verify output JSONL has {text: ...} lines."""
    from mech_interp.datasets.downloader import CORPORA, download_corpus

    fake_docs = ["Hello world.", "Second document.", "Third sentence here."]
    dest = tmp_path / "out.jsonl"

    descriptor = CORPORA["pile-1k"]

    # Patch at the huggingface_hub module level (functions are looked up there
    # at call time since we do `import huggingface_hub` inside the helper).
    with (
        patch(
            "mech_interp.datasets.downloader._iter_via_datasets_library",
            side_effect=ImportError("no datasets"),
        ),
        patch(
            "huggingface_hub.list_repo_files",
            return_value=["data/train-00000-of-00001.jsonl"],
        ),
        patch(
            "huggingface_hub.hf_hub_download",
            side_effect=_fake_jsonl_download(fake_docs, text_field=descriptor.text_field),
        ),
    ):
        result = download_corpus("pile-1k", max_documents=10, dest=dest)

    assert result == dest
    assert dest.exists()
    lines = [json.loads(line) for line in dest.read_text(encoding="utf-8").splitlines() if line]
    assert len(lines) == len(fake_docs)
    for line, expected in zip(lines, fake_docs, strict=True):
        assert line["text"] == expected


@_hf_required
def test_download_corpus_respects_max_documents(tmp_path: Path) -> None:
    """max_documents=2 should write exactly 2 lines even when more are available."""
    from mech_interp.datasets.downloader import CORPORA, download_corpus

    fake_docs = [f"Document number {i}." for i in range(10)]
    dest = tmp_path / "capped.jsonl"
    descriptor = CORPORA["owt-1k"]

    with (
        patch(
            "mech_interp.datasets.downloader._iter_via_datasets_library",
            side_effect=ImportError("no datasets"),
        ),
        patch(
            "huggingface_hub.list_repo_files",
            return_value=[f"data/{descriptor.hf_split}-00000.jsonl"],
        ),
        patch(
            "huggingface_hub.hf_hub_download",
            side_effect=_fake_jsonl_download(fake_docs, text_field=descriptor.text_field),
        ),
    ):
        download_corpus("owt-1k", max_documents=2, dest=dest)

    lines = dest.read_text(encoding="utf-8").splitlines()
    non_empty = [line for line in lines if line.strip()]
    assert len(non_empty) == 2


@_hf_required
def test_download_corpus_unknown_name_raises(tmp_path: Path) -> None:
    from mech_interp.datasets.downloader import download_corpus

    with pytest.raises(ValueError, match="Unknown corpus"):
        download_corpus("does-not-exist", dest=tmp_path / "out.jsonl")


@_hf_required
def test_download_corpus_summary_fields(tmp_path: Path) -> None:
    from mech_interp.datasets.downloader import corpus_download_summary

    dest = tmp_path / "summary_test.jsonl"
    docs = ["Alpha beta gamma.", "Delta epsilon."]
    dest.write_text(
        "\n".join(json.dumps({"text": d}) for d in docs) + "\n",
        encoding="utf-8",
    )

    summary = corpus_download_summary(dest, docs)

    assert summary["line_count"] == 2
    assert isinstance(summary["sha256"], str) and len(str(summary["sha256"])) == 64
    assert summary["total_chars"] == sum(len(d) for d in docs)
    assert str(summary["path"]) == str(dest)


@_hf_required
def test_download_corpus_jsonl_blank_lines_skipped(tmp_path: Path) -> None:
    """Blank and whitespace-only documents must be skipped in the output."""
    from mech_interp.datasets.downloader import CORPORA, download_corpus

    fake_docs = ["Real document.", "  ", "", "Another real doc."]
    dest = tmp_path / "blanks.jsonl"
    descriptor = CORPORA["pile-1k"]

    with (
        patch(
            "mech_interp.datasets.downloader._iter_via_datasets_library",
            side_effect=ImportError("no datasets"),
        ),
        patch(
            "huggingface_hub.list_repo_files",
            return_value=[f"data/{descriptor.hf_split}-00000.jsonl"],
        ),
        patch(
            "huggingface_hub.hf_hub_download",
            side_effect=_fake_jsonl_download(fake_docs, text_field=descriptor.text_field),
        ),
    ):
        download_corpus("pile-1k", max_documents=100, dest=dest)

    lines = [json.loads(line) for line in dest.read_text().splitlines() if line.strip()]
    texts = [line["text"] for line in lines]
    assert "Real document." in texts
    assert "Another real doc." in texts
    assert "" not in texts
    assert "  " not in texts


# ---------------------------------------------------------------------------
# CLI integration: mech download-corpus --list
# ---------------------------------------------------------------------------


@_hf_required
def test_cli_list_exits_zero(tmp_path: Path) -> None:
    """mech download-corpus --list should exit 0 and print corpus names."""
    from typer.testing import CliRunner

    from mech_interp.cli import app
    from mech_interp.datasets.downloader import CORPORA

    runner = CliRunner()
    result = runner.invoke(app, ["download-corpus", "--list"])

    assert result.exit_code == 0, f"Non-zero exit: {result.output}"
    for name in CORPORA:
        assert name in result.output
