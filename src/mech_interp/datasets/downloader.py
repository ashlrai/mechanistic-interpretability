"""Corpus downloader — fetch small real-text subsets from HuggingFace.

Uses only ``huggingface_hub`` (already a transitive dep of transformer-lens)
to stream individual parquet/JSONL shards.  The optional ``datasets`` package
is used when present for a nicer streaming API; the code falls back to the
plain ``huggingface_hub`` HTTP helpers otherwise.

Typical usage::

    from mech_interp.datasets.downloader import download_corpus
    path = download_corpus("pile-1k", dest=Path("data/prompts/pile-1k.jsonl"))

Or via the CLI::

    mech download-corpus --name pile-1k
"""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Corpus registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CorpusDescriptor:
    """Metadata for a named, mech-interp-friendly corpus subset."""

    name: str
    """Canonical key used on the CLI and in YAML specs."""
    hf_repo: str
    """HuggingFace dataset repo id (``owner/name`` or ``owner/name/subdir``)."""
    hf_split: str
    """Dataset split to pull from (e.g. ``"train"``)."""
    text_field: str
    """JSON field that carries the document text in each record."""
    license: str
    """SPDX identifier or plain-English license description."""


CORPORA: dict[str, CorpusDescriptor] = {
    "pile-1k": CorpusDescriptor(
        name="pile-1k",
        hf_repo="NeelNanda/pile-10k",
        hf_split="train",
        text_field="text",
        license="MIT",
    ),
    "owt-1k": CorpusDescriptor(
        name="owt-1k",
        hf_repo="stas/openwebtext-10k",
        hf_split="train",
        text_field="text",
        license="CC0",
    ),
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def download_corpus(
    name: str,
    *,
    max_documents: int = 1000,
    dest: Path,
    cache_dir: Path | None = None,
) -> Path:
    """Fetch the named corpus and write a JSONL file at *dest*.

    Each output line is a JSON object with a single ``"text"`` key.
    Returns *dest*.

    Parameters
    ----------
    name:
        Key from :data:`CORPORA`.
    max_documents:
        Maximum number of documents to write.  The actual count may be less
        if the upstream dataset is smaller.
    dest:
        Target ``.jsonl`` file path.
    cache_dir:
        Override the HuggingFace cache directory.  Defaults to the standard
        ``~/.cache/huggingface/`` location.
    """
    if name not in CORPORA:
        known = ", ".join(sorted(CORPORA))
        raise ValueError(
            f"Unknown corpus '{name}'. Known corpora: {known}. "
            "Pass --list to see all supported names."
        )
    descriptor = CORPORA[name]
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Downloading corpus '%s' from %s (max_documents=%d) → %s",
        name,
        descriptor.hf_repo,
        max_documents,
        dest,
    )

    cache_str = str(cache_dir) if cache_dir is not None else None
    documents = list(
        _iter_documents(descriptor, max_documents=max_documents, cache_dir=cache_str)
    )
    _write_jsonl(documents, dest)

    sha256 = _sha256(dest)
    total_chars = sum(len(d) for d in documents)
    logger.info(
        "Wrote %d documents (%d chars) to %s  sha256=%s",
        len(documents),
        total_chars,
        dest,
        sha256[:16],
    )
    return dest


def corpus_download_summary(dest: Path, documents: list[str]) -> dict[str, object]:
    """Return a summary dict suitable for printing after a download."""
    sha = _sha256(dest)
    return {
        "path": str(dest),
        "line_count": len(documents),
        "total_chars": sum(len(d) for d in documents),
        "sha256": sha,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _iter_documents(
    descriptor: CorpusDescriptor,
    *,
    max_documents: int,
    cache_dir: str | None,
) -> Iterator[str]:
    """Yield up to *max_documents* text strings from *descriptor*."""
    try:
        yield from _iter_via_datasets_library(descriptor, max_documents=max_documents)
        return
    except ImportError:
        logger.debug("'datasets' library not installed; falling back to huggingface_hub")
    except Exception as exc:
        logger.debug("datasets library failed (%s); falling back to huggingface_hub", exc)

    yield from _iter_via_hf_hub(
        descriptor, max_documents=max_documents, cache_dir=cache_dir
    )


def _iter_via_datasets_library(
    descriptor: CorpusDescriptor,
    *,
    max_documents: int,
) -> Iterator[str]:
    """Use the ``datasets`` package (fast streaming, memory-efficient)."""
    from datasets import load_dataset

    ds = load_dataset(
        descriptor.hf_repo,
        split=descriptor.hf_split,
        streaming=True,
        trust_remote_code=False,
    )
    count = 0
    for row in ds:
        if count >= max_documents:
            break
        text = row.get(descriptor.text_field, "")
        if isinstance(text, str):
            text = text.strip()
            if text:
                yield text
                count += 1


def _iter_via_hf_hub(
    descriptor: CorpusDescriptor,
    *,
    max_documents: int,
    cache_dir: str | None,
) -> Iterator[str]:
    """Fall back to the ``huggingface_hub`` HTTP API.

    This path is always available because huggingface_hub is a transitive
    dependency of transformer-lens.  It downloads the first available parquet
    shard and reads it row by row.
    """
    try:
        import huggingface_hub
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required to download corpora. "
            "Install it with: pip install huggingface_hub"
        ) from exc

    # List data files for this split.
    repo_id = descriptor.hf_repo
    file_infos = list(huggingface_hub.list_repo_files(repo_id, repo_type="dataset"))
    split = descriptor.hf_split

    # Prefer parquet shards; fall back to jsonl.
    parquet_files = [
        f for f in file_infos
        if split in f and f.endswith(".parquet")
    ]
    jsonl_files = [
        f for f in file_infos
        if split in f and (f.endswith(".jsonl") or f.endswith(".jsonl.gz"))
    ]

    if parquet_files:
        yield from _read_parquet_shard(
            repo_id, parquet_files[0], descriptor.text_field,
            max_documents=max_documents, cache_dir=cache_dir,
        )
        return

    if jsonl_files:
        yield from _read_jsonl_shard(
            repo_id, jsonl_files[0], descriptor.text_field,
            max_documents=max_documents, cache_dir=cache_dir,
        )
        return

    raise ValueError(
        f"Could not locate any parquet or JSONL shards for '{repo_id}' "
        f"split='{split}' via huggingface_hub. Try installing the 'datasets' package."
    )


def _read_parquet_shard(
    repo_id: str,
    filename: str,
    text_field: str,
    *,
    max_documents: int,
    cache_dir: str | None,
) -> Iterator[str]:
    import huggingface_hub

    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError(
            "pyarrow is required to read parquet shards. "
            "Install it with: pip install pyarrow"
        ) from exc

    local_path = huggingface_hub.hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        cache_dir=cache_dir,
    )
    table = pq.read_table(local_path, columns=[text_field])  # type: ignore[no-untyped-call]
    count = 0
    for batch in table.to_batches():
        col = batch.column(text_field)
        for value in col:
            if count >= max_documents:
                return
            text = str(value)
            text = text.strip()
            if text and text != "None":
                yield text
                count += 1


def _read_jsonl_shard(
    repo_id: str,
    filename: str,
    text_field: str,
    *,
    max_documents: int,
    cache_dir: str | None,
) -> Iterator[str]:
    import huggingface_hub

    local_path = huggingface_hub.hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        cache_dir=cache_dir,
    )
    opener = gzip.open if filename.endswith(".gz") else open
    count = 0
    with opener(local_path, "rt", encoding="utf-8") as fh:
        for line in fh:
            if count >= max_documents:
                return
            stripped = line.strip()
            if not stripped:
                continue
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            text = obj.get(text_field, "")
            if isinstance(text, str):
                text = text.strip()
                if text:
                    yield text
                    count += 1


def _write_jsonl(documents: list[str], dest: Path) -> None:
    with dest.open("w", encoding="utf-8") as fh:
        for doc in documents:
            fh.write(json.dumps({"text": doc}, ensure_ascii=False) + "\n")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
