"""Corpus loading and tokenization utilities for SAE training.

Supports two input formats:
- JSONL: one JSON object per line, must have a ``"text"`` field.
- Plain text: one document per non-empty line.

``tokenize_corpus`` packs documents into fixed-length sequences using
the HookedTransformer's tokenizer. Sequences are truncated or left-padded
to ``seq_len`` tokens; documents longer than ``seq_len`` are truncated at
the token level.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)


def load_text_corpus(
    path: str | Path,
    *,
    max_documents: int | None = None,
) -> list[str]:
    """Load a text corpus from a JSONL or plain-text file.

    Parameters
    ----------
    path:
        Path to a ``.jsonl`` file (each line is ``{"text": "..."}`` or any
        JSON object with a ``"text"`` key) or a ``.txt`` / ``.text`` file
        (one document per non-empty line).
    max_documents:
        If set, stop after collecting this many documents. Useful for fast
        smoke-test runs without reading the full corpus.

    Returns
    -------
    list[str]
        Non-empty document strings, in file order.
    """
    corpus_path = Path(path)
    if not corpus_path.exists():
        raise FileNotFoundError(f"corpus file does not exist: {corpus_path}")
    if not corpus_path.is_file():
        raise ValueError(f"corpus path is not a file: {corpus_path}")

    suffix = corpus_path.suffix.lower()
    if suffix == ".jsonl":
        documents = _load_jsonl_corpus(corpus_path, max_documents=max_documents)
    elif suffix in {".txt", ".text"}:
        documents = _load_text_corpus(corpus_path, max_documents=max_documents)
    else:
        raise ValueError(
            f"unsupported corpus format '{suffix}' for {corpus_path}; "
            "expected .jsonl, .txt, or .text"
        )

    logger.debug("loaded %d documents from %s", len(documents), corpus_path)
    return documents


def tokenize_corpus(
    model: object,
    documents: list[str],
    *,
    seq_len: int = 128,
    max_tokens: int | None = None,
) -> torch.Tensor:
    """Tokenize a list of documents into a fixed-length token tensor.

    Each document is tokenized independently and either truncated to
    ``seq_len`` tokens or zero-padded on the right if shorter. The
    resulting tensor has shape ``(n_docs, seq_len)`` where ``n_docs <=
    len(documents)`` (documents that produce zero tokens are skipped).

    If ``max_tokens`` is set the tensor is capped so that
    ``n_docs * seq_len <= max_tokens`` (i.e. at most
    ``max_tokens // seq_len`` rows are kept).

    Parameters
    ----------
    model:
        A ``HookedTransformer`` instance whose ``.tokenizer`` attribute
        is a HuggingFace fast tokenizer.
    documents:
        Raw text strings to tokenize.
    seq_len:
        Number of tokens per row in the output tensor.
    max_tokens:
        Optional cap on the total number of tokens returned. If set,
        at most ``max_tokens // seq_len`` documents are kept.

    Returns
    -------
    torch.Tensor
        Integer tensor of shape ``(n_docs, seq_len)``.
    """
    import torch

    if not documents:
        raise ValueError("tokenize_corpus requires at least one document")
    if seq_len < 1:
        raise ValueError(f"seq_len must be >= 1, got {seq_len}")

    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None:
        raise AttributeError(
            "model has no 'tokenizer' attribute; expected a HookedTransformer instance"
        )

    max_docs: int | None = None
    if max_tokens is not None:
        if max_tokens < seq_len:
            raise ValueError(
                f"max_tokens={max_tokens} is smaller than seq_len={seq_len}; "
                "no complete sequences can be produced"
            )
        max_docs = max_tokens // seq_len

    rows: list[torch.Tensor] = []
    skipped = 0
    for doc in documents:
        if max_docs is not None and len(rows) >= max_docs:
            break
        encoded = tokenizer.encode(doc, add_special_tokens=False)
        if not encoded:
            skipped += 1
            continue
        token_ids = encoded[:seq_len]
        if len(token_ids) < seq_len:
            token_ids = token_ids + [tokenizer.pad_token_id or 0] * (seq_len - len(token_ids))
        rows.append(torch.tensor(token_ids, dtype=torch.long))

    if skipped:
        logger.debug("tokenize_corpus: skipped %d documents that produced no tokens", skipped)

    if not rows:
        raise ValueError(
            "tokenize_corpus: all documents produced empty token sequences"
        )

    return torch.stack(rows, dim=0)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_jsonl_corpus(path: Path, *, max_documents: int | None) -> list[str]:
    documents: list[str] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, start=1):
            if max_documents is not None and len(documents) >= max_documents:
                break
            stripped = line.strip()
            if not stripped:
                continue
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid JSONL in {path} on line {line_number}: {exc.msg}"
                ) from exc
            if not isinstance(obj, dict):
                raise ValueError(
                    f"invalid JSONL in {path} on line {line_number}: expected a JSON object"
                )
            text = obj.get("text")
            if not isinstance(text, str):
                raise ValueError(
                    f"invalid corpus record in {path} on line {line_number}: "
                    "'text' field must be a string"
                )
            text = text.strip()
            if text:
                documents.append(text)
    return documents


def _load_text_corpus(path: Path, *, max_documents: int | None) -> list[str]:
    documents: list[str] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if max_documents is not None and len(documents) >= max_documents:
                break
            text = line.strip()
            if text and not text.startswith("#"):
                documents.append(text)
    return documents
