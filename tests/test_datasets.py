from pathlib import Path

import pytest

from mech_interp.datasets import PromptDatasetError, PromptRecord, load_prompt_dataset
from mech_interp.experiments.registry import load_experiment_spec


def test_load_jsonl_prompt_dataset_normalizes_records_and_hashes(tmp_path: Path) -> None:
    dataset_path = tmp_path / "facts.jsonl"
    dataset_path.write_text(
        '\n{"id": " capital-france ", "prompt": "  The capital of France is ", '
        '"metadata": {"kind": "factual", "answer": "Paris"}}\n'
        '{"id": "largest-planet", "prompt": "The largest planet is", "answer": "Jupiter"}\n',
        encoding="utf-8",
    )

    dataset = load_prompt_dataset(dataset_path)

    assert dataset.name == "facts"
    assert dataset.prompts == ["The capital of France is", "The largest planet is"]
    assert dataset.records[0].id == "capital-france"
    assert dataset.records[1].metadata["answer"] == "Jupiter"
    assert len(dataset.records[0].sha256) == 64
    assert len(dataset.sha256) == 64

    same_record = PromptRecord(
        id="capital-france",
        prompt="The capital of France is",
        metadata={"answer": "Paris", "kind": "factual"},
    )
    assert dataset.records[0].sha256 == same_record.sha256


def test_load_text_prompt_dataset_skips_blank_lines_and_comments(tmp_path: Path) -> None:
    dataset_path = tmp_path / "plain.txt"
    dataset_path.write_text(
        "# smoke prompts\n\nThe Eiffel Tower is in\r\nThe Colosseum is in\n",
        encoding="utf-8",
    )

    dataset = load_prompt_dataset(dataset_path, name="plain-smoke")

    assert dataset.name == "plain-smoke"
    assert [record.id for record in dataset.records] == ["plain-0001", "plain-0002"]
    assert dataset.prompts == ["The Eiffel Tower is in", "The Colosseum is in"]
    assert dataset.records[0].metadata == {}


def test_dataset_hash_is_stable_across_jsonl_formatting(tmp_path: Path) -> None:
    compact_path = tmp_path / "compact.jsonl"
    spaced_path = tmp_path / "spaced.jsonl"
    compact_path.write_text(
        '{"id":"a","prompt":"Prompt A","metadata":{"b":2,"a":1}}\n',
        encoding="utf-8",
    )
    spaced_path.write_text(
        '{ "metadata": { "a": 1, "b": 2 }, "prompt": "Prompt A", "id": "a" }\n',
        encoding="utf-8",
    )

    assert load_prompt_dataset(compact_path).sha256 == load_prompt_dataset(spaced_path).sha256


def test_load_prompt_dataset_rejects_duplicate_ids(tmp_path: Path) -> None:
    dataset_path = tmp_path / "duplicates.jsonl"
    dataset_path.write_text(
        '{"id": "same", "prompt": "First"}\n{"id": "same", "prompt": "Second"}\n',
        encoding="utf-8",
    )

    with pytest.raises(PromptDatasetError, match="duplicate record id"):
        load_prompt_dataset(dataset_path)


def test_experiment_specs_can_reference_dataset_paths_in_parameters(tmp_path: Path) -> None:
    spec_path = tmp_path / "with-dataset.yaml"
    spec_path.write_text(
        """
name: dataset-reference
family: polysemanticity
backend: transformerlens
parameters:
  dataset_path: data/prompts/factual.jsonl
  dataset_sha256: expected-digest
""",
        encoding="utf-8",
    )

    spec = load_experiment_spec(spec_path)

    assert spec.parameters["dataset_path"] == "data/prompts/factual.jsonl"
    assert spec.parameters["dataset_sha256"] == "expected-digest"


_CORPUS_FILES = {"openwebtext_sample.jsonl"}


def test_curated_prompt_files_load() -> None:
    for path in sorted(Path("data/prompts").glob("*.*")):
        if path.name == "README.md" or path.name in _CORPUS_FILES:
            continue
        dataset = load_prompt_dataset(path)

        assert dataset.records
        assert len(dataset.sha256) == 64


# ---------------------------------------------------------------------------
# Corpus loading tests (load_text_corpus / tokenize_corpus)
# ---------------------------------------------------------------------------

from mech_interp.datasets.corpus import load_text_corpus, tokenize_corpus  # noqa: E402


def test_load_text_corpus_jsonl(tmp_path: Path) -> None:
    corpus_file = tmp_path / "docs.jsonl"
    corpus_file.write_text(
        '{"text": "Hello world"}\n'
        '{"text": "  "}\n'  # blank after strip — should be skipped
        '{"text": "Second document"}\n',
        encoding="utf-8",
    )

    docs = load_text_corpus(corpus_file)

    assert docs == ["Hello world", "Second document"]


def test_load_text_corpus_plain_text(tmp_path: Path) -> None:
    corpus_file = tmp_path / "docs.txt"
    corpus_file.write_text(
        "# comment line\n"
        "\n"
        "First sentence here.\n"
        "Second sentence here.\n",
        encoding="utf-8",
    )

    docs = load_text_corpus(corpus_file)

    assert docs == ["First sentence here.", "Second sentence here."]


def test_load_text_corpus_max_documents(tmp_path: Path) -> None:
    corpus_file = tmp_path / "many.jsonl"
    lines = "\n".join(f'{{"text": "doc {i}"}}' for i in range(20))
    corpus_file.write_text(lines + "\n", encoding="utf-8")

    docs = load_text_corpus(corpus_file, max_documents=5)

    assert len(docs) == 5
    assert docs[0] == "doc 0"
    assert docs[4] == "doc 4"


def test_load_text_corpus_missing_file() -> None:
    import pytest

    with pytest.raises(FileNotFoundError):
        load_text_corpus("/nonexistent/path/corpus.jsonl")


def test_load_text_corpus_bad_text_field(tmp_path: Path) -> None:
    import pytest

    corpus_file = tmp_path / "bad.jsonl"
    corpus_file.write_text('{"text": 42}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="'text' field must be a string"):
        load_text_corpus(corpus_file)


def test_tokenize_corpus_shape_and_padding() -> None:
    """tokenize_corpus with a fake tokenizer produces (n_docs, seq_len) int tensor and mask."""
    import torch

    class _FakeTokenizer:
        pad_token_id = 0

        def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
            # Return one token per word, capped at 200 tokens
            return [hash(w) % 1000 + 1 for w in text.split()][:200]

    class _FakeModel:
        tokenizer = _FakeTokenizer()

    docs = ["hello world", "one two three four five", "a"]
    tokens, mask = tokenize_corpus(_FakeModel(), docs, seq_len=4)

    # Token tensor shape and dtype
    assert tokens.shape == (3, 4)
    assert tokens.dtype == torch.long
    # Mask shape and dtype
    assert mask.shape == (3, 4)
    assert mask.dtype == torch.bool

    # Short document "a" (1 token) should be right-padded with 0s in tokens
    assert tokens[2, 1].item() == 0
    assert tokens[2, 2].item() == 0
    # Corresponding mask positions should be False (padding)
    assert mask[2, 0].item() is True   # the one real token
    assert mask[2, 1].item() is False  # padding
    assert mask[2, 2].item() is False  # padding

    # Full document "one two three four five" truncated to 4 — all mask positions True
    assert mask[1].all().item() is True

    # "hello world" — 2 real tokens, 2 padding
    assert mask[0, 0].item() is True
    assert mask[0, 1].item() is True
    assert mask[0, 2].item() is False
    assert mask[0, 3].item() is False


def test_tokenize_corpus_max_tokens_caps_rows() -> None:
    class _FakeTokenizer:
        pad_token_id = 0

        def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
            return [1] * 10  # always 10 tokens

    class _FakeModel:
        tokenizer = _FakeTokenizer()

    docs = [f"doc {i}" for i in range(20)]
    tokens, mask = tokenize_corpus(_FakeModel(), docs, seq_len=4, max_tokens=16)

    # 16 // 4 = 4 rows maximum
    assert tokens.shape[0] == 4
    assert mask.shape == tokens.shape


def test_tokenize_corpus_requires_nonempty_docs() -> None:
    import pytest

    class _FakeTokenizer:
        pad_token_id = 0

        def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
            return []

    class _FakeModel:
        tokenizer = _FakeTokenizer()

    with pytest.raises(ValueError, match="empty token sequences"):
        tokenize_corpus(_FakeModel(), ["   "], seq_len=4)


def test_tokenize_corpus_mask_filters_padding() -> None:
    """Mask True-count equals the number of real (non-padded) tokens per document."""
    import torch

    class _FakeTokenizer:
        pad_token_id = 0

        def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
            return list(range(1, len(text.split()) + 1))

    class _FakeModel:
        tokenizer = _FakeTokenizer()

    docs = ["a b c", "x y", "one two three four five six"]  # 3, 2, 6 tokens; seq_len=4
    tokens, mask = tokenize_corpus(_FakeModel(), docs, seq_len=4)

    # doc 0: 3 real tokens, 1 pad
    assert mask[0].sum().item() == 3
    # doc 1: 2 real tokens, 2 pads
    assert mask[1].sum().item() == 2
    # doc 2: 6 tokens but seq_len=4 → truncated → all 4 real
    assert mask[2].sum().item() == 4

    # Using the mask to index flat activations (simulating SAE use-case)
    flat_mask = mask.reshape(-1)
    dummy_acts = torch.ones(len(docs) * 4, 8)
    real_acts = dummy_acts[flat_mask]
    assert real_acts.shape[0] == 3 + 2 + 4  # 9 real tokens
