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


def test_curated_prompt_files_load() -> None:
    for path in sorted(Path("data/prompts").glob("*.*")):
        if path.name == "README.md":
            continue
        dataset = load_prompt_dataset(path)

        assert dataset.records
        assert len(dataset.sha256) == 64
