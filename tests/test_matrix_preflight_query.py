from __future__ import annotations

import json
from pathlib import Path

import pytest

from mech_interp.experiments import load_experiment_specs
from mech_interp.experiments.registry import ExperimentSpecValidationError
from mech_interp.orchestration.iteration import IterationCaps, run_bounded_iteration
from mech_interp.orchestration.preflight import (
    inspect_dataset,
    preflight_spec,
    validate_answer_tokens,
)
from mech_interp.storage import SQLiteResultStore
from mech_interp.types import ExperimentResult, ExperimentSpec, RunStatus


def test_matrix_spec_expands_deterministically(tmp_path: Path) -> None:
    spec_dir = tmp_path / "experiments"
    spec_dir.mkdir()
    (spec_dir / "matrix.yaml").write_text(
        """
name: matrix-demo
family: circuit_patching
backend: transformerlens
matrix:
  model: [gpt2-small, tiny]
  parameters.layer: [0, 1]
parameters:
  hook_sites: ["blocks.0.hook_resid_post"]
""",
        encoding="utf-8",
    )

    first = load_experiment_specs(spec_dir).list()
    second = load_experiment_specs(spec_dir).list()

    assert len(first) == 4
    assert [spec.name for spec in first] == [spec.name for spec in second]
    assert {spec.parameters["layer"] for spec in first} == {0, 1}
    assert all(len(spec.parameters["generated_spec_hash"]) == 64 for spec in first)
    assert all("matrix_axes" in spec.parameters for spec in first)


def test_matrix_spec_rejects_duplicate_axis_values(tmp_path: Path) -> None:
    spec_dir = tmp_path / "experiments"
    spec_dir.mkdir()
    (spec_dir / "matrix.yaml").write_text(
        """
name: matrix-demo
family: circuit_patching
backend: transformerlens
matrix:
  parameters.layer: [0, 0]
parameters:
  hook_sites: ["blocks.0.hook_resid_post"]
""",
        encoding="utf-8",
    )

    with pytest.raises(ExperimentSpecValidationError, match="duplicate value"):
        load_experiment_specs(spec_dir)


def test_preflight_reports_bad_prompt_pair() -> None:
    report = preflight_spec(
        ExperimentSpec(
            name="bad",
            family="circuit_patching",
            backend="transformerlens",
            parameters={"prompt_pairs": [{"id": "missing-clean"}]},
        )
    )

    assert not report.ok
    assert any(check.name == "prompt_pairs" and check.status == "error" for check in report.checks)


def test_dataset_inspect_hashes_jsonl(tmp_path: Path) -> None:
    dataset = tmp_path / "pairs.jsonl"
    dataset.write_text('{"clean_prompt":"a","correct_token":" yes"}\n', encoding="utf-8")

    inspected = inspect_dataset(dataset)

    assert inspected["rows"] == 1
    assert inspected["sha256"]
    assert inspected["raw_sha256"]
    assert "clean_prompt" in inspected["fields"]


def test_preflight_accepts_normalized_dataset_hash_and_warns_token_issue(tmp_path: Path) -> None:
    dataset = tmp_path / "pairs.jsonl"
    dataset.write_text(
        '{"id":"clean","prompt":"A clean prompt",'
        '"metadata":{"kind":"clean","pair_id":"p","answer":" two tokens"}}\n'
        '{"id":"corrupt","prompt":"A corrupt prompt",'
        '"metadata":{"kind":"corrupted","pair_id":"p","answer":" no"}}\n',
        encoding="utf-8",
    )
    inspected = inspect_dataset(dataset)
    report = preflight_spec(
        ExperimentSpec(
            name="dataset-spec",
            family="circuit_patching",
            backend="transformerlens",
            parameters={
                "dataset_path": str(dataset),
                "dataset_sha256": inspected["sha256"],
                "hook_sites": ["blocks.0.hook_resid_post"],
            },
        )
    )

    assert report.ok
    assert any(check.name == "dataset_hash" and check.status == "ok" for check in report.checks)
    assert any(
        check.name == "answer_tokens" and check.status == "warning"
        for check in report.checks
    )


def test_validate_answer_tokens_reports_dataset_metadata_tokens(tmp_path: Path) -> None:
    dataset = tmp_path / "pairs.jsonl"
    dataset.write_text(
        '{"id":"clean","prompt":"A clean prompt",'
        '"metadata":{"kind":"clean","pair_id":"p","answer":" two tokens"}}\n',
        encoding="utf-8",
    )

    result = validate_answer_tokens(dataset, "gpt2-small")

    assert not result["valid"]
    assert result["invalid_tokens"] == [
        {
            "line": "1",
            "field": "metadata.answer",
            "token": " two tokens",
            "reason": "multiple_whitespace_tokens",
        }
    ]


def test_query_runs_filters_metric_threshold(tmp_path: Path) -> None:
    store = SQLiteResultStore(tmp_path / "runs.sqlite3", tmp_path / "artifacts")
    spec = ExperimentSpec(
        name="run-a",
        family="circuit_patching",
        backend="transformerlens",
        parameters={"tags": ["interesting"], "hook_sites": ["blocks.0.hook_resid_post"]},
    )
    run = store.create_run(spec)
    store.save_result(
        ExperimentResult(
            run_id=run.id,
            status=RunStatus.SUCCEEDED,
            metrics={"recovery": 0.8},
        )
    )

    rows = store.query_runs(
        family="circuit_patching",
        tag="interesting",
        metric="recovery",
        metric_min=0.5,
    )

    assert [row["run_id"] for row in rows] == [run.id]


def test_bounded_iteration_blocks_duplicates_retry_cap_and_tensor_retention(tmp_path: Path) -> None:
    store = SQLiteResultStore(tmp_path / "runs.sqlite3", tmp_path / "artifacts")
    existing = ExperimentSpec(
        name="already-failed",
        family="circuit_patching",
        backend="transformerlens",
        parameters={"max_retries": 1},
    )
    store.enqueue_experiment_specs([existing])
    item = store.claim_next_queue_item()
    assert item is not None and item.lease_token is not None
    store.mark_queue_item_failed_by_lease(item.id, item.lease_token, "failed once")

    specs = [
        ExperimentSpec(
            name="retain-tensors",
            family="circuit_patching",
            backend="transformerlens",
            parameters={"artifact_policy": {"retain_activation_tensors": True}},
        ),
        existing,
        ExperimentSpec(name="queued", family="circuit_patching", backend="transformerlens"),
        ExperimentSpec(name="queued", family="circuit_patching", backend="transformerlens"),
    ]

    result = run_bounded_iteration(
        store,
        tmp_path / "iteration",
        specs,
        IterationCaps(max_generated_specs=4, max_queued_per_iteration=3, max_failed_retry_count=1),
    )
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))

    assert result.generated == 4
    assert result.queued == 1
    reasons = {
        proposal["spec_name"]: proposal["rejection_reasons"]
        for proposal in manifest["proposals"]
    }
    assert "tensor_retention_blocked" in reasons["retain-tensors"]
    assert "failed_retry_cap" in reasons["already-failed"]
    queued_rejections = [
        proposal["rejection_reasons"]
        for proposal in manifest["proposals"]
        if proposal["spec_name"] == "queued"
    ]
    assert queued_rejections == [[], ["duplicate_candidate"]]
