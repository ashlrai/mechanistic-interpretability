from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from mech_interp.experiments.circuit_patching import CircuitPatchingExperiment
from mech_interp.orchestration import ExperimentRunner
from mech_interp.storage import ArtifactStore, SQLiteResultStore
from mech_interp.types import (
    ActivationPatchRequest,
    ActivationPatchSiteResult,
    CrossModelProbeRequest,
    CrossModelProbeResult,
    ExperimentResult,
    ExperimentRun,
    ExperimentSpec,
    RunStatus,
    utc_now,
)


class FakeCircuitBackend:
    name = "transformerlens"

    def __init__(self, missing_second_site: bool = False) -> None:
        self.missing_second_site = missing_second_site
        self.request: ActivationPatchRequest | None = None

    def load(self) -> None:
        raise AssertionError("Circuit patching should use run_activation_patching directly.")

    def capture_activations(self, prompts: list[str], sites: list[str]) -> dict[str, Any]:
        raise NotImplementedError

    def run_intervention(self, prompt: str, interventions: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    def run_activation_patching(
        self,
        request: ActivationPatchRequest,
    ) -> list[ActivationPatchSiteResult]:
        self.request = request
        results = [
            ActivationPatchSiteResult(
                pair_id=request.prompt_pairs[0].id,
                hook_site=request.hook_sites[0],
                clean_logit_diff=4.0,
                corrupted_logit_diff=-1.0,
                patched_logit_diff=2.0,
                recovery_fraction=0.6,
                activation_norm=12.0,
            )
        ]
        if not self.missing_second_site and len(request.hook_sites) > 1:
            results.append(
                ActivationPatchSiteResult(
                    pair_id=request.prompt_pairs[0].id,
                    hook_site=request.hook_sites[1],
                    clean_logit_diff=4.0,
                    corrupted_logit_diff=-1.0,
                    patched_logit_diff=0.0,
                    recovery_fraction=0.2,
                    activation_norm=8.0,
                )
            )
        return results

    def run_cross_model_probe(
        self,
        request: CrossModelProbeRequest,
    ) -> list[CrossModelProbeResult]:
        raise NotImplementedError


def test_circuit_patching_experiment_writes_ranked_artifacts(tmp_path: Path) -> None:
    backend = FakeCircuitBackend(missing_second_site=True)
    spec = ExperimentSpec(
        name="circuit",
        family="circuit_patching",
        backend="transformerlens",
        parameters={
            "model": "gpt2-small",
            "source_prompt": "The Eiffel Tower is in Paris",
            "target_prompt": "The Eiffel Tower is in Rome",
            "answer_tokens": {"correct": " Paris", "incorrect": " Rome"},
            "layers": [0],
            "patch_sites": ["resid_pre", "mlp_post"],
            "sequence_length": 8,
        },
    )
    run = ExperimentRun(
        id=1,
        spec_name=spec.name,
        family=spec.family,
        backend=spec.backend,
        status=RunStatus.RUNNING,
        artifact_dir=tmp_path,
        created_at=utc_now(),
    )

    result = CircuitPatchingExperiment(backend=backend).run(spec, run)

    assert result.status == RunStatus.SUCCEEDED
    assert result.metrics["prompt_pair_count"] == 1.0
    assert result.metrics["requested_site_count"] == 2.0
    assert result.metrics["patch_result_count"] == 1.0
    assert result.metrics["missing_site_count"] == 1.0
    assert result.metrics["top_recovery_fraction"] == 0.6
    assert backend.request is not None
    assert backend.request.hook_sites == (
        "blocks.0.hook_resid_pre",
        "blocks.0.mlp.hook_post",
    )

    summary = json.loads(Path(result.artifacts["patching_summary"]).read_text(encoding="utf-8"))
    ranked = json.loads(Path(result.artifacts["patching_ranked_json"]).read_text(encoding="utf-8"))
    report = Path(result.artifacts["research_note"]).read_text(encoding="utf-8")

    assert summary["missing_sites"] == [
        {"hook_site": "blocks.0.mlp.hook_post", "pair_id": "pair-0001"}
    ]
    assert ranked[0]["hook_site"] == "blocks.0.hook_resid_pre"
    assert ranked[0]["rank"] == 1
    assert ranked[0]["evidence_label"] == "causal evidence"
    assert "| 1 | pair-0001 | `blocks.0.hook_resid_pre` | 0.6000 | 2.0000 |" in report
    assert "No circuit patch controls were configured" in report


def test_circuit_patching_labels_configured_control_sites(tmp_path: Path) -> None:
    backend = FakeCircuitBackend()
    spec = ExperimentSpec(
        name="circuit-with-controls",
        family="circuit_patching",
        backend="transformerlens",
        parameters={
            "model": "gpt2-small",
            "source_prompt": "The Eiffel Tower is in Paris",
            "target_prompt": "The Eiffel Tower is in Rome",
            "answer_tokens": {"correct": " Paris", "incorrect": " Rome"},
            "layers": [0],
            "patch_sites": ["resid_pre"],
            "control_patch_sites": ["mlp_post"],
            "sequence_length": 8,
        },
    )
    run = ExperimentRun(
        id=1,
        spec_name=spec.name,
        family=spec.family,
        backend=spec.backend,
        status=RunStatus.RUNNING,
        artifact_dir=tmp_path,
        created_at=utc_now(),
    )

    result = CircuitPatchingExperiment(backend=backend).run(spec, run)

    assert backend.request is not None
    assert backend.request.hook_sites == (
        "blocks.0.hook_resid_pre",
        "blocks.0.mlp.hook_post",
    )
    assert result.metrics["control_site_count"] == 1.0
    assert result.metrics["control_result_count"] == 1.0
    summary = json.loads(Path(result.artifacts["patching_summary"]).read_text(encoding="utf-8"))
    ranked = json.loads(Path(result.artifacts["patching_ranked_json"]).read_text(encoding="utf-8"))
    report = Path(result.artifacts["research_note"]).read_text(encoding="utf-8")
    csv_text = Path(result.artifacts["patching_ranked_csv"]).read_text(encoding="utf-8")

    assert summary["control_hook_sites"] == ["blocks.0.mlp.hook_post"]
    assert summary["control_summary"]["evidence_label"] == "control"
    assert summary["control_summary"]["result_count"] == 1
    assert [row["evidence_label"] for row in ranked] == ["causal evidence", "control"]
    assert "rank,pair_id,hook_site" in csv_text
    assert "## Controls" in report
    assert "| 2 | pair-0001 | `blocks.0.mlp.hook_post` | 0.2000 | 0.0000 |" in report


def test_circuit_patching_loads_dataset_pairs_and_checks_hash(tmp_path: Path) -> None:
    dataset_path = tmp_path / "pairs.jsonl"
    dataset_path.write_text(
        '{"id":"clean","prompt":"A clean prompt",'
        '"metadata":{"kind":"clean","pair_id":"p","answer":" yes"}}\n'
        '{"id":"corrupt","prompt":"A corrupt prompt",'
        '"metadata":{"kind":"corrupted","pair_id":"p","answer":" no"}}\n',
        encoding="utf-8",
    )
    backend = FakeCircuitBackend()
    spec = ExperimentSpec(
        name="dataset-circuit",
        family="circuit_patching",
        backend="transformerlens",
        parameters={
            "dataset_path": str(dataset_path),
            "hook_sites": ["blocks.0.hook_resid_pre"],
            "sequence_length": 4,
        },
    )
    run = ExperimentRun(
        id=1,
        spec_name=spec.name,
        family=spec.family,
        backend=spec.backend,
        status=RunStatus.RUNNING,
        artifact_dir=tmp_path,
        created_at=utc_now(),
    )

    result = CircuitPatchingExperiment(backend=backend).run(spec, run)

    assert result.status == RunStatus.SUCCEEDED
    assert backend.request is not None
    assert backend.request.prompt_pairs[0].id == "p"
    assert backend.request.prompt_pairs[0].correct_token == " yes"
    assert backend.request.prompt_pairs[0].incorrect_token == " no"
    summary = json.loads(Path(result.artifacts["patching_summary"]).read_text(encoding="utf-8"))
    assert summary["dataset"]["path"] == str(dataset_path)
    assert len(summary["dataset"]["sha256"]) == 64


def test_circuit_patching_rejects_dataset_hash_mismatch(tmp_path: Path) -> None:
    dataset_path = tmp_path / "pairs.jsonl"
    dataset_path.write_text(
        '{"id":"clean","prompt":"A clean prompt",'
        '"metadata":{"kind":"clean","pair_id":"p","answer":" yes"}}\n'
        '{"id":"corrupt","prompt":"A corrupt prompt",'
        '"metadata":{"kind":"corrupted","pair_id":"p","answer":" no"}}\n',
        encoding="utf-8",
    )
    spec = ExperimentSpec(
        name="dataset-circuit",
        family="circuit_patching",
        backend="transformerlens",
        parameters={
            "dataset_path": str(dataset_path),
            "dataset_sha256": "wrong",
            "hook_sites": ["blocks.0.hook_resid_pre"],
        },
    )
    run = ExperimentRun(
        id=1,
        spec_name=spec.name,
        family=spec.family,
        backend=spec.backend,
        status=RunStatus.RUNNING,
        artifact_dir=tmp_path,
        created_at=utc_now(),
    )

    with pytest.raises(ValueError, match="dataset hash mismatch"):
        CircuitPatchingExperiment(backend=FakeCircuitBackend()).run(spec, run)


def test_runner_dispatches_circuit_patching_family(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeCircuitPatchingExperiment:
        def run(self, spec: ExperimentSpec, run: ExperimentRun) -> ExperimentResult:
            return CircuitPatchingExperiment(backend=FakeCircuitBackend()).run(spec, run)

    monkeypatch.setattr(
        "mech_interp.orchestration.runner.CircuitPatchingExperiment",
        FakeCircuitPatchingExperiment,
    )
    spec = ExperimentSpec(
        name="circuit",
        family="circuit_patching",
        backend="transformerlens",
        parameters={
            "source_prompt": "The Eiffel Tower is in Paris",
            "target_prompt": "The Eiffel Tower is in Rome",
            "answer_tokens": {"correct": " Paris", "incorrect": " Rome"},
            "hook_sites": ["blocks.0.hook_resid_pre"],
            "sequence_length": 8,
        },
    )
    runner = ExperimentRunner(
        result_store=SQLiteResultStore(tmp_path / "runs.sqlite3", tmp_path / "artifacts"),
        artifact_store=ArtifactStore(tmp_path / "artifacts"),
    )

    result = runner.run(spec)

    assert result.status == RunStatus.SUCCEEDED
    assert result.metrics["top_recovery_fraction"] == 0.6
    manifest = runner.artifact_store.read_manifest(1)
    manifest_names = {artifact["name"] for artifact in manifest["artifacts"]}
    assert "patching_summary" in manifest_names
