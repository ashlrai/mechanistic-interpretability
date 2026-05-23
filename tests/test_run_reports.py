import json
from pathlib import Path

from mech_interp.analysis import summarize_recent_runs
from mech_interp.analysis.run_reports import write_aggregate_reports
from mech_interp.storage import ArtifactStore, SQLiteResultStore
from mech_interp.types import ExperimentResult, ExperimentSpec, RunStatus


def test_summarize_recent_runs_counts_statuses_and_families(tmp_path: Path) -> None:
    store = SQLiteResultStore(tmp_path / "runs.sqlite3", tmp_path / "artifacts")
    first = store.create_run(
        ExperimentSpec(name="a", family="polysemanticity", backend="transformerlens")
    )
    second = store.create_run(
        ExperimentSpec(name="b", family="superposition", backend="transformerlens")
    )
    store.save_result(ExperimentResult(run_id=first.id, status=RunStatus.SUCCEEDED))
    store.save_result(ExperimentResult(run_id=second.id, status=RunStatus.FAILED))

    summary = summarize_recent_runs(store)

    assert summary["run_count"] == 2
    assert summary["statuses"] == {"failed": 1, "succeeded": 1}
    assert summary["families"] == {"polysemanticity": 1, "superposition": 1}
    assert summary["backends"] == {"transformerlens": 2}


def test_write_aggregate_reports_includes_patching_and_failures(tmp_path: Path) -> None:
    store = SQLiteResultStore(tmp_path / "runs.sqlite3", tmp_path / "artifacts")
    artifacts = ArtifactStore(tmp_path / "artifacts")
    run = store.create_run(
        ExperimentSpec(name="circuit", family="circuit_patching", backend="transformerlens")
    )
    summary = artifacts.write_json(
        run.id,
        "patching_summary.json",
        {"missing_sites": [], "top_results": []},
    )
    ranked_path = artifacts.run_dir(run.id) / "patching_ranked_results.json"
    ranked_path.write_text(
        json.dumps(
            [
                {
                    "rank": 1,
                    "pair_id": "p",
                    "hook_site": "blocks.0.hook_resid_pre",
                    "clean_logit_diff": 2.0,
                    "corrupted_logit_diff": 0.0,
                    "patched_logit_diff": 1.5,
                    "recovery_fraction": 0.75,
                    "activation_norm": 3.0,
                    "evidence_label": "causal evidence",
                },
                {
                    "rank": 2,
                    "pair_id": "p",
                    "hook_site": "blocks.0.mlp.hook_post",
                    "clean_logit_diff": 2.0,
                    "corrupted_logit_diff": 0.0,
                    "patched_logit_diff": 0.1,
                    "recovery_fraction": 0.05,
                    "activation_norm": 1.0,
                    "evidence_label": "control",
                }
            ]
        ),
        encoding="utf-8",
    )
    store.save_result(
        ExperimentResult(
            run_id=run.id,
            status=RunStatus.SUCCEEDED,
            artifacts={
                "patching_summary": str(summary.path),
                "patching_ranked_json": str(ranked_path),
            },
        )
    )
    failed = store.create_run(
        ExperimentSpec(name="failed", family="circuit_patching", backend="transformerlens")
    )
    store.save_result(ExperimentResult(run_id=failed.id, status=RunStatus.FAILED, notes="boom"))

    reports = write_aggregate_reports(store, tmp_path / "reports")

    note = reports.research_note.read_text(encoding="utf-8")
    payload = json.loads(reports.summary_json.read_text(encoding="utf-8"))
    csv_text = reports.top_sites_csv.read_text(encoding="utf-8")
    assert reports.summary_json.exists()
    assert "blocks.0.hook_resid_pre" in note
    assert "blocks.0.mlp.hook_post" in note
    assert "Run 2 (failed): boom" in note
    assert "evidence_label" in csv_text
    assert "recovery_fraction" in csv_text
    assert payload["top_circuit_patching_sites"][0]["rank"] == 1
    assert payload["top_circuit_patching_sites"][0]["evidence_label"] == "causal evidence"
    assert payload["circuit_patching_control_sites"][0]["evidence_label"] == "control"


def test_write_aggregate_reports_includes_cross_model_probe(tmp_path: Path) -> None:
    store = SQLiteResultStore(tmp_path / "runs.sqlite3", tmp_path / "artifacts")
    artifacts = ArtifactStore(tmp_path / "artifacts")
    run = store.create_run(
        ExperimentSpec(
            name="probe",
            family="cross_model_representation_probe",
            backend="transformerlens",
        )
    )
    summary = artifacts.write_json(
        run.id,
        "cross_model_probe_summary.json",
        {
            "source_model": "source",
            "target_model": "target",
            "source_hook_site": "source.site",
            "target_hook_site": "target.site",
        },
    )
    store.save_result(
        ExperimentResult(
            run_id=run.id,
            status=RunStatus.SUCCEEDED,
            metrics={
                "eval_mean_cosine_similarity": 0.82,
                "eval_variance_explained": 0.51,
            },
            artifacts={"cross_model_probe_summary": str(summary.path)},
        )
    )

    reports = write_aggregate_reports(store, tmp_path / "reports")

    note = reports.research_note.read_text(encoding="utf-8")
    payload = json.loads(reports.summary_json.read_text(encoding="utf-8"))
    assert "Cross-Model Representation Probes" in note
    assert "source -> target" in note
    assert "correlational alignment" in note
    assert payload["cross_model_representation_probes"][0]["eval_variance_explained"] == 0.51
