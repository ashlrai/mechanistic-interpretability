"""Extended tests for run_reports.py covering SAE, ACDC, and environment provenance."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mech_interp.analysis.run_reports import (
    INSPECTOR_BY_FAMILY,
    _load_run_artifacts_from_dir,
    environment_provenance,
    inspect_run_family,
    write_aggregate_reports,
)
from mech_interp.storage import ArtifactStore, SQLiteResultStore
from mech_interp.types import ExperimentResult, ExperimentSpec, RunStatus

# ---------------------------------------------------------------------------
# environment_provenance
# ---------------------------------------------------------------------------


def test_environment_provenance_reads_env_json(tmp_path: Path) -> None:
    run_dir = tmp_path / "run-000001"
    run_dir.mkdir()
    env = {
        "seed": 42,
        "python_version": "3.11.0",
        "uv_lock_sha256": "abcdef1234567890",
        "package_versions": {"torch": "2.3.0"},
    }
    (run_dir / "environment.json").write_text(json.dumps(env), encoding="utf-8")

    result = environment_provenance(run_dir)

    assert result is not None
    assert result["torch_version"] == "2.3.0"
    assert result["seed"] == 42
    assert result["uv_lock_sha"] == "abcdef123456"
    assert result["python_version"] == "3.11.0"


def test_environment_provenance_returns_none_when_missing(tmp_path: Path) -> None:
    assert environment_provenance(tmp_path / "nonexistent") is None


# ---------------------------------------------------------------------------
# INSPECTOR_BY_FAMILY registry
# ---------------------------------------------------------------------------


def test_inspector_circuit_patching_top_sites(tmp_path: Path) -> None:
    ranked = [
        {
            "rank": 1,
            "hook_site": "blocks.0.hook_resid_pre",
            "recovery_fraction": 0.8,
            "evidence_label": "causal evidence",
        },
        {
            "rank": 2,
            "hook_site": "blocks.1.hook_resid_pre",
            "recovery_fraction": 0.6,
            "evidence_label": "causal evidence",
        },
    ]
    (tmp_path / "patching_ranked_results.json").write_text(json.dumps(ranked), encoding="utf-8")

    result = INSPECTOR_BY_FAMILY["circuit_patching"](tmp_path, top_n=5)

    assert result["family"] == "circuit_patching"
    assert result["total_ranked"] == 2
    assert result["top_sites"][0]["hook_site"] == "blocks.0.hook_resid_pre"


def test_inspector_polysemanticity_sae_live_features(tmp_path: Path) -> None:
    analysis = {
        "features": [
            {"feature_index": 0, "dead": False, "max_activation": 3.0, "coherence_score": 0.9},
            {"feature_index": 1, "dead": True, "max_activation": 0.0, "coherence_score": 0.0},
            {"feature_index": 2, "dead": False, "max_activation": 1.5, "coherence_score": 0.7},
        ],
        "reconstruction_mse": 0.05,
    }
    (tmp_path / "feature_analysis.json").write_text(json.dumps(analysis), encoding="utf-8")

    result = INSPECTOR_BY_FAMILY["polysemanticity_sae"](tmp_path, top_n=5)

    assert result["family"] == "polysemanticity_sae"
    assert result["live_feature_count"] == 2
    assert result["total_features"] == 3
    assert result["reconstruction_mse"] == 0.05
    assert result["top_features"][0]["max_activation"] == 3.0


def test_inspector_acdc_lite_survivors(tmp_path: Path) -> None:
    circuit = {
        "faithfulness": 0.88,
        "nodes": [
            {"node_id": "a", "pruned": False, "importance": 0.9},
            {"node_id": "b", "pruned": True, "importance": 0.1},
            {"node_id": "c", "pruned": False, "importance": 0.7},
        ],
    }
    edges = [
        {"src": "a", "dst": "c", "weight": 0.5},
    ]
    (tmp_path / "circuit.json").write_text(json.dumps(circuit), encoding="utf-8")
    (tmp_path / "edges.json").write_text(json.dumps(edges), encoding="utf-8")

    result = INSPECTOR_BY_FAMILY["acdc_lite"](tmp_path, top_n=5)

    assert result["family"] == "acdc_lite"
    assert result["survivor_count"] == 2
    assert result["total_nodes"] == 3
    assert result["faithfulness"] == 0.88
    assert result["top_edges"][0]["src"] == "a"


def test_inspect_run_family_warns_on_unknown_family(tmp_path: Path) -> None:
    with pytest.warns(UserWarning, match="No inspector registered for family"):
        result = inspect_run_family("unknown_future_family", tmp_path, top_n=5)
    assert "artifacts" in result


# ---------------------------------------------------------------------------
# _load_run_artifacts_from_dir
# ---------------------------------------------------------------------------


def test_load_run_artifacts_from_dir_picks_up_all_known_files(tmp_path: Path) -> None:
    run_dir = tmp_path / "run-000001"
    run_dir.mkdir()
    (run_dir / "feature_analysis.json").write_text(json.dumps({"features": []}), encoding="utf-8")
    (run_dir / "circuit.json").write_text(json.dumps({"nodes": []}), encoding="utf-8")
    (run_dir / "environment.json").write_text(json.dumps({"seed": 1}), encoding="utf-8")
    (run_dir / "result.json").write_text(json.dumps({"status": "succeeded"}), encoding="utf-8")

    loaded = _load_run_artifacts_from_dir(run_dir)

    assert "feature_analysis" in loaded
    assert "circuit" in loaded
    assert "environment" in loaded
    assert "result" in loaded


def test_load_run_artifacts_from_dir_returns_empty_for_missing_dir(tmp_path: Path) -> None:
    loaded = _load_run_artifacts_from_dir(tmp_path / "nonexistent")
    assert loaded == {}


# ---------------------------------------------------------------------------
# write_aggregate_reports — SAE section
# ---------------------------------------------------------------------------


def _write_sae_run(
    store: SQLiteResultStore,
    artifacts: ArtifactStore,
    *,
    live_features: int = 3,
    dead_features: int = 1,
    mse: float = 0.04,
) -> int:
    run = store.create_run(
        ExperimentSpec(name="sae-test", family="polysemanticity_sae", backend="transformerlens")
    )
    features = [
        {"feature_index": i, "dead": False, "max_activation": float(i), "coherence_score": 0.8}
        for i in range(live_features)
    ] + [
        {
            "feature_index": live_features + j,
            "dead": True,
            "max_activation": 0.0,
            "coherence_score": 0.0,
        }
        for j in range(dead_features)
    ]
    analysis = {"features": features, "reconstruction_mse": mse}
    fa_record = artifacts.write_json(run.id, "feature_analysis.json", analysis)
    store.save_result(
        ExperimentResult(
            run_id=run.id,
            status=RunStatus.SUCCEEDED,
            artifacts={"feature_analysis": str(fa_record.path)},
        )
    )
    return run.id


def _write_acdc_run(
    store: SQLiteResultStore,
    artifacts: ArtifactStore,
    *,
    survivors: int = 2,
    total_nodes: int = 5,
    faithfulness: float = 0.85,
) -> int:
    run = store.create_run(
        ExperimentSpec(name="acdc-test", family="acdc_lite", backend="transformerlens")
    )
    nodes = [
        {
            "node_id": f"n{i}",
            "pruned": i >= survivors,
            "importance": float(survivors - i) / survivors,
        }
        for i in range(total_nodes)
    ]
    circuit = {"faithfulness": faithfulness, "nodes": nodes, "model": "gpt2-small"}
    circuit_record = artifacts.write_json(run.id, "circuit.json", circuit)
    store.save_result(
        ExperimentResult(
            run_id=run.id,
            status=RunStatus.SUCCEEDED,
            artifacts={"circuit": str(circuit_record.path)},
        )
    )
    return run.id


def test_aggregate_report_includes_sae_section(tmp_path: Path) -> None:
    store = SQLiteResultStore(tmp_path / "runs.sqlite3", tmp_path / "artifacts")
    artifacts = ArtifactStore(tmp_path / "artifacts")
    _write_sae_run(store, artifacts, live_features=4, mse=0.037)

    reports = write_aggregate_reports(store, tmp_path / "reports")

    note = reports.research_note.read_text(encoding="utf-8")
    payload = json.loads(reports.summary_json.read_text(encoding="utf-8"))

    assert "SAE Run Summaries" in note
    assert "sae-test" in note
    assert "sae_run_summaries" in payload
    assert payload["sae_run_summaries"][0]["live_feature_count"] == 4
    assert payload["sae_run_summaries"][0]["reconstruction_mse"] == 0.037


def test_aggregate_report_includes_acdc_section(tmp_path: Path) -> None:
    store = SQLiteResultStore(tmp_path / "runs.sqlite3", tmp_path / "artifacts")
    artifacts = ArtifactStore(tmp_path / "artifacts")
    _write_acdc_run(store, artifacts, survivors=3, faithfulness=0.91)

    reports = write_aggregate_reports(store, tmp_path / "reports")

    note = reports.research_note.read_text(encoding="utf-8")
    payload = json.loads(reports.summary_json.read_text(encoding="utf-8"))

    assert "ACDC-Lite Run Summaries" in note
    assert "acdc-test" in note
    assert "acdc_run_summaries" in payload
    assert payload["acdc_run_summaries"][0]["survivor_count"] == 3
    assert payload["acdc_run_summaries"][0]["faithfulness"] == 0.91


def test_aggregate_report_shows_empty_sections_when_no_new_families(tmp_path: Path) -> None:
    """Report renders SAE + ACDC sections even when no such runs exist."""
    store = SQLiteResultStore(tmp_path / "runs.sqlite3", tmp_path / "artifacts")

    reports = write_aggregate_reports(store, tmp_path / "reports")

    note = reports.research_note.read_text(encoding="utf-8")
    assert "SAE Run Summaries" in note
    assert "ACDC-Lite Run Summaries" in note
    assert "No polysemanticity_sae runs" in note
    assert "No acdc_lite runs" in note
