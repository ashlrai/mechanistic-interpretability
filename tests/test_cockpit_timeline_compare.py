"""Tests for the /timeline and /runs/<a>/compare/<b> cockpit pages."""
from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from mech_interp.cockpit import create_app
from mech_interp.config.loader import AppConfig, ProjectConfig
from mech_interp.storage import SQLiteResultStore
from mech_interp.types import ExperimentResult, ExperimentSpec, RunStatus


def _config(tmp_path: Path) -> AppConfig:
    return AppConfig(
        project=ProjectConfig(
            artifact_dir=tmp_path / "artifacts",
            database_path=tmp_path / "runs.sqlite3",
        )
    )


# ---------------------------------------------------------------------------
# Timeline tests
# ---------------------------------------------------------------------------


def test_cockpit_timeline_renders_runs_grouped_by_family(tmp_path: Path) -> None:
    """Three runs across two families all appear by spec name on /timeline."""
    config = _config(tmp_path)
    store = SQLiteResultStore(config.project.database_path, config.project.artifact_dir)
    run1 = store.create_run(
        ExperimentSpec(name="sae_alpha", family="polysemanticity_sae", backend="transformerlens")
    )
    run2 = store.create_run(
        ExperimentSpec(name="sae_beta", family="polysemanticity_sae", backend="transformerlens")
    )
    run3 = store.create_run(
        ExperimentSpec(name="circuit_gamma", family="circuit_patching", backend="transformerlens")
    )
    store.save_result(
        ExperimentResult(run_id=run1.id, status=RunStatus.SUCCEEDED, artifacts={})
    )
    store.save_result(
        ExperimentResult(run_id=run2.id, status=RunStatus.FAILED, artifacts={})
    )
    store.save_result(
        ExperimentResult(run_id=run3.id, status=RunStatus.SUCCEEDED, artifacts={})
    )
    client = TestClient(create_app(config))

    response = client.get("/timeline")

    assert response.status_code == 200
    # Each run's spec name should appear somewhere in the SVG tooltip or legend
    assert "sae_alpha" in response.text
    assert "sae_beta" in response.text
    assert "circuit_gamma" in response.text
    # Both family labels should be in the swimlane section
    assert "polysemanticity_sae" in response.text
    assert "circuit_patching" in response.text
    # SVG should be present
    assert "<svg" in response.text


def test_cockpit_timeline_empty_state_is_friendly(tmp_path: Path) -> None:
    """Empty DB: /timeline must render a friendly empty-state SVG, not crash."""
    config = _config(tmp_path)
    client = TestClient(create_app(config))

    response = client.get("/timeline")

    assert response.status_code == 200
    # Should contain the SVG empty-state message, not a traceback
    assert "No runs" in response.text or "<svg" in response.text
    assert "500" not in response.text


def test_cockpit_timeline_filter_by_family(tmp_path: Path) -> None:
    """Family filter on /timeline keeps only matching runs."""
    config = _config(tmp_path)
    store = SQLiteResultStore(config.project.database_path, config.project.artifact_dir)
    store.create_run(
        ExperimentSpec(name="sae_run", family="polysemanticity_sae", backend="transformerlens")
    )
    store.create_run(
        ExperimentSpec(name="cp_run", family="circuit_patching", backend="transformerlens")
    )
    client = TestClient(create_app(config))

    response = client.get("/timeline?family=polysemanticity_sae")

    assert response.status_code == 200
    assert "sae_run" in response.text
    # cp_run spec name should not appear in SVG tooltip
    assert "cp_run" not in response.text


# ---------------------------------------------------------------------------
# Compare-runs tests
# ---------------------------------------------------------------------------


def test_cockpit_compare_renders_side_by_side(tmp_path: Path) -> None:
    """Two SAE runs with different n_features show both spec names and metric diff."""
    config = _config(tmp_path)
    store = SQLiteResultStore(config.project.database_path, config.project.artifact_dir)
    run_a = store.create_run(
        ExperimentSpec(
            name="sae_small",
            family="polysemanticity_sae",
            backend="transformerlens",
            parameters={"n_features": 64},
        )
    )
    run_b = store.create_run(
        ExperimentSpec(
            name="sae_large",
            family="polysemanticity_sae",
            backend="transformerlens",
            parameters={"n_features": 256},
        )
    )
    store.save_result(
        ExperimentResult(
            run_id=run_a.id,
            status=RunStatus.SUCCEEDED,
            metrics={"mse": 0.5, "live_fraction": 0.8},
            artifacts={},
        )
    )
    store.save_result(
        ExperimentResult(
            run_id=run_b.id,
            status=RunStatus.SUCCEEDED,
            metrics={"mse": 0.3, "live_fraction": 0.9},
            artifacts={},
        )
    )
    client = TestClient(create_app(config))

    response = client.get(f"/runs/{run_a.id}/compare/{run_b.id}")

    assert response.status_code == 200
    # Both spec names in header
    assert "sae_small" in response.text
    assert "sae_large" in response.text
    # Metric keys present
    assert "mse" in response.text
    assert "live_fraction" in response.text
    # Highlight badge for >5% diff — mse drops 40%, live_fraction rises 12.5%
    assert "better" in response.text or "worse" in response.text
    # Parameter diff: n_features should appear
    assert "n_features" in response.text


def test_cockpit_compare_400_on_family_mismatch(tmp_path: Path) -> None:
    """Comparing runs from different families returns 400 with a clear message."""
    config = _config(tmp_path)
    store = SQLiteResultStore(config.project.database_path, config.project.artifact_dir)
    run_cp = store.create_run(
        ExperimentSpec(name="cp_run", family="circuit_patching", backend="transformerlens")
    )
    run_sae = store.create_run(
        ExperimentSpec(name="sae_run", family="polysemanticity_sae", backend="transformerlens")
    )
    client = TestClient(create_app(config))

    response = client.get(f"/runs/{run_cp.id}/compare/{run_sae.id}")

    assert response.status_code == 400
    body = response.text.lower()
    assert "different families" in body or "matching families" in body


def test_cockpit_compare_handles_missing_metrics_gracefully(tmp_path: Path) -> None:
    """A run with no metrics renders without crashing; missing values shown as dash."""
    config = _config(tmp_path)
    store = SQLiteResultStore(config.project.database_path, config.project.artifact_dir)
    run_a = store.create_run(
        ExperimentSpec(name="cp_a", family="circuit_patching", backend="transformerlens")
    )
    run_b = store.create_run(
        ExperimentSpec(name="cp_b", family="circuit_patching", backend="transformerlens")
    )
    store.save_result(
        ExperimentResult(
            run_id=run_a.id,
            status=RunStatus.SUCCEEDED,
            metrics={"recovery_fraction": 0.75},
            artifacts={},
        )
    )
    # run_b has no result at all
    client = TestClient(create_app(config))

    response = client.get(f"/runs/{run_a.id}/compare/{run_b.id}")

    assert response.status_code == 200
    assert "recovery_fraction" in response.text
