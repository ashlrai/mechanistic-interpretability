"""Tests for `mech compare-runs --left N --right M` CLI command."""
from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from mech_interp.cli import app
from mech_interp.config.loader import AppConfig, ProjectConfig
from mech_interp.storage import SQLiteResultStore
from mech_interp.types import ExperimentResult, ExperimentSpec, RunStatus

runner = CliRunner()


def _make_store(tmp_path: Path) -> tuple[SQLiteResultStore, AppConfig]:
    config = AppConfig(
        project=ProjectConfig(
            artifact_dir=tmp_path / "artifacts",
            database_path=tmp_path / "runs.sqlite3",
        )
    )
    store = SQLiteResultStore(config.project.database_path, config.project.artifact_dir)
    return store, config


def test_cli_compare_runs_prints_run_ids_and_differing_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """compare-runs prints both run IDs and highlights the differing metric."""
    store, config = _make_store(tmp_path)

    run_a = store.create_run(
        ExperimentSpec(
            name="sae_run_a",
            family="polysemanticity_sae",
            backend="transformerlens",
            parameters={"n_features": 64},
        )
    )
    run_b = store.create_run(
        ExperimentSpec(
            name="sae_run_b",
            family="polysemanticity_sae",
            backend="transformerlens",
            parameters={"n_features": 128},
        )
    )
    store.save_result(
        ExperimentResult(
            run_id=run_a.id,
            status=RunStatus.SUCCEEDED,
            metrics={"mse": 0.8, "live_fraction": 0.6},
            artifacts={},
        )
    )
    store.save_result(
        ExperimentResult(
            run_id=run_b.id,
            status=RunStatus.SUCCEEDED,
            metrics={"mse": 0.4, "live_fraction": 0.9},
            artifacts={},
        )
    )

    # Patch load_config so the CLI uses the tmp_path store
    import mech_interp.cli as cli_module

    monkeypatch.setattr(cli_module, "load_config", lambda: config)

    result = runner.invoke(app, ["compare-runs", "--left", str(run_a.id), "--right", str(run_b.id)])

    assert result.exit_code == 0, result.output
    # Both run IDs present
    assert str(run_a.id) in result.output
    assert str(run_b.id) in result.output
    # Differing metric key present
    assert "mse" in result.output
    assert "live_fraction" in result.output


def test_cli_compare_runs_rejects_family_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """compare-runs exits with code 1 and a clear message on family mismatch."""
    store, config = _make_store(tmp_path)

    run_cp = store.create_run(
        ExperimentSpec(name="cp_run", family="circuit_patching", backend="transformerlens")
    )
    run_sae = store.create_run(
        ExperimentSpec(name="sae_run", family="polysemanticity_sae", backend="transformerlens")
    )

    import mech_interp.cli as cli_module

    monkeypatch.setattr(cli_module, "load_config", lambda: config)

    result = runner.invoke(
        app, ["compare-runs", "--left", str(run_cp.id), "--right", str(run_sae.id)]
    )

    assert result.exit_code == 1
    out = result.output.lower()
    assert "different families" in out or "matching families" in out


def test_cli_compare_runs_missing_run_exits_nonzero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """compare-runs exits with code 1 when a run ID doesn't exist."""
    store, config = _make_store(tmp_path)

    run_a = store.create_run(
        ExperimentSpec(name="sae_run_a", family="polysemanticity_sae", backend="transformerlens")
    )

    import mech_interp.cli as cli_module

    monkeypatch.setattr(cli_module, "load_config", lambda: config)

    result = runner.invoke(
        app, ["compare-runs", "--left", str(run_a.id), "--right", "9999"]
    )

    assert result.exit_code == 1
    assert "9999" in result.output
