"""Unit tests for `mech sweep` and `mech sweep-report` CLI commands."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml
from typer.testing import CliRunner

from mech_interp import cli
from mech_interp.config.loader import AppConfig, ProjectConfig
from mech_interp.experiments.registry import load_experiment_specs_from_file
from mech_interp.types import ExperimentResult, RunStatus

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_base_spec(path: Path) -> Path:
    spec = {
        "name": "polysemanticity-sae-smoke",
        "family": "polysemanticity_sae",
        "backend": "transformerlens",
        "description": "test base spec",
        "parameters": {
            "model": "gpt2-small",
            "hook_site": "blocks.0.hook_resid_pre",
            "n_features": 128,
            "k": 8,
            "epochs": 1,
            "batch_size": 64,
            "learning_rate": 0.001,
            "seed": 42,
            "device": "cpu",
            "corpus_path": "data/prompts/openwebtext_sample.jsonl",
            "seq_len": 64,
            "max_tokens": 500,
        },
    }
    path.write_text(yaml.dump(spec), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# mech sweep --dry-run (no execute)
# ---------------------------------------------------------------------------


class TestSweepDryRun:
    def test_writes_matrix_yaml(self, tmp_path: Path) -> None:
        base = _write_base_spec(tmp_path / "base.yaml")
        output = tmp_path / "sweep.yaml"

        result = CliRunner().invoke(
            cli.app,
            [
                "sweep",
                "--base", str(base),
                "--axis", "parameters.n_features=64,128,256",
                "--output", str(output),
            ],
        )

        assert result.exit_code == 0, result.output
        assert output.exists(), "output YAML was not written"

    def test_output_yaml_has_matrix_block(self, tmp_path: Path) -> None:
        base = _write_base_spec(tmp_path / "base.yaml")
        output = tmp_path / "sweep.yaml"

        CliRunner().invoke(
            cli.app,
            [
                "sweep",
                "--base", str(base),
                "--axis", "parameters.n_features=64,128,256",
                "--output", str(output),
            ],
        )

        data = yaml.safe_load(output.read_text(encoding="utf-8"))
        assert "matrix" in data
        assert "parameters.n_features" in data["matrix"]
        assert data["matrix"]["parameters.n_features"] == [64, 128, 256]

    def test_output_yaml_round_trips_through_registry(self, tmp_path: Path) -> None:
        base = _write_base_spec(tmp_path / "base.yaml")
        output = tmp_path / "sweep.yaml"

        CliRunner().invoke(
            cli.app,
            [
                "sweep",
                "--base", str(base),
                "--axis", "parameters.n_features=64,128,256",
                "--output", str(output),
            ],
        )

        specs = load_experiment_specs_from_file(output)
        assert len(specs) == 3
        feature_counts = sorted(
            s.parameters["n_features"] for s in specs
        )
        assert feature_counts == [64, 128, 256]

    def test_two_axes_cartesian_product(self, tmp_path: Path) -> None:
        base = _write_base_spec(tmp_path / "base.yaml")
        output = tmp_path / "sweep.yaml"

        CliRunner().invoke(
            cli.app,
            [
                "sweep",
                "--base", str(base),
                "--axis", "parameters.n_features=64,128",
                "--axis", "parameters.k=4,8",
                "--output", str(output),
            ],
        )

        specs = load_experiment_specs_from_file(output)
        assert len(specs) == 4  # 2 x 2

    def test_reports_spec_count_in_output(self, tmp_path: Path) -> None:
        base = _write_base_spec(tmp_path / "base.yaml")
        output = tmp_path / "sweep.yaml"

        result = CliRunner().invoke(
            cli.app,
            [
                "sweep",
                "--base", str(base),
                "--axis", "parameters.n_features=64,128,256",
                "--output", str(output),
            ],
        )

        assert "3" in result.output

    def test_numeric_values_parsed_as_numbers(self, tmp_path: Path) -> None:
        """Integer-valued axes must be emitted as ints, not strings."""
        base = _write_base_spec(tmp_path / "base.yaml")
        output = tmp_path / "sweep.yaml"

        CliRunner().invoke(
            cli.app,
            [
                "sweep",
                "--base", str(base),
                "--axis", "parameters.n_features=64,128",
                "--output", str(output),
            ],
        )

        data = yaml.safe_load(output.read_text(encoding="utf-8"))
        values = data["matrix"]["parameters.n_features"]
        assert all(isinstance(v, int) for v in values), f"expected ints, got {values}"

    def test_no_axes_exits_nonzero(self, tmp_path: Path) -> None:
        base = _write_base_spec(tmp_path / "base.yaml")
        output = tmp_path / "sweep.yaml"

        result = CliRunner().invoke(
            cli.app,
            [
                "sweep",
                "--base", str(base),
                "--output", str(output),
            ],
        )

        assert result.exit_code != 0

    def test_missing_base_exits_nonzero(self, tmp_path: Path) -> None:
        result = CliRunner().invoke(
            cli.app,
            [
                "sweep",
                "--base", str(tmp_path / "nonexistent.yaml"),
                "--axis", "parameters.n_features=64",
                "--output", str(tmp_path / "out.yaml"),
            ],
        )
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# mech sweep --execute (mocked runner)
# ---------------------------------------------------------------------------


class TestSweepExecute:
    def test_execute_calls_runner_for_each_spec(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        base = _write_base_spec(tmp_path / "base.yaml")
        output = tmp_path / "sweep.yaml"
        config = AppConfig(
            project=ProjectConfig(
                artifact_dir=tmp_path / "artifacts",
                database_path=tmp_path / "runs.sqlite3",
            )
        )
        monkeypatch.setattr(cli, "load_config", lambda: config)

        mock_result = ExperimentResult(run_id=1, status=RunStatus.SUCCEEDED, metrics={"loss": 0.3})
        mock_runner = MagicMock()
        mock_runner.run.return_value = mock_result

        with patch("mech_interp.cli.ExperimentRunner", return_value=mock_runner):
            result = CliRunner().invoke(
                cli.app,
                [
                    "sweep",
                    "--base", str(base),
                    "--axis", "parameters.n_features=64,128",
                    "--output", str(output),
                    "--execute",
                ],
            )

        assert result.exit_code == 0, result.output
        assert mock_runner.run.call_count == 2

    def test_execute_shows_summary_table(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        base = _write_base_spec(tmp_path / "base.yaml")
        output = tmp_path / "sweep.yaml"
        config = AppConfig(
            project=ProjectConfig(
                artifact_dir=tmp_path / "artifacts",
                database_path=tmp_path / "runs.sqlite3",
            )
        )
        monkeypatch.setattr(cli, "load_config", lambda: config)

        mock_result = ExperimentResult(run_id=1, status=RunStatus.SUCCEEDED, metrics={})
        mock_runner = MagicMock()
        mock_runner.run.return_value = mock_result

        with patch("mech_interp.cli.ExperimentRunner", return_value=mock_runner):
            result = CliRunner().invoke(
                cli.app,
                [
                    "sweep",
                    "--base", str(base),
                    "--axis", "parameters.n_features=64,128",
                    "--output", str(output),
                    "--execute",
                ],
            )

        # Rich table headings
        assert "status" in result.output.lower() or "Status" in result.output


# ---------------------------------------------------------------------------
# mech sweep-report
# ---------------------------------------------------------------------------


class TestSweepReportCommand:
    def _make_run_artifacts(
        self, run_dir: Path, axis_values: dict[str, object], metrics: dict[str, object]
    ) -> None:
        run_dir.mkdir(parents=True, exist_ok=True)
        spec = {
            "name": f"sweep-test-{run_dir.name}",
            "family": "polysemanticity_sae",
            "backend": "transformerlens",
            "parameters": {
                "matrix": "sweep-test",
                "matrix_axes": axis_values,
                **{k: v for k, v in axis_values.items()},
            },
        }
        result_data = {
            "run_id": 1,
            "status": "succeeded",
            "metrics": metrics,
            "artifacts": {},
            "notes": "",
        }
        (run_dir / "spec.json").write_text(json.dumps(spec), encoding="utf-8")
        (run_dir / "result.json").write_text(json.dumps(result_data), encoding="utf-8")

    def test_sweep_report_writes_files(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        artifact_dir = tmp_path / "artifacts"
        db_path = tmp_path / "runs.sqlite3"
        config = AppConfig(
            project=ProjectConfig(artifact_dir=artifact_dir, database_path=db_path)
        )
        monkeypatch.setattr(cli, "load_config", lambda: config)

        # Write two fake run dirs that look like a sweep
        self._make_run_artifacts(
            artifact_dir / "run-000001", {"n_features": 64}, {"loss": 0.5}
        )
        self._make_run_artifacts(
            artifact_dir / "run-000002", {"n_features": 128}, {"loss": 0.4}
        )

        output_dir = tmp_path / "reports"
        result = CliRunner().invoke(
            cli.app,
            [
                "sweep-report",
                "--output-dir", str(output_dir),
                "--prefix", "sweep-test",
            ],
        )

        assert result.exit_code == 0, result.output
        assert (output_dir / "sweep_report.json").exists()
        assert (output_dir / "sweep_report.md").exists()
