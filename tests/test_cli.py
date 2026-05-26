import csv
import json
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from mech_interp import cli
from mech_interp.config.loader import AppConfig, ProjectConfig
from mech_interp.storage import ArtifactStore, SQLiteResultStore
from mech_interp.types import ExperimentResult, ExperimentSpec, RunStatus


def test_validate_command_accepts_default_experiments() -> None:
    result = CliRunner().invoke(cli.app, ["validate"])

    assert result.exit_code == 0
    assert "Validated" in result.output and "experiment spec" in result.output


def test_validate_command_fails_invalid_specs(tmp_path: Path) -> None:
    spec_path = tmp_path / "bad.yaml"
    spec_path.write_text(
        """
name: bad
family: nope
backend: transformerlens
""",
        encoding="utf-8",
    )

    result = CliRunner().invoke(cli.app, ["validate", "--directory", str(tmp_path)])

    assert result.exit_code == 1
    assert "Invalid experiment specs" in result.output
    assert "bad.yaml" in result.output
    assert "unsupported family 'nope'" in result.output


def test_inspect_run_prints_run_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, run_id = _create_temp_run(tmp_path)
    monkeypatch.setattr(cli, "load_config", lambda: config)

    result = CliRunner().invoke(cli.app, ["inspect-run", str(run_id)])

    assert result.exit_code == 0
    assert '"run_id": 1' in result.output
    assert '"name": "demo"' in result.output
    assert '"accuracy": 0.75' in result.output
    assert '"manifest"' in result.output
    assert '"result.json"' in result.output


def test_export_run_writes_json_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, run_id = _create_temp_run(tmp_path)
    output = tmp_path / "bundle.json"
    monkeypatch.setattr(cli, "load_config", lambda: config)

    result = CliRunner().invoke(cli.app, ["export-run", str(run_id), "--output", str(output)])

    assert result.exit_code == 0
    bundle = json.loads(output.read_text(encoding="utf-8"))
    assert bundle["run_id"] == run_id
    assert bundle["spec"]["name"] == "demo"
    assert bundle["result"]["metrics"] == {"accuracy": 0.75}
    assert bundle["manifest"]["artifacts"][0]["name"] == "result.json"


def test_query_runs_csv_includes_metadata_and_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_dir = tmp_path / "artifacts"
    database_path = tmp_path / "runs.sqlite3"
    config = AppConfig(
        project=ProjectConfig(artifact_dir=artifact_dir, database_path=database_path)
    )
    store = SQLiteResultStore(database_path, artifact_dir)
    run = store.create_run(
        ExperimentSpec(
            name="query-demo",
            family="circuit_patching",
            backend="transformerlens",
            parameters={"tags": ["interesting"], "hypothesis": "test hypothesis"},
        )
    )
    store.save_result(
        ExperimentResult(
            run_id=run.id,
            status=RunStatus.SUCCEEDED,
            metrics={"recovery": 0.75},
        )
    )
    monkeypatch.setattr(cli, "load_config", lambda: config)

    result = CliRunner().invoke(
        cli.app,
        [
            "query-runs",
            "--family",
            "circuit_patching",
            "--output-format",
            "csv",
        ],
    )

    rows = list(csv.DictReader(StringIO(result.output)))
    assert result.exit_code == 0
    assert rows[0]["spec_name"] == "query-demo"
    assert json.loads(rows[0]["tags"]) == ["interesting"]
    assert rows[0]["hypothesis"] == "test hypothesis"
    assert json.loads(rows[0]["metrics"]) == {"recovery": 0.75}


def test_query_runs_rejects_unknown_output_format(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = AppConfig(
        project=ProjectConfig(
            artifact_dir=tmp_path / "artifacts",
            database_path=tmp_path / "runs.sqlite3",
        )
    )
    monkeypatch.setattr(cli, "load_config", lambda: config)

    result = CliRunner().invoke(cli.app, ["query-runs", "--output-format", "xml"])

    assert result.exit_code == 1
    assert "--output-format must be one of" in result.output


def _write_sae_artifacts(run_dir: Path, model: str = "gpt2-small") -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    spec = {
        "name": "seed-sae",
        "family": "polysemanticity_sae",
        "backend": "transformerlens",
        "parameters": {"model": model, "hook_site": "blocks.0.hook_resid_pre"},
    }
    analysis = {
        "features": [
            {
                "feature_index": 0,
                "dead": False,
                "max_activation": 2.5,
                "coherence_score": 0.9,
                "top_prompts": [
                    {"prompt": "The cat sat on the mat"},
                    {"prompt": "A dog lay on the rug"},
                ],
            },
        ],
        "reconstruction_mse": 0.042,
    }
    config = {"n_features": 256, "k": 32}
    (run_dir / "spec.json").write_text(json.dumps(spec), encoding="utf-8")
    (run_dir / "feature_analysis.json").write_text(json.dumps(analysis), encoding="utf-8")
    (run_dir / "sae_weights.safetensors.json").write_text(json.dumps(config), encoding="utf-8")


def test_iterate_from_run_dry_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_dir = tmp_path / "artifacts"
    database_path = tmp_path / "runs.sqlite3"
    config = AppConfig(
        project=ProjectConfig(artifact_dir=artifact_dir, database_path=database_path)
    )
    monkeypatch.setattr(cli, "load_config", lambda: config)

    run_dir = tmp_path / "source-run"
    _write_sae_artifacts(run_dir)
    output_dir = tmp_path / "proposals"

    result = CliRunner().invoke(
        cli.app,
        [
            "iterate-from-run",
            "--family", "polysemanticity_sae",
            "--artifact-dir", str(run_dir),
            "--output", str(output_dir),
            "--limit", "1",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0
    assert "dry-run" in result.output
    assert "1 proposal(s)" in result.output


def test_iterate_from_run_unknown_family_exits_1(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_dir = tmp_path / "artifacts"
    database_path = tmp_path / "runs.sqlite3"
    config = AppConfig(
        project=ProjectConfig(artifact_dir=artifact_dir, database_path=database_path)
    )
    monkeypatch.setattr(cli, "load_config", lambda: config)

    result = CliRunner().invoke(
        cli.app,
        [
            "iterate-from-run",
            "--family", "circuit_patching",
            "--artifact-dir", str(tmp_path),
            "--output", str(tmp_path / "proposals"),
        ],
    )

    assert result.exit_code == 1
    assert "No per-run proposal generator" in result.output


def test_iterate_from_run_executes_with_mocked_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_dir = tmp_path / "artifacts"
    database_path = tmp_path / "runs.sqlite3"
    config = AppConfig(
        project=ProjectConfig(artifact_dir=artifact_dir, database_path=database_path)
    )
    monkeypatch.setattr(cli, "load_config", lambda: config)

    run_dir = tmp_path / "source-run"
    _write_sae_artifacts(run_dir)
    output_dir = tmp_path / "proposals"

    mock_result = ExperimentResult(run_id=99, status=RunStatus.SUCCEEDED)
    mock_runner = MagicMock()
    mock_runner.run.return_value = mock_result

    with patch("mech_interp.cli.ExperimentRunner", return_value=mock_runner):
        result = CliRunner().invoke(
            cli.app,
            [
                "iterate-from-run",
                "--family", "polysemanticity_sae",
                "--artifact-dir", str(run_dir),
                "--output", str(output_dir),
                "--limit", "1",
            ],
        )

    assert result.exit_code == 0
    assert "succeeded" in result.output or "1 run(s)" in result.output


def _create_temp_run(tmp_path: Path) -> tuple[AppConfig, int]:
    artifact_dir = tmp_path / "artifacts"
    database_path = tmp_path / "runs.sqlite3"
    config = AppConfig(
        project=ProjectConfig(
            artifact_dir=artifact_dir,
            database_path=database_path,
        )
    )
    store = SQLiteResultStore(
        database_path,
        artifact_dir,
        resolved_config={"project": {"artifact_dir": str(artifact_dir)}},
    )
    run = store.create_run(
        ExperimentSpec(
            name="demo",
            family="polysemanticity",
            backend="transformerlens",
            parameters={"layers": [0]},
        )
    )
    artifact_store = ArtifactStore(artifact_dir)
    result_record = artifact_store.write_json(run.id, "result.json", {"ok": True})
    manifest = artifact_store.write_manifest(run.id, [result_record])
    store.save_result(
        ExperimentResult(
            run_id=run.id,
            status=RunStatus.SUCCEEDED,
            metrics={"accuracy": 0.75},
            artifacts={"manifest": str(manifest.path)},
            notes="completed",
        )
    )
    return config, run.id
