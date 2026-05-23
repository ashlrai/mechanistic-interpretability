from __future__ import annotations

import json
from pathlib import Path

import pytest

from mech_interp.config.loader import AppConfig, ProjectConfig
from mech_interp.experiments.registry import load_experiment_spec
from mech_interp.orchestration.proposals import propose_followups


def test_propose_followups_writes_valid_specs_and_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reports = tmp_path / "artifacts" / "reports"
    reports.mkdir(parents=True)
    (reports / "latest_summary.json").write_text(
        json.dumps(
            {
                "top_circuit_patching_sites": [
                    {
                        "run_id": 7,
                        "spec_name": "source",
                        "hook_site": "blocks.0.hook_resid_pre",
                        "recovery_fraction": 0.9,
                    }
                ],
                "failed_runs": [],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "mech_interp.orchestration.proposals.load_config",
        lambda: AppConfig(project=ProjectConfig(artifact_dir=tmp_path / "artifacts")),
    )

    result = propose_followups("circuit_patching", tmp_path / "proposed", limit=1)

    assert len(result.spec_paths) == 1
    spec = load_experiment_spec(result.spec_paths[0])
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert spec.family == "circuit_patching"
    assert manifest["proposals"][0]["source_run_ids"] == [7]
    assert manifest["guardrail"] == "Generated specs are not executed automatically."
