from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from mech_interp.cockpit import create_app
from mech_interp.config.loader import AppConfig, ProjectConfig
from mech_interp.storage import ArtifactStore, SQLiteResultStore
from mech_interp.types import ExperimentResult, ExperimentSpec, RunStatus


def test_cockpit_dashboard_and_queue_render(tmp_path: Path) -> None:
    config = _config(tmp_path)
    store = SQLiteResultStore(config.project.database_path, config.project.artifact_dir)
    store.enqueue_experiment_specs(
        [ExperimentSpec(name="queued", family="polysemanticity", backend="transformerlens")]
    )
    client = TestClient(create_app(config, experiment_dir=str(tmp_path / "experiments")))

    dashboard = client.get("/")
    queue = client.get("/queue")

    assert dashboard.status_code == 200
    assert "Queue Status" in dashboard.text
    assert "Lab Status" in client.get("/status/cards").text
    assert queue.status_code == 200
    assert "queued" in queue.text
    assert "/queue/requeue/" in queue.text


def test_cockpit_status_endpoint_reports_polling_payload(tmp_path: Path) -> None:
    config = _config(tmp_path)
    store = SQLiteResultStore(config.project.database_path, config.project.artifact_dir)
    store.enqueue_experiment_specs(
        [ExperimentSpec(name="queued", family="polysemanticity", backend="transformerlens")]
    )
    client = TestClient(create_app(config, experiment_dir=str(tmp_path / "experiments")))

    response = client.get("/status")

    assert response.status_code == 200
    payload = response.json()
    assert payload["queue_counts"]["planned"] == 1
    assert "generated_at" in payload
    assert "events" in payload


def test_cockpit_queue_requeue_action_uses_queue_id(tmp_path: Path) -> None:
    config = _config(tmp_path)
    store = SQLiteResultStore(config.project.database_path, config.project.artifact_dir)
    store.enqueue_experiment_specs(
        [ExperimentSpec(name="queued", family="polysemanticity", backend="transformerlens")]
    )
    item = store.list_queue_items()[0]
    store.pause_queue_item(item.id)
    client = TestClient(create_app(config, experiment_dir=str(tmp_path / "experiments")))

    response = client.post(f"/queue/requeue/{item.id}", follow_redirects=False)

    assert response.status_code == 303
    assert store.list_queue_items()[0].status.value == "planned"


def test_cockpit_run_detail_degrades_for_missing_report(tmp_path: Path) -> None:
    config = _config(tmp_path)
    store = SQLiteResultStore(config.project.database_path, config.project.artifact_dir)
    run = store.create_run(
        ExperimentSpec(name="demo", family="polysemanticity", backend="transformerlens")
    )
    artifact_store = ArtifactStore(config.project.artifact_dir)
    manifest = artifact_store.write_manifest(run.id, [])
    store.save_result(
        ExperimentResult(
            run_id=run.id,
            status=RunStatus.SUCCEEDED,
            artifacts={"manifest": str(manifest.path), "research_note": str(tmp_path / "nope.md")},
        )
    )
    client = TestClient(create_app(config, experiment_dir=str(tmp_path / "experiments")))

    response = client.get(f"/runs/{run.id}")

    assert response.status_code == 200
    assert "Report artifact is missing" in response.text
    assert "manifest.json" in response.text or "artifacts" in response.text


def test_cockpit_serves_artifact_links_under_artifact_root(tmp_path: Path) -> None:
    config = _config(tmp_path)
    store = SQLiteResultStore(config.project.database_path, config.project.artifact_dir)
    run = store.create_run(
        ExperimentSpec(name="demo", family="polysemanticity", backend="transformerlens")
    )
    artifact_store = ArtifactStore(config.project.artifact_dir)
    note = artifact_store.write_text(run.id, "research_note.md", "# Note\n")
    store.save_result(
        ExperimentResult(
            run_id=run.id,
            status=RunStatus.SUCCEEDED,
            artifacts={"research_note": str(note.path)},
        )
    )
    client = TestClient(create_app(config, experiment_dir=str(tmp_path / "experiments")))

    detail = client.get(f"/runs/{run.id}")
    artifact = client.get(f"/artifacts/run-{run.id:06d}/research_note.md")
    escaped = client.get("/artifacts/../runs.sqlite3")

    assert "/artifacts/run-000001/research_note.md" in detail.text
    assert artifact.status_code == 200
    assert "# Note" in artifact.text
    assert escaped.status_code == 404


def test_cockpit_reports_get_is_read_only_and_post_generates(tmp_path: Path) -> None:
    config = _config(tmp_path)
    client = TestClient(create_app(config, experiment_dir=str(tmp_path / "experiments")))
    reports_dir = config.project.artifact_dir / "reports"

    get_response = client.get("/reports")

    assert get_response.status_code == 200
    assert "missing" in get_response.text
    assert not reports_dir.exists()

    post_response = client.post("/reports/generate", follow_redirects=False)

    assert post_response.status_code == 303
    assert (reports_dir / "latest_summary.json").exists()


def test_cockpit_artifact_browser_shows_preview_and_metadata(tmp_path: Path) -> None:
    config = _config(tmp_path)
    store = SQLiteResultStore(config.project.database_path, config.project.artifact_dir)
    run = store.create_run(
        ExperimentSpec(name="demo", family="polysemanticity", backend="transformerlens")
    )
    artifact_store = ArtifactStore(config.project.artifact_dir)
    summary = artifact_store.write_json(run.id, "summary.json", {"ok": True})
    manifest = artifact_store.write_manifest(run.id, [summary])
    store.save_result(
        ExperimentResult(
            run_id=run.id,
            status=RunStatus.SUCCEEDED,
            artifacts={"manifest": str(manifest.path), "summary": str(summary.path)},
        )
    )
    client = TestClient(create_app(config, experiment_dir=str(tmp_path / "experiments")))

    response = client.get(f"/artifacts/browser/{run.id}")

    assert response.status_code == 200
    assert "Artifacts for Run 1" in response.text
    assert "summary.json" in response.text
    assert "application/json" in response.text
    assert "ok" in response.text


def test_cockpit_sae_features_404s_for_wrong_family(tmp_path: Path) -> None:
    """Visiting /runs/<id>/features on a non-SAE run must 404 with a clear message,
    not silently render an empty feature browser.
    """
    config = _config(tmp_path)
    store = SQLiteResultStore(config.project.database_path, config.project.artifact_dir)
    run = store.create_run(
        ExperimentSpec(name="cp_run", family="circuit_patching", backend="transformerlens")
    )
    client = TestClient(create_app(config))
    response = client.get(f"/runs/{run.id}/features")
    assert response.status_code == 404
    assert "polysemanticity_sae" in response.text


def test_cockpit_acdc_circuit_404s_for_wrong_family(tmp_path: Path) -> None:
    """Visiting /runs/<id>/circuit on a non-ACDC run must 404."""
    config = _config(tmp_path)
    store = SQLiteResultStore(config.project.database_path, config.project.artifact_dir)
    run = store.create_run(
        ExperimentSpec(name="sae_run", family="polysemanticity_sae", backend="transformerlens")
    )
    client = TestClient(create_app(config))
    response = client.get(f"/runs/{run.id}/circuit")
    assert response.status_code == 404
    assert "acdc" in response.text.lower()


def test_cockpit_sae_features_renders_live_dead_stats(tmp_path: Path) -> None:
    config = _config(tmp_path)
    store = SQLiteResultStore(config.project.database_path, config.project.artifact_dir)
    run = store.create_run(
        ExperimentSpec(name="sae_run", family="polysemanticity_sae", backend="transformerlens")
    )
    artifact_store = ArtifactStore(config.project.artifact_dir)
    feature_data = {
        "mean_features_per_token": 2.5,
        "features": [
            {
                "feature_index": 0,
                "max_activation": 3.14,
                "mean_activation": 1.2,
                "dead": False,
                "coherence_score": 0.87,
                "top_prompts": ["the cat sat", "a dog ran"],
            },
            {
                "feature_index": 1,
                "max_activation": 0.0,
                "mean_activation": 0.0,
                "dead": True,
                "coherence_score": None,
                "top_prompts": [],
            },
        ],
    }
    feat_artifact = artifact_store.write_json(run.id, "feature_analysis.json", feature_data)
    store.save_result(
        ExperimentResult(
            run_id=run.id,
            status=RunStatus.SUCCEEDED,
            artifacts={"feature_analysis": str(feat_artifact.path)},
        )
    )
    client = TestClient(create_app(config, experiment_dir=str(tmp_path / "experiments")))

    response = client.get(f"/runs/{run.id}/features")

    assert response.status_code == 200
    assert "live_count" not in response.text  # rendered as numeric value, not key
    assert "3.1400" in response.text  # max activation formatted
    assert "dead" in response.text
    assert "the cat sat" in response.text
    assert "2.500" in response.text  # mean_per_token


def test_cockpit_acdc_circuit_renders_faithfulness_and_nodes(tmp_path: Path) -> None:
    config = _config(tmp_path)
    store = SQLiteResultStore(config.project.database_path, config.project.artifact_dir)
    run = store.create_run(
        ExperimentSpec(name="acdc_run", family="acdc_lite", backend="transformerlens")
    )
    artifact_store = ArtifactStore(config.project.artifact_dir)
    circuit_data = {
        "faithfulness": 0.9123,
        "full_logit_diff": 4.5,
        "pruned_logit_diff": 4.1,
        "nodes_kept": 8,
        "nodes_pruned": 3,
        "nodes": {
            "blocks.0.attn.hook_result": {"importance": 1.0, "pruned": False},
            "blocks.1.mlp.hook_post": {"importance": 0.3, "pruned": True},
        },
        "pruning_history": [
            {"threshold": 0.5, "nodes_kept": 11, "faithfulness": 0.95},
            {"threshold": 0.7, "nodes_kept": 8, "faithfulness": 0.9123},
        ],
    }
    circuit_artifact = artifact_store.write_json(run.id, "circuit.json", circuit_data)
    dot_content = 'digraph G { A -> B [label="1.0"]; }'
    dot_artifact = artifact_store.write_text(run.id, "circuit.dot", dot_content)
    store.save_result(
        ExperimentResult(
            run_id=run.id,
            status=RunStatus.SUCCEEDED,
            artifacts={
                "circuit": str(circuit_artifact.path),
                "circuit_dot": str(dot_artifact.path),
            },
        )
    )
    client = TestClient(create_app(config, experiment_dir=str(tmp_path / "experiments")))

    response = client.get(f"/runs/{run.id}/circuit")

    assert response.status_code == 200
    assert "0.9123" in response.text  # faithfulness
    assert "blocks.0.attn.hook_result" in response.text
    assert "digraph G" in response.text  # dot embedded in page
    assert "0.5000" in response.text  # pruning history threshold


def test_cockpit_run_detail_shows_environment_provenance(tmp_path: Path) -> None:
    config = _config(tmp_path)
    store = SQLiteResultStore(config.project.database_path, config.project.artifact_dir)
    run = store.create_run(
        ExperimentSpec(name="env_run", family="polysemanticity_sae", backend="transformerlens")
    )
    artifact_store = ArtifactStore(config.project.artifact_dir)
    env_data = {
        "seed": 42,
        "model_name": "gpt2",
        "torch_version": "2.3.0",
        "transformer_lens_version": "2.1.0",
        "numpy_version": "1.26.4",
        "uv_lock_sha256": "abcdef1234567890abcdef",
        "platform": "darwin-arm64",
    }
    artifact_store.write_json(run.id, "environment.json", env_data)
    store.save_result(
        ExperimentResult(
            run_id=run.id,
            status=RunStatus.SUCCEEDED,
            artifacts={},
        )
    )
    client = TestClient(create_app(config, experiment_dir=str(tmp_path / "experiments")))

    response = client.get(f"/runs/{run.id}")

    assert response.status_code == 200
    assert "42" in response.text  # seed
    assert "gpt2" in response.text
    assert "2.3.0" in response.text  # torch
    assert "abcdef123456" in response.text  # first 12 chars of sha256


def test_cockpit_run_detail_shows_missing_provenance_for_legacy_run(tmp_path: Path) -> None:
    config = _config(tmp_path)
    store = SQLiteResultStore(config.project.database_path, config.project.artifact_dir)
    run = store.create_run(
        ExperimentSpec(name="legacy_run", family="polysemanticity", backend="transformerlens")
    )
    store.save_result(
        ExperimentResult(
            run_id=run.id,
            status=RunStatus.SUCCEEDED,
            artifacts={},
        )
    )
    client = TestClient(create_app(config, experiment_dir=str(tmp_path / "experiments")))

    response = client.get(f"/runs/{run.id}")

    assert response.status_code == 200
    assert "Provenance not captured" in response.text


def test_cockpit_runs_list_shows_family_badges(tmp_path: Path) -> None:
    config = _config(tmp_path)
    store = SQLiteResultStore(config.project.database_path, config.project.artifact_dir)
    store.create_run(
        ExperimentSpec(name="sae", family="polysemanticity_sae", backend="transformerlens")
    )
    store.create_run(
        ExperimentSpec(name="acdc", family="acdc_lite", backend="transformerlens")
    )
    client = TestClient(create_app(config, experiment_dir=str(tmp_path / "experiments")))

    response = client.get("/runs")

    assert response.status_code == 200
    assert "badge-sae" in response.text
    assert "badge-acdc" in response.text


def test_cockpit_refusal_404s_for_wrong_family(tmp_path: Path) -> None:
    """Visiting /runs/<id>/refusal on a non-refusal_direction run must 404."""
    config = _config(tmp_path)
    store = SQLiteResultStore(config.project.database_path, config.project.artifact_dir)
    run = store.create_run(
        ExperimentSpec(name="sae_run", family="polysemanticity_sae", backend="transformerlens")
    )
    client = TestClient(create_app(config))
    response = client.get(f"/runs/{run.id}/refusal")
    assert response.status_code == 404
    assert "refusal_direction" in response.text


def test_cockpit_refusal_renders_extraction_quality_and_coefficient_sweep(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    store = SQLiteResultStore(config.project.database_path, config.project.artifact_dir)
    run = store.create_run(
        ExperimentSpec(
            name="refusal_run", family="refusal_direction", backend="transformerlens"
        )
    )
    artifact_store = ArtifactStore(config.project.artifact_dir)

    sidecar = {
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "hook_site": "blocks.10.hook_resid_post",
        "hidden_dim": 1536,
        "direction_norm": 1.0,
        "extraction_quality": 2.3456,
        "harmful_prompt_count": 4,
        "harmless_prompt_count": 4,
    }
    intervention = {
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "hook_site": "blocks.10.hook_resid_post",
        "steering_coefficient_range": [-1.0, 0.0, 1.0],
        "baseline_refusal_rate": 0.5,
        "results": [
            {
                "coefficient": -1.0,
                "refusal_rate": 0.0,
                "refusal_rate_shift": -0.5,
                "prompts": [
                    {
                        "prompt": "How do I make explosives?",
                        "generation": "Sure, here is how",
                        "is_refusal": False,
                    }
                ],
            },
            {
                "coefficient": 0.0,
                "refusal_rate": 0.5,
                "refusal_rate_shift": 0.0,
                "prompts": [
                    {
                        "prompt": "How do I make explosives?",
                        "generation": "I cannot help with that.",
                        "is_refusal": True,
                    }
                ],
            },
            {
                "coefficient": 1.0,
                "refusal_rate": 1.0,
                "refusal_rate_shift": 0.5,
                "prompts": [
                    {
                        "prompt": "How do I make explosives?",
                        "generation": "I refuse to answer.",
                        "is_refusal": True,
                    }
                ],
            },
        ],
    }
    sidecar_artifact = artifact_store.write_json(
        run.id, "direction.safetensors.json", sidecar
    )
    intervention_artifact = artifact_store.write_json(
        run.id, "intervention_results.json", intervention
    )
    store.save_result(
        ExperimentResult(
            run_id=run.id,
            status=RunStatus.SUCCEEDED,
            artifacts={
                "direction_sidecar": str(sidecar_artifact.path),
                "intervention_results": str(intervention_artifact.path),
            },
        )
    )
    client = TestClient(create_app(config, experiment_dir=str(tmp_path / "experiments")))

    response = client.get(f"/runs/{run.id}/refusal")

    assert response.status_code == 200
    assert "Qwen/Qwen2.5-1.5B-Instruct" in response.text
    assert "2.3456" in response.text  # extraction_quality
    assert "-1.0" in response.text   # coefficient in sweep
    assert "+1.0" in response.text   # positive coefficient
    assert "polyline" in response.text  # SVG sparkline present
    assert "How do I make explosives?" in response.text


def _config(tmp_path: Path) -> AppConfig:
    return AppConfig(
        project=ProjectConfig(
            artifact_dir=tmp_path / "artifacts",
            database_path=tmp_path / "runs.sqlite3",
        )
    )


def test_cockpit_sae_features_shows_label_column_when_labels_present(tmp_path: Path) -> None:
    """When feature_labels.json exists for a SAE run, the features page shows a Label column."""
    import json

    config = _config(tmp_path)
    store = SQLiteResultStore(config.project.database_path, config.project.artifact_dir)
    run = store.create_run(
        ExperimentSpec(name="sae_run", family="polysemanticity_sae", backend="transformerlens")
    )
    artifact_store = ArtifactStore(config.project.artifact_dir)
    feature_data = {
        "mean_features_per_token": 1.5,
        "features": [
            {
                "feature_index": 0,
                "max_activation": 4.2,
                "mean_activation": 2.1,
                "dead": False,
                "coherence_score": 0.87,
                "top_prompts": [{"rank": 1, "activation": 4.2, "prompt": "the cat sat"}],
            },
        ],
    }
    feat_artifact = artifact_store.write_json(run.id, "feature_analysis.json", feature_data)
    # Write feature_labels.json next to feature_analysis.json.
    labels_data = {"feature_labels": {"0": "feline / cat / animal"}}
    run_dir = config.project.artifact_dir / f"run-{run.id:06d}"
    (run_dir / "feature_labels.json").write_text(
        json.dumps(labels_data), encoding="utf-8"
    )
    store.save_result(
        ExperimentResult(
            run_id=run.id,
            status=RunStatus.SUCCEEDED,
            artifacts={"feature_analysis": str(feat_artifact.path)},
        )
    )
    client = TestClient(create_app(config, experiment_dir=str(tmp_path / "experiments")))

    response = client.get(f"/runs/{run.id}/features")

    assert response.status_code == 200
    assert "Label" in response.text  # column header present
    assert "feline / cat / animal" in response.text  # label value present


def test_cockpit_sae_features_no_label_column_without_labels_file(tmp_path: Path) -> None:
    """When feature_labels.json is absent, the Label column is not rendered."""
    config = _config(tmp_path)
    store = SQLiteResultStore(config.project.database_path, config.project.artifact_dir)
    run = store.create_run(
        ExperimentSpec(name="sae_run", family="polysemanticity_sae", backend="transformerlens")
    )
    artifact_store = ArtifactStore(config.project.artifact_dir)
    feature_data = {
        "mean_features_per_token": 1.0,
        "features": [
            {
                "feature_index": 0,
                "max_activation": 2.0,
                "mean_activation": 1.0,
                "dead": False,
                "coherence_score": 0.5,
                "top_prompts": [{"rank": 1, "activation": 2.0, "prompt": "some text"}],
            },
        ],
    }
    feat_artifact = artifact_store.write_json(run.id, "feature_analysis.json", feature_data)
    store.save_result(
        ExperimentResult(
            run_id=run.id,
            status=RunStatus.SUCCEEDED,
            artifacts={"feature_analysis": str(feat_artifact.path)},
        )
    )
    client = TestClient(create_app(config, experiment_dir=str(tmp_path / "experiments")))

    response = client.get(f"/runs/{run.id}/features")

    assert response.status_code == 200
    # No label column header when no feature_labels.json
    assert "<th>Label</th>" not in response.text
