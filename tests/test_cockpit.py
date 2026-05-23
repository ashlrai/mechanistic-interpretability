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


def _config(tmp_path: Path) -> AppConfig:
    return AppConfig(
        project=ProjectConfig(
            artifact_dir=tmp_path / "artifacts",
            database_path=tmp_path / "runs.sqlite3",
        )
    )
