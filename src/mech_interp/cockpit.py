from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from mech_interp.analysis.run_reports import write_aggregate_reports
from mech_interp.config.loader import AppConfig
from mech_interp.experiments import load_experiment_specs
from mech_interp.orchestration import ExperimentRunner, ExperimentRunQueue
from mech_interp.storage import ArtifactStore, SQLiteResultStore
from mech_interp.storage.sqlite_store import QueueStatus

REPORT_ARTIFACTS = (
    "latest_research_note.md",
    "latest_summary.json",
    "circuit_patching_top_sites.csv",
)
TEXT_PREVIEW_SUFFIXES = {".csv", ".json", ".md", ".txt", ".yaml", ".yml"}
MAX_PREVIEW_BYTES = 4096


def create_app(config: AppConfig, experiment_dir: str = "experiments") -> FastAPI:
    app = FastAPI(title="Mechanistic Interpretability Cockpit")
    templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

    def store() -> SQLiteResultStore:
        return SQLiteResultStore(config.project.database_path, config.project.artifact_dir)

    @app.get("/", response_class=HTMLResponse)
    def dashboard(request: Request) -> HTMLResponse:
        db = store()
        runs = db.list_runs(limit=10)
        queue_items = ExperimentRunQueue(db).list()
        top_sites = _latest_top_sites(config.project.artifact_dir)
        status_cards = _status_cards(
            db,
            {
                "database_path": str(config.project.database_path),
                "artifact_root": str(config.project.artifact_dir),
                "experiment_dir": experiment_dir,
            },
        )
        return templates.TemplateResponse(
            request,
            "dashboard.html",
            {
                "runs": runs,
                "queue_items": queue_items,
                "top_sites": top_sites,
                "status_cards": status_cards,
            },
        )

    @app.get("/queue", response_class=HTMLResponse)
    def queue_page(request: Request) -> HTMLResponse:
        registry = load_experiment_specs(experiment_dir)
        db = store()
        return templates.TemplateResponse(
            request,
            "queue.html",
            {
                "queue_items": ExperimentRunQueue(db).list(),
                "specs": registry.list(),
            },
        )

    @app.post("/queue/enqueue")
    def enqueue_queue() -> RedirectResponse:
        registry = load_experiment_specs(experiment_dir)
        ExperimentRunQueue(store()).plan(registry.list())
        return RedirectResponse("/queue", status_code=303)

    @app.post("/queue/run-once")
    def run_queue_once() -> RedirectResponse:
        registry = load_experiment_specs(experiment_dir)
        db = store()
        queue = ExperimentRunQueue(db)
        runner = ExperimentRunner(
            result_store=db,
            artifact_store=ArtifactStore(config.project.artifact_dir),
        )
        queue.run_once({spec.name: spec for spec in registry.list()}, runner)
        return RedirectResponse("/queue", status_code=303)

    @app.post("/queue/pause/{queue_id}")
    def pause_queue(queue_id: int) -> RedirectResponse:
        return _queue_action_redirect(store(), queue_id, "pause")

    @app.post("/queue/resume/{queue_id}")
    def resume_queue(queue_id: int) -> RedirectResponse:
        return _queue_action_redirect(store(), queue_id, "resume")

    @app.post("/queue/cancel/{queue_id}")
    def cancel_queue(queue_id: int) -> RedirectResponse:
        return _queue_action_redirect(store(), queue_id, "cancel")

    @app.post("/queue/requeue/{queue_id}")
    def requeue_queue(queue_id: int) -> RedirectResponse:
        return _queue_action_redirect(store(), queue_id, "requeue")

    @app.get("/status")
    def status() -> dict[str, Any]:
        db = store()
        return {
            "generated_at": datetime.now(UTC).isoformat(),
            "queue_counts": _queue_counts(db.queue_counts_by_status()),
            "recent_runs": [
                {
                    "id": run.id,
                    "spec_name": run.spec_name,
                    "family": run.family,
                    "backend": run.backend,
                    "status": run.status.value,
                    "created_at": run.created_at.isoformat(),
                }
                for run in db.list_runs(limit=10)
            ],
            "events": [
                {
                    "id": event.id,
                    "run_id": event.run_id,
                    "queue_id": event.queue_id,
                    "attempt_id": event.attempt_id,
                    "type": event.event_type,
                    "message": event.message,
                    "payload": event.payload,
                    "created_at": event.created_at.isoformat(),
                }
                for event in db.list_run_events(limit=20)
            ],
            "database_path": str(config.project.database_path),
            "artifact_root": str(config.project.artifact_dir),
            "experiment_dir": experiment_dir,
        }

    @app.get("/status/cards", response_class=HTMLResponse)
    def status_cards(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(
            request,
            "status_cards.html",
            {
                "status_cards": _status_cards(
                    store(),
                    {
                        "database_path": str(config.project.database_path),
                        "artifact_root": str(config.project.artifact_dir),
                        "experiment_dir": experiment_dir,
                    },
                )
            },
        )

    @app.get("/runs", response_class=HTMLResponse)
    def runs_page(
        request: Request,
        family: str | None = None,
        status: str | None = None,
    ) -> HTMLResponse:
        runs = store().list_runs(limit=100)
        if family:
            runs = [run for run in runs if run.family == family]
        if status:
            runs = [run for run in runs if run.status.value == status]
        return templates.TemplateResponse(request, "runs.html", {"runs": runs})

    @app.get("/runs/{run_id}", response_class=HTMLResponse)
    def run_detail(request: Request, run_id: int) -> HTMLResponse:
        db = store()
        run = next((item for item in db.list_runs(limit=500) if item.id == run_id), None)
        result = db.get_result(run_id)
        spec = db.get_run_spec(run_id) or {}
        manifest = _read_json_artifact(result.artifacts.get("manifest") if result else None)
        report_preview = _read_text_artifact(
            result.artifacts.get("research_note") if result else None
        )
        return templates.TemplateResponse(
            request,
            "run_detail.html",
            {
                "run": run,
                "result": result,
                "spec": json.dumps(spec, indent=2, sort_keys=True),
                "manifest": manifest,
                "report_preview": report_preview,
                "artifact_links": _artifact_links(
                    result.artifacts if result else {},
                    config.project.artifact_dir,
                ),
            },
        )

    @app.get("/reports", response_class=HTMLResponse)
    def reports_page(request: Request) -> HTMLResponse:
        reports_dir = config.project.artifact_dir / "reports"
        return templates.TemplateResponse(
            request,
            "reports.html",
            {
                "reports": [
                    _artifact_link(path, config.project.artifact_dir)
                    for path in (reports_dir / name for name in REPORT_ARTIFACTS)
                ]
            },
        )

    @app.post("/reports/generate")
    def generate_reports() -> RedirectResponse:
        write_aggregate_reports(store(), config.project.artifact_dir / "reports")
        return RedirectResponse("/reports", status_code=303)

    @app.get("/artifacts/browser/{run_id}", response_class=HTMLResponse)
    def artifact_browser(request: Request, run_id: int) -> HTMLResponse:
        db = store()
        run = next((item for item in db.list_runs(limit=500) if item.id == run_id), None)
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found.")
        result = db.get_result(run_id)
        manifest_path = (
            result.artifacts.get("manifest")
            if result and "manifest" in result.artifacts
            else str(config.project.artifact_dir / f"run-{run_id:06d}" / "manifest.json")
        )
        manifest = _read_json_artifact(manifest_path)
        return templates.TemplateResponse(
            request,
            "artifact_browser.html",
            {
                "run": run,
                "result": result,
                "manifest": manifest,
                "artifacts": _artifact_browser_entries(
                    result.artifacts if result else {},
                    manifest,
                    config.project.artifact_dir,
                ),
            },
        )

    @app.post("/queue/requeue-stale")
    def requeue_stale(seconds: int = 3600) -> RedirectResponse:
        ExperimentRunQueue(store()).requeue_stale(seconds)
        return RedirectResponse("/queue", status_code=303)

    @app.get("/artifacts/{artifact_path:path}")
    def artifact_file(artifact_path: str) -> FileResponse:
        root = config.project.artifact_dir.resolve()
        path = (root / artifact_path).resolve()
        if not path.is_file() or not _is_relative_to(path, root):
            raise HTTPException(status_code=404, detail="Artifact not found.")
        return FileResponse(path)

    return app


def _latest_top_sites(artifact_dir: Path) -> list[dict[str, Any]]:
    path = artifact_dir / "reports" / "latest_summary.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(payload, dict):
        return []
    top_sites = payload.get("top_circuit_patching_sites", [])
    return [item for item in top_sites[:10] if isinstance(item, dict)]


def _status_cards(db: SQLiteResultStore, paths: dict[str, str]) -> dict[str, Any]:
    events = db.list_run_events(limit=5)
    return {
        **paths,
        "queue_counts": _queue_counts(db.queue_counts_by_status()),
        "latest_event": events[0] if events else None,
        "updated_at": datetime.now(UTC),
    }


def _queue_counts(raw_counts: dict[str, int]) -> dict[str, int]:
    counts = {status.value: 0 for status in QueueStatus}
    counts.update(raw_counts)
    return counts


def _queue_action_redirect(
    db: SQLiteResultStore,
    queue_id: int,
    action: str,
) -> RedirectResponse:
    queue = ExperimentRunQueue(db)
    try:
        getattr(queue, action)(queue_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return RedirectResponse("/queue", status_code=303)


def _read_json_artifact(path_value: str | None) -> dict[str, Any] | None:
    if path_value is None:
        return None
    try:
        payload = json.loads(Path(path_value).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"missing_or_invalid": path_value}
    return payload if isinstance(payload, dict) else {"value": payload}


def _read_text_artifact(path_value: str | None) -> str:
    if path_value is None:
        return "No Markdown report artifact was recorded for this run."
    try:
        return Path(path_value).read_text(encoding="utf-8")
    except OSError:
        return f"Report artifact is missing: {path_value}"


def _artifact_links(artifacts: dict[str, str], artifact_root: Path) -> list[dict[str, Any]]:
    links: list[dict[str, Any]] = []
    for name, path in artifacts.items():
        link = _artifact_link(Path(path), artifact_root)
        link["name"] = name
        links.append(link)
    return links


def _artifact_browser_entries(
    result_artifacts: dict[str, str],
    manifest: dict[str, Any] | None,
    artifact_root: Path,
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for item in (manifest or {}).get("artifacts", []):
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            continue
        entry = _artifact_entry(
            name=str(item.get("name") or Path(str(item["path"])).name),
            path=Path(str(item["path"])),
            artifact_root=artifact_root,
            manifest_item=item,
        )
        entries.append(entry)
        seen_paths.add(entry["path"])

    for name, path_value in result_artifacts.items():
        if path_value in seen_paths:
            continue
        entries.append(
            _artifact_entry(
                name=name,
                path=Path(path_value),
                artifact_root=artifact_root,
                manifest_item={},
            )
        )
    return entries


def _artifact_entry(
    *,
    name: str,
    path: Path,
    artifact_root: Path,
    manifest_item: dict[str, Any],
) -> dict[str, Any]:
    link = _artifact_link(path, artifact_root)
    link["name"] = name
    link["media_type"] = str(manifest_item.get("media_type") or link.get("media_type") or "")
    link["sha256"] = str(manifest_item.get("sha256") or "")
    link["metadata"] = manifest_item.get("metadata") or {}
    link["preview"] = _artifact_preview(path, artifact_root)
    return link


def _artifact_link(path: Path, artifact_root: Path) -> dict[str, Any]:
    resolved_root = artifact_root.resolve()
    display_path = str(path)
    try:
        resolved_path = _resolve_artifact_path(path, artifact_root)
        if not _is_relative_to(resolved_path, resolved_root):
            return {"name": path.name, "path": display_path, "href": "", "exists": False}
        exists = resolved_path.is_file()
        stat = resolved_path.stat() if exists else None
        return {
            "name": path.name,
            "path": display_path,
            "href": f"/artifacts/{resolved_path.relative_to(resolved_root)}",
            "exists": exists,
            "size_bytes": stat.st_size if stat else None,
            "media_type": _media_type_from_path(resolved_path),
        }
    except OSError:
        return {"name": path.name, "path": display_path, "href": "", "exists": False}


def _artifact_preview(path: Path, artifact_root: Path) -> str:
    try:
        resolved_path = _resolve_artifact_path(path, artifact_root)
        if (
            not _is_relative_to(resolved_path, artifact_root.resolve())
            or not resolved_path.is_file()
            or resolved_path.suffix.lower() not in TEXT_PREVIEW_SUFFIXES
        ):
            return ""
        preview = resolved_path.read_bytes()[:MAX_PREVIEW_BYTES].decode(
            "utf-8",
            errors="replace",
        )
    except OSError:
        return ""
    if resolved_path.suffix.lower() == ".json":
        try:
            return json.dumps(json.loads(preview), indent=2, sort_keys=True)
        except json.JSONDecodeError:
            return preview
    return preview


def _resolve_artifact_path(path: Path, artifact_root: Path) -> Path:
    return path.resolve() if path.is_absolute() else (artifact_root / path).resolve()


def _media_type_from_path(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return "application/json"
    if suffix == ".csv":
        return "text/csv"
    if suffix in {".md", ".txt", ".yaml", ".yml"}:
        return "text/plain"
    if suffix == ".npz":
        return "application/x-numpy-npz"
    return "application/octet-stream"


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True
