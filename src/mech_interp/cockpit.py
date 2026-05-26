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
        environment = _read_environment(run_id, result, config.project.artifact_dir)
        return templates.TemplateResponse(
            request,
            "run_detail.html",
            {
                "run": run,
                "result": result,
                "spec": json.dumps(spec, indent=2, sort_keys=True),
                "manifest": manifest,
                "report_preview": report_preview,
                "environment": environment,
                "artifact_links": _artifact_links(
                    result.artifacts if result else {},
                    config.project.artifact_dir,
                ),
            },
        )

    @app.get("/runs/{run_id}/features", response_class=HTMLResponse)
    def sae_features(request: Request, run_id: int) -> HTMLResponse:
        db = store()
        run = next((item for item in db.list_runs(limit=500) if item.id == run_id), None)
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found.")
        if run.family != "polysemanticity_sae":
            raise HTTPException(
                status_code=404,
                detail=(
                    f"Run {run_id} is family '{run.family}', not 'polysemanticity_sae'; "
                    "SAE feature browser is only available for SAE runs."
                ),
            )
        result = db.get_result(run_id)
        feature_analysis_path = _find_artifact_path(
            "feature_analysis",
            result,
            config.project.artifact_dir / f"run-{run_id:06d}" / "feature_analysis.json",
        )
        raw = _read_json_artifact(
            str(feature_analysis_path) if feature_analysis_path else None
        )
        features, stats = _parse_sae_features(raw)
        # Load optional feature labels (written by `mech label-features`).
        feature_labels_path = (
            config.project.artifact_dir / f"run-{run_id:06d}" / "feature_labels.json"
        )
        feature_labels = _load_feature_labels(feature_labels_path)
        # Attach labels to each feature dict.
        for feat in features:
            feat["label"] = feature_labels.get(str(feat["feature_index"]))
        return templates.TemplateResponse(
            request,
            "sae_features.html",
            {"run": run, "features": features, "stats": stats, "has_labels": bool(feature_labels)},
        )

    @app.get("/runs/{run_id}/circuit", response_class=HTMLResponse)
    def acdc_circuit(request: Request, run_id: int) -> HTMLResponse:
        db = store()
        run = next((item for item in db.list_runs(limit=500) if item.id == run_id), None)
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found.")
        if run.family not in {"acdc_lite", "acdc_edge"}:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"Run {run_id} is family '{run.family}', not an ACDC family; "
                    "circuit view is only available for acdc_lite / acdc_edge runs."
                ),
            )
        result = db.get_result(run_id)
        run_dir = config.project.artifact_dir / f"run-{run_id:06d}"
        circuit_json_path = _find_artifact_path("circuit", result, run_dir / "circuit.json")
        circuit_dot_path = _find_artifact_path(
            "circuit_dot", result, run_dir / "circuit.dot"
        )
        circuit_data = _read_json_artifact(
            str(circuit_json_path) if circuit_json_path else None
        )
        dot_source: str | None = None
        if circuit_dot_path and circuit_dot_path.is_file():
            try:
                dot_source = circuit_dot_path.read_text(encoding="utf-8")
            except OSError:
                pass
        circuit, nodes, pruning_history = _parse_acdc_circuit(circuit_data)
        return templates.TemplateResponse(
            request,
            "acdc_circuit.html",
            {
                "run": run,
                "circuit": circuit,
                "dot_source": dot_source,
                "nodes": nodes,
                "pruning_history": pruning_history,
            },
        )

    @app.get("/runs/{run_id}/refusal", response_class=HTMLResponse)
    def refusal_direction(request: Request, run_id: int) -> HTMLResponse:
        db = store()
        run = next((item for item in db.list_runs(limit=500) if item.id == run_id), None)
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found.")
        if run.family != "refusal_direction":
            raise HTTPException(
                status_code=404,
                detail=(
                    f"Run {run_id} is family '{run.family}', not 'refusal_direction'; "
                    "refusal sweep view is only available for refusal_direction runs."
                ),
            )
        result = db.get_result(run_id)
        run_dir = config.project.artifact_dir / f"run-{run_id:06d}"
        direction_sidecar_path = _find_artifact_path(
            "direction_sidecar", result, run_dir / "direction.safetensors.json"
        )
        intervention_path = _find_artifact_path(
            "intervention_results", result, run_dir / "intervention_results.json"
        )
        direction_json = _read_json_artifact(
            str(direction_sidecar_path) if direction_sidecar_path else None
        )
        intervention_json = _read_json_artifact(
            str(intervention_path) if intervention_path else None
        )
        summary, generations, charts_by_prompt = _parse_refusal_direction(
            direction_json, intervention_json
        )
        return templates.TemplateResponse(
            request,
            "refusal_direction.html",
            {
                "run": run,
                "summary": summary,
                "generations": generations,
                "charts_by_prompt": charts_by_prompt,
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
        return True
    except ValueError:
        return False


def _find_artifact_path(
    key: str,
    result: Any,
    fallback: Path,
) -> Path | None:
    """Return the artifact path from result.artifacts[key], or fallback if it exists."""
    if result is not None:
        value = result.artifacts.get(key)
        if value:
            p = Path(value)
            if p.is_file():
                return p
    if fallback.is_file():
        return fallback
    return None


def _read_environment(
    run_id: int,
    result: Any,
    artifact_dir: Path,
) -> dict[str, Any] | None:
    """Load environment.json for a run, returning None if absent."""
    # Check explicit artifact key first, then canonical path.
    candidate: Path | None = None
    if result is not None:
        value = result.artifacts.get("environment")
        if value:
            candidate = Path(value)
    if candidate is None or not candidate.is_file():
        candidate = artifact_dir / f"run-{run_id:06d}" / "environment.json"
    try:
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _parse_refusal_direction(
    direction_json: dict[str, Any] | None,
    intervention_json: dict[str, Any] | None,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    """Parse direction sidecar + intervention results into display structures.

    Returns:
        summary: header-card fields (model, hook_site, hidden_dim, direction_norm,
                 extraction_quality, harmful_prompt_count, harmless_prompt_count)
        generations: flat list of per-(prompt, coefficient) rows for the table
        charts_by_prompt: mapping prompt -> sorted list of {coefficient, refusal_rate}
                          for SVG sparkline rendering in the template
    """
    summary: dict[str, Any] = {
        "model": (direction_json or {}).get("model", ""),
        "hook_site": (direction_json or {}).get("hook_site", ""),
        "hidden_dim": (direction_json or {}).get("hidden_dim", 0),
        "direction_norm": float((direction_json or {}).get("direction_norm", 0.0)),
        "extraction_quality": float((direction_json or {}).get("extraction_quality", 0.0)),
        "harmful_prompt_count": int((direction_json or {}).get("harmful_prompt_count", 0)),
        "harmless_prompt_count": int((direction_json or {}).get("harmless_prompt_count", 0)),
    }

    results: list[dict[str, Any]] = []
    if isinstance((intervention_json or {}).get("results"), list):
        results = intervention_json["results"]  # type: ignore[index]

    generations: list[dict[str, Any]] = []
    charts_by_prompt: dict[str, list[dict[str, Any]]] = {}

    for row in results:
        if not isinstance(row, dict):
            continue
        coeff = float(row.get("coefficient", 0.0))
        refusal_rate = float(row.get("refusal_rate", 0.0))
        for prompt_row in row.get("prompts", []):
            if not isinstance(prompt_row, dict):
                continue
            prompt = str(prompt_row.get("prompt", ""))
            gen_text = str(prompt_row.get("generation", ""))
            is_refusal = bool(prompt_row.get("is_refusal", False))
            generations.append(
                {
                    "prompt": prompt,
                    "coefficient": coeff,
                    "generation": gen_text,
                    "generation_snippet": gen_text[:200],
                    "generation_full": gen_text,
                    "is_refusal": is_refusal,
                    "refusal_rate": refusal_rate,
                }
            )
        # chart data: one point per coefficient for each prompt (use aggregate refusal_rate)
        # We also collect per-coefficient points for the prompt-level chart.
        for prompt_row in row.get("prompts", []):
            if not isinstance(prompt_row, dict):
                continue
            prompt = str(prompt_row.get("prompt", ""))
            if prompt not in charts_by_prompt:
                charts_by_prompt[prompt] = []
            charts_by_prompt[prompt].append(
                {"coefficient": coeff, "refusal_rate": refusal_rate}
            )

    # Sort each chart series by coefficient ascending.
    for prompt in charts_by_prompt:
        charts_by_prompt[prompt].sort(key=lambda p: p["coefficient"])

    # Sort generations by coefficient ascending.
    generations.sort(key=lambda g: g["coefficient"])

    return summary, generations, charts_by_prompt


def _parse_sae_features(
    raw: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Parse feature_analysis.json into a flat feature list and summary stats."""
    if raw is None:
        return [], {"total": 0, "live": 0, "dead": 0, "mean_per_token": 0.0}

    # Accept either {"features": [...]} or a list at top level.
    feature_list: list[Any] = []
    if isinstance(raw.get("features"), list):
        feature_list = raw["features"]
    elif isinstance(raw, dict):
        # Some runs write the array directly under a numeric-keyed dict or
        # as the root object when it's actually a list stored as dict.
        feature_list = [raw] if raw.get("feature_index") is not None else list(raw.values())

    normalised: list[dict[str, Any]] = []
    for item in feature_list:
        if not isinstance(item, dict):
            continue
        normalised.append(
            {
                "feature_index": int(item.get("feature_index", 0)),
                "max_activation": float(item.get("max_activation", 0.0)),
                "mean_activation": float(item.get("mean_activation", 0.0)),
                "dead": bool(item.get("dead", False)),
                "coherence_score": (
                    float(item["coherence_score"])
                    if item.get("coherence_score") is not None
                    else None
                ),
                "top_prompts": list(item.get("top_prompts") or []),
            }
        )

    # Sort by max_activation descending by default.
    normalised.sort(key=lambda f: f["max_activation"], reverse=True)

    total = len(normalised)
    dead_count = sum(1 for f in normalised if f["dead"])
    stats: dict[str, Any] = {
        "total": total,
        "live": total - dead_count,
        "dead": dead_count,
        "mean_per_token": float(raw.get("mean_features_per_token", 0.0)),
    }
    return normalised, stats


def _parse_acdc_circuit(
    raw: dict[str, Any] | None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Parse circuit.json into summary, sorted node list, and pruning history."""
    if raw is None:
        empty: dict[str, Any] = {
            "faithfulness": 0.0,
            "full_logit_diff": 0.0,
            "pruned_logit_diff": 0.0,
            "nodes_kept": 0,
            "nodes_pruned": 0,
        }
        return empty, [], []

    circuit: dict[str, Any] = {
        "faithfulness": float(raw.get("faithfulness", 0.0)),
        "full_logit_diff": float(raw.get("full_logit_diff", 0.0)),
        "pruned_logit_diff": float(raw.get("pruned_logit_diff", 0.0)),
        "nodes_kept": int(raw.get("nodes_kept", 0)),
        "nodes_pruned": int(raw.get("nodes_pruned", 0)),
    }

    # Nodes: accept {"nodes": {"name": {"importance": ..., "pruned": ...}, ...}}
    # or {"nodes": [{"name": ..., "importance": ..., "pruned": ...}, ...]}
    raw_nodes = raw.get("nodes", {})
    node_items: list[tuple[str, dict[str, Any]]] = []
    if isinstance(raw_nodes, dict):
        node_items = [
            (str(name), info) for name, info in raw_nodes.items() if isinstance(info, dict)
        ]
    elif isinstance(raw_nodes, list):
        node_items = [
            (str(item.get("name", "")), item) for item in raw_nodes if isinstance(item, dict)
        ]
    nodes: list[dict[str, Any]] = [
        {
            "name": name,
            "importance": float(info.get("importance", 0.0)),
            "pruned": bool(info.get("pruned", False)),
        }
        for name, info in node_items
    ]
    nodes.sort(key=lambda n: n["importance"], reverse=True)
    top_nodes = nodes[:20]

    # Pruning history: list of steps
    raw_history = raw.get("pruning_history", [])
    pruning_history: list[dict[str, Any]] = []
    if isinstance(raw_history, list):
        for step in raw_history:
            if isinstance(step, dict):
                pruning_history.append(
                    {
                        "threshold": step.get("threshold"),
                        "nodes_kept": step.get("nodes_kept"),
                        "faithfulness": step.get("faithfulness"),
                    }
                )

    return circuit, top_nodes, pruning_history


def _load_feature_labels(labels_path: Path) -> dict[str, str]:
    """Load feature_labels.json, returning {str(feature_index): label}.

    Returns an empty dict if the file does not exist or is malformed.
    """
    try:
        payload = json.loads(labels_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    raw_labels = payload.get("feature_labels", {})
    if not isinstance(raw_labels, dict):
        return {}
    return {str(k): str(v) for k, v in raw_labels.items()}
