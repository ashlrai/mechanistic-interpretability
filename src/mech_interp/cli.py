from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

from mech_interp.analysis import summarize_recent_runs
from mech_interp.analysis.run_reports import write_aggregate_reports
from mech_interp.config import load_config
from mech_interp.experiments import load_experiment_specs
from mech_interp.experiments.registry import ExperimentSpecValidationError
from mech_interp.orchestration import (
    ActivationEstimate,
    ExperimentRunner,
    ExperimentRunQueue,
    ResourcePolicy,
)
from mech_interp.orchestration.iteration import IterationCaps, propose_and_enqueue_iteration
from mech_interp.orchestration.preflight import (
    inspect_dataset,
    preflight_spec,
    validate_answer_tokens,
)
from mech_interp.orchestration.proposals import propose_followups, propose_from_run
from mech_interp.providers import configured_providers
from mech_interp.storage import ArtifactStore, SQLiteResultStore

app = typer.Typer(help="Local mechanistic interpretability research platform.")
queue_app = typer.Typer(help="Manage the local resumable experiment queue.")
dataset_app = typer.Typer(help="Inspect and validate local datasets.")
app.add_typer(queue_app, name="queue")
app.add_typer(dataset_app, name="dataset")
console = Console()
DEFAULT_REPORT_OUTPUT = Path("artifacts/reports")
DEFAULT_PROPOSAL_OUTPUT = Path("experiments/proposed")


@app.command("config")
def show_config() -> None:
    """Print the resolved application config."""
    config = load_config()
    console.print_json(json.dumps(config.model_dump(mode="json"), indent=2))


@app.command("experiments")
def list_experiments(directory: str = "experiments") -> None:
    """List experiment specs discovered from YAML files."""
    registry = load_experiment_specs(directory)
    table = Table(title="Experiment Specs")
    table.add_column("Name")
    table.add_column("Family")
    table.add_column("Backend")
    table.add_column("Description")
    for spec in registry.list():
        table.add_row(spec.name, spec.family, spec.backend, spec.description)
    console.print(table)


@app.command("validate")
def validate_experiments(
    directory: str = typer.Option(
        "experiments",
        help="Directory containing experiment YAML files.",
    ),
) -> None:
    """Validate experiment specs without creating runs or artifacts."""
    try:
        registry = load_experiment_specs(directory)
    except ExperimentSpecValidationError as exc:
        console.print(f"[red]Invalid experiment specs:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    specs = registry.list()
    console.print(f"Validated {len(specs)} experiment spec(s) in {directory}.")


@app.command("providers")
def check_providers(
    timeout: float = typer.Option(2.0, help="Provider request timeout in seconds."),
) -> None:
    """Show configured local provider endpoints and reachability."""
    config = load_config()
    providers = configured_providers(config, timeout=timeout)
    table = Table(title="Configured Providers")
    table.add_column("Provider")
    table.add_column("Endpoint")
    table.add_column("Reachable")
    table.add_column("Models")
    table.add_column("Error")

    for provider in providers:
        health = provider.health_sync()
        table.add_row(
            health.provider,
            health.base_url,
            "yes" if health.reachable else "no",
            ", ".join(health.models) if health.models else "-",
            health.error or "-",
        )

    console.print(table)


@app.command("init-store")
def init_store() -> None:
    """Initialize the local SQLite result store."""
    config = load_config()
    store = SQLiteResultStore(
        database_path=config.project.database_path,
        artifact_dir=config.project.artifact_dir,
    )
    store.initialize()
    console.print(f"Initialized result store at {config.project.database_path}")


@queue_app.command("plan")
def queue_plan(
    directory: str = typer.Option(
        "experiments",
        help="Directory containing experiment YAML files.",
    ),
) -> None:
    """Enqueue all discovered experiment specs that are not already queued."""
    config = load_config()
    registry = load_experiment_specs(directory)
    store = SQLiteResultStore(config.project.database_path, config.project.artifact_dir)
    plan = ExperimentRunQueue(store).plan(registry.list())
    console.print(
        f"Queued {plan.enqueued} new experiment spec(s); {plan.total} discovered in {directory}."
    )


@queue_app.command("next")
def queue_next() -> None:
    """Claim the next planned or failed experiment spec for running."""
    config = load_config()
    store = SQLiteResultStore(config.project.database_path, config.project.artifact_dir)
    item = ExperimentRunQueue(store).claim_next()
    if item is None:
        console.print("No planned or failed experiment specs are available.")
        return
    console.print(item.spec_name)


@queue_app.command("list")
def queue_list() -> None:
    """List queued experiment specs and their resumable state."""
    config = load_config()
    store = SQLiteResultStore(config.project.database_path, config.project.artifact_dir)
    table = Table(title="Experiment Queue")
    table.add_column("ID")
    table.add_column("Spec")
    table.add_column("Status")
    table.add_column("Retries")
    table.add_column("Lease")
    table.add_column("Phase")
    table.add_column("Error")
    for item in ExperimentRunQueue(store).list():
        detail = item.error or (f"run {item.run_id}" if item.run_id is not None else "-")
        table.add_row(
            str(item.id),
            item.spec_name,
            item.status.value,
            str(item.retry_count),
            item.lease_token[:8] if item.lease_token else "-",
            item.current_phase or "-",
            detail,
        )
    console.print(table)


@queue_app.command("run")
def queue_run(
    once: bool = typer.Option(False, "--once", help="Run one queued experiment and exit."),
    loop: bool = typer.Option(False, "--loop", help="Continuously poll and run queued work."),
    poll_interval: float = typer.Option(5.0, min=0.1, help="Loop poll interval in seconds."),
    directory: str = typer.Option(
        "experiments",
        help="Directory containing experiment YAML files.",
    ),
) -> None:
    """Execute claimed queue items through the experiment runner."""
    if once == loop:
        console.print("[red]Choose exactly one of --once or --loop.[/red]")
        raise typer.Exit(code=1)
    config = load_config()
    registry = load_experiment_specs(directory)
    store = SQLiteResultStore(config.project.database_path, config.project.artifact_dir)
    queue = ExperimentRunQueue(store)
    runner = ExperimentRunner(
        result_store=store,
        artifact_store=ArtifactStore(config.project.artifact_dir),
    )
    specs_by_name = {spec.name: spec for spec in registry.list()}
    if once:
        result = queue.run_once(specs_by_name, runner)
        if result is None:
            console.print("No queued experiment specs are available.")
            return
        console.print(f"Run {result.run_id} finished with status {result.status.value}.")
        return
    while True:
        before_events = store.list_run_events(limit=1)
        result = queue.run_once(specs_by_name, runner)
        for event in reversed(store.list_run_events(limit=20)):
            if before_events and event.id <= before_events[0].id:
                continue
            console.print(
                f"[{event.created_at.isoformat()}] {event.event_type}: {event.message}"
            )
        if result is None:
            time.sleep(poll_interval)


@queue_app.command("requeue-stale")
def queue_requeue_stale(
    stale_after_seconds: int = typer.Option(3600, min=1, help="Age threshold for running items."),
) -> None:
    """Move stale running queue items back to planned."""
    config = load_config()
    store = SQLiteResultStore(config.project.database_path, config.project.artifact_dir)
    requeued = ExperimentRunQueue(store).requeue_stale(stale_after_seconds)
    console.print(f"Requeued {len(requeued)} stale item(s).")


@queue_app.command("pause")
def queue_pause(queue_id: Annotated[int, typer.Argument(min=1)]) -> None:
    """Pause a queue item by id."""
    _mutate_queue_item(queue_id, "pause", "Paused")


@queue_app.command("resume")
def queue_resume(queue_id: Annotated[int, typer.Argument(min=1)]) -> None:
    """Resume a paused queue item by id."""
    _mutate_queue_item(queue_id, "resume", "Resumed")


@queue_app.command("cancel")
def queue_cancel(queue_id: Annotated[int, typer.Argument(min=1)]) -> None:
    """Cancel a queue item by id."""
    _mutate_queue_item(queue_id, "cancel", "Cancelled")


@queue_app.command("requeue")
def queue_requeue(queue_id: Annotated[int, typer.Argument(min=1)]) -> None:
    """Requeue a queue item by id."""
    _mutate_queue_item(queue_id, "requeue", "Requeued")


@queue_app.command("status")
def queue_status() -> None:
    """Show queue counts by status."""
    config = load_config()
    store = SQLiteResultStore(config.project.database_path, config.project.artifact_dir)
    console.print_json(json.dumps(store.queue_counts_by_status(), indent=2))


@app.command("run")
def run_experiments(
    name: str | None = typer.Option(None, help="Run a single experiment by name."),
    directory: str = typer.Option(
        "experiments",
        help="Directory containing experiment YAML files.",
    ),
) -> None:
    """Run experiment specs through the local orchestration and storage spine."""
    config = load_config()
    registry = load_experiment_specs(directory)
    specs = registry.list() if name is None else [registry.get(name)]
    result_store = SQLiteResultStore(config.project.database_path, config.project.artifact_dir)
    runner = ExperimentRunner(
        result_store=result_store,
        artifact_store=ArtifactStore(config.project.artifact_dir),
    )
    results = runner.run_many(specs)

    table = Table(title="Experiment Results")
    table.add_column("Run ID")
    table.add_column("Status")
    table.add_column("Metric Count")
    table.add_column("Manifest")
    for result in results:
        table.add_row(
            str(result.run_id),
            result.status.value,
            str(len(result.metrics)),
            result.artifacts.get("manifest", ""),
        )
    console.print(table)


@app.command("runs")
def list_runs(limit: int = 20) -> None:
    """List recent experiment runs from the local SQLite store."""
    config = load_config()
    store = SQLiteResultStore(config.project.database_path, config.project.artifact_dir)
    table = Table(title="Recent Runs")
    table.add_column("Run ID")
    table.add_column("Spec")
    table.add_column("Family")
    table.add_column("Backend")
    table.add_column("Status")
    for run in store.list_runs(limit=limit):
        table.add_row(
            str(run.id),
            run.spec_name,
            run.family,
            run.backend,
            run.status.value,
        )
    console.print(table)


@app.command("summarize-runs")
def summarize_runs(limit: int = typer.Option(100, min=1)) -> None:
    """Summarize recent runs by status, family, and backend."""
    config = load_config()
    store = SQLiteResultStore(config.project.database_path, config.project.artifact_dir)
    console.print_json(json.dumps(summarize_recent_runs(store, limit=limit), indent=2))


@app.command("report-runs")
def report_runs(
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Directory for aggregate report artifacts."),
    ] = DEFAULT_REPORT_OUTPUT,
    limit: int = typer.Option(100, min=1),
) -> None:
    """Write aggregate research summaries for recent runs."""
    config = load_config()
    store = SQLiteResultStore(config.project.database_path, config.project.artifact_dir)
    reports = write_aggregate_reports(store, output, limit=limit)
    console.print(f"Wrote aggregate reports to {reports.output_dir}")


@app.command("preflight")
def preflight(
    name: str | None = typer.Option(None, help="Preflight a single experiment by name."),
    directory: str = typer.Option(
        "experiments",
        help="Directory containing experiment YAML files.",
    ),
) -> None:
    """Validate runtime readiness for specs without running experiments."""
    registry = load_experiment_specs(directory)
    specs = registry.list() if name is None else [registry.get(name)]
    failed = False
    for spec in specs:
        report = preflight_spec(spec)
        table = Table(title=f"Preflight: {spec.name}")
        table.add_column("Check")
        table.add_column("Status")
        table.add_column("Message")
        for check in report.checks:
            table.add_row(check.name, check.status, check.message)
        console.print(table)
        failed = failed or not report.ok
    if failed:
        raise typer.Exit(code=1)


@dataset_app.command("inspect")
def dataset_inspect(path: Annotated[Path, typer.Argument(exists=True)]) -> None:
    """Inspect dataset rows, fields, size, and SHA-256."""
    console.print_json(json.dumps(inspect_dataset(path), indent=2))


@dataset_app.command("validate-tokens")
def dataset_validate_tokens(
    path: Annotated[Path, typer.Argument(exists=True)],
    model: str = typer.Option(..., help="Model name used for answer-token validation context."),
) -> None:
    """Validate answer-token fields in a dataset with lightweight local checks."""
    result = validate_answer_tokens(path, model)
    console.print_json(json.dumps(result, indent=2))
    if not result["valid"]:
        raise typer.Exit(code=1)


@app.command("query-runs")
def query_runs(
    family: str | None = None,
    status: str | None = None,
    backend: str | None = None,
    model: str | None = None,
    dataset_hash: str | None = None,
    tag: str | None = None,
    metric: str | None = None,
    metric_min: float | None = None,
    hook_site: str | None = None,
    layer: int | None = None,
    matrix_id: int | None = None,
    output_format: str = typer.Option("table", "--output-format", help="table, json, or csv."),
    limit: int = typer.Option(100, min=1),
) -> None:
    """Search indexed runs by spec metadata, metrics, tags, and matrix linkage."""
    output_format = output_format.lower()
    if output_format not in {"table", "json", "csv"}:
        console.print("[red]--output-format must be one of: table, json, csv.[/red]")
        raise typer.Exit(code=1)

    config = load_config()
    store = SQLiteResultStore(config.project.database_path, config.project.artifact_dir)
    rows = store.query_runs(
        family=family,
        status=status,
        backend=backend,
        model=model,
        dataset_hash=dataset_hash,
        tag=tag,
        metric=metric,
        metric_min=metric_min,
        hook_site=hook_site,
        layer=layer,
        matrix_id=matrix_id,
        limit=limit,
    )
    if output_format == "json":
        console.print_json(json.dumps(rows, indent=2, default=str))
        return
    if output_format == "csv":
        writer = csv.DictWriter(sys.stdout, fieldnames=_QUERY_RUN_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(_query_run_csv_row(row))
        return

    if not rows:
        console.print("No runs matched query.")
        return

    table = Table(title="Run Query")
    columns = ("run_id", "spec_name", "family", "backend", "status", "metric", "tags", "matrix")
    for column in columns:
        table.add_column(column)
    for row in rows:
        table.add_row(
            str(row["run_id"]),
            str(row["spec_name"]),
            str(row["family"]),
            str(row["backend"]),
            str(row["status"]),
            _metric_summary(row, metric),
            ", ".join(str(tag_value) for tag_value in row.get("tags", [])) or "-",
            str(row["matrix_id"]) if row.get("matrix_id") is not None else "-",
        )
    console.print(table)


@app.command("propose-followups")
def propose_followup_specs(
    family: str = typer.Option("circuit_patching", help="Experiment family to propose for."),
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Directory for proposed YAML specs."),
    ] = DEFAULT_PROPOSAL_OUTPUT,
    limit: int = typer.Option(20, min=1),
) -> None:
    """Generate deterministic follow-up specs from aggregate reports."""
    result = propose_followups(family, output, limit=limit)
    console.print(
        f"Wrote {len(result.spec_paths)} proposed spec(s) and manifest {result.manifest_path}."
    )


@app.command("propose-from-run")
def propose_from_run_command(
    family: str = typer.Option(
        ...,
        help="Family of the source run (polysemanticity_sae, acdc_lite).",
    ),
    artifact_dir: Annotated[
        Path,
        typer.Option("--artifact-dir", "-a", help="Run artifact directory to read."),
    ] = Path("."),
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Directory for proposed YAML specs."),
    ] = DEFAULT_PROPOSAL_OUTPUT,
    limit: int = typer.Option(5, min=1),
) -> None:
    """Generate per-run follow-up specs from a single run's artifacts.

    Closes the agentic loop: a SAE run becomes circuit_patching probes for its
    top features; an ACDC-lite run becomes activation captures over its
    surviving nodes.
    """
    result = propose_from_run(family, artifact_dir, output, limit=limit)
    console.print(
        f"Wrote {len(result.spec_paths)} proposed spec(s) and manifest {result.manifest_path}."
    )


@app.command("iterate")
def iterate(
    family: str = typer.Option("circuit_patching", help="Experiment family to propose for."),
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Directory for iteration proposal manifests."),
    ] = DEFAULT_PROPOSAL_OUTPUT,
    max_generated_specs: int = typer.Option(50, min=1),
    max_queued_per_iteration: int = typer.Option(10, min=1),
    max_failed_retry_count: int = typer.Option(2, min=0),
    allow_tensor_retention: bool = typer.Option(False, help="Allow retained tensor artifacts."),
) -> None:
    """Generate, preflight, rank, and enqueue bounded local follow-up specs."""
    config = load_config()
    store = SQLiteResultStore(config.project.database_path, config.project.artifact_dir)
    result = propose_and_enqueue_iteration(
        store,
        family,
        output,
        IterationCaps(
            max_generated_specs=max_generated_specs,
            max_queued_per_iteration=max_queued_per_iteration,
            max_failed_retry_count=max_failed_retry_count,
            allow_tensor_retention=allow_tensor_retention,
        ),
    )
    console.print(
        f"Generated {result.generated} candidate(s), queued {result.queued}, "
        f"wrote {result.manifest_path}."
    )


@app.command("cockpit")
def cockpit(
    host: str = typer.Option("127.0.0.1", help="Bind host."),
    port: int = typer.Option(8000, min=1, max=65535, help="Bind port."),
    directory: str = typer.Option(
        "experiments",
        help="Directory containing experiment YAML files.",
    ),
) -> None:
    """Run the local FastAPI/HTMX research cockpit."""
    import uvicorn

    from mech_interp.cockpit import create_app

    uvicorn.run(create_app(load_config(), experiment_dir=directory), host=host, port=port)


@app.command("inspect-run")
def inspect_run(run_id: Annotated[int, typer.Argument(min=1)]) -> None:
    """Print stored spec, config, result, and manifest data for a run."""
    bundle = _run_bundle(run_id)
    console.print_json(json.dumps(bundle, indent=2, sort_keys=True))


@app.command("export-run")
def export_run(
    run_id: Annotated[int, typer.Argument(min=1)],
    output: Annotated[Path, typer.Option("--output", "-o", help="JSON file to write.")],
) -> None:
    """Export stored run data as a JSON bundle."""
    bundle = _run_bundle(run_id)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(bundle, default=str, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    console.print(f"Exported run {run_id} to {output}")


@app.command("estimate-activations")
def estimate_activations(
    batch_size: int = typer.Option(..., min=1),
    sequence_length: int = typer.Option(..., min=1),
    hidden_size: int = typer.Option(..., min=1),
    hook_count: int = typer.Option(..., min=1),
    dtype: str = typer.Option("float16"),
    max_ram_gib: float = typer.Option(128.0, min=1.0),
    max_activation_fraction: float = typer.Option(0.35, min=0.01, max=1.0),
) -> None:
    """Estimate activation-cache memory for a proposed experiment batch."""
    estimate = ActivationEstimate(
        batch_size=batch_size,
        sequence_length=sequence_length,
        hidden_size=hidden_size,
        hook_count=hook_count,
        dtype=dtype,
    )
    policy = ResourcePolicy(
        max_ram_gib=max_ram_gib,
        max_activation_fraction=max_activation_fraction,
    )
    policy.validate_activation_estimate(estimate)
    console.print(
        {
            "estimated_gib": round(estimate.gib, 4),
            "max_activation_gib": round(policy.max_activation_gib, 4),
        }
    )


def _run_bundle(run_id: int) -> dict[str, Any]:
    config = load_config()
    store = SQLiteResultStore(config.project.database_path, config.project.artifact_dir)
    spec = store.get_run_spec(run_id)
    if spec is None:
        console.print(f"[red]Run {run_id} was not found.[/red]")
        raise typer.Exit(code=1)

    result = store.get_result(run_id)
    result_payload: dict[str, Any] | None = None
    if result is not None:
        result_payload = {
            "run_id": result.run_id,
            "status": result.status.value,
            "metrics": result.metrics,
            "artifacts": result.artifacts,
            "notes": result.notes,
        }

    return {
        "run_id": run_id,
        "spec": spec,
        "config": store.get_run_config(run_id) or {},
        "result": result_payload,
        "manifest": _read_manifest(config.project.artifact_dir, run_id, result_payload),
    }


def _read_manifest(
    artifact_dir: Path,
    run_id: int,
    result_payload: dict[str, Any] | None,
) -> dict[str, Any] | None:
    manifest_path = None
    if result_payload is not None:
        artifacts = result_payload.get("artifacts", {})
        if isinstance(artifacts, dict):
            manifest_value = artifacts.get("manifest")
            if isinstance(manifest_value, str):
                manifest_path = Path(manifest_value)

    candidate_paths = [
        path
        for path in (
            manifest_path,
            artifact_dir / f"run-{run_id:06d}" / "manifest.json",
        )
        if path is not None
    ]
    for path in candidate_paths:
        if path.exists():
            with path.open("r", encoding="utf-8") as manifest_file:
                manifest = json.load(manifest_file)
            if not isinstance(manifest, dict):
                raise ValueError(f"Manifest {path} did not contain a JSON object.")
            return manifest
    return None


_QUERY_RUN_COLUMNS = (
    "run_id",
    "spec_name",
    "family",
    "backend",
    "status",
    "created_at",
    "spec_sha256",
    "tags",
    "hypothesis",
    "matrix_id",
    "metrics",
)


def _mutate_queue_item(queue_id: int, action: str, label: str) -> None:
    config = load_config()
    queue = ExperimentRunQueue(
        SQLiteResultStore(config.project.database_path, config.project.artifact_dir)
    )
    try:
        item = getattr(queue, action)(queue_id)
    except (KeyError, RuntimeError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    console.print(f"{label} queue item {item.id}; status is {item.status.value}.")


def _query_run_csv_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": row.get("run_id"),
        "spec_name": row.get("spec_name"),
        "family": row.get("family"),
        "backend": row.get("backend"),
        "status": row.get("status"),
        "created_at": row.get("created_at"),
        "spec_sha256": row.get("spec_sha256") or "",
        "tags": _json_cell(row.get("tags", [])),
        "hypothesis": row.get("hypothesis") or "",
        "matrix_id": row.get("matrix_id") if row.get("matrix_id") is not None else "",
        "metrics": _json_cell(row.get("metrics", {})),
    }


def _metric_summary(row: dict[str, Any], metric: str | None) -> str:
    metrics = row.get("metrics", {})
    if not isinstance(metrics, dict) or not metrics:
        return "-"
    if metric:
        value = metrics.get(metric)
        return f"{metric}={value}" if value is not None else "-"
    return ", ".join(f"{key}={value}" for key, value in sorted(metrics.items()))


def _json_cell(value: Any) -> str:
    return json.dumps(value, default=str, sort_keys=True)
