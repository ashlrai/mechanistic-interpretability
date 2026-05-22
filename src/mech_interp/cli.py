from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

from mech_interp.analysis import summarize_recent_runs
from mech_interp.config import load_config
from mech_interp.experiments import load_experiment_specs
from mech_interp.experiments.registry import ExperimentSpecValidationError
from mech_interp.orchestration import (
    ActivationEstimate,
    ExperimentRunner,
    ExperimentRunQueue,
    ResourcePolicy,
)
from mech_interp.providers import configured_providers
from mech_interp.storage import ArtifactStore, SQLiteResultStore

app = typer.Typer(help="Local mechanistic interpretability research platform.")
queue_app = typer.Typer(help="Manage the local resumable experiment queue.")
app.add_typer(queue_app, name="queue")
console = Console()


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
    table.add_column("Error")
    for item in ExperimentRunQueue(store).list():
        table.add_row(
            str(item.id),
            item.spec_name,
            item.status.value,
            str(item.retry_count),
            item.error or "-",
        )
    console.print(table)


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
