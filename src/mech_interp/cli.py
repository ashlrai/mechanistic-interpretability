from __future__ import annotations

import json

import typer
from rich.console import Console
from rich.table import Table

from mech_interp.config import load_config
from mech_interp.experiments import load_experiment_specs
from mech_interp.orchestration import ActivationEstimate, ExperimentRunner, ResourcePolicy
from mech_interp.providers import configured_providers
from mech_interp.storage import ArtifactStore, SQLiteResultStore

app = typer.Typer(help="Local mechanistic interpretability research platform.")
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
