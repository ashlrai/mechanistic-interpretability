from __future__ import annotations

import csv
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

from mech_interp.analysis import summarize_recent_runs
from mech_interp.analysis.feature_labeler import (
    AnthropicFeatureLabeler,
    FeatureLabeler,
    HeuristicFeatureLabeler,
    OllamaFeatureLabeler,
    label_run_features,
)
from mech_interp.analysis.run_reports import (
    _load_run_artifacts_from_dir,
    environment_provenance,
    inspect_run_family,
    write_aggregate_reports,
)
from mech_interp.analysis.sweep_reports import summarize_sweep, write_sweep_report
from mech_interp.cockpit_compare import (
    build_env_diff_rows,
    build_metric_rows,
    build_param_diff_rows,
)
from mech_interp.config import load_config
from mech_interp.experiments import load_experiment_specs
from mech_interp.experiments.registry import (
    ExperimentSpecValidationError,
    load_experiment_spec,
    load_experiment_specs_from_file,
)
from mech_interp.orchestration import (
    ActivationEstimate,
    ExperimentRunner,
    ExperimentRunQueue,
    ResourcePolicy,
)
from mech_interp.orchestration.iterate_loop import (
    _gather_proposal_paths,
    iterate_from_run,
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
DEFAULT_CORPUS_OUTPUT_DIR = Path("data/prompts")
console = Console()
DEFAULT_REPORT_OUTPUT = Path("artifacts/reports")
DEFAULT_PROPOSAL_OUTPUT = Path("experiments/proposed")


@app.command("config")
def show_config() -> None:
    """Print the resolved application config."""
    config = load_config()
    console.print_json(json.dumps(config.model_dump(mode="json"), indent=2))


@app.command("download-corpus")
def download_corpus_command(
    name: Annotated[
        str | None,
        typer.Option("--name", "-n", help="Corpus name to download (e.g. pile-1k, owt-1k)."),
    ] = None,
    max_documents: int = typer.Option(  # noqa: B008
        1000,
        "--max-documents",
        "-m",
        min=1,
        help="Maximum number of documents to fetch.",
    ),
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Output JSONL path. Default: data/prompts/<name>.jsonl"),  # noqa: E501
    ] = None,
    list_corpora: bool = typer.Option(  # noqa: B008
        False,
        "--list",
        help="Print all supported corpus names and exit.",
    ),
) -> None:
    """Download a real text corpus from HuggingFace for SAE training.

    Uses only huggingface_hub (already a transitive dep of transformer-lens).
    Pass --list to see supported corpus names.
    """
    from mech_interp.datasets.downloader import CORPORA, corpus_download_summary, download_corpus

    if list_corpora:
        table = Table(title="Supported Corpora")
        table.add_column("Name")
        table.add_column("HF Repo")
        table.add_column("Split")
        table.add_column("License")
        for descriptor in sorted(CORPORA.values(), key=lambda d: d.name):
            table.add_row(
                descriptor.name,
                descriptor.hf_repo,
                descriptor.hf_split,
                descriptor.license,
            )
        console.print(table)
        return

    if name is None:
        console.print("[red]--name is required (or pass --list to see options).[/red]")
        raise typer.Exit(code=1)

    dest = output or (DEFAULT_CORPUS_OUTPUT_DIR / f"{name}.jsonl")
    try:
        dest_path = download_corpus(name, max_documents=max_documents, dest=dest)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    # Reload documents for summary (file is small — at most max_documents lines)
    from mech_interp.datasets.corpus import load_text_corpus
    documents = load_text_corpus(dest_path, max_documents=None)
    summary = corpus_download_summary(dest_path, documents)
    console.print_json(json.dumps(summary, indent=2))
    console.print(f"[green]Corpus '{name}' written to {dest_path}[/green]")


@app.command("list-saes")
def list_saes_command() -> None:
    """Print all pretrained SAEs registered for download from HuggingFace."""
    from mech_interp.sae.registry import SAE_REGISTRY

    table = Table(title="Pretrained SAEs")
    table.add_column("Name")
    table.add_column("Model")
    table.add_column("Hook")
    table.add_column("Features")
    table.add_column("License")
    for descriptor in sorted(SAE_REGISTRY.values(), key=lambda d: d.name):
        table.add_row(
            descriptor.name,
            str(descriptor.config.get("model_name", "—")),
            str(descriptor.config.get("hook_site", "—")),
            str(descriptor.config.get("n_features", "—")),
            descriptor.license,
        )
    console.print(table)


@app.command("download-sae")
def download_sae_command(
    name: Annotated[str, typer.Option("--name", "-n", help="SAE name (see mech list-saes).")],
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Cache directory (default: data/saes/cache)."),
    ] = None,
) -> None:
    """Download an SAE's weights from HuggingFace into a local cache."""
    from mech_interp.sae.registry import download_sae

    dest_dir = output or Path("data/saes/cache")
    try:
        local_path = download_sae(name, dest_dir=dest_dir)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    console.print(f"[green]SAE '{name}' cached at {local_path}[/green]")


@app.command("analyze-sae")
def analyze_sae_command(
    name: Annotated[str, typer.Option("--name", "-n", help="SAE name (see mech list-saes).")],
    prompts: Annotated[
        Path,
        typer.Option("--prompts", "-p", help="JSONL or TXT corpus file."),
    ],
    max_tokens: int = typer.Option(2000, "--max-tokens", "-m", min=1),  # noqa: B008
    device: str = typer.Option("cpu", "--device"),  # noqa: B008
) -> None:
    """Load a pretrained SAE and run feature analysis on a prompt corpus.

    Captures activations from the SAE's registered model at the registered hook
    site, runs them through the SAE, and writes a new run record with
    feature_analysis.json so the result shows up in mech runs and the cockpit.
    """
    from mech_interp.backends.instrumented import TransformerLensBackend
    from mech_interp.datasets.corpus import load_text_corpus
    from mech_interp.sae import compute_feature_analysis
    from mech_interp.sae.registry import SAE_REGISTRY, load_pretrained_sae
    from mech_interp.storage import ArtifactStore, SQLiteResultStore
    from mech_interp.storage.artifacts import resolve_run_artifact_dir
    from mech_interp.types import ExperimentSpec

    if name not in SAE_REGISTRY:
        console.print(f"[red]Unknown SAE '{name}'. Run `mech list-saes`.[/red]")
        raise typer.Exit(code=1)
    descriptor = SAE_REGISTRY[name]

    documents = load_text_corpus(prompts, max_documents=None)
    if not documents:
        console.print(f"[red]Corpus at {prompts} is empty.[/red]")
        raise typer.Exit(code=1)

    config = load_config()
    db = SQLiteResultStore(config.project.database_path, config.project.artifact_dir)
    artifact_store = ArtifactStore(config.project.artifact_dir)
    spec = ExperimentSpec(
        name=f"analyze-sae-{name}",
        family="polysemanticity_sae",
        backend="transformerlens",
        description=f"Feature analysis of pretrained SAE {name} from {descriptor.hf_repo}.",
        parameters={
            "source": "pretrained_sae",
            "sae_name": name,
            "model": descriptor.config["model_name"],
            "hook_site": descriptor.config["hook_site"],
            "prompts_path": str(prompts),
            "max_tokens": max_tokens,
            "device": device,
        },
    )
    run = db.create_run(spec)
    artifact_store.write_json(
        run.id,
        "spec.json",
        {
            "name": spec.name,
            "family": spec.family,
            "backend": spec.backend,
            "description": spec.description,
            "parameters": spec.parameters,
        },
    )
    artifact_dir = resolve_run_artifact_dir(run)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    sae, sae_config = load_pretrained_sae(name, device=device)
    backend = TransformerLensBackend(
        model_name=str(descriptor.config["model_name"]),
        device=device,
    )
    backend.load()
    truncated_prompts = documents[: min(len(documents), max_tokens)]
    captured = backend.capture_activations(truncated_prompts, [descriptor.config["hook_site"]])
    activation = captured[descriptor.config["hook_site"]]

    import torch

    if activation.ndim == 3:
        batch, seq, d_model = activation.shape
        flat = activation.reshape(batch * seq, d_model).to(dtype=torch.float32)
        prompt_for_token = [truncated_prompts[i] for i in range(batch) for _ in range(seq)]
    else:
        flat = activation.to(dtype=torch.float32)
        prompt_for_token = truncated_prompts

    analysis = compute_feature_analysis(sae, flat.to(device), prompt_for_token)  # type: ignore[arg-type]
    analysis_path = artifact_dir / "feature_analysis.json"
    analysis_path.write_text(
        json.dumps(analysis.as_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    config_path = artifact_dir / "sae_config.json"
    config_path.write_text(
        json.dumps(sae_config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    console.print(
        f"[green]Analyzed SAE '{name}' over {flat.shape[0]} tokens; "
        f"{analysis.live_count}/{analysis.n_features} live features. "
        f"Run id: {run.id}. Artifacts: {artifact_dir}[/green]"
    )


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
def list_runs(
    limit: int = 20,
    include_archived: bool = typer.Option(
        False, "--include-archived", help="Include archived runs."
    ),
) -> None:
    """List recent experiment runs from the local SQLite store."""
    config = load_config()
    store = SQLiteResultStore(config.project.database_path, config.project.artifact_dir)
    table = Table(title="Recent Runs")
    table.add_column("Run ID")
    table.add_column("Spec")
    table.add_column("Family")
    table.add_column("Backend")
    table.add_column("Status")
    for run in store.list_runs(limit=limit, include_archived=include_archived):
        table.add_row(
            str(run.id),
            run.spec_name,
            run.family,
            run.backend,
            run.status.value,
        )
    console.print(table)


@app.command("archive-runs")
def archive_runs(
    before_run_id: Annotated[
        int, typer.Option("--before-run-id", min=1, help="Archive placeholder runs with id < N.")
    ],
    dry_run: bool = typer.Option(
        False, "--dry-run", help="List runs that would be archived without modifying anything."
    ),
) -> None:
    """Archive stale placeholder runs before a given run ID.

    Moves artifact directories to artifacts/archived/ and stamps archived_at
    in the database.  Only targets runs with family in {polysemanticity,
    superposition} — the families that fell back to SpecValidationExperiment
    before the placeholder gate was introduced.

    Run with --dry-run first to review what will be archived.
    """
    config = load_config()
    store = SQLiteResultStore(config.project.database_path, config.project.artifact_dir)
    runs = store.list_placeholder_runs_before(before_run_id)
    if not runs:
        console.print("No placeholder runs found matching the criteria.")
        return

    table = Table(title=f"{'[dry-run] ' if dry_run else ''}Runs to archive")
    table.add_column("Run ID")
    table.add_column("Family")
    table.add_column("Status")
    table.add_column("Artifact Dir")
    for run in runs:
        artifact_dir = config.project.artifact_dir / f"run-{run.id:06d}"
        table.add_row(
            str(run.id),
            run.family,
            run.status.value,
            str(artifact_dir) if artifact_dir.exists() else f"{artifact_dir} (missing)",
        )
    console.print(table)

    if dry_run:
        console.print(f"[yellow]Dry run: {len(runs)} run(s) would be archived.[/yellow]")
        return

    run_ids = [run.id for run in runs]
    archived = store.archive_runs(run_ids, config.project.artifact_dir)
    console.print(
        f"Archived {len(archived)} run(s). Artifact directories moved to artifacts/archived/."
    )


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


@app.command("iterate-from-run")
def iterate_from_run_command(
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
        typer.Option("--output", "-o", help="Root directory for generated specs and manifests."),
    ] = DEFAULT_PROPOSAL_OUTPUT,
    limit: int = typer.Option(5, min=1, help="Max follow-up specs per depth level."),
    max_depth: int = typer.Option(1, min=1, help="Maximum recursion depth."),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Generate proposals without executing them (equivalent to propose-from-run).",
    ),
) -> None:
    """Closed-loop iterate: generate proposals, execute them, and optionally recurse.

    With --dry-run, behaves identically to mech propose-from-run (writes specs, no execution).
    Without --dry-run, each generated spec is loaded and run through ExperimentRunner,
    then successful child runs are recursed into up to --max-depth levels.
    """
    config = load_config()
    result_store = SQLiteResultStore(config.project.database_path, config.project.artifact_dir)
    runner: ExperimentRunner | None = None
    if not dry_run:
        runner = ExperimentRunner(
            result_store=result_store,
            artifact_store=ArtifactStore(config.project.artifact_dir),
        )

    try:
        loop_result = iterate_from_run(
            family,
            artifact_dir,
            output,
            limit=limit,
            max_depth=max_depth,
            execute=not dry_run,
            runner=runner,
        )
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    proposals = _gather_proposal_paths(loop_result)
    table = Table(title=f"iterate-from-run  family={family}  depth={max_depth}")
    table.add_column("Name")
    table.add_column("Status")
    table.add_column("Run ID")
    table.add_column("Notes")
    for rec in proposals:
        table.add_row(
            rec["name"],
            rec["status"],
            str(rec["child_run_id"]) if rec["child_run_id"] is not None else "-",
            rec["notes"] or "-",
        )
    console.print(table)
    mode = "dry-run" if dry_run else f"executed {loop_result.total_runs} run(s)"
    console.print(
        f"Generated {len(proposals)} proposal(s); {mode}; "
        f"max depth reached: {loop_result.max_depth_reached}."
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
def inspect_run(
    run_id: Annotated[int, typer.Argument(min=1)],
    top_n: int = typer.Option(5, min=1, help="Top N items in family-specific summary."),
) -> None:
    """Print stored spec, config, result, and manifest data for a run.

    Includes a family-specific summary block (top features for SAE; top edges +
    faithfulness for ACDC; top recovery sites for circuit_patching) and a one-line
    environment provenance header (torch version, seed, uv.lock sha).
    """
    config = load_config()
    store = SQLiteResultStore(config.project.database_path, config.project.artifact_dir)
    spec = store.get_run_spec(run_id)
    if spec is None:
        console.print(f"[red]Run {run_id} was not found.[/red]")
        raise typer.Exit(code=1)

    artifact_dir = config.project.artifact_dir / f"run-{run_id:06d}"

    # Environment provenance header
    env = environment_provenance(artifact_dir)
    if env is not None:
        env_line = (
            f"[bold]env:[/bold] torch={env.get('torch_version') or '?'}  "
            f"seed={env.get('seed')}  "
            f"uv.lock={env.get('uv_lock_sha') or 'n/a'}  "
            f"python={env.get('python_version') or '?'}"
        )
        console.print(env_line)

    # Family-specific summary
    family = spec.get("family", "") if isinstance(spec, dict) else ""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        family_summary = inspect_run_family(family, artifact_dir, top_n=top_n)
    for w in caught:
        console.print(f"[yellow]warning:[/yellow] {w.message}")
    console.print_json(json.dumps(family_summary, indent=2, default=str))

    bundle = _run_bundle(run_id)
    console.print_json(json.dumps(bundle, indent=2, sort_keys=True))


@app.command("export-run")
def export_run(
    run_id: Annotated[int, typer.Argument(min=1)],
    output: Annotated[Path, typer.Option("--output", "-o", help="JSON file to write.")],
) -> None:
    """Export stored run data as a JSON bundle.

    Includes all known artifact files found in the run directory (artifact-agnostic
    walk), so SAE feature_analysis.json, circuit.json, edges.json, environment.json,
    and direction.safetensors.json are all captured automatically.
    """
    config = load_config()
    bundle = _run_bundle(run_id)
    artifact_dir = config.project.artifact_dir / f"run-{run_id:06d}"
    bundle["all_artifacts"] = _load_run_artifacts_from_dir(artifact_dir)
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


@app.command("label-features")
def label_features(
    run_id: Annotated[
        int | None,
        typer.Option("--run-id", min=1, help="SAE run ID to label."),
    ] = None,
    labeler_name: str = typer.Option(
        "heuristic",
        "--labeler",
        help="Labeler backend: heuristic, ollama, or anthropic.",
    ),
    ollama_model: str = typer.Option(
        "llama3.2:3b", "--ollama-model", help="Ollama model name."
    ),
    ollama_host: str = typer.Option(
        "http://localhost:11434", "--ollama-host", help="Ollama server base URL."
    ),
    max_features: int = typer.Option(
        50, "--max-features", min=1, help="Cap on features to label."
    ),
) -> None:
    """Label SAE features with human-readable descriptions using an LLM.

    Reads feature_analysis.json from the run artifact directory and writes
    feature_labels.json next to it.  Prints a sample of the top-5 labels.
    """
    config = load_config()
    store = SQLiteResultStore(config.project.database_path, config.project.artifact_dir)

    if run_id is None:
        runs = [r for r in store.list_runs(limit=50) if r.family == "polysemanticity_sae"]
        if not runs:
            console.print("[red]No polysemanticity_sae runs found. Use --run-id N.[/red]")
            raise typer.Exit(code=1)
        run_id = runs[0].id
        console.print(f"No --run-id given; using most recent SAE run: {run_id}")

    artifact_dir = config.project.artifact_dir / f"run-{run_id:06d}"
    if not artifact_dir.is_dir():
        console.print(f"[red]Artifact directory not found: {artifact_dir}[/red]")
        raise typer.Exit(code=1)

    chosen = labeler_name.lower()
    labeler: FeatureLabeler
    if chosen == "heuristic":
        labeler = HeuristicFeatureLabeler()
    elif chosen == "ollama":
        labeler = OllamaFeatureLabeler(host=ollama_host, model=ollama_model)
    elif chosen == "anthropic":
        try:
            labeler = AnthropicFeatureLabeler()
        except (ImportError, ValueError) as exc:
            console.print(f"[red]Cannot create AnthropicFeatureLabeler: {exc}[/red]")
            raise typer.Exit(code=1) from exc
    else:
        console.print(
            f"[red]Unknown labeler '{chosen}'. Choose: heuristic, ollama, anthropic.[/red]"
        )
        raise typer.Exit(code=1)

    try:
        labels = label_run_features(artifact_dir, labeler, max_features=max_features)
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    output_path = artifact_dir / "feature_labels.json"
    console.print(f"Labeled {len(labels)} features -> {output_path}")

    sample = sorted(labels.items(), key=lambda kv: kv[0])[:5]
    if sample:
        table = Table(title=f"Top-5 Feature Labels (run {run_id})")
        table.add_column("Feature Index")
        table.add_column("Label")
        for idx, lbl in sample:
            table.add_row(str(idx), lbl)
        console.print(table)


@app.command("compare-runs")
def compare_runs_cli(
    left: Annotated[int, typer.Option("--left", min=1, help="Run ID for the A side.")],
    right: Annotated[int, typer.Option("--right", min=1, help="Run ID for the B side.")],
) -> None:
    """Print a side-by-side diff of two same-family runs as a Rich table."""
    config = load_config()
    store = SQLiteResultStore(config.project.database_path, config.project.artifact_dir)

    all_runs = store.list_runs(limit=500)
    run_a = next((r for r in all_runs if r.id == left), None)
    run_b = next((r for r in all_runs if r.id == right), None)

    if run_a is None:
        console.print(f"[red]Run {left} not found.[/red]")
        raise typer.Exit(code=1)
    if run_b is None:
        console.print(f"[red]Run {right} not found.[/red]")
        raise typer.Exit(code=1)
    if run_a.family != run_b.family:
        console.print(
            f"[red]Runs {left} and {right} have different families "
            f"('{run_a.family}' vs '{run_b.family}'). "
            "Comparison requires matching families.[/red]"
        )
        raise typer.Exit(code=1)

    result_a = store.get_result(left)
    result_b = store.get_result(right)
    spec_a = store.get_run_spec(left) or {}
    spec_b = store.get_run_spec(right) or {}
    metrics_a: dict[str, float] = result_a.metrics if result_a else {}
    metrics_b: dict[str, float] = result_b.metrics if result_b else {}
    params_a: dict[str, Any] = spec_a.get("parameters", {}) if isinstance(spec_a, dict) else {}
    params_b: dict[str, Any] = spec_b.get("parameters", {}) if isinstance(spec_b, dict) else {}

    # Header
    console.print(
        f"[bold]Compare Run {left} (A)  vs  Run {right} (B)[/bold]  family={run_a.family}"
    )
    console.print(f"  A: {run_a.spec_name}  [{run_a.status.value}]")
    console.print(f"  B: {run_b.spec_name}  [{run_b.status.value}]")
    console.print()

    # Metrics table
    metric_rows = build_metric_rows(metrics_a, metrics_b)
    if metric_rows:
        tbl = Table(title="Metrics")
        tbl.add_column("Metric")
        tbl.add_column(f"Run {left} (A)")
        tbl.add_column(f"Run {right} (B)")
        tbl.add_column("Diff")
        for row in metric_rows:
            va = f"{row['val_a']:.6g}" if row["val_a"] is not None else "—"
            vb = f"{row['val_b']:.6g}" if row["val_b"] is not None else "—"
            if row["pct_diff"] is not None:
                sign = "+" if row["pct_diff"] > 0 else ""
                diff_str = f"{sign}{row['pct_diff'] * 100:.1f}%"
                if row["highlight"]:
                    if row["badge_class"] == "better":
                        colour = "green"
                    elif row["badge_class"] == "worse":
                        colour = "red"
                    else:
                        colour = "yellow"
                    diff_str = f"[{colour}]{diff_str}[/{colour}]"
            else:
                diff_str = "—"
            tbl.add_row(row["key"], va, vb, diff_str)
        console.print(tbl)
    else:
        console.print("[dim]No metrics recorded for either run.[/dim]")

    # Parameters diff
    param_rows = build_param_diff_rows(params_a, params_b)
    if param_rows:
        ptbl = Table(title="Changed Parameters")
        ptbl.add_column("Parameter")
        ptbl.add_column(f"Run {left} (A)")
        ptbl.add_column(f"Run {right} (B)")
        for row in param_rows:
            ptbl.add_row(
                row["key"],
                str(row["val_a"]) if row["val_a"] is not None else "—",
                str(row["val_b"]) if row["val_b"] is not None else "—",
            )
        console.print(ptbl)

    # Environment diff
    from mech_interp.cockpit import _read_environment
    env_a = _read_environment(left, result_a, config.project.artifact_dir)
    env_b = _read_environment(right, result_b, config.project.artifact_dir)
    env_rows = build_env_diff_rows(env_a, env_b)
    diff_env = [r for r in env_rows if r["differs"]]
    if diff_env:
        etbl = Table(title="Environment Drift (differing keys)")
        etbl.add_column("Key")
        etbl.add_column(f"Run {left} (A)")
        etbl.add_column(f"Run {right} (B)")
        for row in diff_env:
            etbl.add_row(
                row["key"],
                str(row["val_a"]) if row["val_a"] is not None else "—",
                str(row["val_b"]) if row["val_b"] is not None else "—",
            )
        console.print(etbl)


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
    candidate_paths: list[Path] = []
    if result_payload is not None:
        artifacts = result_payload.get("artifacts", {})
        if isinstance(artifacts, dict):
            manifest_value = artifacts.get("manifest")
            if isinstance(manifest_value, str):
                candidate_paths.append(Path(manifest_value))
    candidate_paths.append(artifact_dir / f"run-{run_id:06d}" / "manifest.json")

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


# ---------------------------------------------------------------------------
# mech sweep
# ---------------------------------------------------------------------------


def _parse_axis_value(raw: str) -> Any:
    """Try to parse *raw* as int, then float, then return as-is (str)."""
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    # List literal like "[0]" or "[0,1,2]"
    stripped = raw.strip()
    if stripped.startswith("[") and stripped.endswith("]"):
        inner = stripped[1:-1].strip()
        if not inner:
            return []
        return [_parse_axis_value(item.strip()) for item in inner.split(",")]
    return raw


def _parse_axis(spec: str) -> tuple[str, list[Any]]:
    """Parse ``name=v1,v2,v3`` into ``(name, [v1, v2, v3])``.

    Values that look like integer or float literals are converted; strings are
    kept as-is.  The special case ``name=[0],[1],[2]`` (list-valued axes) is
    handled by treating comma-separated bracketed items as atomic tokens.
    """
    if "=" not in spec:
        raise ValueError(f"Invalid --axis value {spec!r}: expected 'name=v1,v2,...'")
    name, _, raw_values = spec.partition("=")
    name = name.strip()
    if not name:
        raise ValueError(f"Invalid --axis value {spec!r}: axis name is empty")

    # Tokenise respecting bracket nesting so "[0],[1]" becomes ["[0]","[1]"]
    tokens: list[str] = []
    depth = 0
    current: list[str] = []
    for ch in raw_values:
        if ch == "[":
            depth += 1
            current.append(ch)
        elif ch == "]":
            depth -= 1
            current.append(ch)
        elif ch == "," and depth == 0:
            tokens.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
    if current:
        tokens.append("".join(current).strip())

    values = [_parse_axis_value(t) for t in tokens if t]
    if not values:
        raise ValueError(f"Invalid --axis value {spec!r}: no values after '='")
    return name, values


@app.command("sweep")
def sweep_command(
    base: Annotated[
        Path,
        typer.Option("--base", "-b", help="Base experiment YAML to sweep over."),
    ],
    axis: Annotated[
        list[str] | None,
        typer.Option(
            "--axis",
            "-a",
            help="Axis spec: 'name=v1,v2,v3'.  Repeat for multiple axes.",
        ),
    ] = None,
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Path for the generated matrix YAML."),
    ] = Path("experiments/sweeps/sweep.yaml"),
    execute: bool = typer.Option(  # noqa: B008
        False,
        "--execute",
        help="Run the generated sweep specs immediately after writing the YAML.",
    ),
) -> None:
    """Generate (and optionally run) a matrix sweep from a base experiment spec.

    Parses --axis name=v1,v2,v3 arguments, builds a matrix: YAML on top of the
    base spec, writes it to --output, then optionally executes all generated
    specs via ExperimentRunner.
    """
    if not axis:
        console.print("[red]At least one --axis is required.[/red]")
        raise typer.Exit(code=1)

    # Validate base spec loads cleanly
    try:
        base_spec = load_experiment_spec(base)
    except ExperimentSpecValidationError as exc:
        console.print(f"[red]Could not load base spec from {base}: {exc}[/red]")
        raise typer.Exit(code=1) from exc

    # Parse axes
    parsed_axes: dict[str, list[Any]] = {}
    for ax in axis:
        try:
            name, values = _parse_axis(ax)
        except ValueError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(code=1) from exc
        parsed_axes[name] = values

    # Re-load raw YAML to preserve exact structure (comments aside)
    import yaml as _yaml  # already a dep

    with base.open("r", encoding="utf-8") as fh:
        base_raw: dict[str, Any] = _yaml.safe_load(fh) or {}

    # Strip any existing matrix block from the base and inject ours
    base_raw.pop("matrix", None)
    base_raw["matrix"] = parsed_axes

    # Write output YAML
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as fh:
        _yaml.dump(base_raw, fh, default_flow_style=False, allow_unicode=True, sort_keys=False)

    # Validate round-trip
    try:
        generated_specs = load_experiment_specs_from_file(output)
    except ExperimentSpecValidationError as exc:
        console.print(f"[red]Generated matrix YAML is invalid: {exc}[/red]")
        raise typer.Exit(code=1) from exc

    n = len(generated_specs)
    console.print(
        f"Wrote sweep YAML to [bold]{output}[/bold] ({n} spec(s) from "
        f"{len(parsed_axes)} axis/axes over base '{base_spec.name}')."
    )

    if not execute:
        return

    # Execute
    config = load_config()
    result_store = SQLiteResultStore(config.project.database_path, config.project.artifact_dir)
    runner = ExperimentRunner(
        result_store=result_store,
        artifact_store=ArtifactStore(config.project.artifact_dir),
    )

    results = []
    for spec in generated_specs:
        console.print(f"  Running [cyan]{spec.name}[/cyan] …")
        result = runner.run(spec)
        results.append(result)

    report = summarize_sweep(generated_specs, results)

    table = Table(title="Sweep Results")
    table.add_column("Spec")
    table.add_column("Status")
    for axis_name in sorted(report.axes.keys()):
        table.add_column(axis_name)
    table.add_column("Metrics")

    for row in report.runs:
        cells = [row["name"], row["status"]]
        for axis_name in sorted(report.axes.keys()):
            cells.append(str(row["axis_values"].get(axis_name, "")))
        cells.append(
            ", ".join(f"{k}={v}" for k, v in sorted(row["metrics"].items()))
            or "-"
        )
        table.add_row(*cells)

    console.print(table)


# ---------------------------------------------------------------------------
# mech sweep-report
# ---------------------------------------------------------------------------


@app.command("sweep-report")
def sweep_report_command(
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir", "-o", help="Directory where report files are written."),
    ] = Path("artifacts/reports"),
    prefix: Annotated[
        str,
        typer.Option(
            "--prefix",
            "-p",
            help="Only include run artifact dirs whose name starts with this prefix.",
        ),
    ] = "",
) -> None:
    """Walk recent run artifact dirs and write sweep_report.json + sweep_report.md.

    Scans the artifact directory from config for run-XXXXXX subdirectories that
    contain a spec.json with 'matrix_axes' in parameters.  Pairs them with their
    result.json and produces a SweepReport.
    """
    config = load_config()
    artifact_root = config.project.artifact_dir

    from mech_interp.types import (  # noqa: PLC0415
        ExperimentResult,
        ExperimentSpec,
        RunStatus,
    )

    specs: list[ExperimentSpec] = []
    results: list[ExperimentResult] = []

    run_dirs = sorted(artifact_root.glob("run-??????"))
    for run_dir in run_dirs:
        spec_path = run_dir / "spec.json"
        result_path = run_dir / "result.json"
        if not spec_path.exists():
            continue

        spec_data: dict[str, Any] = json.loads(spec_path.read_text(encoding="utf-8"))
        params = spec_data.get("parameters") or {}
        if "matrix_axes" not in params:
            continue

        spec_name: str = spec_data.get("name", run_dir.name)
        if prefix and not spec_name.startswith(prefix):
            continue

        spec = ExperimentSpec(
            name=spec_name,
            family=spec_data.get("family", ""),
            backend=spec_data.get("backend", ""),
            description=spec_data.get("description", ""),
            parameters=params,
        )
        specs.append(spec)

        if result_path.exists():
            result_data: dict[str, Any] = json.loads(result_path.read_text(encoding="utf-8"))
            try:
                status = RunStatus(result_data.get("status", "planned"))
            except ValueError:
                status = RunStatus.FAILED
            result = ExperimentResult(
                run_id=int(result_data.get("run_id", 0)),
                status=status,
                metrics=result_data.get("metrics") or {},
                notes=result_data.get("notes") or "",
            )
            results.append(result)

    if not specs:
        console.print(
            f"[yellow]No sweep run dirs found under {artifact_root}"
            + (f" matching prefix '{prefix}'" if prefix else "")
            + ".[/yellow]"
        )
        raise typer.Exit(code=0)

    report = summarize_sweep(specs, results)
    json_path, md_path = write_sweep_report(report, output_dir)
    console.print(
        f"Sweep report: {len(specs)} run(s), {len(report.axes)} axis/axes.\n"
        f"  JSON: [bold]{json_path}[/bold]\n"
        f"  MD:   [bold]{md_path}[/bold]"
    )


@app.command("calibrate-tuned-lens")
def calibrate_tuned_lens(
    model: Annotated[
        str, typer.Option("--model", help="TransformerLens model name")
    ] = "gpt2-small",
    prompts_path: Annotated[
        str, typer.Option("--prompts", help="JSONL file of prompts")
    ] = "data/prompts/factual.jsonl",
    epochs: Annotated[
        int, typer.Option("--epochs", help="Training epochs")
    ] = 50,
    lr: Annotated[
        float, typer.Option("--lr", help="Adam learning rate")
    ] = 1e-3,
    output: Annotated[
        str, typer.Option("--output", help="Output safetensors path")
    ] = "data/tuned-lens/gpt2-small.safetensors",
    device: Annotated[
        str, typer.Option("--device", help="Torch device")
    ] = "cpu",
    seed: Annotated[
        int, typer.Option("--seed", help="Random seed")
    ] = 42,
) -> None:
    """Train per-layer tuned-lens affine transforms and save to a safetensors file.

    Each layer transform is initialised to identity and optimised with Adam to
    minimise the KL divergence between the projected distribution at that layer
    and the final-layer distribution (soft labels).
    """
    try:
        from mech_interp.analysis.tuned_lens_calibration import (
            load_prompts_from_jsonl,
            save_tuned_lens,
            train_tuned_lens,
        )
        from mech_interp.backends import create_instrumented_backend
    except ImportError as exc:
        console.print(f"[red]Missing dependencies: {exc}[/red]")
        console.print(
            "Run [bold]uv sync --extra interp[/bold] to install torch + transformer-lens."
        )
        raise typer.Exit(code=1) from exc

    prompts_file = Path(prompts_path)
    if not prompts_file.is_file():
        console.print(f"[red]Prompts file not found: {prompts_file}[/red]")
        raise typer.Exit(code=1)

    console.print(f"Loading prompts from [bold]{prompts_file}[/bold]...")
    training_prompts = load_prompts_from_jsonl(prompts_file)
    if not training_prompts:
        console.print("[red]No prompts found in the JSONL file.[/red]")
        raise typer.Exit(code=1)
    console.print(f"  {len(training_prompts)} prompt(s) loaded.")

    console.print(f"Loading model [bold]{model}[/bold] on {device}...")
    backend = create_instrumented_backend(
        "transformerlens", {"model_name": model, "device": device}
    )
    backend.load()
    tl_model = getattr(backend, "model", None)
    if tl_model is None:
        console.print("[red]Failed to load model.[/red]")
        raise typer.Exit(code=1)

    console.print(f"Training tuned-lens transforms ({epochs} epochs, lr={lr}, seed={seed})...")
    transforms = train_tuned_lens(
        tl_model, training_prompts, epochs=epochs, lr=lr, seed=seed, device=device
    )

    output_path = Path(output)
    saved = save_tuned_lens(transforms, output_path)
    console.print(f"[green]Saved tuned-lens transforms to[/green] [bold]{saved}[/bold]")
    console.print(f"  Layers trained: {sorted(transforms.keys())}")
