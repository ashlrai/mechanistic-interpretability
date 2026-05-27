# CLI Reference

The `mech` CLI is the primary interface to the platform. All commands are implemented in
`src/mech_interp/cli.py` via [Typer](https://typer.tiangolo.com/).

## Quick reference

```bash
mech --help                     # list all commands
mech <command> --help           # options for a specific command
```

## Core commands

### `mech run`

Run a single experiment spec.

```bash
mech run --name <spec-name>
mech run --spec experiments/polysemanticity.yaml
mech run --name polysemanticity-sae-layer0 --dry-run
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--name` | — | Experiment name (resolves from `experiments/`) |
| `--spec` | — | Path to a YAML spec |
| `--dry-run` | false | Validate spec without executing |
| `--force` | false | Re-run even if a result exists |

### `mech sweep`

Generate and optionally execute a parameter sweep.

```bash
mech sweep \
  --base experiments/polysemanticity.yaml \
  --axis "parameters.seed=1,2,3,4,5" \
  --output experiments/sweeps/my_sweep.yaml \
  --execute
```

Options:

| Flag | Description |
|------|-------------|
| `--base` | Base YAML spec to sweep over |
| `--axis` | Axis spec in `key=v1,v2,...` format; repeatable for multi-axis sweeps |
| `--output` | Write generated sweep YAML to this path |
| `--execute` | Run all generated specs immediately |

### `mech validate`

Run a fast smoke test to verify the platform is working.

```bash
mech validate
```

Checks: TransformerLens forward pass, storage layer, experiment registry.

### `mech cockpit`

Open the local web cockpit (FastAPI + Jinja2) for browsing runs, artifacts, and reports.

```bash
mech cockpit
mech cockpit --port 8080
```

### `mech demo`

Launch the Gradio interactive demo.

```bash
mech demo
mech demo --port 7861 --share
```

Requires: `pip install mech-interpretability[gradio]`

---

## Analysis commands

### `mech analyze-sae-stability`

Compute pairwise seed stability across an SAE sweep.

```bash
mech analyze-sae-stability \
  --sweep experiments/sweeps/sae_seed_stability.yaml \
  --output artifacts/stability_report.json \
  --live-only
```

Options:

| Flag | Description |
|------|-------------|
| `--sweep` | Path to sweep YAML with multiple runs |
| `--output` | Write JSON stability report to this path |
| `--live-only` | Restrict matching to live (non-dead) features only |

### `mech analyze-feature-splits`

Compute feature splitting analysis between a parent and child SAE run.

```bash
mech analyze-feature-splits --parent-run 44 --child-run 46
```

### `mech analyze-sae`

Analyze a single SAE run: dead features, decoder norms, top activating prompts.

```bash
mech analyze-sae --run-id 51
```

### `mech sae-scale-report`

Generate a detailed scale report for a large SAE run.

```bash
mech sae-scale-report --run-id 51
```

### `mech label-features`

Auto-label SAE features using a heuristic or LLM labeler.

```bash
mech label-features --run-id 51 --labeler heuristic --max-features 20
mech label-features --run-id 51 --labeler anthropic
mech label-features --run-id 51 --labeler ollama
```

### `mech compare-runs`

Compare two runs side-by-side.

```bash
mech compare-runs --run-a 44 --run-b 46
```

---

## Data commands

### `mech download-corpus`

Download a named corpus to a local JSONL file.

```bash
mech download-corpus --name openwebtext --max-documents 100 --output data/prompts/openwebtext.jsonl
mech download-corpus --name pile-1k --max-documents 1000 --output data/prompts/pile-1k.jsonl
```

### `mech list-saes`

List SAEs available in the SAE registry (pre-trained weights).

```bash
mech list-saes
```

### `mech download-sae`

Download a pre-trained SAE from the registry.

```bash
mech download-sae --model gpt2 --layer 0
```

---

## Run management commands

### `mech runs`

List all stored runs.

```bash
mech runs
mech runs --limit 20 --family polysemanticity_sae
```

### `mech inspect-run`

Show full details for a specific run.

```bash
mech inspect-run --run-id 51
```

### `mech export-run`

Export a run's artifacts as a ZIP.

```bash
mech export-run --run-id 51 --output my_run.zip
```

### `mech archive-runs`

Archive (compress) old run artifacts.

```bash
mech archive-runs --before-days 30
```

### `mech query-runs`

SQL-style query over the run database.

```bash
mech query-runs --filter "family=polysemanticity_sae" --metric explained_variance
```

---

## Orchestration commands

### `mech propose-followups`

Use the AI planner to propose follow-up experiment specs from recent runs.

```bash
mech propose-followups
```

### `mech propose-from-run`

Propose follow-ups from a specific run's results.

```bash
mech propose-from-run --run-id 51
```

### `mech iterate`

Run the full agentic iterate loop: propose, execute, analyze, repeat.

```bash
mech iterate --steps 5
```

### `mech iterate-from-run`

Start an iterate loop seeded from an existing run.

```bash
mech iterate-from-run --run-id 51 --steps 3
```

### `mech preflight`

Validate that a run spec is feasible given current memory and hardware.

```bash
mech preflight --spec experiments/polysemanticity_sae_gpt2_medium.yaml
```

---

## Utility commands

### `mech config`

Show the current platform configuration.

```bash
mech config
```

### `mech providers`

Check which model providers are available.

```bash
mech providers
```

### `mech init-store`

Initialize the SQLite artifact store at `~/.mech_interp/runs.db`.

```bash
mech init-store
```

### `mech estimate-activations`

Estimate activation cache size for a given spec before running.

```bash
mech estimate-activations --spec experiments/polysemanticity_sae_gpt2_medium.yaml
```
