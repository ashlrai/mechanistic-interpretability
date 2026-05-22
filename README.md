# Local Mechanistic Interpretability Platform

This project is a local, modular research platform for running large batches of controlled
mechanistic interpretability experiments on local language models. The goal is to discover how
models compute by targeting polysemanticity, superposition, activation-level interventions, and
circuit-level behavior.

The platform is designed for an Apple Silicon MacBook Pro with 128 GB RAM. It uses local model
execution, local storage, and local orchestration. No cloud model APIs are part of the core design.

## Research Vision

The system should make it easy to run thousands of repeatable experiments across prompts, features,
layers, heads, activation sites, and intervention strategies. Each run should be:

- Controlled: explicit model, prompt set, seed, backend, and experiment config.
- Inspectable: activations, metrics, artifacts, and notes are stored locally.
- Reproducible: configs and run metadata are saved before experiment execution.
- Modular: experiments can be added without rewriting model loading or storage code.
- Agent-friendly: a planning or coding agent can inspect specs, run batches, and summarize results.

## Architecture

There are two different kinds of local model access:

1. **Instrumented backends** expose internals such as activations, hooks, and interventions. These
   are required for mechanistic interpretability. TransformerLens is the first-class backend in this
   scaffold, with nnsight and MLX-native backends reserved behind the same interface.
2. **Generation providers** expose black-box text generation. Ollama and LM Studio are useful for
   local prompting, baselines, and dataset generation, but their OpenAI-compatible APIs do not expose
   circuit internals. They are intentionally separate from interpretability backends.

Core packages:

- `mech_interp.backends`: adapters for instrumented models.
- `mech_interp.datasets`: prompt dataset loaders, normalized records, and reproducibility hashes.
- `mech_interp.providers`: adapters for black-box local generation providers.
- `mech_interp.experiments`: experiment specs, registry, and initial experiment families.
- `mech_interp.orchestration`: run planning and resource policy for local batches.
- `mech_interp.storage`: SQLite run metadata plus filesystem artifact locations.
- `mech_interp.config`: YAML configuration loading.

## Project Layout

```text
.
├── configs/                 # Local backend/model/experiment settings
├── experiments/             # Runnable experiment spec files
├── artifacts/               # Generated run metadata, tensors, reports, and logs
├── data/                    # Local datasets and prompt corpora
├── notebooks/               # Exploratory analysis notebooks
├── scripts/                 # Operational helper scripts
├── src/mech_interp/         # Python package
└── tests/                   # Smoke and unit tests
```

## Setup

Install `uv` if needed, then create the local development environment:

```bash
uv sync --group dev
```

For interpretability backends:

```bash
uv sync --group dev --extra interp
```

For Apple Silicon MLX support:

```bash
uv sync --group dev --extra interp --extra apple
```

Copy the environment example if you want shell-level defaults:

```bash
cp .env.example .env
```

## Local Model Providers

Ollama default endpoint:

```text
http://localhost:11434
```

LM Studio default OpenAI-compatible endpoint:

```text
http://localhost:1234/v1
```

These providers are useful for prompt generation, black-box comparisons, and evaluation. For
activation patching, probing, and circuit discovery, use an instrumented backend that loads model
weights directly.

## Running Checks

```bash
uv run --group dev python -m pytest
uv run --group dev ruff check .
uv run --group dev mypy src tests
uv run --group dev mech validate
```

Or run the local check script:

```bash
bash scripts/check.sh
```

List registered experiment families:

```bash
uv run --group dev mech experiments
```

Validate experiment YAML without creating runs:

```bash
uv run --group dev mech validate
```

Check local provider reachability:

```bash
uv run --group dev mech providers --timeout 2
```

Estimate activation-cache memory for a planned batch:

```bash
uv run --group dev mech estimate-activations \
  --batch-size 4 \
  --sequence-length 128 \
  --hidden-size 768 \
  --hook-count 12
```

Initialize the local result store:

```bash
uv run --group dev mech init-store
```

Inspect the active config:

```bash
uv run --group dev mech config
```

Run every placeholder experiment spec through the local orchestration spine:

```bash
uv run --group dev mech run
```

List recent experiment runs:

```bash
uv run --group dev mech runs
```

Summarize recent runs:

```bash
uv run --group dev mech summarize-runs --limit 100
```

Inspect or export a run bundle:

```bash
uv run --group dev mech inspect-run 1
uv run --group dev mech export-run 1 --output artifacts/run-1-export.json
```

Plan and claim resumable queue work:

```bash
uv run --group dev mech queue plan
uv run --group dev mech queue next
uv run --group dev mech queue list
```

Or run the local smoke script:

```bash
bash scripts/smoke.sh
```

## Current Execution Flow

The current runner is intentionally lightweight. It validates and persists experiment specs, creates
SQLite run records, writes per-run artifacts, and records placeholder metrics. That gives the
project a stable execution spine before model-backed TransformerLens experiments are enabled.

```text
YAML spec -> registry -> runner -> SQLite run -> per-run artifacts -> SQLite result
```

Specs can opt into the TransformerLens smoke runner by setting `parameters.runner` to
`transformerlens_smoke`. That path captures selected activation sites when the optional
TransformerLens dependencies are installed, and ordinary tests use fakes so CI never downloads
models.

Specs can opt into real activation summary capture by setting `parameters.runner` to
`activation_capture`. The runner calls the configured instrumented backend, captures requested hook
sites, and writes `activation_summary.json` with shape, dtype, mean, std, max, and sparsity where
those summaries are available.

## Prompt Datasets

Prompt datasets live under `data/prompts/` and can be loaded with:

```python
from mech_interp.datasets import load_prompt_dataset

dataset = load_prompt_dataset("data/prompts/factual.jsonl")
print(dataset.sha256)
print(dataset.prompts)
```

Supported formats:

- JSONL: one object per line with a required `prompt`, optional `id`, optional `metadata`, and any
  extra fields folded into metadata.
- Plain text: one prompt per non-empty line, with `#` comment lines ignored.

`PromptRecord.sha256` and `PromptDataset.sha256` are computed from normalized record content, so
hashes are stable across JSON field ordering and formatting changes. Experiment specs can reference
datasets without runner changes by storing paths and optional expected hashes in `parameters`:

```yaml
parameters:
  dataset_path: data/prompts/factual.jsonl
  dataset_sha256: "<expected digest>"
```

An optional example lives at `examples/transformerlens_smoke.yaml`. It is not in the default
`experiments/` directory because it requires the optional TransformerLens dependency and may trigger
a local model download:

```bash
uv sync --group dev --extra interp
uv run --group dev --extra interp mech run \
  --directory examples \
  --name transformerlens-activation-smoke
```

There is also an activation-capture example:

```bash
uv run --group dev --extra interp mech run \
  --directory examples \
  --name activation-capture-smoke
```

## Local-First Verification

This repo intentionally does not use GitHub Actions. Local checks are the source of truth:

```bash
bash scripts/check.sh
```

## Experiment Roadmap

The first implementation modules should be built in this order:

1. TransformerLens backend: model loading, activation cache capture, hook registration, and
   intervention execution.
2. SQLite run tracking: full experiment lifecycle, metrics, artifact manifests, and summaries.
3. Polysemanticity probes: feature activation sweeps across curated prompt sets.
4. Superposition sweeps: sparse feature and residual stream analyses.
5. Circuit experiments: activation patching, causal tracing, attention head scans, and MLP path
   attribution.
6. Agent orchestration: generate experiment matrices, schedule local batches, summarize failures,
   and write research reports.

## Resource Strategy

The scaffold assumes a 128 GB RAM Apple Silicon machine. The orchestration layer should eventually
plan batches around:

- model size and precision,
- activation cache size,
- number of prompts and token length,
- layer/head/site sweep dimensions,
- artifact retention policy,
- whether experiments can stream results instead of retaining all activations.

The default config is conservative. Increase batch sizes only after measuring memory pressure for a
specific model and experiment family.
