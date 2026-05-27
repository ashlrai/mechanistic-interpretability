# Mechanistic Interpretability Platform

**[Documentation](https://ashlrai.github.io/mechanistic-interpretability)** | **[Investigations](docs/investigations/)** | **[Publications](docs/publications/)**

Local mech-interp research at the speed of curiosity. 15 experiment families,
SQLite-backed run history, closed-loop agentic followup, Gradio UI, HuggingFace
backend for any AutoModelForCausalLM, and a 19-second quickstart.

## Quickstart

```bash
uv sync --group dev --extra interp
mech demo
```

## What it lets you do

- **Reproduce canonical circuits**: `mech run --name acdc-edge-ioi-gpt2-small` runs
  edge-level ACDC on 30 IOI prompts and recovers the Wang et al. name-mover heads.
- **Audit refusal across models**: `mech audit-refusal` runs the full 4-stage
  pipeline (direction extraction → circuit patching → causal scrubbing → ablation)
  on any instruct model.
- **Load pretrained SAEs**: `mech list-saes` then `mech download-sae` pulls
  HuggingFace-hosted sparse autoencoders ready for feature analysis.
- **Apply steering vectors**: `mech list-steering` + `mech apply-steering --vector <name>`
  runs baseline-vs-steered generation side-by-side.
- **Use any HF model**: `backend: huggingface` in any YAML loads any
  `AutoModelForCausalLM` with the same hook-site API as TransformerLens.
- **Explore results**: `mech cockpit` serves a local FastAPI/HTMX dashboard;
  `mech gradio` launches an interactive 4-panel Gradio demo.

Discover all 37 commands: `mech help`

---

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

## Getting started

The fastest path to a real mech-interp result is `mech demo`.  It runs three
narrative-coherent experiments on **gpt2-small** (already cached for most
TransformerLens users), prints a Rich-rendered summary, and saves a 3-panel
figure — all in under 5 minutes on a MacBook Pro:

```bash
uv sync --group dev --extra interp
uv run --group dev --extra interp mech demo
```

Example output:

```
mech demo — running 3 experiments on gpt2-small …
Experiments complete in 19.2s

╭──────────────────── mech demo — gpt2-small factual recall ────────────────────╮
│  Experiment               Finding                                    Value     │
│  Direct Logit Attribution Top writing component: L0_mlp            +2.841     │
│  Logit Lens               rank drops over 12 layers (never top-5)  rank 37    │
│  Circuit Patching         Top causal site: L8·resid_pre            93% recov. │
╰───────────────────────────────────────────────────────────────────────────────╯

What just happened:
  1. DLA decomposed every component's contribution to the final logit in a single forward pass.
  2. Logit Lens revealed how the model's best guess evolves layer by layer.
  3. Circuit Patching causally verified the top DLA component via activation patching.
  4. Together: something writes it, somewhere it commits, patching confirms causality.
  5. All results are deterministic (seed=42) — re-run to confirm.

Full walkthrough: notebooks/05_research_walkthrough.ipynb
Saved chart:      artifacts/demo/<timestamp>/summary.png
```

Pass `--output-dir` to control where artifacts land, or `--skip-chart` to skip
the matplotlib figure:

```bash
uv run --group dev --extra interp mech demo --output-dir /tmp/my-demo
uv run --group dev --extra interp mech demo --skip-chart
```

---

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

The runner validates and persists experiment specs, seeds torch/numpy/random deterministically,
writes an `environment.json` fingerprint (library versions, `uv.lock` SHA, seed, model name) per
run, dispatches to the registered family, and persists artifacts + results into SQLite.

```text
YAML spec -> registry -> runner -> seed + env fingerprint -> family -> SQLite run + artifacts
```

If a YAML targets a family with no registered implementation, the runner now raises
`FamilyNotImplementedError` rather than silently falling back to a placeholder runner. To opt back
into the placeholder for scratch runs, set `MECH_INTERP_ALLOW_PLACEHOLDER=1` in the environment.

### Real experiment families

- **circuit_patching** — clean/corrupted prompt-pair patching with recovery fractions and control
  hook sites. Verified on gpt2-small (`experiments/circuit_patching.yaml`).
- **polysemanticity_sae** — Top-K sparse autoencoder (Gao et al., 2024) on residual-stream
  activations. Trains a feature dictionary, ranks top-activating prompts per feature, writes
  `sae_weights.safetensors` + `feature_analysis.json` (`experiments/polysemanticity.yaml`).
- **acdc_lite** — node-level automatic circuit discovery (Conmy et al., 2023). Scores every
  (layer, head) attention node and (layer, MLP) node by ablation impact, prunes below a threshold,
  reports faithfulness and a GraphViz dot (`experiments/acdc_lite.yaml`).
- **acdc_edge** — full edge-level ACDC (Conmy et al., 2023). Scores individual
  (source-hook → dest-hook) edges rather than whole nodes; produces a sparser, more precise
  circuit graph (`experiments/acdc_edge.yaml`).
- **refusal_direction** — extracts the principal refusal direction from an instruct model's
  residual stream using contrastive harmful/harmless prompt pairs, then measures how much ablating
  that direction recovers refusal behaviour (Arditi et al., 2024).
- **cross_model_representation_probe** — ridge regression across model pairs.
- **activation_capture** / **transformerlens_smoke** runners (via `parameters.runner`).

### Agentic loop

After a run completes, family-specific `ProposalGenerator`s emit follow-up specs:

- SAE run → `mech propose-from-run --family polysemanticity_sae --artifact-dir <run>` generates
  `circuit_patching` probes that test the top features for causal weight.
- ACDC run → `mech propose-from-run --family acdc_lite --artifact-dir <run>` generates an
  `activation_capture` spec across surviving nodes.

Every generated spec is round-tripped through the YAML validator before being written to disk, so
malformed proposals fail immediately rather than at queue time.

To close the loop end-to-end (propose + execute + recurse):

```bash
uv run --group dev mech iterate-from-run --family polysemanticity_sae \
  --artifact-dir artifacts/run-000042 --max-depth 2
```

Use `--dry-run` to generate specs without executing them (equivalent to `propose-from-run`).

Archive stale placeholder runs that pollute the cockpit:

```bash
uv run --group dev mech archive-runs --before-run-id N --dry-run   # preview
uv run --group dev mech archive-runs --before-run-id N             # execute
```

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

### Cockpit

The local research cockpit serves a FastAPI/HTMX UI at `http://localhost:8000`:

```bash
uv run --group dev mech cockpit
```

Key routes:

- `/runs/<id>/features` — SAE feature explorer: top-activating prompts, dead-feature rate,
  and reconstruction MSE for a `polysemanticity_sae` run.
- `/runs/<id>/circuit` — ACDC circuit viewer: surviving nodes/edges, faithfulness score,
  and an embedded GraphViz dot for `acdc_lite` and `acdc_edge` runs.
- `/runs/<id>` — Reproducibility panel: environment fingerprint (torch version, seed,
  `uv.lock` SHA), result notes, and artifact manifest for any run family.

## Local-First Verification

Local checks are the source of truth for main-branch pushes. A minimal GitHub Actions workflow
runs `bash scripts/check.sh` on pull requests (fast tests, ruff, mypy, `mech validate`) to give
external contributors an automated gate. Integration tests that require model downloads are
excluded from CI.

```bash
bash scripts/check.sh
```

To also run integration tests locally (requires model downloads):

```bash
RUN_INTEGRATION_TESTS=1 bash scripts/check.sh
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
