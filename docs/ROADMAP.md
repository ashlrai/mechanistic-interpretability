# Roadmap

This project should grow from a reliable local experiment spine into a serious mechanistic
interpretability lab. Keep each stage runnable before moving to the next.

## Stage 1: Research Spine — DONE

- ✅ Validate experiment YAML before execution.
- ✅ Persist resolved specs, configs, run status, metrics, and artifact manifests.
- ✅ Add provider health checks for Ollama and LM Studio.
- ✅ Add a TransformerLens smoke path that does not require heavyweight model downloads in tests.
- ✅ Keep local checks green on every push.
- ✅ PR-only CI via GitHub Actions (`check.sh`: pytest fast tests, ruff, mypy, `mech validate`).

## Stage 2: TransformerLens Execution — DONE

- ✅ Load small supported models through the instrumented backend (gpt2-small verified).
- ✅ Capture activations for selected hook names with deterministic seeding.
- ✅ Save tensor summaries by default and full tensors only when explicitly requested.
- ✅ Smoke experiments for residual stream, MLP output, and activation patching.
- ✅ Real integration tests against gpt2-small validate the patching pipeline produces
  expected recovery fractions (>0.9 for residual stream, near-zero for MLP control).

## Stage 3: First Real Experiment Families — DONE

- ✅ **Polysemanticity (SAE)**: Top-K sparse autoencoder (Gao et al., 2024) on residual-stream
  activations. Produces interpretable feature dictionaries with per-feature top-activating
  prompts and dead-feature stats.
- ⏭ **Superposition**: deferred. Plumbing reserved; no real implementation yet.
- ✅ **Circuit patching**: clean/corrupted prompt pairs with mean-ablation recovery fractions
  and control hook sites.
- ✅ **ACDC-lite**: node-level automatic circuit discovery (Conmy et al., 2023) — scores every
  (layer, head) attention node and (layer, MLP) node by ablation logit-diff impact, prunes
  below a threshold, and reports faithfulness + GraphViz dot.
- ✅ **Edge-level ACDC**: full edge-level circuit discovery (Conmy et al., 2023) — scores
  individual (source-hook → dest-hook) edges for a sparser, higher-resolution circuit graph.
- ✅ **Refusal direction**: extracts the principal refusal direction from instruct-model
  residual streams via contrastive prompt pairs and measures ablation impact (Arditi et al., 2024).
- ✅ **Cross-model representation probe**: ridge regression across model pairs.

## Stage 4: Local Scale — IN PROGRESS

- ✅ Per-run environment fingerprint (torch / numpy / TransformerLens versions, uv.lock SHA,
  seed) written to artifacts/environment.json for reproducibility.
- ✅ Resumable run queues and failure recovery (existing).
- ✅ Report generation (per-run research notes; aggregate summary report).
- ◻︎ Resource planning for batch size, activation retention, dtype, and model size on
  Apple Silicon (MPS) is partially scaffolded — needs validation on instruct-tuned models.

## Stage 5: Agentic Research Loop — DONE

- ✅ Multi-family `ProposalGenerator` registry. Each family declares its own follow-up
  strategy (SAE → circuit_patching probes on top features; ACDC-lite → activation_capture
  over surviving nodes).
- ✅ `mech propose-from-run` CLI for per-run follow-up generation.
- ✅ Generated specs round-trip through the registry validator before they're queued.
- ✅ `mech iterate-from-run` closes the loop end-to-end: generate proposals, execute them,
  and recurse up to `--max-depth` levels without manual intervention.
- ✅ `mech archive-runs` removes stale placeholder runs from the cockpit and default listings.
- ◻︎ Anomaly detection and auto-triage of failed runs.

## Stage 6: Frontier Directions — FUTURE

- **Corpus-scale SAE training**: stream activations from a large text corpus (e.g. Pile
  subset) in chunks to train SAEs on millions of tokens without holding the full activation
  matrix in RAM.
- **Auto-interpretability**: pipe top-activating prompts per feature through a local LLM
  (Ollama) to generate human-readable feature labels and confidence scores in the loop.
- **Multi-model SAE comparison**: train SAEs on aligned hook sites across model families
  (GPT-2 vs Pythia vs Gemma) and score cross-model feature overlap.
- **Crosscoders**: base vs fine-tuned model diffing to isolate fine-tuning-specific circuits.
- **Representation steering**: apply extracted directions (refusal, sentiment, factuality)
  as activation additions and measure downstream behavioural shift.
