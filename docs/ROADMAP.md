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
- ✅ **Cross-model representation probe**: ridge regression across model pairs.

## Stage 4: Local Scale — IN PROGRESS

- ✅ Per-run environment fingerprint (torch / numpy / TransformerLens versions, uv.lock SHA,
  seed) written to artifacts/environment.json for reproducibility.
- ✅ Resumable run queues and failure recovery (existing).
- ✅ Report generation (per-run research notes; aggregate summary report).
- ◻︎ Resource planning for batch size, activation retention, dtype, and model size on
  Apple Silicon (MPS) is partially scaffolded — needs validation on instruct-tuned models.

## Stage 5: Agentic Research Loop — IN PROGRESS

- ✅ Multi-family `ProposalGenerator` registry. Each family declares its own follow-up
  strategy (SAE → circuit_patching probes on top features; ACDC-lite → activation_capture
  over surviving nodes).
- ✅ `mech propose-from-run` CLI for per-run follow-up generation.
- ✅ Generated specs round-trip through the registry validator before they're queued.
- ◻︎ Closed-loop automation (run → propose → enqueue → run) is wired but not the default
  flow; users still execute generated specs manually.
- ◻︎ Anomaly detection and auto-triage of failed runs.

## Stage 6: Frontier Directions — FUTURE

- Refusal-direction extraction and representation steering on small instruct models
  (Qwen2.5-1.5B-Instruct or Gemma-2-2b-it). Requires TransformerLens model loading for
  modern instruct architectures, which may need a custom HF wrapper on this platform.
- Crosscoders for base vs fine-tuned model diffing.
- Auto-interpretability: use an LLM to label SAE features in the loop.
- Full edge-level ACDC (as opposed to the node-level lite version we ship today).
