# Changelog

All notable changes to this project will be documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.0-preview] - 2026-05-27

First public-preview release of the local, agent-driven mechanistic
interpretability research platform.

### Added

**14 experiment families**
- `circuit_patching` — activation patching with noising baselines
- `acdc_lite` — node-level ACDC automatic circuit discovery on GPT-2 small
- `acdc_edge` — edge-level ACDC with KL-weighted source ablation
- `polysemanticity_sae` — Top-K Sparse Autoencoder training + feature analysis
- `attribution_patching` — gradient × activation attribution across layers
- `causal_scrubbing` — hypothesis-driven causal scrubbing pipeline
- `refusal_direction` — Arditi / RepE refusal direction extraction (CAA)
- `caa_steering` — multi-layer contrastive activation addition steering
- `logit_lens` / `tuned_lens` — residual stream prediction across layers
- `sparse_probing` — linear probing on residual stream activations
- `sae_cross_model` — cross-model SAE feature alignment
- `direct_logit_attribution` — DLA per-head and per-MLP decomposition
- `crosscoder` — cross-model SAE feature comparison (crosscoder architecture)
- `cross_model_representation_probe` — probing shared representations across models

**Agentic orchestration loop**
- `ProposalGenerator` registry — LLM-driven experiment proposal from prior results
- `IterateLoop` — autonomous hypothesis → run → analyse → propose cycle
- `PrefightChecker` — resource / config validation before any run

**Gradio web UI** (`mech cockpit` + `gradio_app.py`)
- 4-panel interactive analysis: activations, SAE features, steering, circuit view
- Timeline, compare-runs, and artifact-browser views

**Universal HuggingFace backend** (`hf_adapter.py`)
- Causal-LM adapter supporting any HF model via TransformerLens site translation
- `hf_site_translator.py` — maps `hook_resid_pre` → HF module paths automatically

**Steering vector library**
- Pre-extracted steering vectors (refusal / sentiment / helpfulness) in safetensors
- `mech apply-steering` CLI command
- `steering/registry.py` — load, inspect, and compose steering vectors

**SAE-Lens compatibility shim** (`sae/compat.py`)
- Drop-in compatibility layer for sae_lens ≥ 3.0 pretrained SAE registry
- `mech sae list/download/analyze` CLI subcommands

**Research findings (publishable)**
- *SAE Replication Crisis*: seed variance in Top-K SAE feature counts exceeds
  inter-scale variance; live-feature counts are not reproducible without fixed
  seeds. Paper, tweet thread, and reproducible artifact bundle included.
- *Abliteration Robustness*: Qwen2.5-1.5B-Instruct refusal direction is
  separable from capability representations; mean ablation degrades refusal
  at k ≥ 8 with <2 % capability loss on factual benchmarks.

**Infrastructure**
- `sqlite_store.py` — local SQLite artifact/run database with full-text search
- `mkdocs-material` documentation site deployed to GitHub Pages
- `mech validate` — schema-validate all experiment YAML configs
- CI workflow (GitHub Actions) for PR-gate linting, typing, and unit tests
- Docs-deploy workflow with `mkdocs build --strict` gate
- `scripts/check.sh` — fast (default) and full CI gate modes

### Changed
- `pyproject.toml` extras now have upper-bound version pins on all deps
- `huggingface_hub` added explicitly to `interp` extra

### Fixed
- `tests/test_caa_steering.py`, `test_refusal_direction.py`, `test_crosscoder.py`,
  `test_sae_lens_compat.py` — top-level `import torch` replaced with
  `pytest.importorskip` so base-env collection never fails in CI
- `scripts/check.sh` — added `mkdocs build --strict` as a gate; added `--fast`
  / `--full` modes with documented trade-off

---

## [Pre-release commits] — prior to 0.1.0-preview

| SHA | Summary |
|-----|---------|
| `cdca1ab` | Initial platform scaffold — CLI skeleton, experiment base class, config loader |
| `5a1c6f5` | Build experiment validation and local run tooling |
| `d4418ef` | Use local validation instead of GitHub Actions |
| `03df044` | Add usable local experiment workflows |
| `566c362` | Keep activation capture artifacts per run |
| `2f8c3c2` | Add reliable local lab OS (smoke.sh, serve_docs.sh) |
| `5e8ed0e` | Fix: make local smoke loop usable |
| `3568f93` | Foundation: seeds, env fingerprint, placeholder gate, real integration tests |
| `bf8f98a` | Top-K SAE family replaces the placeholder |
| `2136cd9` | Node-level ACDC-lite for automatic circuit discovery |
| `113644c` | Multi-family ProposalGenerator registry closes the agentic loop |
| `4698f34` | Add MPS validation smoke tests and defensive float32 casts |
| `bc76439` | Add minimal PR-only CI workflow |
| `7fed977` | Consolidate artifact dirs and add archive-runs command |
| `3b94d60` | Wire SAE training to accept a tokens corpus |
| `41ba9aa` | Cockpit upgrades: SAE features, ACDC circuit, env provenance |
| `1d7f65e` | Add refusal_direction family (Arditi / RepE pipeline) |
| `e0bd21c` | Add edge-level ACDC with KL-weighted source ablation |
| `aa2821b` | Fix: end-to-end CLI verification — runner artifact dirs + SAE corpus load |
| `b99f583` | Refactor: simplify ACDC, SAE, orchestration, and cockpit helpers |

[0.1.0-preview]: https://github.com/masonwyatt/mechanistic-interpretability/releases/tag/v0.1.0-preview
