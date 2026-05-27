# Notebooks

Interactive notebooks live in `notebooks/` at the repo root. They are the primary narrative
format for investigations — code, charts, and commentary together.

## Viewing notebooks

GitHub renders `.ipynb` files natively. Click any link below to read the notebook
in your browser without cloning.

!!! note "Execution note"
    These notebooks load artifacts from specific experiment runs (run IDs and artifact paths
    are hardcoded to the original research environment). To re-run them end-to-end, first
    reproduce the relevant sweep with `mech sweep --execute`, then update the artifact path
    constants at the top of each notebook.

## Notebook index

| Notebook | Investigation | Status |
|----------|---------------|--------|
| [06 — SAE Replication Crisis](https://github.com/ashlrai/mechanistic-interpretability/blob/main/notebooks/06_sae_replication_crisis.ipynb) | [Investigation #3](../investigations/sae_replication_crisis.md) | Complete — pairwise cosine heatmaps, distribution histograms, 4-condition comparison |
| [07 — GPT-2 Factual Recall Story](https://github.com/ashlrai/mechanistic-interpretability/blob/main/notebooks/07_gpt2_factual_recall_story.ipynb) | [Investigation #2](../investigations/gpt2_factual_recall.md) | Complete — logit lens, DLA, circuit patching, SAE feature cluster |
| [08 — Feature Splitting](https://github.com/ashlrai/mechanistic-interpretability/blob/main/notebooks/08_feature_splitting.ipynb) | [Investigation #4](../investigations/feature_splitting.md) | Complete — 4-size comparison, fidelity bar chart |
| [09 — Refusal Audit](https://github.com/ashlrai/mechanistic-interpretability/blob/main/notebooks/09_refusal_audit.ipynb) | [Investigation #1](../investigations/refusal_audit.md) | Complete — direction quality sweep, CAA compliance plots |
| [10 — SAE at Scale](https://github.com/ashlrai/mechanistic-interpretability/blob/main/notebooks/10_sae_at_scale.ipynb) | [Investigation #5](../investigations/sae_at_scale.md) | Complete — 2048-feature dictionary, feature cluster analysis |
| [05 — Research Walkthrough](https://github.com/ashlrai/mechanistic-interpretability/blob/main/notebooks/05_research_walkthrough.ipynb) | General | Tutorial — walks through a full experiment from spec to artifact |
| [01 — Circuit Patching](https://github.com/ashlrai/mechanistic-interpretability/blob/main/notebooks/01_circuit_patching.ipynb) | General | Tutorial — activation patching basics |
| [02 — Polysemanticity SAE](https://github.com/ashlrai/mechanistic-interpretability/blob/main/notebooks/02_polysemanticity_sae.ipynb) | General | Tutorial — train and inspect a small SAE |
| [03 — ACDC Lite](https://github.com/ashlrai/mechanistic-interpretability/blob/main/notebooks/03_acdc_lite.ipynb) | General | Tutorial — automated circuit discovery |
| [04 — Agentic Loop](https://github.com/ashlrai/mechanistic-interpretability/blob/main/notebooks/04_agentic_loop.ipynb) | General | Tutorial — orchestration and iterate loop |

## Reproducing a notebook from scratch

### SAE Replication Crisis (notebook 06)

```bash
# 1. Train 5 SAEs at layer 0
mech sweep \
  --base experiments/polysemanticity.yaml \
  --axis "parameters.seed=1,2,3,4,5" \
  --output experiments/sweeps/sae_seed_stability_layer0.yaml \
  --execute

# 2. Generate the stability report
mech analyze-sae-stability \
  --sweep experiments/sweeps/sae_seed_stability_layer0.yaml \
  --output artifacts/stability_layer0_128.json \
  --live-only

# 3. Update REPORT_PATH in notebook cell 1 to point at your new artifact
# 4. jupyter nbconvert --execute --to notebook --inplace notebooks/06_sae_replication_crisis.ipynb
```

### GPT-2 Factual Recall (notebook 07)

```bash
# Reproduce the six runs (adjust run IDs after execution)
mech run --name logit-lens-factual
mech run --name direct-logit-attribution-factual
mech run --name attribution-patching-factual
mech run --name circuit-patching-factual
mech run --name polysemanticity-sae-layer9-factual
mech run --name causal-scrubbing-factual

# Then update run ID constants at top of notebook 07
```
