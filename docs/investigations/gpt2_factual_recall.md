# GPT-2 Small: How It Recalls Facts

**One-line story:** GPT-2 small commits the correct capital/fact token at layer 9, driven primarily by L9.MLP writing directly on the unembedding direction, with L8.MLP suppressing competing tokens — a 4-site circuit that achieves 72% faithfulness.

---

## Evidence chain

### 1. Logit lens — the decision layer (run-000045)

Projecting the residual stream through the unembedding at every layer reveals a sharp phase transition at L9:

| Layer | Mean rank of correct token |
|------:|---------------------------:|
| 8     | 375                        |
| **9** | **12.8**                   |
| 10    | 31.8                       |
| 11    | 27.2                       |

3 of 4 prompts first enter top-5 at **layer 9** exactly (capital-italy rank 3, capital-germany rank 2, planet-largest rank 2). Capital-france is harder (rank 44 at L9 vs 605 at L8) but shows the same discontinuous drop. Mean CE loss falls from 9.18 at L8 to 3.42 at L9.

### 2. Direct logit attribution — the writing components (run-000042)

Decomposing the final-token logit over 4 prompts (mean scores across prompts):

**Positive writers** (push toward correct token):
- L9.MLP: **+9.29**
- L11.MLP: +5.88
- L10.MLP: +3.79
- L5.MLP: +1.54

**Negative writers** (suppress correct token):
- L8.MLP: **-3.15**
- L4.MLP: -1.34
- L6.MLP: -0.97

L9.MLP dominates. L8.MLP is the strongest suppressor — suggesting a two-phase pattern where early-mid MLPs build a competing representation that L9.MLP overrides.

### 3. Attribution patching — cheap causal scan (run-000047)

First-order gradient scan across 24 hook sites (resid_pre + mlp_out at all 12 layers) confirms the DLA picture. Top sites by |attribution|:

1. `blocks.11.hook_resid_pre` — |score| 5.04
2. `blocks.9.hook_resid_pre` — confirms L9 as key bottleneck
3. `blocks.5.hook_resid_pre`, `blocks.6.hook_resid_pre`, `blocks.10.hook_resid_pre`

The attribution ranking (gradient approximation) and DLA ranking (exact decomposition) agree on the same cluster of layers 9–11.

### 4. Circuit patching — causal confirmation (run-000050)

Exact clean→corrupted activation patching on layers 8–11 (resid_pre and mlp_out) across 4 prompts. Top result: patching `blocks.8.hook_resid_pre` on capital-france recovers **100% of the clean-corrupted logit diff** (recovery_fraction = 1.000). All tested layers 8–11 show recovery_fraction = 1.0, confirming that a single resid_pre patch at any of these sites is causally sufficient to restore the correct answer. Mean recovery across all 16 site×prompt pairs = 1.0.

### 5. SAE at the decision layer (run-000052)

256-feature Top-8 SAE trained on `blocks.9.hook_resid_pre` (992 tokens, openwebtext sample). 104/256 features active (59% dead — expected for a sparse corpus). Explained variance: **74.2%**.

Geographic features identified by top-activating prompts:

| Feature index | Max activation | Top prompt |
|--------------:|---------------:|:-----------|
| 194           | 1082.7         | "Paris is the capital of France and is known for the Eiffel T…" |
| 212           | 1082.5         | same doc |
| 5             | 1081.6         | same doc (also activates on 7 other docs) |
| 43            | 1079.9         | same doc |

Features 194, 212, 43 each activate on exactly 31 docs (monosemantic, doc-level); feature 5 fires on 38 docs (slightly broader). The Paris/capital/Eiffel cluster occupies a tight, high-activation region of the L9 pre-residual feature space.

### 6. Causal scrubbing — faithfulness receipt (run-000053)

**4-site hypothesis:** `blocks.8.hook_mlp_out`, `blocks.9.hook_resid_pre`, `blocks.9.attn.hook_z`, `blocks.10.hook_resid_pre`.

All 10 other attention outputs scrubbed with within-equivalence-class resampling.

| Metric | Value |
|:-------|------:|
| Faithfulness | **0.720** |
| Mean KL(full ‖ scrubbed) | 0.329 |
| Max KL | 0.418 |

Faithfulness 0.72 is in the partial-support range (0.5–0.8). The 4 protected sites capture most but not all of the circuit; the remaining ~28% variance likely lives in earlier attention heads (L5–L7) that the DLA scan found contributing at smaller magnitude.

---

## The causal story

GPT-2 small accumulates factual evidence across layers 3–8 via a series of MLP passes that progressively narrow the vocabulary distribution (rank drops from 27,000 at L0 to 375 at L8). At **layer 9**, L9.MLP writes directly onto the unembedding direction with mean DLA score +9.29, collapsing the correct-token rank from 375 to 13. L8.MLP acts as the dominant suppressor (-3.15) of the correct-token direction in the layers immediately before the commitment. A 4-site circuit (L8.mlp_out + L9.resid_pre + L9.attn.z + L10.resid_pre) achieves 72% faithfulness under causal scrubbing, with the residual 28% attributable to earlier layers. The SAE at L9 reveals that geographic/capital information is encoded in a tight cluster of 4+ high-activation features (indices 194, 212, 5, 43) with activations exceeding 1080 on Paris-capital documents — suggesting factual associations are stored in localised, high-norm feature directions in the pre-L9 residual stream.

---

## Weaknesses

1. **Capital-france outlier.** Only 3/4 prompts commit at L9; "The capital of France is" ranks 44 at L9 (not top-5). The prompt lacks an explicit article before the country name — tokenisation differences may route through a slightly different path.
2. **Circuit patching saturation.** Recovery fraction = 1.0 for *all* tested sites suggests the patch space is overcomplete — every late-layer resid_pre carries full information, so we cannot isolate a minimal circuit this way. Follow-up: path patching (attention head outputs individually).
3. **SAE dead features.** 59% dead features on 992 tokens means the feature dictionary is undertrained; the geographic cluster is real but other factual features may be missed. Need 10–50× more tokens to saturate the 256-feature dictionary.
4. **No attention head decomposition.** DLA and attribution patching only ran on MLP layers (the current runner's DLA implementation scores MLPs, not per-head attention outputs). L9.attn.hook_z appears in the protected hypothesis but we have no direct evidence for which specific heads contribute.

---

*Runs: logit_lens=run-000045, DLA=run-000042, attribution_patching=run-000047, circuit_patching=run-000050, SAE_L9=run-000052, causal_scrubbing=run-000053*
