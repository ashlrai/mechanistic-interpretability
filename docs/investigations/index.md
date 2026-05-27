# Investigations

All investigations run on local hardware (Apple Silicon MBP, 128 GB RAM) using the
platform's experiment families. Each one has exact reproduce commands at the bottom.

## Summary table

| # | Investigation | Model | Headline number | Status |
|---|---------------|-------|-----------------|--------|
| 1 | [Qwen Refusal Audit](refusal_audit.md) | Qwen2.5-1.5B-Instruct | Refusal direction extractable (quality 4.2 @ L12), but single-layer steering **fails** | Complete |
| 2 | [GPT-2 Factual Recall](gpt2_factual_recall.md) | GPT-2 small | 4-site circuit, **72% faithfulness**, commits at layer 9 | Complete |
| 3 | [SAE Replication Crisis](sae_replication_crisis.md) | GPT-2 small | Median best-match cosine **0.323–0.500**; stability @ ≥0.9 = **0%** | Complete |
| 4 | [Feature Splitting](feature_splitting.md) | GPT-2 small | 128→256: **0.714** mean fidelity (clean); 256→512: **0.601** (partial); 512→1024: **0.421** (reshuffle) | Complete |
| 5 | [SAE at Scale](sae_at_scale.md) | GPT-2 medium | 2048-feature SAE, L12, Pile-1k corpus; **52 live features**, top cluster: geographic/demographic | Complete |

## What each investigation tested

### 1 — Qwen Refusal Audit (negative result)

4-stage mechanistic audit of `Qwen2.5-1.5B-Instruct` refusal behavior.
Stage 1 extracted a refusal direction with quality 4.1–4.2 at layers 10–12.
Stage 2 showed single-layer CAA steering at ±3 does not flip compliance.
Stage 3 circuit patching found no dominant attention head cluster.
Stage 4 causal scrubbing showed <30% faithfulness — the mechanism is distributed.

**Implication:** small instruct models may have more robust (distributed) safety properties
than the abliteration literature implicitly assumes.

### 2 — GPT-2 Small Factual Recall

Logit lens, DLA, attribution patching, circuit patching, and SAE analysis on
factual recall prompts ("The capital of France is…"). Sharp phase transition at layer 9:
mean rank drops from 375 at L8 to 12.8 at L9. L9.MLP writes the answer; L8.MLP
suppresses competing tokens. Circuit achieves 72% faithfulness under causal scrubbing.

### 3 — SAE Replication Crisis

Five identical Top-K SAEs trained on GPT-2 small (layer 0 and layer 6) with seeds 1–5.
Pairwise Hungarian matching on decoder cosines. Four conditions:
layer-0/full, layer-0/live-only, layer-6/live-only, layer-6/512-feature/live-only.
Stability fraction at cosine ≥ 0.9 is **0% in all conditions at layer 6**.
The dead-feature confound (66% dead features) inflates the problem at full-matrix analysis.

### 4 — Feature Splitting

Four SAEs (128, 256, 512, 1024 features) trained on GPT-2 small layer 0.
Clean splitting (mean fidelity ≥ 0.80) at 128→256, partial specialisation at 256→512,
reshuffle at 512→1024. Larger dictionaries at this layer produce more equivalent bases,
not sharper concepts.

### 5 — SAE at Scale

2048-feature Top-K SAE on GPT-2 medium (345M), layer 12 mid-network, 1000-document
Pile corpus. 52 live features at 20k training tokens. Top 5 features cluster around
geographic/demographic representations. Wall-clock: ~6 min 15 sec on Apple Silicon CPU.
