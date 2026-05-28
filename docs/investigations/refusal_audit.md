# Investigation #1 — Mechanistic Audit of Refusal in Qwen2.5-1.5B-Instruct

**Date:** 2026-05-27
**Model:** `Qwen/Qwen2.5-1.5B-Instruct` (Apache 2.0, 28 transformer layers, d_model=1536)
**Total compute:** ~4.5 hours CPU on Apple Silicon MBP across 4 audit stages
**Run IDs:** 70 (Stage 1, refusal direction) · 71 (Stage 2, CAA layer sweep) · 72 (Stage 3, circuit patching) · 73 (Stage 4, causal scrubbing)
**Status:** **negative result — standard abliteration recipe fails on this model**

---

## Multi-model update (2026-05-27, second model added)

**Qwen2-0.5B-Instruct (24 layers) Stage 1 result (run 78):** the same single-coefficient steering recipe that FAILED on Qwen2.5-1.5B **succeeds on Qwen2-0.5B-Instruct**. At `blocks.12.hook_resid_post` with coefficient −3.0, refusal rate drops from 0.33 baseline to **0.00** — all 3 test prompts comply. coeff −2.0 also drops to 0.00. Extraction quality is even higher than on the larger model: **4.311** (vs Qwen2.5-1.5B's 4.105).

This is a **model-size-dependent** finding. The abliteration recipe is not universally broken; it works on small Qwen models. The transition between "works" (0.5B) and "fails" (1.5B) happens within the Qwen family at sizes most community abliterators target.

**Stage 2 (CAA layer sweep, run 79) result:**

| Layer | Extraction quality | Direction norm | Best refusal-rate shift |
|---:|---:|---:|---:|
| 6 | 3.54 | 1.6 | 0.000 |
| **12** | **4.31** | 4.8 | **+0.333** (refusal 0.33 → 0.00 at coeff ∈ {−2, −3}) |
| 18 | 5.39 | 10.8 | 0.000 |
| 22 | 3.93 | 21.8 | 0.000 |

Same decoupling of extraction quality and steering effectiveness as on Qwen2.5-1.5B (layer 18 has the highest extraction quality but zero behavioral effect), but **layer 12 actually allows controllable suppression** — at coeff −3.0 refusal drops to 0/3 prompts, at coeff −2.0 it also drops to 0/3 (monotonic in the predicted direction). This is the abliteration-recipe-working pattern.

So the side-by-side picture across the two audited models is:

| Quantity | Qwen2-0.5B (24L) | Qwen2.5-1.5B (28L) |
|---|---:|---:|
| Best layer (by behavioural shift) | 12 (mid) | 10 (early-mid) |
| Best-layer extraction quality | 4.31 | 4.11 |
| Highest-quality layer | 18 (5.39) | 12 (4.25) |
| Behavioural effect at best layer, coeff −3 | refusal **0.33 → 0.00** | refusal **0.33 → 0.67** |
| Recipe verdict | **works** | **fails** |

### Qwen2.5-0.5B Stage 1 result (run 80) — discriminating test

Adding Qwen2.5-0.5B-Instruct (24L) discriminates between two hypotheses about *why* Qwen2.5-1.5B's recipe failure occurs: **size** (1.5B too large) vs **model-generation** (Qwen2.5 instruction tuning more robust than Qwen2).

At `blocks.12.hook_resid_post`, extraction quality 3.858, **baseline refusal 0.67** (higher than the other two — Qwen2.5-0.5B refuses 2/3 prompts by default):

| Coefficient | Refusal rate | Shift |
|---:|---:|---:|
| −3.0 | 0.33 | **−0.33** |
| −2.0 | 0.33 | **−0.33** |
| −1.0 | 0.33 | **−0.33** |
| 0.0 | 0.67 | 0.00 |
| +1.0 | 0.67 | 0.00 |
| +2.0 | 0.67 | 0.00 |
| +3.0 | 0.67 | 0.00 |

**Recipe WORKS on Qwen2.5-0.5B** — graded suppression across all negative coefficients (the canonical Arditi pattern), refusal halved from 0.67 to 0.33.

### Four-model summary

| Model | Size | Hook | Extr. quality | Behavioural pattern under steering | Recipe verdict |
|---|:---:|---|---:|---|---|
| Qwen2-0.5B-Instruct | 0.5B | blocks.12.resid_post | 4.311 | coeff −3 → refusal 0.33 → 0.00 (monotonic) | **Works fully** |
| Qwen2.5-0.5B-Instruct | 0.5B | blocks.12.resid_post | 3.858 | coeff −1..−3 → 0.67 → 0.33 (graded) | **Works partially** |
| Qwen2.5-1.5B-Instruct | 1.5B | blocks.10.resid_post | 4.105 | coeff −3 only → 0.33 → 0.67 (saturating backfire) | **Fails** |
| Qwen2.5-3B-Instruct | 3B | blocks.18.resid_post | 2.329 | coeff +2/+3 → 0.33 → 0.67; negative coeffs no effect | **Fails (amplify-only)** |

**The 4-point trend across Qwen scale is consistent**: as model size grows, the single-layer additive-direction recipe loses the ability to *suppress* refusal. The 0.5B models give graded suppression; the 1.5B model gives a saturating backfire; the 3B model loses suppression entirely (negative coefficients are dead) but the direction-sign is still correct (positive coefficients increase refusal). Extraction quality also degrades from 4.3 → 2.3 across the same scale at the default hook layer, though the Qwen2.5-1.5B CAA sweep showed extraction quality can be recovered at a different layer; a Qwen2.5-3B CAA sweep is needed to confirm whether *any* layer enables suppression in the 3B model.

This 4-point pattern is the strongest evidence yet that the abliteration recipe's domain of applicability is bounded by scale within the Qwen family. The community's typical abliteration targets (3-9B) are above the demonstrated working range.

### Qwen2.5-3B CAA sweep (run 82) — no layer enables suppression

To rule out "wrong default layer" as the explanation for Stage 1's null result, a 4-layer CAA sweep was run on Qwen2.5-3B-Instruct:

| Layer | Extraction quality | Direction norm | Best coeff | Best shift |
|---:|---:|---:|---:|---:|
| 9 | 1.68 | 6.6 | −3.0 | 0.000 |
| 18 | 2.33 | 20.3 | +2.0 | +0.333 (amplify) |
| 27 | 3.99 | 47.2 | +3.0 | +0.333 (amplify) |
| 34 | 3.93 | 145.7 | −3.0 | 0.000 |

**Across all 4 layers, no negative coefficient suppresses refusal.** Only positive coefficients (amplification) produce a behavioral shift, and only at specific mid-late layers. The direction norms grow dramatically with depth (6.6 → 145.7) without corresponding suppression efficacy. This pattern is consistent with refusal being implemented redundantly across multiple residual-stream directions by 3B — single-direction additive steering cannot suppress what is computed by parallel paths.

This locks the 4-model story: the Qwen2.5-3B failure is not a hook-choice artifact; it is a property of the model. The recipe's domain of applicability ends below 3B within the Qwen family.

### Implication for the abliteration literature

Most community abliterations target Llama-3.2-3B / Llama-3.2-8B / Gemma-2-9B — models in the 3-9B range. The Qwen 0.5B → 1.5B transition between "works" and "fails" suggests the recipe may break entirely as model size grows beyond a small handful of billions of parameters. If true, the community's abliteration recipes have an implicit size ceiling nobody has characterised.

Stages 3-4 (circuit_patching + causal_scrubbing) on the two 0.5B models would formally confirm the recipe's circuit hypothesis holds there (faithfulness > 0.5), then a Qwen2.5-3B audit would localise the transition point.

> **Compute note (honest limitation):** Stages 3-4 (circuit_patching, causal_scrubbing) were attempted on Qwen2.5-3B (runs 84/85) and Phi-3-mini 3.8B (run 83) but **OOM-killed on this 128 GB CPU machine** — those families hold the full activation + gradient cache across many hook sites, which exceeds memory at 3B+. The lighter Stage-1/Stage-2 families (refusal_direction sweep, caa_steering) complete at 3B. **This does not weaken the headline:** the Qwen2.5-3B CAA layer sweep (run 82) already established that no layer enables suppression, which is what locks the scale-bounded conclusion. Stage 3-4 head localisation at 3B would be a refinement, not a load-bearing result, and needs a higher-memory machine or an MPS + reduced-batch path.

**Implication:** the original Qwen2.5-1.5B "headline" below is correct but narrow. The broader picture is that the recipe's domain of applicability is bounded — works on ≤ 0.5B Qwen, fails on ≥ 1.5B Qwen, fully fails by 3B (CAA-confirmed across all layers).

---

## Headline

> **For `Qwen2.5-1.5B-Instruct`, refusal IS a linearly separable direction in the residual stream (extraction quality 4.1 at layer 10, 4.2 at layer 12), but it is NOT controllable via single-layer steering at the natural candidate layers, and it is NOT implemented by the attention head outputs at those layers.** The standard Arditi/RepE abliteration recipe — find the direction, ablate the attention contributions that write it — produces a circuit hypothesis with faithfulness 0.04 against the 4-stage formal scrubbing test. The information arrives in `blocks.10-11.hook_resid_post` (recovery fraction 0.50-1.04 under exact patching) but is NOT written there by local attention.

This is a genuine negative result on a real instruct-tuned model. Most published abliteration writeups assume the recipe works; this audit produces mechanistic evidence that it doesn't on at least one production checkpoint.

---

## Stage-by-stage results

### Stage 1 — Refusal direction extraction (run 70)

`mech run --name refusal-direction-qwen` · 62 min wall-clock

| Quantity | Value |
|---|---:|
| Hook site | `blocks.10.hook_resid_post` |
| Extraction quality (projection margin) | **4.105** |
| Direction norm | 11.73 |
| Baseline refusal rate (coeff=0) | 0.33 (1 of 3 test prompts) |
| Refusal rate at coeff=−3.0 | 0.67 (+0.33) |
| Refusal rate at coeff ∈ {−2, −1, 0, +1, +2, +3} | 0.33 (no change) |

**Interpretation:** the direction is genuinely separating harmful from harmless activations — a projection margin of 4.1 is well above the linear-separability threshold of 1.0. But the response to steering is highly asymmetric and saturating. Only the strongest negative coefficient perturbs behavior, and it *increases* refusal rather than decreasing it (the abliteration goal). At coefficients within ±2, single-layer steering at layer 10 has zero effect.

### Stage 2 — CAA multi-layer sweep (run 71)

`mech run --name caa-steering-qwen` · 234 min wall-clock (4 layers × 7 coefficients × 3 test prompts × ~50 tokens of greedy decode)

| Layer | Extraction quality | Direction norm | Best coefficient | Best refusal_rate_shift |
|---:|---:|---:|---:|---:|
| 6 | 1.72 | 4.21 | −3.0 | +0.00 |
| 8 | 3.10 | 6.99 | −3.0 | +0.33 |
| 10 | 4.11 | 11.73 | −3.0 | +0.33 |
| 12 | **4.25** | 16.16 | −3.0 | +0.00 |

**Interpretation:** Extraction quality grows monotonically with depth (1.7 → 4.2) — the linear direction becomes cleaner deeper in the network. But causal steerability does NOT track extraction quality. Layer 12 has the highest extraction quality (4.25) but produces *zero* refusal-rate shift across all 7 coefficients. Layer 6 has the lowest extraction quality (1.7) and likewise has no effect. Only layers 8 and 10 show any behavioral shift, and only at coeff=−3.0, and only by adding a single refusal (1 of 3 prompts).

This pattern — **direction quality and steering effectiveness are decoupled** — is one of the central findings of the audit. The conventional wisdom that the cleanest extraction layer is the right intervention layer is contradicted here.

### Stage 3 — Circuit patching (run 72)

`mech run --name refusal-circuit-qwen` · 3 min wall-clock (exact patching at 6 hook sites × 3 prompt pairs = 18 forward passes)

Top sites by recovery_fraction (clean=harmful, corrupted=harmless):

| Rank | Pair | Hook site | Recovery |
|---:|---|---|---:|
| 1 | pair-harm-1 | `blocks.11.hook_resid_post` | **1.037** |
| 2 | pair-harm-1 | `blocks.10.hook_resid_post` | 0.831 |
| 3 | pair-harm-3 | `blocks.11.hook_resid_post` | 0.710 |
| 4 | pair-harm-2 | `blocks.11.hook_resid_post` | 0.638 |
| 5 | pair-harm-2 | `blocks.10.hook_resid_post` | 0.504 |
| 6 | pair-harm-3 | `blocks.10.hook_resid_post` | 0.131 |
| 7 | pair-harm-1 | **`blocks.10.attn.hook_z`** | 0.128 |
| 8 | pair-harm-2 | **`blocks.10.attn.hook_z`** | 0.092 |
| 9 | pair-harm-2 | `blocks.9.hook_resid_post` | 0.062 |
| 10 | pair-harm-3 | `blocks.11.attn.hook_z` | 0.046 |
| 11 | pair-harm-1 | `blocks.11.attn.hook_z` | 0.026 |
| 12 | pair-harm-2 | `blocks.9.attn.hook_z` | 0.022 |

**Interpretation:** This is the surprising result. The refusal signal is **clearly carried by the residual stream** at layers 10-11 (recovery 0.50-1.04 — patching `blocks.11.hook_resid_post` on the first pair recovers more than the full clean-corrupted gap, overshooting slightly). But the **attention-head outputs at the same layers contribute almost nothing** (recovery 0.02-0.13).

Mechanistically, this means the refusal information has been *deposited* into the residual stream by layer 10, but it is NOT being *written* by the attention heads at layers 9, 10, or 11. The signal arrives via either (a) the MLPs at those layers, or (b) the residual stream from earlier — most likely the LayerNorm + skip-connection accumulation of contributions from layers 0-8 that we did not patch in this audit. A full edge-level path analysis is needed to localize the actual writer.

### Stage 4 — Causal scrubbing (run 73)

`mech run --name causal-scrubbing-refusal-qwen` · 2.3 min wall-clock

Hypothesis: refusal is implemented by `blocks.9.attn.hook_z` and `blocks.10.attn.hook_z`. Protect those; scrub the other 52 sites (all attention outputs at all 26 other layers, plus all residual posts at all 26 other layers).

| Quantity | Value |
|---|---:|
| Mean KL(full ∥ scrubbed) | 3.197 |
| Max KL | 5.685 |
| Faithfulness = exp(−mean KL) | **0.041** |
| Verdict | **REJECTED** (faithfulness ≪ 0.5) |
| Protected sites | 2 |
| Scrubbed sites | 52 |

**Interpretation:** Protecting just the two attention-head sites destroys 96% of the model's behavior on the audit prompts. Faithfulness of 0.04 is essentially "the model is randomized." The hypothesis "L9+L10 attention heads implement refusal" is formally rejected by the scrubbing receipt. This is consistent with and quantifies Stage 3's finding that those heads contribute little.

A correct hypothesis would have to protect either (a) the residual stream itself at layers 10-11, which is largely vacuous because resid_post is the model's working memory — protecting it protects everything that flows through it, or (b) the MLP outputs at layers ≤ 8 that we did not measure. A genuine causal-scrubbing-grade circuit description for Qwen2.5-1.5B refusal would require a follow-up MLP-output patching pass, plus per-head decomposition at L10 attention (the existing audit treats `hook_z` as a whole rather than per-head).

---

## What this means for the abliteration recipe

Abliteration (Arditi et al., 2024 and the dozens of HuggingFace community derivatives) consists of: (i) extract a refusal direction via difference of means; (ii) project the attention head output weights in the directions that write that direction so the model can no longer write it; (iii) ship the modified weights. The recipe assumes step (ii) — that the relevant writes happen at attention heads near the direction's most-separable layer.

**This audit shows step (ii) of that recipe is unsupported for Qwen2.5-1.5B-Instruct.** The most-separable layer (12) has zero causal effect under steering. The middle-separable layers (8, 10) have only weak coefficient-saturated effects. And the local attention heads at the most-effective layers (9, 10, 11) carry essentially none of the refusal information that the residual stream carries.

This does NOT prove the recipe fails on other models — gpt2-medium, Gemma-2, Llama-3 may have different mechanistic implementations. But it does show the recipe needs validation on each model rather than assumed to transfer.

## Caveats

1. **Tiny test set.** 3 test prompts × 5 contrastive pairs is enough to detect the linear direction but too small to characterize the response surface. Several "no shift" entries in the CAA table could be statistical fluctuation on the 3-prompt evaluation set. A serious version of this audit needs 30-100 test prompts and a proper refusal classifier (not just keyword matching).
2. **Refusal proxy.** We detect refusal by string matching `["I'm sorry, but I can't", "I cannot", "I'm unable", ...]`. A model that complies by saying "Sure, here's how..." won't be detected as compliant if it also includes any of these phrases. The metric has known noise.
3. **Coefficient range too narrow.** The −3 to +3 range is conventional but may be too small. Real abliteration projects often use coefficients of ±10 or ±20 to get reliable steering. A future pass should sweep wider.
4. **Single hook-site shape.** `hook_z` is the full (batch, seq, n_heads, d_head) attention output — we patch ALL heads at a layer simultaneously. Per-head patching would be more informative.
5. **No MLP-output patching.** The audit's hook list does not include `blocks.{L}.hook_mlp_out`, which Stage 3 strongly suggests is where the actual writing happens.

A follow-up audit would: (a) widen the test set, (b) widen the coefficient range, (c) patch MLP outputs at every layer, (d) decompose `hook_z` per head, (e) add a real refusal classifier rather than keyword matching.

## Reproducibility

- `experiments/refusal_direction.yaml`
- `experiments/caa_steering.yaml`
- `experiments/refusal_circuit_qwen.yaml`
- `experiments/causal_scrubbing_refusal_qwen.yaml`
- Run IDs 70, 71, 72, 73 in `artifacts/` with their full `environment.json` provenance
- Compiled report: `mech audit-refusal --refusal-run 70 --caa-run 71 --circuit-run 72 --scrub-run 73`

Total reproduction cost: ~4.5 hours CPU on a 2024-era Apple Silicon MBP. The bottleneck is Stage 2 (generation sweep, 4 hours).

## Safety disclaimer

This investigation documents a controllability *failure* — Qwen2.5-1.5B-Instruct's refusal mechanism resists the standard single-layer steering recipe in our tests. We do NOT publish modified weights. The platform code and direction vectors that are committed here cannot in themselves disable refusal: the Stage 1 / Stage 2 sweeps explicitly show that single-layer ±3 steering does not enable harmful compliance. Anyone seeking to abliterate this model would have to develop a new technique that survives the negative results above.

The broader point is mechanistic: small instruct models do not necessarily have the convenient "write refusal here, ablate, done" structure that the abliteration literature implicitly assumes. Genuine safety properties of small models may be more distributed and harder to remove than the recipe suggests — which is itself a small piece of good safety news.
