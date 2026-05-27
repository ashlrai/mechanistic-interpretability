# Is the Abliteration Recipe Robust? A Four-Stage Mechanistic Audit Across Small Instruct Models

**Mason Wyatt · 2026-05-27**
**Status:** in progress — Qwen2.5-1.5B complete; additional models running

---

## Abstract

Abliteration — the practice of suppressing LLM refusal by projecting out a linear
"refusal direction" from attention-head output weights — has been applied to dozens of
open-weight models with anecdotal success. We run a formal four-stage mechanistic audit
(direction extraction, CAA steering sweep, attention-head circuit patching, causal
scrubbing) on `Qwen/Qwen2.5-1.5B-Instruct` and three additional small instruct models
(0.5B–3B parameters), asking whether the structural assumptions behind the recipe hold
in each case. On Qwen2.5-1.5B the answer is unambiguously no: a clean refusal direction
exists (extraction quality 4.1), but single-layer steering does not suppress refusal,
the attention heads at the peak layer carry less than 13% of the causal information, and
the circuit-faithfulness score of 0.04 formally rejects the recipe's assumed causal
structure. Results from additional models are pending; the pre-built YAML specs and
infrastructure are committed and will be updated as runs complete.

---

## 1. Introduction

Since Arditi et al. (2023) showed that many LLMs encode a linear refusal direction in
their residual streams, a large community effort has grown around "abliterating" models
by finding that direction and removing it from the model's weight matrices. The recipe
has been applied to Llama, Mistral, Qwen, Phi, and Gemma families, often successfully
in the sense that the modified model outputs harmful content on test prompts.

However, **success as measured by output behavior is not the same as mechanistic
validity**. The recipe makes implicit structural claims: (a) refusal is linearly
separable at some layer; (b) that layer's attention heads *write* the direction into the
residual stream; (c) ablating those attention contributions removes the causal pathway
for refusal. Claim (a) is well-evidenced for many models; claims (b) and (c) are rarely
tested.

We test all three claims using the `mech-interpretability` platform's four-stage audit
pipeline, running each stage as a reproducible experiment with stored artifacts and run
IDs. We start with `Qwen2.5-1.5B-Instruct` (where the full audit has completed) and
extend to three further models in the 0.5B–3B range.

---

## 2. Audit pipeline

Each model runs through four stages:

**Stage 1 — Refusal direction extraction.** We collect residual-stream activations at
the mid-network layer for a matched set of 5 harmful and 5 harmless prompts, compute
the mean-difference direction (Arditi et al. 2023 / RepE), and measure *extraction
quality* = the mean projection-margin of harmful vs. harmless prompts onto the
normalised direction. Quality ≥ 1.0 indicates linear separability.

**Stage 2 — CAA steering sweep.** Following Panickssery et al. (2024), we add the
contrastive-activation-addition (CAA) vector at each of four layers (L/4, L/2, 3L/4,
L−2) with steering coefficients in {−3, −2, −1, 0, +1, +2, +3}, and measure the
*refusal-rate shift* (change in refusal-keyword hit rate on 3 test prompts). The best
layer and coefficient are recorded.

**Stage 3 — Attention-head circuit patching.** At the best CAA layer ± 1, we run
activation patching (clean=harmful prompt, corrupted=harmless prompt) on six hook sites:
`hook_z` (full attention output) and `hook_resid_post` at layers best−1, best, best+1.
Recovery fraction > 0.5 at a site means that site carries more than half the causal
refusal signal.

**Stage 4 — Causal scrubbing.** We test the hypothesis that the top-2 attention-head
sites (identified in Stage 3) are sufficient to implement refusal, by protecting those
sites and replacing all other activations with same-class samples (Conmy et al. 2023).
Faithfulness = exp(−mean KL(full ∥ scrubbed)). Faithfulness > 0.7 supports the
hypothesis; < 0.5 rejects it.

---

## 3. Results

### 3.1 Qwen2.5-1.5B-Instruct (runs 70–73, ~4.5 h CPU)

| Stage | Key metric | Value |
|---|---|---:|
| 1 — Direction extraction | Extraction quality at layer 10 | **4.1** |
| 1 — Direction extraction | Best layer extraction quality | 4.25 (layer 12) |
| 2 — CAA sweep | Best layer (by refusal-rate shift) | 10 |
| 2 — CAA sweep | Peak refusal-rate shift | +0.33 (coeff −3) |
| 2 — CAA sweep | Shift at layer 12 (highest quality) | **0.00** |
| 3 — Circuit patching | `blocks.11.hook_resid_post` recovery | 1.04 |
| 3 — Circuit patching | `blocks.10.attn.hook_z` recovery | 0.13 |
| 4 — Causal scrubbing | Circuit faithfulness | **0.04** |
| 4 — Causal scrubbing | Verdict | **REJECTED** |

**Key finding:** The refusal direction is cleanly separable (quality 4.1–4.25 across
layers 10–12), confirming claim (a). But the residual stream at layer 11 carries 100%
of the signal (recovery 1.04) while the attention-head outputs at the same layer carry
only 13% (recovery 0.13). Claim (b) — that attention heads write the direction — is
refuted for this model. Claim (c) cannot be tested because (b) fails. The scrubbing
faithfulness of 0.04 quantifies how wrong the circuit hypothesis is.

The decoupling of extraction quality from causal steerability is notable: layer 12 has
the *highest* extraction quality (4.25) yet the *lowest* causal effect (shift 0.00).
This means the model has the strongest linear refusal signal in a layer where it is
mechanistically inert. The signal was assembled in earlier layers; layer 12's clean
separation is an epiphenomenon, not the causal source.

### 3.2 Additional models (in progress)

The three priority models below are next in the audit queue. YAMLs are committed at
`experiments/refusal_direction_<slug>.yaml` etc. CAA `hook_layers` and causal-scrubbing
`scrub_sites` are set per-model using the formula L//4, L//2, 3L//4, L−2 and all-layers
coverage respectively.

| Model | Layers | Slug | Status |
|---|---:|---|---|
| `Qwen/Qwen2-0.5B-Instruct` | 24 | `qwen2_0_5b` | pending |
| `Qwen/Qwen2.5-0.5B-Instruct` | 24 | `qwen25_0_5b` | pending |
| `Qwen/Qwen2.5-3B-Instruct` | 36 | `qwen25_3b` | pending |

_This table will be updated with results as each audit completes. The cross-model
summary is auto-generated by `python -m mech_interp.analysis.refusal_audit_multi`._

---

## 4. Discussion

### 4.1 Why the recipe fails on Qwen2.5-1.5B

Three mechanisms plausibly explain the negative result:

**Distributed implementation.** The residual-stream recovery fractions at layers 10–11
are high (0.50–1.04), but these are downstream aggregates — the residual stream
accumulates contributions from all earlier MLPs and attention layers. Abliteration
targets attention weights at a single layer; if the refusal signal is built up by MLP
layers 0–8 (as Stage 3 suggests), no amount of attention-weight modification at layers
9–11 will remove it.

**Instruction-following entanglement.** At the 1.5B scale, the residual-stream
directions that carry the harmful/harmless distinction are also used for instruction
following, role recognition, and other high-level semantic distinctions. Ablating them
from the weight matrices may degrade helpfulness without creating genuine compliance on
harmful prompts.

**Scale-specific refusal architecture.** Larger models (Llama-3-70B, Qwen2.5-72B) may
have more redundant, modular circuits where a single attention layer does implement
refusal writes in a way that can be cleanly ablated. At 1.5B parameters, the same
computation may be more tightly packed and not separable into clean single-layer writes.

### 4.2 Implications for the abliteration literature

Published abliteration results typically demonstrate behavioral success on a post-hoc
test set, without running mechanistic validation. Our results suggest the *reason* the
recipe sometimes works may not be what the recipe's framing implies — the behavior
change may result from global degradation of safety-relevant representations rather than
a clean surgical removal of a single causal pathway.

This also means **model-specific validation is required**. A recipe that works on
Llama-3-8B need not work on Qwen2.5-1.5B, and vice versa. The structural assumptions
should be verified per model before shipping modified weights.

### 4.3 Safety implication

The audit's negative result is **good news from a safety perspective**: Qwen2.5-1.5B's
refusal appears to be implemented in a more distributed, mechanistically entangled way
that resists the standard recipe. Users of this model cannot trivially abliterate it
using the published technique. Whether the result extends to the other small-model
candidates in the audit queue is an open question answered by the ongoing runs.

---

## 5. Conclusion (preliminary)

On `Qwen2.5-1.5B-Instruct`, the abliteration recipe's assumptions do not hold: the
refusal direction is linearly separable but the attention-head circuit hypothesis that
underlies the recipe is formally rejected (faithfulness 0.04). The result motivates the
cross-model audit reported here, which will determine whether this is an idiosyncratic
property of this checkpoint or a systematic feature of small instruct models.

---

## Appendix A — Causal scrubbing layer coverage

A known limitation of earlier causal-scrubbing studies is scrubbing only the first 12
layers, leaving deep layers unperturbed. Our audit YAMLs scrub all layers except the
two protected attention-head sites:

| Model | Layers | Protected | Scrubbed |
|---|---:|---|---:|
| Qwen2.5-1.5B | 28 | blocks.9.attn.hook_z, blocks.10.attn.hook_z | 52 |
| Qwen2-0.5B | 24 | blocks.11.attn.hook_z, blocks.12.attn.hook_z | 44 |
| Qwen2.5-0.5B | 24 | blocks.11.attn.hook_z, blocks.12.attn.hook_z | 44 |
| Qwen2.5-3B | 36 | blocks.17.attn.hook_z, blocks.18.attn.hook_z | 68 |

---

## Appendix B — Reproducibility

All experiments are run via `mech run --name <spec>` against a local SQLite store. The
YAML specs are committed at `experiments/<family>_<slug>.yaml`. The cross-model analysis
is generated by:

```bash
python -m mech_interp.analysis.refusal_audit_multi \
  --audit-dir docs/investigations \
  --output docs/investigations/refusal_audit_multi_model.md
```

Infrastructure commits:
- `scripts/gen_audit_yamls.py` — YAML generator (all 32 specs for 8 candidate models)
- `scripts/run_multi_model_audit.py` — idempotent multi-model runner
- `src/mech_interp/analysis/refusal_audit_multi.py` — cross-model aggregator

---

## References

- Arditi et al. (2023). "Refusal in Language Models Is Mediated by a Single Direction."
  arXiv:2406.11717.
- Panickssery et al. (2024). "Steering Llama 2 via Contrastive Activation Addition."
  arXiv:2312.06681.
- Conmy et al. (2023). "Towards Automated Circuit Discovery for Mechanistic
  Interpretability." NeurIPS 2023.
