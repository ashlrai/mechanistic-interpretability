# How reproducible are SAE features across seeds? A small-scale measurement (and a scale control)

*Cross-posted from [the platform's findings page](https://ashlrai.github.io/mechanistic-interpretability/). Reproduction scripts + raw data linked at the bottom.*

## Summary

I trained Top-K Sparse Autoencoders on GPT-2 small with identical hyperparameters and different random seeds, then measured how well the learned feature dictionaries align across seeds (optimal bipartite matching on decoder-cosine matrices, restricted to live features). The headline:

**SAE feature reproducibility improves substantially with training scale, but even after removing the dead-feature confound and training on 30× more data, the median cross-seed best-match cosine is only ~0.47 — far below the ~0.9 bar implied when we call something "the X feature of model M". Only ~4% of features reach that bar.**

This is a small-scale measurement (GPT-2 small, ≤30K training tokens, Top-K SAEs). It is **not** evidence that production-scale SAEs are irreproducible — the trend with scale is upward, and I did not test the 1M–1B-token regime real SAEs use. I'm posting it because the scale-dependence itself is the interesting part, and because I couldn't find a systematic seed-stability measurement in the literature.

## The measurement

Five seeds, identical config, GPT-2 small `blocks.6.hook_resid_pre`, 512 features, k=32. For each seed pair, match each live feature in run A to its best partner in run B and record the cosine of the matched decoder directions.

| Training tokens | Mean dead-feature ratio | Live-only median best-match cosine | Features ≥ 0.9 |
|---:|---:|---:|---:|
| 992 (openwebtext sample) | 0.65 | 0.257 | 0.0% |
| **30,186 (pile-1k)** | **0.085** | **0.472** | **4.33%** |

The first row is easy to dismiss: at ~1000 tokens, 65% of features never activate (they stay at their random init), which drags the matched cosines down. So I ran the **scale control** — same config, 30× the tokens. The dead-feature ratio collapses to 8.5%, and the median best-match cosine rises from 0.26 to 0.47.

Two honest conclusions from that single control:

1. **The instability is not purely a low-data artifact.** Removing the confound and adding 30× data still leaves the median at 0.47 with only 4% of features reaching 0.9.
2. **Reproducibility is clearly scale-sensitive and rising.** 0.26 → 0.47 going 30×. I cannot rule out that it continues toward 0.9 at production scale. That extrapolation is the key open question, not a settled result.

(For context: two random unit vectors in 768-dim have cosine ≈ 0 with std 0.036, so a *median* matched cosine of 0.47 is ~13σ above chance — live features are genuinely partially aligned across seeds, not independent. "Partial alignment that improves with scale" is the accurate description, not "the dictionaries are unrelated.")

## Why I think it's worth measuring

The mech-interp community often treats SAE features as approximations to intrinsic model properties — "the X feature", durable natural-language labels assigned from a single training run, crosscoder feature-matching across models. All of that implicitly assumes seed-stability. The amount of seed-stability is rarely reported. This is a small attempt to put a number on it, with a scale control so the number isn't just a training-budget artifact.

If the ~0.47-at-30K-tokens figure holds up (and especially if it plateaus below 0.9 at larger scale, which I have **not** shown), then single-run feature labels are partly run-specific, and cross-seed / cross-model feature comparisons should compare distributions rather than specific paired features. If instead it climbs to ~0.9 at production scale, the concern dissolves. I genuinely don't know which, and the honest contribution here is the scale-control method plus the two data points.

## Caveats (these are load-bearing)

- **Small scale.** 30K tokens is still 30–30,000× below production SAEs. The upward trend with scale means the headline number is a lower bound on reproducibility, not an upper bound.
- **One model, one recipe.** GPT-2 small, Top-K SAEs. L1 / JumpReLU / gated SAEs and larger models may differ.
- **Greedy bipartite matching** (no scipy in the env) — an upper bound on the optimal assignment; true Hungarian matching would give equal-or-higher cosines, i.e. the real reproducibility is *at least* this good.
- **No statistical testing** — bootstrap CIs and a permutation test against the random baseline are still TODO.
- **n=5 seeds.** Enough to see the effect, not enough for tight intervals.

## Reproducibility

From a fresh clone (≈5 min on a 2024 Apple Silicon laptop for the small run; ~15 min for the scale control):

```bash
uv sync --group dev --extra interp
# small-scale (original):
mech sweep --base experiments/polysemanticity_sae_layer6_512.yaml \
  --axis "parameters.seed=1,2,3,4,5" --execute
# scale control (30x tokens, pile-1k):
mech download-corpus --name pile-1k
mech sweep --base experiments/polysemanticity_sae_layer6_512_scale.yaml \
  --axis "parameters.seed=1,2,3,4,5" --execute
```

Raw pairwise data: `docs/publications/sae_replication_artifacts/` (small-scale) and `scale_control.json` (30× run). Every run records an `environment.json` with library versions, `uv.lock` SHA-256, seed, and model weight hash.

Platform: https://github.com/ashlrai/mechanistic-interpretability (MIT). The `mech sweep` pattern generalizes to the publishable scope (multiple models × layers × dictionary sizes × ≥20 seeds × ≥1M tokens, ≈a couple GPU-hours).

## Two related findings from the same platform

**Abliteration recipe degrades with model scale (4 Qwen models).** Auditing the standard Arditi/RepE refusal-direction recipe across Qwen 0.5B→3B: single-layer additive steering suppresses refusal fully at 0.5B (refusal 0.33→0.00), partially at 0.5B-newer (0.67→0.33), backfires at 1.5B (0.33→0.67), and loses suppression entirely at 3B (a 4-layer CAA sweep finds no layer that suppresses). Each model's refusal direction extracts cleanly (separation quality 2.3–4.3) — extraction quality and steering efficacy are decoupled. **Caveat: 3 test prompts per model with keyword-based refusal detection — coarse; the per-model numbers are noisy, but the monotonic direction across 4 models is the signal.** Detail: `docs/investigations/refusal_audit.md`.

**Edge-level ACDC recovers late-layer IOI heads only.** On the canonical Wang et al. 2022 IOI task, our KL-weighted edge-ACDC approximation recovers 3 of 12 canonical heads (both backup name-movers + one name-mover at 100%/33%), misses the entire upstream chain (s-inhibition, induction, duplicate-token), faithfulness 0.26. This is a documented late-layer bias of the approximation, reported honestly rather than as "the circuit". Detail: `docs/investigations/ioi_canonical_reproduction.md`.

## Asks

1. **If you have seed-stability data on real-scale SAEs, please share it** — the one number I most want is the live-only median best-match cosine at ≥1M tokens. That settles the extrapolation.
2. **If you have GPU budget**, the `mech sweep` command scales to the proper experiment (multiple models/layers/sizes, 20+ seeds, 1M+ tokens) in one line.
3. **If a systematic SAE seed-stability measurement already exists, link it** — I searched and didn't find one; happy to be corrected and to defer.

— Mason Wyatt
