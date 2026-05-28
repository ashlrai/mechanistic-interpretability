# SAEs are not unique solutions: feature dictionaries diverge across random seeds

*Cross-posted from [the platform's findings page](https://ashlrai.github.io/mechanistic-interpretability/publications/sae_replication_crisis/). Full paper draft + reproduction script at the same link.*

## Summary

I trained the same Top-K Sparse Autoencoder on GPT-2 small five times with identical hyperparameters and different random seeds, then asked whether the feature dictionaries are the same. Using optimal bipartite matching on decoder-cosine matrices between every seed pair:

| Condition | Median best-match cosine | Stability fraction at ≥ 0.9 |
|---|---:|---:|
| Layer 0, 128 features, full matrix | 0.095 | 0.16% |
| Layer 0, 128 features, **live-only** | **0.500** | 0.48% |
| Layer 6, 128 features, live-only | 0.323 | 0.00% |
| Layer 6, 512 features, live-only | 0.257 | 0.00% |

**No condition crosses the 0.9 cosine threshold implicit in the "the X feature" claims that pervade the SAE literature.** Deeper layers and larger dictionaries produce *less* stability — consistent with a degenerate-basis story (a richer manifold has more equivalent overcomplete solutions).

## Why I think this matters

The mech-interp community treats SAE features as approximations to intrinsic model properties. Anthropic's "Towards Monosemanticity" speaks of *the* X feature of a model. Auto-interpretability pipelines build natural-language descriptions of individual features and treat them as durable handles. This implicitly assumes SAE training is approximately deterministic up to seed — that two SAEs trained with identical hyperparameters and different initialisations recover (modulo permutation and small noise) the same dictionary.

This work tests that assumption directly. At the scales tested, it fails. Even among live features (excluding the dead-feature confound), the matched cosines are far below the threshold needed to call two features "the same direction."

## What this implies

- **Single-run feature labels** ("feature 47 detects bananas") are training-run properties, not model properties, unless the run is specified.
- **Auto-interp comparisons** across seeds need to match feature *distributions*, not specific paired features.
- **Crosscoder-based model diffing** (Lindsey et al., 2024) is comparing feature distributions between models when each model's features are already seed-unstable — the diff signal is mixed with seed noise.

## Caveats — this is a preliminary report

- **1 model** (GPT-2 small). Should test gpt2-medium, Pythia, Llama at minimum.
- **~1000 training tokens** is 3 orders of magnitude below published SAEs (which use ≥ 10⁶). Larger scale will reduce the dead-feature confound but unlikely to close the 0.5 → 0.9 gap.
- **1 SAE recipe** (Top-K). L1 SAEs, JumpReLU SAEs, gated SAEs may have different seed-stability profiles.
- **No statistical testing** yet — bootstrap CIs on the median + permutation test against random-vector baseline still needed.
- **Refusal-detection metric is keyword-based** and noisy.

## Reproducibility

Every run wrote an `environment.json` artifact with `torch`/`numpy`/`transformer-lens` versions, the `uv.lock` SHA-256, the seed, and a sample of model weight hashes. From a fresh clone:

```bash
uv sync --group dev --extra interp
mech sweep --base experiments/polysemanticity.yaml \
  --axis "parameters.seed=1,2,3,4,5" --execute
mech analyze-sae-stability --runs 1,2,3,4,5 --live-only
```

Wall-clock: ~5 minutes on a 2024-era Apple Silicon MBP.

The platform I built to run this lives at https://github.com/ashlrai/mechanistic-interpretability. It's MIT-licensed, has 15 experiment families, and a Gradio demo. The same `mech sweep` pattern would generalize to the publishable scope (3 models × 5 layers × 4 sizes × 20 seeds = ~1200 runs, ~1.7 GPU-hours on an A100).

## Related platform finding: ACDC late-layer bias on IOI

While building the platform I also ran our edge-level ACDC approximation (KL-weighted-by-layer-gap, not true path patching) on the canonical Wang et al. 2022 IOI task. **It recovers 3 of 12 canonical heads — the two backup_name_movers (10,7) + (11,10) at 100% recall plus name_mover (10,0).** It misses the entire upstream chain (s_inhibition L7-8, induction L5, duplicate_token L0-3). Faithfulness 0.259, flagged as partial.

This is consistent with the approximation's documented late-layer bias: KL weighted by 1/layer_gap favors writer heads near the answer, misses long-range causal chains. The platform reports it honestly rather than claiming "the IOI circuit" — see `docs/investigations/ioi_canonical_reproduction.md`.

This matters here because the same algorithm was used in the abliteration audit below — the negative result holds, but specific head attributions in that audit should be treated as candidates not conclusions.

## Asks

1. **If you've trained SAEs and have your own seed-stability data, please compare notes.** Especially curious about larger models + L1 / JumpReLU recipes.
2. **If you have GPU access and want to run the publishable-minimum version**, the `mech sweep` command above generalizes to it in one line. Happy to help set it up.
3. **If this result is already known and I missed the citation**, please link it. I searched and didn't find a systematic seed-stability test; happy to be corrected.

— Mason Wyatt
