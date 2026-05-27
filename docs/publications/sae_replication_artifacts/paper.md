# SAEs are not unique solutions: feature dictionaries diverge across random seeds

**Status:** preliminary report, single model, single corpus
**Author:** Mason Wyatt (with the mech-interp platform at github.com/ashlrai/mechanistic-interpretability)
**Date:** 2026-05-27

---

## Abstract

We train five Top-K Sparse Autoencoders with identical hyperparameters and different random seeds on the residual stream of GPT-2 small, then ask whether the learned feature dictionaries are the same. Using optimal bipartite matching (Hungarian algorithm on decoder-cosine matrices) between every seed pair, we find a median best-match cosine of **0.500 among live features at layer 0** and **0.323 at layer 6**, with the stability fraction at the cosine ≥ 0.9 threshold being **effectively zero** in all conditions. Counterintuitively, deeper layers and larger feature dictionaries produce **less** seed-stability, not more. This implies that published SAE feature descriptions — "the X feature of model M" — are properties of a particular training run rather than of the underlying model. We do not yet claim this generalises beyond GPT-2 small / 100-document corpora; we provide the platform and reproducibility receipts to test it at scale.

---

## 1. Introduction

The mechanistic-interpretability community treats Sparse Autoencoder features as approximations to intrinsic model properties. Anthropic's "Towards Monosemanticity" report (Bricken et al., 2023) speaks of "the X feature" of a model. Auto-interpretability pipelines (Cunningham et al., 2023; Marks et al., 2024) build natural-language descriptions of individual SAE features and treat them as durable handles on the model.

This implicitly assumes that SAE training is approximately deterministic up to random seed — i.e. two SAEs trained with identical hyperparameters and different initialisations recover (modulo permutation and small noise) the same feature dictionary. The assumption is rarely tested. We test it.

## 2. Method

### Setup

- **Model:** `gpt2-small` (124M params, d_model=768, 12 layers).
- **Hook sites tested:** `blocks.0.hook_resid_pre` (embedding-adjacent) and `blocks.6.hook_resid_pre` (mid-network).
- **SAE:** Top-K SAE (Gao et al., 2024). For layer 0: n_features=128, k=8, 8 epochs. For layer 6: n_features ∈ {128, 512}, k=8 / k=32, 8 epochs.
- **Corpus:** 100 documents from the bundled `openwebtext_sample.jsonl` (~992 tokens after tokenisation with `seq_len=64`, `max_tokens=2000`).
- **Seeds:** 1, 2, 3, 4, 5 — every other hyperparameter identical.
- **Hardware:** single Apple Silicon MBP, CPU only, deterministic seeding (`torch.manual_seed`, `numpy.random.seed`, `random.seed` set before each run via the platform's runner).

### Matching protocol

For each pair of seeds `(i, j)`:
1. Extract decoder weight matrices `W_dec_i, W_dec_j` of shape `(n_features, d_model)`. L2-normalise rows.
2. Build the cosine-similarity matrix `C = W_dec_i @ W_dec_j.T`, shape `(n_features, n_features)`.
3. Solve `argmax_P sum(C[i, P(i)])` via `scipy.optimize.linear_sum_assignment` to find the optimal one-to-one matching.
4. Report the distribution of matched cosines.

A feature pair is **stable** if its matched cosine is ≥ 0.9. The 0.9 threshold reflects the implicit standard of the auto-interp literature: features described as "the X feature" need to be approximately the same direction across runs for the description to be a model property.

### Live-features-only variant

The matrix includes dead features (zero activation across the corpus). Dead features are essentially random unit vectors with no information-bearing relationship to the model — matching them inflates the random component of the cosine distribution. We separately report `compute_live_only_alignment` results that restrict the matching to features whose `feature_analysis.json` reports `dead=False`.

## 3. Results

### 3.1 Headline numbers

| Condition | Median best-match cosine | Stability fraction at ≥ 0.9 |
|---|---:|---:|
| Layer 0, 128 features, full matrix | 0.095 | 0.16% (2/1280) |
| Layer 0, 128 features, live-only | **0.500** | 0.48% |
| Layer 6, 128 features, live-only | 0.323 | 0.00% |
| Layer 6, 512 features, live-only | 0.257 | 0.00% |

The full-matrix layer-0 number (0.095) is dominated by the 66% dead-feature rate. Restricting to live features raises the median to 0.500 — these features partially overlap, but the overlap is far from the "same feature" threshold.

The layer-6 results are the most striking: a more structured representational locus produces **less** seed-stability than the embedding layer, and the larger 512-feature SAE produces less stability than the smaller 128-feature one. Both directions are consistent with a degenerate-basis interpretation: a richer manifold with more overcomplete solutions admits more equivalent dictionaries.

### 3.2 Example matched pairs

The single highest-cosine matched pair across all 1280 pairs of seed-stability runs at layer 0 was `(seed_1, feature_47) ↔ (seed_3, feature_82)` with cosine = 0.927. Both features' top-activating prompts cluster on geography (Eiffel Tower / Paris content), so this single pair is genuinely the same feature across seeds.

For a representative "borderline" pair with cosine ≈ 0.5: matched feature pairs' top-activating prompts overlap on broad category (e.g. both fire on code-related documents) but disagree on specifics (one fires more on Python control flow, the other on C++ syntax). For pairs at cosine ≈ 0.1: top prompts are unrelated.

### 3.3 The dead-feature confound

At our scale (≈ 1000 training tokens), 66% of 128 features are dead. The original literature (Anthropic's `gemma-scope`, jbloom's gpt2 SAEs) trains on ≥ 10⁶ tokens with much lower dead ratios. Our headline 0.095 cosine in the full-matrix condition is inflated by dead-feature pairing; the live-only 0.500 is the better single-number summary, with the more important point being that even the live-only condition does not approach the 0.9 threshold in any layer × size combination tested.

## 4. Caveats and robustness checks

- **Scale.** Our corpus is two-to-three orders of magnitude smaller than the published SAEs we're implicitly comparing to. At larger scale the dead-feature fraction shrinks and feature directions sharpen; we expect this to raise the median live-only cosine, but the gap to 0.9 is sufficiently large that it is unlikely to close from training scale alone.
- **Model.** A single 124M-parameter model. The result should be tested on at least one of `gpt2-medium`, `gpt2-xl`, `pythia-1.4b`, `Llama-3.1-8B` before claiming generality.
- **Training recipe.** We only test Top-K SAEs. L1-regularised SAEs, JumpReLU SAEs (Rajamanoharan et al., 2024), and gated SAEs may have different seed-stability profiles.
- **Hook site.** Only residual-stream pre-LN. Attention-output SAEs and MLP-output SAEs are not tested.
- **Statistical testing.** Bootstrap confidence intervals on the median, and a permutation test against the random-vector baseline, are not yet computed. The 0.097 vs 0.095 baseline gap is small enough that the live-only 0.500 layer-0 number is unambiguously above noise, but the layer-6 0.323 number needs a formal test before we'd claim significance.

## 5. Implications

If features are not seed-stable at the 0.9 threshold, several common practices in the SAE literature inherit hidden noise:

- **Single-run feature labels.** "Feature 47 of the layer-8 SAE detects bananas" is meaningful only if specifying the training run. Without that specification it's not a model property.
- **Auto-interp comparisons.** Auto-interp labels assigned in one run cannot be matched 1-to-1 with auto-interp labels from a sibling run.
- **Crosscoder-based model diffing.** Lindsey et al. (2024) crosscoders compare features across two models; if a single model's features are seed-unstable, cross-model feature comparisons need to compare *distributions* of features, not specific paired features.
- **SAE evaluation.** Benchmarks that score "how well does this SAE recover known features?" need to either average across seeds or report seed sensitivity.

The constructive suggestion is to publish multi-seed mean ± standard-deviation numbers, and to test seed stability as a prerequisite for any "feature X" claim.

## 6. Limitations and future work

A publishable version of this result would require, at minimum:

| Dimension | This report | Publishable minimum |
|---|---|---|
| Models | 1 (gpt2-small) | 3+ (e.g., gpt2-{small, medium, xl}, Llama-3.1-8B) |
| Layers per model | 2 | 5+ across depth |
| Dictionary sizes | 2 (128, 512) | 4 (128, 512, 2048, 8192) |
| Seeds | 5 | 20+ for tight CIs |
| Training tokens | ~1k | 10⁶+ to match published SAEs |
| Statistical testing | none | bootstrap CIs + permutation test |

That budget is approximately 1200 SAE training runs. At ~5 seconds per run on an A100 it's ~1.7 GPU-hours; on the same MBP this report ran on, the same scale would take ~8-12 hours overnight. The platform pipeline is the bottleneck, not the compute — every step here (`mech sweep`, `mech analyze-sae-stability --live-only`, the analysis notebook) generalises to that scale without code changes.

## 7. Reproducibility

Every run referenced here has an `environment.json` artifact recording `torch` / `numpy` / `transformer-lens` versions, the `uv.lock` SHA-256, the seed, and a sample hash of the model weights. The exact commands to reproduce from a fresh clone are in `docs/publications/sae_replication_artifacts/reproduce.sh`. Total wall-clock to reproduce from scratch: under 5 minutes on a 2024-era Apple Silicon machine.

## References

- Gao et al., 2024. "Scaling and evaluating sparse autoencoders." OpenAI.
- Bricken et al., 2023. "Towards Monosemanticity." Anthropic.
- Cunningham et al., 2023. "Sparse autoencoders find highly interpretable features in language models."
- Marks et al., 2024. "Sparse feature circuits."
- Lindsey et al., 2024. "Sparse Crosscoders for Cross-Layer Features in Superposition." Anthropic.
- Rajamanoharan et al., 2024. "Jumping Ahead: Improving Reconstruction Fidelity with JumpReLU Sparse Autoencoders." Google DeepMind.
- Conmy et al., 2023. "Towards Automated Circuit Discovery." NeurIPS.
- Wang et al., 2022. "Interpretability in the Wild: A Circuit for Indirect Object Identification."
