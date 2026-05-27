# SAE Replication Crisis — tweet thread draft

A 12-tweet thread for mech-interp Twitter. Tone: direct, numbers-first, no hype. Each tweet's draft text is below; line breaks within a tweet are explicit.

---

**1/12** We trained the same Top-K SAE on GPT-2 small 5 times — same hyperparameters, same data, only the seed differed.

The feature dictionaries are essentially independent of each other.

Here's what that means for "the X feature of model M" claims 🧵

**2/12** Setup:
- Top-K SAE, n_features=128, k=8, 8 epochs
- gpt2-small `blocks.0.hook_resid_pre`
- 100-doc corpus (~1000 tokens)
- Seeds 1, 2, 3, 4, 5
- All else identical (torch/numpy/random all pinned)

For each pair of seeds we run Hungarian matching on the decoder-cosine matrix.

**3/12** Layer-0, all features:
**Median best-match cosine: 0.095**
Stability at ≥0.9: **0.16%** (2 of 1280 pairs)

That's basically a random unit vector in 768d.

But — 66% of features at this scale are dead. So this number is inflated. Restrict to live features:

**4/12** Layer-0, **live features only**:
**Median best-match cosine: 0.500**
Stability at ≥0.9: 0.48%

Live features partially overlap, but the overlap is nowhere near "same feature."

You'd think a more structured layer would be more stable. Let's check layer 6:

**5/12** Layer-6, live features only:
- 128 features: median = **0.323**
- 512 features: median = **0.257**

Deeper layers and larger dictionaries are *less* seed-stable, not more.

Consistent with a degenerate-basis story: more overcomplete solutions = more equivalent dictionaries.

**6/12** Concretely, what does this mean?

Single best-match pair found across all 1280 pairs: cosine 0.927. Both features fire on Paris/Eiffel-Tower content — this is genuinely "the same feature."

But it's 1 in 1280.

The other 99.92% are different directions, even with optimal matching.

**7/12** Implication 1 — auto-interp labels

If you label features in seed-1 with an LLM, those labels apply ONLY to seed-1. You can't carry them over to seed-2.

Auto-interp pipelines that don't pin seeds are labelling artifacts of training, not the model.

**8/12** Implication 2 — single-run feature claims

"Feature 47 of layer 8 detects X" is meaningful iff you specify the training seed.

Without that, it's not a model property. It's a training-run property.

This affects almost every SAE feature description in the literature.

**9/12** Implication 3 — crosscoders / model diffing

Crosscoders (Lindsey et al. 2024) match features across two MODELS.

If a single model's features are seed-unstable, model-vs-model feature comparison needs to be a comparison of *distributions*, not specific paired features.

**10/12** What this is NOT:

- 1 model (gpt2-small). Not a claim about Gemma, Llama, Claude.
- 1 corpus (100 docs). Not at published-SAE scale.
- 1 SAE recipe (Top-K). Not a claim about L1 SAEs or JumpReLU.

What we'd need to publish this for real: 3 models × 5 layers × 4 sizes × 20 seeds = ~1200 runs. ~1.7 A100-hours.

**11/12** Full reproducibility:

Every run wrote an `environment.json` with torch/numpy/transformer-lens versions, uv.lock SHA-256, seed, model weight hash.

`reproduce.sh` runs the whole experiment in under 5 min on a 2024 Apple Silicon machine.

github.com/ashlrai/mechanistic-interpretability

**12/12** If you've trained SAEs and have your own seed-stability data, I would love to compare.

If you have GPU and want to run the publishable-minimum version (1200 SAEs, multi-model), the platform's `mech sweep` does this in one line. DMs open.

---

## Notes for posting

- Replace "github.com/ashlrai/mechanistic-interpretability" with the actual canonical URL once chosen.
- Pin tweet 1 with the heatmap from notebook 06 (cell that renders the pairwise-stability matrix).
- Reply-with-figure to tweet 4 with the live-only distribution histogram.
- If the response is strong, prepare a 2-paragraph response template for: "have you tried X SAE recipe?" — be honest that we haven't.
