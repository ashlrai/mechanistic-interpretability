# Investigation #4 — Feature Splitting Across SAE Sizes

**Date**: 2026-05-27  
**Runs**: 44 (128 feat), 46 (256 feat), 48 (512 feat), 49 (1024 feat)  
**Notebook**: `notebooks/08_feature_splitting.ipynb`

## Setup

Four Top-K SAEs (k=8) trained on GPT-2 small layer-0 residual stream activations from
a 100-document mixed corpus (geography, biology, code, history).  All use the same
seed=42 and 8 training epochs.  For each consecutive 2× pair, we compute decoder cosine
similarity between every live parent feature and every child feature, reporting the
top-3 children above a 0.3 cosine threshold.

**Mean split fidelity** = mean best-child cosine across all live parent features.
Thresholds: ≥0.80 clean splitting; 0.50–0.80 partial specialisation; <0.50 reshuffle.

## Results

| Pair       | Parent live | Child live | Mean fidelity | Split dist (0/1/2/3+) |
|------------|-------------|------------|:-------------:|------------------------|
| 128 → 256  | 44          | 51         | **0.758**     | 0 / 6 / 3 / 35        |
| 256 → 512  | 51          | 56         | **0.665**     | 0 / 7 / 8 / 36        |
| 512 → 1024 | 56          | 73         | **0.735**     | 1 / 4 / 5 / 46        |

All three transitions are in the partial-specialisation band (0.50–0.80).  No pair
crosses the clean-splitting threshold of 0.80.

## Example Splits

**128→256, parent feat 5** (geography/biology prompts):
- Child 227, cos=0.818 → Paris/Eiffel Tower + Python BinaryTree (mixed)
- Child 127, cos=0.673 → Fibonacci + Paris (code/geography bleed)
- Child 91, cos=0.528 → Python language + Sahara Desert

Interpretation: the parent was broadly "general knowledge"; the children
specialise weakly but don't cleanly separate domains.

**512→1024, parent feat 17** (Python code — BinaryTree, enumerate):
- Child 17, cos=0.720 → BinaryTree + enumerate (closely tracks parent)
- Child 513, cos=0.714 → enumerate + BinaryTree (near-duplicate direction)
- Child 287, cos=0.432 → same code prompts (third, weaker copy)

Interpretation: the 1024-SAE allocated **two near-duplicate** directions for this
feature instead of splitting it semantically — consistent with superposition
theory (nearby directions are used for related but distinct tokens).

**512→1024, parent feat 39** (quantum mechanics / code):
- Child 39, cos=0.894 → Python code (BinaryTree, Fibonacci) — clean inheritance
- Child 556, cos=0.576 → Python language description + DNA (partial)
- Child 93, cos=0.490 → Shakespeare + vaccines (unrelated fragment)

Interpretation: the best child (cos=0.894) is a clean specialisation; the
trailing children are noise/fragmentation rather than semantic refinement.

## Interpretation

The platform data **partially supports** Anthropic's clean-splitting claim:

1. **Structure is preserved.** Almost all live parent features find at least one
   child with cosine >0.3 (only 0–1 parents per pair have zero qualifying children).
   The dictionary is not wholly reshuffling.

2. **But splitting is not clean.** Mean fidelities of 0.665–0.758 mean the best child
   typically explains ~70% of the parent direction but doesn't fully inherit it.
   True clean splitting would require cosines >0.90.

3. **Near-duplicate directions are common.** Several parent features produce two
   child features with near-identical cosines (~0.71/0.71), suggesting the larger
   SAE allocates multiple directions to the same concept rather than splitting into
   semantically distinct sub-concepts.

4. **Live-feature growth is slow.** Doubling the dictionary only adds ~7–17 live
   features (44→51→56→73), meaning >90% of the additional capacity is absorbed by
   dead features on this small corpus.

## Caveats

- **Small corpus**: 992 tokens is far below the scale used in Anthropic's work.
  With more diverse data, splitting may become cleaner.
- **Only 8 epochs**: longer training would reduce dead feature count and sharpen
  decoder directions.
- **Layer 0 only**: residual stream at layer 0 is largely embedding-level; later
  layers with richer representations may split more cleanly.
- **min_cosine=0.3**: raising to 0.5 would cut the "3+ children" bucket but
  improve the semantic precision of reported splits.

## Reproducing

```bash
# Train the 4 SAEs
mech sweep --base experiments/polysemanticity.yaml \
  --axis parameters.n_features=128,256,512,1024 \
  --output experiments/sweeps/sae_feature_splitting.yaml \
  --execute

# Compute splits for each pair (adjust run IDs to match your db)
mech analyze-feature-splits --parent-run 44 --child-run 46
mech analyze-feature-splits --parent-run 46 --child-run 48
mech analyze-feature-splits --parent-run 48 --child-run 49
```
