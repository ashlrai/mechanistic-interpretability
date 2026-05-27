# Experiment Families

Each experiment family is a registered class that implements a standardized `run()` interface.
Families are declared in `src/mech_interp/experiments/families.py` and implemented in
`src/mech_interp/experiments/<family>.py`.

## Registered families

| Family key | Description | Source paper(s) | Implementation |
|-----------|-------------|-----------------|----------------|
| `polysemanticity` | Baseline polysemanticity probe — measures superposition in MLP neurons | Elhage et al. 2022, "Toy Models of Superposition" | `experiments/polysemanticity.py` |
| `polysemanticity_sae` | Top-K Sparse Autoencoder training on a hook site; reports live/dead features, explained variance, decoder norms | Cunningham et al. 2023; Bricken et al. 2023; Gao et al. 2024 | `experiments/polysemanticity_sae.py` |
| `superposition` | Direct superposition measurement via singular value analysis of weight matrices | Elhage et al. 2022 | `experiments/polysemanticity.py` |
| `circuit_patching` | Activation patching (resample ablation) at any (layer, site) pair to measure causal contribution | Wang et al. 2022, "Interpretability in the Wild" | `experiments/circuit_patching.py` |
| `acdc_lite` | Lightweight ACDC — edge-level circuit discovery without full path enumeration | Conmy et al. 2023, "Towards Automated Circuit Discovery" | `experiments/acdc_lite.py` |
| `acdc_edge` | Full ACDC with IOI validation; exhaustive edge attribution and pruning | Conmy et al. 2023 | `experiments/acdc_edge.py` |
| `refusal_direction` | Extract and characterize the linear refusal direction in residual stream via PCA/SVM on harmful vs. harmless activations | Arditi et al. 2024, "Refusal in LLMs is mediated by a single direction" | `experiments/refusal_direction.py` |
| `caa_steering` | Contrastive Activation Addition — inject a steering vector and measure behavioral shift | Turner et al. 2023, "Activation Addition: Steering Language Models Without Optimization" | `experiments/caa_steering.py` |
| `logit_lens` | Project residual stream at every layer through the unembedding; track per-token rank and probability | nostalgebraist 2020; Belrose et al. 2023, "Eliciting Latent Predictions from Transformers" | `experiments/logit_lens.py` |
| `direct_logit_attribution` | Decompose final logit into per-component (MLP, attention head, embed) contributions | Elhage et al. 2021, "A Mathematical Framework for Transformer Circuits" | `experiments/direct_logit_attribution.py` |
| `attribution_patching` | Gradient x activation attribution patching — efficient approximation of activation patching | Nanda 2023, "Attribution Patching: Activation Patching At Industrial Scale" | `experiments/attribution_patching.py` |
| `sparse_probing` | Fit sparse linear probes on residual stream activations to detect linear representations | Gurnee et al. 2023, "Finding Neurons in a Haystack" | `experiments/sparse_probing.py` |
| `cross_model_representation_probe` | Compare representations across model families by projecting into a shared probe space | — | `experiments/cross_model_representation_probe.py` |
| `sae_cross_model` | Compare SAE feature dictionaries learned from different models on the same corpus | Lindsey et al. 2024, "Sparse Crosscoders for Cross-Layer Features" | `experiments/sae_cross_model.py` |
| `crosscoder` | Train a sparse crosscoder to find shared features across layers or model variants | Lindsey et al. 2024 | `experiments/crosscoder.py` |
| `causal_scrubbing` | Causal scrubbing — test hypotheses about computational circuits by re-running forward passes with controlled substitutions | Chan et al. 2022, "Causal Scrubbing" | `experiments/causal_scrubbing.py` |

## Backend support

| Family | TransformerLens | NNsight | MLX |
|--------|----------------|---------|-----|
| `polysemanticity_sae` | Yes | — | — |
| `circuit_patching` | Yes | — | — |
| `acdc_lite` / `acdc_edge` | Yes | — | — |
| `refusal_direction` | Yes | Yes | — |
| `caa_steering` | Yes | Yes | — |
| `logit_lens` | Yes | — | — |
| `direct_logit_attribution` | Yes | — | — |
| `attribution_patching` | Yes | — | — |
| `sparse_probing` | Yes | Yes | — |
| `causal_scrubbing` | Yes | — | — |
| `crosscoder` | Yes | — | — |

## Adding a new family

1. Add a `StrEnum` entry to `ExperimentFamily` in `families.py`.
2. Implement `class MyFamily(BaseExperiment)` in `experiments/my_family.py`.
3. Register it in `experiments/registry.py`.
4. Add a YAML spec under `experiments/`.
5. Add unit tests in `tests/test_my_family.py`.

See `experiments/polysemanticity_sae.py` for a reference implementation.
