# Mechanistic Interpretability Platform

**Local mechanistic interpretability research at the speed of curiosity.**

A modular platform for running controlled mech-interp experiments on local language models — no cloud APIs, no multi-hour setup, no guesswork about what the model is actually doing.

<div class="grid cards" markdown>

-   **SAE Replication Crisis**

    ---

    Trained 5 identical Top-K SAEs on GPT-2 small with different seeds.
    **Median best-match cosine: 0.500 at layer 0, 0.323 at layer 6.**
    Stability fraction at the ≥0.9 threshold: effectively zero.

    The feature dictionaries are not unique solutions.

    [:octicons-arrow-right-24: Read the investigation](investigations/sae_replication_crisis.md)

-   **Qwen Refusal Audit — Negative Result**

    ---

    For `Qwen2.5-1.5B-Instruct`, refusal IS linearly separable (quality 4.2 at layer 12),
    but it is **NOT controllable** via single-layer steering.
    The standard abliteration recipe fails on this model.

    Small instruct models may have more distributed safety properties than the literature assumes.

    [:octicons-arrow-right-24: Read the investigation](investigations/refusal_audit.md)

-   **GPT-2 Small: How It Recalls Facts**

    ---

    A 4-site circuit achieves **72% faithfulness** on factual recall.
    The correct token is committed at **layer 9** — sharp phase transition from rank 375 to 12.8.
    L9.MLP writes directly on the unembedding direction; L8.MLP suppresses competitors.

    [:octicons-arrow-right-24: Read the investigation](investigations/gpt2_factual_recall.md)

</div>

---

## Try it in 19 seconds

```bash
# Install (Apple Silicon recommended; CPU works too)
pip install mech-interpretability[interp]

# Run the Gradio demo — no config needed
mech demo
```

The demo spins up a local Gradio app at `http://localhost:7860`. You can run circuit patching,
logit lens, and SAE feature analysis directly in the browser against GPT-2 small.

---

## What the platform does

The platform treats each experiment as a YAML spec. You write the spec, the platform handles
model loading, activation capture, sweep parallelism, artifact storage, and report generation.

```yaml
# experiments/polysemanticity.yaml
name: polysemanticity-sae-layer0
family: polysemanticity_sae
model: gpt2
hook_site: blocks.0.hook_resid_pre
parameters:
  n_features: 128
  k: 8
  seed: 42
corpus: openwebtext-100
```

```bash
mech run --name polysemanticity-sae-layer0
mech analyze-sae-stability --sweep experiments/sweeps/sae_seed_stability.yaml
mech report --run-id 42
```

**Supported experiment families:**
circuit patching · ACDC · refusal direction · CAA steering · logit lens ·
DLA · attribution patching · sparse probing · SAE crosscoder · causal scrubbing

---

## Gradio Demo — 4-panel overview

| Panel | What you see |
|-------|-------------|
| **Circuit Patching** | Activation patch any (layer, site) pair; watch logit diffs update live |
| **Logit Lens** | Token probability by layer — see exactly when the model commits to an answer |
| **SAE Features** | Top activating prompts per feature, dead-feature count, decoder cosine matrix |
| **Steering** | CAA or refusal-direction vector injection; compare outputs side-by-side |

---

## Install & hardware requirements

=== "Apple Silicon (recommended)"

    ```bash
    pip install mech-interpretability[interp,apple]
    ```

    Runs GPT-2 small in <2 s on M-series. Full Qwen2.5-1.5B audits in ~4.5 hours CPU.

=== "CUDA"

    ```bash
    pip install mech-interpretability[interp]
    ```

    Standard TransformerLens + PyTorch. Tested on A100 for the publishable-scale experiments.

=== "CPU only"

    ```bash
    pip install mech-interpretability
    ```

    All CLI and cockpit features work. Experiment runners require `[interp]`.

---

## Links

- [GitHub repository](https://github.com/ashlrai/mechanistic-interpretability)
- [Investigations index](investigations/index.md) — all findings with headline numbers
- [Publications](publications/index.md) — paper drafts and tweet threads
- [CLI reference](reference/cli.md)
- [Getting started guide](getting-started.md)
