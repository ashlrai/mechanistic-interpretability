# Getting Started

## Requirements

- Python 3.11+
- Apple Silicon MBP with 16+ GB RAM (recommended) or any x86 machine with CUDA
- ~5 GB disk for GPT-2 weights and experiment artifacts

## Install

=== "Full (interp + apple)"

    ```bash
    pip install mech-interpretability[interp,apple]
    ```

=== "Interp only (CUDA / CPU)"

    ```bash
    pip install mech-interpretability[interp]
    ```

=== "From source"

    ```bash
    git clone https://github.com/ashlrai/mechanistic-interpretability
    cd mechanistic-interpretability
    pip install -e .[interp]
    ```

## Verify the install

```bash
mech validate
```

This runs a fast TransformerLens smoke test on GPT-2 small (CPU, ~10 s) and confirms
that the experiment registry, storage layer, and CLI all work.

Expected output:

```
TransformerLens smoke OK  (gpt2, 1 forward pass)
Storage layer OK          (sqlite at ~/.mech_interp/runs.db)
Experiment registry OK    (16 families registered)
All checks passed.
```

## Run the Gradio demo

```bash
mech demo
```

Opens `http://localhost:7860`. No configuration needed — the demo uses GPT-2 small
and ships with built-in prompt sets for all four panels.

## Your first experiment

The fastest path to a real result is the SAE seed-stability sweep on GPT-2 small:

```bash
# Train 5 SAEs with seeds 1-5 (takes ~3 min on Apple Silicon)
mech sweep \
  --base experiments/polysemanticity.yaml \
  --axis "parameters.seed=1,2,3,4,5" \
  --output experiments/sweeps/sae_seed_stability.yaml \
  --execute

# Compute pairwise alignment
mech analyze-sae-stability \
  --sweep experiments/sweeps/sae_seed_stability.yaml \
  --output artifacts/my_stability_report.json

# Open the results in the cockpit
mech cockpit
```

Navigate to the **SAE Stability** tab to see the pairwise cosine heatmap.

## Reproduce a specific investigation

Each investigation page has an exact command block at the bottom. For example,
[SAE Replication Crisis](investigations/sae_replication_crisis.md) gives you the
five sweep commands that reproduce the full 4-condition comparison.

## Next steps

- [Investigations index](investigations/index.md) — read what we've found
- [Experiment families reference](reference/families.md) — understand what each family does
- [CLI reference](reference/cli.md) — full command listing
