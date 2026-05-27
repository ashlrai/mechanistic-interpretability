#!/usr/bin/env bash
# Reproduce the SAE replication crisis result from a fresh clone of the repo.
# Total wall-clock: ~5 minutes on a 2024-era Apple Silicon machine.
#
# Prerequisites:
#   - uv installed
#   - clone of github.com/ashlrai/mechanistic-interpretability
#   - working directory = the repo root
set -euo pipefail

# 1. Set up the dev environment with the interp extra (torch + transformer-lens).
uv sync --group dev --extra interp

# 2. Initialize the local result store (creates artifacts/runs.sqlite3).
uv run mech init-store

# 3. Train 5 layer-0, 128-feature SAEs with seeds 1..5.
uv run mech sweep \
  --base experiments/polysemanticity.yaml \
  --axis "parameters.seed=1,2,3,4,5" \
  --output experiments/sweeps/sae_seed_stability.yaml \
  --execute

# 4. Train 5 layer-6, 128-feature SAEs (mid-network probe).
uv run mech sweep \
  --base experiments/polysemanticity_sae_layer6.yaml \
  --axis "parameters.seed=1,2,3,4,5" \
  --output experiments/sweeps/sae_seed_stability_layer6.yaml \
  --execute

# 5. Train 5 layer-6, 512-feature SAEs (larger dictionary).
uv run mech sweep \
  --base experiments/polysemanticity_sae_layer6_512.yaml \
  --axis "parameters.seed=1,2,3,4,5" \
  --output experiments/sweeps/sae_seed_stability_layer6_512.yaml \
  --execute

# 6. Compute pairwise stability reports.
#    The exact run-IDs depend on your local state — list them with `mech runs`
#    and substitute below. The reports land alongside the original investigation.
#
#    Example (replace IDs with your actual run IDs):
#
#    uv run mech analyze-sae-stability --runs 1,2,3,4,5 \
#      --output artifacts/seed_stability_layer0.json
#    uv run mech analyze-sae-stability --runs 6,7,8,9,10 --live-only \
#      --output artifacts/seed_stability_layer0_liveonly.json
#    uv run mech analyze-sae-stability --runs 11,12,13,14,15 --live-only \
#      --output artifacts/seed_stability_layer6_liveonly.json
#    uv run mech analyze-sae-stability --runs 16,17,18,19,20 --live-only \
#      --output artifacts/seed_stability_layer6_512_liveonly.json

# 7. The notebook narrates the analysis end-to-end.
echo "Now open notebooks/06_sae_replication_crisis.ipynb and run all cells."
