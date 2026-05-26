#!/usr/bin/env bash
set -euo pipefail

# Smoke the spine: validate every YAML, list registered families, run each of
# the real families end-to-end against the lightweight defaults, and dump the
# resulting run table. Skips ``superposition`` because that family isn't wired
# yet (the runner's placeholder gate would otherwise block it).

uv run --group dev mech validate
uv run --group dev mech experiments

uv run --group dev --extra interp mech run --name circuit-patching-smoke
uv run --group dev --extra interp mech run --name polysemanticity-sae-smoke
uv run --group dev --extra interp mech run --name acdc-lite-gpt2-factual

uv run --group dev mech runs
