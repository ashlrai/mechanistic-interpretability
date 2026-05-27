#!/usr/bin/env bash
# demo.sh — Run every registered experiment family end-to-end through the CLI.
# Prints: spec name | family | elapsed seconds | exit status | key metric.
# Skips families that require HF_TOKEN or large multi-model downloads:
#   refusal_direction, sae_cross_model, cross_model_representation_probe, crosscoder.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UV="uv run --group dev --extra interp"

# ─── colour helpers ────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; RED='\033[0;31m'; CYAN='\033[0;36m'; RESET='\033[0m'

pass() { printf "${GREEN}PASS${RESET}"; }
fail() { printf "${RED}FAIL${RESET}"; }

# ─── run one spec ──────────────────────────────────────────────────────────────
run_spec() {
    local spec_name="$1"
    local family="$2"
    local key_metric="$3"

    printf "\n${CYAN}%-45s${RESET} family=%-32s  " "$spec_name" "$family"
    local t0; t0=$(date +%s)

    local out
    if out=$(cd "$REPO_ROOT" && $UV mech run --name "$spec_name" 2>&1); then
        local t1; t1=$(date +%s)
        local elapsed=$(( t1 - t0 ))
        local metric=""
        if [[ -n "$key_metric" ]]; then
            metric=$(echo "$out" | grep -oE "${key_metric}[^,\n]*" | head -1 || true)
        fi
        printf "  %ds  " "$elapsed"
        pass
        printf "  %s\n" "${metric:-}"
    else
        local t1; t1=$(date +%s)
        local elapsed=$(( t1 - t0 ))
        printf "  %ds  " "$elapsed"
        fail
        printf "\n"
        echo "  ↳ last line: $(echo "$out" | tail -1)"
    fi
}

# ─── validate all YAMLs first ─────────────────────────────────────────────────
echo "=== Validate all experiment specs ==="
cd "$REPO_ROOT"
$UV mech validate
echo ""

echo "=== List registered families ==="
$UV mech experiments
echo ""

echo "=== Running experiment families ==="
echo "────────────────────────────────────────────────────────────────────────────"

# direct_logit_attribution
run_spec "direct-logit-attribution-factual"  "direct_logit_attribution"  "Top positive"

# circuit_patching
run_spec "circuit-patching-smoke"            "circuit_patching"           "recovery"

# attribution_patching
run_spec "attribution-patching-factual-recall" "attribution_patching"    "top_k"

# acdc_lite
run_spec "acdc-lite-gpt2-factual"            "acdc_lite"                  "surviving_nodes"

# polysemanticity_sae
run_spec "polysemanticity-sae-smoke"         "polysemanticity_sae"        "reconstruction_loss"

# sparse_probing
run_spec "sparse-probing-factual-vs-random"  "sparse_probing"             "test_accuracy"

# Skipped (require HF_TOKEN or large multi-model setup):
echo ""
echo "Skipped (require HF_TOKEN or large multi-model downloads):"
echo "  refusal_direction              → needs instruct model + HF_TOKEN"
echo "  sae_cross_model                → needs two models + HF_TOKEN"
echo "  cross_model_representation_probe → needs two models"
echo "  crosscoder                     → needs gpt2 + distilgpt2 (slow)"

echo ""
echo "────────────────────────────────────────────────────────────────────────────"
echo "=== Recent runs ==="
cd "$REPO_ROOT" && $UV mech runs --limit 20

echo ""
echo "=== Aggregate report ==="
cd "$REPO_ROOT" && $UV mech report-runs || echo "(report-runs not available in this environment)"

echo ""
echo "=== Demo complete ==="
