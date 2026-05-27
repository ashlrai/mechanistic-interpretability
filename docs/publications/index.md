# Publications

Publication artifacts from the platform's investigations.

## Artifacts

| Title | Type | Investigation | Status |
|-------|------|---------------|--------|
| [SAE Replication Crisis — preprint draft](sae_replication_crisis.md) | Paper (preprint) | Investigation #3 | Draft — needs multi-model + GPU scale-up |
| [SAE Replication Crisis — tweet thread](sae_replication_crisis_thread.md) | 12-tweet thread | Investigation #3 | Draft ready to post |

## Publication pipeline

The platform is designed to go from investigation to artifact with minimal friction:

1. **Run the investigation** — `mech run` / `mech sweep`
2. **Generate the report** — `mech report --run-id <id>`
3. **Write the narrative** — under `docs/investigations/`
4. **Produce artifacts** — paper draft + tweet thread under `docs/publications/`

The SAE replication crisis is the first investigation to reach the publication stage.
The refusal audit (Investigation #1) and factual recall story (Investigation #2) are
candidates for the next round once the platform's multi-model sweep infrastructure
is exercised at publishable scale.

## Scale gap

The current results are single-model, single-corpus, small dictionary size.
The publishable-minimum version of the SAE replication crisis requires:

- 3+ models (GPT-2 small, GPT-2 medium, Pythia-1.4B at minimum)
- 5 layers per model
- 4 dictionary sizes (128, 256, 512, 1024)
- 20+ seeds
- ~1200 training runs, ~10 GPU-hours on A100

The analysis pipeline (`mech analyze-sae-stability --live-only`) is ready.
Compute and model diversity are the remaining gap.
