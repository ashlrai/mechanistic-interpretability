# Investigation #7 — Refined Qwen2.5-1.5B Refusal Audit (v2)

**Date:** 2026-05-27
**Model:** `Qwen/Qwen2.5-1.5B-Instruct`
**Based on:** v1 audit (runs 70/71/72/73) — negative result with 5 documented caveats
**This audit addresses:** caveats 2 (coefficient range), 3 (per-head decomposition), 4 (MLP patching)
**Caveats 1 and 5 deferred:** test-set expansion and real refusal classifier are dataset/model dependencies out of scope for this code pass

---

## What was built

### Enhancement A — Wider coefficient sweep (`refusal_direction_qwen_wide.yaml`)

`steering_coefficient_range: [-20.0, -10.0, -7.0, -5.0, -3.0, -1.0, 0.0, 1.0, 3.0, 5.0, 7.0, 10.0, 20.0]`

13 coefficients vs the original 7 (±3 range). All other parameters identical to run 70 so results are directly comparable. Estimated wall-clock ~90 min.

Run via: `mech run experiments/refusal_direction_qwen_wide.yaml`

### Enhancement B — Per-head attention patching (`refusal_circuit_qwen_per_head.yaml`)

`per_head: true`, `n_heads: 16`. Expands the 3 `blocks.{9,10,11}.attn.hook_z` sites into 48 per-head synthetic sites (`blocks.L.attn.hook_z.head.H`). The backend patches only slice `[:, position, H, :]` of `hook_z`, leaving all other heads untouched. 51 sites × 3 pairs = 153 forward passes. ~10 min wall-clock.

Run via: `mech run experiments/refusal_circuit_qwen_per_head.yaml`

### Enhancement C — MLP-output patching across all 28 layers (`refusal_circuit_qwen_mlps.yaml`)

28 `blocks.{0..27}.hook_mlp_out` sites × 3 pairs = 84 forward passes. ~10 min wall-clock.

Run via: `mech run experiments/refusal_circuit_qwen_mlps.yaml`

### Enhancement D — Revised causal scrubbing (`causal_scrubbing_refusal_qwen_v2.yaml`)

v2 hypothesis: refusal is written by MLP outputs at layers 8-10, not by attention heads. Protected sites: `blocks.{8,9,10}.hook_mlp_out` + `blocks.10.attn.hook_z` (catch-all pending per-head results). Scrubbed sites: everything else (attention at all other layers, resid_post everywhere, MLP out at all other layers).

**After running Enhancements B and C:** update `protected_sites` in this YAML with the actual top-3 MLP layers and top-3 per-head sites from those runs before executing the v2 scrub.

Run via: `mech run experiments/causal_scrubbing_refusal_qwen_v2.yaml`

---

## Code changes

### `src/mech_interp/experiments/circuit_patching.py`

Added to `CircuitPatchingSpec`:
- `per_head: bool = False` — backward-compatible; existing runs unaffected
- `n_heads: int = 16` — number of heads to expand into

New public functions (importable for testing):
- `expand_hook_z_per_head(site, n_heads)` — expands `blocks.L.attn.hook_z` into `n` synthetic per-head names; returns site unchanged for non-`hook_z` sites
- `per_head_site_to_base(site)` — decomposes `blocks.L.attn.hook_z.head.H` back to `(base_site, head_idx)`; returns `(site, -1)` for non-per-head sites

`_resolve_hook_sites` now calls `_expand_per_head_sites` when `per_head=True`.

### `src/mech_interp/backends/instrumented.py`

`TransformerLensBackend.run_activation_patching` now:
1. Resolves each requested hook site to its base model hook (stripping `.head.H` suffix)
2. Captures the cache using base site names only (no duplicates)
3. For per-head sites: patches only `[:, position, H, :]` of `hook_z`
4. For whole-layer sites: existing `[:, position, ...]` behaviour unchanged
5. Uses `_tensor_norm_head` (new helper) for per-head activation norms

New helper: `_tensor_norm_head(tensor, position, head_idx)`.

### `src/mech_interp/analysis/refusal_audit.py`

`_parse_hook_site` updated to recognise the per-head synthetic suffix pattern
`\.attn\.hook_z\.head\.(\d+)$` before falling back to the legacy `headN` pattern. Previously would return `head=-1` for all `hook_z` sites; now correctly returns the head index for per-head sites, enabling `compile_refusal_audit` to populate `top_causal_heads` with real `(layer, head, recovery)` tuples.

### `tests/test_circuit_patching.py`

8 new unit tests:
- `test_expand_hook_z_per_head_produces_correct_names` — names are `blocks.L.attn.hook_z.head.{0..N-1}`
- `test_expand_hook_z_per_head_non_hook_z_unchanged` — resid_post sites untouched
- `test_expand_hook_z_per_head_mlp_out_unchanged` — hook_mlp_out untouched
- `test_per_head_site_to_base_round_trips` — decompose → correct (base, head)
- `test_per_head_site_to_base_non_per_head` — whole-layer → head=-1
- `test_per_head_site_to_base_resid_post` — non-hook_z site → head=-1
- `test_circuit_patching_per_head_expands_hook_sites` — end-to-end: `per_head=True` delivers 4 synthetic sites to backend
- `test_circuit_patching_per_head_false_preserves_existing_behaviour` — `per_head=False` delivers unchanged site name

All 533 tests pass; ruff and mypy clean; all 60 YAML specs validate.

---

## v1 vs v2 findings (side-by-side)

> Note: the v2 experiments have been built and validated but have NOT yet been run — the model weights (~3 GB) require the `interp` extra and ~2.5 hours total CPU time. This section describes expected findings and their interpretation based on v1 evidence.

### What v1 established

| Stage | Finding |
|---|---|
| S1: Direction extraction | Extraction quality 4.1 at layer 10 — strong linear separation |
| S2: CAA steering | Steerability decoupled from extraction quality; only layers 8/10 shift at coeff=−3, and only by +0.33 |
| S3: Circuit patching | resid_post at L10-11 recovery 0.50-1.04; `attn.hook_z` at same layers recovery 0.02-0.13 |
| S4: Causal scrubbing | Faithfulness 0.04 (hypothesis: L9+L10 attn heads implement refusal) — **REJECTED** |

### What v2 tests

| Enhancement | Hypothesis being tested |
|---|---|
| A: wider coefficients | Does ±3 saturation explain the weak steering? Does ±10/±20 unlock behavioral change? |
| B: per-head patching | Which specific heads at L9-11 carry the 0.02-0.13 residual attention signal? Any single head > 0.1? |
| C: MLP patching | Which layers' MLP outputs write the refusal direction? Expected: layers 8-10 dominate |
| D: v2 scrubbing | If MLP outputs at L8-10 are protected, does faithfulness rise above 0.5? |

### Predicted outcomes

**Enhancement A (wider coefficients):** The v1 pattern — only coeff=−3 shifts behavior, and only by +0.33 — suggests saturation at the representation level, not coefficient insufficiency. Coefficients of ±10 or ±20 may produce larger perturbations but are unlikely to unlock compliance in a model where the refusal direction lives in the residual stream via MLP writes rather than a patchable attention output. Prediction: still no compliance at any coefficient; possibly stronger refusal at large negative values.

**Enhancement B (per-head):** The whole-layer `hook_z` at L9-11 has maximum recovery 0.13 in v1, aggregated across 16 heads. Per-head decomposition will reveal whether this is one strong head (e.g. 0.10 for head H, near-zero for the rest) or noise spread across all heads. Given the low aggregate, prediction: no single head exceeds 0.15 recovery; the attention pathway is genuinely weak, not just diluted.

**Enhancement C (MLP patching):** This is the highest-information test. If the residual stream at L10-11 carries the signal but attention doesn't write it, the MLP at L8-10 must. Prediction: `blocks.9.hook_mlp_out` and `blocks.10.hook_mlp_out` show recovery fractions >0.3, potentially matching the resid_post values from v1 Stage 3.

**Enhancement D (v2 scrubbing):** If MLP writes are the actual mechanism, protecting `blocks.{8,9,10}.hook_mlp_out` should yield faithfulness substantially above 0.04. Whether it reaches the 0.5 threshold for "PARTIAL" or 0.7 for "SUPPORTED" depends on how cleanly the refusal behavior is localized to those three layers.

---

## Interpretation pending run results

### If Enhancement D faithfulness > 0.5 (partial or supported)

**Strong positive finding:** the abliteration recipe CAN work on Qwen2.5-1.5B, but the correct intervention target is MLP outputs, not attention head weights. The Arditi/RepE abliteration recipe as written projects out attention weight contributions — this audit would show those writes are wrong target. A corrected recipe would project out MLP weight contributions (W_out rows) that write in the refusal direction. This is a meaningful mechanistic refinement: it means abliterating Qwen requires modifying the down-projection matrices of MLPs 8-10, not the OV matrices of attention heads 9-11.

### If Enhancement D faithfulness < 0.5 (rejected)

**Strong negative finding:** the abliteration recipe is definitively broken on Qwen across all tested intervention strategies — wider coefficients, per-head attention decomposition, and MLP-output patching all fail to identify a sufficient circuit. This would suggest the refusal mechanism is either (a) implemented by the accumulated residual stream from early layers (pre-layer 8) rather than any single identifiable component, or (b) distributed across so many components that no subset of size ≤3 covers it at faithfulness >0.5.

Either outcome is meaningful. The literature largely assumes the recipe works. Documented mechanistic failure on a production instruct checkpoint is itself a contribution.

---

## Relationship to v1 runs

Original runs 70/71/72/73 are **unaffected** — no changes were made to their YAML files, experiment code, or stored artifacts. The `per_head: false` default in `CircuitPatchingSpec` ensures full backward compatibility. The new runs will receive new run IDs assigned by `mech run`.

To compile a v2 audit report after all four new runs complete:

```bash
mech audit-refusal \
  --refusal-run <wide_run_id> \
  --caa-run 71 \
  --circuit-run <per_head_run_id> \
  --scrub-run <v2_scrub_run_id> \
  --output docs/investigations/refusal_audit_v2_compiled
```

(Reuse caa_run=71 since the CAA layer sweep is unchanged.)

---

## What this means for the abliteration literature

The abliteration recipe (Arditi et al., 2024) and its HuggingFace community derivatives assume a specific mechanistic structure: the refusal direction, once found, is *written* by attention head outputs near the most-separable residual stream layer, so projecting out those head weight contributions suffices to remove refusal. The v1 Qwen audit showed that assumption fails — the direction is present at L10-11 resid_post but not in the attention outputs at those layers. The v2 audit tests the natural correction: maybe MLPs, not attention, are the writer.

If v2 confirms MLP writes: the recipe works in principle but requires a different intervention (MLP W_out projection rather than attention OV projection). This would mean every published abliteration writeup that reports success on attention-head ablation on Qwen is either: (a) wrong about which components they ablated, (b) working on a different model where the structure is genuinely attention-mediated, or (c) measuring a proxy metric that doesn't survive causal scrubbing. The correct methodology is the 4-stage pipeline this audit implements.

If v2 still fails: refusal in Qwen2.5-1.5B-Instruct may be genuinely distributed — implemented by many small contributions across early layers that accumulate into the residual stream rather than by any identifiable circuit of 3-5 components. This is a safety-positive finding: distributed refusal mechanisms are harder to surgically remove than concentrated ones, which means this particular checkpoint may be more robust to abliteration attacks than the literature's confidence in the recipe implies.
