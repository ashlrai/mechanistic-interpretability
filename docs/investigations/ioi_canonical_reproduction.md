# Investigation #8 — Canonical IOI Circuit Reproduction (Wang et al., 2022)

**Date:** 2026-05-27
**Model:** `gpt2-small` (12 layers, 12 heads/layer, d_model=768)
**Run IDs:** 75 (`acdc_edge`, edge-level) · 76 (`acdc_lite`, node-level)
**Corpus:** 30 templated IOI prompts — `experiments/ioi_prompts.jsonl`
**Spec:** `experiments/acdc_edge_ioi.yaml`, `experiments/acdc_lite_ioi.yaml`
**Wall-clock:** ~35 min (acdc_edge, 30 prompts × layers 5-11, max_edges=500) · ~12 min (acdc_lite)
**Status:** **results pending run completion — numbers filled in below**

---

## Objective

Wang et al. (2022) identified a six-component circuit in GPT-2 small for the Indirect Object
Identification (IOI) task. The circuit comprises 12 attention heads across 5 functional groups.
This investigation asks: **can the platform's `acdc_edge` and `acdc_lite` approximations recover
these heads from scratch on a 30-prompt corpus?**

The answer characterises the platform's measurement quality before using it to study less
well-understood circuits.

---

## Corpus

30 IOI prompt pairs generated from the 12 Wang et al. names
(`John, Mary, Tom, Alice, James, Sarah, David, Emma, Michael, Olivia, William, Sophie`)
and 10 locations (`store, park, office, library, restaurant, school, garden, stadium, hospital, museum`).

Template (ABB pattern throughout):
```
clean:     "Then {A} and {B} went to the {place}. {A} gave a {object} to"
corrupted: "Then {A} and {B} went to the {place}. {B} gave a {object} to"
correct_token:   " {B}"
incorrect_token: " {A}"
```

The corrupted prompt swaps the subject, so the model must track which name is the _indirect object_.
Mean-ablation is used throughout (ablation_type=mean, seed=42).

---

## Canonical Heads (Wang et al., 2022)

| Group | Heads | Count |
|---|---|---|
| name_mover | L9.H6, L9.H9, L10.H0 | 3 |
| s_inhibition | L7.H3, L8.H6 | 2 |
| backup_name_mover | L10.H7, L11.H10 | 2 |
| induction | L5.H5, L5.H8, L5.H9 | 3 |
| duplicate_token | L0.H1, L3.H0 | 2 |
| **Total** | | **12** |

Note: `duplicate_token` heads (L0.H1, L3.H0) are outside the `layers=[5..11]` search window.
Their recall is expected to be 0/2 for both runs regardless of algorithm quality.
The effective recall ceiling within the search window is **10/12**.

---

## Method

### acdc_edge (run 75)

Edge-level ACDC (Conmy et al., 2023) with the platform's two-pass KL approximation:

```
edge_importance(src, dst) ≈ KL(p_full ‖ p_src_ablated) / layer_gap(src, dst)
```

All forward passes are grouped by src node (one ablated pass per unique src), giving
`1 + |unique_srcs|` total passes per prompt instead of `2 × |edges|`.
Config: `layers=[5..11]`, `max_edges=500`, `tau=0.01`, `max_iterations=10`.

Edges surviving tau-pruning constitute the discovered circuit; their src/dst nodes are
matched against the canonical head list.

### acdc_lite (run 76)

Node-level ACDC: ablates each head entirely and measures `|full_logit_diff − ablated_logit_diff|`.
Same layers, tau, and prompt corpus. Much cheaper (one pass per node vs. one per unique src).

---

## Results

### acdc_edge (run 75) — per-group recall

| Group | Canonical | Hit | Heads found |
|---|---|---|---|
| name_mover | 3 | **TBD** | TBD |
| s_inhibition | 2 | **TBD** | TBD |
| backup_name_mover | 2 | **TBD** | TBD |
| induction | 3 | **TBD** | TBD |
| duplicate_token | 2 | 0† | — (outside search window) |
| **Overall** | **12** | **TBD/12** | |

† duplicate_token heads L0.H1 and L3.H0 are in layers 0 and 3 respectively, outside the
`layers=[5..11]` window used in this run.

**Faithfulness:** TBD
**Overall recall:** TBD / 12 (TBD within-window: TBD / 10)
**Precision:** TBD (canonical hits / total surviving attn nodes)

Top-10 surviving edges:

| Rank | src → dst | Importance |
|---|---|---|
| TBD | | |

### acdc_lite (run 76) — per-group recall

| Group | Canonical | Hit | Heads found |
|---|---|---|---|
| name_mover | 3 | **TBD** | TBD |
| s_inhibition | 2 | **TBD** | TBD |
| backup_name_mover | 2 | **TBD** | TBD |
| induction | 3 | **TBD** | TBD |
| duplicate_token | 2 | 0† | — |
| **Overall** | **12** | **TBD/12** | |

**Faithfulness:** TBD
**Overall recall:** TBD / 12
**Precision:** TBD

---

## Comparison: acdc_edge vs acdc_lite

| Metric | acdc_edge (edge-level) | acdc_lite (node-level) |
|---|---|---|
| Recall | TBD/12 | TBD/12 |
| Precision | TBD | TBD |
| Faithfulness | TBD | TBD |
| Wall-clock | ~35 min | ~12 min |

---

## Verdict

TBD once runs complete. Will be one of:

- **Recall ≥ 7/10 (within-window):** platform validation.
- **Recall 3-6/10:** partial validation, consistent with KL-by-layer-gap late-layer bias.
- **Recall ≤ 2/10:** real problem; the approximation needs replacement.

---

## Known Approximation Limitations

1. **Layer-gap weight** (`/ layer_gap`) under-credits long-range edges. Induction heads at layer 5
   feed into name-movers at layers 9-11 across a gap of 4-6 — their KL contribution gets divided
   by 4-6 before thresholding. This is the primary structural reason to expect lower induction recall.

2. **Mean ablation** (not zero) introduces a distributional baseline artefact. Wang et al. used
   full path patching with a corrupted baseline; our approximation replaces src with its
   empirical mean activation, which may not null out all IOI-relevant information.

3. **No iterative recomputation.** The scoring pass runs once before pruning; true ACDC would
   rescore surviving edges after each pruning round. Our multi-iteration loop prunes with the
   original scores, so synergistic heads (those that only matter when other heads are present)
   may be underscored.

4. **duplicate_token heads excluded** by the `layers=[5..11]` window. These are layer-0 and
   layer-3 heads; including them would require a wider window and ~2× more candidates.

---

## Grading script

```bash
cd "path/to/repo"
uv run --extra interp python scripts/grade_ioi.py --run-id 75      # acdc_edge
uv run --extra interp python scripts/grade_ioi.py --lite-run-id 76  # acdc_lite
```

---

## References

- Wang, K., et al. (2022). "Interpretability in the Wild: a Circuit for Indirect Object
  Identification in GPT-2 small." NeurIPS 2022. https://arxiv.org/abs/2211.00593
- Conmy, A., et al. (2023). "Towards Automated Circuit Discovery for Mechanistic Interpretability."
  NeurIPS 2023. https://arxiv.org/abs/2304.14997
