# Wave 6 Integration Report

**Date:** 2026-05-27  
**Gate result:** 644 passed, 7 skipped · ruff clean · mypy clean (147 files) · 61 specs validated

---

## Agent landing order and status

| Agent | Commit | Status | Fixes needed |
|---|---|---|---|
| IOI reproduction | `663ab3a` | Clean | None — YAML corpus + experiment specs only |
| HF adapter | `7f6eed7` | Needed fixes | ruff errors in `hf_adapter.py` + test file; `backends/__init__.py` missing export |
| GitHub Pages (mkdocs) | `a51cde4` | Clean | None |
| Steering library | `76886de` | Needed minor fix | `load_steering_vector` imported locally inside function, breaking `unittest.mock.patch` |

---

## What broke and what was fixed

### `backends/hf_adapter.py` (HF adapter agent)
- Unused `numpy` import, `TYPE_CHECKING` block with unused `torch`/`transformers` stubs
- `_encode` and `_logits_at_position` assigned `torch = _require_torch()` but never used the variable
- Loop variable `io` unused in `run_with_cache`
- Unused `_tl_logits_at_position` import inside `run_activation_patching`
- **Fixed:** stripped all dead imports/assignments; renamed `io` → `_io`

### `tests/test_hf_site_translator.py` (HF adapter agent)
- Import block ordering (ruff I001)
- **Fixed:** `ruff --fix`

### `tests/integration/test_hf_backend_e2e.py` (HF adapter agent)
- Long line in `_SKIP_REASON`
- Stale `# type: ignore[name-defined]` on four test functions (unused after mypy override added)
- Malformed `# mypy: needed because...` comment parsed as a mypy directive
- **Fixed:** split long line; removed stale ignores; removed malformed comment

### `src/mech_interp/backends/__init__.py` (HF adapter agent)
- `HuggingFaceBackend` not exported — `test_hf_adapter.py` imports it from the package
- **Fixed:** added `from mech_interp.backends.hf_adapter import HuggingFaceBackend`

### `src/mech_interp/cli.py` (steering agent)
- `load_steering_vector` imported locally inside `apply_steering_command`; `unittest.mock.patch("mech_interp.cli.load_steering_vector", ...)` failed because the name didn't exist at module level
- **Fixed:** hoisted import to module level; removed redundant local import

### `tests/test_cli_apply_steering.py` (steering agent)
- Unused `pytest` import; one long line; `fake_load` noqa placement
- **Fixed:** `ruff --fix` + manual wrap

### `pyproject.toml` (pre-existing + Wave 6 gap)
- `[[tool.mypy.overrides]]` only covered `mlx_lm`, `nnsight`, `transformer_lens` — Wave 6 added `safetensors`, `huggingface_hub`, `datasets`, `pyarrow`, `torch`, `transformers` as new deps
- `ignore_errors` override used `tests.test_hf_adapter` (wrong — no `tests/__init__.py`)
- **Fixed:** extended override list; corrected module names to `test_hf_adapter`, `test_hf_backend_e2e`

### `src/mech_interp/datasets/downloader.py`
- Stale `# type: ignore[import-untyped]` on `datasets` import (now covered by override)
- **Fixed:** removed comment

---

## Final platform inventory

### Commands (37 total)

**Setup:** `init-store`, `providers`, `config`, `download-corpus`

**Run experiments:** `validate`, `experiments`, `run`, `runs`, `query-runs`, `inspect-run`,
`export-run`, `sweep`, `sweep-report`, `preflight`, `estimate-activations`

**Closed-loop:** `propose-followups`, `propose-from-run`, `iterate-from-run`, `iterate`,
`archive-runs`, `summarize-runs`, `report-runs`

**Pretrained artifacts:** `list-saes`, `download-sae`, `analyze-sae`, `analyze-sae-stability`,
`analyze-feature-splits`, `list-steering`, `apply-steering`, `list-hf-architectures`

**Audit:** `audit-refusal`, `label-features`, `sae-scale-report`, `calibrate-tuned-lens`,
`compare-runs`

**UI / demo:** `demo`, `cockpit`, `gradio`

Discovery: `mech help`

### Experiment families (15)
`circuit_patching`, `polysemanticity_sae`, `acdc_lite`, `acdc_edge`, `attribution_patching`,
`causal_scrubbing`, `caa_steering`, `refusal_direction`, `direct_logit_attribution`, `logit_lens`,
`sparse_probing`, `cross_model_representation_probe`, `crosscoder`, `sae_cross_model`,
`activation_capture`

### Backends
- TransformerLens (primary)
- HuggingFace universal adapter (`backend: huggingface` — any `AutoModelForCausalLM`)
- NNsight, MLX (optional)

### Investigations (6)
`refusal_audit`, `refusal_audit_v2`, `gpt2_factual_recall`, `feature_splitting`,
`sae_replication_crisis`, `sae_at_scale`

### Publications (2)
`sae_replication_crisis`, `abliteration_robustness`

---

## What's left for v1

1. **IOI reproduction result** — `663ab3a` added the 30-prompt corpus and extended YAML specs, but
   `docs/investigations/ioi_canonical_reproduction.md` and `scripts/grade_ioi.py` are untracked.
   Need to commit those and report the actual head-recovery fraction.

2. **SAE-lens compat** — no `mech list-sae-lens-releases` or `mech load-sae-lens` commands in
   HEAD. The SAE-lens agent either didn't commit or landed in a different worktree. Verify and
   merge if present.

3. **check.sh uses `--group dev` without `--extra interp`** for unit tests — 3 test files
   (`test_caa_steering`, `test_crosscoder`, `test_refusal_direction`) import torch directly and
   fail collection without `--extra interp`. Either add `--extra interp` to check.sh's pytest
   invocation or guard those imports under `pytest.importorskip`.

4. **Gradio app** has an unverified launch path — needs a smoke test that it starts without
   crashing before tagging v1.

5. **Docs site** (`mkdocs-material`) needs a `mkdocs build` verification pass and a link audit
   since several investigation pages were added after the site structure was committed.
