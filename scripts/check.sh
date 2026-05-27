#!/usr/bin/env bash
# Local CI gate.
#
# MODES
#   --fast (default)  Lint + type checks + unit tests only.
#                     Tests that need torch/transformer-lens are skipped via
#                     pytest.importorskip — no 2 GB dep install required.
#                     Typical run: ~15 s.
#
#   --full            Same as --fast PLUS the full interp suite (loads real
#                     model weights via --extra interp) and the integration
#                     tests when RUN_INTEGRATION_TESTS=1.
#                     Typical run: 5-10 min depending on model cache state.
#
# The trade-off: --fast keeps the CI gate lightweight; torch/transformer-lens
# are NOT pulled into the base dev environment.  Unit tests that import torch
# at module-level are guarded by pytest.importorskip so they skip cleanly in
# --fast mode rather than erroring at collection time.

set -euo pipefail

MODE="fast"
if [[ "${1:-}" == "--full" ]]; then
  MODE="full"
fi

echo "==> check.sh mode: ${MODE}"

# ── Lint ─────────────────────────────────────────────────────────────────────
uv run --group dev ruff check src tests

# ── Type checks ──────────────────────────────────────────────────────────────
uv run --group dev mypy src tests

# ── Unit tests ───────────────────────────────────────────────────────────────
if [[ "${MODE}" == "full" ]]; then
  uv run --group dev --extra interp python -m pytest --ignore=tests/integration -q
else
  uv run --group dev python -m pytest --ignore=tests/integration -q
fi

# ── Validate experiment manifests ────────────────────────────────────────────
uv run --group dev mech validate

# ── Docs build (strict) ──────────────────────────────────────────────────────
uv run --group dev mkdocs build --strict --quiet

# ── Integration tests (opt-in) ───────────────────────────────────────────────
if [[ "${MODE}" == "full" ]] || [[ "${RUN_INTEGRATION_TESTS:-0}" == "1" ]]; then
  uv run --group dev --extra interp python -m pytest tests/integration -v
fi

echo "==> All checks passed."
