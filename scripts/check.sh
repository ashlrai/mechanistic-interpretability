#!/usr/bin/env bash
set -euo pipefail

# Local CI gate. Fast tests + lint + type checks run by default; the gpt2-small
# integration suite is opt-in because each test loads real model weights and
# takes ~30s.

uv run --group dev python -m pytest --ignore=tests/integration -q
uv run --group dev ruff check src tests
uv run --group dev mypy src tests
uv run --group dev mech validate

if [[ "${RUN_INTEGRATION_TESTS:-0}" == "1" ]]; then
  uv run --group dev --extra interp python -m pytest tests/integration -v
fi
