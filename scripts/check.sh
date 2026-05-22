#!/usr/bin/env bash
set -euo pipefail

uv run --group dev python -m pytest
uv run --group dev ruff check .
uv run --group dev mypy src tests
uv run --group dev mech validate
