#!/usr/bin/env bash
set -euo pipefail

uv run --group dev mech experiments
uv run --group dev mech run
uv run --group dev mech runs
