#!/usr/bin/env bash
set -euo pipefail

# Serve the docs site locally with live reload.
# Requires the dev dependency group: uv sync --group dev
#
# Usage:
#   bash scripts/serve_docs.sh
#   bash scripts/serve_docs.sh --dev-addr 0.0.0.0:8001
#
# Open http://127.0.0.1:8000 in your browser once the server starts.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

exec uv run --group dev mkdocs serve --dev-addr 127.0.0.1:8000 "$@"
