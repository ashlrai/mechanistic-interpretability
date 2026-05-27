#!/usr/bin/env bash
# Create the v0.1.0-preview GitHub Release (draft + prerelease).
#
# Prerequisites:
#   gh auth login   (if not already authenticated)
#   git tag v0.1.0-preview && git push origin v0.1.0-preview
#
# Usage:
#   bash scripts/create_release.sh
#
# The release is created as --draft so you can review before publishing.
# Remove --draft to publish immediately.

set -euo pipefail

NOTES_FILE="docs/release-notes/v0.1.0-preview.md"
TITLE="v0.1.0-preview — Local mech-interp at MBP scale"
TAG="v0.1.0-preview"

# Verify the notes file exists before calling gh.
if [[ ! -f "${NOTES_FILE}" ]]; then
  echo "ERROR: release notes not found at ${NOTES_FILE}" >&2
  exit 1
fi

echo "==> Creating GitHub Release: ${TAG}"
echo "    Title:  ${TITLE}"
echo "    Notes:  ${NOTES_FILE}"
echo "    Flags:  --draft --prerelease"
echo ""

gh release create "${TAG}" \
  --title "${TITLE}" \
  --notes-file "${NOTES_FILE}" \
  --draft \
  --prerelease

echo ""
echo "==> Draft release created. Review and publish at:"
echo "    https://github.com/$(gh repo view --json nameWithOwner -q .nameWithOwner)/releases"
