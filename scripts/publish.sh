#!/usr/bin/env bash
# Build and upload the package to PyPI. Set PYPI_TOKEN in a gitignored .env at repo root,
# or export PYPI_TOKEN in your shell (PyPI API token, including the pypi- prefix).

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -f "${ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${ROOT}/.env"
  set +a
fi

if [[ -z "${PYPI_TOKEN:-}" ]]; then
  echo "error: PYPI_TOKEN is not set. Add it to .env in the repo root or export it." >&2
  exit 1
fi

# Avoid uploading stale wheels/sdists from older package names or versions.
rm -rf "${ROOT}/dist" "${ROOT}/build"

uv build
UV_PUBLISH_TOKEN="${PYPI_TOKEN}" uv publish
