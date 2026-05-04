#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT_DIR/.uv-cache}"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required. Install it from https://docs.astral.sh/uv/ and rerun this script."
  exit 1
fi

uv sync

if command -v npm >/dev/null 2>&1; then
  cd "$ROOT_DIR/frontend"
  npm install --cache "$ROOT_DIR/.npm-cache"
else
  echo "npm was not found. Install Node.js before running the frontend."
fi
