#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR/frontend"

if [[ ! -d node_modules ]]; then
  npm install --cache "$ROOT_DIR/.npm-cache"
fi

PORT="${PORT:-5173}"
npm run dev -- --port "$PORT"
