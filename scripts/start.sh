#!/usr/bin/env bash
# EEGNet Architecture Playground - one-command local start (macOS / Linux / Git Bash).
# Usage: scripts/start.sh [auto|cuda|cpu] [--rebuild]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TORCH="${1:-auto}"
if [ "$TORCH" = "auto" ]; then
  if command -v nvidia-smi >/dev/null 2>&1; then TORCH=cuda; else TORCH=cpu; fi
fi
echo "PyTorch build: $TORCH"
(cd "$ROOT/backend" && uv sync --extra "$TORCH")
if [ "${2:-}" = "--rebuild" ] || [ ! -f "$ROOT/frontend/dist/index.html" ]; then
  (cd "$ROOT/frontend" && npm ci && npm run build)
fi
echo "Open http://127.0.0.1:8000 (Ctrl+C to stop)"
cd "$ROOT/backend" && exec uv run --extra "$TORCH" python -m app.main
