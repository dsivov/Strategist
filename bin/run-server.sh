#!/usr/bin/env bash
# Launch the POC benchmark server.
# Open http://localhost:8443/ to use the dual-panel UI.
set -euo pipefail
cd "$(dirname "$0")/.."

# Load .env if present
if [ -f .env ]; then
  set -a; . ./.env; set +a
fi

PORT="${PORT:-8443}"
HOST="${HOST:-127.0.0.1}"

if [ -z "${ANTHROPIC_API_KEY:-}" ] || [ -z "${GEMINI_API_KEY_1:-}" ]; then
  echo "ERROR: ANTHROPIC_API_KEY and GEMINI_API_KEY_1 must be set in .env" >&2
  echo "       (cp .env.example .env, then fill in the keys)" >&2
  exit 1
fi

echo "POC server starting on http://${HOST}:${PORT}/"
echo "(db.py: JSON-backed mode — scenarios from data/benchmark/v1_scenarios.json)"
exec python3 -m uvicorn server.main:app --host "$HOST" --port "$PORT"
