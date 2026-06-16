#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/config.env" ]; then
  # shellcheck disable=SC1091
  source "$SCRIPT_DIR/config.env"
fi

FUNASR_SERVICE_PORT="${FUNASR_SERVICE_PORT:-8899}"

PIDS=""
if command -v fuser >/dev/null 2>&1; then
  PIDS="$(fuser "${FUNASR_SERVICE_PORT}/tcp" 2>/dev/null || true)"
elif command -v lsof >/dev/null 2>&1; then
  PIDS="$(lsof -ti "tcp:${FUNASR_SERVICE_PORT}" 2>/dev/null || true)"
fi

if [ -n "$PIDS" ]; then
  # shellcheck disable=SC2086
  kill $PIDS 2>/dev/null || true
  sleep 2
  # shellcheck disable=SC2086
  kill -9 $PIDS 2>/dev/null || true
fi

pkill -f "serve_vllm.py.*--port ${FUNASR_SERVICE_PORT}" 2>/dev/null || true

echo "Stopped FunASR vLLM service on port ${FUNASR_SERVICE_PORT} if it was running."
