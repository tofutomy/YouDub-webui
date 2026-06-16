#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/config.env" ]; then
  # shellcheck disable=SC1091
  source "$SCRIPT_DIR/config.env"
fi

FUNASR_SERVICE_PORT="${FUNASR_SERVICE_PORT:-8899}"
SERVER_URL="${SERVER_URL:-http://127.0.0.1:$FUNASR_SERVICE_PORT/asr}"
SAMPLE_AUDIO="${1:-${SAMPLE_AUDIO:-}}"

if [ -z "$SAMPLE_AUDIO" ]; then
  SAMPLE_AUDIO="$SCRIPT_DIR/sample.wav"
  curl -L "https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/BAC009S0764W0121.wav" -o "$SAMPLE_AUDIO"
fi

curl -sS -X POST "$SERVER_URL" \
  -F "file=@$SAMPLE_AUDIO" \
  -F "language=中文" \
  -F "timestamp=true" \
  -F "spk=false"
echo
