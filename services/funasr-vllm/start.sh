#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/config.env" ]; then
  # shellcheck disable=SC1091
  source "$SCRIPT_DIR/config.env"
fi

FUNASR_SOURCE_DIR="${FUNASR_SOURCE_DIR:-$HOME/ai/upstream/FunASR}"
VENV_DIR="${VENV_DIR:-$HOME/ai/funasr-service/.venv}"
FUNASR_SERVICE_PORT="${FUNASR_SERVICE_PORT:-8899}"
FUNASR_SERVICE_MODEL="${FUNASR_SERVICE_MODEL:-FunAudioLLM/Fun-ASR-Nano-2512}"
FUNASR_GPU_MEMORY_UTILIZATION="${FUNASR_GPU_MEMORY_UTILIZATION:-0.8}"
FUNASR_MAX_MODEL_LEN="${FUNASR_MAX_MODEL_LEN:-4096}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export VLLM_USE_PRECOMPILED="${VLLM_USE_PRECOMPILED:-1}"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/env.sh"

SERVE_SCRIPT="$FUNASR_SOURCE_DIR/examples/industrial_data_pretraining/fun_asr_nano/serve_vllm.py"
if [ ! -f "$SERVE_SCRIPT" ]; then
  echo "Missing $SERVE_SCRIPT. Run ./update.sh first." >&2
  exit 1
fi
PATCH_PYTHON=""
if [ -x "$VENV_DIR/bin/python" ]; then
  PATCH_PYTHON="$VENV_DIR/bin/python"
elif [ -n "${PYTHON_BIN:-}" ] && command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PATCH_PYTHON="$PYTHON_BIN"
elif command -v python3 >/dev/null 2>&1; then
  PATCH_PYTHON="python3"
elif command -v python >/dev/null 2>&1; then
  PATCH_PYTHON="python"
else
  echo "Could not find python to patch serve_vllm.py." >&2
  exit 1
fi
"$PATCH_PYTHON" "$SCRIPT_DIR/patch_serve_vllm.py" "$SERVE_SCRIPT"

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
CUDA_LIB_DIRS="$(python - <<'PY'
from pathlib import Path
import site

dirs = []
for base in site.getsitepackages():
    torch_lib = Path(base) / "torch" / "lib"
    if torch_lib.is_dir():
        dirs.append(str(torch_lib))
    nvidia = Path(base) / "nvidia"
    if not nvidia.exists():
        continue
    dirs.extend(str(path) for path in nvidia.glob("*/lib") if path.is_dir())
print(":".join(dirs))
PY
)"
export LD_LIBRARY_PATH="${CUDA_LIB_DIRS}${CUDA_LIB_DIRS:+:}${LD_LIBRARY_PATH:-}"
if ! command -v nvcc >/dev/null 2>&1; then
  echo "Notice: nvcc is not installed. VLLM_USE_PRECOMPILED=$VLLM_USE_PRECOMPILED; if vLLM still tries to compile CUDA code, install a CUDA toolkit in WSL." >&2
fi
exec python "$SERVE_SCRIPT" \
  --port "$FUNASR_SERVICE_PORT" \
  --model "$FUNASR_SERVICE_MODEL" \
  --gpu-memory-utilization "$FUNASR_GPU_MEMORY_UTILIZATION" \
  --max-model-len "$FUNASR_MAX_MODEL_LEN" \
  "$@"
