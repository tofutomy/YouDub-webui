#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/config.env" ]; then
  # shellcheck disable=SC1091
  source "$SCRIPT_DIR/config.env"
fi

FUNASR_SOURCE_DIR="${FUNASR_SOURCE_DIR:-$HOME/ai/upstream/FunASR}"
VENV_DIR="${VENV_DIR:-$HOME/ai/funasr-service/.venv}"
PYTHON_BIN="${PYTHON_BIN:-}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
VLLM_CUDA_VERSION="${VLLM_CUDA_VERSION:-129}"
VLLM_VERSION="${VLLM_VERSION:-}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu$VLLM_CUDA_VERSION}"
export VLLM_USE_PRECOMPILED="${VLLM_USE_PRECOMPILED:-1}"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/env.sh"

if [ -z "$PYTHON_BIN" ]; then
  for candidate in python3.12 python3.11 python3.10 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
      PYTHON_BIN="$candidate"
      break
    fi
  done
fi

if [ -z "$PYTHON_BIN" ]; then
  echo "Could not find python3.12, python3.11, python3.10, or python3." >&2
  echo "Install one first, for example: sudo apt update && sudo apt install -y python3 python3-venv python3-pip" >&2
  exit 1
fi

UV_BIN="${UV_BIN:-}"
if [ -z "$UV_BIN" ]; then
  if command -v uv >/dev/null 2>&1; then
    UV_BIN="$(command -v uv)"
  else
    UV_BOOTSTRAP_DIR="${UV_BOOTSTRAP_DIR:-$HOME/.local/share/youdub-uv-bootstrap}"
    "$PYTHON_BIN" -m venv "$UV_BOOTSTRAP_DIR"
    "$UV_BOOTSTRAP_DIR/bin/python" -m pip install -U pip uv
    UV_BIN="$UV_BOOTSTRAP_DIR/bin/uv"
  fi
fi

mkdir -p "$(dirname "$FUNASR_SOURCE_DIR")" "$(dirname "$VENV_DIR")"
"$UV_BIN" python install "$PYTHON_VERSION"
if [ -x "$VENV_DIR/bin/python" ]; then
  EXISTING_VERSION="$("$VENV_DIR/bin/python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
  if [ "$EXISTING_VERSION" != "$PYTHON_VERSION" ]; then
    echo "Existing venv uses Python $EXISTING_VERSION, but PYTHON_VERSION=$PYTHON_VERSION." >&2
    echo "Remove or rename $VENV_DIR, then run ./update.sh again." >&2
    exit 1
  fi
fi
"$UV_BIN" venv "$VENV_DIR" --python "$PYTHON_VERSION" --seed

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
"$UV_BIN" pip install -U pip
"$UV_BIN" pip install -U torch torchaudio --index-url "$PYTORCH_INDEX_URL"
"$UV_BIN" pip install -U safetensors tiktoken websockets regex fastapi uvicorn python-multipart requests soundfile pydub
VLLM_REQUIREMENT="vllm${VLLM_VERSION:+==$VLLM_VERSION}"
echo "Installing $VLLM_REQUIREMENT with torch backend cu$VLLM_CUDA_VERSION"
UV_TORCH_BACKEND="cu$VLLM_CUDA_VERSION" "$UV_BIN" pip install -U "$VLLM_REQUIREMENT" \
  --torch-backend="cu$VLLM_CUDA_VERSION" \
  --extra-index-url "$PYTORCH_INDEX_URL"

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
  echo "Notice: nvcc is not installed. VLLM_USE_PRECOMPILED=$VLLM_USE_PRECOMPILED; this is OK if vLLM uses prebuilt kernels." >&2
fi

python - <<'PY'
from pathlib import Path
import importlib.util
import subprocess
import sys
import torch

print("python=", sys.version.split()[0])
print("torch=", torch.__version__, "torch_cuda=", torch.version.cuda, "cuda_available=", torch.cuda.is_available())
spec = importlib.util.find_spec("vllm")
print("vllm_location=", spec.origin if spec else "missing")
try:
    import vllm
    import vllm._C
    print("vllm=", vllm.__version__)
    print("vllm_C=", vllm._C.__file__)
except Exception as exc:
    print("vllm_import_error=", repr(exc))
    for base in sys.path:
        path = Path(base)
        if path.exists():
            for lib in path.glob("**/libcudart.so*"):
                print("found_cuda_runtime=", lib)
    c_candidates = [
        Path(base) / "vllm" / "_C.abi3.so"
        for base in sys.path
        if (Path(base) / "vllm" / "_C.abi3.so").exists()
    ]
    for candidate in c_candidates:
        print("ldd", candidate)
        subprocess.run(["ldd", str(candidate)], check=False)
    raise
PY

if [ -d "$FUNASR_SOURCE_DIR/.git" ]; then
  git -C "$FUNASR_SOURCE_DIR" pull --ff-only
else
  git clone https://github.com/modelscope/FunASR.git "$FUNASR_SOURCE_DIR"
fi

SERVE_SCRIPT="$FUNASR_SOURCE_DIR/examples/industrial_data_pretraining/fun_asr_nano/serve_vllm.py"
python "$SCRIPT_DIR/patch_serve_vllm.py" "$SERVE_SCRIPT"
"$UV_BIN" pip install -e "$FUNASR_SOURCE_DIR"
python -c "import torch, vllm, funasr; print('cuda=', torch.cuda.is_available()); print('gpu=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'); print('vllm=', vllm.__version__)"
