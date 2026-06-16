#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/config.env" ]; then
  # shellcheck disable=SC1091
  source "$SCRIPT_DIR/config.env"
fi

VENV_DIR="${VENV_DIR:-$HOME/ai/funasr-service/.venv}"
export VLLM_USE_PRECOMPILED="${VLLM_USE_PRECOMPILED:-1}"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/env.sh"

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

python - <<'PY'
from pathlib import Path
import importlib.metadata
import importlib.util
import subprocess
import sys

def version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"

print("python=", sys.version)
os = __import__("os")
print("VLLM_USE_PRECOMPILED=", os.environ.get("VLLM_USE_PRECOMPILED"))
print("CUDA_HOME=", os.environ.get("CUDA_HOME"))
print("CC=", os.environ.get("CC"))
print("CXX=", os.environ.get("CXX"))
print("CUDAHOSTCXX=", os.environ.get("CUDAHOSTCXX"))
for name in (
    "torch",
    "vllm",
    "nvidia-cuda-runtime-cu12",
    "nvidia-cuda-runtime-cu13",
    "nvidia-cublas-cu12",
    "nvidia-cublas-cu13",
):
    print(f"{name}=", version(name))

try:
    import torch
    print("torch_cuda=", torch.version.cuda, "cuda_available=", torch.cuda.is_available())
except Exception as exc:
    print("torch_error=", repr(exc))

print("cuda_runtimes:")
for base in sys.path:
    path = Path(base)
    if path.exists():
        for lib in path.glob("**/libcudart.so*"):
            print(" ", lib)

spec = importlib.util.find_spec("vllm")
print("vllm_spec=", spec.origin if spec else "missing")
c_candidates = [
    Path(base) / "vllm" / "_C.abi3.so"
    for base in sys.path
    if (Path(base) / "vllm" / "_C.abi3.so").exists()
]
for candidate in c_candidates:
    print("ldd", candidate)
    subprocess.run(["ldd", str(candidate)], check=False)

try:
    import vllm
    import vllm._C
    print("vllm_import=ok", vllm.__version__, vllm._C.__file__)
except Exception as exc:
    print("vllm_import_error=", repr(exc))
PY
