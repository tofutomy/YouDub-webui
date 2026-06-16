# FunASR vLLM Service

This folder contains WSL/Linux helper scripts for running FunASR + vLLM as a
separate ASR service for YouDub.

## Layout

Recommended runtime layout inside WSL:

```text
~/ai/funasr-service/.venv
~/ai/upstream/FunASR
```

The YouDub repository only stores the scripts. The large virtual environment,
models, and upstream FunASR checkout stay outside the YouDub worktree.

## Install Or Update

```bash
cd /mnt/f/AI/YouDub-webui/services/funasr-vllm
cp config.env.example config.env
./update.sh
```

`update.sh` creates or updates the virtual environment, installs CUDA PyTorch,
installs vLLM, clones or updates `modelscope/FunASR`, and installs FunASR in
editable mode.

The script uses system Python only to bootstrap `uv`. The actual FunASR/vLLM
runtime is created with `PYTHON_VERSION=3.12` by default, because vLLM supports
Python 3.10-3.13 and newer Python versions may select incompatible wheels.
You can change `PYTHON_VERSION` in `config.env` if needed.

The default vLLM wheel variant is `VLLM_CUDA_VERSION=129`, matching vLLM's
current CUDA 12.9 binary line. The script installs PyTorch from the matching
`cu129` index and installs vLLM with `uv pip --torch-backend=cu129`.

On Ubuntu/Debian, install the venv support package first if `python -m venv`
is unavailable:

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip git curl
```

If you previously created the service venv with an unsupported Python version
such as 3.14, remove or rename it before running `./update.sh` again:

```bash
mv ~/ai/funasr-service/.venv ~/ai/funasr-service/.venv.py314
./update.sh
```

An error like `ImportError: libcudart.so.13: cannot open shared object file`
means the environment installed a CUDA 13 vLLM wheel while the machine runtime
is CUDA 12.x. Recreate the service venv with `PYTHON_VERSION=3.12` and
`VLLM_CUDA_VERSION=129`.

## Start

```bash
cd /mnt/f/AI/YouDub-webui/services/funasr-vllm
./start.sh
```

The default service listens on port `8899` and exposes the FunASR vLLM `/asr`
endpoint.

`start.sh` patches the upstream `/asr` endpoint for YouDub:

- it accepts `audio_path` so same-machine WSL can read audio directly from
  `/mnt/<drive>/...` instead of receiving multipart uploads;
- it converts stereo audio to mono before resampling, avoiding incorrect
  duration expansion for stereo WAV files.

Restrict readable audio paths with:

```env
FUNASR_ALLOWED_AUDIO_ROOTS=/mnt/g/FFOutput:/mnt/f/AI/YouDub-webui/workfolder
```

vLLM reserves GPU memory up front. On a 24 GB desktop GPU, start with:

```env
FUNASR_GPU_MEMORY_UTILIZATION=0.45
```

Raise it only if requests fail with KV cache or out-of-memory errors. Lower it
if YouDub needs to run Demucs, VoxCPM, or other CUDA workloads at the same time.

## Smoke Test

```bash
cd /mnt/f/AI/YouDub-webui/services/funasr-vllm
./smoke_test.sh
```

Or pass a local audio file:

```bash
./smoke_test.sh /path/to/audio.wav
```

## Diagnose vLLM/CUDA

```bash
cd /mnt/f/AI/YouDub-webui/services/funasr-vllm
./diagnose.sh
```

This prints the installed Torch/vLLM versions, CUDA runtime packages, detected
`libcudart.so*` files, and `ldd` output for `vllm._C`.

If startup fails with `Could not find nvcc`, first keep
`VLLM_USE_PRECOMPILED=1` in `config.env` and retry `./start.sh`. If vLLM still
tries to compile CUDA code, install a CUDA toolkit inside WSL so `nvcc` exists.
For the default `torch==*+cu129` environment, prefer CUDA Toolkit 12.9 and set:

```env
CUDA_HOME=/usr/local/cuda-12.9
```

Do not install Linux NVIDIA display drivers inside WSL. NVIDIA's WSL guide
recommends installing the toolkit package only, such as `cuda-toolkit-12-x`,
and avoiding `cuda`, `cuda-12-x`, or `cuda-drivers` meta-packages.

CUDA 12.9 supports GCC up to 14. If nvcc fails with `unsupported GNU version`,
install GCC/G++ 14 or 13 and let the scripts select it automatically:

```bash
sudo apt update
sudo apt install -y gcc-14 g++-14
```

If your distro does not provide 14:

```bash
sudo apt install -y gcc-13 g++-13
```

You can also pin the compiler in `config.env`:

```env
CC=/usr/bin/gcc-14
CXX=/usr/bin/g++-14
CUDAHOSTCXX=/usr/bin/g++-14
```

## YouDub Configuration

Set these values in the YouDub `.env` file:

```env
FUNASR_REMOTE_BASE_URL=http://127.0.0.1:8899
FUNASR_REMOTE_ENDPOINT=/asr
FUNASR_REMOTE_TIMEOUT=900
FUNASR_REMOTE_SPK=false
FUNASR_REMOTE_AUTOSTART=true
FUNASR_REMOTE_AUTOSTOP=true
FUNASR_REMOTE_START_TIMEOUT=300
```

Then choose a `funasr:*` ASR model in YouDub. When
`FUNASR_REMOTE_BASE_URL` is set, YouDub sends FunASR ASR requests to this
service instead of importing local `funasr` or `vllm`.

With `FUNASR_REMOTE_AUTOSTART=true`, YouDub checks `FUNASR_REMOTE_BASE_URL`
before a remote Nano run. If nothing is listening, it runs `bash ./start.sh`
through WSL and waits until the HTTP service responds. With
`FUNASR_REMOTE_AUTOSTOP=true`, YouDub stops only the service it started for
that ASR run. A manually started service is left running. Auto-start output is
written to `services/funasr-vllm/autostart.log`.

To stop the service manually:

```bash
cd /mnt/f/AI/YouDub-webui/services/funasr-vllm
bash ./stop.sh
```

## Canary Updates

For safer rolling updates, duplicate the runtime folder and test first:

```text
~/ai/funasr-service-canary/.venv
~/ai/upstream/FunASR
```

Point `VENV_DIR` in `config.env` at the canary venv, run `./update.sh`, then
run `./smoke_test.sh`. Update the stable venv only after the canary works.
