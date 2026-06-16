#!/usr/bin/env python3
"""Patch upstream FunASR serve_vllm.py for YouDub.

Patches:
1. Convert stereo to mono before resampling.
2. Allow POST /asr to read a same-machine ``audio_path`` instead of requiring
   multipart upload.
"""

from __future__ import annotations

import sys
from pathlib import Path


OLD = """    if sr != 16000:
        import librosa
        audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=16000)
        sr = 16000
    if audio_data.ndim > 1:
        audio_data = audio_data[:, 0]
    audio_data = audio_data.astype(np.float32)
"""

OLD_ASR_SIGNATURE = """async def asr_endpoint(
    file: UploadFile = File(...),
    language: str = Form(default=None),
    hotwords: str = Form(default=""),
    spk: bool = Form(default=False),
    timestamp: bool = Form(default=True),
):
"""

NEW_ASR_SIGNATURE = """async def asr_endpoint(
    file: UploadFile | None = File(default=None),
    audio_path: str = Form(default=""),
    language: str = Form(default=None),
    hotwords: str = Form(default=""),
    spk: bool = Form(default=False),
    timestamp: bool = Form(default=True),
):
"""

OLD_ASR_READ = """    content = await file.read()
    audio_data, sr = sf.read(io.BytesIO(content))
"""

NEW_ASR_READ = """    if audio_path:
        allowed_roots = [
            os.path.abspath(root)
            for root in os.getenv("FUNASR_ALLOWED_AUDIO_ROOTS", "/mnt").split(":")
            if root.strip()
        ]
        requested_path = os.path.abspath(audio_path)
        if allowed_roots and not any(
            requested_path == root or requested_path.startswith(root + os.sep)
            for root in allowed_roots
        ):
            return JSONResponse(status_code=403, content={"error": f"audio_path is outside allowed roots: {audio_path}"})
        audio_data, sr = sf.read(requested_path)
    else:
        if file is None:
            return JSONResponse(status_code=422, content={"error": "file or audio_path is required"})
        content = await file.read()
        audio_data, sr = sf.read(io.BytesIO(content))
"""

NEW = """    if audio_data.ndim > 1:
        audio_data = audio_data.mean(axis=1)
    if sr != 16000:
        import librosa
        audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=16000)
        sr = 16000
    audio_data = audio_data.astype(np.float32)
"""


def _patch_stereo_block(text: str, path: Path) -> tuple[str, bool]:
    if NEW in text:
        return text, False
    if OLD in text:
        return text.replace(OLD, NEW), True

    lines = text.splitlines(keepends=True)
    def_idx = next((i for i, line in enumerate(lines) if line.startswith("def process_audio(")), None)
    if def_idx is None:
        print(f"Warning: could not find process_audio in {path}", file=sys.stderr)
        return text, False

    vad_idx = next(
        (i for i in range(def_idx + 1, len(lines)) if lines[i].lstrip().startswith("# VAD segmentation")),
        None,
    )
    if vad_idx is None:
        print(f"Warning: could not find VAD marker in process_audio in {path}", file=sys.stderr)
        return text, False

    if any("audio_data.mean(axis=1)" in line for line in lines[def_idx:vad_idx]):
        return text, False

    start_idx = next(
        (
            i
            for i in range(def_idx + 1, vad_idx)
            if lines[i].startswith("    if sr != 16000:") or lines[i].startswith("    if audio_data.ndim > 1:")
        ),
        None,
    )
    if start_idx is None:
        print(
            f"Warning: could not find stereo-resample block in {path}; "
            "continuing with /asr path-mode patch.",
            file=sys.stderr,
        )
        return text, False

    lines[start_idx:vad_idx] = [line + "\n" for line in NEW.rstrip("\n").splitlines()]
    return "".join(lines), True


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: patch_serve_vllm.py /path/to/serve_vllm.py", file=sys.stderr)
        return 2

    path = Path(sys.argv[1])
    text = path.read_text(encoding="utf-8")
    changed = False
    text, stereo_changed = _patch_stereo_block(text, path)
    changed = changed or stereo_changed

    if NEW_ASR_SIGNATURE not in text:
        if OLD_ASR_SIGNATURE not in text:
            print(f"Could not find expected /asr signature block in {path}", file=sys.stderr)
            return 1
        text = text.replace(OLD_ASR_SIGNATURE, NEW_ASR_SIGNATURE)
        changed = True

    if NEW_ASR_READ not in text:
        if OLD_ASR_READ not in text:
            print(f"Could not find expected /asr read block in {path}", file=sys.stderr)
            return 1
        text = text.replace(OLD_ASR_READ, NEW_ASR_READ)
        changed = True

    if not changed:
        print(f"serve_vllm.py already patched: {path}")
        return 0

    path.write_text(text, encoding="utf-8")
    print(f"Patched serve_vllm.py for YouDub: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
