from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

import torch

from ..config import REPO_ROOT
from ..devices import resolve_device


def _device() -> str:
    return resolve_device("demucs").selected


def _is_cuda(device_str: str) -> bool:
    return "cuda" in device_str.lower()


def _demucs_progress(info: dict, shifts: int) -> int:
    models = max(1, int(info.get("models") or 1))
    model_index = max(0, int(info.get("model_idx_in_bag") or 0))
    shift_index = max(0, int(info.get("shift_idx") or 0))
    audio_length = max(0, int(info.get("audio_length") or 0))
    segment_offset = max(0, int(info.get("segment_offset") or 0))
    segment_ratio = min(segment_offset / audio_length, 1) if audio_length else 0
    total_units = max(1, models * shifts)
    completed_units = model_index * shifts + shift_index + segment_ratio
    return max(0, min(99, int(completed_units / total_units * 100)))


DEFAULT_DEMUCS_MODEL = "htdemucs_ft"
DEFAULT_DEMUCS_SHIFTS = 1


def _enable_amp(separator: object) -> None:
    """Monkey-patch the Separator to use AMP (float16) inference.

    RTX 3090 (Ampere, sm_86) has dedicated Tensor Cores for float16.
    With only ~1.5 GB model weights, AMP uses ~2.5 GB total — well within 24 GB.
    Expected speedup: 1.5–2× on the model forward pass.
    """
    import functools
    from demucs import apply as _apply_mod

    _original_apply = _apply_mod.apply_model

    @functools.wraps(_original_apply)
    def _apply_model_amp(*args, **kwargs):
        device = kwargs.get("device") or (args[0].device if hasattr(args[0], "device") else "cpu")
        device_str = str(device) if not isinstance(device, str) else device
        if _is_cuda(device_str):
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                return _original_apply(*args, **kwargs)
        return _original_apply(*args, **kwargs)

    _apply_mod.apply_model = _apply_model_amp  # type: ignore[assignment]


def _compile_model(separator: object) -> None:
    """Compile sub-models with torch.compile for additional speedup.

    First call is slow (compilation), but subsequent calls are 10-30% faster.
    Only effective on PyTorch >= 2.0.
    """
    if not hasattr(torch, "compile"):
        return
    model = separator._model
    if hasattr(model, "models"):
        for i, sub in enumerate(model.models):
            model.models[i] = torch.compile(sub, mode="reduce-overhead")  # type: ignore[union-attr]
    else:
        separator._model = torch.compile(model, mode="reduce-overhead")  # type: ignore[assignment]


def separate_audio(
    video_file: Path,
    session: Path,
    progress_callback: Callable[[int, str], None] | None = None,
    demucs_model: str | None = None,
    shifts: int | None = None,
    use_amp: bool | None = None,
) -> tuple[Path, Path]:
    demucs_path = _demucs_source_path()
    sys.path.insert(0, str(demucs_path))

    from demucs.api import Separator, save_audio

    media_dir = session / "media"
    vocals_file = media_dir / "audio_vocals.wav"
    bgm_file = media_dir / "audio_bgm.wav"
    if vocals_file.exists() and bgm_file.exists():
        return vocals_file, bgm_file

    model_name = demucs_model or DEFAULT_DEMUCS_MODEL
    shifts = shifts if shifts is not None else DEFAULT_DEMUCS_SHIFTS
    device = _device()

    # Default: enable AMP on CUDA (RTX 3090 / Ampere+ benefits significantly)
    if use_amp is None:
        use_amp = _is_cuda(device)

    def report_progress(info: dict) -> None:
        if progress_callback is None:
            return
        progress = _demucs_progress(info, shifts)
        progress_callback(progress, f"Separating audio {progress}%")

    separator = Separator(
        model=model_name,
        device=device,
        progress=True,
        shifts=shifts,
        callback=report_progress,
    )

    if use_amp and _is_cuda(device):
        _enable_amp(separator)

    _, separated = separator.separate_audio_file(str(video_file))

    vocals = separated["vocals"]
    bgm = None
    for stem, source in separated.items():
        if stem == "vocals":
            continue
        bgm = source if bgm is None else bgm + source

    save_audio(vocals, str(vocals_file), samplerate=separator.samplerate)
    save_audio(bgm, str(bgm_file), samplerate=separator.samplerate)
    return vocals_file, bgm_file


def _demucs_source_path() -> Path:
    demucs_path = REPO_ROOT / "submodule" / "demucs"
    api_file = demucs_path / "demucs" / "api.py"
    if api_file.exists():
        return demucs_path
    raise RuntimeError(
        "Demucs source submodule is missing or incomplete. "
        "Clone this repository with git and run: git submodule update --init --recursive. "
        "Do not use GitHub Download ZIP because it does not include submodules."
    )
