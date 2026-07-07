"""MOSS-Audio-4B speech recognition adapter.

Uses the ``MossAudioModel`` and ``MossAudioProcessor`` from the
MOSS-Audio repository (``OpenMOSS/MOSS-Audio``).  MOSS-Audio is a
4.6B-parameter audio understanding model built on Qwen3-4B that
supports time-aware ASR — timestamps are embedded as text markers
(e.g. ``[0.0s] Hello [2.0s] world``) in the generated output.

Requires: ``git clone https://github.com/OpenMOSS/MOSS-Audio && cd
MOSS-Audio && pip install -e ".[torch-runtime]"``

VAD pre-segmentation uses FunASR's ``fsmn-vad`` (same as ARK-ASR).
"""

from __future__ import annotations

import gc
import json
import logging
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
from pydub import AudioSegment

from ._time_utils import to_ms as _to_ms
from ._lang_map import LANG_TO_MOSS_AUDIO as _LANG_MAP

logger = logging.getLogger(__name__)

_MODEL = None
_PROCESSOR = None


# ---------------------------------------------------------------------------
# 模型路径与设备
# ---------------------------------------------------------------------------

_REPO_ID = "OpenMOSS-Team/MOSS-Audio-4B-Instruct"


def _model_dir() -> str | None:
    """Return the local model directory if it exists, else ``None``."""
    cache = os.getenv("MODEL_CACHE_DIR", "").strip()
    if cache:
        candidate = Path(cache) / "models" / "OpenMOSS-Team" / "MOSS-Audio-4B-Instruct"
        if candidate.is_dir():
            return str(candidate)
    return None


def _ensure_local_path(repo_id: str) -> str:
    """Ensure *repo_id* is a local directory, downloading via huggingface_hub if needed.

    modelscope monkey-patches HuggingFace's ``from_pretrained`` and breaks
    on remote repo IDs.  By downloading to ``MODEL_CACHE_DIR`` first and
    returning the local path, we bypass the patcher entirely.
    """
    if os.path.isdir(repo_id):
        return repo_id

    cache = os.getenv("MODEL_CACHE_DIR", "").strip()
    if not cache:
        return repo_id

    local_dir = str(Path(cache) / "models" / repo_id.replace("/", os.sep))
    if os.path.isdir(local_dir):
        return local_dir

    from huggingface_hub import snapshot_download
    logger.info("Downloading %s to %s", repo_id, local_dir)
    return snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
    )


def _resolve_device() -> str:
    """Resolve the compute device."""
    device = os.getenv("DEVICE", "cuda").strip()
    if device == "auto":
        import torch
        if torch.cuda.is_available():
            return "cuda:0"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if device == "cuda":
        return "cuda:0"
    return device


# ---------------------------------------------------------------------------
# 模型加载 / 释放
# ---------------------------------------------------------------------------

def _load_model():
    """Load MOSS-Audio model and processor (lazy singleton)."""
    global _MODEL, _PROCESSOR
    if _MODEL is not None:
        return _MODEL, _PROCESSOR

    # 尝试导入 MOSS-Audio 自定义类
    try:
        from src.modeling_moss_audio import MossAudioModel
        from src.processing_moss_audio import MossAudioProcessor
    except ImportError:
        raise RuntimeError(
            "MOSS-Audio is not installed.  Clone the repository and install it:\n"
            "  git clone https://github.com/OpenMOSS/MOSS-Audio.git\n"
            "  cd MOSS-Audio\n"
            "  pip install -e '.[torch-runtime]'\n"
            "See https://github.com/OpenMOSS/MOSS-Audio for details."
        )

    model_path = _model_dir() or _REPO_ID
    model_path = _ensure_local_path(model_path)
    device = _resolve_device()

    logger.info("Loading MOSS-Audio-4B from %s on %s", model_path, device)

    _MODEL = MossAudioModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        dtype="auto",
        device_map=device,
    )
    _MODEL.eval()

    _PROCESSOR = MossAudioProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
        enable_time_marker=True,
    )

    return _MODEL, _PROCESSOR


def release_model() -> None:
    """Release MOSS-Audio from GPU memory."""
    global _MODEL, _PROCESSOR
    if _MODEL is None:
        return
    _MODEL = None
    _PROCESSOR = None

    for _ in range(3):
        gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass

    _cleanup_spawn_children()
    logger.info("MOSS-Audio models released from GPU memory")


def _cleanup_spawn_children() -> None:
    """Terminate orphan multiprocessing.spawn children."""
    try:
        import psutil
    except ImportError:
        return
    try:
        current = psutil.Process()
        for child in current.children(recursive=True):
            try:
                cmdline = child.cmdline()
                if any("multiprocessing.spawn" in part for part in cmdline):
                    child.terminate()
                    try:
                        child.wait(timeout=5)
                    except psutil.TimeoutExpired:
                        child.kill()
                    logger.info("Terminated orphan spawn child PID=%d", child.pid)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    except Exception as exc:
        logger.warning("Orphan spawn child cleanup skipped: %s", exc)


# ---------------------------------------------------------------------------
# VAD 切段（与 ARK-ASR 共用逻辑）
# ---------------------------------------------------------------------------

_VAD_MODEL = None


def _load_vad_model():
    """Load the FSMN-VAD model (lazy singleton)."""
    global _VAD_MODEL
    if _VAD_MODEL is not None:
        return _VAD_MODEL

    from funasr import AutoModel

    device = os.getenv("FUNASR_DEVICE") or os.getenv("DEVICE", "auto")
    if device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"

    vad_model_id = os.getenv("FUNASR_VAD_MODEL", "fsmn-vad")
    _VAD_MODEL = AutoModel(model=vad_model_id, device=device)
    return _VAD_MODEL


def _vad_segments(audio_path: str) -> list[tuple[int, int]]:
    """Run FSMN-VAD and return ``[(start_ms, end_ms), ...]`` segments.

    Each segment is guaranteed ≤ 30 000 ms.  Segments shorter than
    200 ms are dropped (likely noise).
    """
    vad = _load_vad_model()
    result = vad.generate(input=audio_path, cache={})

    raw_segments: list[tuple[int, int]] = []
    for item in result:
        ts = item.get("timestamp") or []
        for pair in ts:
            if not isinstance(pair, (list, tuple)) or len(pair) < 2:
                continue
            start_ms = int(pair[0])
            end_ms = int(pair[1])
            if end_ms <= start_ms:
                continue
            if end_ms - start_ms < 200:
                continue
            raw_segments.append((start_ms, end_ms))

    MAX_MS = 30_000
    segments: list[tuple[int, int]] = []
    for start_ms, end_ms in raw_segments:
        while end_ms - start_ms > MAX_MS:
            segments.append((start_ms, start_ms + MAX_MS))
            start_ms += MAX_MS
        segments.append((start_ms, end_ms))

    return segments


# ---------------------------------------------------------------------------
# 音频加载
# ---------------------------------------------------------------------------

def _load_audio(audio_path: str, sample_rate: int = 16000) -> np.ndarray:
    """Load audio as a 1-D float32 numpy array at *sample_rate* Hz.

    Uses torchaudio for loading (same as MOSS-Audio's ``audio_io``).
    Falls back to pydub if torchaudio fails.
    """
    try:
        import torchaudio
        waveform, orig_sr = torchaudio.load(audio_path)
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if orig_sr != sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, orig_freq=orig_sr, new_freq=sample_rate
            )
        return waveform.squeeze(0).cpu().numpy()
    except Exception:
        # Fallback: pydub
        audio = AudioSegment.from_file(audio_path)
        samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
        if audio.channels == 2:
            samples = samples.reshape(-1, 2).mean(axis=1)
        samples /= 32768.0
        if audio.frame_rate != sample_rate:
            import librosa
            samples = librosa.resample(samples, orig_sr=audio.frame_rate, target_sr=sample_rate)
        return samples


def _extract_segment(
    audio_path: str, start_ms: int, end_ms: int, sr: int = 16000
) -> np.ndarray:
    """Extract a time segment from *audio_path* as a float32 numpy array."""
    audio = AudioSegment.from_file(audio_path)
    segment = audio[start_ms:end_ms]
    samples = np.array(segment.get_array_of_samples(), dtype=np.float32)
    if segment.channels == 2:
        samples = samples.reshape(-1, 2).mean(axis=1)
    samples /= 32768.0
    if segment.frame_rate != sr:
        import librosa
        samples = librosa.resample(samples, orig_sr=segment.frame_rate, target_sr=sr)
    return samples


# ---------------------------------------------------------------------------
# 时间标记解析
# ---------------------------------------------------------------------------

# 匹配 [X.XX] 或 [X] 格式的时间标记（MOSS-Audio 输出无 "s" 后缀）
_TIME_MARKER_RE = re.compile(r'\[(\d+(?:\.\d+)?)\]')


def _parse_timestamped_text(text: str) -> list[dict[str, Any]] | None:
    """Parse MOSS-Audio's time-marked output into utterances.

    MOSS-Audio outputs timestamps in the format ``[0.06]text [2.64][3.04]text``
    where timestamps are in seconds.  Consecutive timestamps without text
    between them (e.g. ``[2.64][3.04]``) mark a boundary: the last one is
    the start of the next sentence.

    Returns a list of ``{"text": str, "start_ms": int}`` or ``None``
    if no time markers are found.
    """
    # Step 1: 找到所有时间标记的位置
    markers = list(_TIME_MARKER_RE.finditer(text))
    if not markers:
        return None

    # Step 2: 对每个标记，判断它后面是否紧跟文本（非标记内容）
    # 如果标记后紧跟另一个标记，则它是边界标记，跳过
    # 只取"最后一个连续标记"作为句子的 start_ms
    entries: list[dict[str, Any]] = []
    i = 0
    while i < len(markers):
        # 找到连续标记序列 [m_i][m_i+1]...[m_j]，其中 m_j 后面有文本
        j = i
        while j + 1 < len(markers):
            # 检查 markers[j] 和 markers[j+1] 之间是否有文本
            between = text[markers[j].end():markers[j + 1].start()].strip()
            if between:
                break
            j += 1

        # markers[j] 是这个序列的最后一个标记
        start_s = float(markers[j].group(1))
        start_ms = _to_ms(start_s)

        # 文本范围：markers[j].end() → 下一个标记前（或文本末尾）
        text_start = markers[j].end()
        if j + 1 < len(markers):
            text_end = markers[j + 1].start()
        else:
            text_end = len(text)

        chunk = text[text_start:text_end].strip()
        if chunk:
            entries.append({"text": chunk, "start_ms": start_ms})

        i = j + 1

    if not entries:
        return None

    # Step 3: 为每个 entry 计算 end_ms（下一个 entry 的 start_ms）
    for k, entry in enumerate(entries):
        if k + 1 < len(entries):
            entry["end_ms"] = entries[k + 1]["start_ms"]
        else:
            entry["end_ms"] = None  # 后续由 VAD 段边界填充

    return entries


def _distribute_with_time_markers(
    text: str,
    vad_start_ms: int,
    vad_end_ms: int,
) -> list[dict[str, Any]]:
    """Parse time markers from *text* and build utterances with timestamps.

    MOSS-Audio's time markers are relative to the input audio segment.
    We offset them by *vad_start_ms* to get absolute timestamps, then
    clamp to the VAD segment range ``[vad_start_ms, vad_end_ms]``.

    Falls back to punctuation-based splitting if no time markers found.
    """
    parsed = _parse_timestamped_text(text)
    if parsed:
        # 时间标记是相对于输入音频段的，需要加上 VAD 段起始偏移
        for u in parsed:
            u["start_ms"] += vad_start_ms
            if u["end_ms"] is not None:
                u["end_ms"] += vad_start_ms
        # 用 VAD 段边界填充缺失的 end_ms
        for u in parsed:
            if u["end_ms"] is None:
                u["end_ms"] = vad_end_ms
            # 确保 start/end 在 VAD 段范围内
            u["start_ms"] = max(u["start_ms"], vad_start_ms)
            u["end_ms"] = min(u["end_ms"], vad_end_ms)
            if u["end_ms"] <= u["start_ms"]:
                u["end_ms"] = min(vad_end_ms, u["start_ms"] + 1)
        return [
            {
                "text": u["text"],
                "start_time": u["start_ms"],
                "end_time": u["end_ms"],
                "words": [],
            }
            for u in parsed
        ]

    # 无时间标记 → 按标点断句 + 按比例分配
    return _distribute_by_punctuation(text, vad_start_ms, vad_end_ms)


_SENTENCE_PUNCT = set(".!?!。！？；;：:")


def _split_text_to_sentences(text: str) -> list[str]:
    """Split *text* by sentence-ending punctuation, preserving punct."""
    if not text or not text.strip():
        return []
    parts: list[str] = []
    current: list[str] = []
    for ch in text:
        current.append(ch)
        if ch in _SENTENCE_PUNCT:
            chunk = "".join(current).strip()
            if chunk:
                parts.append(chunk)
            current = []
    trailing = "".join(current).strip()
    if trailing:
        parts.append(trailing)
    return parts if len(parts) > 1 else ([text.strip()] if text.strip() else [])


def _distribute_by_punctuation(
    text: str, start_ms: int, end_ms: int
) -> list[dict[str, Any]]:
    """Split *text* by punctuation and distribute timestamps proportionally."""
    sentences = _split_text_to_sentences(text)
    if len(sentences) <= 1:
        return [{"text": text, "start_time": start_ms, "end_time": end_ms, "words": []}]

    total_chars = max(1, sum(len(s) for s in sentences))
    duration = end_ms - start_ms
    result: list[dict[str, Any]] = []
    cursor = start_ms
    for i, sent in enumerate(sentences):
        weight = len(sent) / total_chars
        if i == len(sentences) - 1:
            sent_end = end_ms
        else:
            sent_end = cursor + int(round(duration * weight))
            sent_end = min(sent_end, end_ms)
        result.append({"text": sent, "start_time": cursor, "end_time": sent_end, "words": []})
        cursor = sent_end
    return result


# ---------------------------------------------------------------------------
# ASR 推理
# ---------------------------------------------------------------------------

def _transcribe_chunk(
    audio_array: np.ndarray,
    model,
    processor,
    device: str,
) -> str:
    """Transcribe a single audio chunk → text string with time markers."""
    import torch

    prompt = "Please transcribe this audio with timestamps."
    inputs = processor(
        text=prompt,
        audios=[audio_array],
        return_tensors="pt",
    )
    inputs = inputs.to(device)
    if inputs.get("audio_data") is not None:
        inputs["audio_data"] = inputs["audio_data"].to(model.dtype)

    # 构建音频输入掩码
    audio_input_mask = inputs["input_ids"] == processor.audio_token_id
    inputs["audio_input_mask"] = audio_input_mask

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=False,
            num_beams=1,
            use_cache=True,
        )

    input_len = inputs["input_ids"].shape[1]
    text = processor.decode(
        generated_ids[0, input_len:],
        skip_special_tokens=True,
    )
    # 清理可能的 <think>...</think> 标签（Thinking 变体）
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    return text


# ---------------------------------------------------------------------------
# 公开接口
# ---------------------------------------------------------------------------

def recognize_speech(
    vocals_file: Path, session: Path, language: str, *, model_id: str | None = None
) -> Path:
    """Recognize speech using MOSS-Audio-4B with FSMN-VAD pre-segmentation.

    Pipeline:
    1. Run ``fsmn-vad`` to get time-stamped speech segments.
    2. For each segment, transcribe with MOSS-Audio-4B.
    3. Parse time markers from output to get sentence-level timestamps.
    4. Assemble ``asr.json``.

    Returns the path to ``metadata/asr.json``.
    """
    metadata_dir = session / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    output_file = metadata_dir / "asr.json"
    if output_file.exists():
        return output_file

    vocals_path = str(vocals_file)

    # Step 1: VAD 切段
    logger.info("MOSS-Audio: running FSMN-VAD on %s", vocals_file.name)
    segments = _vad_segments(vocals_path)
    if not segments:
        dur_ms = len(AudioSegment.from_file(vocals_file))
        segments = [(0, dur_ms)]

    # Step 2: 加载模型
    model, processor = _load_model()
    device = _resolve_device()

    # Step 3: 逐段识别
    utterances: list[dict[str, Any]] = []
    full_text_parts: list[str] = []
    raw_segments: list[dict[str, Any]] = []

    for idx, (start_ms, end_ms) in enumerate(segments):
        audio_array = _extract_segment(vocals_path, start_ms, end_ms)
        if audio_array.size == 0:
            continue

        text = _transcribe_chunk(audio_array, model, processor, device)
        if not text:
            continue

        raw_segments.append({
            "start_ms": start_ms,
            "end_ms": end_ms,
            "index": idx,
            "raw_text": text,
        })

        # 解析时间标记或回退到标点断句
        chunk_utterances = _distribute_with_time_markers(text, start_ms, end_ms)
        utterances.extend(chunk_utterances)
        full_text_parts.append(text)

        if (idx + 1) % 10 == 0:
            logger.info("MOSS-Audio: processed %d/%d segments", idx + 1, len(segments))

    if not utterances:
        raise RuntimeError("MOSS-Audio did not produce any utterances.")

    # Step 4: 保存原始结果
    duration_ms = len(AudioSegment.from_file(vocals_file))
    raw_output = metadata_dir / "asr.raw.moss_audio.json"
    raw_output.write_text(
        json.dumps(
            {
                "segments": raw_segments,
                "full_text": " ".join(full_text_parts),
                "utterances": utterances,
                "language": language,
                "model": "OpenMOSS-Team/MOSS-Audio-4B-Instruct",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # Step 5: 写入 asr.json
    full_text = " ".join(full_text_parts)
    payload = {
        "audio_info": {"duration": duration_ms},
        "result": {
            "text": full_text,
            "utterances": utterances,
        },
    }
    output_file.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info(
        "MOSS-Audio: recognized %d segments -> %s", len(utterances), output_file.name
    )
    return output_file
