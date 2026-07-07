"""ARK-ASR-3B speech recognition adapter (HuggingFace Transformers backend).

Uses the ``AutoArk-AI/ARK-ASR-3B`` model via standard HuggingFace
``AutoModelForCausalLM`` with ``trust_remote_code=True``.  ARK-ASR-3B
is a 4B-scale autoregressive ASR model that outputs text only (no
timestamps).  To provide sentence-level timestamps required by YouDub's
pipeline, we pair it with FunASR's ``fsmn-vad`` to pre-segment the audio
into ≤30 s chunks, then transcribe each chunk independently.

Install: no extra dependencies needed — uses ``transformers`` (already
in requirements) and ``fsmn-vad`` from ``funasr`` (already in
requirements).
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
from ._lang_map import LANG_TO_ARK_ASR as _LANG_MAP

logger = logging.getLogger(__name__)

_MODEL = None
_PROCESSOR = None
_TOKENIZER = None


# ---------------------------------------------------------------------------
# 模型路径与设备
# ---------------------------------------------------------------------------

def _model_dir() -> str | None:
    """Return the local model directory if it exists, else ``None``."""
    cache = os.getenv("MODEL_CACHE_DIR", "").strip()
    if cache:
        candidate = Path(cache) / "models" / "AutoArk-AI" / "ARK-ASR-3B"
        if candidate.is_dir():
            return str(candidate)
    return None


def _ensure_local_path(repo_id: str) -> str:
    """Ensure *repo_id* is a local directory, downloading via huggingface_hub if needed.

    modelscope monkey-patches ``AutoProcessor.from_pretrained`` and breaks on
    remote repo IDs (``TypeError: 'tuple' object does not support item
    assignment``).  By downloading to ``MODEL_CACHE_DIR`` first and returning
    the local path, we bypass the patcher entirely.
    """
    # Already a local path that exists
    if os.path.isdir(repo_id):
        return repo_id

    cache = os.getenv("MODEL_CACHE_DIR", "").strip()
    if not cache:
        # No cache dir — let transformers handle download (may fail if
        # modelscope patcher is active, but we have no alternative).
        return repo_id

    local_dir = str(Path(cache) / "models" / repo_id.replace("/", os.sep))
    if os.path.isdir(local_dir):
        return local_dir

    from huggingface_hub import snapshot_download
    logger.info("Downloading %s to %s", repo_id, local_dir)
    return snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        ignore_patterns=["*.safetensors.index.json", "*.bin"],
    )


def _resolve_device() -> str:
    """Resolve the compute device, mirroring Qwen3-ASR's logic."""
    device = os.getenv("DEVICE", "cuda").strip()
    if device == "auto":
        import torch
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        return "cuda:0"
    return device


# ---------------------------------------------------------------------------
# 模型加载 / 释放
# ---------------------------------------------------------------------------

def _load_model():
    """Load ARK-ASR-3B model, processor and tokenizer (lazy singleton)."""
    global _MODEL, _PROCESSOR, _TOKENIZER
    if _MODEL is not None:
        return _MODEL, _PROCESSOR, _TOKENIZER

    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

    model_path = _model_dir() or "AutoArk-AI/ARK-ASR-3B"
    model_path = _ensure_local_path(model_path)
    device = _resolve_device()
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32

    logger.info("Loading ARK-ASR-3B from %s on %s", model_path, device)

    _PROCESSOR = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    _TOKENIZER = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    _MODEL = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=dtype,
        attn_implementation="sdpa",
    ).to(device)
    _MODEL.eval()

    return _MODEL, _PROCESSOR, _TOKENIZER


def release_model() -> None:
    """Release ARK-ASR-3B from GPU memory.

    Mirrors Qwen3-ASR's cleanup: multi-pass GC + CUDA cache clear +
    orphan spawn child termination.
    """
    global _MODEL, _PROCESSOR, _TOKENIZER
    if _MODEL is None:
        return
    _MODEL = None
    _PROCESSOR = None
    _TOKENIZER = None

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
    logger.info("ARK-ASR-3B models released from GPU memory")


def _cleanup_spawn_children() -> None:
    """Terminate orphan multiprocessing.spawn children (same as Qwen3-ASR)."""
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
# VAD 切段
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

    Each segment is guaranteed ≤ 30 000 ms to satisfy ARK-ASR-3B's
    ``audio_max_length = 30 * 16000`` limit.  Segments shorter than
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

    # 拆分超过 30s 的段
    MAX_MS = 30_000
    segments: list[tuple[int, int]] = []
    for start_ms, end_ms in raw_segments:
        while end_ms - start_ms > MAX_MS:
            segments.append((start_ms, start_ms + MAX_MS))
            start_ms += MAX_MS
        segments.append((start_ms, end_ms))

    return segments


# ---------------------------------------------------------------------------
# 音频截取
# ---------------------------------------------------------------------------

def _extract_segment(
    audio_path: str, start_ms: int, end_ms: int, sr: int = 16000
) -> np.ndarray:
    """Extract a time segment from *audio_path* as a float32 numpy array.

    Uses pydub for format-agnostic loading, then slices by ms offsets.
    Returns a 1-D float32 array normalised to [-1, 1].
    """
    audio = AudioSegment.from_file(audio_path)
    segment = audio[start_ms:end_ms]

    samples = np.array(segment.get_array_of_samples(), dtype=np.float32)
    if segment.channels == 2:
        samples = samples.reshape(-1, 2).mean(axis=1)
    # pcm_s16le → normalise
    samples /= 32768.0
    # Resample if needed
    if segment.frame_rate != sr:
        import librosa
        samples = librosa.resample(samples, orig_sr=segment.frame_rate, target_sr=sr)
    return samples


# ---------------------------------------------------------------------------
# ASR 推理
# ---------------------------------------------------------------------------

def _build_bad_words_ids(tokenizer) -> list[list[int]]:
    """Build ``bad_words_ids`` to suppress non-ASR special tokens.

    ARK-ASR uses a chat-template format with tokens like ``<|user|>``,
    ``<|audio|>``, ``<|begin_of_audio|>`` etc.  During ASR generation we
    only want the EOS token to be allowed among special tokens.
    """
    eos_ids = tokenizer.eos_token_id
    keep_ids = {eos_ids} if isinstance(eos_ids, int) else set(eos_ids or [])
    bad_ids = set(tokenizer.all_special_ids) - keep_ids
    bad_ids.update(
        token_id
        for token, token_id in tokenizer.get_added_vocab().items()
        if token.startswith("<") and token.endswith(">") and token_id not in keep_ids
    )
    return [[token_id] for token_id in sorted(bad_ids)]


def _transcribe_chunk(
    audio_array: np.ndarray,
    model,
    processor,
    tokenizer,
    bad_words_ids: list[list[int]],
    device: str,
    dtype,
) -> str:
    """Transcribe a single audio chunk (numpy float32 array) → text string."""
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "array": audio_array, "sampling_rate": 16000},
                {"type": "text", "text": "Please transcribe this audio."},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        return_tensors="pt",
        sampling_rate=16000,
        audio_padding="longest",
        text_kwargs={"padding": "longest"},
        audio_max_length=30 * 16000,
    )
    inputs = inputs.to(device)
    if "audios" in inputs:
        inputs["audios"] = inputs["audios"].to(dtype=dtype)

    import torch
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=512,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            bad_words_ids=bad_words_ids,
        )
    decoded = tokenizer.batch_decode(
        outputs[:, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )
    text = (decoded[0] if decoded else "").strip()
    # 清理可能残留的 prompt 回显
    text = re.sub(r"^Please transcribe this audio\.?\s*", "", text, flags=re.IGNORECASE)
    return text.strip()


# ---------------------------------------------------------------------------
# 句子级拆分
# ---------------------------------------------------------------------------

_SENTENCE_END_RE = re.compile(r'(?<=[.!?\u3002\uff01\uff1f])\s+')

def _split_text_to_sentences(text: str) -> list[str]:
    """Split *text* into sentences by sentence-ending punctuation.

    Handles English (.!?), Chinese (。！？), and mixed punctuation.
    Preserves the punctuation at the end of each sentence.
    """
    if not text or not text.strip():
        return []

    # 按句末标点切分，保留标点
    # 使用正向后顾断言：在标点后、空白前切分
    parts: list[str] = []
    current: list[str] = []
    for ch in text:
        current.append(ch)
        if ch in '.!?!\u3002\uff01\uff1f\uff1b':
            chunk = ''.join(current).strip()
            if chunk:
                parts.append(chunk)
            current = []
    # 尾部无标点的残留
    trailing = ''.join(current).strip()
    if trailing:
        parts.append(trailing)
    return parts if len(parts) > 1 else ([text.strip()] if text.strip() else [])


def _distribute_utterances(
    text: str,
    start_ms: int,
    end_ms: int,
) -> list[dict[str, Any]]:
    """Split *text* by punctuation and distribute timestamps proportionally.

    Returns a list of utterance dicts with ``text``, ``start_time``,
    ``end_time``, ``words: []``.  If the text has only one sentence (or
    no punctuation), returns a single utterance spanning the full range.
    """
    sentences = _split_text_to_sentences(text)
    if len(sentences) <= 1:
        return [{
            "text": text,
            "start_time": start_ms,
            "end_time": end_ms,
            "words": [],
        }]

    # 按字符数比例分配时间
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
            # 确保不超过总结束时间
            sent_end = min(sent_end, end_ms)
        result.append({
            "text": sent,
            "start_time": cursor,
            "end_time": sent_end,
            "words": [],
        })
        cursor = sent_end
    return result


# ---------------------------------------------------------------------------
# 公开接口
# ---------------------------------------------------------------------------

def recognize_speech(
    vocals_file: Path, session: Path, language: str, *, model_id: str | None = None
) -> Path:
    """Recognize speech using ARK-ASR-3B with FSMN-VAD pre-segmentation.

    Pipeline:
    1. Run ``fsmn-vad`` to get time-stamped speech segments.
    2. For each segment, extract audio and transcribe with ARK-ASR-3B.
    3. Assemble ``asr.json`` with VAD timestamps as sentence-level boundaries.

    Returns the path to ``metadata/asr.json``.
    """
    metadata_dir = session / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    output_file = metadata_dir / "asr.json"
    if output_file.exists():
        return output_file

    vocals_path = str(vocals_file)

    # Step 1: VAD 切段
    logger.info("ARK-ASR: running FSMN-VAD on %s", vocals_file.name)
    segments = _vad_segments(vocals_path)
    if not segments:
        # VAD 未检测到语音，使用整段音频
        dur_ms = len(AudioSegment.from_file(vocals_file))
        segments = [(0, dur_ms)]

    # Step 2: 加载模型
    model, processor, tokenizer = _load_model()
    device = _resolve_device()
    dtype = next(model.parameters()).dtype
    bad_words_ids = _build_bad_words_ids(tokenizer)

    # Step 3: 逐段识别
    utterances: list[dict[str, Any]] = []
    full_text_parts: list[str] = []
    ark_lang = _LANG_MAP.get(language)

    for idx, (start_ms, end_ms) in enumerate(segments):
        audio_array = _extract_segment(vocals_path, start_ms, end_ms)
        if audio_array.size == 0:
            continue

        text = _transcribe_chunk(
            audio_array, model, processor, tokenizer, bad_words_ids, device, dtype
        )
        if not text:
            continue

        # 按句末标点拆分长段，按比例分配时间戳
        chunk_utterances = _distribute_utterances(text, start_ms, end_ms)
        utterances.extend(chunk_utterances)
        full_text_parts.append(text)

        if (idx + 1) % 10 == 0:
            logger.info("ARK-ASR: processed %d/%d segments", idx + 1, len(segments))

    if not utterances:
        raise RuntimeError("ARK-ASR-3B did not produce any utterances.")

    # Step 4: 保存原始结果
    duration_ms = len(AudioSegment.from_file(vocals_file))
    raw_output = metadata_dir / "asr.raw.arkasr.json"
    raw_output.write_text(
        json.dumps(
            {
                "segments": [
                    {"start_ms": s, "end_ms": e, "index": i,
                     "text": full_text_parts[i] if i < len(full_text_parts) else ""}
                    for i, (s, e) in enumerate(segments)
                ],
                "full_text": " ".join(full_text_parts),
                "utterances": utterances,
                "language": language,
                "model": "AutoArk-AI/ARK-ASR-3B",
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
        "ARK-ASR: recognized %d segments -> %s", len(utterances), output_file.name
    )
    return output_file
