"""FunASR speech recognition adapter with vLLM support for LLM-based models.

Dispatches to one of two inference backends:

* **Standard** :func:`funasr.auto.auto_model.AutoModel` for non-LLM models
  (SenseVoice, Paraformer, ...).
* **vLLM** :func:`funasr.auto.auto_model_vllm.AutoModelVLLM` for LLM-based
  models (Fun-ASR-Nano, GLM-ASR-Nano, LLMASR) that need vLLM's
  PagedAttention + Continuous Batching to drive the autoregressive LLM
  decoder — see https://github.com/modelscope/FunASR/blob/main/docs/vllm_guide.md

The dispatch is governed by the ``funasr.use_vllm`` setting
(``auto``/``on``/``off``) and the availability of the ``vllm`` Python
package. With ``auto`` the vLLM path is only used for LLM-based models
and falls back to the standard path with a warning when vllm is not
installed.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Callable

from pydub import AudioSegment

logger = logging.getLogger(__name__)

_STANDARD_MODEL = None
_STANDARD_MODEL_NAME: str | None = None

_VLLM_MODEL = None
_VLLM_MODEL_NAME: str | None = None

# LLM-based FunASR models that require the vLLM engine for batch decoding.
# The standard AutoModel batch path raises
# ``NotImplementedError("batch decoding is not implemented")`` in
# funasr/models/fun_asr_nano/model.py when ``len(data_in) > 1``.
# ``Fun-ASR`` covers both ``Fun-ASR-Nano-2512`` and ``Fun-ASR-MLT-Nano-2512``.
_LLM_BASED_MODEL_KEYWORDS = (
    "Fun-ASR",
    "GLM-ASR-Nano",
    "LLMASR",
)

# Per-character timestamps come back from vLLM via CTC forced alignment;
# we re-group them into coarse sentence segments using these terminators.
_SENTENCE_PUNCTUATION = re.compile(r"([。！？.!?\n]+)")

_LANG_TO_FUNASR = {
    "zh": "zh",
    "en": "en",
    "ja": "ja",
    "ko": "ko",
    "yue": "yue",
}

_LANG_TO_FUNASR_VLLM = {
    "zh": "中文",
    "en": "英文",
    "ja": "日文",
    "ko": "韩文",
    "yue": "粤语",
}


def _get_model_id() -> str:
    return os.getenv("FUNASR_MODEL", "iic/SenseVoiceSmall")


def _get_vad_model() -> str:
    return os.getenv("FUNASR_VAD_MODEL", "fsmn-vad")


def _is_llm_based_model(model_id: str) -> bool:
    """Return True if *model_id* is an LLM-based FunASR model.

    These models use an autoregressive LLM decoder, which the standard
    AutoModel batch path cannot drive. The vLLM engine wraps the LLM
    in PagedAttention + Continuous Batching to handle multi-segment
    inputs natively.
    """
    return any(keyword in model_id for keyword in _LLM_BASED_MODEL_KEYWORDS)


def _vllm_available() -> bool:
    """Return True when the ``vllm`` Python package is importable."""
    return importlib.util.find_spec("vllm") is not None


def _resolve_vllm_mode(model_id: str) -> tuple[bool, str | None]:
    """Decide whether to use the vLLM engine for *model_id*.

    Returns ``(use_vllm, warning_message)``. When the user explicitly
    chose ``on`` but vllm is missing, raises :class:`RuntimeError`; when
    ``auto`` and vllm is missing, returns ``(False, warning)`` so the
    caller can fall back to the standard path.
    """
    from .. import database

    choice = (database.get_funasr_settings().get("use_vllm") or "auto").strip().lower()
    if choice == "off":
        return False, None
    if choice == "on":
        if not _vllm_available():
            raise RuntimeError(
                "FunASR vLLM is forced on (FUNASR_USE_VLLM=on) but the "
                "vllm package is not installed. Run `pip install vllm` or "
                "set FUNASR_USE_VLLM=auto/off in settings."
            )
        return True, None
    # auto
    if _is_llm_based_model(model_id):
        if _vllm_available():
            return True, None
        warning = (
            f"FunASR model '{model_id}' requires the vLLM engine for "
            "batch decoding, but the vllm package is not installed. "
            "Falling back to the standard AutoModel path — this will "
            "raise NotImplementedError for LLM-based models. Install "
            "vllm (pip install vllm) or set FUNASR_USE_VLLM=off."
        )
        return False, warning
    return False, None


def _load_standard_model(model_id: str | None = None):
    """Load a standard :class:`funasr.AutoModel` (SenseVoice, Paraformer, ...)."""
    global _STANDARD_MODEL, _STANDARD_MODEL_NAME
    target = model_id or _get_model_id()
    if _STANDARD_MODEL is not None and _STANDARD_MODEL_NAME == target:
        return _STANDARD_MODEL

    from funasr import AutoModel

    device = os.getenv("FUNASR_DEVICE") or os.getenv("DEVICE", "auto")
    if device == "auto":
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"

    kwargs = dict(
        model=target,
        vad_model=_get_vad_model(),
        vad_kwargs={"max_single_segment_time": 30000},
        device=device,
    )
    if "Fun-ASR-Nano" in target or "Fun-ASR" in target:
        kwargs["trust_remote_code"] = True

    _STANDARD_MODEL = AutoModel(**kwargs)
    _STANDARD_MODEL_NAME = target
    return _STANDARD_MODEL


def _load_vllm_model(model_id: str | None = None):
    """Load a vLLM-backed FunASR model (Fun-ASR-Nano, GLM-ASR-Nano, ...).

    Reference: https://github.com/modelscope/FunASR/blob/main/docs/vllm_guide.md
    """
    global _VLLM_MODEL, _VLLM_MODEL_NAME
    target = model_id or _get_model_id()
    if _VLLM_MODEL is not None and _VLLM_MODEL_NAME == target:
        return _VLLM_MODEL

    from funasr.auto.auto_model_vllm import AutoModelVLLM

    device = os.getenv("FUNASR_DEVICE") or os.getenv("DEVICE", "auto")
    if device == "auto":
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"

    _VLLM_MODEL = AutoModelVLLM(
        model=target,
        hub="ms",  # ModelScope
        device=device,
        dtype="bf16",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.8,
        max_model_len=4096,
    )
    _VLLM_MODEL_NAME = target
    return _VLLM_MODEL


def _to_ms(seconds: float) -> int:
    return int(round(float(seconds) * 1000))


def _convert_standard_result(result: dict, duration_ms: int) -> list[dict]:
    """Convert a standard FunASR AutoModel result to YouDub utterances."""
    utterances: list[dict] = []
    sentences = result.get("sentence_info") or result.get("sentences") or []
    if sentences:
        for sent in sentences:
            text = sent.get("text", "").strip()
            if not text:
                continue
            start_raw = sent.get("start", 0)
            end_raw = sent.get("end", 0)
            # FunASR may return ms or seconds depending on the model.
            if start_raw > 100000:
                start_ms = int(start_raw)
                end_ms = int(end_raw)
            else:
                start_ms = _to_ms(start_raw / 1000.0) if start_raw > 1000 else _to_ms(start_raw)
                end_ms = _to_ms(end_raw / 1000.0) if end_raw > 1000 else _to_ms(end_raw)
            utterances.append({
                "text": text,
                "start_time": start_ms,
                "end_time": end_ms,
                "words": [],
            })
    else:
        text = result.get("text", "").strip()
        if text:
            utterances.append({
                "text": text,
                "start_time": 0,
                "end_time": duration_ms,
                "words": [],
            })
    return utterances


def _group_chars_to_sentences(text: str, timestamps: list[dict]) -> list[dict]:
    """Group a per-character timestamp stream into sentence-level segments.

    The vLLM engine returns per-character timestamps via CTC forced
    alignment; FunASR does not produce sentence boundaries. We split
    on sentence-final punctuation and emit one segment per chunk,
    taking the first/last character timestamp as start/end. The
    asr_fix stage will refine these boundaries.

    ``timestamps`` is a list of dicts with at least ``token``,
    ``start_time``, ``end_time`` (in seconds). Characters in ``text``
    that cannot be aligned to a timestamp inherit the previous chunk's
    end time.
    """
    if not text or not timestamps:
        return []

    # Flatten each timestamp dict into a stream of (char, start, end)
    # tuples; one timestamp entry may contain several characters.
    char_stream: list[tuple[str, float, float]] = []
    for ts in timestamps:
        token = ts.get("token", "")
        start = float(ts.get("start_time", 0.0) or 0.0)
        end = float(ts.get("end_time", start) or start)
        for ch in token:
            char_stream.append((ch, start, end))

    sentences: list[dict] = []
    cur_chars: list[str] = []
    cur_start: float | None = None
    cur_end: float = 0.0
    si = 0  # index into char_stream

    def _emit() -> None:
        nonlocal cur_chars, cur_start, cur_end
        if not cur_chars:
            return
        chunk = "".join(cur_chars).strip()
        if chunk:
            sentences.append({
                "text": chunk,
                "start_time": _to_ms(cur_start or 0.0),
                "end_time": _to_ms(cur_end),
                "words": [],
            })
        cur_chars = []
        cur_start = None
        cur_end = 0.0

    for ch in text:
        if ch.isspace():
            continue
        if si < len(char_stream):
            _, c_start, c_end = char_stream[si]
            si += 1
        elif cur_chars:
            # Ran out of timestamps; reuse the last known end.
            c_start, c_end = (cur_start if cur_start is not None else 0.0), cur_end
        else:
            c_start, c_end = cur_end, cur_end
        if cur_start is None:
            cur_start = c_start
        cur_end = c_end
        cur_chars.append(ch)
        if _SENTENCE_PUNCTUATION.match(ch):
            _emit()
    _emit()  # trailing chunk without sentence punctuation
    return sentences


def _convert_vllm_result(result: dict, duration_ms: int) -> list[dict]:
    """Convert an :class:`AutoModelVLLM` result to YouDub utterances.

    The vLLM engine returns per-character timestamps, not sentence
    segments. We group characters by punctuation to form coarse
    segments; the asr_fix stage will refine sentence boundaries.
    """
    text = (result.get("text") or "").strip()
    if not text:
        return []
    timestamps = result.get("timestamps") or []
    sentences = _group_chars_to_sentences(text, timestamps)
    if sentences:
        return sentences
    # Fallback: no sentence grouping possible; one utterance covering
    # the whole audio with no internal timestamps.
    return [{
        "text": text,
        "start_time": 0,
        "end_time": duration_ms,
        "words": [],
    }]


def _finalize_asr_json(
    vocals_file: Path,
    session: Path,
    result: Any,
    convert_fn: Callable[[dict, int], list[dict]],
) -> Path:
    """Write the unified ``asr.json`` payload from a model result.

    ``convert_fn`` maps a single raw result dict to a list of YouDub
    utterance dicts; this lets the standard and vLLM paths share the
    on-disk format.
    """
    metadata_dir = session / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    output_file = metadata_dir / "asr.json"

    if not result:
        raise RuntimeError("FunASR did not return any results.")

    raw = result[0] if isinstance(result, list) else result
    duration_ms = len(AudioSegment.from_file(vocals_file))
    utterances = convert_fn(raw, duration_ms)
    if not utterances:
        raise RuntimeError("FunASR did not return any segments.")

    full_text = raw.get("text", "").strip()
    if not full_text:
        full_text = " ".join(u["text"] for u in utterances)

    payload = {
        "audio_info": {"duration": duration_ms},
        "result": {
            "text": full_text,
            "utterances": utterances,
        },
    }
    output_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_file


def _recognize_speech_standard(
    vocals_file: Path, session: Path, language: str, model_id: str | None
) -> Path:
    """Standard FunASR AutoModel path (SenseVoice, Paraformer, ...)."""
    model = _load_standard_model(model_id)
    funasr_lang = _LANG_TO_FUNASR.get(language, "auto")

    is_nano = "Fun-ASR-Nano" in (model_id or _get_model_id())

    generate_kwargs: dict = dict(
        input=str(vocals_file),
        cache={},
        language=funasr_lang,
        use_itn=True,
        merge_vad=True,
        merge_length_s=15,
    )
    if not is_nano:
        generate_kwargs["batch_size_s"] = 60

    result = model.generate(**generate_kwargs)
    return _finalize_asr_json(vocals_file, session, result, _convert_standard_result)


def _recognize_speech_vllm(
    vocals_file: Path, session: Path, language: str, model_id: str | None
) -> Path:
    """vLLM-backed FunASR path (Fun-ASR-Nano, GLM-ASR-Nano, LLMASR)."""
    model = _load_vllm_model(model_id)
    funasr_lang = _LANG_TO_FUNASR_VLLM.get(language)

    target = model_id or _get_model_id()
    # GLM-ASR-Nano does not support long segments; disable dynamic silence.
    dynamic_silence = "GLM-ASR-Nano" not in target

    result = model.generate(
        str(vocals_file),
        language=funasr_lang,
        itn=True,
        max_new_tokens=512,
        dynamic_silence=dynamic_silence,
    )
    return _finalize_asr_json(vocals_file, session, result, _convert_vllm_result)


def recognize_speech(
    vocals_file: Path, session: Path, language: str, *, model_id: str | None = None
) -> Path:
    """Recognize speech using the appropriate FunASR backend.

    Dispatches to the vLLM engine for LLM-based models
    (Fun-ASR-Nano, GLM-ASR-Nano, LLMASR) when vllm is installed and
    ``funasr.use_vllm`` is not ``off``; otherwise uses the standard
    ``funasr.AutoModel``.
    """
    metadata_dir = session / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    output_file = metadata_dir / "asr.json"
    if output_file.exists():
        return output_file

    target = model_id or _get_model_id()
    use_vllm, warning = _resolve_vllm_mode(target)
    if warning:
        logger.warning(warning)
    if use_vllm:
        return _recognize_speech_vllm(vocals_file, session, language, model_id)
    return _recognize_speech_standard(vocals_file, session, language, model_id)
