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
_SENTENCE_PUNCTUATION = re.compile(r"([.!?;:\n\u3002\uff01\uff1f\uff1b\uff1a]+)")
_SOFT_SENTENCE_PUNCTUATION = re.compile(r"([,\u3001\uff0c])")
_SENSEVOICE_TAG = re.compile(r"<\|[^|>]+\|>")
_MAX_FALLBACK_CHARS = 120
_NO_SPACE_BEFORE = set(".,!?;:%)]}\u3002\uff0c\u3001\uff01\uff1f\uff1b\uff1a")
_NO_SPACE_AFTER = set("([{\u201c\u2018")
_APOSTROPHES = {"'", "\u2019"}

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


def _time_value_to_ms(value, default: int = 0) -> int:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if number > 100000:
        return int(round(number))
    return _to_ms(number / 1000.0) if number > 1000 else _to_ms(number)


def _clean_asr_text(text: str) -> str:
    cleaned = _SENSEVOICE_TAG.sub("", text or "")
    return re.sub(r"\s+", " ", cleaned).strip()


def _split_long_chunk(chunk: str) -> list[str]:
    if len(chunk) <= _MAX_FALLBACK_CHARS:
        return [chunk]
    parts: list[str] = []
    current: list[str] = []
    for char in chunk:
        current.append(char)
        if _SOFT_SENTENCE_PUNCTUATION.match(char) and len("".join(current).strip()) >= 40:
            parts.append("".join(current).strip())
            current = []
    trailing = "".join(current).strip()
    if trailing:
        parts.append(trailing)
    if len(parts) > 1 and all(len(part) <= _MAX_FALLBACK_CHARS * 1.5 for part in parts):
        return parts

    words = chunk.split()
    if len(words) <= 1:
        return [chunk]
    parts = []
    current_words: list[str] = []
    for word in words:
        tentative = " ".join([*current_words, word])
        if current_words and len(tentative) > _MAX_FALLBACK_CHARS:
            parts.append(" ".join(current_words))
            current_words = [word]
        else:
            current_words.append(word)
    if current_words:
        parts.append(" ".join(current_words))
    return parts


def _split_text_to_sentences(text: str) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    for char in _clean_asr_text(text):
        if char.isspace() and not current:
            continue
        current.append(char)
        if _SENTENCE_PUNCTUATION.match(char):
            chunk = "".join(current).strip()
            if chunk:
                chunks.extend(_split_long_chunk(chunk))
            current = []
    trailing = "".join(current).strip()
    if trailing:
        chunks.extend(_split_long_chunk(trailing))
    return chunks


def _fallback_sentences_by_duration(text: str, duration_ms: int) -> list[dict]:
    chunks = _split_text_to_sentences(text)
    if not chunks:
        return []
    weights = [max(1, len(chunk.strip())) for chunk in chunks]
    cursor = 0
    utterances: list[dict] = []
    for index, (chunk, weight) in enumerate(zip(chunks, weights)):
        elapsed_weight = sum(weights[: index + 1])
        end_ms = duration_ms if index == len(chunks) - 1 else int(round(duration_ms * elapsed_weight / sum(weights)))
        if end_ms <= cursor:
            end_ms = min(duration_ms, cursor + 1)
        utterances.append({
            "text": chunk,
            "start_time": cursor,
            "end_time": end_ms,
            "words": [],
        })
        cursor = end_ms
    return utterances


def _timestamp_pair_to_ms(ts) -> tuple[int, int] | None:
    if isinstance(ts, dict):
        start = ts.get("start_time", ts.get("start"))
        end = ts.get("end_time", ts.get("end"))
        return _time_value_to_ms(start), _time_value_to_ms(end)
    if isinstance(ts, (list, tuple)) and len(ts) >= 2:
        return int(round(float(ts[0]))), int(round(float(ts[1])))
    return None


def _is_cjk_char(char: str) -> bool:
    if not char:
        return False
    code = ord(char)
    return (
        0x3400 <= code <= 0x9FFF
        or 0xF900 <= code <= 0xFAFF
        or 0x3040 <= code <= 0x30FF
        or 0xAC00 <= code <= 0xD7AF
    )


def _tokens_to_text(tokens: list[str]) -> str:
    text = ""
    previous_token = ""
    for raw_token in tokens:
        token = str(raw_token or "").strip()
        if not token:
            continue
        if not text:
            text = token
            previous_token = token
            continue
        previous = text[-1]
        first = token[0]
        if (
            token in _APOSTROPHES
            or previous in _APOSTROPHES
            or (len(previous_token) == 1 and previous_token.isalpha() and token.isdigit())
            or first in _NO_SPACE_BEFORE
            or previous in _NO_SPACE_AFTER
            or (_is_cjk_char(previous) and _is_cjk_char(first))
        ):
            text += token
        else:
            text += " " + token
        previous_token = token
    return text.strip()


def _word_items_to_utterance(items: list[dict]) -> dict | None:
    if not items:
        return None
    text = _tokens_to_text([item["text"] for item in items])
    if not text:
        return None
    return {
        "text": text,
        "start_time": items[0]["start_time"],
        "end_time": items[-1]["end_time"],
        "words": items,
    }


def _word_list_timestamps_to_sentences(words: list | None, timestamps: list) -> list[dict]:
    if not isinstance(words, list) or not words or not timestamps:
        return []
    if len(timestamps) < max(1, int(len(words) * 0.8)):
        return []

    utterances: list[dict] = []
    current: list[dict] = []

    def emit() -> None:
        nonlocal current
        utterance = _word_items_to_utterance(current)
        if utterance:
            utterances.append(utterance)
        current = []

    for raw_word, ts in zip(words, timestamps):
        token = str(raw_word or "").strip()
        if not token:
            continue
        pair = _timestamp_pair_to_ms(ts)
        if pair is None:
            return []
        start_ms, end_ms = pair
        if end_ms <= start_ms:
            return []
        current.append({"text": token, "start_time": start_ms, "end_time": end_ms})
        current_text = _tokens_to_text([item["text"] for item in current])
        if (
            _SENTENCE_PUNCTUATION.fullmatch(token)
            or _SENTENCE_PUNCTUATION.search(token)
            or _SOFT_SENTENCE_PUNCTUATION.fullmatch(token)
            or _SOFT_SENTENCE_PUNCTUATION.search(token)
        ):
            emit()

    emit()
    return utterances


def _word_timestamps_to_sentences(text: str, timestamps: list) -> list[dict]:
    words = [match.group(0) for match in re.finditer(r"\S+", text)]
    if not words or not timestamps:
        return []
    # SenseVoice/standard FunASR often returns word-level timestamps for
    # whitespace-delimited languages. Treat the timestamps as word-level when
    # they are close to the word count; otherwise let the char-level fallback try.
    if abs(len(timestamps) - len(words)) > max(2, int(len(words) * 0.2)):
        return []

    utterances: list[dict] = []
    cur_words: list[dict] = []
    cur_text: list[str] = []
    for word, ts in zip(words, timestamps):
        pair = _timestamp_pair_to_ms(ts)
        if pair is None:
            return []
        start_ms, end_ms = pair
        if end_ms <= start_ms:
            return []
        word_item = {"text": word, "start_time": start_ms, "end_time": end_ms}
        cur_words.append(word_item)
        cur_text.append(word)
        if _SENTENCE_PUNCTUATION.search(word):
            utterances.append({
                "text": " ".join(cur_text).strip(),
                "start_time": cur_words[0]["start_time"],
                "end_time": cur_words[-1]["end_time"],
                "words": cur_words,
            })
            cur_words = []
            cur_text = []

    if cur_words:
        utterances.append({
            "text": " ".join(cur_text).strip(),
            "start_time": cur_words[0]["start_time"],
            "end_time": cur_words[-1]["end_time"],
            "words": cur_words,
        })
    return utterances


def _timestamp_pairs_to_sentences(text: str, timestamps: list) -> list[dict]:
    text = _clean_asr_text(text)
    if not text or not timestamps:
        return []
    word_level = _word_timestamps_to_sentences(text, timestamps)
    if word_level:
        return word_level

    normalized: list[dict] = []
    chars = (char for char in text if not char.isspace())
    for token, ts in zip(chars, timestamps):
        if isinstance(ts, dict):
            start = ts.get("start_time", ts.get("start"))
            end = ts.get("end_time", ts.get("end"))
            normalized.append({
                "token": ts.get("token", token),
                "start_time": _time_value_to_ms(start) / 1000.0,
                "end_time": _time_value_to_ms(end) / 1000.0,
            })
        elif isinstance(ts, (list, tuple)) and len(ts) >= 2:
            normalized.append({
                "token": token,
                "start_time": float(ts[0]) / 1000.0,
                "end_time": float(ts[1]) / 1000.0,
            })
    nonspace_count = sum(1 for char in text if not char.isspace())
    if len(normalized) < max(1, int(nonspace_count * 0.8)):
        return []
    return _group_chars_to_sentences(text, normalized)


def _timestamps_are_usable(utterances: list[dict], duration_ms: int) -> bool:
    if not utterances:
        return False
    previous_end = -1
    for utterance in utterances:
        start = int(utterance.get("start_time", 0))
        end = int(utterance.get("end_time", 0))
        if end <= start:
            return False
        if start < previous_end:
            return False
        previous_end = end
    return True


def _convert_standard_result(result: dict, duration_ms: int) -> list[dict]:
    """Convert a standard FunASR AutoModel result to YouDub utterances."""
    utterances: list[dict] = []
    sentences = result.get("sentence_info") or result.get("sentences") or []
    if sentences:
        for sent in sentences:
            text = _clean_asr_text(sent.get("text") or sent.get("sentence") or "")
            if not text:
                continue
            start_ms = _time_value_to_ms(sent.get("start", sent.get("start_time")), 0)
            end_ms = _time_value_to_ms(sent.get("end", sent.get("end_time")), duration_ms)
            if end_ms <= start_ms:
                end_ms = min(duration_ms, start_ms + 1)
            utterances.append({
                "text": text,
                "start_time": start_ms,
                "end_time": end_ms,
                "words": [],
            })
    else:
        text = _clean_asr_text(result.get("text", ""))
        if text:
            timestamps = result.get("timestamp") or result.get("timestamps") or []
            timestamped = _word_list_timestamps_to_sentences(result.get("words"), timestamps)
            if not timestamped:
                timestamped = _timestamp_pairs_to_sentences(text, timestamps)
            if _timestamps_are_usable(timestamped, duration_ms):
                utterances.extend(timestamped)
            else:
                utterances.extend(_fallback_sentences_by_duration(text, duration_ms))
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
            if cur_chars and not cur_chars[-1].isspace():
                cur_chars.append(" ")
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
    raw_output_file = metadata_dir / "asr.raw.funasr.json"
    raw_output_file.write_text(
        json.dumps(raw, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

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
        sentence_timestamp=True,
        output_timestamp=True,
        return_time_stamps=True,
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
