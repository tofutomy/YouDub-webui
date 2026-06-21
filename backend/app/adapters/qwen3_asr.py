"""Qwen3-ASR speech recognition adapter (transformers backend).

Uses the ``qwen-asr`` package's transformers backend for local inference on
Windows/Linux without requiring vLLM.  Timestamps are produced by the
companion ``Qwen3-ForcedAligner-0.6B`` model.

Install: ``pip install -U qwen-asr``
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

from pydub import AudioSegment

logger = logging.getLogger(__name__)

_MODEL = None
_ALIGNER = None

# Sentence-level punctuation used to split word timestamps into utterances.
_SENTENCE_END = re.compile(r"[.!?;:\u3002\uff01\uff1f\uff1b\uff1a]+$")
_CLAUSE_END = re.compile(r"[,\u3001\uff0c\u2014\u2013]+$")

# Language code → human-readable name for Qwen3-ASR.
_LANG_MAP = {
    "zh": "Chinese",
    "en": "English",
    "ja": "Japanese",
    "ko": "Korean",
    "yue": "Cantonese",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "pt": "Portuguese",
    "ru": "Russian",
    "ar": "Arabic",
    "it": "Italian",
    "th": "Thai",
    "vi": "Vietnamese",
    "id": "Indonesian",
    "tr": "Turkish",
    "hi": "Hindi",
    "ms": "Malay",
    "nl": "Dutch",
}


def _model_dir() -> str | None:
    """Return the local model directory if it exists, else ``None``."""
    cache = os.getenv("MODEL_CACHE_DIR", "").strip()
    if cache:
        candidate = Path(cache) / "models" / "Qwen" / "Qwen3-ASR-1.7B"
        if candidate.is_dir():
            return str(candidate)
    # Fallback: let qwen-asr download automatically.
    return None


def _aligner_dir() -> str | None:
    """Return the local ForcedAligner directory if it exists."""
    cache = os.getenv("MODEL_CACHE_DIR", "").strip()
    if cache:
        candidate = Path(cache) / "models" / "Qwen" / "Qwen3-ForcedAligner-0.6B"
        if candidate.is_dir():
            return str(candidate)
    return None


def _resolve_device() -> str:
    device = os.getenv("DEVICE", "cuda").strip()
    if device == "auto":
        import torch
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        return "cuda:0"
    return device


def _load_model():
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    import torch
    from qwen_asr import Qwen3ASRModel

    model_path = _model_dir() or "Qwen/Qwen3-ASR-1.7B"
    device = _resolve_device()
    aligner_path = _aligner_dir() or "Qwen/Qwen3-ForcedAligner-0.6B"

    logger.info("Loading Qwen3-ASR from %s on %s", model_path, device)
    _MODEL = Qwen3ASRModel.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map=device,
        max_inference_batch_size=8,
        max_new_tokens=2048,
        forced_aligner=aligner_path,
        forced_aligner_kwargs=dict(
            dtype=torch.bfloat16,
            device_map=device,
        ),
    )
    return _MODEL


def _to_ms(seconds: float) -> int:
    return int(round(seconds * 1000))


def _split_text_into_sentences(text: str) -> list[str]:
    """Split text into sentences using sentence-ending punctuation.

    Returns a list of non-empty sentence strings (punctuation preserved).
    """
    # Split after sentence-ending punctuation, keeping the delimiter.
    # Skip '.' that is a decimal point (digit.digit) to avoid splitting
    # numbers like "4.5" or "3.14" into two sentences.
    parts: list[str] = []
    current: list[str] = []
    for i, ch in enumerate(text):
        current.append(ch)
        if ch in ".!?\u3002\uff01\uff1f":
            if ch == "." and i > 0 and i + 1 < len(text):
                if text[i - 1].isdigit() and text[i + 1].isdigit():
                    continue
            parts.append("".join(current).strip())
            current = []
    trailing = "".join(current).strip()
    if trailing:
        parts.append(trailing)
    return [s for s in parts if s]


def _tokenize(text: str) -> list[str]:
    """Extract whitespace-separated tokens from text."""
    return [t for t in text.split() if t]


def _group_words_to_utterances(
    words: list[dict[str, Any]],
    full_text: str,
) -> list[dict[str, Any]]:
    """Group word-level timestamp items into sentence-level utterances.

    The ForcedAligner returns word timestamps *without* punctuation tokens.
    We use the punctuated ``full_text`` to determine sentence boundaries,
    then sequentially match each sentence's words against the timestamp list.
    """
    if not words:
        return []

    sentences = _split_text_into_sentences(full_text)
    if len(sentences) <= 1:
        # Only one sentence (or no punctuation) — return as-is.
        text = " ".join(w["text"] for w in words).strip()
        return [{
            "text": text or full_text,
            "start_time": words[0]["start_time"],
            "end_time": words[-1]["end_time"],
            "words": list(words),
        }]

    utterances: list[dict[str, Any]] = []
    wi = 0  # current index into words list

    for sentence in sentences:
        sentence_tokens = _tokenize(sentence)
        if not sentence_tokens:
            continue

        # How many word-timestamp entries belong to this sentence.
        # We greedily consume tokens from the word list that appear in
        # the sentence text (ignoring punctuation differences).
        matched: list[dict[str, Any]] = []
        needed = len(sentence_tokens)
        consumed = 0

        while wi < len(words) and consumed < needed:
            matched.append(words[wi])
            wi += 1
            consumed += 1

        # Consume trailing punctuation-only tokens if any (unlikely
        # with ForcedAligner, but safe).
        while wi < len(words):
            nxt = words[wi]["text"]
            if nxt in ".!?,;:\u3002\uff0c\u3001\uff01\uff1f\uff1b\uff1a":
                matched.append(words[wi])
                wi += 1
            else:
                break

        if matched:
            utterances.append({
                "text": sentence,
                "start_time": matched[0]["start_time"],
                "end_time": matched[-1]["end_time"],
                "words": matched,
            })

    # Any remaining words go into a trailing utterance.
    if wi < len(words):
        remaining = words[wi:]
        text = " ".join(w["text"] for w in remaining).strip()
        # Try to find the corresponding trailing text.
        remaining_sentences = full_text
        for utt in utterances:
            remaining_sentences = remaining_sentences.replace(utt["text"], "", 1)
        remaining_text = remaining_sentences.strip()
        utterances.append({
            "text": remaining_text or text,
            "start_time": remaining[0]["start_time"],
            "end_time": remaining[-1]["end_time"],
            "words": remaining,
        })

    return utterances


def _convert_result(result: Any, duration_ms: int) -> dict[str, Any]:
    """Convert a Qwen3-ASR transcribe result to YouDub asr.json payload."""
    text = getattr(result, "text", "") or ""
    language = getattr(result, "language", "") or ""
    time_stamps = getattr(result, "time_stamps", None) or []

    # Convert ForcedAligner timestamps to word items.
    words: list[dict[str, Any]] = []
    for ts in time_stamps:
        # Each ts has .text, .start_time, .end_time (seconds, float).
        token = str(getattr(ts, "text", "")).strip()
        if not token:
            continue
        start_s = float(getattr(ts, "start_time", 0.0) or 0.0)
        end_s = float(getattr(ts, "end_time", start_s) or start_s)
        words.append({
            "text": token,
            "start_time": _to_ms(start_s),
            "end_time": _to_ms(end_s),
        })

    utterances = _group_words_to_utterances(words, text)

    # Fallback: no timestamps at all.
    if not utterances:
        utterances = [{
            "text": text,
            "start_time": 0,
            "end_time": duration_ms,
            "words": [],
        }]

    return {
        "audio_info": {"duration": duration_ms},
        "result": {
            "text": text,
            "utterances": utterances,
        },
    }


def recognize_speech(
    vocals_file: Path, session: Path, language: str, *, model_id: str | None = None
) -> Path:
    """Recognize speech using Qwen3-ASR with transformers backend."""
    metadata_dir = session / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    output_file = metadata_dir / "asr.json"
    if output_file.exists():
        return output_file

    model = _load_model()
    qwen_lang = _LANG_MAP.get(language)  # None → auto-detect

    logger.info("Qwen3-ASR transcribing %s (language=%s)", vocals_file.name, qwen_lang)
    results = model.transcribe(
        audio=str(vocals_file),
        language=qwen_lang,
        return_time_stamps=True,
    )

    if not results:
        raise RuntimeError("Qwen3-ASR did not return any results.")

    result = results[0]

    # Save raw result for debugging.
    raw_output = metadata_dir / "asr.raw.qwen3asr.json"
    raw_data = {
        "text": getattr(result, "text", ""),
        "language": getattr(result, "language", ""),
        "time_stamps": [
            {
                "text": str(getattr(ts, "text", "")),
                "start_time": float(getattr(ts, "start_time", 0.0) or 0.0),
                "end_time": float(getattr(ts, "end_time", 0.0) or 0.0),
            }
            for ts in (getattr(result, "time_stamps", None) or [])
        ],
    }
    raw_output.write_text(
        json.dumps(raw_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    duration_ms = len(AudioSegment.from_file(vocals_file))
    payload = _convert_result(result, duration_ms)

    output_file.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_file
