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
from pathlib import Path
from typing import Any

import gc

from pydub import AudioSegment

from ._time_utils import to_ms as _to_ms
from ._lang_map import LANG_TO_QWEN3 as _LANG_MAP

logger = logging.getLogger(__name__)

_MODEL = None
_ALIGNER = None


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


def release_model() -> None:
    """Release Qwen3-ASR and ForcedAligner models from GPU memory.

    Call this after ``recognize_speech`` completes to free VRAM when the
    model is no longer needed.  Subsequent calls to ``recognize_speech``
    will reload the models automatically.
    """
    global _MODEL, _ALIGNER
    if _MODEL is None and _ALIGNER is None:
        return
    _MODEL = None
    _ALIGNER = None
    # Multiple GC passes are needed because PyTorch models often contain
    # reference cycles between modules, tensors, and autograd graphs.
    for _ in range(3):
        gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass
    logger.info("Qwen3-ASR models released from GPU memory")


def _group_words_to_utterances(
    words: list[dict[str, Any]],
    full_text: str,
) -> list[dict[str, Any]]:
    """Group word-level timestamp items into sentence-level utterances.

    The ForcedAligner returns word timestamps *without* punctuation tokens.
    We first strip all punctuation from ``full_text`` to create a *cleaned*
    text, then match each word token sequentially against the cleaned text to
    determine each word's character position.  Sentence boundaries are derived
    from sentence-ending punctuation in the original ``full_text`` (with
    decimal-point protection for numbers like ``4.5``).

    This approach works correctly for CJK languages (Japanese, Chinese, etc.)
    where words are NOT separated by whitespace, as well as for space-delimited
    languages like English.
    """
    if not words:
        return []

    _ALL_PUNCT = set(".!?,;:\u3002\uff0c\u3001\uff01\uff1f\uff1b\uff1a")
    _SENTENCE_PUNCT = set(".!?\u3002\uff01\uff1f\uff1b\uff1a")

    # ------------------------------------------------------------------
    # Step 1: Build cleaned text (no punctuation) and record the mapping
    #         from cleaned-text positions back to original-text positions.
    # ------------------------------------------------------------------
    cleaned_chars: list[str] = []
    clean_to_orig: list[int] = []  # clean_pos → orig_pos
    for oi, ch in enumerate(full_text):
        if ch not in _ALL_PUNCT:
            cleaned_chars.append(ch)
            clean_to_orig.append(oi)
    cleaned = "".join(cleaned_chars)

    # ------------------------------------------------------------------
    # Step 2: Collect sentence-ending punctuation positions in the
    #         original text (skip decimal dots: digit.digit).
    # ------------------------------------------------------------------
    sentence_punct_positions: list[int] = []
    for oi, ch in enumerate(full_text):
        if ch in _SENTENCE_PUNCT:
            if ch == "." and oi > 0 and oi + 1 < len(full_text):
                if full_text[oi - 1].isdigit() and full_text[oi + 1].isdigit():
                    continue
            sentence_punct_positions.append(oi)

    # ------------------------------------------------------------------
    # Step 3: Match each word against the cleaned text sequentially,
    #         recording its position in the original text.
    # ------------------------------------------------------------------
    word_positions: list[tuple[dict[str, Any], int]] = []  # (word, orig_pos)
    ci = 0  # current position in cleaned text
    for word in words:
        token = word["text"]
        if token in _ALL_PUNCT:
            continue  # skip punctuation tokens from the aligner

        matched = False
        while ci < len(cleaned):
            remaining_clean = cleaned[ci:]
            if remaining_clean.startswith(token):
                orig_pos = clean_to_orig[ci]
                word_positions.append((word, orig_pos))
                ci += len(token)
                matched = True
                break
            # Fallback: token might be a sub-word fragment that doesn't
            # match the cleaned text exactly (e.g., Japanese morphemes
            # split differently by the aligner).  Skip one character
            # and try again.
            ci += 1

        if not matched:
            # Exhausted cleaned text — append word at the last known
            # position so it is not lost.
            last_pos = word_positions[-1][1] if word_positions else 0
            word_positions.append((word, last_pos))

    if not word_positions:
        return []

    # ------------------------------------------------------------------
    # Step 4: Assign each word to a sentence using the punctuation
    #         positions as sentence boundaries.
    # ------------------------------------------------------------------
    utterances: list[dict[str, Any]] = []
    current_words: list[dict[str, Any]] = []
    current_first_orig: int = 0  # original-text position of first word in sentence
    sentence_start_ci = 0  # cleaned-text offset of current sentence start
    punct_idx = 0  # index into sentence_punct_positions

    for word, orig_pos in word_positions:
        if not current_words:
            current_first_orig = orig_pos
        current_words.append(word)

        # Determine the cleaned-text position of this word to check if a
        # sentence boundary falls between this word and the next.
        # Find the cleaned-text offset that corresponds to orig_pos.
        # We search forward in clean_to_orig from sentence_start_ci.
        word_clean_end = 0
        for c in range(sentence_start_ci, len(clean_to_orig)):
            if clean_to_orig[c] >= orig_pos:
                # This cleaned position is at or past the word's original
                # position.  The word occupies [orig_pos, orig_pos+len(token)).
                word_clean_end = c + len(word["text"])
                break

        # Check if there's a sentence-ending punctuation in the original
        # text between the end of this word and the start of the next word.
        # The "next word start" in original text is the next clean_to_orig
        # entry after word_clean_end.
        next_orig_start = len(full_text)  # default: end of text
        if word_clean_end < len(clean_to_orig):
            next_orig_start = clean_to_orig[word_clean_end]

        # Any sentence-ending punctuation between orig_pos and
        # next_orig_start marks a sentence boundary.
        has_boundary = False
        boundary_orig = -1
        while punct_idx < len(sentence_punct_positions):
            pp = sentence_punct_positions[punct_idx]
            if pp >= next_orig_start:
                break  # punctuation is after the next word — not yet
            if pp >= orig_pos:
                has_boundary = True
                boundary_orig = pp
                punct_idx += 1
                break
            punct_idx += 1  # skip punctuation before this word

        if has_boundary:
            # Flush: sentence text spans from first word's original
            # position to the punctuation character (inclusive).
            sentence_text = full_text[current_first_orig:boundary_orig + 1].strip()
            if sentence_text:
                utterances.append({
                    "text": sentence_text,
                    "start_time": current_words[0]["start_time"],
                    "end_time": current_words[-1]["end_time"],
                    "words": list(current_words),
                })
            current_words = []
            sentence_start_ci = word_clean_end

    # Trailing words without sentence-ending punctuation.
    if current_words:
        sentence_text = full_text[current_first_orig:].strip()
        if sentence_text:
            utterances.append({
                "text": sentence_text,
                "start_time": current_words[0]["start_time"],
                "end_time": current_words[-1]["end_time"],
                "words": list(current_words),
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
    del results  # release batch reference early

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
    del result  # release ASR result (holds aligner output) early

    output_file.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_file
