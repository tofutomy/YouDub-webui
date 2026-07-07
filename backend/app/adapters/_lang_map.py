"""Shared language code → name / backend-alias mappings.

Centralises the language maps that were previously duplicated across
``funasr_asr``, ``remote_funasr_asr``, ``qwen3_asr`` and ``sources``.
Each adapter can look up the mapping it needs without redefining it.
"""

from __future__ import annotations


# Human-readable language names (used by prompts and UI).
LANG_NAMES: dict[str, str] = {
    "en": "English",
    "zh": "Simplified Chinese",
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

# FunASR standard AutoModel language codes.
LANG_TO_FUNASR: dict[str, str] = {
    "zh": "zh",
    "en": "en",
    "ja": "ja",
    "ko": "ko",
    "yue": "yue",
}

# FunASR vLLM engine language labels (Chinese names expected by the engine).
LANG_TO_FUNASR_VLLM: dict[str, str] = {
    "zh": "中文",
    "en": "英文",
    "ja": "日文",
    "ko": "韩文",
    "yue": "粤语",
}

# Remote FunASR service language labels.
LANG_TO_REMOTE_FUNASR: dict[str, str] = {
    "zh": "中文",
    "en": "英文",
    "ja": "日本語",
    "ko": "한국어",
    "yue": "粤语",
}

# OpenAI-compatible language codes (used by the remote transcription endpoint).
LANG_TO_OPENAI: dict[str, str] = {
    "zh": "zh",
    "en": "en",
    "ja": "ja",
    "ko": "ko",
    "yue": "yue",
}

# Qwen3-ASR language labels (English names expected by the model).
LANG_TO_QWEN3: dict[str, str] = {
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


# ARK-ASR-3B language labels (English names expected by the model).
# ARK-ASR supports 19 languages; we map only the ones relevant to YouDub.
LANG_TO_ARK_ASR: dict[str, str] = {
    "zh": "Chinese",
    "en": "English",
    "ja": "Japanese",
    "ko": "Korean",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "pt": "Portuguese",
    "it": "Italian",
    "nl": "Dutch",
    "pl": "Polish",
}

# MOSS-Audio-4B language labels (same as Qwen3 — based on Qwen3-4B backbone).
LANG_TO_MOSS_AUDIO: dict[str, str] = LANG_TO_QWEN3.copy()


def language_name(code: str | None, default: str = "") -> str:
    """Return the human-readable name for a language code."""
    if not code:
        return default
    return LANG_NAMES.get(code, default or code)
