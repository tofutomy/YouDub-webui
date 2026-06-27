from __future__ import annotations

import pytest

from backend.app.languages import (
    DEFAULT_ASR_LANGUAGE,
    DEFAULT_TARGET_LANGUAGE,
    LanguageConfigError,
    direction_to_language_pair,
    language_pair_to_direction,
    normalize_language_pair,
)


def test_empty_language_config_defaults_to_en_zh() -> None:
    assert normalize_language_pair() == (DEFAULT_ASR_LANGUAGE, DEFAULT_TARGET_LANGUAGE)
    assert normalize_language_pair("", "   ") == ("en", "zh")


@pytest.mark.parametrize(
    ("asr_language", "expected"),
    [
        ("en", ("en", "zh")),
        ("zh", ("zh", "en")),
        ("ja", ("ja", "zh")),
    ],
)
def test_asr_language_only_is_completed_from_supported_directions(asr_language, expected) -> None:
    assert normalize_language_pair(asr_language=asr_language) == expected


@pytest.mark.parametrize(
    ("target_language", "expected"),
    [
        ("zh", ("en", "zh")),
        ("en", ("zh", "en")),
    ],
)
def test_target_language_only_is_completed_from_supported_directions(target_language, expected) -> None:
    assert normalize_language_pair(target_language=target_language) == expected


def test_direction_and_language_pair_conversion() -> None:
    assert direction_to_language_pair("ja-zh") == ("ja", "zh")
    assert language_pair_to_direction("zh", "en") == "zh-en"


@pytest.mark.parametrize(
    ("asr_language", "target_language"),
    [
        ("ja", "en"),
        ("zh", "zh"),
        ("fr", "zh"),
    ],
)
def test_unsupported_language_pair_raises(asr_language, target_language) -> None:
    with pytest.raises(LanguageConfigError):
        normalize_language_pair(asr_language, target_language)
