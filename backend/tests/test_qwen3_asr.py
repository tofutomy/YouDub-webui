"""Tests for backend.app.adapters.qwen3_asr — sentence splitting logic."""
from __future__ import annotations

import pytest

from backend.app.adapters.qwen3_asr import (
    _split_text_into_sentences,
    _tokenize,
)


# ---------------------------------------------------------------------------
# _split_text_into_sentences
# ---------------------------------------------------------------------------


class TestSplitTextIntoSentences:
    """Verify sentence-boundary detection around decimal numbers."""

    def test_decimal_not_split(self) -> None:
        """4.5 is a decimal and must NOT trigger a sentence split."""
        result = _split_text_into_sentences("The value is 4.5 meters.")
        assert result == ["The value is 4.5 meters."]

    def test_pi_not_split(self) -> None:
        result = _split_text_into_sentences("Pi equals 3.14 approximately.")
        assert result == ["Pi equals 3.14 approximately."]

    def test_version_number_not_split(self) -> None:
        result = _split_text_into_sentences("Upgrade to v2.0 now.")
        assert result == ["Upgrade to v2.0 now."]

    def test_normal_period_splits(self) -> None:
        """A regular sentence-ending period must still trigger a split."""
        result = _split_text_into_sentences("Hello world. Goodbye.")
        assert result == ["Hello world.", "Goodbye."]

    def test_chinese_punctuation_splits(self) -> None:
        result = _split_text_into_sentences("你好世界。再见！")
        assert result == ["你好世界。", "再见！"]

    def test_multiple_decimals_in_text(self) -> None:
        result = _split_text_into_sentences("Speed is 3.5m/s. Rate is 1.2x.")
        assert result == ["Speed is 3.5m/s.", "Rate is 1.2x."]

    def test_decimal_at_end_no_trailing_period(self) -> None:
        result = _split_text_into_sentences("Version 2.1")
        assert result == ["Version 2.1"]

    def test_trailing_period_after_decimal(self) -> None:
        """'4.5.' — the trailing period is a real sentence boundary."""
        result = _split_text_into_sentences("It is 4.5. Next sentence.")
        assert result == ["It is 4.5.", "Next sentence."]

    def test_empty_text(self) -> None:
        assert _split_text_into_sentences("") == []

    def test_no_punctuation(self) -> None:
        result = _split_text_into_sentences("no punctuation here")
        assert result == ["no punctuation here"]


# ---------------------------------------------------------------------------
# _tokenize
# ---------------------------------------------------------------------------


class TestTokenize:
    def test_basic(self) -> None:
        assert _tokenize("hello world") == ["hello", "world"]

    def test_empty(self) -> None:
        assert _tokenize("") == []

    def test_whitespace_only(self) -> None:
        assert _tokenize("   ") == []
