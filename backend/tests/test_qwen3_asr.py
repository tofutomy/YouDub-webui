"""Tests for backend.app.adapters.qwen3_asr — utterance grouping logic."""
from __future__ import annotations

import pytest

from backend.app.adapters.qwen3_asr import (
    _group_words_to_utterances,
)


def _w(text: str, start: int, end: int) -> dict:
    """Helper to create a word dict with ms timestamps."""
    return {"text": text, "start_time": start, "end_time": end}


# ---------------------------------------------------------------------------
# _group_words_to_utterances
# ---------------------------------------------------------------------------


class TestGroupWordsToUtterances:
    """Verify sentence grouping for various languages."""

    def test_empty_words(self) -> None:
        assert _group_words_to_utterances([], "Hello.") == []

    def test_single_sentence_no_punctuation(self) -> None:
        """No punctuation → single utterance."""
        words = [_w("hello", 0, 500), _w("world", 500, 1000)]
        result = _group_words_to_utterances(words, "hello world")
        assert len(result) == 1
        assert result[0]["text"] == "hello world"
        assert result[0]["start_time"] == 0
        assert result[0]["end_time"] == 1000
        assert len(result[0]["words"]) == 2

    def test_english_two_sentences(self) -> None:
        words = [_w("hello", 0, 500), _w("world", 500, 1000),
                 _w("goodbye", 1500, 2000)]
        result = _group_words_to_utterances(words, "hello world. goodbye")
        assert len(result) == 2
        assert result[0]["text"] == "hello world."
        assert result[0]["start_time"] == 0
        assert result[0]["end_time"] == 1000
        assert result[1]["text"] == "goodbye"
        assert result[1]["start_time"] == 1500

    def test_japanese_sentences(self) -> None:
        """Japanese text without spaces — the key fix target."""
        # こんにちは。君が応募して来てくれた人？
        words = [
            _w("こんにちは", 14720, 15200),
            _w("君", 18640, 18880),
            _w("が", 18880, 19200),
            _w("応募", 20320, 20640),
            _w("し", 20640, 20720),
            _w("て", 20720, 20800),
            _w("来", 20800, 20880),
            _w("て", 20880, 21040),
            _w("くれ", 21040, 21280),
            _w("た", 21280, 21440),
            _w("人", 21440, 21760),
        ]
        full_text = "こんにちは。君が応募して来てくれた人？"
        result = _group_words_to_utterances(words, full_text)

        assert len(result) == 2
        # First sentence: こんにちは。
        assert result[0]["text"] == "こんにちは。"
        assert result[0]["start_time"] == 14720
        assert result[0]["end_time"] == 15200
        assert len(result[0]["words"]) == 1
        # Second sentence: 君が応募して来てくれた人？
        assert result[1]["text"] == "君が応募して来てくれた人？"
        assert result[1]["start_time"] == 18640
        assert result[1]["end_time"] == 21760
        assert len(result[1]["words"]) == 10

    def test_chinese_sentences(self) -> None:
        words = [_w("你好", 0, 500), _w("世界", 500, 1000),
                 _w("再见", 1500, 2000)]
        result = _group_words_to_utterances(words, "你好世界。再见")
        assert len(result) == 2
        assert result[0]["text"] == "你好世界。"
        assert result[1]["text"] == "再见"

    def test_decimal_not_split(self) -> None:
        """Decimal dot (digit.digit) must NOT trigger a sentence split."""
        words = [_w("value", 0, 500), _w("is", 500, 700),
                 _w("4", 700, 800), _w("5", 800, 900),
                 _w("meters", 900, 1400)]
        result = _group_words_to_utterances(words, "value is 4.5 meters.")
        assert len(result) == 1
        assert result[0]["text"] == "value is 4.5 meters."

    def test_trailing_no_punctuation(self) -> None:
        """Trailing content without sentence-ending punctuation."""
        words = [_w("hello", 0, 500), _w("world", 500, 1000)]
        result = _group_words_to_utterances(words, "hello. world")
        assert len(result) == 2
        assert result[0]["text"] == "hello."
        assert result[1]["text"] == "world"

    def test_many_japanese_sentences(self) -> None:
        """Simulate a realistic multi-sentence Japanese ASR output."""
        words = [
            _w("チロ", 28480, 28880),
            _w("が", 28880, 29040),
            _w("自慢", 29040, 29440),
            _w("で", 29440, 29520),
            _w("応募", 29520, 29840),
            _w("し", 29840, 29920),
            _w("て", 29920, 30000),
            _w("来", 30000, 30080),
            _w("て", 30080, 30160),
            _w("くれ", 30160, 30400),
            _w("た", 30400, 30560),
            _w("ん", 30560, 30640),
            _w("でしょ", 30640, 30960),
        ]
        full_text = "チロが自慢で応募して来てくれたんでしょ？"
        result = _group_words_to_utterances(words, full_text)
        assert len(result) == 1
        assert result[0]["text"] == full_text
        assert result[0]["start_time"] == 28480
        assert result[0]["end_time"] == 30960
        assert len(result[0]["words"]) == 13
