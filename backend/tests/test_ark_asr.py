"""Tests for backend.app.adapters.ark_asr — VAD segmentation and text assembly."""
from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import json
import pytest
import numpy as np

from backend.app.adapters.ark_asr import (
    _build_bad_words_ids,
    _distribute_utterances,
    _extract_segment,
    _split_text_to_sentences,
    _transcribe_chunk,
    _vad_segments,
)


# ---------------------------------------------------------------------------
# _vad_segments
# ---------------------------------------------------------------------------


class TestVadSegments:
    """Verify VAD segment parsing and splitting."""

    @patch("backend.app.adapters.ark_asr._load_vad_model")
    def test_empty_vad_result(self, mock_load: MagicMock) -> None:
        mock_vad = MagicMock()
        mock_vad.generate.return_value = []
        mock_load.return_value = mock_vad
        assert _vad_segments("test.wav") == []

    @patch("backend.app.adapters.ark_asr._load_vad_model")
    def test_basic_segments(self, mock_load: MagicMock) -> None:
        mock_vad = MagicMock()
        mock_vad.generate.return_value = [
            {"timestamp": [[0, 5000], [6000, 12000]]}
        ]
        mock_load.return_value = mock_vad
        result = _vad_segments("test.wav")
        assert result == [(0, 5000), (6000, 12000)]

    @patch("backend.app.adapters.ark_asr._load_vad_model")
    def test_split_long_segment(self, mock_load: MagicMock) -> None:
        """Segments > 30s should be split into ≤30s chunks."""
        mock_vad = MagicMock()
        mock_vad.generate.return_value = [
            {"timestamp": [[0, 70000]]}
        ]
        mock_load.return_value = mock_vad
        result = _vad_segments("test.wav")
        assert len(result) == 3
        assert result[0] == (0, 30000)
        assert result[1] == (30000, 60000)
        assert result[2] == (60000, 70000)

    @patch("backend.app.adapters.ark_asr._load_vad_model")
    def test_drop_short_segments(self, mock_load: MagicMock) -> None:
        """Segments < 200ms should be dropped as noise."""
        mock_vad = MagicMock()
        mock_vad.generate.return_value = [
            {"timestamp": [[0, 5000], [5100, 5200], [6000, 10000]]}
        ]
        mock_load.return_value = mock_vad
        result = _vad_segments("test.wav")
        assert result == [(0, 5000), (6000, 10000)]

    @patch("backend.app.adapters.ark_asr._load_vad_model")
    def test_invalid_timestamp_pairs(self, mock_load: MagicMock) -> None:
        """Malformed pairs should be skipped."""
        mock_vad = MagicMock()
        mock_vad.generate.return_value = [
            {"timestamp": [[5000, 1000], [6000, 12000], None]}
        ]
        mock_load.return_value = mock_vad
        result = _vad_segments("test.wav")
        assert result == [(6000, 12000)]


# ---------------------------------------------------------------------------
# _build_bad_words_ids
# ---------------------------------------------------------------------------


class TestBuildBadWordsIds:
    """Verify bad_words_ids construction from tokenizer special tokens."""

    def test_filters_special_tokens(self) -> None:
        tokenizer = MagicMock()
        tokenizer.eos_token_id = 2
        tokenizer.all_special_ids = [0, 1, 2, 3, 4]
        tokenizer.get_added_vocab.return_value = {
            "<|user|>": 10,
            "<|audio|>": 11,
        }
        result = _build_bad_words_ids(tokenizer)
        # EOS (2) should NOT be in bad_words_ids
        bad_ids = {ids[0] for ids in result}
        assert 2 not in bad_ids
        # Other special tokens should be filtered
        assert 0 in bad_ids
        assert 1 in bad_ids
        # Added vocab tokens should be filtered
        assert 10 in bad_ids
        assert 11 in bad_ids


# ---------------------------------------------------------------------------
# _transcribe_chunk
# ---------------------------------------------------------------------------


class TestTranscribeChunk:
    """Verify single-chunk transcription logic."""

    def test_returns_cleaned_text(self) -> None:
        model = MagicMock()
        processor = MagicMock()
        tokenizer = MagicMock()
        device = "cpu"
        dtype = None

        # Mock processor.apply_chat_template to return a mock dict
        mock_inputs = MagicMock()
        mock_inputs.to.return_value = mock_inputs
        mock_inputs.__contains__ = lambda self, k: k == "audios"
        mock_inputs.__getitem__ = lambda self, k: {"audios": MagicMock()}
        mock_inputs.__iter__ = lambda self: iter(["input_ids", "audios"])
        processor.apply_chat_template.return_value = mock_inputs

        # Mock tokenizer
        tokenizer.pad_token_id = 0
        tokenizer.eos_token_id = 2
        mock_output = MagicMock()
        mock_output.__getitem__ = lambda self, k: [100, 101]
        model.generate.return_value = mock_output
        mock_inputs.__getitem__ = lambda self, k: {"input_ids": MagicMock(shape=[0, 10]), "audios": MagicMock()}[k] if k in ("input_ids", "audios") else MagicMock()
        tokenizer.batch_decode.return_value = ["Hello world"]

        audio = np.zeros(16000, dtype=np.float32)
        result = _transcribe_chunk(
            audio, model, processor, tokenizer, [], device, dtype
        )
        assert result == "Hello world"

    def test_strips_prompt_echo(self) -> None:
        """The prompt echo should be stripped from the output."""
        model = MagicMock()
        processor = MagicMock()
        tokenizer = MagicMock()

        mock_inputs = MagicMock()
        mock_inputs.to.return_value = mock_inputs
        mock_inputs.__contains__ = lambda self, k: False
        mock_inputs.__getitem__ = lambda self, k: {"input_ids": MagicMock(shape=[0, 10])}[k]
        processor.apply_chat_template.return_value = mock_inputs

        tokenizer.pad_token_id = 0
        tokenizer.eos_token_id = 2
        model.generate.return_value = MagicMock(__getitem__=lambda s, k: [100, 101])
        tokenizer.batch_decode.return_value = [
            "Please transcribe this audio. Hello world"
        ]

        audio = np.zeros(16000, dtype=np.float32)
        result = _transcribe_chunk(
            audio, model, processor, tokenizer, [], "cpu", None
        )
        assert result == "Hello world"


# ---------------------------------------------------------------------------
# recognize_speech (integration mock)
# ---------------------------------------------------------------------------


class TestRecognizeSpeech:
    """Verify the full recognize_speech pipeline with mocked components."""

    @patch("backend.app.adapters.ark_asr._vad_segments")
    @patch("backend.app.adapters.ark_asr._load_model")
    @patch("backend.app.adapters.ark_asr._transcribe_chunk")
    @patch("backend.app.adapters.ark_asr._extract_segment")
    @patch("backend.app.adapters.ark_asr._build_bad_words_ids")
    @patch("backend.app.adapters.ark_asr.AudioSegment")
    def test_basic_pipeline(
        self,
        mock_audio_seg: MagicMock,
        mock_bad_words: MagicMock,
        mock_extract: MagicMock,
        mock_transcribe: MagicMock,
        mock_load: MagicMock,
        mock_vad: MagicMock,
        tmp_path: Path,
    ) -> None:
        from backend.app.adapters.ark_asr import recognize_speech

        # Setup mocks
        mock_vad.return_value = [(0, 5000), (6000, 10000)]
        mock_model = MagicMock()
        mock_model.parameters.return_value = iter([MagicMock(dtype=None)])
        mock_load.return_value = (mock_model, MagicMock(), MagicMock())
        mock_bad_words.return_value = []
        mock_extract.return_value = np.zeros(16000, dtype=np.float32)
        mock_transcribe.side_effect = ["Hello world.", "Goodbye world."]

        # Mock AudioSegment for duration
        mock_audio = MagicMock()
        mock_audio.__len__ = MagicMock(return_value=10000)
        mock_audio_seg.from_file.return_value = mock_audio

        # Create session dir
        session = tmp_path / "session"
        session.mkdir()
        vocals = tmp_path / "vocals.wav"
        vocals.touch()

        result = recognize_speech(vocals, session, "en")
        assert result.exists()

        data = json.loads(result.read_text(encoding="utf-8"))
        assert data["audio_info"]["duration"] == 10000
        assert len(data["result"]["utterances"]) == 2
        assert data["result"]["utterances"][0]["text"] == "Hello world."
        assert data["result"]["utterances"][0]["start_time"] == 0
        assert data["result"]["utterances"][0]["end_time"] == 5000
        assert data["result"]["utterances"][1]["text"] == "Goodbye world."
        assert data["result"]["utterances"][1]["start_time"] == 6000
        assert data["result"]["utterances"][1]["end_time"] == 10000
        assert data["result"]["text"] == "Hello world. Goodbye world."

    @patch("backend.app.adapters.ark_asr._vad_segments")
    @patch("backend.app.adapters.ark_asr._load_model")
    @patch("backend.app.adapters.ark_asr._transcribe_chunk")
    @patch("backend.app.adapters.ark_asr._extract_segment")
    @patch("backend.app.adapters.ark_asr._build_bad_words_ids")
    @patch("backend.app.adapters.ark_asr.AudioSegment")
    def test_no_vad_segments_falls_back_to_full_audio(
        self,
        mock_audio_seg: MagicMock,
        mock_bad_words: MagicMock,
        mock_extract: MagicMock,
        mock_transcribe: MagicMock,
        mock_load: MagicMock,
        mock_vad: MagicMock,
        tmp_path: Path,
    ) -> None:
        """When VAD returns no segments, use the full audio as one chunk."""
        from backend.app.adapters.ark_asr import recognize_speech

        mock_vad.return_value = []
        mock_model = MagicMock()
        mock_model.parameters.return_value = iter([MagicMock(dtype=None)])
        mock_load.return_value = (mock_model, MagicMock(), MagicMock())
        mock_bad_words.return_value = []
        mock_extract.return_value = np.zeros(16000, dtype=np.float32)
        mock_transcribe.return_value = "Full audio text."

        mock_audio = MagicMock()
        mock_audio.__len__ = MagicMock(return_value=15000)
        mock_audio_seg.from_file.return_value = mock_audio

        session = tmp_path / "session"
        session.mkdir()
        vocals = tmp_path / "vocals.wav"
        vocals.touch()

        result = recognize_speech(vocals, session, "en")
        data = json.loads(result.read_text(encoding="utf-8"))
        assert len(data["result"]["utterances"]) == 1
        assert data["result"]["utterances"][0]["start_time"] == 0
        assert data["result"]["utterances"][0]["end_time"] == 15000


# ---------------------------------------------------------------------------
# _split_text_to_sentences
# ---------------------------------------------------------------------------


class TestSplitTextToSentences:
    """Verify sentence splitting by punctuation."""

    def test_empty_text(self) -> None:
        assert _split_text_to_sentences("") == []

    def test_single_sentence_no_punct(self) -> None:
        result = _split_text_to_sentences("hello world")
        assert result == ["hello world"]

    def test_two_english_sentences(self) -> None:
        result = _split_text_to_sentences("Hello world. Goodbye world.")
        assert result == ["Hello world.", "Goodbye world."]

    def test_multiple_punctuation_types(self) -> None:
        result = _split_text_to_sentences("Really? Yes! Of course.")
        assert result == ["Really?", "Yes!", "Of course."]

    def test_chinese_sentences(self) -> None:
        result = _split_text_to_sentences("你好世界。再见世界。")
        assert result == ["你好世界。", "再见世界。"]

    def test_mixed_language(self) -> None:
        result = _split_text_to_sentences("Hello你好。Bye再见.")
        assert len(result) == 2

    def test_trailing_no_punct(self) -> None:
        result = _split_text_to_sentences("First sentence. Last part")
        assert result == ["First sentence.", "Last part"]

    def test_preserves_single_sentence(self) -> None:
        """If only one sentence (even with punct), return as single item."""
        result = _split_text_to_sentences("One sentence only.")
        # One sentence with punct → still 1 item since split needs >1
        assert len(result) == 1
        assert result[0] == "One sentence only."


# ---------------------------------------------------------------------------
# _distribute_utterances
# ---------------------------------------------------------------------------


class TestDistributeUtterances:
    """Verify timestamp distribution across split sentences."""

    def test_single_sentence_passthrough(self) -> None:
        result = _distribute_utterances("hello world", 0, 5000)
        assert len(result) == 1
        assert result[0]["text"] == "hello world"
        assert result[0]["start_time"] == 0
        assert result[0]["end_time"] == 5000
        assert result[0]["words"] == []

    def test_two_sentences_proportional(self) -> None:
        # "Hi." (3 chars) + "Hello world." (12 chars) = 15 chars
        result = _distribute_utterances("Hi. Hello world.", 0, 1500)
        assert len(result) == 2
        assert result[0]["text"] == "Hi."
        assert result[0]["start_time"] == 0
        # ~3/15 = 20% of 1500ms = 300ms
        assert result[0]["end_time"] == 300
        assert result[1]["text"] == "Hello world."
        assert result[1]["start_time"] == 300
        assert result[1]["end_time"] == 1500

    def test_three_sentences_cover_full_range(self) -> None:
        result = _distribute_utterances("A. B. C.", 1000, 4000)
        assert len(result) == 3
        assert result[0]["start_time"] == 1000
        assert result[-1]["end_time"] == 4000
        # 时间戳连续
        for i in range(len(result) - 1):
            assert result[i]["end_time"] == result[i + 1]["start_time"]

    def test_chinese_sentences(self) -> None:
        result = _distribute_utterances("你好。世界。", 0, 10000)
        assert len(result) == 2
        assert result[0]["text"] == "你好。"
        assert result[1]["text"] == "世界。"
        assert result[-1]["end_time"] == 10000
