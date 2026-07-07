"""Tests for backend.app.adapters.moss_audio_asr — timestamp parsing and text assembly."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from backend.app.adapters.moss_audio_asr import (
    _distribute_by_punctuation,
    _distribute_with_time_markers,
    _force_split,
    _parse_timestamped_text,
    _split_text_to_sentences,
    _vad_segments,
)


# ---------------------------------------------------------------------------
# _parse_timestamped_text
# ---------------------------------------------------------------------------


class TestParseTimestampedText:
    """Verify time-marker parsing from MOSS-Audio output."""

    def test_no_markers(self) -> None:
        assert _parse_timestamped_text("Hello world.") is None

    def test_single_marker(self) -> None:
        result = _parse_timestamped_text("[0.06] Hello world.")
        assert result is not None
        assert len(result) == 1
        assert result[0]["text"] == "Hello world."
        assert result[0]["start_ms"] == 60
        assert result[0]["end_ms"] is None

    def test_two_markers_with_text_between(self) -> None:
        result = _parse_timestamped_text("[0.06]Hello. [2.64]World.")
        assert result is not None
        assert len(result) == 2
        assert result[0]["text"] == "Hello."
        assert result[0]["start_ms"] == 60
        assert result[0]["end_ms"] == 2640
        assert result[1]["text"] == "World."
        assert result[1]["start_ms"] == 2640
        assert result[1]["end_ms"] is None

    def test_paired_boundary_timestamps(self) -> None:
        """[2.64][3.04] means end=2.64s, next start=3.04s."""
        result = _parse_timestamped_text("[0.06]Hello.[2.64][3.04]World.")
        assert result is not None
        assert len(result) == 2
        assert result[0]["text"] == "Hello."
        assert result[0]["start_ms"] == 60
        assert result[0]["end_ms"] == 3040  # last of paired markers
        assert result[1]["text"] == "World."
        assert result[1]["start_ms"] == 3040

    def test_real_moss_output(self) -> None:
        """Test with actual MOSS-Audio output format."""
        text = (
            "[0.06]A lot of you have asked me if I'm gonna be playing R E nine."
            "[2.64][3.04]Good news, I am."
            "[12.37][12.79]But I'm gonna be starting that um tomorrow."
        )
        result = _parse_timestamped_text(text)
        assert result is not None
        assert len(result) == 3
        assert result[0]["start_ms"] == 60
        assert result[0]["end_ms"] == 3040
        assert result[1]["start_ms"] == 3040
        assert result[1]["end_ms"] == 12790
        assert result[2]["start_ms"] == 12790

    def test_integer_timestamps(self) -> None:
        result = _parse_timestamped_text("[0]A [2]B [4]C")
        assert result is not None
        assert len(result) == 3
        assert result[0]["start_ms"] == 0
        assert result[1]["start_ms"] == 2000
        assert result[2]["start_ms"] == 4000

    def test_empty_text_between_markers_skipped(self) -> None:
        result = _parse_timestamped_text("[0.06] [2.64] Hello")
        assert result is not None
        assert len(result) == 1
        assert result[0]["text"] == "Hello"


# ---------------------------------------------------------------------------
# _distribute_with_time_markers
# ---------------------------------------------------------------------------


class TestDistributeWithTimeMarkers:
    """Verify utterance assembly with time markers and VAD boundaries."""

    def test_with_time_markers(self) -> None:
        text = "[0.06]Hello. [2.64]World."
        result = _distribute_with_time_markers(text, 0, 5000)
        assert len(result) == 2
        assert result[0]["text"] == "Hello."
        assert result[0]["start_time"] == 60
        assert result[0]["end_time"] == 2640
        assert result[1]["text"] == "World."
        assert result[1]["start_time"] == 2640
        assert result[1]["end_time"] == 5000  # filled by VAD end

    def test_paired_boundary_markers(self) -> None:
        text = "[0.06]Hello.[2.64][3.04]World."
        result = _distribute_with_time_markers(text, 0, 5000)
        assert len(result) == 2
        assert result[0]["text"] == "Hello."
        assert result[0]["start_time"] == 60
        assert result[0]["end_time"] == 3040
        assert result[1]["text"] == "World."
        assert result[1]["start_time"] == 3040
        assert result[1]["end_time"] == 5000

    def test_time_markers_offset_by_vad_start(self) -> None:
        """Timestamps are relative to the segment; offset by vad_start_ms."""
        text = "[0.06]Hello. [2.64]World."
        result = _distribute_with_time_markers(text, 6000, 10000)
        assert len(result) == 2
        assert result[0]["start_time"] == 6060  # 60 + 6000
        assert result[0]["end_time"] == 8640    # 2640 + 6000
        assert result[1]["start_time"] == 8640
        assert result[1]["end_time"] == 10000   # VAD end

    def test_fallback_to_punctuation(self) -> None:
        """Without time markers, falls back to punctuation splitting."""
        text = "Hello world. Goodbye world."
        result = _distribute_with_time_markers(text, 0, 10000)
        assert len(result) == 2
        assert result[0]["text"] == "Hello world."
        assert result[1]["text"] == "Goodbye world."


# ---------------------------------------------------------------------------
# _distribute_by_punctuation
# ---------------------------------------------------------------------------


class TestDistributeByPunctuation:
    """Verify punctuation-based timestamp distribution."""

    def test_single_sentence(self) -> None:
        result = _distribute_by_punctuation("Hello world", 0, 5000)
        assert len(result) == 1
        assert result[0]["start_time"] == 0
        assert result[0]["end_time"] == 5000

    def test_two_sentences(self) -> None:
        result = _distribute_by_punctuation("Hi. Hello world.", 0, 1500)
        assert len(result) == 2
        assert result[0]["text"] == "Hi."
        assert result[-1]["end_time"] == 1500

    def test_three_sentences_cover_full_range(self) -> None:
        result = _distribute_by_punctuation("A. B. C.", 1000, 4000)
        assert len(result) == 3
        assert result[0]["start_time"] == 1000
        assert result[-1]["end_time"] == 4000
        for i in range(len(result) - 1):
            assert result[i]["end_time"] == result[i + 1]["start_time"]


# ---------------------------------------------------------------------------
# _vad_segments
# ---------------------------------------------------------------------------


class TestVadSegments:
    """Verify VAD segment parsing."""

    @patch("backend.app.adapters.moss_audio_asr._load_vad_model")
    def test_empty_vad_result(self, mock_load: MagicMock) -> None:
        mock_vad = MagicMock()
        mock_vad.generate.return_value = []
        mock_load.return_value = mock_vad
        assert _vad_segments("test.wav") == []

    @patch("backend.app.adapters.moss_audio_asr._load_vad_model")
    def test_basic_segments(self, mock_load: MagicMock) -> None:
        mock_vad = MagicMock()
        mock_vad.generate.return_value = [{"timestamp": [[0, 5000], [6000, 12000]]}]
        mock_load.return_value = mock_vad
        result = _vad_segments("test.wav")
        assert result == [(0, 5000), (6000, 12000)]

    @patch("backend.app.adapters.moss_audio_asr._load_vad_model")
    def test_split_long_segment(self, mock_load: MagicMock) -> None:
        mock_vad = MagicMock()
        mock_vad.generate.return_value = [{"timestamp": [[0, 70000]]}]
        mock_load.return_value = mock_vad
        result = _vad_segments("test.wav")
        assert len(result) == 3
        assert result[0] == (0, 30000)
        assert result[1] == (30000, 60000)
        assert result[2] == (60000, 70000)

    @patch("backend.app.adapters.moss_audio_asr._load_vad_model")
    def test_vad_fallback_single_segment(self, mock_load: MagicMock) -> None:
        """When VAD returns one huge segment, fall back to fixed splitting."""
        mock_vad = MagicMock()
        mock_vad.generate.return_value = [{"timestamp": [[0, 300000]]}]
        mock_load.return_value = mock_vad
        result = _vad_segments("test.wav", duration_ms=300000)
        # Should be force-split into 30s chunks
        assert len(result) == 10
        assert result[0] == (0, 30000)
        assert result[-1] == (270000, 300000)

    @patch("backend.app.adapters.moss_audio_asr._load_vad_model")
    def test_vad_fallback_no_segments(self, mock_load: MagicMock) -> None:
        """When VAD returns nothing, fall back to fixed splitting."""
        mock_vad = MagicMock()
        mock_vad.generate.return_value = []
        mock_load.return_value = mock_vad
        result = _vad_segments("test.wav", duration_ms=60000)
        assert len(result) == 2
        assert result[0] == (0, 30000)
        assert result[1] == (30000, 60000)


# ---------------------------------------------------------------------------
# _force_split
# ---------------------------------------------------------------------------


class TestForceSplit:
    """Verify fixed-size chunk splitting."""

    def test_short_audio(self) -> None:
        result = _force_split(10000)
        assert result == [(0, 10000)]

    def test_exact_chunk(self) -> None:
        result = _force_split(30000)
        assert result == [(0, 30000)]

    def test_two_chunks(self) -> None:
        result = _force_split(45000)
        assert result == [(0, 30000), (30000, 45000)]

    def test_many_chunks(self) -> None:
        result = _force_split(300000)
        assert len(result) == 10
        assert result[0] == (0, 30000)
        assert result[-1] == (270000, 300000)

    def test_custom_chunk_size(self) -> None:
        result = _force_split(25000, chunk_ms=10000)
        assert result == [(0, 10000), (10000, 20000), (20000, 25000)]


# ---------------------------------------------------------------------------
# recognize_speech (integration mock)
# ---------------------------------------------------------------------------


class TestRecognizeSpeech:
    """Verify the full pipeline with mocked components."""

    @patch("backend.app.adapters.moss_audio_asr._vad_segments")
    @patch("backend.app.adapters.moss_audio_asr._load_model")
    @patch("backend.app.adapters.moss_audio_asr._transcribe_chunk")
    @patch("backend.app.adapters.moss_audio_asr._extract_segment")
    @patch("backend.app.adapters.moss_audio_asr._load_audio")
    def test_basic_pipeline_with_time_markers(
        self,
        mock_load_audio: MagicMock,
        mock_extract: MagicMock,
        mock_transcribe: MagicMock,
        mock_load: MagicMock,
        mock_vad: MagicMock,
        tmp_path: Path,
    ) -> None:
        from backend.app.adapters.moss_audio_asr import recognize_speech

        mock_vad.return_value = [(0, 5000), (6000, 10000)]
        mock_load.return_value = (MagicMock(), MagicMock())
        mock_load_audio.return_value = np.zeros(160000, dtype=np.float32)  # 10s at 16kHz
        mock_extract.return_value = np.zeros(16000, dtype=np.float32)
        mock_transcribe.side_effect = [
            "[0.06]Hello world.[3.04]Goodbye.",
            "[0.06]How are you?",
        ]

        session = tmp_path / "session"
        session.mkdir()
        vocals = tmp_path / "vocals.wav"
        vocals.touch()

        result = recognize_speech(vocals, session, "en")
        data = json.loads(result.read_text(encoding="utf-8"))

        # First segment: 2 utterances from time markers
        # Second segment: 1 utterance
        assert len(data["result"]["utterances"]) == 3
        assert data["result"]["utterances"][0]["text"] == "Hello world."
        assert data["result"]["utterances"][0]["start_time"] == 60
        assert data["result"]["utterances"][0]["end_time"] == 3040
        assert data["result"]["utterances"][1]["text"] == "Goodbye."
        assert data["result"]["utterances"][1]["start_time"] == 3040
        assert data["result"]["utterances"][2]["text"] == "How are you?"
        assert data["result"]["utterances"][2]["start_time"] == 6060  # 60 + 6000
        assert data["result"]["utterances"][2]["end_time"] == 10000

    @patch("backend.app.adapters.moss_audio_asr._vad_segments")
    @patch("backend.app.adapters.moss_audio_asr._load_model")
    @patch("backend.app.adapters.moss_audio_asr._transcribe_chunk")
    @patch("backend.app.adapters.moss_audio_asr._extract_segment")
    @patch("backend.app.adapters.moss_audio_asr._load_audio")
    def test_fallback_without_time_markers(
        self,
        mock_load_audio: MagicMock,
        mock_extract: MagicMock,
        mock_transcribe: MagicMock,
        mock_load: MagicMock,
        mock_vad: MagicMock,
        tmp_path: Path,
    ) -> None:
        """When no time markers, falls back to punctuation splitting."""
        from backend.app.adapters.moss_audio_asr import recognize_speech

        mock_vad.return_value = [(0, 10000)]
        mock_load.return_value = (MagicMock(), MagicMock())
        mock_load_audio.return_value = np.zeros(160000, dtype=np.float32)
        mock_extract.return_value = np.zeros(16000, dtype=np.float32)
        mock_transcribe.return_value = "Hello world. Goodbye world."

        session = tmp_path / "session"
        session.mkdir()
        vocals = tmp_path / "vocals.wav"
        vocals.touch()

        result = recognize_speech(vocals, session, "en")
        data = json.loads(result.read_text(encoding="utf-8"))

        assert len(data["result"]["utterances"]) == 2
        assert data["result"]["utterances"][0]["text"] == "Hello world."
        assert data["result"]["utterances"][0]["start_time"] == 0
        assert data["result"]["utterances"][1]["text"] == "Goodbye world."
        assert data["result"]["utterances"][1]["end_time"] == 10000
