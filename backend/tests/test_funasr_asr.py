from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from backend.app import database
from backend.app.adapters import funasr_asr
from backend.app.adapters.funasr_asr import (
    _convert_standard_result,
    _convert_vllm_result,
    _group_chars_to_sentences,
    _is_llm_based_model,
    _load_standard_model,
    _load_vllm_model,
    _resolve_vllm_mode,
    _vllm_available,
    recognize_speech,
)


# ---------------------------------------------------------------------------
# Model classification
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model_id,expected",
    [
        ("iic/SenseVoiceSmall", False),
        ("iic/Paraformer-large", False),
        ("FunAudioLLM/Fun-ASR-Nano-2512", True),
        ("FunAudioLLM/Fun-ASR-MLT-Nano-2512", True),
        ("FunAudioLLM/GLM-ASR-Nano", True),
        ("some-org/LLMASR-model", True),
    ],
)
def test_is_llm_based_model(model_id: str, expected: bool) -> None:
    assert _is_llm_based_model(model_id) is expected


# ---------------------------------------------------------------------------
# vllm availability / dispatch
# ---------------------------------------------------------------------------


def test_vllm_available_returns_bool(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(funasr_asr.importlib.util, "find_spec", lambda name: object() if name == "vllm" else None)
    assert _vllm_available() is True

    monkeypatch.setattr(funasr_asr.importlib.util, "find_spec", lambda name: None)
    assert _vllm_available() is False


def _stub_settings(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setattr(
        database,
        "get_funasr_settings",
        lambda: {"use_vllm": value},
    )


def test_resolve_vllm_mode_off_disables(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_settings(monkeypatch, "off")
    use_vllm, warning = _resolve_vllm_mode("FunAudioLLM/Fun-ASR-Nano-2512")
    assert use_vllm is False
    assert warning is None


def test_resolve_vllm_mode_auto_for_nano_with_vllm(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_settings(monkeypatch, "auto")
    monkeypatch.setattr(funasr_asr, "_vllm_available", lambda: True)
    use_vllm, warning = _resolve_vllm_mode("FunAudioLLM/Fun-ASR-Nano-2512")
    assert use_vllm is True
    assert warning is None


def test_resolve_vllm_mode_auto_for_nano_without_vllm_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_settings(monkeypatch, "auto")
    monkeypatch.setattr(funasr_asr, "_vllm_available", lambda: False)
    use_vllm, warning = _resolve_vllm_mode("FunAudioLLM/Fun-ASR-Nano-2512")
    assert use_vllm is False
    assert warning is not None
    assert "NotImplementedError" in warning


def test_resolve_vllm_mode_auto_for_non_llm_skips_vllm(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_settings(monkeypatch, "auto")
    monkeypatch.setattr(funasr_asr, "_vllm_available", lambda: True)
    use_vllm, warning = _resolve_vllm_mode("iic/SenseVoiceSmall")
    assert use_vllm is False
    assert warning is None


def test_resolve_vllm_mode_on_without_vllm_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_settings(monkeypatch, "on")
    monkeypatch.setattr(funasr_asr, "_vllm_available", lambda: False)
    with pytest.raises(RuntimeError, match="vllm package is not installed"):
        _resolve_vllm_mode("FunAudioLLM/Fun-ASR-Nano-2512")


def test_resolve_vllm_mode_on_with_vllm_enables(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_settings(monkeypatch, "on")
    monkeypatch.setattr(funasr_asr, "_vllm_available", lambda: True)
    use_vllm, warning = _resolve_vllm_mode("iic/SenseVoiceSmall")  # model id is ignored when on
    assert use_vllm is True
    assert warning is None


# ---------------------------------------------------------------------------
# recognize_speech dispatch
# ---------------------------------------------------------------------------


def test_recognize_speech_returns_existing_output(tmp_path: Path) -> None:
    session = tmp_path / "session"
    (session / "metadata").mkdir(parents=True)
    asr_file = session / "metadata" / "asr.json"
    asr_file.write_text("{}", encoding="utf-8")
    vocals = tmp_path / "vocals.wav"
    vocals.write_bytes(b"")
    result = recognize_speech(vocals, session, "zh")
    assert result == asr_file


def test_recognize_speech_dispatches_to_vllm_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    session = tmp_path / "session"
    (session / "metadata").mkdir(parents=True)
    (session / "media").mkdir(parents=True)
    vocals = session / "media" / "vocals.wav"
    vocals.write_bytes(b"\x00" * 100)

    _stub_settings(monkeypatch, "auto")
    monkeypatch.setattr(funasr_asr, "_vllm_available", lambda: True)

    vllm_model = MagicMock()
    vllm_model.generate.return_value = [
        {
            "key": "vocals",
            "text": "你好世界。",
            "timestamps": [
                {"token": "你", "start_time": 0.0, "end_time": 0.4},
                {"token": "好", "start_time": 0.4, "end_time": 0.8},
                {"token": "世", "start_time": 0.8, "end_time": 1.2},
                {"token": "界", "start_time": 1.2, "end_time": 1.6},
                {"token": "。", "start_time": 1.6, "end_time": 1.7},
            ],
        }
    ]
    monkeypatch.setattr(funasr_asr, "_load_vllm_model", lambda model_id=None: vllm_model)

    # Avoid actually reading audio metadata in the test environment.
    monkeypatch.setattr(
        funasr_asr.AudioSegment,
        "from_file",
        lambda *_args, **_kwargs: MagicMock(__len__=lambda self: 2000),
    )

    out = recognize_speech(
        vocals, session, "zh", model_id="FunAudioLLM/Fun-ASR-Nano-2512"
    )
    vllm_model.generate.assert_called_once()
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["result"]["text"] == "你好世界。"
    assert len(payload["result"]["utterances"]) == 1
    assert payload["result"]["utterances"][0]["text"] == "你好世界。"
    assert payload["result"]["utterances"][0]["start_time"] == 0
    assert payload["result"]["utterances"][0]["end_time"] == 1700


def test_recognize_speech_dispatches_to_standard_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    session = tmp_path / "session"
    (session / "metadata").mkdir(parents=True)
    (session / "media").mkdir(parents=True)
    vocals = session / "media" / "vocals.wav"
    vocals.write_bytes(b"\x00" * 100)

    _stub_settings(monkeypatch, "off")
    monkeypatch.setattr(funasr_asr, "_vllm_available", lambda: False)

    standard_model = MagicMock()
    standard_model.generate.return_value = [
        {
            "text": "hello world",
            "sentence_info": [
                {"text": "hello world", "start": 0, "end": 1000},
            ],
        }
    ]
    monkeypatch.setattr(funasr_asr, "_load_standard_model", lambda model_id=None: standard_model)

    monkeypatch.setattr(
        funasr_asr.AudioSegment,
        "from_file",
        lambda *_args, **_kwargs: MagicMock(__len__=lambda self: 1000),
    )

    out = recognize_speech(vocals, session, "en", model_id="iic/SenseVoiceSmall")
    standard_model.generate.assert_called_once()
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["result"]["text"] == "hello world"
    assert len(payload["result"]["utterances"]) == 1
    assert payload["result"]["utterances"][0]["text"] == "hello world"


# ---------------------------------------------------------------------------
# Standard result conversion
# ---------------------------------------------------------------------------


def test_convert_standard_result_with_sentences() -> None:
    result = {
        "text": "你好世界",
        "sentence_info": [
            {"text": "你好", "start": 0, "end": 1500},
            {"text": "世界", "start": 1500, "end": 3000},
        ],
    }
    utterances = _convert_standard_result(result, 3000)
    assert len(utterances) == 2
    assert utterances[0] == {"text": "你好", "start_time": 0, "end_time": 1500, "words": []}
    assert utterances[1] == {"text": "世界", "start_time": 1500, "end_time": 3000, "words": []}


def test_convert_standard_result_with_ms_values() -> None:
    result = {
        "text": "hello",
        "sentence_info": [{"text": "hello", "start": 200000, "end": 203000}],
    }
    utterances = _convert_standard_result(result, 3000)
    assert utterances[0]["start_time"] == 200000
    assert utterances[0]["end_time"] == 203000


def test_convert_standard_result_fallback_to_text() -> None:
    result = {"text": "  hello world  "}
    utterances = _convert_standard_result(result, 5000)
    assert utterances == [{"text": "hello world", "start_time": 0, "end_time": 5000, "words": []}]


def test_convert_standard_result_empty_text() -> None:
    assert _convert_standard_result({"text": ""}, 1000) == []


# ---------------------------------------------------------------------------
# vLLM char-to-sentence grouping
# ---------------------------------------------------------------------------


def test_group_chars_to_sentences_splits_on_punctuation() -> None:
    text = "你好世界。今天天气好！"
    timestamps = [
        {"token": "你", "start_time": 0.0, "end_time": 0.4},
        {"token": "好", "start_time": 0.4, "end_time": 0.8},
        {"token": "世", "start_time": 0.8, "end_time": 1.2},
        {"token": "界", "start_time": 1.2, "end_time": 1.6},
        {"token": "。", "start_time": 1.6, "end_time": 1.7},
        {"token": "今", "start_time": 1.8, "end_time": 2.2},
        {"token": "天", "start_time": 2.2, "end_time": 2.6},
        {"token": "天", "start_time": 2.6, "end_time": 3.0},
        {"token": "气", "start_time": 3.0, "end_time": 3.4},
        {"token": "好", "start_time": 3.4, "end_time": 3.8},
        {"token": "！", "start_time": 3.8, "end_time": 3.9},
    ]
    sentences = _group_chars_to_sentences(text, timestamps)
    assert [s["text"] for s in sentences] == ["你好世界。", "今天天气好！"]
    assert sentences[0]["start_time"] == 0
    assert sentences[0]["end_time"] == 1700
    assert sentences[1]["start_time"] == 1800
    assert sentences[1]["end_time"] == 3900


def test_group_chars_to_sentences_handles_missing_punctuation() -> None:
    text = "你好世界"
    timestamps = [
        {"token": "你", "start_time": 0.0, "end_time": 0.5},
        {"token": "好", "start_time": 0.5, "end_time": 1.0},
        {"token": "世", "start_time": 1.0, "end_time": 1.5},
        {"token": "界", "start_time": 1.5, "end_time": 2.0},
    ]
    sentences = _group_chars_to_sentences(text, timestamps)
    assert len(sentences) == 1
    assert sentences[0]["text"] == "你好世界"
    assert sentences[0]["start_time"] == 0
    assert sentences[0]["end_time"] == 2000


def test_group_chars_to_sentences_empty_inputs() -> None:
    assert _group_chars_to_sentences("", [{"token": "x", "start_time": 0, "end_time": 1}]) == []
    assert _group_chars_to_sentences("hello", []) == []


def test_group_chars_to_sentences_inherits_last_timestamp() -> None:
    # Text has more characters than timestamps; trailing chars reuse the
    # last known end time.
    text = "你好世界"
    timestamps = [
        {"token": "你", "start_time": 0.0, "end_time": 0.5},
        {"token": "好", "start_time": 0.5, "end_time": 1.0},
    ]
    sentences = _group_chars_to_sentences(text, timestamps)
    assert len(sentences) == 1
    assert sentences[0]["start_time"] == 0
    assert sentences[0]["end_time"] == 1000  # last known end


# ---------------------------------------------------------------------------
# vLLM result conversion
# ---------------------------------------------------------------------------


def test_convert_vllm_result_with_timestamps() -> None:
    result = {
        "text": "你好世界。",
        "timestamps": [
            {"token": "你", "start_time": 0.0, "end_time": 0.5},
            {"token": "好", "start_time": 0.5, "end_time": 1.0},
            {"token": "世", "start_time": 1.0, "end_time": 1.5},
            {"token": "界", "start_time": 1.5, "end_time": 2.0},
            {"token": "。", "start_time": 2.0, "end_time": 2.1},
        ],
    }
    utterances = _convert_vllm_result(result, 2100)
    assert len(utterances) == 1
    assert utterances[0]["text"] == "你好世界。"
    assert utterances[0]["start_time"] == 0
    assert utterances[0]["end_time"] == 2100


def test_convert_vllm_result_fallback_without_timestamps() -> None:
    result = {"text": "hello world"}
    utterances = _convert_vllm_result(result, 5000)
    assert utterances == [
        {"text": "hello world", "start_time": 0, "end_time": 5000, "words": []}
    ]


def test_convert_vllm_result_empty_text() -> None:
    assert _convert_vllm_result({"text": ""}, 1000) == []


# ---------------------------------------------------------------------------
# Model loader caching (smoke test, no real GPU access)
# ---------------------------------------------------------------------------


def test_load_standard_model_returns_cached_instance(monkeypatch: pytest.MonkeyPatch) -> None:
    funasr_asr._STANDARD_MODEL = MagicMock(name="cached")
    funasr_asr._STANDARD_MODEL_NAME = "iic/SenseVoiceSmall"
    result = _load_standard_model("iic/SenseVoiceSmall")
    assert result is funasr_asr._STANDARD_MODEL
    # Cleanup
    funasr_asr._STANDARD_MODEL = None
    funasr_asr._STANDARD_MODEL_NAME = None


def test_load_vllm_model_returns_cached_instance(monkeypatch: pytest.MonkeyPatch) -> None:
    funasr_asr._VLLM_MODEL = MagicMock(name="cached")
    funasr_asr._VLLM_MODEL_NAME = "FunAudioLLM/Fun-ASR-Nano-2512"
    result = _load_vllm_model("FunAudioLLM/Fun-ASR-Nano-2512")
    assert result is funasr_asr._VLLM_MODEL
    # Cleanup
    funasr_asr._VLLM_MODEL = None
    funasr_asr._VLLM_MODEL_NAME = None
