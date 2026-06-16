from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import httpx

from backend.app.adapters import remote_funasr_asr


def _json_response(payload: dict) -> httpx.Response:
    return httpx.Response(200, json=payload, request=httpx.Request("POST", "http://test/asr"))


def _patch_audio_duration(monkeypatch, duration_ms: int = 5000) -> None:
    monkeypatch.setattr(
        remote_funasr_asr.AudioSegment,
        "from_file",
        lambda *_args, **_kwargs: MagicMock(__len__=lambda self: duration_ms),
    )


def test_is_configured_uses_base_url(monkeypatch) -> None:
    monkeypatch.delenv("FUNASR_REMOTE_BASE_URL", raising=False)
    assert remote_funasr_asr.is_configured() is False

    monkeypatch.setenv("FUNASR_REMOTE_BASE_URL", "http://127.0.0.1:8899")
    assert remote_funasr_asr.is_configured() is True


def test_requires_remote_vllm_only_for_llm_models() -> None:
    assert remote_funasr_asr.requires_remote_vllm("iic/SenseVoiceSmall") is False
    assert remote_funasr_asr.requires_remote_vllm("FunAudioLLM/Fun-ASR-Nano-2512") is True
    assert remote_funasr_asr.requires_remote_vllm("FunAudioLLM/GLM-ASR-Nano") is True
    assert remote_funasr_asr.requires_remote_vllm("some-org/LLMASR-model") is True


def test_recognize_speech_converts_asr_response(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("FUNASR_REMOTE_BASE_URL", "http://127.0.0.1:8899")
    monkeypatch.setenv("FUNASR_REMOTE_ENDPOINT", "/asr")
    monkeypatch.setenv("FUNASR_REMOTE_PATH_MODE", "off")
    _patch_audio_duration(monkeypatch, 3200)

    captured = {}

    def fake_post(url, *, files, data, timeout):
        captured["url"] = url
        captured["data"] = data
        assert files["file"][0] == "vocals.wav"
        return _json_response(
            {
                "text": "hello world",
                "duration": 3.2,
                "segments": [
                    {
                        "text": "hello",
                        "start": 0.0,
                        "end": 1.2,
                        "speaker": "SPK0",
                        "words": [{"word": "hello", "start": 0.0, "end": 1.2}],
                    },
                    {"text": "world", "start": 1.4, "end": 3.2},
                ],
            }
        )

    monkeypatch.setattr(remote_funasr_asr.httpx, "post", fake_post)

    session = tmp_path / "session"
    vocals = tmp_path / "vocals.wav"
    vocals.write_bytes(b"audio")

    out = remote_funasr_asr.recognize_speech(
        vocals, session, "en", model_id="FunAudioLLM/Fun-ASR-Nano-2512"
    )

    assert captured["url"] == "http://127.0.0.1:8899/asr"
    assert captured["data"]["language"] == "英文"
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["audio_info"]["duration"] == 3200
    assert payload["result"]["text"] == "hello world"
    assert payload["result"]["utterances"] == [
        {
            "text": "hello",
            "start_time": 0,
            "end_time": 1200,
            "words": [{"text": "hello", "start_time": 0, "end_time": 1200}],
            "additions": {"speaker": "SPK0"},
        },
        {"text": "world", "start_time": 1400, "end_time": 3200, "words": []},
    ]


def test_recognize_speech_uses_path_mode_for_asr(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("FUNASR_REMOTE_BASE_URL", "http://127.0.0.1:8899")
    monkeypatch.setenv("FUNASR_REMOTE_ENDPOINT", "/asr")
    monkeypatch.setenv("FUNASR_REMOTE_PATH_MODE", "auto")
    _patch_audio_duration(monkeypatch, 1200)

    captured = {}

    def fake_post(url, *, files, data, timeout):
        captured["files"] = files
        captured["data"] = data
        return _json_response({"text": "path mode", "duration": 1.2, "segments": []})

    monkeypatch.setattr(remote_funasr_asr.httpx, "post", fake_post)

    session = tmp_path / "session"
    vocals = tmp_path / "vocals.wav"
    vocals.write_bytes(b"audio")
    out = remote_funasr_asr.recognize_speech(vocals, session, "en")

    assert captured["files"] is None
    assert captured["data"]["audio_path"].startswith("/mnt/")
    debug = json.loads((session / "metadata" / "asr.remote_request.json").read_text(encoding="utf-8"))
    assert debug["transport"] == "path"
    assert debug["remote_audio_path"] == captured["data"]["audio_path"]
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["result"]["text"] == "path mode"


def test_recognize_speech_autostarts_and_stops_service(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("FUNASR_REMOTE_BASE_URL", "http://127.0.0.1:8899")
    monkeypatch.setenv("FUNASR_REMOTE_ENDPOINT", "/asr")
    monkeypatch.setenv("FUNASR_REMOTE_PATH_MODE", "off")
    monkeypatch.setenv("FUNASR_REMOTE_AUTOSTART", "true")
    monkeypatch.setenv("FUNASR_REMOTE_AUTOSTOP", "true")
    _patch_audio_duration(monkeypatch, 1000)

    events = []
    process = MagicMock()

    monkeypatch.setattr(remote_funasr_asr, "_service_is_running", lambda: False)
    monkeypatch.setattr(remote_funasr_asr, "_start_service", lambda: events.append("start") or process)
    monkeypatch.setattr(
        remote_funasr_asr,
        "_wait_for_service",
        lambda started_process: events.append(("wait", started_process)),
    )
    monkeypatch.setattr(
        remote_funasr_asr,
        "_stop_service",
        lambda started_process: events.append(("stop", started_process)),
    )

    def fake_post(url, *, files, data, timeout):
        events.append("post")
        return _json_response({"text": "auto service", "duration": 1.0, "segments": []})

    monkeypatch.setattr(remote_funasr_asr.httpx, "post", fake_post)

    session = tmp_path / "session"
    vocals = tmp_path / "vocals.wav"
    vocals.write_bytes(b"audio")
    remote_funasr_asr.recognize_speech(vocals, session, "en")

    assert events == ["start", ("wait", process), "post", ("stop", process)]


def test_recognize_speech_keeps_existing_service_running(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("FUNASR_REMOTE_BASE_URL", "http://127.0.0.1:8899")
    monkeypatch.setenv("FUNASR_REMOTE_ENDPOINT", "/asr")
    monkeypatch.setenv("FUNASR_REMOTE_PATH_MODE", "off")
    monkeypatch.setenv("FUNASR_REMOTE_AUTOSTART", "true")
    _patch_audio_duration(monkeypatch, 1000)

    monkeypatch.setattr(remote_funasr_asr, "_service_is_running", lambda: True)
    monkeypatch.setattr(
        remote_funasr_asr,
        "_start_service",
        lambda: (_ for _ in ()).throw(AssertionError("should not start")),
    )
    monkeypatch.setattr(
        remote_funasr_asr,
        "_stop_service",
        lambda _process: (_ for _ in ()).throw(AssertionError("should not stop")),
    )
    monkeypatch.setattr(
        remote_funasr_asr.httpx,
        "post",
        lambda *_args, **_kwargs: _json_response({"text": "already up", "duration": 1.0, "segments": []}),
    )

    session = tmp_path / "session"
    vocals = tmp_path / "vocals.wav"
    vocals.write_bytes(b"audio")
    out = remote_funasr_asr.recognize_speech(vocals, session, "en")

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["result"]["text"] == "already up"


def test_recognize_speech_splits_remote_words_into_sentences(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("FUNASR_REMOTE_BASE_URL", "http://127.0.0.1:8899")
    monkeypatch.setenv("FUNASR_REMOTE_ENDPOINT", "/asr")
    monkeypatch.setenv("FUNASR_REMOTE_PATH_MODE", "off")
    _patch_audio_duration(monkeypatch, 2000)

    def fake_post(url, *, files, data, timeout):
        return _json_response({
            "text": "Hello world. Good news.",
            "duration": 2.0,
            "segments": [{
                "text": "Hello world. Good news.",
                "start": 0.0,
                "end": 2.0,
                "words": [
                    {"word": "Hello", "start": 0.0, "end": 0.2},
                    {"word": " world", "start": 0.2, "end": 0.4},
                    {"word": ".", "start": 0.4, "end": 0.5},
                    {"word": " Good", "start": 1.0, "end": 1.2},
                    {"word": " news", "start": 1.2, "end": 1.5},
                    {"word": ".", "start": 1.5, "end": 1.6},
                ],
            }],
        })

    monkeypatch.setattr(remote_funasr_asr.httpx, "post", fake_post)

    session = tmp_path / "session"
    vocals = tmp_path / "vocals.wav"
    vocals.write_bytes(b"audio")
    out = remote_funasr_asr.recognize_speech(vocals, session, "en")
    payload = json.loads(out.read_text(encoding="utf-8"))

    assert [item["text"] for item in payload["result"]["utterances"]] == [
        "Hello world.",
        "Good news.",
    ]
    assert payload["result"]["utterances"][0]["start_time"] == 0
    assert payload["result"]["utterances"][0]["end_time"] == 500
    assert len(payload["result"]["utterances"][0]["words"]) == 3


def test_recognize_speech_converts_openai_verbose_json(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("FUNASR_REMOTE_BASE_URL", "http://127.0.0.1:8899")
    monkeypatch.setenv("FUNASR_REMOTE_ENDPOINT", "/v1/audio/transcriptions")
    _patch_audio_duration(monkeypatch, 2000)

    captured = {}

    def fake_post(url, *, files, data, timeout):
        captured["url"] = url
        captured["data"] = data
        return _json_response(
            {
                "task": "transcribe",
                "language": "en",
                "duration": 2.0,
                "text": "hello",
                "segments": [{"id": 0, "start": 0.0, "end": 2.0, "text": "hello"}],
            }
        )

    monkeypatch.setattr(remote_funasr_asr.httpx, "post", fake_post)

    session = tmp_path / "session"
    vocals = tmp_path / "vocals.wav"
    vocals.write_bytes(b"audio")
    out = remote_funasr_asr.recognize_speech(vocals, session, "en", model_id="iic/SenseVoiceSmall")

    assert captured["url"] == "http://127.0.0.1:8899/v1/audio/transcriptions"
    assert captured["data"]["model"] == "sensevoice"
    assert captured["data"]["response_format"] == "verbose_json"
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["result"]["utterances"][0]["text"] == "hello"


def test_recognize_speech_raises_on_empty_response(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("FUNASR_REMOTE_BASE_URL", "http://127.0.0.1:8899")
    monkeypatch.setenv("FUNASR_REMOTE_PATH_MODE", "off")
    _patch_audio_duration(monkeypatch, 1000)
    monkeypatch.setattr(
        remote_funasr_asr.httpx,
        "post",
        lambda *_args, **_kwargs: _json_response({"text": "", "segments": []}),
    )

    session = tmp_path / "session"
    vocals = tmp_path / "vocals.wav"
    vocals.write_bytes(b"audio")

    try:
        remote_funasr_asr.recognize_speech(vocals, session, "en")
    except RuntimeError as exc:
        assert "did not return any segments" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")
