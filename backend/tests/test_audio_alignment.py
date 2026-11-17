from __future__ import annotations

import json

import numpy as np
import pytest
import soundfile as sf

from backend.app.adapters import audio, ffmpeg


def test_dense_audio_reserves_time_and_rewrites_existing_subtitles(monkeypatch, tmp_path):
    tts = tmp_path / "segments" / "tts"
    media = tmp_path / "media"
    metadata = tmp_path / "metadata"
    for directory in (tts, media, metadata):
        directory.mkdir(parents=True)
    source = media / "video_source.mp4"
    source.touch()
    rate = 8000
    rows = [
        {"start_time": i * 350, "end_time": (i + 1) * 350, "dst": "这是完整译文", "dst_lang": "zh"}
        for i in range(12)
    ]
    for index in range(1, len(rows) + 1):
        sf.write(tts / f"{index:04d}.wav", np.full(7000, 0.25, dtype=np.float32), rate)
    draft = metadata / "translation.zh.json"
    draft.write_text(json.dumps({"translation": rows}), encoding="utf-8")
    srt = metadata / "subtitles.zh.srt"
    srt.write_text("旧字幕", encoding="utf-8")
    monkeypatch.setattr(audio, "probe_video_duration", lambda _: 4.5)
    received = []

    def fit(clip, source_rate, sample_rate, frames):
        assert source_rate == sample_rate == rate
        received.append((len(clip), frames))
        return np.full(frames, 0.25, dtype=np.float32)

    monkeypatch.setattr(audio, "_fit_clip", fit)
    dubbing, timings = audio.merge_tts_audio(draft, tts, tmp_path)
    result = json.loads(timings.read_text(encoding="utf-8"))["translation"]
    assert len(received) == len(result) == len(rows)
    assert any(maximum < raw for raw, maximum in received)
    assert result[-1]["actual_end_time"] == round(sf.info(dubbing).duration * 1000)
    assert result[-1]["actual_end_time"] <= 4400
    for original, aligned in zip(rows, result, strict=True):
        assert 0 <= aligned["actual_start_time"] - original["start_time"] <= audio.MAX_LAG_MS
        assert aligned["actual_end_time"] <= original["end_time"] + audio.MAX_LAG_MS
    assert all(a["actual_end_time"] <= b["actual_start_time"] for a, b in zip(result, result[1:]))
    assert ffmpeg._find_or_generate_srt(tmp_path, timings) == srt
    assert ffmpeg._srt_time(result[-1]["actual_end_time"]) in srt.read_text(encoding="utf-8")
    assert "旧字幕" not in srt.read_text(encoding="utf-8")


def test_impossible_tail_fails_without_overwriting_cached_audio(monkeypatch, tmp_path):
    tts = tmp_path / "tts"
    tts.mkdir()
    source = tmp_path / "source.mp4"
    source.touch()
    old = tmp_path / "tmp" / "audio_dubbing.wav"
    old.parent.mkdir()
    old.write_bytes(b"previous audio")
    sf.write(tts / "0001.wav", np.full(16000, 0.25, dtype=np.float32), 8000)
    draft = tmp_path / "translation.zh.json"
    draft.write_text(json.dumps({"translation": [{"start_time": 900, "end_time": 990, "dst": "译文"}]}), encoding="utf-8")
    monkeypatch.setattr(audio, "probe_video_duration", lambda _: 1.0)
    with pytest.raises(RuntimeError, match="第 1 句"):
        audio.merge_tts_audio(draft, tts, tmp_path, video_file=source)
    assert old.read_bytes() == b"previous audio"
    assert not (tmp_path / "metadata" / "timings.json").exists()
