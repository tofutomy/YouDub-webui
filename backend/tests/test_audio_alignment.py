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
    sf.write(tts / "0001.wav", np.full(120000, 0.25, dtype=np.float32), 8000)
    draft = tmp_path / "translation.zh.json"
    draft.write_text(json.dumps({"translation": [{"start_time": 900, "end_time": 990, "dst": "译文"}]}), encoding="utf-8")
    monkeypatch.setattr(audio, "probe_video_duration", lambda _: 1.0)
    with pytest.raises(RuntimeError, match="第 1 句"):
        audio.merge_tts_audio(draft, tts, tmp_path, video_file=source)
    assert old.read_bytes() == b"previous audio"
    assert not (tmp_path / "metadata" / "timings.json").exists()



def test_short_tail_borrows_from_previous_lines_and_preserves_video_end(monkeypatch, tmp_path):
    tts = tmp_path / "tts"
    tts.mkdir()
    source = tmp_path / "source.mp4"
    source.touch()
    draft = tmp_path / "translation.zh.json"
    rows = [
        {"start_time": 120, "end_time": 1612, "dst": "那么，我们下期再见。"},
        {"start_time": 1663, "end_time": 1980, "dst": "回见！"},
        {"start_time": 1980, "end_time": 2313, "dst": "再见！"},
    ]
    draft.write_text(json.dumps({"translation": rows}), encoding="utf-8")
    rate = 8000
    for index, duration in enumerate((1995, 960, 644), start=1):
        sf.write(tts / f"{index:04d}.wav", np.full(duration * 8, 0.25, dtype=np.float32), rate)
    monkeypatch.setattr(audio, "probe_video_duration", lambda _: 2.3)
    _, timings = audio.merge_tts_audio(draft, tts, tmp_path, video_file=source)
    timing = json.loads(timings.read_text(encoding="utf-8"))
    assert timing["tail_freeze_ms"] == 0
    aligned = timing["translation"]
    assert [row["dst"] for row in aligned] == [row["dst"] for row in rows]
    assert all(a["actual_end_time"] <= b["actual_start_time"] for a, b in zip(aligned, aligned[1:]))
    assert all(0 < row["actual_end_time"] - row["actual_start_time"] for row in aligned)
    for original, result in zip(rows, aligned, strict=True):
        assert -audio.RECOVERY_LEAD_MS <= result["actual_start_time"] - original["start_time"] <= audio.MAX_LAG_MS
    assert aligned[1]["actual_start_time"] < rows[1]["start_time"]
    assert aligned[2]["actual_end_time"] <= 2200



def test_following_line_recovers_a_small_delay_at_preferred_tempo(monkeypatch, tmp_path):
    tts = tmp_path / "tts"
    tts.mkdir()
    source = tmp_path / "source.mp4"
    source.touch()
    draft = tmp_path / "translation.zh.json"
    rows = [
        {"start_time": 0, "end_time": 500, "dst": "第一句"},
        {"start_time": 500, "end_time": 1500, "dst": "第二句"},
        {"start_time": 1500, "end_time": 2200, "dst": "第三句"},
    ]
    draft.write_text(json.dumps({"translation": rows}), encoding="utf-8")
    rate = 8000
    for index, duration in enumerate((1000, 900, 500), start=1):
        sf.write(tts / f"{index:04d}.wav", np.full(duration * 8, 0.25, dtype=np.float32), rate)
    monkeypatch.setattr(audio, "probe_video_duration", lambda _: 2.4)
    monkeypatch.setattr(audio, "_fit_clip", lambda clip, _, __, frames: clip[:frames])
    _, timings = audio.merge_tts_audio(draft, tts, tmp_path, video_file=source)
    aligned = json.loads(timings.read_text(encoding="utf-8"))["translation"]
    assert aligned[1]["actual_start_time"] > rows[1]["start_time"]
    assert aligned[1]["actual_end_time"] <= rows[1]["end_time"]
    assert aligned[2]["actual_start_time"] == rows[2]["start_time"]
