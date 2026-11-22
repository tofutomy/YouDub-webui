from __future__ import annotations

import json
import subprocess

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
        return np.full(frames, 0.25, dtype=np.float32), len(clip) / frames

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


@pytest.mark.parametrize("first_ms, action", [(3060, "extreme_tempo"), (15000, "skipped")])
def test_extreme_tempo_before_skipping_and_next_line_keeps_its_audio(monkeypatch, tmp_path, first_ms, action):
    tts = tmp_path / "tts"
    tts.mkdir()
    source = tmp_path / "source.mp4"
    source.touch()
    rate = 8000
    sf.write(tts / "0001.wav", np.full(first_ms * 8, 0.25, dtype=np.float32), rate)
    sf.write(tts / "0002.wav", np.full(rate // 2, -0.5, dtype=np.float32), rate)
    draft = tmp_path / "translation.zh.json"
    rows = [{"start_time": 0, "end_time": 100, "dst": "拥挤的句子"},
            {"start_time": 1500, "end_time": 2000, "dst": "后面的句子"}]
    draft.write_text(json.dumps({"translation": rows}), encoding="utf-8")
    monkeypatch.setattr(audio, "probe_video_duration", lambda _: 3.0)
    with pytest.raises(audio._AlignmentError):
        audio._alignment_windows(rows, [first_ms * 8, rate // 2], rate, 23200,
                                 audio.RECOVERY_LEAD_MS, audio.PLANNING_TEMPO, tail_freeze_ms=3000)
    dub, timings = audio.merge_tts_audio(draft, tts, tmp_path, video_file=source)
    timeline = json.loads(timings.read_text(encoding="utf-8"))
    first, second = timeline["translation"]
    anomaly, = timeline["anomalies"]
    assert anomaly == {"sentence": 1, "start_time": 0, "end_time": 100, "action": action,
                       **({"tempo": first["tempo"]} if action == "extreme_tempo" else {})}
    assert second["actual_start_time"] == 1500
    samples, _ = sf.read(dub)
    assert np.allclose(samples[12000:], -0.5)
    subtitle = ffmpeg.write_srt(timings, tmp_path).read_text(encoding="utf-8")
    assert "后面的句子" in subtitle
    if action == "skipped":
        assert first["dubbing_skipped"] and not samples[:12000].any()
        assert "拥挤的句子" not in subtitle
    else:
        assert audio.PLANNING_TEMPO < first["tempo"] <= audio.MAX_TEMPO
        assert samples[:8800].any() and "拥挤的句子" in subtitle


def test_encoder_error_preserves_cached_dubbing(monkeypatch, tmp_path):
    tts = tmp_path / "tts"
    tts.mkdir()
    sf.write(tts / "0001.wav", np.full(16000, 0.25), 8000)
    source = tmp_path / "source.mp4"
    source.touch()
    draft = tmp_path / "translation.zh.json"
    draft.write_text(json.dumps({"translation": [{"start_time": 0, "end_time": 100}]}), encoding="utf-8")
    old = tmp_path / "tmp" / "audio_dubbing.wav"
    old.parent.mkdir()
    old.write_bytes(b"previous audio")
    monkeypatch.setattr(audio, "probe_video_duration", lambda _: 1.0)

    def fail_encoding(*args, **kwargs):
        raise subprocess.CalledProcessError(1, "ffmpeg")

    monkeypatch.setattr(audio.subprocess, "run", fail_encoding)
    with pytest.raises(subprocess.CalledProcessError):
        audio.merge_tts_audio(draft, tts, tmp_path, video_file=source)
    assert old.read_bytes() == b"previous audio"
    assert not (tmp_path / "metadata" / "timings.json").exists()


def test_skips_only_after_filter_exhausts_full_window_at_max_tempo(monkeypatch, tmp_path):
    rate = 8000
    sf.write(tmp_path / "0001.wav", np.full(rate * 2, 0.25), rate)
    sf.write(tmp_path / "0002.wav", np.full(rate // 2, -0.5), rate)
    source = tmp_path / "source.mp4"
    source.touch()
    draft = tmp_path / "translation.zh.json"
    rows = [{"start_time": 0, "end_time": 700, "dst": "第一句"},
            {"start_time": 2000, "end_time": 2500, "dst": "第二句"}]
    draft.write_text(json.dumps({"translation": rows}), encoding="utf-8")
    monkeypatch.setattr(audio, "probe_video_duration", lambda _: 3.0)
    tempos = []

    def oversized_output(command, **kwargs):
        filters = command[command.index("-af") + 1].split(",")
        tempos.append(np.prod([float(f.removeprefix("atempo=")) for f in filters]))
        return subprocess.CompletedProcess(command, 0, stdout=np.zeros(2 * rate, dtype="<f4").tobytes())

    monkeypatch.setattr(audio.subprocess, "run", oversized_output)
    dub, timings = audio.merge_tts_audio(draft, tmp_path, tmp_path, video_file=source)
    timeline = json.loads(timings.read_text(encoding="utf-8"))
    # 首选窗口和完整窗口都必须真正试过极限语速。
    assert sum(np.isclose(tempos, audio.MAX_TEMPO)) == 2
    assert timeline["anomalies"] == [{"sentence": 1, "start_time": 0, "end_time": 700, "action": "skipped"}]
    assert timeline["translation"][1]["actual_start_time"] == 2000
    samples, _ = sf.read(dub)
    assert not samples[:2 * rate].any()
    assert np.allclose(samples[2 * rate:], -0.5)



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
    monkeypatch.setattr(audio, "_fit_clip", lambda clip, _, __, frames: (clip[:frames], len(clip) / frames))
    _, timings = audio.merge_tts_audio(draft, tts, tmp_path, video_file=source)
    aligned = json.loads(timings.read_text(encoding="utf-8"))["translation"]
    assert aligned[1]["actual_start_time"] > rows[1]["start_time"]
    assert aligned[1]["actual_end_time"] <= rows[1]["end_time"]
    assert aligned[2]["actual_start_time"] == rows[2]["start_time"]
