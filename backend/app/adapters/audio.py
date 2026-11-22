from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable

import numpy as np
import soundfile as sf
from pydub import AudioSegment

from ..config import ffmpeg_binary
from .ffmpeg import MAX_TAIL_FREEZE_MS, probe_video_duration

MAX_LAG_MS = 1000
RECOVERY_LEAD_MS = 300
RECOVERY_TEMPO = 2.2
PREFERRED_TEMPO = 1.55
PLANNING_TEMPO = 2.8
MAX_TEMPO = 3.0
TAIL_GUARD_MS = 100
TEMPO_PAD_MS = 45
TEMPO_MARGIN_MS = 20
ALIGNMENT_VERSION = 4


def split_audio_by_translation(vocals_file: Path, translation_file: Path, session: Path) -> Path:
    output_dir = session / "segments" / "vocals"
    output_dir.mkdir(parents=True, exist_ok=True)
    data = json.loads(translation_file.read_text(encoding="utf-8"))
    audio = AudioSegment.from_file(vocals_file)

    for index, item in enumerate(data["translation"], start=1):
        output_file = output_dir / f"{index:04d}.wav"
        if output_file.exists():
            continue
        start = max(0, int(item["start_time"]) - 80)
        end = min(len(audio), int(item["end_time"]) + 160)
        audio[start:end].export(output_file, format="wav")

    return output_dir


def _speech_span(file: Path) -> tuple[int, int, int]:
    samples, sample_rate = sf.read(file, dtype="float32", always_2d=True)
    level = np.abs(samples).max(axis=1)
    # 仅收掉片段首尾接近无声的部分，保留辅音和收尾余量。
    voiced = np.flatnonzero(level >= max(0.001, float(level.max(initial=0)) * 0.003))
    if not len(voiced):
        return 0, len(samples), sample_rate
    guard = round(sample_rate * 0.045)
    start = max(0, int(voiced[0]) - guard)
    end = min(len(samples), int(voiced[-1]) + 1 + guard)
    return start, end - start, sample_rate


class _AlignmentError(RuntimeError):
    pass


def _sync_error(index: int, segment: dict) -> _AlignmentError:
    start = segment["start_time"] / 1000
    return _AlignmentError(
        f"第 {index + 1} 句（原片 {int(start // 60):02d}:{start % 60:06.3f}）"
        "配音在允许的延迟和语速内放不下。请缩短该处译文或重新生成该处配音。"
    )


def _clip_budget(frames: int, sample_rate: int, tempo: float) -> int:
    pad = round(TEMPO_PAD_MS * sample_rate / 1000)
    margin = round(TEMPO_MARGIN_MS * sample_rate / 1000)
    return min(frames, math.ceil((frames + pad) / tempo) + margin)


def _alignment_windows(
    translation: list[dict], lengths: list[int], sample_rate: int, video_end: int,
    lead_ms: int, tempo: float, *, tail_freeze_ms: int = 0, allow_skips: bool = False,
) -> list[tuple[int, int, int]]:
    lag = round(MAX_LAG_MS * sample_rate / 1000)
    lead = round(lead_ms * sample_rate / 1000)
    windows = [(0, 0, 0)] * len(translation)
    next_start = video_end + round(tail_freeze_ms * sample_rate / 1000)
    # 从片尾倒推，为后续句子留出完整语音的空间。允许的提前量只在排不开时使用。
    for index in reversed(range(len(translation))):
        item = translation[index]
        start = max(0, math.floor(item["start_time"] * sample_rate / 1000))
        source_end = math.floor(item["end_time"] * sample_rate / 1000)
        # 定格只给尾句增加空间，其余句仍要在各自的同步窗口里结束。
        end = next_start if tail_freeze_ms and index == len(translation) - 1 else min(next_start, source_end + lag)
        latest_start = min(start + lag, end - _clip_budget(lengths[index], sample_rate, tempo))
        earliest = max(0, start - lead)
        if latest_start < earliest:
            if not allow_skips:
                raise _sync_error(index, item)
            # 不为放不下的句子挤占其他句子；实际混音时仍在剩余窗口尝试极限语速。
            windows[index] = start, earliest, end
            continue
        windows[index] = start, latest_start, end
        next_start = latest_start
    return windows


def _plan_alignment(
    translation: list[dict], lengths: list[int], sample_rate: int, video_end: int,
) -> list[tuple[int, int, int]]:
    attempts = (
        (0, PLANNING_TEMPO, 0),
        (RECOVERY_LEAD_MS, RECOVERY_TEMPO, 0),
        (RECOVERY_LEAD_MS, PLANNING_TEMPO, 0),
        (RECOVERY_LEAD_MS, RECOVERY_TEMPO, MAX_TAIL_FREEZE_MS),
        (RECOVERY_LEAD_MS, PLANNING_TEMPO, MAX_TAIL_FREEZE_MS),
    )
    for lead, tempo, freeze in attempts:
        try:
            return _alignment_windows(translation, lengths, sample_rate, video_end, lead, tempo, tail_freeze_ms=freeze)
        except _AlignmentError:
            continue
    return _alignment_windows(translation, lengths, sample_rate, video_end, RECOVERY_LEAD_MS, MAX_TEMPO,
                              tail_freeze_ms=MAX_TAIL_FREEZE_MS, allow_skips=True)


def _fit_clip(
    clip: np.ndarray, source_rate: int, sample_rate: int, frames: int,
) -> tuple[np.ndarray, float] | None:
    natural = math.ceil(len(clip) * sample_rate / source_rate)
    if natural <= frames and source_rate == sample_rate:
        return clip, 1.0

    pad = round(TEMPO_PAD_MS * source_rate / 1000) if natural > frames else 0
    data = np.pad(clip, (0, pad)).astype("<f4", copy=False).tobytes()
    safety = round(0.01 * sample_rate)
    tempo = min(MAX_TEMPO, max(1.0, (len(clip) + pad) * sample_rate / source_rate / max(1, frames - safety)))
    for attempt in range(4):
        if attempt == 3:
            tempo = MAX_TEMPO
        factors = (math.sqrt(tempo),) * 2 if tempo > 2 else (tempo,)
        filters = ",".join(f"atempo={factor:.8f}" for factor in factors)
        result = subprocess.run(
            [ffmpeg_binary(), "-nostdin", "-v", "error", "-f", "f32le", "-ar", str(source_rate), "-ac", "1",
             "-i", "pipe:0", "-af", filters, "-ar", str(sample_rate), "-f", "f32le", "pipe:1"],
            input=data, capture_output=True, check=True,
        )
        audio = np.frombuffer(result.stdout, dtype="<f4")
        if 0 < len(audio) <= frames:
            return audio, tempo
        if tempo >= MAX_TEMPO:
            return None
        tempo = min(MAX_TEMPO, tempo * max(1.025, len(audio) / max(1, frames - safety)))
    return None


def merge_tts_audio(
    translation_file: Path, tts_dir: Path, session: Path, *, video_file: Path | None = None,
    progress_callback: Callable[[int, str], None] | None = None,
) -> tuple[Path, Path]:
    dubbing_file = session / "tmp" / "audio_dubbing.wav"
    timings_file = session / "metadata" / "timings.json"
    source_video = video_file if video_file is not None else session / "media" / "video_source.mp4"
    dubbing_file.parent.mkdir(parents=True, exist_ok=True)
    timings_file.parent.mkdir(parents=True, exist_ok=True)

    translation = json.loads(translation_file.read_text(encoding="utf-8"))["translation"]
    if not translation:
        raise RuntimeError("未找到需要合成的配音片段。")
    tts_files = [tts_dir / f"{i:04d}.wav" for i in range(1, len(translation) + 1)]
    newest_input = max(path.stat().st_mtime_ns for path in (translation_file, source_video, *tts_files))
    if dubbing_file.exists() and timings_file.exists() and min(
        dubbing_file.stat().st_mtime_ns, timings_file.stat().st_mtime_ns,
    ) >= newest_input:
        try:
            previous = json.loads(timings_file.read_text(encoding="utf-8"))
        except ValueError:
            previous = {}
        if previous.get("alignment_version") == ALIGNMENT_VERSION:
            return dubbing_file, timings_file

    spans = [_speech_span(path) for path in tts_files]
    sample_rate = spans[0][2]
    lengths = [math.ceil(size * sample_rate / rate) for _, size, rate in spans]
    duration = probe_video_duration(source_video)
    video_frames = math.floor(duration * sample_rate)
    guard = min(round(TAIL_GUARD_MS * sample_rate / 1000), video_frames // 10)
    windows = _plan_alignment(translation, lengths, sample_rate, video_frames - guard)
    anomalies = []

    with TemporaryDirectory(prefix="dubbing-", dir=dubbing_file.parent) as staging:
        scratch_dub = Path(staging) / dubbing_file.name
        scratch_timing = Path(staging) / timings_file.name
        with sf.SoundFile(scratch_dub, "w", sample_rate, 1) as output:
            cursor = 0
            for index, (item, file, span, raw_frames, window) in enumerate(
                zip(translation, tts_files, spans, lengths, windows, strict=True)
            ):
                source_start, latest_start, latest_end = window
                start = max(cursor, min(source_start, latest_start))
                source_end = math.floor(item["end_time"] * sample_rate / 1000)
                # 前一句占用的时间优先用正常语速范围内的余量补回来。
                source_span = max(0, source_end - max(source_start, start))
                preferred = min(raw_frames, max(source_span, _clip_budget(raw_frames, sample_rate, PREFERRED_TEMPO)))
                budget = min(preferred, latest_end - start)
                fitted = None
                if budget > 0 and start <= source_start + round(MAX_LAG_MS * sample_rate / 1000):
                    offset, size, source_rate = span
                    clip, _ = sf.read(file, start=offset, frames=size, dtype="float32", always_2d=True)
                    mono = clip.mean(axis=1)
                    fitted = _fit_clip(mono, source_rate, sample_rate, budget)
                    if fitted is None and budget < latest_end - start:
                        # 变速滤镜的实际长度会波动。优先使用窗口余量，保留后续句的预留。
                        fitted = _fit_clip(mono, source_rate, sample_rate, latest_end - start)
                position = {"sentence": index + 1, "start_time": item["start_time"], "end_time": item["end_time"]}
                if fitted is None:
                    anomalies.append({**position, "action": "skipped"})
                    item["dubbing_skipped"] = True
                    if progress_callback:
                        progress_callback(round(100 * (index + 1) / len(translation)), f"第 {index + 1} 句无法对齐，已跳过")
                    continue
                audio, tempo = fitted
                item["tempo"] = round(tempo, 8)
                if tempo > PLANNING_TEMPO:
                    anomalies.append({**position, "action": "extreme_tempo", "tempo": item["tempo"]})
                gap = start - cursor
                silence = np.zeros(min(gap, sample_rate * 5), dtype=np.float32)
                while gap:
                    output.write(silence[:gap])
                    gap -= min(gap, len(silence))
                output.write(audio)
                cursor = start + len(audio)
                item["actual_start_time"] = round(start * 1000 / sample_rate)
                item["actual_end_time"] = round(cursor * 1000 / sample_rate)
                if progress_callback:
                    progress_callback(round(100 * (index + 1) / len(translation)),
                                      f"已对齐 {index + 1}/{len(translation)} 句配音")
            if cursor == 0:
                # 全部跳过时仍提供有效的静音输入，后续合成保留原片背景音。
                output.write(np.zeros(1, dtype=np.float32))

        extension = max(0, math.ceil((cursor + guard - video_frames) * 1000 / sample_rate))
        # 需要定格时至少补一秒，为视频帧取整和尾句收音留余量。
        tail_freeze_ms = max(1000, extension) if extension else 0
        scratch_timing.write_text(
            json.dumps({"alignment_version": ALIGNMENT_VERSION, "translation": translation,
                        "tail_freeze_ms": tail_freeze_ms, "anomalies": anomalies}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        # 时间轴作为最后提交的产物，重跑时不会复用只写了一半的音轨。
        timings_file.unlink(missing_ok=True)
        scratch_dub.replace(dubbing_file)
        scratch_timing.replace(timings_file)
    return dubbing_file, timings_file
