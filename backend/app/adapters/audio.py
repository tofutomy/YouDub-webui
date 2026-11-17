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
from .ffmpeg import probe_video_duration

MAX_LAG_MS = 1000
PREFERRED_TEMPO = 1.55
PLANNING_TEMPO = 2.8
MAX_TEMPO = 3.0
TAIL_GUARD_MS = 100
TEMPO_PAD_MS = 45
TEMPO_MARGIN_MS = 20
ALIGNMENT_VERSION = 1


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


def _sync_error(index: int, segment: dict) -> RuntimeError:
    start = segment["start_time"] / 1000
    return RuntimeError(
        f"第 {index + 1} 句（原片 {int(start // 60):02d}:{start % 60:06.3f}）"
        "配音在允许的延迟和语速内放不下。请缩短该处译文或重新生成该处配音。"
    )


def _clip_budget(frames: int, sample_rate: int, tempo: float) -> int:
    pad = round(TEMPO_PAD_MS * sample_rate / 1000)
    margin = round(TEMPO_MARGIN_MS * sample_rate / 1000)
    return min(frames, math.ceil((frames + pad) / tempo) + margin)


def _latest_ends(translation: list[dict], lengths: list[int], sample_rate: int, video_end: int) -> list[int]:
    lag = round(MAX_LAG_MS * sample_rate / 1000)
    latest_ends = [0] * len(translation)
    next_latest_start = video_end
    # 从片尾往前给每句预留最短可播的时间，防止前面的长句吃掉后续的时间。
    for index in reversed(range(len(translation))):
        item = translation[index]
        start = max(0, math.floor(item["start_time"] * sample_rate / 1000))
        source_end = math.floor(item["end_time"] * sample_rate / 1000)
        end = min(next_latest_start, source_end + lag)
        latest_ends[index] = end
        next_latest_start = min(start + lag, end - _clip_budget(lengths[index], sample_rate, PLANNING_TEMPO))
        if next_latest_start < start:
            raise _sync_error(index, item)
    return latest_ends


def _fit_clip(clip: np.ndarray, source_rate: int, sample_rate: int, frames: int) -> np.ndarray | None:
    natural = math.ceil(len(clip) * sample_rate / source_rate)
    if natural <= frames and source_rate == sample_rate:
        return clip

    pad = round(TEMPO_PAD_MS * source_rate / 1000) if natural > frames else 0
    data = np.pad(clip, (0, pad)).astype("<f4", copy=False).tobytes()
    safety = round(0.01 * sample_rate)
    tempo = min(MAX_TEMPO, max(1.0, (len(clip) + pad) * sample_rate / source_rate / max(1, frames - safety)))
    for _ in range(4):
        factors = (math.sqrt(tempo),) * 2 if tempo > 2 else (tempo,)
        filters = ",".join(f"atempo={factor:.8f}" for factor in factors)
        result = subprocess.run(
            [ffmpeg_binary(), "-nostdin", "-v", "error", "-f", "f32le", "-ar", str(source_rate), "-ac", "1",
             "-i", "pipe:0", "-af", filters, "-ar", str(sample_rate), "-f", "f32le", "pipe:1"],
            input=data, capture_output=True, check=True,
        )
        audio = np.frombuffer(result.stdout, dtype="<f4")
        if 0 < len(audio) <= frames:
            return audio
        if not len(audio) or tempo >= MAX_TEMPO:
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
    latest_ends = _latest_ends(translation, lengths, sample_rate, video_frames - guard)

    with TemporaryDirectory(prefix="dubbing-", dir=dubbing_file.parent) as staging:
        scratch_dub = Path(staging) / dubbing_file.name
        scratch_timing = Path(staging) / timings_file.name
        with sf.SoundFile(scratch_dub, "w", sample_rate, 1) as output:
            cursor = 0
            for index, (item, file, span, raw_frames, latest_end) in enumerate(
                zip(translation, tts_files, spans, lengths, latest_ends, strict=True)
            ):
                start = max(cursor, int(item["start_time"] * sample_rate // 1000))
                source_span = max(0, int((item["end_time"] - item["start_time"]) * sample_rate // 1000))
                preferred = min(raw_frames, max(source_span, _clip_budget(raw_frames, sample_rate, PREFERRED_TEMPO)))
                budget = min(preferred, latest_end - start)
                if budget < 1:
                    raise _sync_error(index, item)
                offset, size, source_rate = span
                clip, _ = sf.read(file, start=offset, frames=size, dtype="float32", always_2d=True)
                audio = _fit_clip(clip.mean(axis=1), source_rate, sample_rate, budget)
                if audio is None:
                    raise _sync_error(index, item)
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

        scratch_timing.write_text(
            json.dumps({"alignment_version": ALIGNMENT_VERSION, "translation": translation}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        # 时间轴作为最后提交的产物，重跑时不会复用只写了一半的音轨。
        timings_file.unlink(missing_ok=True)
        scratch_dub.replace(dubbing_file)
        scratch_timing.replace(timings_file)
    return dubbing_file, timings_file
