from __future__ import annotations

import json
import math
import re
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

from ..config import ffmpeg_binary, ffprobe_binary

SUBTITLE_PUNCTUATION = {"，", ",", "；", ";", "：", ":", "。", "?", "？", "!", "！", "、"}
SUBTITLE_PROTECTED_PAIRS = {"《": "》", "（": "）", "【": "】", "「": "」", "『": "』"}
SUBTITLE_CLOSING_QUOTES = {'"', "'", "」", "』", "》", "）", "】", "\u201d", "\u2019", "]"}
SUBTITLE_MIN_FRAGMENT_LEN = 5
SUBTITLE_MIN_DURATION_MS = 200
SUBTITLE_DURATION_FLOOR_MS = 600


SUBTITLE_FONTS = {
    "zh": "Noto Sans CJK SC",
    "en": "Arial",
}

SUBTITLE_FONT_SIZES = {
    "zh": {"portrait": 12, "landscape": 24},
    "en": {"portrait": 9, "landscape": 18},
}


def _subtitle_style(font: str, size: int, margin_v: int) -> str:
    return (
        f"FontName={font},"
        f"FontSize={size},"
        "PrimaryColour=&H00FFFFFF,"
        "OutlineColour=&H00000000,"
        "BorderStyle=1,"
        "Outline=2,"
        "Alignment=2,"
        f"MarginV={margin_v}"
    )


def _srt_time(ms: int) -> str:
    hours = ms // 3_600_000
    ms -= hours * 3_600_000
    minutes = ms // 60_000
    ms -= minutes * 60_000
    seconds = ms // 1000
    millis = ms - seconds * 1000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def _split_protected(text: str) -> list[str]:
    segments: list[str] = []
    buf: list[str] = []
    inside = None
    for ch in text:
        if inside is None and ch in SUBTITLE_PROTECTED_PAIRS:
            inside = SUBTITLE_PROTECTED_PAIRS[ch]
            buf.append(ch)
            continue
        if inside is not None and ch == inside:
            inside = None
            buf.append(ch)
            continue
        if inside is None and ch in SUBTITLE_PUNCTUATION:
            chunk = "".join(buf).strip()
            if chunk:
                segments.append(chunk)
            buf.clear()
            continue
        buf.append(ch)
    tail = "".join(buf).strip()
    if tail:
        segments.append(tail)
    return segments


def _attach_closing_quotes(segments: list[str]) -> list[str]:
    fixed: list[str] = []
    for seg in segments:
        if seg and seg[0] in SUBTITLE_CLOSING_QUOTES and fixed:
            fixed[-1] = f"{fixed[-1]}{seg}".strip()
            continue
        fixed.append(seg.strip())
    return fixed


def _merge_short_fragments(segments: list[str]) -> list[str]:
    merged: list[str] = []
    i = 0
    while i < len(segments):
        cur = segments[i]
        if len(cur.strip()) < SUBTITLE_MIN_FRAGMENT_LEN and i + 1 < len(segments):
            segments[i + 1] = f"{cur}{segments[i + 1]}".strip()
            i += 1
            continue
        merged.append(cur)
        i += 1
    return merged


def _strip_trailing_punct(segments: list[str]) -> list[str]:
    cleaned: list[str] = []
    for item in segments:
        text = item.strip()
        if not text:
            continue
        if text.endswith(("，", ",", "。")):
            text = text[:-1]
        cleaned.append(re.sub(r"\s+", " ", text).strip())
    return cleaned


def split_subtitle_text(text: str) -> list[str]:
    original = (text or "").strip()
    if not original:
        return []
    segments = _split_protected(original)
    if not segments:
        return [original]
    segments = _attach_closing_quotes(segments)
    segments = _merge_short_fragments(segments)
    cleaned = _strip_trailing_punct(segments)
    return cleaned or [original]


def _allocate_durations(fragments: list[str], total_duration: int) -> list[int]:
    if len(fragments) == 1:
        return [total_duration]
    weights = [max(1, len(f.replace(" ", ""))) for f in fragments]
    total_weight = sum(weights)
    durations: list[int] = []
    remaining = total_duration
    minimum = SUBTITLE_MIN_DURATION_MS if total_duration >= SUBTITLE_DURATION_FLOOR_MS else SUBTITLE_MIN_DURATION_MS // 2
    minimum = min(minimum, total_duration // len(fragments))
    for index, weight in enumerate(weights[:-1]):
        share = round(total_duration * weight / total_weight)
        ceiling = remaining - minimum * (len(fragments) - index - 1)
        allocated = max(minimum, min(share, ceiling))
        durations.append(allocated)
        remaining -= allocated
    durations.append(remaining)
    return durations


def _segment_times(item: dict) -> tuple[int, int]:
    start = int(item.get("actual_start_time", item["start_time"]))
    end = int(item.get("actual_end_time", item["end_time"]))
    return start, end


def _dst_lang(translation: list[dict]) -> str:
    for item in translation:
        lang = item.get("dst_lang")
        if lang:
            return lang
    return "zh"


def _dst_text(item: dict) -> str:
    return item.get("dst") or item.get("zh") or ""


def write_srt(translation_file: Path, session: Path) -> Path:
    data = json.loads(translation_file.read_text(encoding="utf-8"))
    translation = data["translation"]
    dst_lang = _dst_lang(translation)
    output_file = session / "metadata" / f"subtitles.{dst_lang}.srt"
    lines: list[str] = []
    idx = 1
    for item in translation:
        if item.get("dubbing_skipped"):
            continue
        start, end = _segment_times(item)
        if end <= start:
            continue
        fragments = split_subtitle_text(_dst_text(item))
        if not fragments:
            continue
        if len(fragments) * 100 > end - start:
            fragments = [_dst_text(item).strip()]
        cursor = start
        for fragment, duration in zip(fragments, _allocate_durations(fragments, end - start)):
            lines.extend([str(idx), f"{_srt_time(cursor)} --> {_srt_time(cursor + duration)}", fragment, ""])
            cursor += duration
            idx += 1
    output_file.write_text("\n".join(lines), encoding="utf-8")
    return output_file


def probe_video_duration(video_file: Path) -> float:
    result = subprocess.run(
        [ffprobe_binary(), "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=duration:format=duration", "-of", "json", str(video_file)],
        capture_output=True, text=True, check=True,
    )
    data = json.loads(result.stdout)
    # 优先使用画面流的时长，封装总时长有时会被较长的原声拉长。
    for entry in [*data.get("streams", []), data.get("format", {})]:
        try:
            duration = float(entry.get("duration", "nan"))
        except (TypeError, ValueError):
            continue
        if math.isfinite(duration) and duration > 0:
            return duration
    raise RuntimeError("无法确定原视频画面的时长。")


def probe_video_size(video_file: Path) -> tuple[int, int] | None:
    result = subprocess.run(
        [
            ffprobe_binary(),
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "csv=p=0",
            str(video_file),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None

    lines = result.stdout.strip().splitlines()
    if not lines:
        return None
    parts = lines[0].split(",", maxsplit=1)
    if len(parts) != 2:
        return None
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        return None


def get_video_orientation(video_file: Path) -> str:
    size = probe_video_size(video_file)
    if size is None:
        return "landscape"
    width, height = size
    return "portrait" if height > width else "landscape"


def subtitle_style_for_orientation(orientation: str, font: str, lang: str = "zh") -> str:
    sizes = SUBTITLE_FONT_SIZES.get(lang, SUBTITLE_FONT_SIZES["zh"])
    margin_v = 70 if orientation == "portrait" else 5
    return _subtitle_style(font, size=sizes[orientation], margin_v=margin_v)


def _subtitle_filter_path(subtitle_file: Path, session: Path) -> str:
    try:
        return subtitle_file.resolve().relative_to(session.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError("Subtitle file must be inside the session directory.") from exc


def subtitle_filter(video_file: Path, subtitle_file: Path, session: Path) -> str:
    lang = subtitle_file.stem.rsplit(".", 1)[-1]
    font = SUBTITLE_FONTS.get(lang, "Arial")
    style = subtitle_style_for_orientation(get_video_orientation(video_file), font, lang)
    sub_path = _subtitle_filter_path(subtitle_file, session)
    return f"subtitles=filename='{sub_path}':force_style='{style}'"


def _find_or_generate_srt(session: Path, timings_file: Path) -> Path | None:
    # 成片字幕和配音使用同一份最终时间轴，兼容只有旧字幕的任务。
    if timings_file.exists():
        return write_srt(timings_file, session)
    return next((session / "metadata").glob("subtitles.*.srt"), None)


MAX_TAIL_FREEZE_MS = 3000


def merge_video(video_file: Path, dubbing_file: Path, bgm_file: Path, timings_file: Path, session: Path, *, add_subtitles: bool = True) -> Path:
    tmp_dir = session / "tmp"
    media_dir = session / "media"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    media_dir.mkdir(parents=True, exist_ok=True)
    final_video = media_dir / "video_final.mp4"
    if final_video.exists():
        last_render = final_video.stat().st_mtime_ns
        inputs = (video_file, dubbing_file, bgm_file, timings_file)
        if all(path.stat().st_mtime_ns <= last_render for path in inputs if path.exists()):
            return final_video

    session_dir = session.resolve()
    video_input = video_file.resolve()
    dubbing_input = dubbing_file.resolve()
    bgm_input = bgm_file.resolve()
    mixed_audio = tmp_dir / "audio_mixed.m4a"
    mixed_audio_output = mixed_audio.resolve()
    final_video_output = final_video.resolve()
    timeline = json.loads(timings_file.read_text(encoding="utf-8")) if timings_file.is_file() else {}
    tail_freeze_ms = timeline.get("tail_freeze_ms", 0)
    if type(tail_freeze_ms) is not int or not 0 <= tail_freeze_ms <= MAX_TAIL_FREEZE_MS:
        raise RuntimeError("片尾定格时长无效，请从「合并音频」级联重跑。")
    total_duration = 0.0
    if tail_freeze_ms or dubbing_input.is_file():
        total_duration = probe_video_duration(video_input) + tail_freeze_ms / 1000
    if dubbing_input.is_file():
        import soundfile as sf

        if sf.info(dubbing_input).duration > total_duration:
            raise RuntimeError("配音轨超过成片画面时长。请从「合并音频」级联重跑，避免截去片尾配音。")
    subprocess.run(
        [
            ffmpeg_binary(),
            "-y",
            "-i",
            str(dubbing_input),
            "-i",
            str(bgm_input),
            "-filter_complex",
            "[0:a]volume=1.0[a0];[1:a]volume=0.30[a1];[a0][a1]amix=inputs=2:duration=longest:normalize=0[aout]",
            "-map",
            "[aout]",
            "-c:a",
            "aac",
            str(mixed_audio_output),
        ],
        check=True,
    )

    # Build the final video command, conditionally including subtitles
    cmd = [
        ffmpeg_binary(),
        "-y",
        "-i",
        str(video_input),
        "-i",
        str(mixed_audio_output),
    ]

    video_filters = []
    if tail_freeze_ms:
        # 先延长画面再生成字幕，使新增帧仍按真实配音时间显示字幕。
        video_filters.append("tpad=stop_mode=clone:stop=-1")
        # 成片按时间截止，避免定格滤镜按帧四舍五入导致少补一帧。
        cmd.extend(["-t", f"{total_duration:.6f}"])
    if add_subtitles:
        subtitles = _find_or_generate_srt(session, timings_file)
        if subtitles:
            video_filters.append(subtitle_filter(video_input, subtitles, session_dir))
    if video_filters:
        cmd.extend(["-vf", ",".join(video_filters)])

    cmd.extend([
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "23",
        "-c:a",
        "aac",
        "-movflags",
        "+faststart",
        "-af",
        "apad",
        "-shortest",
    ])

    # 渲染失败时保留上一版成片，完成后再一次性替换。
    with TemporaryDirectory(prefix="render-", dir=media_dir) as staging:
        output = Path(staging).resolve() / final_video.name
        subprocess.run([*cmd, str(output)], check=True, cwd=session_dir)
        output.replace(final_video_output)
    return final_video
