"""下载 YouTube 原始英文字幕和封面，并对齐已有转录校对用词。"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from difflib import SequenceMatcher
from html import unescape
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.parse import parse_qs, urlsplit

from yt_dlp.networking.exceptions import RequestError

from ..config import ffmpeg_binary
from ..glossary import Glossary

CAPTION = Path("metadata/source.en.vtt")
CAPTION_INFO = Path("metadata/source.en.info.json")
COVER = Path("media/cover_source.jpg")
_MAX_CAPTION_BYTES = 4_000_000
_MAX_COVER_BYTES = 12_000_000
_TIMESTAMP = r"(?:\d{2,}:)?\d{2}:\d{2}\.\d{3}"
_CUE = re.compile(rf"^({_TIMESTAMP})\s+-->\s+({_TIMESTAMP})(?:\s|$)")
_INLINE = re.compile(rf"<{_TIMESTAMP}>")
_WORD = re.compile(r"[a-zA-Z]+(?:['’][a-zA-Z]+)*|\d+(?:[.,]\d+)*")
_TAG = re.compile(r"<[^>]+>")


@dataclass(frozen=True)
class Cue:
    start: int
    end: int
    text: str


def _time_ms(value: str) -> int:
    *clock, seconds = value.split(":")
    whole, ms = seconds.split(".")
    clock = [0, *map(int, clock)] if len(clock) == 1 else list(map(int, clock))
    return ((clock[0] * 60 + clock[1]) * 60 + int(whole)) * 1000 + int(ms)


def read_cues(text: str, *, automatic: bool = False) -> list[Cue]:
    text = text.lstrip("\ufeff").replace("\r\n", "\n")
    if not text.startswith("WEBVTT"):
        return []
    cues = []
    for block in re.split(r"\n[ \t]*\n", text):
        lines = block.splitlines()
        timing = next((i for i, line in enumerate(lines[:2]) if _CUE.match(line.strip())), None)
        if timing is None:
            continue
        match = _CUE.match(lines[timing].strip())
        start, end = map(_time_ms, match.groups())
        body = lines[timing + 1:]
        timed_lines = [line for line in body if _INLINE.search(line)] if automatic else []
        if timed_lines:
            # 滚动字幕会重复上一行，只取带有新词时间戳的行。
            body = timed_lines
        elif automatic and end - start <= 50:
            continue
        cleaned = " ".join(unescape(_TAG.sub("", " ".join(body))).split())
        if end <= start or not _WORD.search(cleaned):
            continue
        if cues and cues[-1].text.casefold() == cleaned.casefold() and start <= cues[-1].end + 150:
            continue
        cues.append(Cue(start, end, cleaned))
    return sorted(cues, key=lambda cue: cue.start)


def _save(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, prefix=".source-", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        temporary.write_bytes(content)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _fetch(ydl, url: str, maximum: int) -> bytes:
    with ydl.urlopen(url) as response:
        data = response.read(maximum + 1)
    if not data or len(data) > maximum:
        raise ValueError("无效的原视频附属文件")
    return data


def _english_vtt(info):
    for kind, key in (("manual", "subtitles"), ("automatic", "automatic_captions")):
        tracks = info.get(key) or {}
        languages = [name for name in tracks if re.fullmatch(r"en(?:[-_][a-zA-Z0-9]+)*", name, re.I)]
        languages.sort(key=lambda name: (name.lower() not in {"en", "en-orig"}, name.lower()))
        seen = set()
        for language in languages:
            for track in tracks[language]:
                url = track.get("url", "")
                if (track.get("ext") != "vtt" or not url or url in seen
                        or "tlang" in parse_qs(urlsplit(url).query, keep_blank_values=True)):
                    continue
                seen.add(url)
                yield kind, language, url


def _download_caption(ydl, info, session: Path) -> None:
    if (session / CAPTION).is_file() and (session / CAPTION_INFO).is_file():
        return
    for kind, language, url in _english_vtt(info):
        try:
            raw = _fetch(ydl, url, _MAX_CAPTION_BYTES)
            if not read_cues(raw.decode("utf-8-sig"), automatic=kind == "automatic"):
                continue
            _save(session / CAPTION, raw)
            _save(session / CAPTION_INFO, json.dumps({"kind": kind, "language": language}).encode())
            return
        except (RequestError, OSError, ValueError):
            continue


def _jpeg(raw: bytes) -> bytes:
    if raw.startswith(b"\xff\xd8"):
        return raw
    process = subprocess.run(
        [ffmpeg_binary(), "-v", "error", "-i", "pipe:0", "-frames:v", "1", "-f", "image2pipe",
         "-vcodec", "mjpeg", "-q:v", "2", "pipe:1"],
        input=raw, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=12, check=True,
    )
    if not process.stdout.startswith(b"\xff\xd8"):
        raise ValueError("封面文件无法解码")
    return process.stdout


def _download_cover(ydl, info, session: Path) -> None:
    if (session / COVER).is_file():
        return
    thumbs = [item for item in info.get("thumbnails", []) if item.get("url")]
    if not thumbs and info.get("thumbnail"):
        thumbs.append({"url": info["thumbnail"]})
    thumbs.sort(key=lambda thumb: (
        (thumb.get("width") or 0) * (thumb.get("height") or 0),
        urlsplit(thumb["url"]).path.lower().endswith((".jpg", ".jpeg")),
    ), reverse=True)
    for thumb in thumbs[:5]:
        try:
            _save(session / COVER, _jpeg(_fetch(ydl, thumb["url"], _MAX_COVER_BYTES)))
            return
        except (RequestError, OSError, ValueError, subprocess.SubprocessError):
            continue


def download_youtube_assets(ydl, info, session: Path) -> None:
    _download_caption(ydl, info, session)
    _download_cover(ydl, info, session)


def _tokens(text: str):
    return [word.group().casefold().replace("’", "'") for word in _WORD.finditer(text)]


def _known_caption_term(ref: list[str], index: int, glossary: Glossary) -> bool:
    # 自动字幕的改单词必须属于本任务词库中的一个真实术语。
    sample = " ".join(ref[max(0, index - 5): index + 6])
    names = {key for key, _ in glossary.names.search(sample, set(sample.split()))} | glossary.custom.keys()
    for left in range(max(0, index - 5), index + 1):
        for right in range(index + 1, min(len(ref), index + 6) + 1):
            if " ".join(ref[left:right]) in names:
                return True
    return False


def correct_terms(utterances, cues: list[Cue], *, automatic: bool, glossary: Glossary):
    texts = [utterance["text"] for utterance in utterances]
    words = [_tokens(text) for text in texts]
    cue_index = 0
    corrections = []
    for index, utterance in enumerate(utterances):
        start, end = utterance["start_time"], utterance["end_time"]
        while cue_index < len(cues) and cues[cue_index].end < start - 1200:
            cue_index += 1
        reference = []
        for position in range(cue_index, len(cues)):
            cue = cues[position]
            if cue.start > end + 1200:
                break
            reference.extend(match.group() for match in _WORD.finditer(cue.text))
        if not reference or not words[index]:
            continue
        before = words[index - 1][-8:] if index else []
        after = words[index + 1][:8] if index + 1 < len(words) else []
        draft = before + words[index] + after
        caption = [word.casefold().replace("’", "'") for word in reference]
        opcodes = SequenceMatcher(None, draft, caption, autojunk=False).get_opcodes()
        locations = list(_WORD.finditer(texts[index]))
        edits = []
        for previous, op, following in zip(opcodes, opcodes[1:], opcodes[2:]):
            kind, a, b, c, d = op
            if (kind != "replace" or b - a != 1 or d - c != 1 or not len(before) <= a < len(before) + len(locations)
                    or previous[0] != "equal" or previous[2] - previous[1] < 2
                    or following[0] != "equal" or following[2] - following[1] < 2):
                continue
            if (not draft[a].isalpha() or not caption[c].isalpha()
                    or SequenceMatcher(None, draft[a], caption[c]).ratio() < 0.69):
                continue
            if automatic and not _known_caption_term(caption, c, glossary):
                continue
            old_word = locations[a - len(before)]
            edits.append((old_word.start(), old_word.end(), reference[c]))
            corrections.append({"start_time": start, "from": old_word.group(), "to": reference[c]})
        for left, right, new in reversed(edits):
            utterance["text"] = utterance["text"][:left] + new + utterance["text"][right:]
    return corrections


def apply_caption_corrections(path: Path, session: Path, glossary_id: str | None, custom: str) -> int:
    from ..glossary import select_glossary

    caption = session / CAPTION
    info = session / CAPTION_INFO
    if not caption.is_file() or not info.is_file():
        return 0
    try:
        kind = json.loads(info.read_text(encoding="utf-8")).get("kind")
        if kind not in {"manual", "automatic"}:
            return 0
        cues = read_cues(caption.read_text(encoding="utf-8-sig"), automatic=kind == "automatic")
        if not cues:
            return 0
        meta = json.loads((session / "metadata" / "ytdlp_info.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return 0
    result = json.loads(path.read_text(encoding="utf-8"))
    glossary = select_glossary(glossary_id, meta, custom)
    edits = correct_terms(result["result"]["utterances"], cues, automatic=kind == "automatic", glossary=glossary)
    if edits:
        result["result"]["text"] = " ".join(row["text"] for row in result["result"]["utterances"])
        result["source_caption_corrections"] = edits
        _save(path, json.dumps(result, ensure_ascii=False, indent=2).encode("utf-8"))
    return len(edits)
