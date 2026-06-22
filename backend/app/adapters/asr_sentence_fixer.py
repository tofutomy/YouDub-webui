from __future__ import annotations

import json
import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Filler-word filter — multi-language (en / zh / ja)
# ---------------------------------------------------------------------------

# Multi-character filler phrases (longest-first so shorter substrings don't
# eat parts of longer phrases).
_FILLER_PHRASES: list[str] = sorted([
    # English
    "you know", "i mean", "sort of", "kind of",
    # Chinese
    "就是说", "怎么说", "这样子", "就是说呢", "那个那个", "这个这个",
    # Japanese
    "あのう", "ええと", "あのー", "えーと", "そうですね", "なんか",
    "はぁ", "ふぅ", "はぁー", "うぅ", "あぁ",
], key=len, reverse=True)

# Word-level fillers (English / romanised interjections).
_FILLER_WORDS: set[str] = {
    "um", "uh", "ah", "er", "hmm", "mm", "mmm", "Mmhmm", "hm", "huh", "eh", "oh",
    "ugh", "ahem", "erm", "like", "well", "so", "actually",
    "basically", "literally", "right", "okay", "yeah", "yep", "yup",
    "nope", "nah", "mhm", "pff", "tsk", "shh", "whoa", "wow",
    "ooh", "aah", "haha", "hehe", "hoho", "lala", "dada", "baba",
    "gaga", "dunno", "gonna", "wanna", "gotta", "kinda", "sorta",
    "yay", "boo", "phew", "duh", "meh", "bleh", "blah", "wah",
    "eww", "yuck", "yum", "ouch", "oops", "whoops", "huhuhu",
}

# CJK single-character fillers — each char is one token.
_FILLER_CHARS_CJK: set[str] = set(
    # Chinese
    "嗯啊呃哦噢唔哼哈呵哎唉呀哟嘛呢吧哇咦嘿诶呐咯咧啵嘞喽喔哒咩喵呱"
    "咔嘭咚啪吱嘶嘻嘘噗咻嘣嘎啦嘤咿吖嗷咕嘟哔啵呵嘛嘿呐呗嘞啰咯噢嚯"
    # Japanese hiragana common fillers
    "あのえうんまさはいへふーらおやれわをがぎぐげござじずぜぞだぢづでど"
    "ばびぶべぼぱぴぷぺぽゃゅょっ"
)

# Punctuation / symbol regex (Unicode + CJK punctuation).
# Python's built-in re doesn't support \p{…} — use explicit ranges.
_PUNCT_RE = re.compile(
    r"[\s"
    r"\u0020-\u002F"   # ASCII punctuation & symbols  !"#$%&'()*+,-./
    r"\u003A-\u0040"   # :;<=>?@
    r"\u005B-\u0060"   # [\]^_`
    r"\u007B-\u007E"   # {|}~
    r"\u00A0-\u00BF"   # Latin-1 punctuation
    r"\u2000-\u206F"   # General Punctuation
    r"\u3000-\u303F"   # CJK Symbols and Punctuation
    r"\uFE30-\uFE4F"   # CJK Compatibility Forms
    r"\uFF01-\uFF0F"   # Fullwidth ASCII variants
    r"\uFF1A-\uFF20"
    r"\uFF3B-\uFF40"
    r"\uFF5B-\uFF65"
    r"]+",
    re.UNICODE,
)


def _strip_punctuation(text: str) -> str:
    """Remove punctuation / whitespace / symbols, keep only word characters."""
    return _PUNCT_RE.sub(" ", text).strip()


def _is_all_filler(text: str) -> bool:
    """Return True when *every* content token is a filler word/char.

    Punctuation is ignored so that ``嗯。`` or ``um,`` are considered 100 % filler.
    """
    cleaned = _strip_punctuation(text)
    if not cleaned:
        return True  # only punctuation → treat as filler

    # 1. Remove multi-char filler phrases first.
    for phrase in _FILLER_PHRASES:
        cleaned = cleaned.replace(phrase, " ")

    # 2. Split into tokens.
    parts = cleaned.split()
    tokens: list[str] = []
    for part in parts:
        # If every character is CJK, split into individual chars.
        cjk = [c for c in part if _is_cjk(c)]
        if len(cjk) == len(part):
            tokens.extend(cjk)
        else:
            tokens.append(part.lower())

    if not tokens:
        return True

    return all(t in _FILLER_WORDS or t in _FILLER_CHARS_CJK for t in tokens)


def _is_cjk(ch: str) -> bool:
    cp = ord(ch)
    return (
        0x4E00 <= cp <= 0x9FFF    # CJK Unified
        or 0x3400 <= cp <= 0x4DBF  # CJK Extension A
        or 0x3040 <= cp <= 0x309F  # Hiragana
        or 0x30A0 <= cp <= 0x30FF  # Katakana
    )


def _filter_fillers(utterances: list[dict], log_path: Path | None = None) -> list[dict]:
    """Remove utterances that consist entirely of filler words / punctuation.

    Returns the filtered list and, when *log_path* is given, writes a JSON
    record of every removed sentence.
    """
    kept: list[dict] = []
    removed: list[dict] = []
    for idx, u in enumerate(utterances):
        if _is_all_filler(u.get("text", "")):
            removed.append({
                "original_index": idx,
                "text": u["text"],
                "start_time": u.get("start_time"),
                "end_time": u.get("end_time"),
            })
        else:
            kept.append(u)

    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(
            json.dumps(
                {
                    "original_count": len(utterances),
                    "kept_count": len(kept),
                    "filtered_count": len(removed),
                    "removed": removed,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    return kept


def _start_pad(idx: int, utts: list, start_pad: int, end_pad: int, min_gap: int) -> int:
    orig_start = utts[idx]["start_time"]
    if idx == 0:
        return max(0, orig_start - start_pad)

    prev_end = utts[idx - 1]["end_time"]
    gap = orig_start - prev_end
    total = start_pad + end_pad

    if gap >= total + min_gap:
        return orig_start - start_pad
    if gap > min_gap:
        share = int((gap - min_gap) * start_pad / total)
        return orig_start - share
    return prev_end + gap // 2


def _end_pad(idx: int, utts: list, duration: int, start_pad: int, end_pad: int, min_gap: int) -> int:
    orig_end = utts[idx]["end_time"]
    if idx == len(utts) - 1:
        return min(duration, orig_end + end_pad) if duration else orig_end + end_pad

    next_start = utts[idx + 1]["start_time"]
    gap = next_start - orig_end
    total = start_pad + end_pad

    if gap >= total + min_gap:
        return orig_end + end_pad
    if gap > min_gap:
        share = int((gap - min_gap) * end_pad / total)
        return orig_end + share
    return orig_end + gap // 2


def _apply_padding(utts: list, duration: int, start_pad: int, end_pad: int) -> list:
    if not utts:
        return utts

    min_gap = 50
    result = []
    for idx in range(len(utts)):
        new_start = _start_pad(idx, utts, start_pad, end_pad, min_gap)
        new_end = _end_pad(idx, utts, duration, start_pad, end_pad, min_gap)
        clamped_end = min(duration, new_end) if duration else new_end
        result.append({
            **utts[idx],
            "start_time": max(0, new_start),
            "end_time": clamped_end,
        })
    return result


def _normalize(utterances: list) -> list:
    return [
        {"text": u["text"].strip(), "start_time": u["start_time"], "end_time": u["end_time"]}
        for u in utterances if u.get("text", "").strip()
    ]


def fix_asr_sentences(asr_file: Path, session: Path,
                     start_pad: int = 100, end_pad: int = 300,
                     language: str = "en",
                     filter_fillers: bool = False) -> Path:
    output_file = session / "metadata" / "asr_fixed.json"
    if output_file.exists():
        return output_file

    data = json.loads(Path(asr_file).read_text(encoding="utf-8"))
    utterances = data["result"]["utterances"]
    duration = data.get("audio_info", {}).get("duration", 0)

    new_utts = _normalize(utterances)
    if not new_utts:
        raise RuntimeError("ASR result has no utterances.")

    # ---- filler-word filter (before padding) ----
    filtered_count = 0
    if filter_fillers:
        before = len(new_utts)
        filter_log = session / "metadata" / "filtered_sentences.json"
        new_utts = _filter_fillers(new_utts, log_path=filter_log)
        filtered_count = before - len(new_utts)
        if not new_utts:
            raise RuntimeError("All utterances were filtered out as filler words.")
    # --------------------------------------------

    padded = _apply_padding(new_utts, duration, start_pad, end_pad)
    payload = {
        "audio_info": data.get("audio_info", {}),
        "result": {"text": data["result"].get("text", ""), "utterances": padded},
    }
    output_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_file
