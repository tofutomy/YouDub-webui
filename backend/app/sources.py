from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .config import COOKIE_DIR
from .youtube import (
    is_bilibili_url,
    is_local_en_to_zh_url,
    is_local_ja_to_zh_url,
    is_local_zh_to_en_url,
    is_localdir_en_to_zh_url,
    is_localdir_ja_to_zh_url,
    is_localdir_zh_to_en_url,
    is_youtube_url,
)


from .adapters._lang_map import LANG_NAMES

DEFAULT_ASR_LANGUAGE = "en"
DEFAULT_TARGET_LANGUAGE = "zh"


@dataclass(frozen=True)
class SourceConfig:
    name: str
    matches: Callable[[str], bool]
    use_proxy: bool
    cookie_filename: str | None
    asr_language: str
    target_language: str

    @property
    def cookie_path(self) -> Path | None:
        if not self.cookie_filename:
            return None
        return COOKIE_DIR / self.cookie_filename

    @property
    def asr_language_name(self) -> str:
        return LANG_NAMES[self.asr_language]

    @property
    def target_language_name(self) -> str:
        return LANG_NAMES[self.target_language]


SOURCES: list[SourceConfig] = [
    SourceConfig(
        name="youtube",
        matches=is_youtube_url,
        use_proxy=True,
        cookie_filename="youtube.txt",
        asr_language="en",
        target_language="zh",
    ),
    SourceConfig(
        name="local",
        matches=is_local_en_to_zh_url,
        use_proxy=False,
        cookie_filename=None,
        asr_language="en",
        target_language="zh",
    ),
    SourceConfig(
        name="local",
        matches=is_local_zh_to_en_url,
        use_proxy=False,
        cookie_filename=None,
        asr_language="zh",
        target_language="en",
    ),
    SourceConfig(
        name="localdir",
        matches=is_localdir_en_to_zh_url,
        use_proxy=False,
        cookie_filename=None,
        asr_language="en",
        target_language="zh",
    ),
    SourceConfig(
        name="localdir",
        matches=is_localdir_zh_to_en_url,
        use_proxy=False,
        cookie_filename=None,
        asr_language="zh",
        target_language="en",
    ),
    SourceConfig(
        name="local",
        matches=is_local_ja_to_zh_url,
        use_proxy=False,
        cookie_filename=None,
        asr_language="ja",
        target_language="zh",
    ),
    SourceConfig(
        name="localdir",
        matches=is_localdir_ja_to_zh_url,
        use_proxy=False,
        cookie_filename=None,
        asr_language="ja",
        target_language="zh",
    ),
    SourceConfig(
        name="bilibili",
        matches=is_bilibili_url,
        use_proxy=False,
        cookie_filename="bilibili.txt",
        asr_language="zh",
        target_language="en",
    ),
]


def detect_source(url: str) -> SourceConfig:
    for source in SOURCES:
        if source.matches(url):
            return source
    raise ValueError(f"No source matches URL: {url}")


_OPPOSITE_LANG: dict[str, str] = {"en": "zh", "zh": "en", "ja": "zh"}


def _opposite_language(language: str) -> str:
    return _OPPOSITE_LANG.get(language, "zh")


def _task_languages(task: dict, source: SourceConfig) -> tuple[str, str]:
    asr_language = (task.get("asr_language") or "").strip()
    target_language = (task.get("target_language") or "").strip()
    if asr_language and target_language:
        return asr_language, target_language
    if asr_language:
        return asr_language, _opposite_language(asr_language)
    if target_language:
        return _opposite_language(target_language), target_language
    if source.name in ("local", "localdir"):
        return source.asr_language, source.target_language
    return DEFAULT_ASR_LANGUAGE, DEFAULT_TARGET_LANGUAGE


def get_source_for_task(task: dict) -> SourceConfig:
    """Get source config for a task, using task-stored translation direction."""
    source = detect_source(task["url"])
    asr_language, target_language = _task_languages(task, source)
    return SourceConfig(
        name=source.name,
        matches=source.matches,
        use_proxy=source.use_proxy,
        cookie_filename=source.cookie_filename,
        asr_language=asr_language,
        target_language=target_language,
    )
