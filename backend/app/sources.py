from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .adapters._lang_map import LANG_NAMES
from .config import COOKIE_DIR
from .languages import (
    DEFAULT_ASR_LANGUAGE,
    DEFAULT_TARGET_LANGUAGE,
    normalize_task_language_pair,
)
from .youtube import (
    is_bilibili_url,
    is_local_url_format,
    is_localdir_url_format,
    is_youtube_url,
    local_upload_direction,
    localdir_direction,
)


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
        asr_language=DEFAULT_ASR_LANGUAGE,
        target_language=DEFAULT_TARGET_LANGUAGE,
    ),
    SourceConfig(
        name="local",
        matches=is_local_url_format,
        use_proxy=False,
        cookie_filename=None,
        asr_language=DEFAULT_ASR_LANGUAGE,
        target_language=DEFAULT_TARGET_LANGUAGE,
    ),
    SourceConfig(
        name="localdir",
        matches=is_localdir_url_format,
        use_proxy=False,
        cookie_filename=None,
        asr_language=DEFAULT_ASR_LANGUAGE,
        target_language=DEFAULT_TARGET_LANGUAGE,
    ),
    SourceConfig(
        name="bilibili",
        matches=is_bilibili_url,
        use_proxy=False,
        cookie_filename="bilibili.txt",
        asr_language=DEFAULT_ASR_LANGUAGE,
        target_language=DEFAULT_TARGET_LANGUAGE,
    ),
]


def detect_source(url: str) -> SourceConfig:
    for source in SOURCES:
        if source.matches(url):
            return source
    raise ValueError(f"No source matches URL: {url}")


def _legacy_direction(task: dict, source: SourceConfig) -> str | None:
    """旧任务缺语言列时，仅本地 URL 允许从 direction 查询串兜底。"""
    if source.name == "local":
        return local_upload_direction(task["url"])
    if source.name == "localdir":
        return localdir_direction(task["url"])
    return None


def get_source_for_task(task: dict) -> SourceConfig:
    """Get source config for a task, using task-stored translation direction."""
    source = detect_source(task["url"])
    asr_language, target_language = normalize_task_language_pair(
        task,
        fallback_direction=_legacy_direction(task, source),
    )
    return SourceConfig(
        name=source.name,
        matches=source.matches,
        use_proxy=source.use_proxy,
        cookie_filename=source.cookie_filename,
        asr_language=asr_language,
        target_language=target_language,
    )
