from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Callable

import time

import requests
import yt_dlp

from ..sanitize import sanitize_text
from ..sources import SourceConfig
from ..youtube import extract_video_id


FORMAT_CANDIDATES = (
    "bestvideo[height<=1080]+bestaudio/best",
    "bestvideo+bestaudio/best",
    "bv*+ba/b",
    "best",
)

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


def _bootstrap_bilibili_cookie(cookie_path: Path) -> None:
    response = requests.get(
        "https://www.bilibili.com/",
        headers={"User-Agent": DEFAULT_USER_AGENT, "Referer": "https://www.bilibili.com/"},
        timeout=10,
    )
    response.raise_for_status()
    expires = int(time.time()) + 3600 * 24 * 365
    lines = ["# Netscape HTTP Cookie File", ""]
    cookies = dict(response.cookies)
    cookies.setdefault("SESSDATA", "anonymous_for_webpage_playinfo")
    for name, value in cookies.items():
        lines.append("\t".join([".bilibili.com", "TRUE", "/", "FALSE", str(expires), name, value]))
    cookie_path.parent.mkdir(parents=True, exist_ok=True)
    cookie_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _proxy_url(proxy_port: str = "") -> str:
    if proxy_port.strip():
        return f"http://127.0.0.1:{proxy_port.strip()}"
    return os.getenv("HTTP_PROXY") or os.getenv("http_proxy") or ""


def _ensure_cookie(source: SourceConfig) -> None:
    cookie_path = source.cookie_path
    if not cookie_path or source.name != "bilibili":
        return
    if cookie_path.exists() and cookie_path.stat().st_size > 0:
        return
    _bootstrap_bilibili_cookie(cookie_path)


def _ydl_base(
    source: SourceConfig, proxy_port: str = "", firefox_profile: Path | None = None
) -> dict[str, Any]:
    opts: dict[str, Any] = {
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "js_runtimes": {"node": {}},
        "http_headers": {"User-Agent": DEFAULT_USER_AGENT},
    }
    configured_profile = os.getenv("YOUTUBE_FIREFOX_PROFILE") if source.name == "youtube" else None
    profile = firefox_profile or (Path(configured_profile).expanduser() if configured_profile else None)
    if source.name == "youtube" and profile is not None:
        opts.pop("http_headers", None)
        if (profile / "cookies.sqlite").is_file():
            opts["cookiesfrombrowser"] = ("firefox", str(profile.resolve()), None, None)
    else:
        cookie_path = source.cookie_path
        if cookie_path and cookie_path.exists() and cookie_path.stat().st_size > 0:
            opts["cookiefile"] = str(cookie_path)
    if not source.use_proxy:
        opts["proxy"] = ""
        return opts
    proxy = _proxy_url(proxy_port)
    if proxy:
        opts["proxy"] = proxy
    return opts


def _session_path(workfolder: Path, info: dict[str, Any]) -> Path:
    uploader = sanitize_text(str(info.get("uploader") or "unknown"))
    title = sanitize_text(str(info.get("title") or "untitled"))
    video_id = str(info.get("id") or extract_video_id(str(info.get("webpage_url") or "")))
    return workfolder / uploader / f"{title}__{video_id}"


def _managed_component(value: str, fallback: str, max_chars: int) -> str:
    label = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", value.strip())[:max_chars].strip(" .") or fallback
    if re.fullmatch(r"(?i:con|prn|aux|nul|com[1-9¹²³]|lpt[1-9¹²³])", label.split(".")[0]):
        label = "_" + label
    return label


def _managed_session(root: Path, info: dict[str, Any], task_id: str) -> Path:
    author = root / _managed_component(str(info.get("uploader") or info.get("channel") or ""), "未知作者", 60)
    if author.resolve().parent != root.resolve():
        raise ValueError("作者目录必须位于所选工作目录内。")
    author.mkdir(parents=True, exist_ok=True)
    title = _managed_component(str(info.get("title") or ""), "未命名视频", 100)
    video_id = str(info["id"])
    for name in (title, f"{title} [{video_id}]", f"{title} [{video_id}-{task_id[-8:]}]"):
        candidate = author / name
        try:
            candidate.mkdir()
        except FileExistsError:
            continue
        return candidate
    raise FileExistsError("工作目录下已有多个同名任务目录。")


def _is_format_unavailable(exc: Exception) -> bool:
    return "Requested format is not available" in str(exc)


def _remove_partial_outputs(video_file: Path) -> None:
    for candidate in video_file.parent.glob(f"{video_file.name}*"):
        if candidate == video_file:
            continue
        if candidate.is_file():
            candidate.unlink(missing_ok=True)


def _download_with_format_candidates(
    url: str, video_file: Path, source: SourceConfig, proxy_port: str, firefox_profile: Path | None = None
) -> None:
    last_error: Exception | None = None
    for format_selector in FORMAT_CANDIDATES:
        download_opts = {
            **_ydl_base(source, proxy_port, firefox_profile),
            "format": format_selector,
            "merge_output_format": "mp4",
            "outtmpl": str(video_file),
            "retries": 10,
            "fragment_retries": 10,
        }
        if firefox_profile is not None:
            download_opts["outtmpl"] = str(video_file.with_suffix(".%(ext)s"))
            download_opts["postprocessors"] = [{"key": "FFmpegVideoRemuxer", "preferedformat": "mp4"}]
        try:
            with yt_dlp.YoutubeDL(download_opts) as ydl:
                ydl.download([url])
            return
        except Exception as exc:
            last_error = exc
            _remove_partial_outputs(video_file)
            if not _is_format_unavailable(exc):
                continue
    if last_error:
        raise last_error


def download_video(
    url: str, workfolder: Path, source: SourceConfig, proxy_port: str = "",
    *, firefox_profile: Path | None = None, managed_task_id: str | None = None,
    saved_session: Path | None = None, on_session: Callable[[Path], None] | None = None,
) -> tuple[Path, dict[str, Any]]:
    video_id = extract_video_id(url)
    if saved_session is not None:
        resolved_session = saved_session.resolve()
        if len(resolved_session.relative_to(workfolder.resolve()).parts) != 2:
            raise ValueError("任务目录必须位于所选工作目录的作者和视频子目录内。")
        cached_video = resolved_session / "media" / "video_source.mp4"
        cached_metadata = resolved_session / "metadata" / "ytdlp_info.json"
        if cached_metadata.is_file() and cached_video.is_file() and cached_video.stat().st_size > 0:
            cached_info = json.loads(cached_metadata.read_text(encoding="utf-8"))
            if cached_info.get("id") == video_id:
                return resolved_session, cached_info
    _ensure_cookie(source)
    info_opts = _ydl_base(source, proxy_port, firefox_profile)
    with yt_dlp.YoutubeDL(info_opts) as ydl:
        info = ydl.extract_info(url, download=False)

    if str(info.get("id", video_id)) != video_id:
        raise ValueError("The resolved video id does not match the submitted URL.")

    if managed_task_id:
        session = saved_session or _managed_session(workfolder, info, managed_task_id)
        if on_session is not None:
            on_session(session)
    else:
        session = _session_path(workfolder, info)
    media_dir = session / "media"
    metadata_dir = session / "metadata"
    media_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    video_file = media_dir / "video_source.mp4"
    metadata_file = metadata_dir / "ytdlp_info.json"
    if managed_task_id:
        metadata = {key: info.get(key) for key in ("id", "title", "uploader", "channel", "description", "webpage_url", "duration", "upload_date")}
    else:
        metadata = ydl.sanitize_info(info)
    metadata_file.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    if source.name == "youtube":
        from .youtube_assets import download_youtube_assets

        with yt_dlp.YoutubeDL({**info_opts, "socket_timeout": 10}) as assets_client:
            download_youtube_assets(assets_client, info, session)

    if video_file.exists() and video_file.stat().st_size > 0:
        return session, info

    _download_with_format_candidates(url, video_file, source, proxy_port, firefox_profile)

    if not video_file.exists() or video_file.stat().st_size == 0:
        raise RuntimeError("yt-dlp finished without producing media/video_source.mp4")

    return session, info
