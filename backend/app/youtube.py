from __future__ import annotations

import re
from urllib.parse import parse_qs, quote, unquote, urlparse

from .languages import SUPPORTED_DIRECTION_SET


YOUTUBE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")
BILIBILI_BV_RE = re.compile(r"BV[A-Za-z0-9]{10}")
BILIBILI_HOSTS = {"bilibili.com", "www.bilibili.com", "m.bilibili.com"}
LOCAL_UPLOAD_SCHEME = "local"
LOCAL_UPLOAD_HOST = "upload"
LOCAL_UPLOAD_DIRECTIONS = SUPPORTED_DIRECTION_SET
LOCAL_UPLOAD_TASK_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")
LOCALDIR_SCHEME = "localdir"
LOCALDIR_TASK_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def _extract_youtube_id(parsed) -> str | None:
    host = parsed.netloc.lower()
    path = parsed.path.strip("/")

    if host in {"youtu.be", "www.youtu.be"}:
        candidate = path.split("/")[0]
        if YOUTUBE_ID_RE.match(candidate):
            return candidate

    if "youtube.com" not in host:
        return None

    query_id = parse_qs(parsed.query).get("v", [""])[0]
    if YOUTUBE_ID_RE.match(query_id):
        return query_id

    parts = path.split("/")
    for prefix in ("shorts", "embed", "live"):
        if len(parts) >= 2 and parts[0] == prefix and YOUTUBE_ID_RE.match(parts[1]):
            return parts[1]
    return None


def _extract_bilibili_id(parsed) -> str | None:
    host = parsed.netloc.lower()
    if host not in BILIBILI_HOSTS:
        return None
    match = BILIBILI_BV_RE.search(parsed.path)
    if match:
        return match.group(0)
    return None


def extract_video_id(url: str) -> str:
    parsed = urlparse(url.strip())
    video_id = _extract_youtube_id(parsed) or _extract_bilibili_id(parsed)
    if video_id:
        return video_id
    raise ValueError("Only YouTube or Bilibili single-video URLs are supported.")


def is_youtube_url(url: str) -> bool:
    try:
        return _extract_youtube_id(urlparse(url.strip())) is not None
    except ValueError:
        return False


def is_bilibili_url(url: str) -> bool:
    return urlparse(url.strip()).netloc.lower() in BILIBILI_HOSTS


def local_upload_task_id(url: str) -> str:
    parsed = urlparse(url.strip())
    if parsed.scheme != LOCAL_UPLOAD_SCHEME or parsed.netloc != LOCAL_UPLOAD_HOST:
        return ""
    candidate = parsed.path.strip("/").split("/", maxsplit=1)[0]
    if not LOCAL_UPLOAD_TASK_ID_RE.match(candidate):
        return ""
    return candidate


def local_upload_direction(url: str) -> str:
    parsed = urlparse(url.strip())
    if not local_upload_task_id(url):
        return ""
    return parse_qs(parsed.query).get("direction", [""])[0]


def is_local_upload_url(url: str) -> bool:
    return bool(local_upload_task_id(url)) and local_upload_direction(url) in LOCAL_UPLOAD_DIRECTIONS


def is_local_url_format(url: str) -> bool:
    """Check if URL is a local upload (any direction)."""
    return bool(local_upload_task_id(url))


def is_local_en_to_zh_url(url: str) -> bool:
    return is_local_upload_url(url) and local_upload_direction(url) == "en-zh"


def is_local_zh_to_en_url(url: str) -> bool:
    return is_local_upload_url(url) and local_upload_direction(url) == "zh-en"


def is_local_ja_to_zh_url(url: str) -> bool:
    return is_local_upload_url(url) and local_upload_direction(url) == "ja-zh"


# ---------------------------------------------------------------------------
# localdir:// protocol — reads video directly from a local file path
# Format: localdir://{task_id}?direction={dir}&path={encoded_path}&filename={name}
# ---------------------------------------------------------------------------

def make_localdir_url(task_id: str, file_path: str, direction: str, filename: str = "") -> str:
    """Build a ``localdir://`` URL for a task that reads from *file_path*."""
    encoded_path = quote(file_path, safe="")
    name = filename or file_path.rsplit("\\", 1)[-1].rsplit("/", 1)[-1]
    return (
        f"{LOCALDIR_SCHEME}://{task_id}"
        f"?direction={direction}"
        f"&path={encoded_path}"
        f"&filename={quote(name, safe='')}"
    )


def localdir_task_id(url: str) -> str:
    parsed = urlparse(url.strip())
    if parsed.scheme != LOCALDIR_SCHEME:
        return ""
    candidate = parsed.netloc.strip("/")
    if not candidate:
        # Also accept the path form: localdir:///taskid?...
        candidate = parsed.path.strip("/").split("/", maxsplit=1)[0]
    if not LOCALDIR_TASK_ID_RE.match(candidate):
        return ""
    return candidate


def localdir_source_path(url: str) -> str:
    """Return the decoded file path embedded in a ``localdir://`` URL."""
    parsed = urlparse(url.strip())
    if parsed.scheme != LOCALDIR_SCHEME:
        return ""
    encoded = parse_qs(parsed.query).get("path", [""])[0]
    if not encoded:
        return ""
    return unquote(encoded)


def localdir_direction(url: str) -> str:
    parsed = urlparse(url.strip())
    if parsed.scheme != LOCALDIR_SCHEME:
        return ""
    if not localdir_task_id(url):
        return ""
    return parse_qs(parsed.query).get("direction", [""])[0]


def localdir_filename(url: str) -> str:
    """Return the original filename embedded in a ``localdir://`` URL."""
    parsed = urlparse(url.strip())
    if parsed.scheme != LOCALDIR_SCHEME:
        return ""
    return unquote(parse_qs(parsed.query).get("filename", [""])[0])


def is_localdir_url(url: str) -> bool:
    return (
        bool(localdir_task_id(url))
        and bool(localdir_source_path(url))
        and localdir_direction(url) in LOCAL_UPLOAD_DIRECTIONS
    )


def is_localdir_url_format(url: str) -> bool:
    """Check if URL is a localdir upload (any direction)."""
    return bool(localdir_task_id(url)) and bool(localdir_source_path(url))


def is_localdir_en_to_zh_url(url: str) -> bool:
    return is_localdir_url(url) and localdir_direction(url) == "en-zh"


def is_localdir_zh_to_en_url(url: str) -> bool:
    return is_localdir_url(url) and localdir_direction(url) == "zh-en"


def is_localdir_ja_to_zh_url(url: str) -> bool:
    return is_localdir_url(url) and localdir_direction(url) == "ja-zh"
