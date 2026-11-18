from __future__ import annotations

from urllib.parse import urlsplit
from uuid import uuid4


def normalize_openai_base_url(base_url: str) -> str:
    url = base_url.strip().rstrip("/")
    lowered = url.lower()
    for suffix in ("/chat/completions", "/completions"):
        if lowered.endswith(suffix):
            url = url[: -len(suffix)].rstrip("/")
            lowered = url.lower()
    return url or "https://api.openai.com/v1"


def is_opencode_go(base_url: str) -> bool:
    endpoint = urlsplit(base_url)
    path = endpoint.path.rstrip("/") + "/"
    return (endpoint.scheme == "https" and endpoint.hostname == "opencode.ai"
            and path.startswith(("/inference/go/", "/zen/go/")))


def opencode_go_options(base_url: str, session_id: str = "") -> dict:
    if not is_opencode_go(base_url):
        return {}
    return {"default_headers": {"User-Agent": "YouDub-webui/1.0",
                                "x-opencode-session": session_id or str(uuid4())}}
