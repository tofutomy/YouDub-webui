"""YouTube cookie management endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from ..config import YOUTUBE_COOKIE_PATH
from ..schemas import YouTubeCookieUpdate

router = APIRouter(prefix="/api/cookies", tags=["cookies"])


@router.get("/youtube")
def get_youtube_cookie() -> dict:
    exists = YOUTUBE_COOKIE_PATH.exists()
    size = YOUTUBE_COOKIE_PATH.stat().st_size if exists else 0
    updated_at = YOUTUBE_COOKIE_PATH.stat().st_mtime if exists else None
    return {"exists": exists, "size": size, "updated_at": updated_at, "content": ""}


@router.post("/youtube")
def save_youtube_cookie(payload: YouTubeCookieUpdate) -> dict:
    YOUTUBE_COOKIE_PATH.parent.mkdir(parents=True, exist_ok=True)
    content = payload.content.strip()
    if content:
        YOUTUBE_COOKIE_PATH.write_text(content + "\n", encoding="utf-8")
    elif YOUTUBE_COOKIE_PATH.exists():
        YOUTUBE_COOKIE_PATH.unlink()
    return get_youtube_cookie()
