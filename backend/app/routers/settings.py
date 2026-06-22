"""Runtime settings endpoints (OpenAI / yt-dlp / FunASR)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from .. import database
from ..adapters.openai_translate import list_models as list_openai_models
from ..schemas import (
    FunasrSettingsUpdate,
    OpenAIModelsRequest,
    OpenAISettingsUpdate,
    YtdlpSettingsUpdate,
)
from ._common import (
    mask_secret,
    normalize_funasr_use_vllm,
    normalize_proxy_port,
    normalize_translate_concurrency,
)

router = APIRouter(prefix="/api/settings", tags=["settings"])


@router.get("/openai")
def get_openai_settings() -> dict:
    settings = database.get_openai_settings()
    return {
        "base_url": settings["base_url"],
        "api_key": mask_secret(settings["api_key"]),
        "has_api_key": bool(settings["api_key"]),
        "model": settings["model"],
        "translate_concurrency": settings["translate_concurrency"],
    }


@router.post("/openai")
def save_openai_settings(payload: OpenAISettingsUpdate) -> dict:
    database.save_openai_settings(
        payload.base_url,
        payload.api_key,
        payload.model,
        normalize_translate_concurrency(payload.translate_concurrency),
        clear_api_key=payload.clear_api_key,
    )
    return get_openai_settings()


@router.post("/openai/models")
def get_openai_models(payload: OpenAIModelsRequest) -> dict:
    settings = database.get_openai_settings()
    base_url = payload.base_url.strip() or settings["base_url"]
    api_key = payload.api_key.strip() or settings["api_key"]
    try:
        models = list_openai_models(base_url=base_url, api_key=api_key)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Failed to fetch models: {exc}") from exc
    return {"models": models}


@router.get("/ytdlp")
def get_ytdlp_settings() -> dict:
    return database.get_ytdlp_settings()


@router.post("/ytdlp")
def save_ytdlp_settings(payload: YtdlpSettingsUpdate) -> dict:
    database.save_ytdlp_settings(normalize_proxy_port(payload.proxy_port))
    return get_ytdlp_settings()


@router.get("/funasr")
def get_funasr_settings() -> dict:
    return database.get_funasr_settings()


@router.post("/funasr")
def save_funasr_settings(payload: FunasrSettingsUpdate) -> dict:
    database.save_funasr_settings(normalize_funasr_use_vllm(payload.use_vllm))
    return get_funasr_settings()
