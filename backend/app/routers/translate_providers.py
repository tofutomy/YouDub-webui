"""Translate provider CRUD endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Response

from .. import database
from ..adapters.openai_translate import list_models as list_openai_models
from ..schemas import TranslateProviderCreate, TranslateProviderUpdate

router = APIRouter(prefix="/api/translate-providers", tags=["translate-providers"])


def _require_provider(provider_id: str) -> dict[str, Any]:
    provider = database.get_translate_provider(provider_id)
    if provider is None:
        raise HTTPException(status_code=404, detail="Provider not found.")
    return provider


def _serialize_provider(p: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": p["id"],
        "name": p["name"],
        "base_url": p["base_url"],
        "api_key": p["api_key"],
        "has_api_key": bool(p["api_key"]),
        "model": p["model"],
        "is_default": bool(p["is_default"]),
        "created_at": p["created_at"],
        "updated_at": p["updated_at"],
    }


@router.get("")
def list_translate_providers() -> dict[str, Any]:
    providers = database.list_translate_providers()
    return {"providers": [_serialize_provider(p) for p in providers]}


@router.post("", status_code=201)
def create_translate_provider(payload: TranslateProviderCreate) -> dict[str, Any]:
    provider_id = database.create_translate_provider(
        name=payload.name,
        base_url=payload.base_url,
        api_key=payload.api_key,
        model=payload.model,
        is_default=payload.is_default,
    )
    return _serialize_provider(_require_provider(provider_id))


@router.patch("/{provider_id}")
def update_translate_provider(provider_id: str, payload: TranslateProviderUpdate) -> dict[str, Any]:
    _require_provider(provider_id)
    try:
        database.update_translate_provider(
            provider_id,
            name=payload.name,
            base_url=payload.base_url,
            api_key=payload.api_key,
            clear_api_key=payload.clear_api_key,
            model=payload.model,
            is_default=payload.is_default,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return _serialize_provider(_require_provider(provider_id))


@router.delete("/{provider_id}", status_code=204)
def delete_translate_provider(provider_id: str) -> Response:
    if not database.delete_translate_provider(provider_id):
        raise HTTPException(status_code=404, detail="Provider not found.")
    return Response(status_code=204)


@router.post("/{provider_id}/models")
def list_translate_provider_models(provider_id: str) -> dict[str, Any]:
    provider = _require_provider(provider_id)
    base_url = provider["base_url"]
    api_key = provider["api_key"]
    if not base_url:
        settings = database.get_openai_settings()
        base_url = settings["base_url"]
        api_key = api_key or settings["api_key"]
    try:
        models = list_openai_models(base_url=base_url, api_key=api_key)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Failed to fetch models: {exc}") from exc
    return {"models": models}
