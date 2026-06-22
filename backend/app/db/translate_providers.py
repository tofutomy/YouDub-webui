"""Translate provider CRUD operations."""

from __future__ import annotations

import uuid
from typing import Any

from ..config import openai_defaults
from .connection import connect, now_iso
from .settings import get_openai_settings, get_setting


def _sync_openai_settings_from_default_provider(conn) -> None:
    """Copy the default provider's values into openai.* settings keys for backward compat."""
    row = conn.execute(
        "SELECT base_url, api_key, model FROM translate_providers WHERE is_default = 1 LIMIT 1"
    ).fetchone()
    if not row:
        return
    now = now_iso()
    for key, value in [
        ("openai.base_url", row["base_url"]),
        ("openai.api_key", row["api_key"]),
        ("openai.model", row["model"]),
    ]:
        conn.execute(
            """
            INSERT INTO settings (key, value, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
            """,
            (key, value, now),
        )


def list_translate_providers() -> list[dict[str, Any]]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT id, name, base_url, api_key, model, is_default, created_at, updated_at "
            "FROM translate_providers ORDER BY is_default DESC, created_at ASC"
        ).fetchall()
    return [dict(row) for row in rows]


def get_translate_provider(provider_id: str) -> dict[str, Any] | None:
    with connect() as conn:
        row = conn.execute(
            "SELECT * FROM translate_providers WHERE id = ?", (provider_id,)
        ).fetchone()
    return dict(row) if row else None


def get_default_translate_provider() -> dict[str, Any] | None:
    with connect() as conn:
        row = conn.execute(
            "SELECT * FROM translate_providers WHERE is_default = 1 LIMIT 1"
        ).fetchone()
    if row:
        return dict(row)
    # Fallback: first provider
    row = conn.execute(
        "SELECT * FROM translate_providers ORDER BY created_at ASC LIMIT 1"
    ).fetchone()
    return dict(row) if row else None


def create_translate_provider(
    name: str,
    base_url: str,
    api_key: str,
    model: str,
    is_default: bool = False,
    provider_id: str | None = None,
) -> str:
    from ..adapters.openai_client import normalize_openai_base_url

    new_id = provider_id or str(uuid.uuid4())
    now = now_iso()
    with connect() as conn:
        if is_default:
            conn.execute("UPDATE translate_providers SET is_default = 0 WHERE is_default = 1")
        conn.execute(
            """
            INSERT INTO translate_providers (id, name, base_url, api_key, model, is_default, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (new_id, name.strip(), normalize_openai_base_url(base_url), api_key.strip(), model.strip(), int(is_default), now, now),
        )
        _sync_openai_settings_from_default_provider(conn)
    return new_id


def update_translate_provider(
    provider_id: str,
    *,
    name: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    clear_api_key: bool = False,
    model: str | None = None,
    is_default: bool | None = None,
) -> None:
    from ..adapters.openai_client import normalize_openai_base_url

    with connect() as conn:
        existing = conn.execute(
            "SELECT * FROM translate_providers WHERE id = ?", (provider_id,)
        ).fetchone()
        if not existing:
            raise ValueError(f"Provider {provider_id} not found")
        if is_default:
            conn.execute("UPDATE translate_providers SET is_default = 0 WHERE is_default = 1")
        updates: list[str] = []
        params: list[Any] = []
        if name is not None:
            updates.append("name = ?")
            params.append(name.strip())
        if base_url is not None:
            updates.append("base_url = ?")
            params.append(normalize_openai_base_url(base_url))
        if clear_api_key:
            updates.append("api_key = ?")
            params.append("")
        elif api_key is not None:
            cleaned = api_key.strip()
            if cleaned and set(cleaned) != {"*"}:
                updates.append("api_key = ?")
                params.append(cleaned)
        if model is not None:
            updates.append("model = ?")
            params.append(model.strip())
        if is_default is not None:
            updates.append("is_default = ?")
            params.append(int(is_default))
        if updates:
            updates.append("updated_at = ?")
            params.append(now_iso())
            params.append(provider_id)
            conn.execute(
                f"UPDATE translate_providers SET {', '.join(updates)} WHERE id = ?",
                params,
            )
        _sync_openai_settings_from_default_provider(conn)


def delete_translate_provider(provider_id: str) -> bool:
    with connect() as conn:
        existing = conn.execute(
            "SELECT is_default FROM translate_providers WHERE id = ?", (provider_id,)
        ).fetchone()
        if not existing:
            return False
        conn.execute("DELETE FROM translate_providers WHERE id = ?", (provider_id,))
        # If we deleted the default, promote the first remaining one.
        if existing["is_default"]:
            first = conn.execute(
                "SELECT id FROM translate_providers ORDER BY created_at ASC LIMIT 1"
            ).fetchone()
            if first:
                conn.execute(
                    "UPDATE translate_providers SET is_default = 1 WHERE id = ?",
                    (first["id"],),
                )
        _sync_openai_settings_from_default_provider(conn)
    return True


def get_translate_provider_settings(provider_id: str | None) -> dict[str, str]:
    """Return base_url/api_key/model for the given provider, falling back to default."""
    provider = None
    if provider_id:
        provider = get_translate_provider(provider_id)
    if not provider:
        provider = get_default_translate_provider()
    if not provider:
        # Ultimate fallback: legacy openai settings
        return get_openai_settings()
    return {
        "base_url": provider["base_url"],
        "api_key": provider["api_key"],
        "model": provider["model"],
        "translate_concurrency": get_setting(
            "openai.translate_concurrency", openai_defaults()["translate_concurrency"]
        ),
    }
