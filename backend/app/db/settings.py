"""Key/value settings store (openai, ytdlp, funasr)."""

from __future__ import annotations

from ..config import funasr_defaults, openai_defaults, ytdlp_defaults
from .connection import connect, now_iso


def set_setting(key: str, value: str) -> None:
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO settings (key, value, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
            """,
            (key, value, now_iso()),
        )


def get_setting(key: str, default: str = "") -> str:
    with connect() as conn:
        row = conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
    return row["value"] if row else default


def get_openai_settings() -> dict[str, str]:
    from ..adapters.openai_client import normalize_openai_base_url

    defaults = openai_defaults()
    return {
        "base_url": normalize_openai_base_url(get_setting("openai.base_url", defaults["base_url"])),
        "api_key": get_setting("openai.api_key", defaults["api_key"]),
        "model": get_setting("openai.model", defaults["model"]),
        "translate_concurrency": get_setting(
            "openai.translate_concurrency", defaults["translate_concurrency"]
        ),
    }


def save_openai_settings(
    base_url: str,
    api_key: str,
    model: str,
    translate_concurrency: str = "",
    *,
    clear_api_key: bool = False,
) -> None:
    from ..adapters.openai_client import normalize_openai_base_url

    set_setting("openai.base_url", normalize_openai_base_url(base_url))
    cleaned_api_key = api_key.strip()
    if clear_api_key:
        set_setting("openai.api_key", "")
    elif cleaned_api_key and set(cleaned_api_key) != {"*"}:
        set_setting("openai.api_key", cleaned_api_key)
    set_setting("openai.model", model.strip())
    if translate_concurrency.strip():
        set_setting("openai.translate_concurrency", translate_concurrency.strip())


def get_ytdlp_settings() -> dict[str, str]:
    defaults = ytdlp_defaults()
    return {
        "proxy_port": get_setting("ytdlp.proxy_port", defaults["proxy_port"]),
    }


def save_ytdlp_settings(proxy_port: str) -> None:
    set_setting("ytdlp.proxy_port", proxy_port.strip())


VALID_FUNASR_USE_VLLM = {"auto", "on", "off"}


def get_funasr_settings() -> dict[str, str]:
    defaults = funasr_defaults()
    return {
        "use_vllm": get_setting("funasr.use_vllm", defaults["use_vllm"]) or "auto",
    }


def save_funasr_settings(use_vllm: str) -> None:
    value = (use_vllm or "auto").strip().lower()
    if value not in VALID_FUNASR_USE_VLLM:
        raise ValueError(f"use_vllm must be one of {sorted(VALID_FUNASR_USE_VLLM)}")
    set_setting("funasr.use_vllm", value)
