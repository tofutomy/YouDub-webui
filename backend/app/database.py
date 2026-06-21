from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import (
    DB_PATH,
    ensure_runtime_dirs,
    funasr_defaults,
    openai_defaults,
    ytdlp_defaults,
)
from .stages import STAGES


ACTIVE_STATUSES = ("queued", "running")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect() -> sqlite3.Connection:
    ensure_runtime_dirs()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with connect() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS tasks (
              id TEXT PRIMARY KEY,
              url TEXT NOT NULL,
              title TEXT,
              status TEXT NOT NULL,
              current_stage TEXT,
              session_path TEXT,
              final_video_path TEXT,
              error_message TEXT,
              created_at TEXT NOT NULL,
              started_at TEXT,
              completed_at TEXT
            );

            CREATE TABLE IF NOT EXISTS task_stages (
              task_id TEXT NOT NULL,
              name TEXT NOT NULL,
              label TEXT NOT NULL,
              status TEXT NOT NULL,
              progress INTEGER,
              started_at TEXT,
              completed_at TEXT,
              last_message TEXT,
              error_message TEXT,
              PRIMARY KEY (task_id, name),
              FOREIGN KEY (task_id) REFERENCES tasks(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS settings (
              key TEXT PRIMARY KEY,
              value TEXT NOT NULL,
              updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS translate_providers (
              id TEXT PRIMARY KEY,
              name TEXT NOT NULL,
              base_url TEXT NOT NULL DEFAULT '',
              api_key TEXT NOT NULL DEFAULT '',
              model TEXT NOT NULL DEFAULT '',
              is_default INTEGER NOT NULL DEFAULT 0,
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL
            );
            """
        )
        # Seed default provider from .env only when the table is empty (first run).
        provider_count = conn.execute("SELECT COUNT(*) AS cnt FROM translate_providers").fetchone()["cnt"]
        if provider_count == 0:
            defaults = openai_defaults()
            now = now_iso()
            conn.execute(
                """
                INSERT INTO translate_providers (id, name, base_url, api_key, model, is_default, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, 1, ?, ?)
                """,
                ("default", "Default", defaults["base_url"], defaults["api_key"], defaults["model"], now, now),
            )
        # Keep openai.* settings in sync with the default provider for backward compat.
        _sync_openai_settings_from_default_provider(conn)
        for key, value in ytdlp_defaults().items():
            conn.execute(
                """
                INSERT INTO settings (key, value, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
                """,
                (f"ytdlp.{key}", value, now_iso()),
            )
        task_columns = {row["name"] for row in conn.execute("PRAGMA table_info(tasks)").fetchall()}
        if "title" not in task_columns:
            conn.execute("ALTER TABLE tasks ADD COLUMN title TEXT")
        if "asr_language" not in task_columns:
            conn.execute("ALTER TABLE tasks ADD COLUMN asr_language TEXT")
        if "target_language" not in task_columns:
            conn.execute("ALTER TABLE tasks ADD COLUMN target_language TEXT")
        if "add_subtitles" not in task_columns:
            conn.execute("ALTER TABLE tasks ADD COLUMN add_subtitles INTEGER DEFAULT 1")
        if "asr_model" not in task_columns:
            conn.execute("ALTER TABLE tasks ADD COLUMN asr_model TEXT")
        if "stop_requested" not in task_columns:
            conn.execute("ALTER TABLE tasks ADD COLUMN stop_requested INTEGER DEFAULT 0")
        if "translate_mode" not in task_columns:
            conn.execute("ALTER TABLE tasks ADD COLUMN translate_mode TEXT")
        if "validate_translation" not in task_columns:
            conn.execute("ALTER TABLE tasks ADD COLUMN validate_translation INTEGER DEFAULT 0")
        if "tts_mode" not in task_columns:
            conn.execute("ALTER TABLE tasks ADD COLUMN tts_mode TEXT")
        if "translate_provider_id" not in task_columns:
            conn.execute("ALTER TABLE tasks ADD COLUMN translate_provider_id TEXT")
        if "stop_after_translate" not in task_columns:
            conn.execute("ALTER TABLE tasks ADD COLUMN stop_after_translate INTEGER DEFAULT 0")
        stage_columns = {row["name"] for row in conn.execute("PRAGMA table_info(task_stages)").fetchall()}
        if "progress" not in stage_columns:
            conn.execute("ALTER TABLE task_stages ADD COLUMN progress INTEGER")


def backfill_titles_from_metadata() -> None:
    import json
    with connect() as conn:
        rows = conn.execute(
            "SELECT id, session_path FROM tasks WHERE (title IS NULL OR title = '') AND session_path IS NOT NULL"
        ).fetchall()
    for row in rows:
        info_path = Path(row["session_path"]) / "metadata" / "ytdlp_info.json"
        if not info_path.exists():
            continue
        title = (json.loads(info_path.read_text(encoding="utf-8")).get("title") or "").strip()
        if not title:
            continue
        with connect() as conn:
            conn.execute("UPDATE tasks SET title = ? WHERE id = ?", (title, row["id"]))


# ---------------------------------------------------------------------------
# Translate Providers CRUD
# ---------------------------------------------------------------------------

def _sync_openai_settings_from_default_provider(conn: sqlite3.Connection) -> None:
    """Copy the default provider's values into openai.* settings keys for backward compat."""
    row = conn.execute(
        "SELECT base_url, api_key, model FROM translate_providers WHERE is_default = 1 LIMIT 1"
    ).fetchone()
    if not row:
        return
    defaults = openai_defaults()
    concurrency = defaults["translate_concurrency"]
    now = now_iso()
    for key, value in [
        ("openai.base_url", row["base_url"]),
        ("openai.api_key", row["api_key"]),
        ("openai.model", row["model"]),
        ("openai.translate_concurrency", concurrency),
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
    from .adapters.openai_client import normalize_openai_base_url

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
    from .adapters.openai_client import normalize_openai_base_url

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


def fail_stale_active_tasks() -> None:
    message = "Backend restarted before the task completed."
    completed_at = now_iso()
    with connect() as conn:
        active_tasks = conn.execute(
            f"SELECT id, current_stage FROM tasks WHERE status IN ({','.join('?' for _ in ACTIVE_STATUSES)})",
            ACTIVE_STATUSES,
        ).fetchall()
        for task in active_tasks:
            conn.execute(
                """
                UPDATE tasks
                SET status = 'failed', error_message = ?, completed_at = ?
                WHERE id = ?
                """,
                (message, completed_at, task["id"]),
            )
            if task["current_stage"]:
                conn.execute(
                    """
                    UPDATE task_stages
                    SET status = 'failed', error_message = ?, completed_at = ?
                    WHERE task_id = ? AND name = ? AND status IN ('pending', 'running')
                    """,
                    (message, completed_at, task["id"], task["current_stage"]),
                )


def create_task(
    url: str,
    task_id: str | None = None,
    asr_language: str | None = None,
    target_language: str | None = None,
    add_subtitles: bool = True,
    asr_model: str | None = None,
    translate_mode: str | None = None,
    validate_translation: bool = False,
    tts_mode: str | None = None,
    translate_provider_id: str | None = None,
    stop_after_translate: bool = False,
) -> str:
    new_id = task_id or str(uuid.uuid4())
    created_at = now_iso()
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO tasks (id, url, status, current_stage, created_at, asr_language, target_language, add_subtitles, asr_model, translate_mode, validate_translation, tts_mode, translate_provider_id, stop_after_translate)
            VALUES (?, ?, 'queued', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (new_id, url, STAGES[0].name, created_at, asr_language, target_language, int(add_subtitles), asr_model, translate_mode, int(validate_translation), tts_mode, translate_provider_id, int(stop_after_translate)),
        )
        conn.executemany(
            """
            INSERT INTO task_stages (task_id, name, label, status)
            VALUES (?, ?, ?, 'pending')
            """,
            [(new_id, stage.name, stage.label) for stage in STAGES],
        )
    return new_id


def find_task_by_video_id(video_id: str) -> str | None:
    with connect() as conn:
        row = conn.execute(
            "SELECT id FROM tasks WHERE id = ? OR url LIKE ? "
            "ORDER BY created_at DESC, rowid DESC LIMIT 1",
            (video_id, f"%{video_id}%"),
        ).fetchone()
    return row["id"] if row else None


def has_active_task() -> bool:
    with connect() as conn:
        row = conn.execute(
            f"SELECT 1 FROM tasks WHERE status IN ({','.join('?' for _ in ACTIVE_STATUSES)}) LIMIT 1",
            ACTIVE_STATUSES,
        ).fetchone()
    return row is not None


def latest_task_id() -> str | None:
    with connect() as conn:
        row = conn.execute("SELECT id FROM tasks ORDER BY created_at DESC, rowid DESC LIMIT 1").fetchone()
    return row["id"] if row else None


def list_tasks(limit: int = 100) -> list[dict[str, Any]]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT id, url, title, status, current_stage, final_video_path, error_message, "
            "created_at, started_at, completed_at FROM tasks "
            "ORDER BY created_at DESC, rowid DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]


def get_task(task_id: str) -> dict[str, Any] | None:
    with connect() as conn:
        task = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
        if not task:
            return None
        stages = conn.execute(
            """
            SELECT * FROM task_stages
            WHERE task_id = ?
            ORDER BY
              CASE name
                WHEN 'download' THEN 1
                WHEN 'separate' THEN 2
                WHEN 'asr' THEN 3
                WHEN 'asr_fix' THEN 4
                WHEN 'translate' THEN 5
                WHEN 'split_audio' THEN 6
                WHEN 'tts' THEN 7
                WHEN 'merge_audio' THEN 8
                WHEN 'merge_video' THEN 9
                ELSE 99
              END
            """,
            (task_id,),
        ).fetchall()
    result = dict(task)
    result["stages"] = [dict(stage) for stage in stages]
    return result


def get_current_task() -> dict[str, Any] | None:
    task_id = latest_task_id()
    return get_task(task_id) if task_id else None


def delete_task(task_id: str) -> bool:
    with connect() as conn:
        cursor = conn.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
        conn.execute("DELETE FROM task_stages WHERE task_id = ?", (task_id,))
        return cursor.rowcount > 0


def reset_failed_for_resume(task_id: str) -> None:
    with connect() as conn:
        conn.execute(
            """
            UPDATE task_stages
            SET status = 'pending', started_at = NULL, completed_at = NULL,
                progress = NULL, last_message = NULL, error_message = NULL
            WHERE task_id = ? AND status IN ('failed', 'running')
            """,
            (task_id,),
        )
        conn.execute(
            """
            UPDATE tasks
            SET status = 'queued', error_message = NULL, completed_at = NULL,
                started_at = NULL, stop_requested = 0
            WHERE id = ?
            """,
            (task_id,),
        )


def reset_stage_for_rerun(task_id: str, stage_name: str) -> None:
    """Reset the given stage and all subsequent stages to pending."""
    from .stages import STAGE_NAMES

    if stage_name not in STAGE_NAMES:
        raise ValueError(f"Unknown stage: {stage_name}")
    start_index = STAGE_NAMES.index(stage_name)
    stages_to_reset = STAGE_NAMES[start_index:]
    with connect() as conn:
        placeholders = ",".join("?" for _ in stages_to_reset)
        conn.execute(
            f"""
            UPDATE task_stages
            SET status = 'pending', started_at = NULL, completed_at = NULL,
                progress = NULL, last_message = NULL, error_message = NULL
            WHERE task_id = ? AND name IN ({placeholders})
            """,
            [task_id, *stages_to_reset],
        )
        conn.execute(
            """
            UPDATE tasks
            SET status = 'queued', error_message = NULL, completed_at = NULL,
                started_at = NULL, stop_requested = 0, current_stage = ?
            WHERE id = ?
            """,
            (stage_name, task_id),
        )


def reset_single_stage_for_rerun(task_id: str, stage_name: str) -> None:
    """Reset only the given stage to pending, without affecting other stages or task status."""
    from .stages import STAGE_NAMES

    if stage_name not in STAGE_NAMES:
        raise ValueError(f"Unknown stage: {stage_name}")
    with connect() as conn:
        conn.execute(
            """
            UPDATE task_stages
            SET status = 'pending', started_at = NULL, completed_at = NULL,
                progress = NULL, last_message = NULL, error_message = NULL
            WHERE task_id = ? AND name = ?
            """,
            (task_id, stage_name),
        )
        conn.execute(
            """
            UPDATE tasks
            SET status = 'queued', current_stage = ?, stop_requested = 0
            WHERE id = ?
            """,
            (stage_name, task_id),
        )


def update_task(task_id: str, **fields: Any) -> None:
    if not fields:
        return
    assignments = ", ".join(f"{key} = ?" for key in fields)
    values = list(fields.values()) + [task_id]
    with connect() as conn:
        conn.execute(f"UPDATE tasks SET {assignments} WHERE id = ?", values)


def update_stage(task_id: str, name: str, **fields: Any) -> None:
    if not fields:
        return
    assignments = ", ".join(f"{key} = ?" for key in fields)
    values = list(fields.values()) + [task_id, name]
    with connect() as conn:
        conn.execute(f"UPDATE task_stages SET {assignments} WHERE task_id = ? AND name = ?", values)


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
    from .adapters.openai_client import normalize_openai_base_url

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
    from .adapters.openai_client import normalize_openai_base_url

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


def log_path(task_id: str) -> Path:
    from .config import LOG_DIR

    return LOG_DIR / f"{task_id}.log"
