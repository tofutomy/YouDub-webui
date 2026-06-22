"""Schema initialization and migrations."""

from __future__ import annotations

import json
from pathlib import Path

from ..config import openai_defaults, ytdlp_defaults
from ..stages import STAGES
from .connection import connect, now_iso
from .translate_providers import _sync_openai_settings_from_default_provider


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
        # Seed translate_concurrency from .env only if it doesn't exist yet.
        defaults = openai_defaults()
        conn.execute(
            """
            INSERT INTO settings (key, value, updated_at)
            VALUES ('openai.translate_concurrency', ?, ?)
            ON CONFLICT(key) DO NOTHING
            """,
            (defaults["translate_concurrency"], now_iso()),
        )
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
        if "demucs_model" not in task_columns:
            conn.execute("ALTER TABLE tasks ADD COLUMN demucs_model TEXT")
        if "demucs_shifts" not in task_columns:
            conn.execute("ALTER TABLE tasks ADD COLUMN demucs_shifts INTEGER DEFAULT 1")
        if "use_amp" not in task_columns:
            conn.execute("ALTER TABLE tasks ADD COLUMN use_amp INTEGER DEFAULT 1")
        if "filter_fillers" not in task_columns:
            conn.execute("ALTER TABLE tasks ADD COLUMN filter_fillers INTEGER DEFAULT 0")
        stage_columns = {row["name"] for row in conn.execute("PRAGMA table_info(task_stages)").fetchall()}
        if "progress" not in stage_columns:
            conn.execute("ALTER TABLE task_stages ADD COLUMN progress INTEGER")


def backfill_titles_from_metadata() -> None:
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
