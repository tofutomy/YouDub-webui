"""Task and task_stages CRUD operations."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from ..config import LOG_DIR
from ..stages import STAGES
from .connection import ACTIVE_STATUSES, connect, now_iso


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
    filter_fillers: bool = False,
    demucs_model: str | None = None,
    demucs_shifts: int = 1,
) -> str:
    new_id = task_id or str(uuid.uuid4())
    created_at = now_iso()
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO tasks (id, url, status, current_stage, created_at, asr_language, target_language, add_subtitles, asr_model, translate_mode, validate_translation, tts_mode, translate_provider_id, stop_after_translate, filter_fillers, demucs_model, demucs_shifts)
            VALUES (?, ?, 'queued', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (new_id, url, STAGES[0].name, created_at, asr_language, target_language, int(add_subtitles), asr_model, translate_mode, int(validate_translation), tts_mode, translate_provider_id, int(stop_after_translate), int(filter_fillers), demucs_model, demucs_shifts),
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


def list_tasks(limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT id, url, title, status, current_stage, final_video_path, error_message, "
            "created_at, started_at, completed_at FROM tasks "
            "ORDER BY created_at DESC, rowid DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
    return [dict(row) for row in rows]


def count_tasks() -> int:
    with connect() as conn:
        row = conn.execute("SELECT COUNT(*) FROM tasks").fetchone()
    return row[0] if row else 0


def _stage_order_case_sql() -> str:
    """Build the ``CASE name WHEN ... THEN <n>`` clause from ``STAGES``.

    Keeps the ordering in sync with ``stages.STAGES`` so adding a stage
    doesn't require touching this SQL.
    """
    whens = " ".join(
        f"WHEN '{stage.name}' THEN {i + 1}" for i, stage in enumerate(STAGES)
    )
    return f"CASE name {whens} ELSE 99 END"


def get_task(task_id: str) -> dict[str, Any] | None:
    with connect() as conn:
        task = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
        if not task:
            return None
        stages = conn.execute(
            f"""
            SELECT * FROM task_stages
            WHERE task_id = ?
            ORDER BY {_stage_order_case_sql()}
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
    from ..stages import STAGE_NAMES

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
    from ..stages import STAGE_NAMES

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


# Whitelist of columns that may be written via update_task / update_stage.
# Prevents accidental injection of untrusted keys into the dynamic SQL.
_TASK_WRITABLE_FIELDS = frozenset({
    "url", "title", "status", "current_stage", "session_path", "final_video_path",
    "error_message", "started_at", "completed_at", "asr_language", "target_language",
    "add_subtitles", "asr_model", "stop_requested", "translate_mode",
    "validate_translation", "tts_mode", "translate_provider_id",
    "stop_after_translate", "filter_fillers", "demucs_model", "demucs_shifts",
    "use_amp",
})
_STAGE_WRITABLE_FIELDS = frozenset({
    "label", "status", "progress", "started_at", "completed_at",
    "last_message", "error_message",
})


def update_task(task_id: str, **fields: Any) -> None:
    if not fields:
        return
    unknown = set(fields) - _TASK_WRITABLE_FIELDS
    if unknown:
        raise ValueError(f"Unknown task fields: {sorted(unknown)}")
    assignments = ", ".join(f"{key} = ?" for key in fields)
    values = list(fields.values()) + [task_id]
    with connect() as conn:
        conn.execute(f"UPDATE tasks SET {assignments} WHERE id = ?", values)


def update_stage(task_id: str, name: str, **fields: Any) -> None:
    if not fields:
        return
    unknown = set(fields) - _STAGE_WRITABLE_FIELDS
    if unknown:
        raise ValueError(f"Unknown stage fields: {sorted(unknown)}")
    assignments = ", ".join(f"{key} = ?" for key in fields)
    values = list(fields.values()) + [task_id, name]
    with connect() as conn:
        conn.execute(f"UPDATE task_stages SET {assignments} WHERE task_id = ? AND name = ?", values)


def log_path(task_id: str) -> Path:
    return LOG_DIR / f"{task_id}.log"
