"""Helpers for user-requested task stops.

Centralizes the messages and database updates triggered by a stop request so
that the worker loop, the FastAPI endpoint, and the pipeline can all share the
same code path.
"""

from __future__ import annotations

from . import database


# Visible in the UI / log. Keep these as the single source of truth - do not
# hardcode them elsewhere.
STOPPED_MESSAGE = "Stopped by user."
STOPPED_BEFORE_START_MESSAGE = "Stopped by user before start."


def mark_task_as_stopped(task_id: str, *, error_message: str = STOPPED_MESSAGE) -> None:
    """Mark a task as failed because the user requested a stop.

    Sets ``stop_requested=1`` on the task so the frontend can distinguish
    user-stops from genuine errors. If the task had already started running,
    the currently active stage is also marked as failed. For queued tasks that
    were stopped before they could run, stages are left untouched (they remain
    in ``pending``).
    """
    current = database.get_task(task_id)
    if current is None:
        return
    completed_at = database.now_iso()
    started_at = current.get("started_at")
    current_stage = current.get("current_stage")
    if started_at and current_stage and current_stage != "done":
        current_stage_status = None
        for stage_entry in current.get("stages", []):
            if stage_entry["name"] == current_stage:
                current_stage_status = stage_entry["status"]
                break
        # Only mark the stage as failed if it was actually running. When the stop
        # happens right at a stage boundary, current_stage may still point to a
        # previously-succeeded stage — do not overwrite that with "failed".
        if current_stage_status == "running":
            database.update_stage(
                task_id,
                current_stage,
                status="failed",
                completed_at=completed_at,
                error_message=error_message,
                last_message="Stopped",
            )
    database.update_task(
        task_id,
        status="failed",
        error_message=error_message,
        completed_at=completed_at,
        stop_requested=1,
    )
