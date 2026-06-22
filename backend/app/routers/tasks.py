"""Task lifecycle REST endpoints.

Covers task creation (URL / upload / localdir), listing, detail, config
update, delete, rerun, resume, stop, stage rerun/clear, log and final
video artifact.
"""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from urllib.parse import quote

from fastapi import APIRouter, File, Form, HTTPException, Response, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse

from .. import database, worker
from ..adapters.local_video import remove_upload, upload_dir
from ..config import WORKFOLDER
from ..pipeline import run_task, run_task_single_stage
from ..runtime_checks import validate_runtime_device
from ..sanitize import sanitize_text
from ..stops import STOPPED_BEFORE_START_MESSAGE, mark_task_as_stopped
from ..task_config import TaskConfig
from ..youtube import (
    LOCAL_UPLOAD_DIRECTIONS,
    extract_video_id,
    is_local_upload_url,
    is_localdir_url,
    make_localdir_url,
)
from ..schemas import (
    LocaldirTaskCreate,
    TaskCreate,
)

router = APIRouter(prefix="/api/tasks", tags=["tasks"])

ALLOWED_VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".mkv", ".webm", ".avi", ".flv", ".wmv"}
LOCAL_UPLOAD_CHUNK_SIZE = 1024 * 1024
MAX_LOCAL_UPLOAD_BYTES = int(__import__("os").getenv("LOCAL_UPLOAD_MAX_BYTES", str(4 * 1024 * 1024 * 1024)))


def _ensure_runtime_ready() -> None:
    try:
        validate_runtime_device()
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("", status_code=201)
def create_task(payload: TaskCreate) -> dict:
    try:
        video_id = extract_video_id(payload.url)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    existing_id = database.find_task_by_video_id(video_id)
    if existing_id:
        fields = payload.config.to_db_fields(only_set=True)
        if fields:
            database.update_task(existing_id, **fields)
        return database.get_task(existing_id)

    _ensure_runtime_ready()
    task_id = database.create_task(
        payload.url.strip(),
        task_id=video_id,
        **payload.config.to_db_fields(),
    )
    worker.enqueue(task_id)
    return database.get_task(task_id)


def _clean_upload_filename(filename: str | None) -> str:
    original = Path(filename or "").name.strip()
    if not original:
        raise HTTPException(status_code=422, detail="Video filename is required.")
    suffix = Path(original).suffix.lower()
    if suffix not in ALLOWED_VIDEO_SUFFIXES:
        raise HTTPException(status_code=422, detail="Unsupported video file type.")
    safe_stem = sanitize_text(Path(original).stem) or "video"
    return f"{safe_stem}{suffix}"


def _save_uploaded_file(file: UploadFile, destination: Path) -> int:
    total = 0
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as handle:
        while True:
            chunk = file.file.read(LOCAL_UPLOAD_CHUNK_SIZE)
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_LOCAL_UPLOAD_BYTES:
                destination.unlink(missing_ok=True)
                raise HTTPException(status_code=413, detail="Uploaded video is too large.")
            handle.write(chunk)
    if total == 0:
        destination.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail="Uploaded video is empty.")
    return total


@router.post("/upload", status_code=201)
def upload_local_video(
    config: str = Form(""),
    file: UploadFile = File(...),
) -> dict:
    cfg = TaskConfig.model_validate_json(config) if config else TaskConfig()
    asr_language = cfg.asr_language or "en"
    target_language = cfg.target_language or "zh"
    direction = f"{asr_language}-{target_language}"
    if direction not in LOCAL_UPLOAD_DIRECTIONS:
        raise HTTPException(status_code=422, detail="Unsupported local video direction.")

    _ensure_runtime_ready()
    original_name = Path(file.filename or "").name.strip()
    stored_name = _clean_upload_filename(original_name)
    task_id = str(uuid.uuid4())
    target_dir = upload_dir(WORKFOLDER, task_id)
    try:
        _save_uploaded_file(file, target_dir / stored_name)
    except HTTPException:
        remove_upload(WORKFOLDER, task_id)
        raise

    url = f"local://upload/{task_id}?direction={direction}&filename={quote(original_name)}"
    database.create_task(
        url,
        task_id=task_id,
        **cfg.to_db_fields(),
    )
    database.update_task(task_id, title=Path(original_name).stem)
    worker.enqueue(task_id)
    return database.get_task(task_id)


@router.post("/localdir", status_code=201)
def create_localdir_task(payload: LocaldirTaskCreate) -> dict:
    file_path = payload.file_path.strip()
    if not file_path:
        raise HTTPException(status_code=422, detail="File path is required.")

    source_file = Path(file_path)
    if not source_file.is_file():
        raise HTTPException(status_code=422, detail=f"File not found: {file_path}")

    suffix = source_file.suffix.lower()
    if suffix not in ALLOWED_VIDEO_SUFFIXES:
        raise HTTPException(status_code=422, detail=f"Unsupported video file type: {suffix}")

    cfg = payload.config
    asr_language = cfg.asr_language or "en"
    target_language = cfg.target_language or "zh"
    direction = f"{asr_language}-{target_language}"
    if direction not in LOCAL_UPLOAD_DIRECTIONS:
        raise HTTPException(status_code=422, detail="Unsupported direction.")

    _ensure_runtime_ready()

    task_id = str(uuid.uuid4())
    url = make_localdir_url(task_id, file_path, direction, filename=source_file.name)
    database.create_task(url, task_id=task_id, **cfg.to_db_fields())
    database.update_task(task_id, title=source_file.stem)
    worker.enqueue(task_id)
    return database.get_task(task_id)


@router.get("/current")
def current_task() -> dict | None:
    return database.get_current_task()


@router.patch("/{task_id}/config")
def update_task_config(task_id: str, payload: TaskConfig) -> dict:
    task = database.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found.")
    if task["status"] == "running":
        raise HTTPException(status_code=409, detail="Cannot update config of a running task.")
    fields = payload.to_db_fields(only_set=True)
    if fields:
        database.update_task(task_id, **fields)
    return database.get_task(task_id)


@router.get("")
def list_tasks(limit: int = 100, offset: int = 0) -> dict:
    return {"tasks": database.list_tasks(limit=limit, offset=offset), "total": database.count_tasks()}


@router.get("/{task_id}")
def task_detail(task_id: str) -> dict:
    task = database.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found.")
    return task


def _is_inside_workfolder(path: Path) -> bool:
    workfolder = WORKFOLDER.resolve()
    try:
        path.resolve().relative_to(workfolder)
    except ValueError:
        return False
    return True


def _purge_task(task: dict) -> None:
    session_path = task.get("session_path")
    if session_path:
        session_dir = Path(session_path)
        if session_dir.exists() and _is_inside_workfolder(session_dir):
            shutil.rmtree(session_dir)
    log_file = database.log_path(task["id"])
    if log_file.exists():
        log_file.unlink()
    database.delete_task(task["id"])


@router.delete("/{task_id}", status_code=204)
def delete_task(task_id: str) -> Response:
    task = database.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found.")
    if task["status"] == "running":
        raise HTTPException(status_code=409, detail="Cannot delete a running task.")
    _purge_task(task)
    if is_local_upload_url(task["url"]):
        remove_upload(WORKFOLDER, task["id"])
    # Drop any pending stop event so the worker's _stop_events dict
    # doesn't leak entries for deleted tasks.
    worker.cleanup_stop(task_id)
    return Response(status_code=204)


@router.post("/{task_id}/rerun")
def rerun_task(task_id: str) -> dict:
    task = database.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found.")
    if task["status"] == "running":
        raise HTTPException(status_code=409, detail="Cannot rerun a running task.")

    _ensure_runtime_ready()
    url = task["url"]
    preserved_config = TaskConfig.from_task_dict(task)
    _purge_task(task)
    new_id = database.create_task(url, task_id=task_id, **preserved_config.to_db_fields())
    worker.enqueue(new_id)
    return database.get_task(new_id)


@router.post("/{task_id}/resume")
def resume_task(task_id: str) -> dict:
    task = database.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found.")
    if task["status"] != "failed":
        raise HTTPException(status_code=409, detail="Only failed tasks can be resumed.")
    _ensure_runtime_ready()
    database.reset_failed_for_resume(task_id)
    worker.enqueue(task_id)
    return database.get_task(task_id)


@router.post("/{task_id}/stop")
def stop_task(task_id: str) -> dict:
    task = database.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found.")
    status = task["status"]
    if status not in ("running", "queued"):
        raise HTTPException(
            status_code=409,
            detail="Only running or queued tasks can be stopped.",
        )
    worker.request_stop(task_id)
    # For queued tasks the worker will short-circuit, but we mark them as failed
    # immediately so the UI reflects the change without waiting for dequeue.
    if status == "queued":
        mark_task_as_stopped(task_id, error_message=STOPPED_BEFORE_START_MESSAGE)
    return database.get_task(task_id)


@router.post("/{task_id}/rerun-stage/{stage_name}")
def rerun_stage(task_id: str, stage_name: str) -> dict:
    from ..stages import STAGE_NAMES

    task = database.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found.")
    if task["status"] == "running":
        raise HTTPException(status_code=409, detail="Cannot rerun stage of a running task.")
    if stage_name not in STAGE_NAMES:
        raise HTTPException(status_code=422, detail=f"Unknown stage: {stage_name}")
    _ensure_runtime_ready()
    database.reset_stage_for_rerun(task_id, stage_name)
    worker.enqueue(task_id)
    return database.get_task(task_id)


@router.post("/{task_id}/rerun-single-stage/{stage_name}")
def rerun_single_stage(task_id: str, stage_name: str) -> dict:
    from ..stages import STAGE_NAMES

    task = database.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found.")
    if task["status"] == "running":
        raise HTTPException(status_code=409, detail="Cannot rerun stage of a running task.")
    if stage_name not in STAGE_NAMES:
        raise HTTPException(status_code=422, detail=f"Unknown stage: {stage_name}")
    _ensure_runtime_ready()
    database.reset_single_stage_for_rerun(task_id, stage_name)
    run_task_single_stage(task_id, stage_name)
    return database.get_task(task_id)


@router.post("/{task_id}/clear-stage/{stage_name}")
def clear_stage_output(task_id: str, stage_name: str) -> dict:
    from ..stages import STAGE_NAMES

    task = database.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found.")
    if task["status"] == "running":
        raise HTTPException(status_code=409, detail="Cannot clear output of a running task.")
    if stage_name not in STAGE_NAMES:
        raise HTTPException(status_code=422, detail=f"Unknown stage: {stage_name}")
    _clear_stage_output(task, stage_name)
    database.update_stage(
        task_id,
        stage_name,
        status="pending",
        started_at=None,
        completed_at=None,
        progress=None,
        last_message=None,
        error_message=None,
    )
    return database.get_task(task_id)


def _clear_stage_output(task: dict, stage_name: str) -> None:
    """Delete cached output files for a single stage so it actually re-runs."""
    from ..stages import resolve_stage_outputs

    session_path = task.get("session_path")
    if not session_path:
        return
    session = Path(session_path)
    if not session.exists():
        return

    for p in resolve_stage_outputs(session, stage_name):
        if p.exists():
            p.unlink()


@router.get("/{task_id}/log", response_class=PlainTextResponse)
def task_log(task_id: str) -> str:
    task = database.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found.")
    path = database.log_path(task_id)
    return path.read_text(encoding="utf-8") if path.exists() else ""


@router.get("/{task_id}/artifact/final-video")
def final_video(task_id: str, download: bool = False) -> FileResponse:
    task = database.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found.")
    final_path = task.get("final_video_path")
    if not final_path or not Path(final_path).exists():
        raise HTTPException(status_code=404, detail="Final video is not available.")
    name = Path(final_path).name
    if download:
        return FileResponse(final_path, media_type="video/mp4", filename=name)
    headers = {"Content-Disposition": f'inline; filename="{name}"'}
    return FileResponse(final_path, media_type="video/mp4", headers=headers)
