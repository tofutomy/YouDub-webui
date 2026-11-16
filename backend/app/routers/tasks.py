"""Task lifecycle REST endpoints.

Covers task creation (URL / upload / localdir), listing, detail, config
update, delete, rerun, resume, stop, stage rerun/clear, log and final
video artifact.
"""

from __future__ import annotations

import re
import shutil
import sqlite3
import threading
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import quote

from fastapi import APIRouter, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import ValidationError

from .. import database, worker
from ..adapters.local_video import remove_upload, upload_dir
from ..config import WORKFOLDER
from ..pipeline import run_task, run_task_single_stage
from ..runtime_checks import validate_runtime_device
from ..sanitize import sanitize_text
from ..stops import STOPPED_BEFORE_START_MESSAGE, mark_task_as_stopped
from ..task_config import TaskConfig
from ..languages import (
    LanguageConfigError,
    language_pair_to_direction,
    normalize_config_fields,
)
from ..youtube import (
    extract_video_id,
    is_local_upload_url,
    is_localdir_url,
    make_localdir_url,
)
from ..schemas import (
    LocaldirTaskCreate,
    ManagedYouTubeTaskCreate,
    TaskCreate,
)

router = APIRouter(prefix="/api/tasks", tags=["tasks"])

ALLOWED_VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".mkv", ".webm", ".avi", ".flv", ".wmv"}
LOCAL_UPLOAD_CHUNK_SIZE = 1024 * 1024
MAX_LOCAL_UPLOAD_BYTES = int(__import__("os").getenv("LOCAL_UPLOAD_MAX_BYTES", str(4 * 1024 * 1024 * 1024)))


def _ensure_runtime_ready() -> None:
    """Validate that the configured devices (CUDA / MPS / CPU) are available.

    Raises HTTP 409 if a required device is missing, so the client can
    surface the error before queueing a task that would fail at runtime.
    """
    try:
        validate_runtime_device()
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


def _require_task(task_id: str) -> dict[str, Any]:
    """Fetch a task or raise HTTP 404."""
    task = database.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found.")
    return task


def _config_fields_or_422(
    config: TaskConfig,
    *,
    only_set: bool = False,
    current_task: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """将任务配置规范化为数据库字段，语言方向错误统一返回 422。"""
    try:
        return normalize_config_fields(config, only_set=only_set, current_task=current_task)
    except LanguageConfigError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("", status_code=201)
def create_task(payload: TaskCreate) -> dict[str, Any]:
    """Create a task from a video URL (YouTube / Bilibili / etc.).

    If the same video was already submitted, the existing task is returned
    and its configuration is updated with any explicitly-set fields from
    this request (dedup by video ID).
    """
    try:
        video_id = extract_video_id(payload.url)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    # Dedup: reuse the existing task for the same video.
    existing_id = database.find_task_by_video_id(video_id)
    if existing_id:
        existing_task = _require_task(existing_id)
        fields = _config_fields_or_422(payload.config, only_set=True, current_task=existing_task)
        if fields:
            database.update_task(existing_id, **fields)
        return _require_task(existing_id)

    _ensure_runtime_ready()
    task_id = database.create_task(
        payload.url.strip(),
        task_id=video_id,
        **_config_fields_or_422(payload.config),
    )
    worker.enqueue(task_id)
    return _require_task(task_id)


@router.post("/managed-youtube", status_code=201)
def create_managed_youtube_task(payload: ManagedYouTubeTaskCreate, request: Request) -> dict[str, Any]:
    """为本机制作工作台创建共享文件的 YouTube 任务。"""
    if request.client is None or request.client.host not in {"127.0.0.1", "::1"} or request.headers.get("origin"):
        raise HTTPException(status_code=403, detail="Managed tasks require a local server request.")
    if not re.fullmatch(r"[0-9a-f]{32}", payload.job_id):
        raise HTTPException(status_code=422, detail="Invalid workbench job ID.")
    if not re.fullmatch(r"https://www\.youtube\.com/watch\?v=[A-Za-z0-9_-]{11}", payload.url):
        raise HTTPException(status_code=422, detail="Expected a canonical YouTube video URL.")
    workdir = Path(payload.work_directory)
    profile = Path(payload.firefox_profile)
    if not workdir.is_absolute() or not profile.is_absolute() or profile.name != "youtube-firefox-profile":
        raise HTTPException(status_code=422, detail="Invalid work directory or dedicated Firefox profile.")
    paths = {"storage_workdir": str(workdir.resolve()), "firefox_profile": str(profile.resolve())}
    task_id = f"lootgamevideo-{payload.job_id}"

    def existing_task():
        existing = database.get_task(task_id)
        if existing and (existing["url"] != payload.url or any(existing[key] != value for key, value in paths.items())):
            raise HTTPException(status_code=409, detail="This workbench job is already bound to a different source or directory.")
        return existing

    if existing := existing_task():
        return existing
    _ensure_runtime_ready()
    try:
        database.create_task(payload.url, task_id=task_id, **paths, **_config_fields_or_422(payload.config))
    except sqlite3.IntegrityError:
        if existing := existing_task():
            return existing
        raise
    worker.enqueue(task_id)
    return _require_task(task_id)


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
) -> dict[str, Any]:
    """Upload a local video file and create a task.

    The ``config`` form field is a JSON-serialised ``TaskConfig``.  The
    uploaded file is saved to ``WORKFOLDER/_uploads/<task_id>/`` and
    automatically transcoded to h.264+aac MP4 by the pipeline.
    """
    try:
        cfg = TaskConfig.model_validate_json(config) if config else TaskConfig()
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors(include_context=False, include_input=False)) from exc
    fields = _config_fields_or_422(cfg)
    direction = language_pair_to_direction(fields["asr_language"], fields["target_language"])

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
        **fields,
    )
    database.update_task(task_id, title=Path(original_name).stem)
    worker.enqueue(task_id)
    return _require_task(task_id)


@router.post("/localdir", status_code=201)
def create_localdir_task(payload: LocaldirTaskCreate) -> dict[str, Any]:
    """Create a task from a local file path (no upload, reads in-place)."""
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
    fields = _config_fields_or_422(cfg)
    direction = language_pair_to_direction(fields["asr_language"], fields["target_language"])

    _ensure_runtime_ready()

    task_id = str(uuid.uuid4())
    url = make_localdir_url(task_id, file_path, direction, filename=source_file.name)
    database.create_task(url, task_id=task_id, **fields)
    database.update_task(task_id, title=source_file.stem)
    worker.enqueue(task_id)
    return _require_task(task_id)


@router.get("/current")
def current_task() -> dict[str, Any] | None:
    return database.get_current_task()


@router.patch("/{task_id}/config")
def update_task_config(task_id: str, payload: TaskConfig) -> dict[str, Any]:
    task = _require_task(task_id)
    if task["status"] == "running":
        raise HTTPException(status_code=409, detail="Cannot update config of a running task.")
    fields = _config_fields_or_422(payload, only_set=True, current_task=task)
    if fields:
        database.update_task(task_id, **fields)
    return _require_task(task_id)


@router.get("")
def list_tasks(limit: int = 100, offset: int = 0) -> dict[str, Any]:
    return {"tasks": database.list_tasks(limit=limit, offset=offset), "total": database.count_tasks()}


@router.get("/{task_id}")
def task_detail(task_id: str) -> dict[str, Any]:
    return _require_task(task_id)


def _is_inside_workfolder(path: Path) -> bool:
    workfolder = WORKFOLDER.resolve()
    try:
        path.resolve().relative_to(workfolder)
    except ValueError:
        return False
    return True


def _purge_task(task: dict) -> None:
    """Remove a task's session directory, log file, and database row."""
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
    task = _require_task(task_id)
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
def rerun_task(task_id: str) -> dict[str, Any]:
    """Delete the existing task and re-create it from scratch.

    Preserves the task's current configuration so user settings are not
    lost across reruns.
    """
    task = _require_task(task_id)
    if task["status"] == "running":
        raise HTTPException(status_code=409, detail="Cannot rerun a running task.")

    _ensure_runtime_ready()
    if task.get("storage_workdir"):
        # 保留共享目录和原片，重跑阶段时继续复用该任务。
        database.reset_stage_for_rerun(task_id, "download")
        worker.cleanup_stop(task_id)
        worker.enqueue(task_id)
        return _require_task(task_id)
    url = task["url"]
    # Preserve config before purging so the new task inherits user settings.
    preserved_config = TaskConfig.from_task_dict(task)
    _purge_task(task)
    new_id = database.create_task(url, task_id=task_id, **preserved_config.to_db_fields())
    # 清理残留的 stop event，防止重跑时被旧的 stop 请求立即终止
    worker.cleanup_stop(new_id)
    worker.enqueue(new_id)
    return _require_task(new_id)


@router.post("/{task_id}/resume")
def resume_task(task_id: str) -> dict[str, Any]:
    task = _require_task(task_id)
    if task["status"] != "failed":
        raise HTTPException(status_code=409, detail="Only failed tasks can be resumed.")
    _ensure_runtime_ready()
    database.reset_failed_for_resume(task_id)
    # 清理残留的 stop event，防止恢复时被旧的 stop 请求立即终止
    worker.cleanup_stop(task_id)
    worker.enqueue(task_id)
    return _require_task(task_id)


@router.post("/{task_id}/stop")
def stop_task(task_id: str) -> dict[str, Any]:
    task = _require_task(task_id)
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
    return _require_task(task_id)


@router.post("/{task_id}/rerun-stage/{stage_name}")
def rerun_stage(task_id: str, stage_name: str) -> dict[str, Any]:
    from ..stages import STAGE_NAMES

    task = _require_task(task_id)
    if task["status"] == "running":
        raise HTTPException(status_code=409, detail="Cannot rerun stage of a running task.")
    if stage_name not in STAGE_NAMES:
        raise HTTPException(status_code=422, detail=f"Unknown stage: {stage_name}")
    _ensure_runtime_ready()
    database.reset_stage_for_rerun(task_id, stage_name)
    # 清理残留的 stop event，防止级联重跑时被旧的 stop 请求立即终止
    worker.cleanup_stop(task_id)
    worker.enqueue(task_id)
    return _require_task(task_id)


@router.post("/{task_id}/rerun-single-stage/{stage_name}")
def rerun_single_stage(task_id: str, stage_name: str) -> dict[str, Any]:
    from ..stages import STAGE_NAMES

    task = _require_task(task_id)
    if task["status"] == "running":
        raise HTTPException(status_code=409, detail="Cannot rerun stage of a running task.")
    if stage_name not in STAGE_NAMES:
        raise HTTPException(status_code=422, detail=f"Unknown stage: {stage_name}")
    _ensure_runtime_ready()
    database.reset_single_stage_for_rerun(task_id, stage_name)
    # Run in a background thread so the HTTP response returns immediately.
    # Long-running stages (e.g. translate) would otherwise block the event loop.
    threading.Thread(
        target=run_task_single_stage,
        args=(task_id, stage_name),
        daemon=True,
    ).start()
    return _require_task(task_id)


@router.post("/{task_id}/clear-stage/{stage_name}")
def clear_stage_output(task_id: str, stage_name: str) -> dict[str, Any]:
    from ..stages import STAGE_NAMES

    task = _require_task(task_id)
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
    return _require_task(task_id)


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
    _require_task(task_id)
    path = database.log_path(task_id)
    return path.read_text(encoding="utf-8") if path.exists() else ""


@router.get("/{task_id}/artifact/final-video")
def final_video(task_id: str, download: bool = False) -> FileResponse:
    task = _require_task(task_id)
    final_path = task.get("final_video_path")
    if not final_path or not Path(final_path).exists():
        raise HTTPException(status_code=404, detail="Final video is not available.")
    name = Path(final_path).name
    if download:
        return FileResponse(final_path, media_type="video/mp4", filename=name)
    headers = {"Content-Disposition": f'inline; filename="{name}"'}
    return FileResponse(final_path, media_type="video/mp4", headers=headers)
