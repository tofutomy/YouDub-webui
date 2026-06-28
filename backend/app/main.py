"""YouDub API application entry point.

This module is responsible **only** for:
- Application lifespan (startup / shutdown hooks)
- CORS middleware configuration
- Mounting the API routers defined in ``routers/``
- The ``/api/health`` probe

All business-logic endpoints live in the ``routers`` sub-package.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from . import database, worker
from .config import ensure_runtime_dirs
from .pipeline import run_task
from .routers._common import cors_origin_regex, cors_origins
from .routers.cookies import router as cookies_router
from .routers.settings import router as settings_router
from .routers.tasks import router as tasks_router
from .routers.translate_providers import router as providers_router

import logging

logger = logging.getLogger(__name__)


def _cleanup_orphan_spawn_children() -> None:
    """终止当前进程的所有 multiprocessing.spawn 孤儿子进程。

    在 Windows 上，HuggingFace accelerate 的 ``infer_auto_device_map`` 通过
    ``multiprocessing.spawn`` 创建子进程来执行设备映射计算。如果 uvicorn
    在任务执行期间被 reload，这些子进程可能不会被 ``release_model()`` 清理，
    导致它们持有 CUDA context 和大量显存。

    此函数在应用启动时调用，确保每次启动都是干净的状态。
    """
    try:
        import psutil
    except ImportError:
        return

    try:
        current = psutil.Process()
        for child in current.children(recursive=True):
            try:
                cmdline = child.cmdline()
                if any("multiprocessing.spawn" in part for part in cmdline):
                    child.terminate()
                    try:
                        child.wait(timeout=5)
                    except psutil.TimeoutExpired:
                        child.kill()
                    logger.info(
                        "Startup cleanup: terminated orphan spawn child PID=%d",
                        child.pid,
                    )
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    except Exception as exc:
        logger.warning("Startup spawn child cleanup skipped: %s", exc)


# ---------------------------------------------------------------------------
# Lifespan — runs once on startup, yields once, then runs on shutdown.
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise database schema, backfill metadata, and start the worker."""
    # 在启动时清理上一次运行残留的 multiprocessing.spawn 子进程。
    # 这些子进程由 HuggingFace accelerate 的 device_map 推理创建，
    # 持有 CUDA context 和大量显存，必须在启动时清理。
    _cleanup_orphan_spawn_children()
    ensure_runtime_dirs()
    database.init_db()
    database.backfill_titles_from_metadata()
    database.fail_stale_active_tasks()
    worker.start(run_task)
    yield


# ---------------------------------------------------------------------------
# Application assembly
# ---------------------------------------------------------------------------

app = FastAPI(title="YouDub API", lifespan=lifespan)

# CORS — allow the Next.js dev server (localhost:3000 / 3080) and any
# additional origins configured via environment variables.
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins(),
    allow_origin_regex=cors_origin_regex(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API routers — each router owns a prefix and its own tag.
app.include_router(tasks_router)
app.include_router(cookies_router)
app.include_router(settings_router)
app.include_router(providers_router)


@app.get("/api/health")
def health() -> dict[str, str]:
    """Liveness probe used by the frontend and monitoring."""
    return {"status": "ok"}
