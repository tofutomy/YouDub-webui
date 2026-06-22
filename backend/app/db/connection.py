"""SQLite connection helpers and shared constants."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

from .. import config
from ..config import ensure_runtime_dirs


ACTIVE_STATUSES = ("queued", "running")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect() -> sqlite3.Connection:
    ensure_runtime_dirs()
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn
