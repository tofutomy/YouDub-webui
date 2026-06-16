"""Single-thread FIFO worker that runs queued tasks one at a time."""

from __future__ import annotations

import queue
import threading
from typing import Callable

from . import database
from .stops import STOPPED_BEFORE_START_MESSAGE, mark_task_as_stopped


_queue: "queue.Queue[str]" = queue.Queue()
_thread: threading.Thread | None = None
_lock = threading.Lock()

# Per-task cooperative-cancellation state.
# `_stop_events` maps task_id -> Event. The event is set when a stop is requested.
# `_current_task_id` tracks the task currently being executed by the worker loop.
_stop_events: dict[str, threading.Event] = {}
_current_task_id: str | None = None


def enqueue(task_id: str) -> None:
    _queue.put(task_id)


def current_task_id() -> str | None:
    with _lock:
        return _current_task_id


def request_stop(task_id: str) -> bool:
    """Mark a task as stop-requested.

    Returns True when the stop flag was created or already set, False on internal
    failure (which should not normally happen).
    """
    with _lock:
        event = _stop_events.get(task_id)
        if event is None:
            event = threading.Event()
            _stop_events[task_id] = event
        event.set()
        return True


def is_stop_requested(task_id: str) -> bool:
    with _lock:
        event = _stop_events.get(task_id)
    return event is not None and event.is_set()


def consume_stop_event(task_id: str) -> bool:
    """Remove the stop event for a task and return whether it was set."""
    with _lock:
        event = _stop_events.pop(task_id, None)
    return event is not None and event.is_set()


def _loop(runner: Callable[[str], None]) -> None:
    global _current_task_id
    while True:
        task_id = _queue.get()
        try:
            # Skip tasks that were cancelled while waiting in the queue.
            if consume_stop_event(task_id):
                mark_task_as_stopped(task_id, error_message=STOPPED_BEFORE_START_MESSAGE)
                continue
            with _lock:
                _current_task_id = task_id
            runner(task_id)
        finally:
            with _lock:
                _current_task_id = None
            # Always drop the stop flag for a finished task so a future rerun is not
            # silently aborted by a stale request.
            consume_stop_event(task_id)
            _queue.task_done()


def start(runner: Callable[[str], None]) -> None:
    global _thread
    with _lock:
        if _thread is not None:
            return
        _thread = threading.Thread(target=_loop, args=(runner,), daemon=True)
        _thread.start()
    pending = [t for t in database.list_tasks() if t["status"] == "queued"]
    for task in reversed(pending):
        _queue.put(task["id"])
