from __future__ import annotations

import threading

from backend.app import database, worker


def test_worker_picks_up_pending_and_new_tasks(monkeypatch, tmp_path):
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "worker.sqlite")
    database.init_db()
    pre_queued = [
        database.create_task(f"https://www.youtube.com/watch?v=v{i:011d}") for i in range(2)
    ]

    executed: list[str] = []
    target = len(pre_queued) + 1
    done = threading.Event()

    def runner(task_id: str) -> None:
        executed.append(task_id)
        if len(executed) == target:
            done.set()

    monkeypatch.setattr(worker, "_thread", None)
    worker.start(runner)
    worker.enqueue("late-task")

    assert done.wait(timeout=2.0)
    assert executed[:2] == pre_queued
    assert executed[-1] == "late-task"


def test_request_stop_sets_and_consumes_flag():
    assert worker.is_stop_requested("missing") is False

    assert worker.request_stop("abc") is True
    assert worker.is_stop_requested("abc") is True
    # Idempotent: calling again is a no-op that still returns True.
    assert worker.request_stop("abc") is True
    assert worker.is_stop_requested("abc") is True

    # consume returns the prior set state and removes the entry.
    assert worker.consume_stop_event("abc") is True
    assert worker.is_stop_requested("abc") is False
    # Second consume: nothing was set, so False.
    assert worker.consume_stop_event("abc") is False


def test_queued_task_marked_failed_when_stopped_before_dequeue(monkeypatch, tmp_path):
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "worker-stop.sqlite")
    database.init_db()
    task_id = database.create_task("https://www.youtube.com/watch?v=stopped")

    # The task is queued. User requests a stop before the worker picks it up.
    worker.request_stop(task_id)

    executed: list[str] = []
    runner_called = threading.Event()

    def runner(tid: str) -> None:
        executed.append(tid)
        runner_called.set()

    monkeypatch.setattr(worker, "_thread", None)
    worker.start(runner)
    # start() enqueues pending tasks from the database; do not enqueue again.

    # Wait for the worker to observe the stop and mark the task as failed.
    import time
    deadline = time.time() + 3.0
    task = None
    while time.time() < deadline:
        task = database.get_task(task_id)
        if task and task["status"] == "failed":
            break
        time.sleep(0.05)

    assert task is not None
    assert task["status"] == "failed"
    assert "Stopped by user" in (task["error_message"] or "")
    # The runner is never invoked because the worker short-circuits cancelled work.
    assert runner_called.is_set() is False
    assert executed == []
    # Frontend relies on this flag to distinguish user-stops from errors.
    assert task["stop_requested"] == 1

    # Cleanup the worker thread for the next test.
    monkeypatch.setattr(worker, "_thread", None)
    worker.consume_stop_event(task_id)

