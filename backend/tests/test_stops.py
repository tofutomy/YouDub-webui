from __future__ import annotations

from backend.app import config, database, stops


def _init_db(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "stops.sqlite")
    database.init_db()


def test_mark_task_as_stopped_sets_flag_for_queued_task(monkeypatch, tmp_path):
    _init_db(monkeypatch, tmp_path)
    task_id = database.create_task("https://www.youtube.com/watch?v=stops12345")

    stops.mark_task_as_stopped(task_id, error_message=stops.STOPPED_BEFORE_START_MESSAGE)

    task = database.get_task(task_id)
    assert task["status"] == "failed"
    assert task["error_message"] == stops.STOPPED_BEFORE_START_MESSAGE
    assert task["stop_requested"] == 1
    # Queued task never ran: stages stay pending.
    assert all(stage["status"] == "pending" for stage in task["stages"])


def test_mark_task_as_stopped_marks_active_stage_for_running_task(monkeypatch, tmp_path):
    _init_db(monkeypatch, tmp_path)
    task_id = database.create_task("https://www.youtube.com/watch?v=stops67890")
    database.update_task(task_id, status="running", started_at=database.now_iso())
    database.update_stage(task_id, "download", status="running", started_at=database.now_iso())

    stops.mark_task_as_stopped(task_id, error_message=stops.STOPPED_MESSAGE)

    task = database.get_task(task_id)
    assert task["status"] == "failed"
    assert task["error_message"] == stops.STOPPED_MESSAGE
    assert task["stop_requested"] == 1
    # The currently running stage is marked as failed.
    download_stage = next(s for s in task["stages"] if s["name"] == "download")
    assert download_stage["status"] == "failed"
    assert download_stage["error_message"] == stops.STOPPED_MESSAGE


def test_mark_task_as_stopped_does_not_mark_succeeded_stage_when_stopped_at_boundary(monkeypatch, tmp_path):
    """When stop happens between stages, current_stage points to the last
    completed (succeeded) stage — it must not be overwritten as failed."""
    _init_db(monkeypatch, tmp_path)
    task_id = database.create_task("https://www.youtube.com/watch?v=boundarystop")
    database.update_task(
        task_id,
        status="running",
        started_at=database.now_iso(),
        current_stage="download",
    )
    database.update_stage(
        task_id,
        "download",
        status="succeeded",
        started_at=database.now_iso(),
        completed_at=database.now_iso(),
    )

    stops.mark_task_as_stopped(task_id)

    task = database.get_task(task_id)
    assert task["status"] == "failed"
    assert task["stop_requested"] == 1
    download_stage = next(s for s in task["stages"] if s["name"] == "download")
    assert download_stage["status"] == "succeeded"


def test_mark_task_as_stopped_is_noop_for_unknown_task(monkeypatch, tmp_path):
    _init_db(monkeypatch, tmp_path)
    # Should not raise even though the task does not exist.
    stops.mark_task_as_stopped("missing-id")
    assert database.get_task("missing-id") is None

