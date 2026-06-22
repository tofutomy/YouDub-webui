from __future__ import annotations

import re

import pytest
from fastapi.testclient import TestClient

from backend.app import config, database
from backend.app import main
from backend.app import worker
from backend.app.routers import cookies as cookies_router
from backend.app.routers import settings as settings_router
from backend.app.routers import tasks as tasks_router


def configure_tmp_runtime(monkeypatch, tmp_path):
    workfolder = tmp_path / "workfolder"
    workfolder.mkdir()
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "test.sqlite")
    monkeypatch.setattr(config, "YOUTUBE_COOKIE_PATH", tmp_path / "cookies" / "youtube.txt")
    monkeypatch.setattr(cookies_router, "YOUTUBE_COOKIE_PATH", tmp_path / "cookies" / "youtube.txt")
    monkeypatch.setattr(config, "WORKFOLDER", workfolder)
    monkeypatch.setattr(tasks_router, "WORKFOLDER", workfolder)
    monkeypatch.setattr(config, "LOG_DIR", log_dir)
    monkeypatch.setattr(worker, "start", lambda runner: None)
    monkeypatch.setattr(worker, "enqueue", lambda task_id: None)
    monkeypatch.setattr(main.worker, "enqueue", lambda task_id: None)
    monkeypatch.setattr(tasks_router.worker, "enqueue", lambda task_id: None)
    database.init_db()


def test_openai_key_is_masked(monkeypatch, tmp_path):
    configure_tmp_runtime(monkeypatch, tmp_path)
    database.save_openai_settings("https://example.com/v1", "sk-test-secret", "test-model")
    client = TestClient(main.app)

    response = client.get("/api/settings/openai")

    assert response.status_code == 200
    body = response.json()
    assert body["api_key"] == "********"
    assert body["has_api_key"] is True
    assert "sk-test-secret" not in str(body)


def test_create_existing_task_updates_direction(monkeypatch, tmp_path):
    configure_tmp_runtime(monkeypatch, tmp_path)
    enqueued: list[str] = []
    monkeypatch.setattr(main.worker, "enqueue", lambda task_id: enqueued.append(task_id))
    task_id = database.create_task(
        "https://www.bilibili.com/video/BV1UNAbzpEzR",
        task_id="BV1UNAbzpEzR",
        validate_translation=True,
    )
    client = TestClient(main.app)

    response = client.post(
        "/api/tasks",
        json={
            "url": "https://www.bilibili.com/video/BV1UNAbzpEzR",
            "config": {
                "asr_language": "en",
                "target_language": "zh",
                "asr_model": "",
                "translate_mode": "batch",
                "validate_translation": False,
                "tts_mode": "hifi_clone",
            },
        },
    )

    assert response.status_code == 201
    body = response.json()
    assert body["id"] == task_id
    assert body["asr_language"] == "en"
    assert body["target_language"] == "zh"
    assert body["translate_mode"] == "batch"
    assert body["validate_translation"] == 0
    assert body["tts_mode"] == "hifi_clone"
    task = database.get_task(task_id)
    assert task["asr_language"] == "en"
    assert task["validate_translation"] == 0
    assert enqueued == []


def test_masked_openai_key_is_not_saved_back(monkeypatch, tmp_path):
    configure_tmp_runtime(monkeypatch, tmp_path)
    database.save_openai_settings("https://example.com/v1", "sk-test-secret", "test-model")
    client = TestClient(main.app)

    response = client.post(
        "/api/settings/openai",
        json={
            "base_url": "https://example.com/v1",
            "api_key": "********",
            "clear_api_key": False,
            "model": "next-model",
        },
    )

    assert response.status_code == 200
    settings = database.get_openai_settings()
    assert settings["api_key"] == "sk-test-secret"
    assert settings["model"] == "next-model"


def test_openai_key_can_be_cleared(monkeypatch, tmp_path):
    configure_tmp_runtime(monkeypatch, tmp_path)
    database.save_openai_settings("https://example.com/v1", "sk-test-secret", "test-model")
    client = TestClient(main.app)

    response = client.post(
        "/api/settings/openai",
        json={
            "base_url": "https://example.com/v1",
            "api_key": "",
            "clear_api_key": True,
            "model": "next-model",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["api_key"] == ""
    assert body["has_api_key"] is False
    settings = database.get_openai_settings()
    assert settings["api_key"] == ""
    assert settings["model"] == "next-model"


@pytest.mark.parametrize(
    ("api_key", "expected"),
    [
        ("", "sk-test-secret"),
        ("********", "sk-test-secret"),
        ("sk-new", "sk-new"),
    ],
)
def test_openai_key_save_modes_without_clear(monkeypatch, tmp_path, api_key, expected):
    configure_tmp_runtime(monkeypatch, tmp_path)
    database.save_openai_settings("https://example.com/v1", "sk-test-secret", "test-model")
    client = TestClient(main.app)

    response = client.post(
        "/api/settings/openai",
        json={
            "base_url": "https://example.com/v1",
            "api_key": api_key,
            "clear_api_key": False,
            "model": "next-model",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["api_key"] == "********"
    assert body["has_api_key"] is True
    settings = database.get_openai_settings()
    assert settings["api_key"] == expected
    assert settings["model"] == "next-model"


def test_cookie_response_does_not_leak_content(monkeypatch, tmp_path):
    configure_tmp_runtime(monkeypatch, tmp_path)
    client = TestClient(main.app)

    response = client.post("/api/cookies/youtube", json={"content": "secret-cookie-content"})

    assert response.status_code == 200
    assert response.json()["content"] == ""
    assert "secret-cookie-content" not in response.text


def test_task_id_is_video_id_and_dedupes_existing(monkeypatch, tmp_path):
    configure_tmp_runtime(monkeypatch, tmp_path)
    enqueued: list[str] = []
    monkeypatch.setattr(main.worker, "enqueue", lambda task_id: enqueued.append(task_id))
    client = TestClient(main.app)
    payload = {"url": "https://www.youtube.com/watch?v=abcdefghijk"}

    first = client.post("/api/tasks", json=payload)
    second = client.post("/api/tasks", json=payload)

    assert first.status_code == 201
    assert second.status_code == 201
    assert first.json()["id"] == "abcdefghijk"
    assert second.json()["id"] == "abcdefghijk"
    assert enqueued == ["abcdefghijk"]


def test_different_videos_create_separate_tasks(monkeypatch, tmp_path):
    configure_tmp_runtime(monkeypatch, tmp_path)
    enqueued: list[str] = []
    monkeypatch.setattr(main.worker, "enqueue", lambda task_id: enqueued.append(task_id))
    client = TestClient(main.app)

    a = client.post("/api/tasks", json={"url": "https://www.youtube.com/watch?v=abcdefghijk"})
    b = client.post("/api/tasks", json={"url": "https://youtu.be/zyxwvutsrqp"})

    assert a.json()["id"] == "abcdefghijk"
    assert b.json()["id"] == "zyxwvutsrqp"
    assert enqueued == ["abcdefghijk", "zyxwvutsrqp"]


def test_list_tasks_returns_history_newest_first(monkeypatch, tmp_path):
    configure_tmp_runtime(monkeypatch, tmp_path)
    older = database.create_task("https://www.youtube.com/watch?v=oldvideoidx")
    newer = database.create_task("https://www.youtube.com/watch?v=newvideoidx")
    client = TestClient(main.app)

    response = client.get("/api/tasks")

    assert response.status_code == 200
    body = response.json()
    ids = [task["id"] for task in body["tasks"]]
    assert ids == [newer, older]
    assert "stages" not in body["tasks"][0]
    assert set(body["tasks"][0].keys()) >= {"id", "url", "title", "status", "final_video_path"}


def test_task_detail_includes_stage_progress(monkeypatch, tmp_path):
    configure_tmp_runtime(monkeypatch, tmp_path)
    task_id = database.create_task("https://www.youtube.com/watch?v=progressapi")
    database.update_stage(task_id, "tts", progress=42, last_message="Prepared 21/50 TTS clips")
    client = TestClient(main.app)

    response = client.get(f"/api/tasks/{task_id}")

    assert response.status_code == 200
    stages = {stage["name"]: stage for stage in response.json()["stages"]}
    assert stages["tts"]["progress"] == 42
    assert stages["tts"]["last_message"] == "Prepared 21/50 TTS clips"


def test_delete_task_removes_session_log_and_record(monkeypatch, tmp_path):
    configure_tmp_runtime(monkeypatch, tmp_path)
    task_id = database.create_task("https://www.youtube.com/watch?v=delvideoidx", task_id="delvideoidx")
    session = config.WORKFOLDER / "uploader" / "title__delvideoidx"
    (session / "media").mkdir(parents=True)
    (session / "media" / "video_source.mp4").write_bytes(b"mp4")
    database.update_task(task_id, session_path=str(session))
    log_file = database.log_path(task_id)
    log_file.write_text("hello", encoding="utf-8")

    client = TestClient(main.app)
    response = client.delete(f"/api/tasks/{task_id}")

    assert response.status_code == 204
    assert database.get_task(task_id) is None
    assert not session.exists()
    assert not log_file.exists()


def test_delete_task_returns_404_for_unknown(monkeypatch, tmp_path):
    configure_tmp_runtime(monkeypatch, tmp_path)
    client = TestClient(main.app)

    response = client.delete("/api/tasks/does-not-exist")

    assert response.status_code == 404


def test_delete_task_rejects_running_task(monkeypatch, tmp_path):
    configure_tmp_runtime(monkeypatch, tmp_path)
    task_id = database.create_task("https://www.youtube.com/watch?v=runningvidx", task_id="runningvidx")
    database.update_task(task_id, status="running")

    client = TestClient(main.app)
    response = client.delete(f"/api/tasks/{task_id}")

    assert response.status_code == 409
    assert database.get_task(task_id) is not None


def test_clear_asr_stage_only_removes_declared_output(monkeypatch, tmp_path):
    configure_tmp_runtime(monkeypatch, tmp_path)
    task_id = database.create_task("https://www.youtube.com/watch?v=clearasrxyz", task_id="clearasrxyz")
    session = config.WORKFOLDER / "uploader" / "title__clearasrxyz"
    for directory in ("metadata", "media", "segments/vocals", "segments/tts", "tmp"):
        (session / directory).mkdir(parents=True, exist_ok=True)
    for name in (
        "metadata/asr.json",
        "metadata/asr.raw.funasr.json",
        "metadata/asr.raw.remote_funasr.json",
        "metadata/asr.remote_request.json",
        "metadata/asr_fixed.json",
        "metadata/translation.zh.json",
        "metadata/subtitles.zh.srt",
        "metadata/timings.json",
        "tmp/audio_dubbing.wav",
        "media/video_final.mp4",
        "segments/vocals/0001.wav",
        "segments/tts/0001.wav",
    ):
        (session / name).write_bytes(b"cached")
    database.update_task(task_id, session_path=str(session))

    client = TestClient(main.app)
    response = client.post(f"/api/tasks/{task_id}/clear-stage/asr")

    assert response.status_code == 200
    assert not (session / "metadata/asr.json").exists()
    assert (session / "metadata/asr.raw.funasr.json").exists()
    assert (session / "metadata/asr.raw.remote_funasr.json").exists()
    assert (session / "metadata/asr.remote_request.json").exists()
    assert (session / "metadata/asr_fixed.json").exists()
    assert (session / "metadata/translation.zh.json").exists()
    assert (session / "segments/vocals/0001.wav").exists()
    assert (session / "segments/tts/0001.wav").exists()
    assert (session / "media/video_final.mp4").exists()


def test_clear_stage_outputs_match_stage_descriptions(monkeypatch, tmp_path):
    configure_tmp_runtime(monkeypatch, tmp_path)
    task_id = database.create_task("https://www.youtube.com/watch?v=clearstages", task_id="clearstages")
    session = config.WORKFOLDER / "uploader" / "title__clearstages"
    for directory in ("metadata", "media", "segments/vocals", "segments/tts", "tmp"):
        (session / directory).mkdir(parents=True, exist_ok=True)
    for name in (
        "media/video_source.mp4",
        "metadata/ytdlp_info.json",
        "metadata/local_info.json",
        "media/audio_vocals.wav",
        "media/audio_bgm.wav",
        "segments/vocals/0001.wav",
        "segments/vocals/keep.txt",
        "segments/tts/0001.wav",
        "segments/tts/keep.txt",
        "tmp/audio_dubbing.wav",
        "metadata/timings.json",
        "media/video_final.mp4",
    ):
        (session / name).write_bytes(b"cached")
    database.update_task(task_id, session_path=str(session))

    client = TestClient(main.app)

    assert client.post(f"/api/tasks/{task_id}/clear-stage/download").status_code == 200
    assert not (session / "media/video_source.mp4").exists()
    assert not (session / "metadata/ytdlp_info.json").exists()
    assert not (session / "metadata/local_info.json").exists()

    assert client.post(f"/api/tasks/{task_id}/clear-stage/separate").status_code == 200
    assert not (session / "media/audio_vocals.wav").exists()
    assert not (session / "media/audio_bgm.wav").exists()

    assert client.post(f"/api/tasks/{task_id}/clear-stage/split_audio").status_code == 200
    assert not (session / "segments/vocals/0001.wav").exists()
    assert (session / "segments/vocals/keep.txt").exists()

    assert client.post(f"/api/tasks/{task_id}/clear-stage/tts").status_code == 200
    assert not (session / "segments/tts/0001.wav").exists()
    assert (session / "segments/tts/keep.txt").exists()

    assert client.post(f"/api/tasks/{task_id}/clear-stage/merge_audio").status_code == 200
    assert not (session / "tmp/audio_dubbing.wav").exists()
    assert not (session / "metadata/timings.json").exists()

    assert client.post(f"/api/tasks/{task_id}/clear-stage/merge_video").status_code == 200
    assert not (session / "media/video_final.mp4").exists()


def test_rerun_task_purges_session_and_requeues(monkeypatch, tmp_path):
    configure_tmp_runtime(monkeypatch, tmp_path)
    enqueued: list[str] = []
    monkeypatch.setattr(main.worker, "enqueue", lambda task_id: enqueued.append(task_id))

    task_id = database.create_task(
        "https://www.youtube.com/watch?v=rerunvideox",
        task_id="rerunvideox",
        asr_language="zh",
        target_language="en",
        add_subtitles=False,
        asr_model="funasr:iic/SenseVoiceSmall",
        translate_mode="batch",
        validate_translation=True,
        tts_mode="hifi_clone",
    )
    session = config.WORKFOLDER / "uploader" / "title__rerunvideox"
    (session / "media").mkdir(parents=True)
    (session / "media" / "video_source.mp4").write_bytes(b"old")
    database.update_task(task_id, session_path=str(session), status="failed")
    log_file = database.log_path(task_id)
    log_file.write_text("old run", encoding="utf-8")

    client = TestClient(main.app)
    response = client.post(f"/api/tasks/{task_id}/rerun")

    assert response.status_code == 200
    body = response.json()
    assert body["id"] == task_id
    assert body["status"] == "queued"
    assert body["session_path"] is None
    assert body["asr_language"] == "zh"
    assert body["target_language"] == "en"
    assert body["add_subtitles"] == 0
    assert body["asr_model"] == "funasr:iic/SenseVoiceSmall"
    assert body["translate_mode"] == "batch"
    assert body["validate_translation"] == 1
    assert body["tts_mode"] == "hifi_clone"
    assert enqueued == [task_id]
    assert not session.exists()
    assert not log_file.exists()


def test_rerun_task_returns_404_for_unknown(monkeypatch, tmp_path):
    configure_tmp_runtime(monkeypatch, tmp_path)
    client = TestClient(main.app)

    response = client.post("/api/tasks/missing/rerun")

    assert response.status_code == 404


def test_rerun_task_rejects_running_task(monkeypatch, tmp_path):
    configure_tmp_runtime(monkeypatch, tmp_path)
    enqueued: list[str] = []
    monkeypatch.setattr(main.worker, "enqueue", lambda task_id: enqueued.append(task_id))
    task_id = database.create_task("https://www.youtube.com/watch?v=runrerunvid", task_id="runrerunvid")
    database.update_task(task_id, status="running")

    client = TestClient(main.app)
    response = client.post(f"/api/tasks/{task_id}/rerun")

    assert response.status_code == 409
    assert enqueued == []
    assert database.get_task(task_id)["status"] == "running"


def test_stop_running_task_sets_worker_flag(monkeypatch, tmp_path):
    configure_tmp_runtime(monkeypatch, tmp_path)
    requested: list[str] = []
    monkeypatch.setattr(main.worker, "request_stop", lambda task_id: requested.append(task_id) or True)
    task_id = database.create_task("https://www.youtube.com/watch?v=stoprunning", task_id="stoprunning")
    database.update_task(task_id, status="running")

    client = TestClient(main.app)
    response = client.post(f"/api/tasks/{task_id}/stop")

    assert response.status_code == 200
    assert requested == [task_id]
    # The task is still running; the worker is expected to observe the flag and
    # transition it to failed on its own.
    assert database.get_task(task_id)["status"] == "running"


def test_stop_queued_task_marks_it_failed_immediately(monkeypatch, tmp_path):
    configure_tmp_runtime(monkeypatch, tmp_path)
    requested: list[str] = []
    monkeypatch.setattr(main.worker, "request_stop", lambda task_id: requested.append(task_id) or True)
    task_id = database.create_task("https://www.youtube.com/watch?v=stopqueued", task_id="stopqueued")
    # Task starts in 'queued' status (database default).

    client = TestClient(main.app)
    response = client.post(f"/api/tasks/{task_id}/stop")

    assert response.status_code == 200
    assert requested == [task_id]
    task = database.get_task(task_id)
    assert task["status"] == "failed"
    assert "Stopped by user" in (task["error_message"] or "")
    # Frontend relies on this flag to distinguish user-stops from errors.
    assert task["stop_requested"] == 1
    # For a queued task, the stage was never started and stays in 'pending'.
    download_stage = next(s for s in task["stages"] if s["name"] == "download")
    assert download_stage["status"] == "pending"


@pytest.mark.parametrize("status", ["succeeded", "failed"])
def test_stop_task_rejects_non_active_tasks(monkeypatch, tmp_path, status):
    configure_tmp_runtime(monkeypatch, tmp_path)
    requested: list[str] = []
    monkeypatch.setattr(main.worker, "request_stop", lambda task_id: requested.append(task_id) or True)
    task_id = database.create_task("https://www.youtube.com/watch?v=nostop", task_id="nostop")
    database.update_task(task_id, status=status)

    client = TestClient(main.app)
    response = client.post(f"/api/tasks/{task_id}/stop")

    assert response.status_code == 409
    assert requested == []


def test_stop_task_returns_404_for_unknown(monkeypatch, tmp_path):
    configure_tmp_runtime(monkeypatch, tmp_path)
    client = TestClient(main.app)
    response = client.post("/api/tasks/missing-id/stop")
    assert response.status_code == 404


def test_delete_task_skips_session_outside_workfolder(monkeypatch, tmp_path):
    configure_tmp_runtime(monkeypatch, tmp_path)
    task_id = database.create_task("https://www.youtube.com/watch?v=outsidevidx", task_id="outsidevidx")
    outside = tmp_path / "elsewhere" / "session"
    (outside / "media").mkdir(parents=True)
    (outside / "media" / "video_source.mp4").write_bytes(b"mp4")
    database.update_task(task_id, session_path=str(outside))

    client = TestClient(main.app)
    response = client.delete(f"/api/tasks/{task_id}")

    assert response.status_code == 204
    assert database.get_task(task_id) is None
    assert outside.exists(), "Sessions outside WORKFOLDER must not be deleted."


def test_cors_origins_include_runtime_configuration(monkeypatch):
    monkeypatch.setenv("CORS_ALLOW_ORIGINS", "http://172.27.2.90:3000, http://100.94.222.54:3000")

    origins = main.cors_origins()

    assert "http://localhost:3000" in origins
    assert "http://172.27.2.90:3000" in origins
    assert "http://100.94.222.54:3000" in origins


def test_cors_origin_regex_allows_common_development_hosts(monkeypatch):
    monkeypatch.delenv("CORS_ALLOW_ORIGIN_REGEX", raising=False)

    regex = re.compile(main.cors_origin_regex())

    assert regex.fullmatch("http://0.0.0.0:3000")
    assert regex.fullmatch("http://192.168.1.2:3000")
    assert regex.fullmatch("http://10.0.0.5:3000")
    assert regex.fullmatch("http://172.27.2.90:3000")
    assert regex.fullmatch("http://100.94.222.54:3000")
    assert regex.fullmatch("http://localhost:3080")
    assert regex.fullmatch("http://192.168.1.2:3080")
    assert not regex.fullmatch("http://example.com:3000")
    assert not regex.fullmatch("http://192.168.1.2:4000")


def test_openai_models_use_form_key_without_leaking_it(monkeypatch, tmp_path):
    configure_tmp_runtime(monkeypatch, tmp_path)
    captured = {}

    def fake_list_models(*, base_url: str, api_key: str) -> list[str]:
        captured["base_url"] = base_url
        captured["api_key"] = api_key
        return ["gpt-test", "qwen-test"]

    monkeypatch.setattr(settings_router, "list_openai_models", fake_list_models)
    client = TestClient(main.app)

    response = client.post(
        "/api/settings/openai/models",
        json={"base_url": "https://example.com/v1", "api_key": "sk-secret-models"},
    )

    assert response.status_code == 200
    assert response.json() == {"models": ["gpt-test", "qwen-test"]}
    assert captured == {"base_url": "https://example.com/v1", "api_key": "sk-secret-models"}
    assert "sk-secret-models" not in response.text


def test_openai_models_can_use_saved_key(monkeypatch, tmp_path):
    configure_tmp_runtime(monkeypatch, tmp_path)
    database.save_openai_settings("https://saved.example/v1", "sk-saved", "saved-model")
    captured = {}

    def fake_list_models(*, base_url: str, api_key: str) -> list[str]:
        captured["base_url"] = base_url
        captured["api_key"] = api_key
        return ["saved-model"]

    monkeypatch.setattr(settings_router, "list_openai_models", fake_list_models)
    client = TestClient(main.app)

    response = client.post("/api/settings/openai/models", json={"base_url": "", "api_key": ""})

    assert response.status_code == 200
    assert response.json() == {"models": ["saved-model"]}
    assert captured == {"base_url": "https://saved.example/v1", "api_key": "sk-saved"}


def test_openai_settings_include_translate_concurrency(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENAI_TRANSLATE_CONCURRENCY", raising=False)
    configure_tmp_runtime(monkeypatch, tmp_path)
    client = TestClient(main.app)

    response = client.get("/api/settings/openai")

    assert response.status_code == 200
    assert response.json()["translate_concurrency"] == "50"


def test_openai_settings_persists_translate_concurrency(monkeypatch, tmp_path):
    configure_tmp_runtime(monkeypatch, tmp_path)
    client = TestClient(main.app)

    response = client.post(
        "/api/settings/openai",
        json={
            "base_url": "https://example.com/v1",
            "api_key": "",
            "model": "model",
            "translate_concurrency": " 32 ",
        },
    )

    assert response.status_code == 200
    assert response.json()["translate_concurrency"] == "32"
    assert database.get_openai_settings()["translate_concurrency"] == "32"


@pytest.mark.parametrize("value", ["abc", "1.5", "0", "-1", "201"])
def test_openai_settings_rejects_invalid_translate_concurrency(monkeypatch, tmp_path, value):
    configure_tmp_runtime(monkeypatch, tmp_path)
    database.save_openai_settings("https://example.com/v1", "sk-test", "model", "64")
    client = TestClient(main.app)

    response = client.post(
        "/api/settings/openai",
        json={
            "base_url": "https://example.com/v1",
            "api_key": "",
            "clear_api_key": False,
            "model": "model",
            "translate_concurrency": value,
        },
    )

    assert response.status_code == 422
    assert database.get_openai_settings()["translate_concurrency"] == "64"


def test_openai_settings_empty_translate_concurrency_preserves_existing(monkeypatch, tmp_path):
    configure_tmp_runtime(monkeypatch, tmp_path)
    database.save_openai_settings("https://example.com/v1", "sk-test", "model", "64")
    client = TestClient(main.app)

    response = client.post(
        "/api/settings/openai",
        json={
            "base_url": "https://example.com/v1",
            "api_key": "",
            "clear_api_key": False,
            "model": "next-model",
            "translate_concurrency": "",
        },
    )

    assert response.status_code == 200
    assert response.json()["translate_concurrency"] == "64"
    settings = database.get_openai_settings()
    assert settings["translate_concurrency"] == "64"
    assert settings["model"] == "next-model"


@pytest.mark.parametrize("value", ["1", "200"])
def test_openai_settings_accepts_translate_concurrency_boundaries(monkeypatch, tmp_path, value):
    configure_tmp_runtime(monkeypatch, tmp_path)
    client = TestClient(main.app)

    response = client.post(
        "/api/settings/openai",
        json={
            "base_url": "https://example.com/v1",
            "api_key": "",
            "clear_api_key": False,
            "model": "model",
            "translate_concurrency": value,
        },
    )

    assert response.status_code == 200
    assert response.json()["translate_concurrency"] == value
    assert database.get_openai_settings()["translate_concurrency"] == value


def test_resume_task_requeues_failed_task(monkeypatch, tmp_path):
    configure_tmp_runtime(monkeypatch, tmp_path)
    enqueued: list[str] = []
    monkeypatch.setattr(main.worker, "enqueue", lambda task_id: enqueued.append(task_id))
    task_id = database.create_task("https://www.youtube.com/watch?v=resumevideox", task_id="resumevideox")
    database.update_task(task_id, status="failed", error_message="boom", completed_at=database.now_iso())
    database.update_stage(task_id, "asr", status="failed", progress=33, error_message="boom")
    database.update_stage(task_id, "download", status="succeeded")

    client = TestClient(main.app)
    response = client.post(f"/api/tasks/{task_id}/resume")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "queued"
    assert body["error_message"] is None
    stages = {s["name"]: s for s in body["stages"]}
    assert stages["download"]["status"] == "succeeded"
    assert stages["asr"]["status"] == "pending"
    assert stages["asr"]["progress"] is None
    assert stages["asr"]["error_message"] is None
    assert enqueued == [task_id]


def test_resume_task_rejects_non_failed(monkeypatch, tmp_path):
    configure_tmp_runtime(monkeypatch, tmp_path)
    task_id = database.create_task("https://www.youtube.com/watch?v=okvideoxxxx", task_id="okvideoxxxx")
    database.update_task(task_id, status="succeeded")

    client = TestClient(main.app)
    response = client.post(f"/api/tasks/{task_id}/resume")

    assert response.status_code == 409


def test_update_task_config_can_clear_asr_model(monkeypatch, tmp_path):
    configure_tmp_runtime(monkeypatch, tmp_path)
    task_id = database.create_task(
        "https://www.youtube.com/watch?v=asrmodelcfg",
        task_id="asrmodelcfg",
        asr_model="funasr:FunAudioLLM/Fun-ASR-Nano-2512",
    )

    client = TestClient(main.app)
    response = client.patch(f"/api/tasks/{task_id}/config", json={"asr_model": None})

    assert response.status_code == 200
    assert response.json()["asr_model"] is None
    assert database.get_task(task_id)["asr_model"] is None


def test_ytdlp_proxy_port_settings(monkeypatch, tmp_path):
    configure_tmp_runtime(monkeypatch, tmp_path)
    client = TestClient(main.app)

    saved = client.post("/api/settings/ytdlp", json={"proxy_port": "7890"})
    loaded = client.get("/api/settings/ytdlp")

    assert saved.status_code == 200
    assert loaded.status_code == 200
    assert loaded.json() == {"proxy_port": "7890"}


def test_ytdlp_proxy_port_rejects_invalid_value(monkeypatch, tmp_path):
    configure_tmp_runtime(monkeypatch, tmp_path)
    client = TestClient(main.app)

    response = client.post("/api/settings/ytdlp", json={"proxy_port": "70000"})

    assert response.status_code == 422


def test_upload_local_video_creates_task_and_saved_file(monkeypatch, tmp_path):
    configure_tmp_runtime(monkeypatch, tmp_path)
    enqueued: list[str] = []
    monkeypatch.setattr(main.worker, "enqueue", lambda task_id: enqueued.append(task_id))
    client = TestClient(main.app)

    response = client.post(
        "/api/tasks/upload",
        data={"config": '{"asr_language":"zh","target_language":"en"}'},
        files={"file": ("clip.mp4", b"mp4data", "video/mp4")},
    )

    assert response.status_code == 201
    body = response.json()
    assert body["title"] == "clip"
    assert body["url"].startswith(f"local://upload/{body['id']}?direction=zh-en")
    assert enqueued == [body["id"]]
    saved = list((config.WORKFOLDER / "_uploads" / body["id"]).iterdir())
    assert len(saved) == 1
    assert saved[0].read_bytes() == b"mp4data"


def test_create_task_rejects_local_upload_url(monkeypatch, tmp_path):
    configure_tmp_runtime(monkeypatch, tmp_path)
    client = TestClient(main.app)
    response = client.post("/api/tasks", json={"url": "local://upload/fake?direction=en-zh"})
    assert response.status_code == 422


def test_create_task_rejects_unavailable_cuda_runtime(monkeypatch, tmp_path):
    configure_tmp_runtime(monkeypatch, tmp_path)
    monkeypatch.setattr(
        tasks_router,
        "validate_runtime_device",
        lambda: (_ for _ in ()).throw(RuntimeError("DEVICE=cuda is not available")),
    )
    client = TestClient(main.app)
    response = client.post("/api/tasks", json={"url": "https://www.youtube.com/watch?v=okvideoxxxx"})
    assert response.status_code == 409
    assert response.json()["detail"] == "DEVICE=cuda is not available"


def test_delete_local_video_removes_upload(monkeypatch, tmp_path):
    configure_tmp_runtime(monkeypatch, tmp_path)
    client = TestClient(main.app)
    upload = client.post(
        "/api/tasks/upload",
        data={"config": '{"asr_language":"en","target_language":"zh"}'},
        files={"file": ("clip.mp4", b"mp4data", "video/mp4")},
    )
    task_id = upload.json()["id"]
    upload_root = config.WORKFOLDER / "_uploads" / task_id
    assert upload_root.exists()

    response = client.delete(f"/api/tasks/{task_id}")

    assert response.status_code == 204
    assert not upload_root.exists()
    assert database.get_task(task_id) is None
