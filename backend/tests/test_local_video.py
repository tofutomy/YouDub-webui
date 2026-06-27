from __future__ import annotations

import json
import subprocess
from pathlib import Path

from backend.app.adapters import local_video
from backend.app.sources import get_source_for_task
from backend.app.youtube import make_localdir_url


def test_import_local_video_transcodes_with_configured_ffmpeg(monkeypatch, tmp_path):
    task_id = "local-task"
    upload_dir = local_video.upload_dir(tmp_path, task_id)
    upload_dir.mkdir(parents=True)
    source_file = upload_dir / "demo.mov"
    source_file.write_bytes(b"video")
    commands: list[list[str]] = []

    def fake_run(cmd, check=False, **kwargs):
        commands.append(cmd)
        output = Path(cmd[-1])
        output.write_bytes(b"mp4")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setenv("FFMPEG_PATH", "/opt/bin/ffmpeg")
    monkeypatch.setattr(local_video.subprocess, "run", fake_run)

    session, info = local_video.import_local_video(
        f"local://upload/{task_id}?direction=zh-en&filename=demo.mov",
        tmp_path,
        get_source_for_task({"url": f"local://upload/{task_id}?direction=zh-en&filename=demo.mov"}),
    )

    assert session == tmp_path / "local" / f"demo__{task_id}"
    assert info["title"] == "demo"
    assert info["target_language"] == "en"
    assert commands
    assert commands[0][0] == "/opt/bin/ffmpeg"
    assert commands[0][-1] == str(session / "media" / "video_source.mp4")
    metadata = json.loads((session / "metadata" / "local_info.json").read_text(encoding="utf-8"))
    assert metadata["source"] == "local"


def test_import_localdir_video_creates_link_to_source(monkeypatch, tmp_path):
    """localdir imports create a symlink/hardlink to the source (no copy)."""
    task_id = "dir-task-001"
    source_file = tmp_path / "source_video.mkv"
    source_file.write_bytes(b"mkv-content")

    url = make_localdir_url(task_id, str(source_file), "en-zh", "source_video.mkv")
    source = get_source_for_task({"url": url})

    session, info = local_video.import_localdir_video(url, tmp_path, source)

    video_link = session / "media" / "video_source.mp4"
    assert video_link.exists()
    # The link should point back to the original source content
    assert video_link.read_bytes() == b"mkv-content"
    assert info["title"] == "source_video"
    assert info["source"] == "localdir"
    assert info["original_path"] == str(source_file)
    assert info["asr_language"] == "en"
    assert info["target_language"] == "zh"
    metadata = json.loads((session / "metadata" / "local_info.json").read_text(encoding="utf-8"))
    assert metadata["source"] == "localdir"
    assert metadata["original_path"] == str(source_file)


def test_import_localdir_video_skips_when_link_exists(monkeypatch, tmp_path):
    """If video_source.mp4 link already exists, skip creation."""
    task_id = "dir-task-002"
    source_file = tmp_path / "existing.mp4"
    source_file.write_bytes(b"mp4-content")

    url = make_localdir_url(task_id, str(source_file), "zh-en", "existing.mp4")
    source = get_source_for_task({"url": url})

    # First call — creates link
    session, _ = local_video.import_localdir_video(url, tmp_path, source)
    video_link = session / "media" / "video_source.mp4"
    assert video_link.exists()

    # Second call — should not fail or recreate
    session2, info2 = local_video.import_localdir_video(url, tmp_path, source)
    assert session2 == session


def test_import_localdir_video_rejects_missing_file(monkeypatch, tmp_path):
    """Raises FileNotFoundError when source path does not exist."""
    url = make_localdir_url("missing-task", "/nonexistent/video.mp4", "en-zh")
    source = get_source_for_task({"url": url})

    import pytest

    with pytest.raises(FileNotFoundError, match="Source video not found"):
        local_video.import_localdir_video(url, tmp_path, source)
