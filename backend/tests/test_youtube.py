import pytest

from backend.app.youtube import (
    extract_video_id,
    is_bilibili_url,
    is_local_en_to_zh_url,
    is_local_upload_url,
    is_local_zh_to_en_url,
    is_localdir_en_to_zh_url,
    is_localdir_url,
    is_localdir_zh_to_en_url,
    is_youtube_url,
    local_upload_direction,
    local_upload_task_id,
    localdir_direction,
    localdir_filename,
    localdir_source_path,
    localdir_task_id,
    make_localdir_url,
)


def test_extract_video_id_from_watch_url():
    assert extract_video_id("https://www.youtube.com/watch?v=abcdefghijk&t=12s") == "abcdefghijk"


def test_extract_video_id_from_shorts_url():
    assert extract_video_id("https://youtube.com/shorts/abcdefghijk?feature=share") == "abcdefghijk"


def test_rejects_playlist_only_url():
    assert not is_youtube_url("https://www.youtube.com/playlist?list=123")


def test_extract_video_id_from_bilibili_url():
    assert extract_video_id("https://www.bilibili.com/video/BV1xx411c7mD/?spm_id_from=test") == "BV1xx411c7mD"


def test_is_bilibili_url():
    assert is_bilibili_url("https://www.bilibili.com/video/BV1xx411c7mD")
    assert not is_bilibili_url("https://www.youtube.com/watch?v=abcdefghijk")


def test_extract_video_id_rejects_unknown():
    with pytest.raises(ValueError):
        extract_video_id("https://example.com/video/123")

def test_local_upload_helpers_parse_direction_and_task_id():
    url = "local://upload/abc123?direction=zh-en&filename=demo.mp4"

    assert local_upload_task_id(url) == "abc123"
    assert local_upload_direction(url) == "zh-en"
    assert is_local_upload_url(url)
    assert is_local_zh_to_en_url(url)
    assert not is_local_en_to_zh_url(url)


def test_local_upload_helpers_reject_missing_or_unknown_direction():
    assert not is_local_upload_url("local://upload/abc123")
    assert not is_local_upload_url("local://upload/../workfolder?direction=en-zh")
    assert local_upload_task_id("local://upload/../workfolder?direction=en-zh") == ""
    assert not is_local_upload_url("local://upload/abc123?direction=fr-zh")
    assert local_upload_task_id("https://example.com/video.mp4") == ""


def test_make_localdir_url_roundtrip():
    task_id = "test-task-001"
    path = r"G:\videos\test video.mp4"
    url = make_localdir_url(task_id, path, "en-zh")

    assert localdir_task_id(url) == task_id
    assert localdir_source_path(url) == path
    assert localdir_direction(url) == "en-zh"
    assert localdir_filename(url) == "test video.mp4"
    assert is_localdir_url(url)
    assert is_localdir_en_to_zh_url(url)
    assert not is_localdir_zh_to_en_url(url)


def test_localdir_zh_to_en():
    url = make_localdir_url("abc", "/home/user/视频.mp4", "zh-en", "视频.mp4")

    assert is_localdir_url(url)
    assert is_localdir_zh_to_en_url(url)
    assert not is_localdir_en_to_zh_url(url)
    assert localdir_source_path(url) == "/home/user/视频.mp4"


def test_localdir_rejects_missing_path():
    assert not is_localdir_url("localdir://task001?direction=en-zh")
    assert not is_localdir_url("localdir://task001?direction=en-zh&path=")


def test_localdir_rejects_invalid_direction():
    url = make_localdir_url("t1", "/tmp/v.mp4", "fr-de")
    assert not is_localdir_url(url)


def test_localdir_rejects_non_localdir_scheme():
    assert not is_localdir_url("https://example.com/video.mp4")
    assert localdir_task_id("https://example.com/video.mp4") == ""
    assert localdir_source_path("https://example.com/video.mp4") == ""
