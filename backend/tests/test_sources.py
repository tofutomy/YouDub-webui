from __future__ import annotations

from backend.app.sources import get_source_for_task
from backend.app.youtube import make_localdir_url


def test_bilibili_task_without_language_uses_app_default_direction() -> None:
    source = get_source_for_task({"url": "https://www.bilibili.com/video/BV1UNAbzpEzR"})

    assert source.name == "bilibili"
    assert source.asr_language == "en"
    assert source.target_language == "zh"


def test_task_language_pair_is_inferred_from_one_side() -> None:
    source = get_source_for_task({
        "url": "https://www.bilibili.com/video/BV1UNAbzpEzR",
        "target_language": "en",
    })

    assert source.asr_language == "zh"
    assert source.target_language == "en"


def test_localdir_task_detected_as_localdir_source() -> None:
    url = make_localdir_url("task-001", r"G:\videos\test.mp4", "en-zh", "test.mp4")
    source = get_source_for_task({"url": url})

    assert source.name == "localdir"
    assert source.asr_language == "en"
    assert source.target_language == "zh"


def test_localdir_task_zh_to_en() -> None:
    url = make_localdir_url("task-002", "/home/user/视频.mp4", "zh-en", "视频.mp4")
    source = get_source_for_task({"url": url})

    assert source.name == "localdir"
    assert source.asr_language == "zh"
    assert source.target_language == "en"


def test_localdir_task_with_explicit_languages() -> None:
    url = make_localdir_url("task-003", r"C:\v.mp4", "en-zh")
    source = get_source_for_task({"url": url, "asr_language": "zh", "target_language": "en"})

    assert source.name == "localdir"
    assert source.asr_language == "zh"
    assert source.target_language == "en"



def test_legacy_local_upload_url_uses_direction_fallback() -> None:
    source = get_source_for_task({"url": "local://upload/task-004?direction=ja-zh&filename=clip.mp4"})

    assert source.name == "local"
    assert source.asr_language == "ja"
    assert source.target_language == "zh"


def test_database_languages_win_over_legacy_local_url_direction() -> None:
    source = get_source_for_task({
        "url": "local://upload/task-005?direction=ja-zh&filename=clip.mp4",
        "asr_language": "zh",
        "target_language": "en",
    })

    assert source.name == "local"
    assert source.asr_language == "zh"
    assert source.target_language == "en"
