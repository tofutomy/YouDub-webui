from __future__ import annotations

from backend.app.sources import get_source_for_task


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
