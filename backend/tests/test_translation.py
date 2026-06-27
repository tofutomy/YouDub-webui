from __future__ import annotations

import json

import pytest

from backend.app.adapters import openai_translate
from backend.app.adapters.openai_translate import (
    HotwordItem,
    PreprocessResponse,
)
from backend.app.sources import get_source_for_task


YT_SOURCE = get_source_for_task({"url": "https://www.youtube.com/watch?v=abcdefghijk"})
BB_SOURCE = get_source_for_task({
    "url": "https://www.bilibili.com/video/BV1xx411c7mD",
    "asr_language": "zh",
    "target_language": "en",
})


def _write_asr(path, n: int, full_text: str | None = None) -> None:
    utterances = [
        {"text": f"S{i}.", "start_time": i * 1000, "end_time": (i + 1) * 1000}
        for i in range(n)
    ]
    payload = {"result": {"utterances": utterances, "text": full_text or " ".join(u["text"] for u in utterances)}}
    path.write_text(json.dumps(payload), encoding="utf-8")


def _settings() -> dict[str, str]:
    return {"base_url": "https://example.com/v1", "api_key": "sk-test", "model": "model-x"}


def _stub_preprocess(monkeypatch, response: PreprocessResponse | None = None):
    seen: list[dict] = []

    def fake(full_text, meta, source, **kw):
        seen.append({"full_text": full_text, "meta": meta, "source": source, **kw})
        return response or PreprocessResponse()

    monkeypatch.setattr(openai_translate, "preprocess", fake)
    return seen


def _stub_translate_batch(monkeypatch, transform):
    seen: list[dict] = []

    def fake(texts, source, meta, pre, **kw):
        seen.append({"texts": list(texts), "source": source, "meta": meta, "pre": pre, **kw})
        results = [transform(t) for t in texts]
        on_progress = kw.get("on_progress")
        if on_progress:
            for i, r in enumerate(results):
                on_progress(i, r)
        return results

    monkeypatch.setattr(openai_translate, "translate_batch", fake)
    return seen


def test_translate_asr_writes_schema_with_speaker_and_lang(tmp_path, monkeypatch):
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    asr_file = metadata / "asr.json"
    _write_asr(asr_file, 2)

    _stub_preprocess(monkeypatch)
    _stub_translate_batch(monkeypatch, lambda t: f"zh:{t}")

    out = openai_translate.translate_asr(asr_file, tmp_path, _settings(), YT_SOURCE)
    items = json.loads(out.read_text(encoding="utf-8"))["translation"]
    assert [i["dst"] for i in items] == ["zh:S0.", "zh:S1."]
    assert {i["src_lang"] for i in items} == {"en"}
    assert {i["dst_lang"] for i in items} == {"zh"}
    assert {i["speaker"] for i in items} == {"1"}
    assert items[0]["start_time"] == 0


def test_translate_asr_output_filename_uses_target_lang(tmp_path, monkeypatch):
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    asr_file = metadata / "asr.json"
    _write_asr(asr_file, 1)

    _stub_preprocess(monkeypatch)
    _stub_translate_batch(monkeypatch, lambda _t: "x")

    out = openai_translate.translate_asr(asr_file, tmp_path, _settings(), BB_SOURCE)
    assert out.name == "translation.en.json"


def test_translate_asr_passes_meta_and_full_text_to_preprocess(tmp_path, monkeypatch):
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    asr_file = metadata / "asr.json"
    _write_asr(asr_file, 1, full_text="hello world")
    (metadata / "ytdlp_info.json").write_text(
        json.dumps({"title": "T", "uploader": "U", "description": "D"}),
        encoding="utf-8",
    )

    seen = _stub_preprocess(monkeypatch)
    _stub_translate_batch(monkeypatch, lambda t: t)

    openai_translate.translate_asr(asr_file, tmp_path, _settings(), YT_SOURCE)
    assert seen[0]["full_text"] == "hello world"
    assert seen[0]["meta"] == {"title": "T", "uploader": "U", "description": "D"}


def test_translate_asr_invokes_translate_batch_with_all_texts_at_once(tmp_path, monkeypatch):
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    asr_file = metadata / "asr.json"
    _write_asr(asr_file, 5)

    _stub_preprocess(monkeypatch, PreprocessResponse(hotwords=[HotwordItem(src="x", dst="y")]))
    seen = _stub_translate_batch(monkeypatch, lambda t: f"zh:{t}")

    openai_translate.translate_asr(asr_file, tmp_path, _settings(), YT_SOURCE)
    assert len(seen) == 1
    assert seen[0]["texts"] == ["S0.", "S1.", "S2.", "S3.", "S4."]
    assert seen[0]["pre"].hotwords[0].src == "x"


def test_translate_batch_replaces_em_dash_for_zh_target(monkeypatch):
    monkeypatch.setattr(openai_translate, "_call_json", lambda *a, **kw: {"dst": "你好——世界"})
    monkeypatch.setattr(openai_translate, "_client", lambda *a, **kw: object())

    out = openai_translate.translate_batch(
        ["Hello world."], YT_SOURCE, {}, PreprocessResponse(),
        base_url="u", api_key="k", model="m",
    )
    assert out == ["你好，世界"]


def test_translate_batch_does_not_replace_em_dash_for_en_target(monkeypatch):
    monkeypatch.setattr(
        openai_translate, "_call_json", lambda *a, **kw: {"dst": "He said—wait—and left."}
    )
    monkeypatch.setattr(openai_translate, "_client", lambda *a, **kw: object())

    out = openai_translate.translate_batch(
        ["他说——等等——就走了。"], BB_SOURCE, {}, PreprocessResponse(),
        base_url="u", api_key="k", model="m",
    )
    assert out == ["He said—wait—and left."]


def test_translate_batch_uses_shared_system_prompt(monkeypatch):
    captured: list[str] = []
    lock = __import__("threading").Lock()

    def fake_call_json(client, model, system, user):
        with lock:
            captured.append(system)
        return {"dst": f"dst:{user}"}

    monkeypatch.setattr(openai_translate, "_call_json", fake_call_json)
    monkeypatch.setattr(openai_translate, "_client", lambda *a, **kw: object())

    texts = [f"s{i}" for i in range(5)]
    out = openai_translate.translate_batch(
        texts, BB_SOURCE, {}, PreprocessResponse(),
        base_url="u", api_key="k", model="m", concurrency=4,
    )
    assert out == [f"dst:s{i}" for i in range(5)]
    assert len(set(captured)) == 1, "system prompt must be identical across calls for prompt cache"


@pytest.mark.parametrize("value", ["abc", "1.5", "0", "-1", "201", ""])
def test_concurrency_from_bad_saved_values_falls_back_to_default(value):
    assert openai_translate._concurrency_from({"translate_concurrency": value}) == 50


def test_translate_sentence_retries_on_empty_dst(monkeypatch):
    calls = {"n": 0}

    def fake_call_json(client, model, system, user):
        calls["n"] += 1
        return {"dst": ""} if calls["n"] == 1 else {"dst": "ok"}

    monkeypatch.setattr(openai_translate, "_call_json", fake_call_json)

    out = openai_translate.translate_sentence("hello", "en", object(), "m", "sys")
    assert out == "ok"
    assert calls["n"] == 2


def test_translate_sentence_raises_after_retries(monkeypatch):
    def fake_call_json(client, model, system, user):
        raise ValueError("boom")

    monkeypatch.setattr(openai_translate, "_call_json", fake_call_json)

    with pytest.raises(RuntimeError, match="translate_sentence failed"):
        openai_translate.translate_sentence("x", "en", object(), "m", "sys")


def test_preprocess_returns_empty_when_repeatedly_invalid(monkeypatch):
    def fake_call_json(client, model, system, user):
        return {"summary": 123, "hotwords": "bad"}

    monkeypatch.setattr(openai_translate, "_call_json", fake_call_json)
    monkeypatch.setattr(openai_translate, "_client", lambda *a, **kw: object())

    pre = openai_translate.preprocess(
        "text", {"title": "t"}, YT_SOURCE,
        base_url="u", api_key="k", model="m",
    )
    assert pre.summary == ""
    assert pre.hotwords == []
    assert pre.corrections == []


def test_translate_system_prompt_contains_meta_summary_hotwords(monkeypatch):
    pre = PreprocessResponse(
        summary="Recap of the talk.",
        hotwords=[HotwordItem(src="LEGO", dst="乐高")],
    )
    meta = {"title": "Demo", "uploader": "Alice", "description": "Long description"}
    system = openai_translate._translate_system(YT_SOURCE, meta, pre)
    assert "Demo" in system
    assert "Alice" in system
    assert "Long description" in system
    assert "Recap of the talk." in system
    assert "LEGO -> 乐高" in system


# --- Validation and Correction tests ---

def test_validate_batch_returns_score_and_issues(monkeypatch):
    from backend.app.adapters.openai_translate import ValidationBatchResult

    def fake_call_json(client, model, system, user):
        return {
            "score": 75,
            "issues": [
                {
                    "index": 0,
                    "type": "terminology",
                    "problem": "GPU translated inconsistently",
                    "suggestion": "使用 GPU 而不是图形处理器",
                }
            ],
        }

    monkeypatch.setattr(openai_translate, "_call_json", fake_call_json)
    monkeypatch.setattr(openai_translate, "_client", lambda *a, **kw: object())

    result = openai_translate.validate_batch(
        ["The GPU is fast."],
        ["图形处理器很快。"],
        YT_SOURCE, {}, PreprocessResponse(),
        client=object(), model="m",
    )
    assert result.score == 75
    assert len(result.issues) == 1
    assert result.issues[0].index == 0
    assert result.issues[0].type == "terminology"


def test_validate_batch_clamps_score(monkeypatch):
    def fake_call_json(client, model, system, user):
        return {"score": 150, "issues": []}

    monkeypatch.setattr(openai_translate, "_call_json", fake_call_json)
    monkeypatch.setattr(openai_translate, "_client", lambda *a, **kw: object())

    result = openai_translate.validate_batch(
        ["hello"], ["你好"], YT_SOURCE, {}, PreprocessResponse(),
        client=object(), model="m",
    )
    assert result.score == 100


def test_validate_batch_returns_default_on_failure(monkeypatch):
    def fake_call_json(client, model, system, user):
        raise ValueError("api error")

    monkeypatch.setattr(openai_translate, "_call_json", fake_call_json)
    monkeypatch.setattr(openai_translate, "_client", lambda *a, **kw: object())

    result = openai_translate.validate_batch(
        ["hello"], ["你好"], YT_SOURCE, {}, PreprocessResponse(),
        client=object(), model="m",
    )
    assert result.score == 100
    assert result.issues == []


def test_correct_sentence_returns_corrected_text(monkeypatch):
    def fake_call_json(client, model, system, user):
        return {"dst": "GPU 很快。"}

    monkeypatch.setattr(openai_translate, "_call_json", fake_call_json)

    out = openai_translate.correct_sentence(
        "The GPU is fast.", "图形处理器很快。",
        "terminology issue", "Use GPU",
        YT_SOURCE, {}, PreprocessResponse(),
        client=object(), model="m",
    )
    assert out == "GPU 很快。"


def test_correct_sentence_keeps_original_on_failure(monkeypatch):
    def fake_call_json(client, model, system, user):
        raise ValueError("api error")

    monkeypatch.setattr(openai_translate, "_call_json", fake_call_json)

    original = "图形处理器很快。"
    out = openai_translate.correct_sentence(
        "The GPU is fast.", original,
        "terminology issue", "Use GPU",
        YT_SOURCE, {}, PreprocessResponse(),
        client=object(), model="m",
    )
    assert out == original


def test_correct_sentence_replaces_em_dash_for_zh(monkeypatch):
    def fake_call_json(client, model, system, user):
        return {"dst": "你好——世界"}

    monkeypatch.setattr(openai_translate, "_call_json", fake_call_json)

    out = openai_translate.correct_sentence(
        "Hello—world", "x", "issue", "fix",
        YT_SOURCE, {}, PreprocessResponse(),
        client=object(), model="m",
    )
    assert out == "你好，世界"


def test_validate_and_correct_skips_when_score_above_threshold(monkeypatch):
    validate_calls: list[int] = []

    def fake_call_json(client, model, system, user):
        validate_calls.append(1)
        return {"score": 90, "issues": []}

    monkeypatch.setattr(openai_translate, "_call_json", fake_call_json)
    monkeypatch.setattr(openai_translate, "_client", lambda *a, **kw: object())

    texts = ["Hello."]
    dst_list = ["你好。"]
    result, report = openai_translate.validate_and_correct(
        texts, dst_list, YT_SOURCE, {}, PreprocessResponse(),
        base_url="u", api_key="k", model="m", threshold=80,
    )
    assert result == ["你好。"]
    assert report["overall_score"] == 90
    assert report["corrected_count"] == 0


def test_validate_and_correct_corrects_below_threshold(monkeypatch):
    call_log: list[str] = []

    def fake_call_json(client, model, system, user):
        if "检查" in system or "check" in system.lower():
            call_log.append("validate")
            return {
                "score": 60,
                "issues": [
                    {"index": 0, "type": "accuracy", "problem": "wrong", "suggestion": "你好世界。"}
                ],
            }
        else:
            call_log.append("correct")
            return {"dst": "你好世界。"}

    monkeypatch.setattr(openai_translate, "_call_json", fake_call_json)
    monkeypatch.setattr(openai_translate, "_client", lambda *a, **kw: object())

    texts = ["Hello world."]
    dst_list = ["错误翻译。"]
    result, report = openai_translate.validate_and_correct(
        texts, dst_list, YT_SOURCE, {}, PreprocessResponse(),
        base_url="u", api_key="k", model="m", threshold=80,
    )
    assert result == ["你好世界。"]
    assert report["corrected_count"] == 1
    assert "validate" in call_log
    assert "correct" in call_log


def test_translate_asr_with_validation_enabled(tmp_path, monkeypatch):
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    asr_file = metadata / "asr.json"
    _write_asr(asr_file, 1)

    _stub_preprocess(monkeypatch)

    call_log: list[str] = []

    def fake_translate_batch(texts, source, meta, pre, **kw):
        call_log.append("translate")
        return ["你好。"]

    def fake_validate_and_correct(texts, dst_list, source, meta, pre, **kw):
        call_log.append("validate")
        return dst_list, {"overall_score": 90, "batches": [], "corrected_count": 0}

    monkeypatch.setattr(openai_translate, "translate_batch", fake_translate_batch)
    monkeypatch.setattr(openai_translate, "validate_and_correct", fake_validate_and_correct)

    settings = {**_settings(), "validation_enabled": "true"}
    openai_translate.translate_asr(asr_file, tmp_path, settings, YT_SOURCE)

    assert "translate" in call_log
    assert "validate" in call_log
    assert (metadata / "validation.json").exists()


def test_translate_asr_without_validation(tmp_path, monkeypatch):
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    asr_file = metadata / "asr.json"
    _write_asr(asr_file, 1)

    _stub_preprocess(monkeypatch)
    _stub_translate_batch(monkeypatch, lambda t: "你好")

    call_log: list[str] = []

    def fake_validate_and_correct(*args, **kwargs):
        call_log.append("validate")
        return args[1], {}

    monkeypatch.setattr(openai_translate, "validate_and_correct", fake_validate_and_correct)

    openai_translate.translate_asr(asr_file, tmp_path, _settings(), YT_SOURCE)
    assert "validate" not in call_log
    assert not (metadata / "validation.json").exists()
