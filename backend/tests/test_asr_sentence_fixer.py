from __future__ import annotations

import json

import pytest

from backend.app.adapters import asr_sentence_fixer


def _utt(text: str, start: int, end: int) -> dict:
    return {"text": text, "start_time": start, "end_time": end}


def _write_asr(tmp_path, utterances: list, duration: int = 10000, text: str = "") -> tuple:
    session = tmp_path / "session"
    (session / "metadata").mkdir(parents=True)
    asr_file = session / "metadata" / "asr.json"
    payload = {
        "audio_info": {"duration": duration},
        "result": {"text": text, "utterances": utterances},
    }
    asr_file.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return asr_file, session


def test_fix_asr_sentences_passes_through_utterances(tmp_path):
    utts = [_utt("Hello world.", 100, 1200), _utt("How are you?", 1500, 2800)]
    asr_file, session = _write_asr(tmp_path, utts)

    fixed = asr_sentence_fixer.fix_asr_sentences(asr_file, session, start_pad=50, end_pad=100)
    out = json.loads(fixed.read_text(encoding="utf-8"))["result"]["utterances"]

    assert [u["text"] for u in out] == ["Hello world.", "How are you?"]


def test_fix_asr_sentences_drops_empty_text(tmp_path):
    utts = [_utt("Hello.", 0, 500), _utt("   ", 600, 800), _utt("World.", 900, 1500)]
    asr_file, session = _write_asr(tmp_path, utts)

    fixed = asr_sentence_fixer.fix_asr_sentences(asr_file, session)
    out = json.loads(fixed.read_text(encoding="utf-8"))["result"]["utterances"]

    assert [u["text"] for u in out] == ["Hello.", "World."]


def test_fix_asr_sentences_applies_padding_within_gap(tmp_path):
    utts = [_utt("a", 1000, 2000), _utt("b", 3000, 4000)]
    asr_file, session = _write_asr(tmp_path, utts, duration=5000)

    fixed = asr_sentence_fixer.fix_asr_sentences(asr_file, session, start_pad=100, end_pad=300)
    out = json.loads(fixed.read_text(encoding="utf-8"))["result"]["utterances"]

    assert out[0]["start_time"] == 900
    assert out[0]["end_time"] == 2300
    assert out[1]["start_time"] == 2900
    assert out[1]["end_time"] == 4300


def test_fix_asr_sentences_clamps_to_duration(tmp_path):
    utts = [_utt("only", 100, 4900)]
    asr_file, session = _write_asr(tmp_path, utts, duration=5000)

    fixed = asr_sentence_fixer.fix_asr_sentences(asr_file, session, start_pad=200, end_pad=500)
    out = json.loads(fixed.read_text(encoding="utf-8"))["result"]["utterances"]

    assert out[0]["start_time"] == 0  # 100 - 200 -> clamp 0
    assert out[0]["end_time"] == 5000


def test_fix_asr_sentences_raises_when_empty(tmp_path):
    asr_file, session = _write_asr(tmp_path, [_utt("  ", 0, 100)])

    with pytest.raises(RuntimeError):
        asr_sentence_fixer.fix_asr_sentences(asr_file, session)


def test_fix_asr_sentences_reuses_cache(tmp_path):
    utts = [_utt("hi", 0, 500)]
    asr_file, session = _write_asr(tmp_path, utts)

    first = asr_sentence_fixer.fix_asr_sentences(asr_file, session)
    first.write_text('{"already": true}', encoding="utf-8")
    second = asr_sentence_fixer.fix_asr_sentences(asr_file, session)

    assert json.loads(second.read_text(encoding="utf-8")) == {"already": True}


# ---------------------------------------------------------------------------
# _dedup_repeated
# ---------------------------------------------------------------------------

def test_dedup_merges_consecutive_same_short_text():
    utts = [
        _utt("2", 1000, 1100),
        _utt("2", 1120, 1200),
        _utt("2", 1220, 1300),
        _utt("2", 1350, 1400),
        _utt("hello", 2000, 2500),
    ]
    result = asr_sentence_fixer._dedup_repeated(utts)
    assert len(result) == 2
    assert result[0]["text"] == "2"
    assert result[0]["start_time"] == 1000
    assert result[0]["end_time"] == 1400
    assert result[1]["text"] == "hello"


def test_dedup_respects_max_gap():
    utts = [
        _utt("2", 1000, 1100),
        _utt("2", 5000, 5100),  # gap = 3900 > 1000
    ]
    result = asr_sentence_fixer._dedup_repeated(utts)
    assert len(result) == 2
    assert result[0]["start_time"] == 1000
    assert result[0]["end_time"] == 1100
    assert result[1]["start_time"] == 5000


def test_dedup_ignores_long_text():
    utts = [
        _utt("おやすみなさい", 1000, 1700),
        _utt("おやすみなさい", 1750, 2400),
        _utt("おやすみなさい", 2450, 3100),
    ]
    result = asr_sentence_fixer._dedup_repeated(utts)
    assert len(result) == 3  # 6 chars > 4, no merge


def test_dedup_caps_duration_at_4000ms():
    # All within gap ≤ 1000ms → merged into one group, end_time capped
    utts = [
        _utt("2", 1000, 1100),
        _utt("2", 1500, 1600),
        _utt("2", 2000, 2100),
        _utt("2", 2500, 2600),
        _utt("2", 3000, 3100),
        _utt("2", 3500, 3600),
        _utt("2", 4000, 4100),
        _utt("2", 4500, 4600),
        _utt("2", 5000, 5100),  # gap=400, extend → cap = 1000+4000 = 5000
        _utt("2", 5500, 5600),  # gap=500, still merged, end stays 5000
    ]
    result = asr_sentence_fixer._dedup_repeated(utts)
    assert len(result) == 1
    assert result[0]["start_time"] == 1000
    assert result[0]["end_time"] == 5000  # capped at start + 4000


def test_dedup_preserves_order_and_different_texts():
    utts = [
        _utt("1", 100, 200),
        _utt("1", 250, 300),
        _utt("2", 400, 500),
        _utt("2", 520, 600),
        _utt("2", 650, 700),
        _utt("3", 800, 900),
    ]
    result = asr_sentence_fixer._dedup_repeated(utts)
    assert len(result) == 3
    assert [u["text"] for u in result] == ["1", "2", "3"]
    assert result[1]["start_time"] == 400
    assert result[1]["end_time"] == 700


def test_dedup_empty_input():
    assert asr_sentence_fixer._dedup_repeated([]) == []


def test_dedup_single_utterance():
    utts = [_utt("2", 100, 200)]
    result = asr_sentence_fixer._dedup_repeated(utts)
    assert len(result) == 1
    assert result[0]["text"] == "2"


def test_dedup_writes_log(tmp_path):
    log_path = tmp_path / "dedup_log.json"
    utts = [
        _utt("2", 1000, 1100),
        _utt("2", 1150, 1200),
        _utt("ok", 2000, 2500),
    ]
    asr_sentence_fixer._dedup_repeated(utts, log_path=log_path)
    log = json.loads(log_path.read_text(encoding="utf-8"))
    assert log["dedup_removed_count"] == 1
    assert log["dedup_kept_count"] == 2


def test_fix_asr_sentences_dedup_with_filter_fillers(tmp_path):
    """Integration: filter_fillers triggers both filler removal and dedup."""
    utts = [
        _utt("嗯", 100, 200),       # filler → removed by filter
        _utt("2", 1000, 1100),       # short repeated → merged by dedup
        _utt("2", 1120, 1200),
        _utt("2", 1250, 1300),
        _utt("hello world", 2000, 3000),
    ]
    asr_file, session = _write_asr(tmp_path, utts, duration=5000)
    fixed = asr_sentence_fixer.fix_asr_sentences(
        asr_file, session, filter_fillers=True, start_pad=0, end_pad=0,
    )
    out = json.loads(fixed.read_text(encoding="utf-8"))["result"]["utterances"]
    assert len(out) == 2
    assert out[0]["text"] == "2"
    assert out[0]["end_time"] == 1300  # merged
    assert out[1]["text"] == "hello world"
