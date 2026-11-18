import json
from unittest.mock import patch

import pytest

from backend.app.adapters.openai_translate import translate_asr
from backend.app.glossary import detect_glossary, glossary_prompt
from backend.app.sources import get_source_for_task
from backend.app.task_config import TaskConfig


def test_game_selection_and_term_precedence():
    assert detect_glossary({"title": "POE2 Bloodmage build"}) == "poe2"
    assert detect_glossary({"title": "PoE 1 and PoE 2 comparison"}) == "none"
    assert detect_glossary({"title": "New build", "description": "Last Epoch guide"}) == "last_epoch"
    for game in ("poe1", "poe2", "last_epoch", "diablo4"):
        assert glossary_prompt(game, {}, ["Divine Orb, Radiant Lance or Nightmare Dungeons"])[1]
    selected, hint = glossary_prompt("poe1", {}, ["I tried RF and Spirit"], "Righteous Fire => 自定义译名")
    assert selected == "poe1"
    assert "RF => 自定义译名" in hint
    assert "Spirit =>" not in hint
    assert not glossary_prompt("poe1", {}, ["RF"], target_language="en")[1]
    with pytest.raises(ValueError, match="自定义术语"):
        TaskConfig(translation_terms="缺少对照")


@pytest.mark.parametrize("game,title,english,chinese", [
    ('diablo2', 'Diablo II Resurrected', 'Blessed Hammer', '祝福之锤'),
    ('diablo3', 'Diablo III Build', 'Hammer of the Ancients', '先祖之鎚'),
    ('diablo_immortal', 'Diablo: Immortal', 'Disintegrate', '瓦解射线'),
    ('grim_dawn', 'Grim Dawn Build', "Albrecht's Aether Ray", '阿尔布雷特的虚化射线'),
    ('torchlight_infinite', 'Torchlight: Infinite', 'Aegis of Fire', '烈火神盾'),
    ('hero_siege', 'Hero Siege Build', 'Fury of Tarethiel', '塔雷蒂尔之怒'),
    ('slormancer', 'The Slormancer Build', 'Banners of War', '战争之旗'),
    ('titan_quest2', 'Titan Quest II Build', 'Fissure', '裂隙'),
    ('torchlight2', 'Torchlight II Build', 'Magma Spear', '岩浆长矛'),
    ('chronicon', 'Chronicon Build', 'Reflective Shield', '反射之盾'),
])
def test_expanded_game_glossaries_use_verified_names(game, title, english, chinese):
    assert TaskConfig(translation_glossary=game).translation_glossary == game
    assert detect_glossary({"title": title}) == game
    selected, hint = glossary_prompt("auto", {"title": title}, [english])
    assert selected == game
    assert f"{english} => {chinese}" in hint


@pytest.mark.parametrize("mode", ["sentence", "batch"])
def test_glossary_reaches_actual_translation_prompts(tmp_path, mode):
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    (metadata / "ytdlp_info.json").write_text(json.dumps({"title": "PoE 2 Blood Mage"}), encoding="utf-8")
    asr = metadata / "asr_fixed.json"
    asr.write_text(json.dumps({"result": {"utterances": [
        {"text": "Blood Mage uses Lightning Arrow", "start_time": 0, "end_time": 1000}
    ]}}), encoding="utf-8")
    source = get_source_for_task({"url": "https://www.youtube.com/watch?v=abcdefghijk",
                                  "asr_language": "en", "target_language": "zh"})
    settings = {"base_url": "https://opencode.ai/inference/go/openai/v1", "api_key": "test", "model": "test",
                "translate_mode": mode, "translation_glossary": "auto",
                "translation_terms": "Lightning Arrow => 魔改电箭", "validation_enabled": "true"}
    received = []

    def answer(client, model, system, user):
        assert "Blood Mage => 血法师" in system
        assert "Lightning Arrow => 魔改电箭" in system
        received.append(user)
        if system.startswith("You output strict JSON only."):
            return {"summary": "游戏配装"}
        if " -> " in user:
            return {"score": 100, "issues": []}
        return {"translations": ["血法师使用魔改电箭。"]} if mode == "batch" else {"dst": "血法师使用魔改电箭。"}

    with patch("backend.app.adapters.openai_translate._client") as make_client, patch(
        "backend.app.adapters.openai_translate._call_json", side_effect=answer
    ):
        output = translate_asr(asr, tmp_path, settings, source)
    sessions = {call.kwargs["session_id"] for call in make_client.call_args_list}
    assert len(sessions) == 1 and sessions.pop()
    assert len(received) == 3
    assert json.loads(output.read_text(encoding="utf-8"))["translation"][0]["dst"] == "血法师使用魔改电箭。"



@pytest.mark.parametrize("mode", ["sentence", "batch"])
def test_expanded_glossary_finds_terms_beyond_initial_prompt(tmp_path, monkeypatch, mode):
    dictionary = tmp_path / "terms"
    dictionary.mkdir()
    entries = {f"Vanguard Technique {index:02d}": f"先驱技巧{index:02d}" for index in range(45)}
    entries["Shattered Sentinel"] = "破碎哨兵"
    effects = {"#% more Cold Damage": "#% 额外冰霜伤害"}
    (dictionary / "poe2.json").write_text(
        json.dumps({"terms": entries, "effects": effects}, ensure_ascii=False), encoding="utf-8"
    )
    monkeypatch.setattr("backend.app.glossary.GLOSSARY_DIR", dictionary)

    metadata = tmp_path / "metadata"
    metadata.mkdir()
    asr = metadata / "asr_fixed.json"
    lines = list(entries)[:-1] + ["Shattered Sentinel deals 25% more Cold Damage"]
    asr.write_text(json.dumps({"result": {"utterances": [
        {"text": text, "start_time": index * 1000, "end_time": (index + 1) * 1000}
        for index, text in enumerate(lines)
    ]}}), encoding="utf-8")
    settings = {"base_url": "http://localhost", "api_key": "test", "model": "test",
                "translate_mode": mode, "translation_glossary": "poe2"}
    source = get_source_for_task({"url": "https://www.youtube.com/watch?v=abcdefghijk",
                                  "asr_language": "en", "target_language": "zh"})
    found = []

    def answer(client, model, system, user):
        if system.startswith("You output strict JSON only."):
            assert "Shattered Sentinel =>" not in system
            return {"summary": "游戏配装"}
        if "Shattered Sentinel" in user:
            assert "Shattered Sentinel => 破碎哨兵" in system
            assert "#% more Cold Damage => #% 额外冰霜伤害" in system
            assert "Vanguard Technique 00 =>" not in system
            found.append(user)
        if mode == "batch":
            return {"translations": ["已翻译"] * len(user.splitlines())}
        return {"dst": "已翻译"}

    with patch("backend.app.adapters.openai_translate._client"), patch(
        "backend.app.adapters.openai_translate._call_json", side_effect=answer
    ):
        translate_asr(asr, tmp_path, settings, source)
    assert found
