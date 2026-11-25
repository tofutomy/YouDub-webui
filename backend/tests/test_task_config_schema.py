"""任务配置 Schema 与校验接口单元测试。"""

from __future__ import annotations

from fastapi.testclient import TestClient

from backend.app import database, main
from backend.app.glossary import GAMES
from backend.app.task_config import TaskConfig
from backend.tests.test_settings_and_api import configure_tmp_runtime


def test_schema_covers_all_task_config_fields_and_dynamic_options(monkeypatch, tmp_path):
    """验证 Schema 覆盖 TaskConfig 全部字段，且实时同步翻译提供商和游戏术语表。"""
    configure_tmp_runtime(monkeypatch, tmp_path)
    provider_id = database.create_translate_provider(
        name="DeepSeek",
        base_url="https://api.deepseek.com/v1",
        api_key="sk-test",
        model="deepseek-chat",
        is_default=True,
    )
    client = TestClient(main.app)

    response = client.get("/api/task-config/schema")
    assert response.status_code == 200
    schema = response.json()
    assert schema["version"] == 1

    # 默认值与字段键集合必须完整覆盖 TaskConfig
    model_keys = set(TaskConfig.model_fields.keys())
    assert set(schema["defaults"].keys()) == model_keys

    all_fields = [field for stage in schema["stages"] for field in stage["fields"]]
    schema_keys = {key for field in all_fields for key in field["keys"]}
    assert schema_keys == model_keys

    by_id = {field["id"]: field for field in all_fields}
    # 翻译提供商应包含默认项和刚刚创建的 DeepSeek
    provider_values = {opt["value"] for opt in by_id["translate_provider_id"]["options"]}
    assert "" in provider_values
    assert provider_id in provider_values

    # 游戏术语表应包含 auto、none 与全部 GAMES 键
    glossary_values = {opt["value"] for opt in by_id["translation_glossary"]["options"]}
    assert {"auto", "none", *GAMES.keys()}.issubset(glossary_values)

    # 逐任务配置与仅人工配置的作用域标记准确
    assert by_id["direction"]["scope"] == "per_task"
    assert by_id["translation_glossary"]["scope"] == "per_task"
    assert by_id["stop_after_translate"]["scope"] == "manual_only"
    assert by_id["asr_model"]["scope"] == "global"
    assert by_id["separate_mode"]["scope"] == "global"
    assert by_id["tts_mode"]["scope"] == "global"
    assert by_id["add_subtitles"]["scope"] == "global"


def test_validate_task_config_endpoint(monkeypatch, tmp_path):
    """验证配置校验端点能通过合法配置，并拦截非法自定义术语与语言方向。"""
    configure_tmp_runtime(monkeypatch, tmp_path)
    client = TestClient(main.app)

    ok = client.post(
        "/api/task-config/validate",
        json={
            "asr_model": "qwen3asr:Qwen/Qwen3-ASR-1.7B",
            "demucs_model": "",
            "demucs_shifts": 3,
            "filter_fillers": True,
            "translate_mode": "batch",
            "validate_translation": True,
            "translation_terms": "Divine Orb => 神圣石",
            "tts_mode": "hifi_clone",
            "add_subtitles": False,
        },
    )
    assert ok.status_code == 200
    assert ok.json()["config"]["demucs_shifts"] == 3
    assert ok.json()["config"]["add_subtitles"] is False

    bad_terms = client.post(
        "/api/task-config/validate",
        json={"translation_terms": "缺少分隔符"},
    )
    assert bad_terms.status_code == 422

    bad_direction = client.post(
        "/api/task-config/validate",
        json={"asr_language": "fr", "target_language": "zh"},
    )
    assert bad_direction.status_code == 422
