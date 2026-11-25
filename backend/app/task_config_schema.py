"""任务配置 Schema 定义模块（前后端与外部工作台的单一事实来源）。"""

from __future__ import annotations

from typing import Any

from . import database
from .glossary import list_glossary_options
from .languages import (
    DEFAULT_ASR_LANGUAGE,
    DEFAULT_DIRECTION,
    DEFAULT_TARGET_LANGUAGE,
    SUPPORTED_DIRECTIONS,
    normalize_config_fields,
)
from .task_config import TaskConfig

# Schema 协议版本号，结构发生不兼容变更时递增
TASK_CONFIG_SCHEMA_VERSION = 1

# 任务配置各字段的默认值（与前端新建任务默认表单保持一致）
DEFAULT_TASK_CONFIG: dict[str, Any] = {
    "asr_model": "",
    "asr_language": DEFAULT_ASR_LANGUAGE,
    "target_language": DEFAULT_TARGET_LANGUAGE,
    "add_subtitles": True,
    "translate_mode": "sentence",
    "validate_translation": False,
    "stop_after_translate": False,
    "filter_fillers": False,
    "tts_mode": "controllable_clone",
    "translate_provider_id": "",
    "translation_glossary": "auto",
    "translation_terms": "",
    "demucs_model": "",
    "demucs_shifts": 1,
}

# 翻译方向的展示文案映射
_DIRECTION_LABELS: dict[str, dict[str, str]] = {
    "en-zh": {"zh": "英文 -> 中文", "en": "English -> Chinese"},
    "zh-en": {"zh": "中文 -> 英文", "en": "Chinese -> English"},
    "ja-zh": {"zh": "日文 -> 中文", "en": "Japanese -> Chinese"},
}

# 支持的语音识别模型列表（新增 ASR 模型时仅需在此处补充）
ASR_MODEL_OPTIONS: list[dict[str, Any]] = [
    {"value": "", "label": {"zh": "Whisper large-v3-turbo", "en": "Whisper large-v3-turbo"}},
    {"value": "whisper:large-v3", "label": {"zh": "Whisper large-v3", "en": "Whisper large-v3"}},
    {"value": "funasr:iic/SenseVoiceSmall", "label": {"zh": "SenseVoiceSmall", "en": "SenseVoiceSmall"}},
    {"value": "funasr:FunAudioLLM/Fun-ASR-Nano-2512", "label": {"zh": "Fun-ASR-Nano", "en": "Fun-ASR-Nano"}},
    {"value": "qwen3asr:Qwen/Qwen3-ASR-1.7B", "label": {"zh": "Qwen3-ASR", "en": "Qwen3-ASR"}},
    {"value": "arkasr:AutoArk-AI/ARK-ASR-3B", "label": {"zh": "ARK-ASR-3B", "en": "ARK-ASR-3B"}},
    {"value": "mossaudio:MOSS-Audio-4B-Instruct", "label": {"zh": "MOSS-Audio-4B", "en": "MOSS-Audio-4B"}},
]


def _direction_options() -> list[dict[str, Any]]:
    """根据后端支持的语言方向列表动态生成组合下拉选项。"""
    options: list[dict[str, Any]] = []
    for direction in SUPPORTED_DIRECTIONS:
        asr_lang, target_lang = direction.split("-", 1)
        label = _DIRECTION_LABELS.get(direction, {"zh": direction, "en": direction})
        options.append(
            {
                "value": direction,
                "label": label,
                "patch": {"asr_language": asr_lang, "target_language": target_lang},
            }
        )
    return options


def _translate_provider_options() -> list[dict[str, Any]]:
    """从 SQLite 实时读取翻译提供商列表并拼装为下拉选项。"""
    options: list[dict[str, Any]] = [
        {
            "value": "",
            "label": {"zh": "默认提供商", "en": "Default provider"},
        }
    ]
    for provider in database.list_translate_providers():
        suffix = " ★" if provider.get("is_default") else ""
        model = provider.get("model") or ""
        display = f"{provider['name']} ({model}){suffix}" if model else f"{provider['name']}{suffix}"
        options.append(
            {
                "value": provider["id"],
                "label": {"zh": display, "en": display},
                "is_default": bool(provider.get("is_default")),
            }
        )
    return options


def build_task_config_schema() -> dict[str, Any]:
    """构建完整的任务配置 Schema（包含各阶段字段、控件类型、实时选项及默认值）。

    scope 字段含义：
    - global: 通用配置，YouDub 首页、任务弹窗及外部工作台全局预设均显示。
    - per_task: 逐任务配置（如翻译方向/原片语言、游戏术语表），外部工作台全局页默认隐藏。
    - manual_only: 仅人工单任务适用（如翻译后中止），不兼容自动化出片流程。
    """
    stages: list[dict[str, Any]] = [
        {
            "stage": "separate",
            "label": {"zh": "分离模式", "en": "Separation mode"},
            "fields": [
                {
                    "id": "separate_mode",
                    "type": "composite_select",
                    "keys": ["demucs_model", "demucs_shifts"],
                    "scope": "global",
                    "label": {"zh": "分离模式", "en": "Separation mode"},
                    "default": "balanced",
                    "options": [
                        {
                            "value": "balanced",
                            "label": {"zh": "均衡（默认）", "en": "Balanced (default)"},
                            "description": {
                                "zh": "速度快 3 倍，音质损失极小（shifts=1）。",
                                "en": "Fast 3x speed with minimal quality loss (shifts=1).",
                            },
                            "patch": {"demucs_model": "", "demucs_shifts": 1},
                        },
                        {
                            "value": "high_quality",
                            "label": {"zh": "高精度", "en": "High quality"},
                            "description": {
                                "zh": "最佳音质，处理较慢（shifts=3）。",
                                "en": "Best audio quality, slower processing (shifts=3).",
                            },
                            "patch": {"demucs_model": "", "demucs_shifts": 3},
                        },
                    ],
                }
            ],
        },
        {
            "stage": "translate",
            "label": {"zh": "翻译", "en": "Translate"},
            "fields": [
                {
                    "id": "translation_glossary",
                    "key": "translation_glossary",
                    "keys": ["translation_glossary"],
                    "type": "select",
                    "scope": "per_task",
                    "label": {"zh": "翻译术语表", "en": "Translation glossary"},
                    "default": "auto",
                    "options": list_glossary_options(),
                },
                {
                    "id": "translation_terms",
                    "key": "translation_terms",
                    "keys": ["translation_terms"],
                    "type": "textarea",
                    "scope": "global",
                    "label": {"zh": "自定义术语", "en": "Custom terms"},
                    "hint": {
                        "zh": "每行一组，可用制表符或 => 分隔。自定义译名优先。",
                        "en": "Enter one pair per line, separated by a tab or =>. Custom names override built-in names.",
                    },
                    "default": "",
                    "rows": 3,
                    "max_length": 12000,
                },
                {
                    "id": "translate_provider_id",
                    "key": "translate_provider_id",
                    "keys": ["translate_provider_id"],
                    "type": "select",
                    "scope": "global",
                    "label": {"zh": "翻译提供商", "en": "Translate provider"},
                    "default": "",
                    "options": _translate_provider_options(),
                },
                {
                    "id": "direction",
                    "type": "composite_select",
                    "keys": ["asr_language", "target_language"],
                    "scope": "per_task",
                    "label": {"zh": "翻译方向", "en": "Translation direction"},
                    "default": DEFAULT_DIRECTION,
                    "options": _direction_options(),
                },
                {
                    "id": "translate_mode",
                    "key": "translate_mode",
                    "keys": ["translate_mode"],
                    "type": "select",
                    "scope": "global",
                    "label": {"zh": "翻译模式", "en": "Translation mode"},
                    "default": "sentence",
                    "options": [
                        {
                            "value": "sentence",
                            "label": {"zh": "逐句（独立翻译，快）", "en": "Sentence (per-sentence, fast)"},
                        },
                        {
                            "value": "batch",
                            "label": {"zh": "分批（上下文感知，慢）", "en": "Batch (context-aware, slower)"},
                        },
                    ],
                },
                {
                    "id": "validate_translation",
                    "key": "validate_translation",
                    "keys": ["validate_translation"],
                    "type": "boolean",
                    "scope": "global",
                    "label": {
                        "zh": "翻译校验（自动修正问题）",
                        "en": "Validate translation (auto-correct issues)",
                    },
                    "default": False,
                },
                {
                    "id": "stop_after_translate",
                    "key": "stop_after_translate",
                    "keys": ["stop_after_translate"],
                    "type": "boolean",
                    "scope": "manual_only",
                    "label": {
                        "zh": "翻译后中止（仅字幕，不配音）",
                        "en": "Stop after translation (subtitles only, no dubbing)",
                    },
                    "default": False,
                },
            ],
        },
        {
            "stage": "asr",
            "label": {"zh": "语音识别", "en": "ASR"},
            "fields": [
                {
                    "id": "asr_model",
                    "key": "asr_model",
                    "keys": ["asr_model"],
                    "type": "select",
                    "scope": "global",
                    "label": {"zh": "语音识别模型", "en": "ASR model"},
                    "default": "",
                    "options": ASR_MODEL_OPTIONS,
                }
            ],
        },
        {
            "stage": "asr_fix",
            "label": {"zh": "切分句子", "en": "Split sentences"},
            "fields": [
                {
                    "id": "filter_fillers",
                    "key": "filter_fillers",
                    "keys": ["filter_fillers"],
                    "type": "boolean",
                    "scope": "global",
                    "label": {
                        "zh": "过滤语气词 & 重复短句（嗯、啊、um、uh…，重复的短字符也会合并）",
                        "en": "Filter filler words & repeated short utterances (um, uh, 嗯, 啊…, short repeated chars are merged)",
                    },
                    "default": False,
                }
            ],
        },
        {
            "stage": "tts",
            "label": {"zh": "语音合成", "en": "TTS"},
            "fields": [
                {
                    "id": "tts_mode",
                    "key": "tts_mode",
                    "keys": ["tts_mode"],
                    "type": "select",
                    "scope": "global",
                    "label": {"zh": "TTS 克隆模式", "en": "TTS clone mode"},
                    "default": "controllable_clone",
                    "options": [
                        {
                            "value": "controllable_clone",
                            "label": {"zh": "可控声音克隆（默认）", "en": "Controllable clone (default)"},
                            "description": {
                                "zh": "从参考音频提取音色克隆，支持风格控制指令。",
                                "en": "Clones timbre from reference audio. Supports style control instructions.",
                            },
                        },
                        {
                            "value": "hifi_clone",
                            "label": {"zh": "高保真克隆（最高保真度）", "en": "Hi-Fi clone (highest fidelity)"},
                            "description": {
                                "zh": "使用参考音频+精确文本实现最高保真度克隆，控制指令会被忽略。",
                                "en": "Uses reference audio + transcript for maximum fidelity. Style control is ignored.",
                            },
                        },
                    ],
                }
            ],
        },
        {
            "stage": "merge_video",
            "label": {"zh": "视频合成", "en": "Merge video"},
            "fields": [
                {
                    "id": "add_subtitles",
                    "key": "add_subtitles",
                    "keys": ["add_subtitles"],
                    "type": "boolean",
                    "scope": "global",
                    "label": {"zh": "添加字幕", "en": "Add subtitles"},
                    "default": True,
                }
            ],
        },
    ]
    return {
        "version": TASK_CONFIG_SCHEMA_VERSION,
        "defaults": dict(DEFAULT_TASK_CONFIG),
        "stages": stages,
    }


def validate_task_config(config: TaskConfig) -> dict[str, Any]:
    """校验任务配置并返回规范化的 JSON 字典（保留原始布尔/整型/字符串类型）。"""
    # 先验证语言方向成对合法性（非法时抛出 LanguageConfigError）
    normalized_db = normalize_config_fields(config, only_set=True)
    result = config.model_dump(exclude_unset=True)
    # 若传入了语言方向，则使用规范化后的成对语言值
    if "asr_language" in normalized_db:
        result["asr_language"] = normalized_db["asr_language"]
    if "target_language" in normalized_db:
        result["target_language"] = normalized_db["target_language"]
    return result
