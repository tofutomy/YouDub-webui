from __future__ import annotations

from typing import Any, Mapping


DEFAULT_ASR_LANGUAGE = "en"
DEFAULT_TARGET_LANGUAGE = "zh"
DEFAULT_DIRECTION = f"{DEFAULT_ASR_LANGUAGE}-{DEFAULT_TARGET_LANGUAGE}"

SUPPORTED_DIRECTIONS: tuple[str, ...] = ("en-zh", "zh-en", "ja-zh")
SUPPORTED_DIRECTION_SET = frozenset(SUPPORTED_DIRECTIONS)
SUPPORTED_LANGUAGE_PAIRS: tuple[tuple[str, str], ...] = tuple(
    tuple(direction.split("-", 1)) for direction in SUPPORTED_DIRECTIONS
)


class LanguageConfigError(ValueError):
    """语言方向配置不在当前 UI/API 支持范围内。"""


def clean_language(value: Any) -> str | None:
    """把空字符串等同于未传，避免写入半空配置。"""
    if value is None:
        return None
    cleaned = str(value).strip()
    return cleaned or None


def direction_to_language_pair(direction: str | None) -> tuple[str, str]:
    cleaned = (direction or "").strip()
    if cleaned not in SUPPORTED_DIRECTION_SET:
        raise LanguageConfigError(f"Unsupported direction: {direction}")
    asr_language, target_language = cleaned.split("-", 1)
    return asr_language, target_language


def language_pair_to_direction(asr_language: str | None, target_language: str | None) -> str:
    asr = clean_language(asr_language)
    target = clean_language(target_language)
    if not asr or not target:
        raise LanguageConfigError("Both asr_language and target_language are required.")
    direction = f"{asr}-{target}"
    if direction not in SUPPORTED_DIRECTION_SET:
        raise LanguageConfigError(f"Unsupported language direction: {direction}")
    return direction


def normalize_language_pair(
    asr_language: str | None = None,
    target_language: str | None = None,
    *,
    fallback_direction: str | None = None,
) -> tuple[str, str]:
    """按支持方向补齐并校验语言对，缺省统一回退到 en-zh。"""
    asr = clean_language(asr_language)
    target = clean_language(target_language)
    if asr and target:
        direction = language_pair_to_direction(asr, target)
        return direction_to_language_pair(direction)

    if asr:
        for source, dest in SUPPORTED_LANGUAGE_PAIRS:
            if source == asr:
                return source, dest
        raise LanguageConfigError(f"Unsupported ASR language: {asr}")

    if target:
        for source, dest in SUPPORTED_LANGUAGE_PAIRS:
            if dest == target:
                return source, dest
        raise LanguageConfigError(f"Unsupported target language: {target}")

    if fallback_direction:
        try:
            return direction_to_language_pair(fallback_direction)
        except LanguageConfigError:
            pass
    return DEFAULT_ASR_LANGUAGE, DEFAULT_TARGET_LANGUAGE


def normalize_task_language_pair(
    task: Mapping[str, Any],
    *,
    fallback_direction: str | None = None,
) -> tuple[str, str]:
    """运行时兼容旧任务：缺语言列时可从 URL direction 兜底。"""
    return normalize_language_pair(
        task.get("asr_language"),
        task.get("target_language"),
        fallback_direction=fallback_direction,
    )


def normalize_config_fields(
    config: Any,
    *,
    only_set: bool = False,
    current_task: Mapping[str, Any] | None = None,
    fallback_direction: str | None = None,
) -> dict[str, Any]:
    """把 TaskConfig 转成 DB 字段，并保证语言列要么不写、要么成对写入。"""
    fields = config.to_db_fields(only_set=only_set)
    field_set = getattr(config, "model_fields_set", set())
    language_was_provided = any(
        key in field_set and clean_language(getattr(config, key, None))
        for key in ("asr_language", "target_language")
    )

    if only_set and not language_was_provided:
        fields.pop("asr_language", None)
        fields.pop("target_language", None)
        return fields

    current_task = current_task or {}
    request_asr = clean_language(getattr(config, "asr_language", None))
    request_target = clean_language(getattr(config, "target_language", None))
    current_asr = clean_language(current_task.get("asr_language"))
    current_target = clean_language(current_task.get("target_language"))

    if only_set and request_asr and not request_target:
        try:
            asr_language, target_language = normalize_language_pair(request_asr, current_target)
        except LanguageConfigError:
            asr_language, target_language = normalize_language_pair(request_asr)
    elif only_set and request_target and not request_asr:
        try:
            asr_language, target_language = normalize_language_pair(current_asr, request_target)
        except LanguageConfigError:
            asr_language, target_language = normalize_language_pair(target_language=request_target)
    else:
        asr_language, target_language = normalize_language_pair(
            request_asr or current_asr,
            request_target or current_target,
            fallback_direction=fallback_direction,
        )

    fields["asr_language"] = asr_language
    fields["target_language"] = target_language
    return fields
