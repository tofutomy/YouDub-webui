from __future__ import annotations

from typing import Any

from pydantic import BaseModel

_CLEAR_ON_NULL = frozenset({"asr_model", "tts_mode", "translate_provider_id"})
_EMPTY_TO_NONE = frozenset({"asr_model", "translate_mode", "tts_mode", "translate_provider_id"})
_BOOL_TO_INT = frozenset({"add_subtitles", "validate_translation", "stop_after_translate"})


class TaskConfig(BaseModel):
    asr_model: str | None = None
    asr_language: str | None = None
    target_language: str | None = None
    add_subtitles: bool | None = None
    translate_mode: str | None = None
    validate_translation: bool | None = None
    tts_mode: str | None = None
    translate_provider_id: str | None = None
    stop_after_translate: bool | None = None

    def to_db_fields(self, only_set: bool = False) -> dict[str, object]:
        fields: dict[str, object] = {}
        if only_set:
            keys = self.model_fields_set
        else:
            keys = set(self.model_dump().keys())
        for key in keys:
            value = getattr(self, key, None)
            if value is None:
                if only_set and key in _CLEAR_ON_NULL:
                    fields[key] = None
                continue
            if key in _BOOL_TO_INT:
                fields[key] = int(value)
            elif key in _EMPTY_TO_NONE:
                fields[key] = value or None
            else:
                fields[key] = value
        return fields

    @classmethod
    def from_task_dict(cls, task: dict[str, Any]) -> "TaskConfig":
        return cls(
            asr_model=task.get("asr_model"),
            asr_language=task.get("asr_language"),
            target_language=task.get("target_language"),
            add_subtitles=bool(task.get("add_subtitles", 1)),
            translate_mode=task.get("translate_mode"),
            validate_translation=bool(task.get("validate_translation")),
            tts_mode=task.get("tts_mode"),
            translate_provider_id=task.get("translate_provider_id"),
            stop_after_translate=bool(task.get("stop_after_translate")),
        )
