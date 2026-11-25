"""任务配置 Schema 与校验端点。"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from ..languages import LanguageConfigError
from ..task_config import TaskConfig
from ..task_config_schema import build_task_config_schema, validate_task_config

router = APIRouter(prefix="/api/task-config", tags=["task-config"])


@router.get("/schema")
def get_task_config_schema() -> dict[str, Any]:
    """返回所有可配置阶段的元数据、实时选项列表与默认值。"""
    return build_task_config_schema()


@router.post("/validate")
def validate_task_config_endpoint(payload: TaskConfig) -> dict[str, Any]:
    """校验一份任务配置字典，返回规范化后的字段集合。"""
    try:
        validated = validate_task_config(payload)
    except LanguageConfigError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"config": validated}
