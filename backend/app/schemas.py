"""Pydantic request models for the YouDub REST API.

Centralized so routers can import them without circular dependencies on
``main``.
"""

from __future__ import annotations

from pydantic import BaseModel

from .task_config import TaskConfig


class TaskCreate(BaseModel):
    url: str
    config: TaskConfig = TaskConfig()


class LocaldirTaskCreate(BaseModel):
    file_path: str
    config: TaskConfig = TaskConfig()


class TranslateProviderCreate(BaseModel):
    name: str
    base_url: str = ""
    api_key: str = ""
    model: str = ""
    is_default: bool = False


class TranslateProviderUpdate(BaseModel):
    name: str | None = None
    base_url: str | None = None
    api_key: str = ""
    clear_api_key: bool = False
    model: str | None = None
    is_default: bool | None = None


class YouTubeCookieUpdate(BaseModel):
    content: str


class OpenAISettingsUpdate(BaseModel):
    base_url: str
    api_key: str = ""
    clear_api_key: bool = False
    model: str
    translate_concurrency: str = ""


class OpenAIModelsRequest(BaseModel):
    base_url: str = ""
    api_key: str = ""


class YtdlpSettingsUpdate(BaseModel):
    proxy_port: str = ""


class FunasrSettingsUpdate(BaseModel):
    use_vllm: str = "auto"
