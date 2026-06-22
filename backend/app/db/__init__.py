"""``db`` package — SQLite access layer.

Submodules:
- ``connection``: ``connect()``, ``now_iso()``, ``ACTIVE_STATUSES``
- ``migrations``: ``init_db()``, ``backfill_titles_from_metadata()``
- ``tasks``: task + task_stages CRUD
- ``settings``: key/value settings (openai / ytdlp / funasr)
- ``translate_providers``: provider CRUD
"""

from .connection import ACTIVE_STATUSES, connect, now_iso
from .migrations import backfill_titles_from_metadata, init_db
from .settings import (
    VALID_FUNASR_USE_VLLM,
    get_funasr_settings,
    get_openai_settings,
    get_setting,
    get_ytdlp_settings,
    save_funasr_settings,
    save_openai_settings,
    save_ytdlp_settings,
    set_setting,
)
from .tasks import (
    count_tasks,
    create_task,
    delete_task,
    fail_stale_active_tasks,
    find_task_by_video_id,
    get_current_task,
    get_task,
    has_active_task,
    latest_task_id,
    list_tasks,
    log_path,
    reset_failed_for_resume,
    reset_single_stage_for_rerun,
    reset_stage_for_rerun,
    update_stage,
    update_task,
)
from .translate_providers import (
    create_translate_provider,
    delete_translate_provider,
    get_default_translate_provider,
    get_translate_provider,
    get_translate_provider_settings,
    list_translate_providers,
    update_translate_provider,
)

__all__ = [
    "ACTIVE_STATUSES",
    "VALID_FUNASR_USE_VLLM",
    "backfill_titles_from_metadata",
    "connect",
    "count_tasks",
    "create_task",
    "create_translate_provider",
    "delete_task",
    "delete_translate_provider",
    "fail_stale_active_tasks",
    "find_task_by_video_id",
    "get_current_task",
    "get_default_translate_provider",
    "get_funasr_settings",
    "get_openai_settings",
    "get_setting",
    "get_task",
    "get_translate_provider",
    "get_translate_provider_settings",
    "get_ytdlp_settings",
    "has_active_task",
    "init_db",
    "latest_task_id",
    "list_tasks",
    "list_translate_providers",
    "log_path",
    "now_iso",
    "reset_failed_for_resume",
    "reset_single_stage_for_rerun",
    "reset_stage_for_rerun",
    "save_funasr_settings",
    "save_openai_settings",
    "save_ytdlp_settings",
    "set_setting",
    "update_stage",
    "update_task",
    "update_translate_provider",
]
