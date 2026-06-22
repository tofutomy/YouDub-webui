# YouDub-webui 代码审查报告（2026-06）

> 目的：评估项目健康度，识别需要重构的地方，支持可持续开发。

## 📊 总体评分：7.5/10

项目整体架构清晰、测试覆盖扎实、文档完善，处于健康状态。主要问题集中在**单文件过大**和**跨适配器重复代码**两类，属于可持续开发需要逐步偿还的"技术债"，没有阻塞性缺陷。

---

## 🟡 建议改进（按优先级排序）

### P0 — 拆分超大文件（影响可维护性）

**1. `backend/app/main.py`（712 行，30+ 端点）**
- 所有 REST 端点、Pydantic 模型、CORS 配置、辅助函数全堆在一个文件
- 💡 按 FastAPI Router 拆分：
  - `routers/tasks.py`（任务 CRUD + 阶段重跑/清理）
  - `routers/settings.py`（openai/ytdlp/funasr settings）
  - `routers/cookies.py`（YouTube cookie）
  - `routers/translate_providers.py`（提供商 CRUD）
  - `schemas.py`（所有 Pydantic 模型）
  - `main.py` 只保留 lifespan + app 装配

**2. `backend/app/database.py`（587 行，30+ 函数）**
- tasks / stages / settings / translate_providers 四类 CRUD 混在一起
- `init_db()` 的 schema 迁移是一长串 `if "x" not in columns: ALTER TABLE`，随字段增加会越来越脆
- 💡 拆分为 `repositories/tasks.py`、`stages.py`、`settings.py`、`translate_providers.py`；迁移逻辑独立到 `migrations.py`，或引入 alembic

**3. `backend/app/adapters/openai_translate.py`（804 行，全项目最大）**
- preprocess / 句子翻译 / 批量翻译 / 校验 / 修正 / orchestrator 全在一个文件
- `validate_and_correct` 单函数 ~150 行
- 💡 拆分为 `translate/preprocess.py`、`sentence.py`、`batch.py`、`validate.py`、`orchestrator.py`

**4. `backend/app/adapters/funasr_asr.py`（628 行）**
- 模型加载 + vLLM 调度 + 时间戳转换 + 句子分组 + 文本清洗全混在一起
- 💡 拆分为 `funasr/model_loader.py`、`result_converter.py`、`text_segmentation.py`

**5. `apps/web/src/app/tasks/[id]/page.tsx`（812 行，30+ `useState`）**
- 所有弹窗状态、handler、轮询、渲染都在 `TaskDetailPage` 一个组件
- 💡 抽取：
  - `hooks/useTaskPolling.ts`（轮询 + 日志拉取）
  - `components/confirm-dialog.tsx`（复用 delete/rerun/stop/rerun-stage/clear-stage 5 个确认弹窗）
  - `components/stage-config-dialog.tsx`（阶段配置弹窗）
  - `components/danger-zone.tsx`（重跑/删除区）

**6. `apps/web/src/components/settings-dialog.tsx`（526 行）**
- 💡 抽取 `components/provider-form.tsx`（单个 provider 的展开表单）

---

### P1 — 消除跨适配器重复代码

**7. 时间转换函数重复定义**
- `funasr_asr.py` 的 `_to_ms` / `_time_value_to_ms`
- `remote_funasr_asr.py` 的 `_seconds_to_ms`
- `qwen3_asr.py` 的 `_to_ms`
- 💡 抽到 `adapters/_time_utils.py`

**8. 语言映射表重复**
- `funasr_asr.py` 的 `_LANG_TO_FUNASR` / `_LANG_TO_FUNASR_VLLM`
- `remote_funasr_asr.py` 的 `_LANG_TO_FUNASR` / `_LANG_TO_OPENAI`
- `qwen3_asr.py` 的 `_LANG_MAP`
- `sources.py` 的 `LANG_NAMES`
- 💡 统一到 `adapters/_lang_map.py` 或 `sources.py` 作为单一来源

**9. 句子切分逻辑重复**
- `funasr_asr.py`：`_split_text_to_sentences` / `_group_chars_to_sentences` / `_word_timestamps_to_sentences`
- `qwen3_asr.py`：`_group_words_to_utterances`
- `asr_sentence_fixer.py`：断句逻辑
- 都在处理"按 `.!?。！？` 切分 + 小数点保护 + CJK 处理"
- 💡 抽到 `adapters/_text_segmentation.py`，各适配器复用

**10. 阶段产出路径硬编码两处**
- `main.py` 的 `_clear_stage_output`（删除产出）
- `pipeline.py` 的 `_restore_cached_stage`（恢复缓存）
- 两处各自维护一份"阶段 → 文件路径"映射，新增阶段或产出时容易遗漏同步
- 💡 在 `stages.py` 的 `StageSpec` 上增加 `outputs: tuple[str, ...]` 字段，两处都从 STAGES 读取

---

### P2 — 设计模式优化

**11. `pipeline.py` 的 ASR 路由用 if-elif 前缀判断**
```python
if asr_model.startswith("qwen3asr:"): ...
elif asr_model.startswith("funasr:"): ...
else: # whisper
```
- 每加一个 ASR 后端都要改 `pipeline.py`
- 💡 用 registry/strategy：每个 ASR 适配器注册 `prefix` + `recognize(vocals, session, source, model_id)`，`pipeline.py` 查表分发

**12. `database.py` 的 `update_task` / `update_stage` 用 `**fields` 拼 SQL**
- 没有字段白名单，字段名直接拼进 `UPDATE tasks SET {key} = ?`
- 当前调用方都来自代码内部，风险低；但 `update_task` 是公开 API，未来误传用户输入会有 SQL 注入风险
- 💡 加白名单常量 `TASK_WRITABLE_FIELDS` / `STAGE_WRITABLE_FIELDS`，校验后拼接

**13. `database.py` 的 `get_task` 用 `CASE WHEN name WHEN 'download' THEN 1...` 硬编码阶段顺序**
- 与 `stages.py` 的 `STAGES` 重复定义
- 💡 从 `STAGES` 动态生成 CASE 子句

**14. `worker.py` 的 `_stop_events` 字典可能泄漏**
- `request_stop` 创建 `Event`，`consume_stop_event` pop；但若任务被 stop 后直接 delete（未走 worker loop），Event 永远不会被 consume
- 影响：小内存泄漏
- 💡 在 `delete_task` 时调用 `worker.cleanup_stop(task_id)`

**15. `openai_translate.py` 内部 `import shutil` 等函数内 import**
- 重依赖（funasr/whisper/torch）的延迟 import 是合理的，但 `shutil`/`json` 这类标准库没必要延迟
- 💡 提到模块顶部，减少阅读干扰

---

### P3 — 工程化补强

**16. 缺少静态类型检查配置**
- 后端有完整 type hints 但没看到 `mypy.ini` / `pyright` 配置
- 前端 `tsconfig.json` 未开启严格模式
- 💡 加 `pyright` 或 `mypy` strict 检查（可先从 `backend/app` 开始），前端开启 `strict: true`

**17. `i18n.tsx` 单文件 525 行**
- en/zh 两份大对象 + `STAGE_INFO` + Context 全在一个文件
- 💡 按模块拆分：`i18n/en/home.ts`、`i18n/zh/home.ts` 等，或用 `next-intl` 这类成熟方案

**18. 前端 `api.ts` 的 `request` 错误处理丢失 status code**
- 所有错误都变成 `new Error(body.detail)`，调用方拿不到 HTTP status
- 💡 自定义 `ApiError extends Error` 保留 `status`，便于 UI 按 404/409/422 区分处理

**19. `config.py` 的 `openai_defaults()` 在 `init_db()` 中被调用 2 次**
- 小重复，可缓存为局部变量

---

## 🟢 亮点

- **Pipeline 设计优秀**：9 阶段 + 缓存复用 + 单阶段/级联重跑 + 清理产出，状态机清晰
- **协作式取消**：`stop_requested` + 进度检查点 + 阶段边界三重检查，长任务可在 ~2s 内中断
- **翻译校验+修正+checkpoint**：分段校验 → 评分 → 低于阈值修正 → 失败可恢复，工程化程度高
- **测试覆盖扎实**：19 个测试文件 ~2500 行，覆盖 settings/API/pipeline/各适配器
- **安全性细节到位**：API key masking、session 路径 `_is_inside_workfolder` 校验、CORS 正则、cookie 不回显
- **设备管理周到**：组件级设备分配 + MPS 对 Whisper 的 float64 DTW 回退处理
- **文档完善**：`AGENTS.md` 是难得一见的高质量 agent 指南
- **适配器隔离好**：每个 ASR/TTS 后端独立文件，互不污染

---

## 📝 总结

项目处于**健康但开始出现单文件膨胀**的阶段。建议按 P0 → P1 → P2 顺序逐步推进，每个 P0 拆分都伴随测试回归验证。不建议一次性大重构，保持"每次 PR 拆一个文件"的节奏即可。

---

## 🔧 重构执行记录（2026-06-22）

### 已完成

| 编号 | 项目 | 改动 | 测试 |
|------|------|------|------|
| P0-1 | 拆分 `main.py` | 633→71 行，端点分散到 `routers/{tasks,cookies,settings,translate_providers}.py` + `schemas.py` + `routers/_common.py`；`include_router()` 接入；`rerun_single_stage` 加后台线程；`_clear_stage_output` 使用 `resolve_stage_outputs()` | 216/216 ✅ |
| P0-2 | 拆分 `database.py` | 587→17 行 facade，实现移到 `db/{connection,migrations,tasks,settings,translate_providers}.py` | 216/216 ✅ |
| P0-5 | 拆分 `tasks/[id]/page.tsx` | 812→660 行，抽取 `useTaskPolling` hook + `ConfirmDialog`/`StageInfoDialog`/`StageConfigDialog` 组件 | lint 0/0 ✅ |
| P1-7 | 抽取时间转换工具 | 新建 `adapters/_time_utils.py`，`funasr_asr`/`remote_funasr_asr`/`qwen3_asr` 复用 | 216/216 ✅ |
| P1-8 | 统一语言映射表 | 新建 `adapters/_lang_map.py`，4 个适配器 + `sources.py` 复用 | 216/216 ✅ |
| P1-10 | 阶段产出路径集中定义 | `StageSpec` 增加 `outputs` 字段 + `resolve_stage_outputs()`，`_clear_stage_output` 从 60 行降到 12 行 | 216/216 ✅ |
| P2-13 | `get_task` CASE 动态生成 | 从 `STAGES` 动态生成 SQL，不再硬编码阶段顺序 | 216/216 ✅ |
| P2-14 | worker stop_events 泄漏修复 | 新增 `worker.cleanup_stop()`，`delete_task` 时调用 | 216/216 ✅ |
| P2-12 | update_task/stage 字段白名单 | 新增 `_TASK_WRITABLE_FIELDS`/`_STAGE_WRITABLE_FIELDS`，防 SQL 注入 | 216/216 ✅ |

### 暂缓（需先重构测试）

| 编号 | 项目 | 原因 |
|------|------|------|
| P0-3 | 拆分 `openai_translate.py`（804行） | `test_translation.py` 深度 monkeypatch 内部符号（`_client`/`_call_json`/`_translate_system` 等 ~40 处），拆分子模块后 patch 会失效，需先重构测试 |
| P0-4 | 拆分 `funasr_asr.py`（628行） | 同上，`test_funasr_asr.py` 508 行深度依赖内部符号 |
| P0-6 | 拆分 `settings-dialog.tsx` | provider 表单抽取需传递大量回调 props，收益有限（~100行） |

### 未做（P1-9 / P2-11 / P2-15）

- **P1-9** 句子切分逻辑抽取：3 个适配器的断句逻辑差异较大（CJK vs 空格分隔 vs 字符级对齐），强行统一可能引入回归，建议观察后再做
- **P2-11** ASR 路由 strategy 模式：当前 if-elif 仅 3 分支，收益不高
- **P2-15** 标准库 import 提到顶部：小改动，可随手做

### P3 已完成

| 编号 | 项目 | 改动 | 测试 |
|------|------|------|------|
| P3-16 | pyright 配置 | 新建 `pyrightconfig.json`（basic 模式 + 噪音抑制）；修复 `stages.py` 缺失 `Path` 导入 bug；前端 `tsconfig.json` 已有 `strict: true` | pyright 0 errors 0 warnings ✅ |
| P3-18 | ApiError 保留 status code | 新增 `ApiError` 类，`request`/`getTaskLog`/`uploadLocalTask` 统一抛出 `ApiError`，调用方可按 status 分支处理 | lint 0/0 ✅ |
| P3-19 | `openai_defaults()` 重复调用 | `init_db()` 中缓存为局部变量 | 216/216 ✅ |

