# YouDub WebUI - Agent 指南

> 视频本地化工具：下载 -> 人声分离 -> 语音识别 -> 翻译 -> 语音合成 -> 混音输出

## 快速命令

| 用途 | 命令 |
|------|------|
| 启动后端 | `.\.venv\Scripts\uvicorn.exe backend.app.main:app --reload --host 0.0.0.0 --port 8000` |
| 启动前端 | `npm.cmd --prefix apps/web run dev -- --hostname 0.0.0.0 --port 3000` |
| 运行后端测试 | `.\.venv\Scripts\python.exe -m pytest backend\tests` |
| Lint 前端 | `npm.cmd --prefix apps/web run lint` |
| 构建前端 | `npm.cmd --prefix apps/web run build` |
| CLI 跑 pipeline | `.\.venv\Scripts\python.exe scripts\run_pipeline.py <url>` |

PowerShell 可能禁止执行 `npm.ps1`，优先使用 `npm.cmd`。

## 架构概览

```text
Next.js 16 前端 (apps/web, :3000)
    | /api/* rewrite
    v
FastAPI 后端 (backend/app, :8000)
    | SQLite + 单线程 FIFO Worker
    v
yt-dlp / demucs / whisper / funasr / funasr-vllm / openai / voxcpm / ffmpeg
```

- **前端 -> 后端**：`next.config.ts` 中 rewrite `/api/:path*` 到后端地址。默认代理到 `127.0.0.1:8000`。可通过 `NEXT_PUBLIC_API_BASE_URL`（客户端+服务端）或 `NEXT_SERVER_API_BASE_URL`（仅服务端）覆盖。
- **Worker**：单线程串行处理任务，不支持并行。启动时会将 `queued/running` 任务标记为 `failed`（后端重启前未完成）。
- **数据库**：SQLite（`data/youdub.sqlite`），3 张表：`tasks`、`task_stages`、`settings`。数据库 schema 会自动迁移新增的列。
- **上传**：支持本地视频上传（`.mp4/.mov/.m4v/.mkv/.webm/.avi/.flv/.wmv`），默认最大 4GB，上传到 `WORKFOLDER/_uploads/`，自动转码为 h.264+aac MP4。

## 处理 Pipeline（9 阶段）

```text
download -> separate -> asr -> asr_fix -> translate -> split_audio -> tts -> merge_audio -> merge_video
```

阶段定义在 `backend/app/stages.py`，执行逻辑在 `backend/app/pipeline.py`。
每个阶段成功后标记为 `succeeded`，重跑时自动跳过已成功的阶段。
长任务阶段（download、separate、asr、tts）有细进度反馈，最低每 2 秒更新一次进度。

### 阶段重跑与清理规则

| 操作 | API 端点 | 行为 |
|------|----------|------|
| **重跑任务** | `POST /rerun` | 重置所有阶段为 pending，重新排队 |
| **恢复任务** | `POST /resume` | 仅将 failed/running 阶段重置为 pending，重新排队 |
| **级联重跑** | `POST /rerun-stage/{stage}` | 重置目标阶段 + 所有下游阶段为 pending，重新排队 |
| **单阶段重跑** | `POST /rerun-single-stage/{stage}` | 仅重置目标阶段，同步直接执行（不走队列），完成后恢复上游产出 |
| **清理产出** | `POST /clear-stage/{stage}` | 删除该阶段的 output 文件，重置为 pending，不影响下游 |
| **更新配置** | `PATCH /config` | 修改 `asr_model`、`asr_language`、`target_language`、`add_subtitles`（仅非 running 任务） |

- 级联重跑会复用已成功阶段的上游产出。
- 单阶段重跑不会触发重跑后下游阶段，适合某个阶段单独重新执行。
- 清理产出不级联删除，不会影响下游阶段已有的 output。
- 已完成的阶段不会被后续重跑覆盖（除非被重置）。

## 关键目录与文件

### 后端 (`backend/app/`)

| 文件 | 职责 |
|------|------|
| `main.py` | FastAPI 入口，REST API 端点，CORS，任务创建/上传，阶段重跑/清理，配置更新 |
| `config.py` | `.env` 配置加载，路径常量，目录初始化 |
| `database.py` | SQLite CRUD，任务/阶段/设置表，schema 自动迁移 |
| `pipeline.py` | `PipelineRunner`，9 阶段顺序执行，阶段缓存复用，单阶段执行 |
| `worker.py` | 单线程 FIFO worker，后台线程执行任务 |
| `stages.py` | 阶段定义（`StageSpec` 数据类：name + label） |
| `sources.py` | 视频源识别、下载策略、任务语言方向归一化 |
| `devices.py` | 设备管理（auto/cpu/cuda/mps），组件级设备分配 |
| `adapters/` | 各阶段适配器（ytdlp、demucs、whisper、funasr、remote_funasr、openai、voxcpm、ffmpeg、audio、local_video） |

### 前端 (`apps/web/src/`)

| 文件 | 职责 |
|------|------|
| `app/page.tsx` | 首页：URL 任务创建 + 本地上传 + 历史列表，2 秒轮询 |
| `app/tasks/[id]/page.tsx` | 任务详情：阶段进度、日志、视频播放、阶段配置弹窗、重跑/清理按钮 |
| `lib/api.ts` | 后端 REST 接口 TypeScript 封装，含所有端点的请求函数 |
| `lib/i18n.tsx` | 国际化（en/zh），阶段说明和阶段 output 文案，`useI18n()` hook |
| `components/settings-dialog.tsx` | 设置弹窗：Cookie、OpenAI（含模型列表）、代理、FunASR vLLM、翻译并发 |
| `components/ui/` | shadcn/ui 组件库 |

### 服务目录

- `services/funasr-vllm/`：WSL2/Linux 下的 FunASR vLLM 远程服务脚本（start/stop/update/diagnose/smoke_test）
- `submodule/demucs/`：Demucs 人声分离源码，必须 `git submodule update --init --recursive`

## REST API 端点一览

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/tasks` | 创建 URL 任务（YouTube/Bilibili） |
| POST | `/api/tasks/upload` | 上传本地视频创建任务（multipart） |
| GET | `/api/tasks` | 列出任务（`?limit=N`，默认 100） |
| GET | `/api/tasks/current` | 获取最近任务 |
| GET | `/api/tasks/{id}` | 获取任务详情（含阶段列表） |
| PATCH | `/api/tasks/{id}/config` | 更新任务配置（asr_model/direction/subtitles） |
| DELETE | `/api/tasks/{id}` | 删除任务 + session 目录 + 日志 |
| POST | `/api/tasks/{id}/rerun` | 重跑整个任务 |
| POST | `/api/tasks/{id}/resume` | 从失败恢复 |
| POST | `/api/tasks/{id}/rerun-stage/{stage}` | 级联重跑指定阶段及下游 |
| POST | `/api/tasks/{id}/rerun-single-stage/{stage}` | 仅重跑单个阶段 |
| POST | `/api/tasks/{id}/clear-stage/{stage}` | 清理指定阶段产出 |
| GET | `/api/tasks/{id}/log` | 获取任务日志（纯文本） |
| GET | `/api/tasks/{id}/artifact/final-video` | 获取最终视频（`?download=1` 下载） |
| GET/POST | `/api/cookies/youtube` | YouTube Cookie 管理 |
| GET/POST | `/api/settings/openai` | OpenAI 设置（base_url/api_key/model/concurrency） |
| POST | `/api/settings/openai/models` | 列出可用的 OpenAI 模型 |
| GET/POST | `/api/settings/ytdlp` | yt-dlp 代理设置 |
| GET/POST | `/api/settings/funasr` | FunASR vLLM 引擎设置（use_vllm: auto/on/off） |
| GET | `/api/health` | 健康检查 |

## 语言方向规则

- ASR 语言和翻译目标语言来自任务配置中的翻译方向：`asr_language` / `target_language`。
- 如果只传了 `asr_language`，自动推断 `target_language` 为相反方向（`en`↔`zh`），反之亦然。
- YouTube/Bilibili 只决定下载策略、cookie、代理，不再决定语言方向。
- 普通 URL 任务如果缺少方向，统一默认 `en -> zh`。
- 本地上传任务使用上传 URL 中的 `direction`，并在创建任务时写入任务字段。
- 重复提交同一个视频时，后端会把本次提交的方向和 ASR 模型同步回旧任务，然后返回旧任务。
- 旧任务可能缺少 `asr_language/target_language`，会按默认 `en -> zh` 解析。如果实际方向不同，需在任务详情保存正确方向后重跑相关阶段。

## 字幕选项

- 任务创建时可设置 `add_subtitles`（布尔值，默认 `true`）。
- `merge_video` 阶段会根据此选项决定是否将翻译字幕嵌入到最终视频中。
- 可在任务详情通过"更新配置"按钮修改（仅 non-running 任务）。

## 本地上传

- 前端首页提供本地视频上传入口，支持拖拽和文件选择。
- 上传需选择翻译方向：`en -> zh` 或 `zh -> en`。
- 后端接受到 `WORKFOLDER/_uploads/{task_id}/` 目录，分块写入（1MB 块）。
- 上传完成后自动转码为 h.264 + aac MP4 格式。
- 上传限制：默认 4GB，可通过 `LOCAL_UPLOAD_MAX_BYTES` 环境变量调整。
- Next.js 代理请求体限制为 `2000m`，通过 `next.config.ts` 中 `proxyClientMaxBodySize` 设置。

## ASR 模型路径

前端提供 4 个 ASR 选项，路由到不同的执行路径：

| UI 选项 | `asr_model` | 执行路径 |
|--------|-------------|----------|
| Whisper large-v3-turbo | `null` / 空 | Windows 主 `.venv` 中的 `openai-whisper`，默认 `WHISPER_MODEL=large-v3-turbo` |
| Whisper large-v3 | `whisper:large-v3` | 仍走 Whisper 路径，模型名由 `WHISPER_MODEL` 或任务逻辑决定 |
| SenseVoiceSmall | `funasr:iic/SenseVoiceSmall` | Windows 主 `.venv` 中的标准 `funasr.AutoModel` |
| Fun-ASR-Nano | `funasr:FunAudioLLM/Fun-ASR-Nano-2512` | WSL2/Linux 远程 FunASR vLLM 服务 |

### ASR 路由逻辑

- `asr_model` 带 `funasr:*` 前缀 → FunASR 路径（本地或远程，取决于配置）
- `asr_model` 为空/null → Whisper `large-v3-turbo`
- `asr_model` 为 `whisper:large-v3` → Whisper（模型名由 `WHISPER_MODEL` env 决定）
- 可在任务详情"更新配置"修改 ASR 模型（仅 non-running 任务）

### Whisper

- 默认模型是 `large-v3-turbo`，由 `backend/app/adapters/whisper_asr.py` 加载。
- Whisper 识别时会传入任务的 `asr_language`；如果语言方向错，会强制按错误语言解码。
- 切换回默认 turbo 时，任务中的 `asr_model` 应被清空为 `NULL`，不能被后端当作"未传字段"忽略。
- 若遇到 corrupt 缓存的 SHA256 错误，适配器会自动清除缓存并重试。
- MPS 设备下 Whisper 会被强制回退到 `cpu`（因为 word timestamps 使用 float64 DTW，MPS 不支持）。

### SenseVoiceSmall（标准 FunASR）

- SenseVoiceSmall 走标准 FunASR 路径，不使用 vLLM。模型从 modelscope 自动下载。
- 原始返回会保存到 `metadata/asr.raw.funasr.json` 便于排查。
- 标准 FunASR 结果转换优先级：
  1. `sentence_info` / `sentences`
  2. 原始 `words + timestamp`
  3. `text + timestamp`
  4. 无可用时间戳时按文本 fallback
- `words + timestamp` 中标点可能是独立 token。断句应在 `. ? ! ; : , ， 、 。 ！ ？ ； ：` 等标点处发生，逗号也要断。
- 独立 token 会被拼回正常文本，例如 `I` + `'` + `m` -> `I'm`，`R` + `9` -> `R9`。
- SenseVoice 特有的 `<|...|>` 标签在结果处理时会被自动去除。

### Fun-ASR-Nano + vLLM（本地 vLLM 引擎）

- 如果 `FUNASR_USE_VLLM=auto` 且 vLLM 包已安装，LLM-based 模型（Fun-ASR-Nano、GLM-ASR-Nano、LLMASR）会走本地 vLLM 引擎。
- `FUNASR_USE_VLLM=on`：强制使用 vLLM，未安装则报错。
- `FUNASR_USE_VLLM=off`：强制不使用 vLLM，LLM 模型将报 NotImplementedError。
- 本地 vLLM 使用 CTC forced alignment 生成逐字时间戳。
- 此配置独立于远程 vLLM 服务——本地 vLLM 在 Windows 主 `.venv` 中，远程服务在 WSL2/Linux 中。

### Fun-ASR-Nano + 远程 vLLM 服务

- Windows 主 `.venv` 不安装 `vllm`。vLLM 及其 CUDA 依赖只放在 WSL2/Linux 服务环境。
- Nano 远程服务默认目录：`services/funasr-vllm/`
- WSL 运行时默认目录：`~/ai/funasr-service/.venv`
- 上游源码默认目录：`~/ai/upstream/FunASR`
- `.env` 中启用远程服务：

```env
FUNASR_REMOTE_BASE_URL=http://127.0.0.1:8899
FUNASR_REMOTE_ENDPOINT=/asr
FUNASR_REMOTE_PATH_MODE=auto
FUNASR_REMOTE_TIMEOUT=900
FUNASR_REMOTE_SPK=false
FUNASR_REMOTE_AUTOSTART=true
FUNASR_REMOTE_AUTOSTOP=true
FUNASR_REMOTE_START_TIMEOUT=300
FUNASR_REMOTE_MODEL=fun-asr-nano
FUNASR_REMOTE_HOTWORDS=
```

- `FUNASR_REMOTE_PATH_MODE=auto` 会把 Windows 路径（如 `G:\...`）转换成 WSL 路径（如 `/mnt/g/...`），同机服务直接读文件，避免上传音频。`off` 则使用 multipart 上传。
- 远程服务支持两种端点：`/asr`（FunASR serve_vllm.py 协议）或 `/v1/audio/transcriptions`（OpenAI 兼容）。
- 只有 LLM-based FunASR 模型（Fun-ASR-Nano、GLM-ASR-Nano、LLMASR）会走远程 vLLM 服务。
- 如果远程服务未启动且 `FUNASR_REMOTE_AUTOSTART=true`，后端会通过 WSL 执行 `bash ./start.sh`，等待服务响应后请求 `/asr`。
- 如果服务是本次 ASR 自动启动的，且 `FUNASR_REMOTE_AUTOSTOP=true`，ASR 完成后会执行 `bash ./stop.sh` 释放显存。
- 手动启动的服务不会被自动关闭。
- 自动启动日志：`services/funasr-vllm/autostart.log`
- 远程服务的原始返回保存到 `metadata/asr.raw.remote_funasr.json`，请求信息保存到 `metadata/asr.remote_request.json`。
- 远程服务安全性：通过 `FUNASR_ALLOWED_AUDIO_ROOTS` 限制 `/asr` 端点可读取的路径。

### FunASR vLLM 设置端点

- 前端设置弹窗的 FunASR 部分管理 `use_vllm` 设置（前端映射为 vLLM 开关）。
- 此设置控制本地 vLLM 引擎行为，不影响远程 vLLM 服务（远程由 `FUNASR_REMOTE_*` env 控制）。
- 保存时验证值必须在 `auto`/`on`/`off` 之中。

## 设备管理

`backend/app/devices.py` 提供组件级别的设备分配：

| 组件 | 环境变量 | 说明 |
|------|----------|------|
| `demucs` | `DEMUCS_DEVICE` -> `DEVICE` | 人声分离 |
| `whisper` | `WHISPER_DEVICE` -> `DEVICE` | 语音识别 |
| `voxcpm` | 不使用组件变量 | VoxCPM 内部自动选择设备 |

- `DEVICE` 是回退默认值。每个组件都可以单独指定设备。
- `auto` 模式：尝试 cuda → mps → cpu。
- **MPS 特殊处理**：Whisper 在 MPS 下会被强制回退到 `cpu`，因为 word timestamps 使用 float64 DTW（MPS 不支持）。
- CUDA 设备可加索引：`cuda:0`、`cuda`（等价于 `cuda:0`）。
- `validate_runtime_device()` 在任务创建前验证所有受管组件的设备可用性。

## 阶段产出清理规则

`POST /api/tasks/{task_id}/clear-stage/{stage_name}` 只删除该阶段说明中列出的 output，重置该阶段为 pending，不做级联删除。

| 阶段 | 清理内容 |
|------|----------|
| `download` | `media/video_source.mp4`，`metadata/ytdlp_info.json` 或 `metadata/local_info.json` |
| `separate` | `media/audio_vocals.wav`，`media/audio_bgm.wav` |
| `asr` | `metadata/asr.json`、`metadata/asr.raw.funasr.json` 或 `metadata/asr.raw.remote_funasr.json` |
| `asr_fix` | `metadata/asr_fixed.json` |
| `translate` | `metadata/translation.{lang}.json`，`metadata/subtitles.{lang}.srt` |
| `split_audio` | `segments/vocals/*.wav` |
| `tts` | `segments/tts/*.wav` |
| `merge_audio` | `tmp/audio_dubbing.wav`，`metadata/timings.json` |
| `merge_video` | `media/video_final.mp4` |

- 如果需要从某阶段开始重新生成整条下游链路，使用级联重跑（`rerun-stage`）而不是清理产出。
- 清理产出 + 单独重跑单阶段：先清理，再 `rerun-single-stage` 仅重新执行该阶段。

## 环境配置

`.env` 从 `env.txt.example` 复制。所有变量：

### 核心路径与设备

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `WORKFOLDER` | `./workfolder` | 工作目录（下载、中间产物、最终视频、上传目录） |
| `MODEL_CACHE_DIR` | `./data/modelscope` | Modelscope 模型缓存目录 |
| `DEVICE` | `cuda` | 全局设备：`cuda` / `cpu` / `mps` / `auto` |
| `DEMUCS_DEVICE` | (空，回退到 DEVICE) | Demucs 专用设备 |
| `WHISPER_DEVICE` | (空，回退到 DEVICE) | Whisper 专用设备 |
| `FFMPEG_PATH` | `ffmpeg` | ffmpeg 可执行路径 |
| `FFPROBE_PATH` | `ffprobe` | ffprobe 可执行路径 |

### OpenAI / 翻译

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | OpenAI API 地址（也接受 `OPENAI_API_BASE`） |
| `OPENAI_API_KEY` | (空) | API 密钥 |
| `OPENAI_MODEL` | `gpt-4o-mini` | 翻译模型（也接受 `OPENAI_MODEL_NAME`） |
| `OPENAI_TRANSLATE_CONCURRENCY` | `50` | 翻译并发请求数 |

### 上传与 CORS

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `LOCAL_UPLOAD_MAX_BYTES` | `4294967296` (4GB) | 本地上传最大字节数 |
| `CORS_ALLOW_ORIGINS` | (空) | 额外的 CORS 允许源（逗号分隔） |
| `CORS_ALLOW_ORIGIN_REGEX` | (匹配局域网 3000 端口) | CORS 源正则匹配 |

### 代理

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `YTDLP_PROXY_PORT` | (空) | YouTube 代理端口（SOCKS5 支持通过 `httpx[socks]`） |
| `HTTP_PROXY` | (空) | 系统 HTTP 代理 |
| `NO_PROXY` | `localhost,127.0.0.1,::1` | 不走代理的地址 |

### FunASR

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `FUNASR_MODEL` | `iic/SenseVoiceSmall` | 默认 FunASR 模型 |
| `FUNASR_VAD_MODEL` | `fsmn-vad` | VAD 模型 |
| `FUNASR_USE_VLLM` | `auto` | 本地 vLLM 引擎：`auto`/`on`/`off` |

### FunASR 远程 vLLM 服务（全部可选）

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `FUNASR_REMOTE_BASE_URL` | (空) | 远程服务地址 |
| `FUNASR_REMOTE_ENDPOINT` | `/asr` | 远程端点 |
| `FUNASR_REMOTE_PATH_MODE` | `auto` | 路径模式：`auto`/`wsl`/`off` |
| `FUNASR_REMOTE_TIMEOUT` | `900` | HTTP 超时（秒） |
| `FUNASR_REMOTE_SPK` | `false` | 说话人分离 |
| `FUNASR_REMOTE_MODEL` | (空) | 模型别名 |
| `FUNASR_REMOTE_HOTWORDS` | (空) | 热词 |
| `FUNASR_REMOTE_AUTOSTART` | `false` | 自动启动 WSL 服务 |
| `FUNASR_REMOTE_AUTOSTOP` | `true` | 自动停止自动启动的服务 |
| `FUNASR_REMOTE_START_TIMEOUT` | `300` | 等待服务启动超时（秒） |
| `FUNASR_REMOTE_START_COMMAND` | (空) | 自定义启动命令 |
| `FUNASR_REMOTE_STOP_COMMAND` | (空) | 自定义停止命令 |
| `FUNASR_REMOTE_AUDIO_PATH` | (空) | 覆盖远程音频路径 |

### VoxCPM

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `VOXCPM_MODEL` | `OpenBMB/VoxCPM2` | VoxCPM 模型名 |
| `VOXCPM_MODEL_DIR` | (空) | 本地模型目录（可选） |

## Python 依赖边界

- Python 版本：3.12
- Windows 主 `.venv` 使用 `requirements-pytorch-cu128.txt` + `requirements.txt`
- Windows 主 `.venv` 只运行 YouDub 后端、Whisper、SenseVoiceSmall、VoxCPM、Demucs 等本地能力。
- 不要在 Windows 主 `.venv` 里安装 vLLM；之前混装 vLLM 会带来 `compressed-tensors`、`mistral-common`、`opentelemetry-*` 等残留依赖，并可能破坏 `numba/llvmlite`、`transformers`、`numpy`、`protobuf` 组合。
- `requirements.txt` 约束 `transformers>=4.36.2,<5`，保持 FunASR/VoxCPM 在 4.x Transformers API 线上。
- 若 Whisper 报 `Numba requires at least version ... of llvmlite`，先检查：

```powershell
.\.venv\Scripts\python.exe -m pip show numba llvmlite
.\.venv\Scripts\python.exe -m pip check
```

## 开发约定

- 前端框架：Next.js 16 + React 19 + Tailwind v4 + shadcn/ui
- 后端框架：FastAPI + uvicorn
- 测试：pytest，`backend/tests/`，fixture 默认 `DEVICE=cpu`
- 包管理：Python 用 pip；前端用 npm
- 路径分隔符：Windows 用 `.venv\Scripts\`，不要用 `.venv/bin/`
- CORS：默认允许 localhost:3000 和局域网 IP
- 修改后优先跑：

```powershell
.\.venv\Scripts\python.exe -m pytest backend\tests
npm.cmd --prefix apps/web run lint
```

## 常见陷阱

1. **Git Submodule**：必须 `git submodule update --init --recursive`，ZIP 下载不含子模块。
2. **CUDA PyTorch**：先装 `requirements-pytorch-cu128.txt`，再装 `requirements.txt`。
3. **ffmpeg/ffprobe**：系统必须安装，可通过 `FFMPEG_PATH`/`FFPROBE_PATH` 指定。
4. **FFmpeg Shared DLL**：`torchaudio 2.11+` 使用 `torchcodec` 后端，需要 FFmpeg shared DLL。如果装的是 `essentials_build`（静态编译），需要额外下载 `full_build-shared` 版本的 `bin/*.dll` 复制到 `.venv\Lib\site-packages\torchcodec\`。
5. **单线程 Worker**：任务串行执行，不支持并行。
6. **首次运行**：Whisper、VoxCPM、Demucs 模型首次运行会自动下载，需要网络和磁盘空间。
7. **Session 目录**：`WORKFOLDER/{source}/{title}__{task_id}/` 下有 `media/`、`metadata/`、`segments/`、`tmp/`。
8. **代理**：YouTube 需要代理（SOCKS5 通过 `httpx[socks]` 支持）；Bilibili 自动抓取匿名 cookie。
9. **Next.js 上传限制**：代理请求体限制已设置为 `2000m`（`next.config.ts` 中 `proxyClientMaxBodySize`）。如需更大上传，需调整此值。
10. **数据库设置同步**：`.env` 改动需重启后端才生效（`init_db()` 用 `ON CONFLICT DO UPDATE` 同步 settings 默认值）。启动时也会自动从 metadata 回填缺失的任务标题。
11. **旧任务配置**：旧任务可能缺少 `asr_language/target_language`。当前代码会默认按 `en -> zh` 解析；如果实际方向不同，需要在任务详情里保存正确方向后重跑相关阶段。
12. **阶段清理不是级联重跑**：清理产出只删当前阶段 output；需要下游重算时使用级联重跑（`rerun-stage`）。
13. **单阶段重跑不触发下游**：`rerun-single-stage` 只执行目标阶段，之后不会自动跑下游阶段。需要完整链路的用 `rerun-stage`。
14. **MPS + Whisper**：macOS MPS 设备下 Whisper 会自动回退到 CPU，因为 DTW float64 运算不被 MPS 支持。
15. **上传文件格式**：仅支持 `.mp4/.mov/.m4v/.mkv/.webm/.avi/.flv/.wmv`，上传后自动转码为 h.264+aac MP4。
16. **API Key 安全**：保存 OpenAI 设置时，前端不会回显 API Key（显示为 `********`）。设置 `clear_api_key=true` 可清空已保存的 Key。
