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
yt-dlp / demucs / whisper / funasr / openai / voxcpm / ffmpeg
```

- **前端 -> 后端**：`next.config.ts` 中 rewrite `/api/:path*` 到 `http://127.0.0.1:8000/api/:path*`
- **Worker**：单线程串行处理任务，不支持并行
- **数据库**：SQLite（`data/youdub.sqlite`），3 张表：`tasks`、`task_stages`、`settings`

## 处理 Pipeline（9 阶段）

```text
download -> separate -> asr -> asr_fix -> translate -> split_audio -> tts -> merge_audio -> merge_video
```

阶段定义在 `backend/app/stages.py`，执行逻辑在 `backend/app/pipeline.py`。
每个阶段成功后标记为 `succeeded`，重跑时自动跳过已成功的阶段。

## 关键目录与文件

### 后端 (`backend/app/`)

| 文件 | 职责 |
|------|------|
| `main.py` | FastAPI 入口，REST API 端点，CORS，任务配置，阶段产出清理 |
| `config.py` | `.env` 配置加载，路径常量 |
| `database.py` | SQLite CRUD，任务/阶段/设置表 |
| `pipeline.py` | `PipelineRunner`，9 阶段顺序执行，阶段缓存复用 |
| `worker.py` | 单线程 FIFO worker，后台线程执行任务 |
| `stages.py` | 阶段定义（`StageSpec` 数据类） |
| `sources.py` | 视频源识别、下载策略、任务语言方向归一化 |
| `devices.py` | 设备管理（auto/cpu/cuda/mps） |
| `adapters/` | 各阶段适配器（ytdlp、demucs、whisper、funasr、openai、voxcpm、ffmpeg、audio） |

### 前端 (`apps/web/src/`)

| 文件 | 职责 |
|------|------|
| `app/page.tsx` | 首页：任务创建 + 历史列表，2 秒轮询 |
| `app/tasks/[id]/page.tsx` | 任务详情：阶段进度、日志、视频播放、阶段配置 |
| `lib/api.ts` | 后端 REST 接口 TypeScript 封装 |
| `lib/i18n.tsx` | 国际化（en/zh），阶段说明和阶段 output 文案 |
| `components/settings-dialog.tsx` | 设置弹窗：Cookie、OpenAI、代理、语言、FunASR vLLM |
| `components/ui/` | shadcn/ui 组件库 |

### 服务目录

- `services/funasr-vllm/`：WSL2/Linux 下的 FunASR Nano + vLLM 远程服务脚本
- `submodule/demucs/`：Demucs 人声分离源码，必须 `git submodule update --init --recursive`

## 语言方向规则

- ASR 语言和翻译目标语言必须来自任务配置中的翻译方向：`asr_language` / `target_language`。
- YouTube/Bilibili 只决定下载策略、cookie、代理，不再决定语言方向。
- 普通 URL 任务如果缺少方向，统一默认 `en -> zh`。
- 本地上传任务使用上传 URL 中的 `direction`，并在创建任务时写入任务字段。
- 重复提交同一个视频时，后端会把本次提交的方向和 ASR 模型同步回旧任务，然后返回旧任务。

## ASR 模型路径

前端当前提供 4 个 ASR 选项：

| UI 选项 | `asr_model` | 执行路径 |
|--------|-------------|----------|
| Whisper large-v3-turbo | `null` / 空 | Windows 主 `.venv` 中的 `openai-whisper`，默认 `WHISPER_MODEL=large-v3-turbo` |
| Whisper large-v3 | `whisper:large-v3` | 仍走 Whisper 路径，模型名由 `WHISPER_MODEL` 或任务逻辑决定 |
| SenseVoiceSmall | `funasr:iic/SenseVoiceSmall` | Windows 主 `.venv` 中的标准 `funasr.AutoModel` |
| Fun-ASR-Nano | `funasr:FunAudioLLM/Fun-ASR-Nano-2512` | WSL2/Linux 远程 FunASR vLLM 服务 |

### Whisper

- 默认模型是 `large-v3-turbo`，由 `backend/app/adapters/whisper_asr.py` 加载。
- Whisper 识别时会传入任务的 `asr_language`；如果语言方向错，会强制按错误语言解码。
- 切换回默认 turbo 时，任务中的 `asr_model` 应被清空为 `NULL`，不能被后端当作“未传字段”忽略。

### SenseVoiceSmall

- SenseVoiceSmall 走标准 FunASR 路径，不使用 vLLM。
- 原始返回会保存到 `metadata/asr.raw.funasr.json` 便于排查。
- 标准 FunASR 结果转换优先级：
  1. `sentence_info` / `sentences`
  2. 原始 `words + timestamp`
  3. `text + timestamp`
  4. 无可用时间戳时按文本 fallback
- `words + timestamp` 中标点可能是独立 token。断句应在 `. ? ! ; : , ， 、 。 ！ ？ ； ：` 等标点处发生，逗号也要断。
- 独立 token 会被拼回正常文本，例如 `I` + `'` + `m` -> `I'm`，`R` + `9` -> `R9`。

### Fun-ASR-Nano + vLLM

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
```

- `FUNASR_REMOTE_PATH_MODE=auto` 会把 Windows 路径（如 `G:\...`）转换成 WSL 路径（如 `/mnt/g/...`），同机服务直接读文件，避免上传音频。
- 只有 LLM-based FunASR 模型（Fun-ASR-Nano、GLM-ASR-Nano、LLMASR）会走远程 vLLM 服务。
- 如果远程服务未启动且 `FUNASR_REMOTE_AUTOSTART=true`，后端会通过 WSL 执行 `bash ./start.sh`，等待服务响应后请求 `/asr`。
- 如果服务是本次 ASR 自动启动的，且 `FUNASR_REMOTE_AUTOSTOP=true`，ASR 完成后会执行 `bash ./stop.sh` 释放显存。
- 手动启动的服务不会被自动关闭。
- 自动启动日志：`services/funasr-vllm/autostart.log`

## 阶段产出清理规则

`/api/tasks/{task_id}/clear-stage/{stage_name}` 只删除该阶段说明中列出的 output，不做级联删除，避免超出用户预期。

| 阶段 | 清理内容 |
|------|----------|
| `download` | `media/video_source.mp4`，`metadata/ytdlp_info.json` 或 `metadata/local_info.json` |
| `separate` | `media/audio_vocals.wav`，`media/audio_bgm.wav` |
| `asr` | `metadata/asr.json` |
| `asr_fix` | `metadata/asr_fixed.json` |
| `translate` | `metadata/translation.{lang}.json`，`metadata/subtitles.{lang}.srt` |
| `split_audio` | `segments/vocals/*.wav` |
| `tts` | `segments/tts/*.wav` |
| `merge_audio` | `tmp/audio_dubbing.wav`，`metadata/timings.json` |
| `merge_video` | `media/video_final.mp4` |

如果需要从某阶段开始重新生成整条下游链路，使用“重跑阶段”而不是“清理产出”。

## 环境配置

`.env` 从 `env.txt.example` 复制。关键变量：

| 变量 | 说明 |
|------|------|
| `DEVICE` | 全局设备：`cuda` / `cpu` / `mps` / `auto` |
| `OPENAI_BASE_URL` / `OPENAI_API_KEY` / `OPENAI_MODEL` | 翻译 API |
| `OPENAI_TRANSLATE_CONCURRENCY` | 翻译并发数 |
| `WORKFOLDER` | 工作目录（下载、中间产物、最终视频） |
| `MODEL_CACHE_DIR` | 模型缓存目录 |
| `YTDLP_PROXY_PORT` | YouTube 代理端口 |
| `FUNASR_REMOTE_*` | 远程 Nano/vLLM 服务配置 |

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
8. **代理**：YouTube 需要代理；Bilibili 自动抓取匿名 cookie。
9. **Next.js 上传限制**：默认代理请求体 10MB，需在 `next.config.ts` 设置 `experimental.proxyClientMaxBodySize`。
10. **数据库设置同步**：`.env` 改动需重启后端才生效（`init_db()` 用 `ON CONFLICT DO UPDATE` 同步 settings 默认值）。
11. **旧任务配置**：旧任务可能缺少 `asr_language/target_language`。当前代码会默认按 `en -> zh` 解析；如果实际方向不同，需要在任务详情里保存正确方向后重跑相关阶段。
12. **阶段清理不是级联重跑**：清理产出只删当前阶段 output；需要下游重算时使用重跑阶段。
