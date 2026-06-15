# YouDub WebUI — Agent 指南

> 视频本地化工具：下载 → 人声分离 → 语音识别 → 翻译 → 语音合成 → 混音输出

## 快速命令

| 用途 | 命令 |
|------|------|
| 启动后端 | `.\.venv\Scripts\uvicorn.exe backend.app.main:app --reload --host 0.0.0.0 --port 8000` |
| 启动前端 | `npm --prefix apps/web run dev -- --hostname 0.0.0.0 --port 3000` |
| 运行后端测试 | `pytest backend/tests` |
| Lint 前端 | `npm --prefix apps/web run lint` |
| 构建前端 | `npm --prefix apps/web run build` |
| CLI 跑 pipeline | `python scripts/run_pipeline.py <url>` |

## 架构概览

```
Next.js 16 前端 (apps/web, :3000)
    │ /api/* rewrite
    ▼
FastAPI 后端 (backend/app, :8000)
    │ SQLite + 单线程 FIFO Worker
    ▼
yt-dlp / demucs / whisper / openai / voxcpm / ffmpeg
```

- **前端 → 后端**：`next.config.ts` 中 rewrite `/api/:path*` 到 `http://127.0.0.1:8000/api/:path*`
- **Worker**：单线程串行处理任务，不支持并行
- **数据库**：SQLite（`data/youdub.sqlite`），3 张表：`tasks`、`task_stages`、`settings`

## 处理 Pipeline（9 阶段）

```
download → separate → asr → asr_fix → translate → split_audio → tts → merge_audio → merge_video
```

阶段定义在 `backend/app/stages.py`，执行逻辑在 `backend/app/pipeline.py`。
每个阶段成功后标记为 `succeeded`，重跑时自动跳过已成功的阶段。

## 关键目录与文件

### 后端 (`backend/app/`)

| 文件 | 职责 |
|------|------|
| `main.py` | FastAPI 入口，所有 REST API 端点，CORS，lifespan |
| `config.py` | `.env` 配置加载，路径常量 |
| `database.py` | SQLite CRUD，任务/阶段/设置表 |
| `pipeline.py` | `PipelineRunner`，9 阶段顺序执行，阶段缓存复用 |
| `worker.py` | 单线程 FIFO worker，后台线程执行任务 |
| `stages.py` | 阶段定义（`StageSpec` 数据类） |
| `sources.py` | 视频源识别（YouTube/Bilibili/本地） |
| `devices.py` | 设备管理（auto/cpu/cuda/mps） |
| `adapters/` | 各阶段适配器（ytdlp、demucs、whisper、openai、voxcpm、ffmpeg、audio） |

### 前端 (`apps/web/src/`)

| 文件 | 职责 |
|------|------|
| `app/page.tsx` | 首页：任务创建 + 历史列表，2 秒轮询 |
| `app/tasks/[id]/page.tsx` | 任务详情：阶段进度、日志、视频播放 |
| `lib/api.ts` | 后端 REST 接口 TypeScript 封装 |
| `lib/i18n.tsx` | 国际化（en/zh），Context + localStorage |
| `components/settings-dialog.tsx` | 设置弹窗：Cookie、OpenAI、代理、语言 |
| `components/ui/` | shadcn/ui 组件库 |

### 子模块

- `submodule/demucs/`：Demucs 人声分离源码，必须 `git submodule update --init --recursive`

## 环境配置

`.env` 从 `env.txt.example` 复制。关键变量：

| 变量 | 说明 |
|------|------|
| `DEVICE` | 全局设备：`cuda` / `cpu` / `mps` / `auto` |
| `OPENAI_BASE_URL` / `OPENAI_API_KEY` / `OPENAI_MODEL` | 翻译 API |
| `WORKFOLDER` | 工作目录（下载、中间产物、最终视频） |
| `MODEL_CACHE_DIR` | 模型缓存目录 |
| `YTDLP_PROXY_PORT` | YouTube 代理端口 |

## 开发约定

- **Python 版本**：3.12
- **前端框架**：Next.js 16 + React 19 + Tailwind v4 + shadcn/ui
- **后端框架**：FastAPI + uvicorn
- **测试**：pytest，`backend/tests/`，fixture 强制 `DEVICE=cpu`
- **包管理**：Python 用 pip + 阿里云镜像；前端用 npm
- **路径分隔符**：Windows 用 `.venv\Scripts\`，不要用 `.venv/bin/`
- **CORS**：默认允许 localhost:3000 和局域网 IP

## 常见陷阱

1. **Git Submodule**：必须 `git submodule update --init --recursive`，ZIP 下载不含子模块
2. **CUDA PyTorch**：必须先装 `requirements-pytorch-cu128.txt`，再装 `requirements.txt`
3. **ffmpeg/ffprobe**：系统必须安装，可通过 `FFMPEG_PATH`/`FFPROBE_PATH` 指定
4. **FFmpeg Shared DLL**：`torchaudio 2.11+` 使用 `torchcodec` 后端，需要 FFmpeg shared DLL。如果装的是 `essentials_build`（静态编译），需要额外下载 `full_build-shared` 版本的 `bin/*.dll` 复制到 `.venv\Lib\site-packages\torchcodec\`
5. **单线程 Worker**：任务串行执行，不支持并行
6. **首次运行**：Whisper、VoxCPM、Demucs 模型首次运行自动下载，需要网络和磁盘空间
7. **Session 目录**：`WORKFOLDER/{source}/{title}__{task_id}/` 下有 media/、metadata/、segments/ 子目录
8. **代理**：YouTube 需要代理，Bilibili 自动抓取匿名 cookie
9. **Next.js 上传限制**：默认代理请求体 10MB，需在 `next.config.ts` 设置 `experimental.proxyClientMaxBodySize`
10. **数据库设置同步**：`.env` 改动需重启后端才生效（`init_db()` 用 `ON CONFLICT DO UPDATE`）
11. **Fun-ASR-Nano + VAD**：必须安装 `vllm` 并设置 `FUNASR_USE_VLLM=auto`（或在 settings 弹窗选 `Auto`/`On`）。标准 `funasr.AutoModel` 路径在 [`funasr/models/fun_asr_nano/model.py:559`](.venv/Lib/site-packages/funasr/models/fun_asr_nano/model.py) 会因 `NotImplementedError: batch decoding is not implemented` 崩溃，vLLM 引擎通过 PagedAttention + Continuous Batching 内部消化 batch，是官方唯一支持的路径（参考 [vllm_guide.md](https://github.com/modelscope/FunASR/blob/main/docs/vllm_guide.md)）。
