# YouDub WebUI — 配置指南

> 所有配置途径、可调参数及说明

## 配置途径总览

| 途径 | 修改方式 | 生效时机 | 适用场景 |
|------|----------|----------|----------|
| **前端设置弹窗** | 浏览器 UI | 即时生效 | 日常使用（翻译 API、代理、Cookie） |
| **`.env` 文件** | 文本编辑器 | 重启后端 | 服务端配置（设备、路径、模型） |
| **`.env` 高级参数** | 文本编辑器 | 重启后端 | 模型调优（TTS、ASR 参数） |
| **源码硬编码** | 改代码 | 重启后端 | 深度定制（Demucs 模型、编码器、变速） |

---

## 一、前端设置弹窗

点右上角 **"设置"** 按钮，修改后实时保存到 SQLite 数据库。

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| 界面语言 | UI 语言 | `zh` |
| YouTube Cookie | Netscape 格式 Cookie 内容 | 空（也可放文件，见下方） |
| yt-dlp 代理端口 | YouTube 下载代理 | 空 |
| OpenAI Base URL | 翻译 API 地址 | `https://api.openai.com/v1` |
| OpenAI API Key | 翻译 API 密钥 | 空 |
| 模型 | 翻译模型名 | `gpt-4o-mini` |
| 翻译并发数 | 翻译阶段并行请求数 | `50` |

---

## 二、`.env` 文件

从 `env.txt.example` 复制。改后需**重启后端**生效（`uvicorn --reload` 只监控 Python 源码变化）。

### 路径与目录

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `WORKFOLDER` | `./workfolder` | 工作目录（下载视频、中间产物、最终视频） |
| `MODEL_CACHE_DIR` | `./data/modelscope` | AI 模型缓存目录（VoxCPM 等） |

### 设备配置

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `DEVICE` | `cuda` | 全局设备：`cuda` / `cpu` / `mps` / `auto` |
| `DEMUCS_DEVICE` | (继承 DEVICE) | Demucs 人声分离单独指定设备 |
| `WHISPER_DEVICE` | (继承 DEVICE) | Whisper 语音识别单独指定设备 |

> `auto` 按 CUDA → MPS → CPU 顺序选择。Whisper 选 MPS 时会自动退回 CPU（DTW 需要 float64）。

### 翻译 API（与前端设置同步）

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | OpenAI 兼容 API 地址 |
| `OPENAI_API_KEY` | 空 | API 密钥 |
| `OPENAI_MODEL` | `gpt-4o-mini` | 翻译模型名 |
| `OPENAI_TRANSLATE_CONCURRENCY` | `50` | 翻译并发请求数 |

### 下载代理（与前端设置同步）

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `YTDLP_PROXY_PORT` | 空 | yt-dlp 本机代理端口（如 `7897`） |
| `HTTP_PROXY` | 空 | HTTP 代理地址 |
| `NO_PROXY` | `localhost,127.0.0.1,::1` | 代理绕过列表 |

### 工具路径

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `FFMPEG_PATH` | (从 PATH 查找) | ffmpeg 可执行文件路径 |
| `FFPROBE_PATH` | (从 PATH 查找) | ffprobe 可执行文件路径 |

### 上传限制

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `LOCAL_UPLOAD_MAX_BYTES` | `4294967296` (4GB) | 本地视频上传大小限制 |

### CORS

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `CORS_ALLOW_ORIGINS` | 空 | 自定义允许的前端来源（逗号分隔） |
| `CORS_ALLOW_ORIGIN_REGEX` | 空 | 正则匹配允许的来源 |

---

## 三、`.env` 高级参数（模型调优）

这些不在 UI 中暴露，改 `.env` 后重启后端生效。

### Whisper 语音识别

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `WHISPER_MODEL` | `large-v3-turbo` | 模型大小：`tiny` / `base` / `small` / `medium` / `large-v3` / `large-v3-turbo` |
| `WHISPER_DOWNLOAD_ROOT` | (默认缓存 `~/.cache/whisper`) | 模型文件下载/缓存目录 |

> 模型越大越准但越慢。`large-v3-turbo` 是精度和速度的平衡点。

### VoxCPM 语音合成

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `VOXCPM_MODEL` | `OpenBMB/VoxCPM2` | ModelScope 模型 ID |
| `VOXCPM_MODEL_DIR` | (自动下载) | 本地模型目录（如已下载可指定路径跳过下载） |
| `VOXCPM_CFG_VALUE` | `2.0` | Classifier-Free Guidance 强度。越大越忠实原文发音，但多样性降低 |
| `VOXCPM_INFERENCE_TIMESTEPS` | `10` | 扩散去噪步数。越大音质越好，但速度越慢 |
| `VOXCPM_MIN_REFERENCE_MS` | `1200` | 参考音频最短时长（毫秒）。太短会影响音色克隆质量 |
| `VOXCPM_LOAD_DENOISER` | `false` | 是否加载降噪器（额外占用显存） |

### FunASR（备用 ASR）

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `FUNASR_MODEL` | `iic/SenseVoiceSmall` | FunASR 模型名 |
| `FUNASR_VAD_MODEL` | `fsmn-vad` | 语音活动检测模型 |

---

## 四、源码硬编码参数（需改代码）

修改后需重启后端。

### Demucs 人声分离 (`backend/app/adapters/demucs.py`)

| 参数 | 当前值 | 说明 |
|------|--------|------|
| 模型 | `htdemucs_ft` | Hybrid Transformer Demucs 微调版。可选 `htdemucs` / `htdemucs_ft` / `mdx_extra` |
| `shifts` | `3` | TTA 增强次数。改 `1` 可提速约 3x，SDR 降低约 0.1 |

### 翻译 (`backend/app/adapters/openai_translate.py`)

| 参数 | 当前值 | 说明 |
|------|--------|------|
| `TRANSLATE_RETRY` | `2` | 单句翻译失败重试次数 |
| `PREPROCESS_RETRY` | `2` | 预处理（热词提取）失败重试次数 |
| `temperature` | `0.2` | LLM 采样温度。越低越稳定，越高越多样 |
| `response_format` | `{"type": "json_object"}` | 强制 API 返回 JSON |

### 音频处理 (`backend/app/adapters/audio.py`)

| 参数 | 当前值 | 说明 |
|------|--------|------|
| 变速范围 | `[0.8, 1.2]` | TTS 变速对齐的全局范围 |
| 局部变速范围 | `[0.9, 1.1]` | 单个 segment 的微调范围 |
| 起始扩展 | `80ms` | 音频分段起始边界扩展 |
| 结束扩展 | `160ms` | 音频分段结束边界扩展 |

### 视频合成 (`backend/app/adapters/ffmpeg.py`)

| 参数 | 当前值 | 说明 |
|------|--------|------|
| 视频编码 | `libx264 -preset fast -crf 23` | H.264 软编码。可改 `h264_nvenc` 用 GPU 编码 |
| CRF | `23` | 质量因子。越小质量越好（18-28 常用范围） |
| BGM 音量 | `0.30` | 背景音乐混合音量（1.0 为原音量） |
| 配音音量 | `1.0` | 配音混合音量 |

### 句子切分 (`backend/app/adapters/asr_sentence_fixer.py`)

| 参数 | 当前值 | 说明 |
|------|--------|------|
| 起始 padding | `100ms` | 句子起始时间向前扩展 |
| 结束 padding | `300ms` | 句子结束时间向后扩展 |
| 最小间隔 | `50ms` | 相邻句子最小时间间隔 |

### 字幕 (`backend/app/adapters/ffmpeg.py`)

| 参数 | 当前值 | 说明 |
|------|--------|------|
| 字幕样式 | `FontSize=18,PrimaryColour=&H00FFFFFF` | ASS 字幕字体大小和颜色 |
| 字幕位置 | `Alignment=2,MarginV=28` | 底部居中，距底部 28px |

---

## 五、Cookies 文件

| 文件 | 路径 | 说明 |
|------|------|------|
| YouTube Cookie | `data/cookies/youtube.txt` | Netscape 格式，从浏览器导出 |
| Bilibili Cookie | `data/cookies/bilibili.txt` | 自动生成（匿名 cookie） |

**导出 YouTube Cookie**：
1. Edge 安装扩展 [Get cookies.txt LOCALLY](https://microsoftedge.microsoft.com/addons/detail/get-cookiestxt-locally/cclelndahbckbenkjhflpdbgdldlbecc)
2. 打开 YouTube 并登录
3. 点击扩展 → Export → 保存为 `youtube.txt` → 放到 `data/cookies/`

> Cookies 会被 YouTube 定期轮换，过期后需重新导出。

---

## 六、前端配置 (`apps/web/next.config.ts`)

| 配置项 | 当前值 | 说明 |
|--------|--------|------|
| `experimental.proxyClientMaxBodySize` | `2000mb` | 代理请求体大小限制（上传视频用） |
| `allowedDevOrigins` | `["172.27.2.90", "100.94.222.54"]` | 开发模式允许的局域网 IP |
| `rewrites` | `/api/*` → `http://127.0.0.1:8000/api/*` | 前端到后端的 API 代理 |
