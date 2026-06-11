# YouDub WebUI — Pipeline 技术分析报告

> 生成日期：2026-06-10

## 架构总览

```
download → separate → asr → asr_fix → translate → split_audio → tts → merge_audio → merge_video
```

## 各阶段技术详解

### Stage 1: download — 视频下载

| 属性 | 值 |
|------|-----|
| 技术栈 | `yt-dlp` + `requests` |
| 资源类型 | I/O 密集（网络 + 磁盘） |
| 设备需求 | 纯 CPU |

**操作流程：**
1. `yt_dlp.YoutubeDL.extract_info()` 解析视频元数据
2. Bilibili 源：自动抓取匿名 cookie
3. `yt_dlp.YoutubeDL.download()` 多线程分片下载
4. ffmpeg 合并分离的 video/audio 流为 MP4
5. 多 format 候选：`bestvideo[height<=1080]+bestaudio` → `best` 四级降级

**CPU 占用原因：** 网络 I/O 等待 + yt-dlp 的 Node.js 运行时解析签名算法。

---

### Stage 2: separate — Demucs 人声分离 ⚡ 最吃资源

| 属性 | 值 |
|------|-----|
| 技术栈 | `demucs`（Meta）— **htdemucs_ft** 模型 |
| 资源类型 | GPU 推理 + CPU 密集预处理 |
| 设备 | CUDA / CPU / MPS |

**模型架构（HTDemucs）：**
- 双分支混合架构：频率分支（spectrogram）+ 时间分支（waveform）
- 频率分支：STFT → 频率轴卷积编码器 → Transformer → 解码器 → iSTFT
- 时间分支：直接波形卷积编码器 → Transformer → 解码器
- 中间有 `CrossTransformerEncoder` 跨分支注意力（5 层，8 头）
- **Bag of Models**：包含 4 个子模型，加权平均输出

**推理流程：**
```
for sub_model in bag (4个模型):
  for shift in shifts (3次):
    for segment in segments:
      a. STFT（CPU）
      b. 模型前向（GPU）：编码 → Transformer → 解码
      c. iSTFT（CPU）
```

**🔑 CPU 打满的 5 大元凶：**

| # | 原因 | 详情 |
|---|------|------|
| 1 | **STFT/iSTFT** | `torch.stft()` 底层调用 kiss_fft C 库，不走 CUDA kernel。每次 segment 都要做，12 轮（4模型×3shift） |
| 2 | **ffmpeg 子进程解码** | `AudioFile.read()` 调 ffmpeg 子进程（`-threads 1`），`np.fromfile()` + `torch.from_numpy()` 在 CPU |
| 3 | **julius 重采样** | `julius.resample_frac()` 多相滤波器组，纯 CPU tensor 操作 |
| 4 | **CPU↔GPU 数据搬运** | 归一化、`view_as_real`、`permute` 反复在 CPU 和 GPU 间转移 tensor |
| 5 | **Python GIL** | Bag of Models 的循环在 Python 层面串行 |

**Shifts 参数影响：** `shifts=3` 是测试时增强（TTA），对整段音频跑 3 次（随机偏移后取平均），精度更高但计算量 ×3。

---

### Stage 3: asr — Whisper 语音识别

| 属性 | 值 |
|------|-----|
| 技术栈 | OpenAI Whisper |
| 模型 | `large-v3-turbo`（约 809M 参数） |
| 资源类型 | GPU 推理 + CPU 音频预处理 |
| 设备 | CUDA / CPU（MPS 时降级 CPU，因 DTW 需要 float64） |

**操作流程：**
1. `whisper.load_model()` 加载 Transformer 编码器-解码器
2. `model.transcribe()` 核心推理：
   - 音频预处理：STFT（CPU）→ 对数梅尔频谱图（CPU）
   - 编码器前向：Transformer encoder（GPU）
   - 解码器自回归：逐 token 生成（GPU）
   - `word_timestamps=True`：DTW 对齐（CPU，需 float64）

**CPU 占用原因：** STFT + mel 滤波器组在 CPU；DTW 对齐算法在 CPU。

---

### Stage 4: asr_fix — ASR 后处理

| 属性 | 值 |
|------|-----|
| 技术栈 | 纯 Python（json + 数学运算） |
| 资源类型 | CPU 轻量（几乎不消耗） |
| 设备 | 无需 GPU |

**操作流程：**
1. 读取 ASR JSON → 解析 utterances
2. 文本 strip + 过滤空 utterance
3. 对每个 utterance 的 start/end 时间做 padding（起始 100ms，结束 300ms）
4. 检测与前后 utterance 的间隙，避免重叠（min_gap=50ms）
5. 写入 `asr_fixed.json`

**CPU 占用：** 不吃。纯 JSON 解析和整数运算，<1ms 完成。

---

### Stage 5: translate — OpenAI API 翻译

| 属性 | 值 |
|------|-----|
| 技术栈 | `openai` Python SDK → OpenAI-compatible API |
| 资源类型 | I/O 密集（网络请求） |
| 设备 | 纯 CPU（发起 HTTP 请求） |

**操作流程：**
1. **预处理阶段**：构造 prompt 调用 LLM 生成 `{summary, hotwords, corrections}`
2. **翻译阶段**：`ThreadPoolExecutor` 并发翻译（默认 50 并发）
   - 每句调用 `client.chat.completions.create()` → JSON 模式
   - Pydantic 校验 + 最多重试 2 次
   - 后处理：中文翻译将 `——` 替换为 `，`

**CPU 占用原因：** 50 并发线程创建+管理有少量开销，主要瓶颈是 API 延迟。

---

### Stage 6: split_audio — 音频分段

| 属性 | 值 |
|------|-----|
| 技术栈 | `pydub.AudioSegment` |
| 资源类型 | CPU + 磁盘 I/O |
| 设备 | 纯 CPU |

**操作流程：**
1. `AudioSegment.from_file(vocals_file)` 加载完整人声音频（底层调用 ffmpeg）
2. 遍历 translation 的每个 segment：切片 + 导出 WAV
3. 起止时间各扩展 80ms/160ms 边界

**CPU 占用原因：** pydub 的 `from_file` 和 `export` 都调用 ffmpeg 子进程。

---

### Stage 7: tts — VoxCPM 语音合成

| 属性 | 值 |
|------|-----|
| 技术栈 | `VoxCPM`（OpenBMB/VoxCPM2） |
| 模型类型 | 扩散模型（Diffusion） |
| 资源类型 | GPU 推理 |
| 设备 | library-auto（VoxCPM 内部选择 cuda/mps/cpu） |

**操作流程：**
1. 从 ModelScope 下载 `OpenBMB/VoxCPM2` 模型
2. `VoxCPM.from_pretrained()` 加载扩散 TTS 模型
3. 对每个翻译句子：
   - 选择参考音频（当前 segment 对应的人声片段）
   - `model.generate()` 扩散推理：`cfg_value=2.0`，`inference_timesteps=10`
   - `sf.write()` 写入 WAV

**CPU 占用原因：** 参考音频加载在 CPU；扩散模型采样器有 CPU 侧调度逻辑。

---

### Stage 8: merge_audio — 音频合并 + 变速

| 属性 | 值 |
|------|-----|
| 技术栈 | `librosa` + `audiostretchy` + `numpy` + `soundfile` + `pydub` |
| 资源类型 | CPU 密集（信号处理） |
| 设备 | 纯 CPU |

**操作流程：**
1. 计算全局变速比：`desired_total / actual_total`，限制 [0.8, 1.2]
2. 逐段处理：
   - `librosa.load()` 读取 TTS WAV
   - `audiostretchy.stretch_audio()` 时间拉伸（WSOLA / Phase Vocoder）
   - NumPy 数组截断到目标时长
3. `np.concatenate()` 按时间轴拼接所有 segment + 静音填充
4. `sf.write()` 写入最终配音 WAV

**CPU 占用原因：**
- `librosa.load()` 被调用 2N+1 次（N 个 segment × 2 + 1），每次涉及音频解码
- `audiostretchy` 时间拉伸是纯 CPU 信号处理
- NumPy 数组拼接有显著内存分配开销

---

### Stage 9: merge_video — 视频合成

| 属性 | 值 |
|------|-----|
| 技术栈 | `ffmpeg`（外部子进程） |
| 资源类型 | CPU 密集（视频编码） |
| 设备 | 纯 CPU（libx264 软编码） |

**操作流程：**
1. 生成 SRT 字幕文件（按标点分割 + 按字符数权重分配时间）
2. **ffmpeg 音频混合**：配音 + BGM（音量 0.30）→ AAC
3. **ffmpeg 视频合成**：
   - 字幕烧入：`subtitles` 滤镜逐帧文字光栅化
   - 视频编码：`libx264 -preset fast -crf 23`（纯 CPU 软编码）
   - 音频编码：AAC

**CPU 占用原因：** libx264 软编码 + 字幕逐帧渲染。未使用硬件加速（NVENC/QSV）。

---

## 资源消耗总览

| 阶段 | GPU | CPU | 瓶颈类型 |
|------|-----|-----|----------|
| download | ✗ | ★★☆☆☆ | I/O（网络） |
| **separate** | ★★★ | ★★★★★ | **GPU 推理 + CPU 预处理** |
| **asr** | ★★★ | ★★★ | GPU 推理 + CPU STFT |
| asr_fix | ✗ | ☆☆☆☆☆ | 几乎无负载 |
| translate | ✗ | ★☆☆☆☆ | I/O（API 调用） |
| split_audio | ✗ | ★★☆☆☆ | CPU + 磁盘 I/O |
| **tts** | ★★★ | ★★☆☆☆ | GPU 扩散推理 |
| **merge_audio** | ✗ | ★★★☆☆ | CPU 信号处理 |
| **merge_video** | ✗ | ★★★★☆ | CPU 视频编码 |

## 回答：为什么 separate 阶段 CPU 和 GPU 都跑满

**GPU 跑满** 是正常的——htdemucs_ft 模型有 4 个子模型 × 3 次 shift = 12 轮完整推理，每轮包含 Transformer 自注意力和交叉注意力计算。

**CPU 也跑满** 的核心原因是 **STFT/iSTFT 不走 CUDA kernel**。`torch.stft()` 底层调用的是 kiss_fft C 库，即使输入 tensor 在 GPU 上，STFT 计算本身也在 CPU 完成。加上 12 轮推理、ffmpeg 子进程解码、julius 重采样、CPU↔GPU 数据搬运，CPU 成为与 GPU 并列的瓶颈。

## 潜在优化方向

| 优先级 | 优化项 | 预期效果 |
|--------|--------|----------|
| 高 | Demucs `shifts=3` → `shifts=1` | 分离阶段速度 ×3，SDR 降低约 0.1 |
| 高 | merge_video 启用 `-c:v h264_nvenc` | 视频编码从 CPU 卸载到 GPU |
| 中 | merge_audio 减少 `librosa.load()` 调用次数 | 减少重复音频解码 |
| 中 | split_audio 改用 `soundfile` 替代 pydub | 减少 ffmpeg 子进程调用 |
| 低 | Demucs STFT 改用 CUDA 实现 | 需修改 demucs 源码，收益有限 |
