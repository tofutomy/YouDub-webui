"use client"

import { createContext, ReactNode, useContext, useEffect, useMemo, useState } from "react"

export type UiLanguage = "en" | "zh"

const STORAGE_KEY = "youdub-ui-language"

export const LANGUAGE_OPTIONS: { value: UiLanguage; label: string }[] = [
  { value: "en", label: "English" },
  { value: "zh", label: "中文" },
]

type Messages = {
  common: {
    back: string
    cancel: string
    close: string
    loading: string
    sentenceEnd: string
    waiting: string
  }
  home: Record<string, string>
  task: Record<string, string>
  settings: Record<string, string>
  status: Record<string, string>
  stages: Record<string, string>
}

const messages: Record<UiLanguage, Messages> = {
  en: {
    common: {
      back: "Back",
      cancel: "Cancel",
      close: "Close",
      loading: "loading",
      sentenceEnd: ".",
      waiting: "Waiting",
    },
    home: {
      createTitle: "Create new task",
      youtubeLabel: "YouTube URL",
      bilibiliLabel: "Bilibili URL",
      localVideoLabel: "Local video file",
      localDirectionLabel: "Translation direction",
      localEnZh: "English -> Chinese",
      localZhEn: "Chinese -> English",
      addSubtitles: "Add subtitles",
      addSubtitlesHelp: "Embed subtitles into the final video",
      asrModelLabel: "ASR model",
      asrWhisperTurbo: "Whisper large-v3-turbo",
      asrWhisperLarge: "Whisper large-v3",
      asrSenseVoice: "SenseVoiceSmall",
      asrFunAsrNano: "Fun-ASR-Nano",
      asrQwen3Asr: "Qwen3-ASR",
      groupInput: "Input source",
      groupAsr: "ASR",
      groupTranslate: "Translate",
      translateModeLabel: "Translation mode",
      translateModeSentence: "Sentence (per-sentence, fast)",
      translateModeBatch: "Batch (context-aware, slower)",
      groupMergeVideo: "Merge video",
      submitting: "Submitting",
      createTask: "Create task",
      taskHistory: "Task history",
      empty: "No tasks yet. Submit a URL or upload a local video above to start.",
      loadError: "Failed to load tasks",
      createError: "Failed to create task",
    },
    task: {
      overview: "Task overview",
      title: "Title",
      taskId: "Task ID",
      created: "Created",
      started: "Started",
      completed: "Completed",
      session: "Session",
      loading: "Loading task...",
      finalVideo: "Final video",
      download: "Download",
      stages: "Stages",
      resumeHelp: "Resume from the failed stage. Already-succeeded stages will be reused from cache.",
      resuming: "Resuming",
      resumeTask: "Resume task",
      stopTask: "Stop task",
      stopping: "Stopping",
      stopTitle: "Stop this task?",
      stopDescription:
        "The task will be stopped at the next safe checkpoint (stage boundary or progress update). The current stage will run to completion, then the task will be marked as failed. Already-succeeded stages and their outputs are kept, so you can resume later.",
      confirmStop: "Confirm stop",
      stopError: "Failed to stop task",
      stoppedNotice: "This task was stopped manually.",
      rerunStage: "Rerun from here",
      rerunStageTitle: "Rerun from this stage?",
      rerunStageDescription:
        "This stage and all subsequent stages will be reset and re-run. Already-succeeded stages before this one will be reused from cache.",
      rerunningStage: "Rerunning",
      rerunSingleStage: "Rerun this stage",
      rerunSingleStageTitle: "Rerun this stage only?",
      rerunSingleStageDescription:
        "Only this stage will be reset and re-run. All other stages and their outputs will remain unchanged.",
      rerunningSingleStage: "Rerunning",
      stageConfig: "Config",
      stageConfigTitle: "Stage configuration",
      stageConfigSaved: "Configuration saved.",
      stageConfigError: "Failed to save configuration.",
      clearStage: "Clear output",
      clearStageTitle: "Clear stage output?",
      clearStageDescription: "This will delete the cached output files for this stage. The stage will need to be re-run to regenerate them.",
      clearStageDone: "Stage output cleared.",
      clearStageError: "Failed to clear stage output.",
      runLog: "Run log",
      emptyLog: "Logs will appear once the task starts.",
      dangerZone: "Danger zone",
      rerunHelp: "Wipe the session directory and run this URL again from scratch.",
      rerunTask: "Rerun task",
      rerunTitle: "Rerun this task?",
      rerunDescription:
        "Existing log, session directory and final video will be deleted, then the same URL is re-queued under the same task id.",
      rerunning: "Rerunning",
      confirmRerun: "Confirm rerun",
      deleteHelp:
        "Delete this task, its run log, and the entire session directory under",
      deleteTask: "Delete task",
      deleteTitle: "Delete this task?",
      deleteDescription:
        "This permanently removes the task record, its log file, and the entire session directory. This action cannot be undone.",
      deleting: "Deleting",
      confirmDelete: "Confirm delete",
      runningLocked: "Running tasks cannot be rerun or deleted. Wait until it finishes or fails.",
      loadError: "Failed to load task",
      deleteError: "Failed to delete task",
      rerunError: "Failed to rerun task",
      resumeError: "Failed to resume task",
      stageInfoTitle: "Stage info",
    },
    settings: {
      button: "Settings",
      title: "Runtime settings",
      description: "Stored locally by the FastAPI backend.",
      language: "Interface language",
      cookie: "YouTube cookie",
      savedCookie: "******** saved YouTube cookie ********",
      cookiePlaceholder: "Paste Netscape cookie content",
      proxyPort: "yt-dlp proxy port",
      baseUrl: "OpenAI base URL",
      apiKey: "OpenAI API key",
      apiKeyPlaceholder: "Leave blank to keep existing key",
      hideApiKey: "Hide API key",
      showApiKey: "Show API key",
      model: "Model",
      selectModel: "Select model",
      loading: "Loading",
      getModels: "Get models",
      translateConcurrency: "Translate concurrency",
      concurrencyHelp: "Parallel OpenAI requests during the translate stage. Increase if your provider allows it.",
      save: "Save settings",
      keySaved: "OpenAI key is saved.",
      saved: "Settings saved.",
      saveError: "Failed to save settings",
      noModels: "No models returned.",
      loadModelsError: "Failed to load models",
      funasrUseVllm: "FunASR vLLM engine",
      funasrUseVllmHelp:
        "Use the vLLM inference engine for LLM-based ASR models (e.g. Fun-ASR-Nano). Auto: enabled when the model needs it and vllm is installed. On: force on (requires vllm). Off: force off (Nano will raise NotImplementedError).",
      funasrVllmAuto: "Auto (recommended)",
      funasrVllmOn: "Always on",
      funasrVllmOff: "Always off",
    },
    status: {
      queued: "queued",
      running: "running",
      succeeded: "succeeded",
      failed: "failed",
      pending: "pending",
    },
    stages: {
      download: "Download",
      separate: "Demucs",
      asr: "Whisper",
      asr_fix: "Split sentences",
      translate: "Translate",
      split_audio: "Split audio",
      tts: "VoxCPM",
      merge_audio: "Merge audio",
      merge_video: "Merge video",
      done: "Done",
    },
  },
  zh: {
    common: {
      back: "返回",
      cancel: "取消",
      close: "关闭",
      loading: "加载中",
      sentenceEnd: "。",
      waiting: "等待中",
    },
    home: {
      createTitle: "新建任务",
      youtubeLabel: "YouTube 链接",
      bilibiliLabel: "Bilibili 链接",
      localVideoLabel: "本地视频文件",
      localDirectionLabel: "翻译方向",
      localEnZh: "英文 -> 中文",
      localZhEn: "中文 -> 英文",
      addSubtitles: "添加字幕",
      addSubtitlesHelp: "将字幕嵌入最终视频",
      asrModelLabel: "语音识别模型",
      asrWhisperTurbo: "Whisper large-v3-turbo",
      asrWhisperLarge: "Whisper large-v3",
      asrSenseVoice: "SenseVoiceSmall",
      asrFunAsrNano: "Fun-ASR-Nano",
      asrQwen3Asr: "Qwen3-ASR",
      groupInput: "输入源",
      groupAsr: "语音识别",
      groupTranslate: "翻译",
      translateModeLabel: "翻译模式",
      translateModeSentence: "逐句（独立翻译，快）",
      translateModeBatch: "分批（上下文感知，慢）",
      groupMergeVideo: "视频合成",
      submitting: "提交中",
      createTask: "创建任务",
      taskHistory: "任务历史",
      empty: "暂无任务。输入链接或上传本地视频后即可开始。",
      loadError: "加载任务失败",
      createError: "创建任务失败",
    },
    task: {
      overview: "任务概览",
      title: "标题",
      taskId: "任务 ID",
      created: "创建时间",
      started: "开始时间",
      completed: "完成时间",
      session: "会话目录",
      loading: "正在加载任务...",
      finalVideo: "最终视频",
      download: "下载",
      stages: "处理阶段",
      resumeHelp: "从失败阶段继续执行。已经成功的阶段会复用缓存结果。",
      resuming: "继续中",
      resumeTask: "继续任务",
      stopTask: "停止任务",
      stopping: "停止中",
      stopTitle: "确认停止这个任务？",
      stopDescription:
        "任务将在下一个安全检查点（阶段边界或进度更新处）停止。当前阶段会运行到结束，然后把任务标记为失败。已经成功的阶段及其产出会保留，之后可以继续执行。",
      confirmStop: "确认停止",
      stopError: "停止任务失败",
      stoppedNotice: "此任务已被手动停止。",
      rerunStage: "从此处重跑",
      rerunStageTitle: "确认从此阶段重跑？",
      rerunStageDescription: "该阶段及后续所有阶段将被重置并重新执行。该阶段之前已成功的阶段会复用缓存结果。",
      rerunningStage: "重跑中",
      rerunSingleStage: "重跑此阶段",
      rerunSingleStageTitle: "确认重跑此阶段？",
      rerunSingleStageDescription: "仅重置并重新执行该阶段。其他所有阶段及其产出保持不变。",
      rerunningSingleStage: "重跑中",
      stageConfig: "配置",
      stageConfigTitle: "阶段配置",
      stageConfigSaved: "配置已保存。",
      stageConfigError: "保存配置失败",
      clearStage: "清理产出",
      clearStageTitle: "确认清理阶段产出？",
      clearStageDescription: "将删除该阶段的缓存产出文件，需要重新运行该阶段才能重新生成。",
      clearStageDone: "阶段产出已清理。",
      clearStageError: "清理阶段产出失败",
      runLog: "运行日志",
      emptyLog: "任务开始后会显示日志。",
      dangerZone: "危险操作",
      rerunHelp: "清空会话目录，并从头重新运行这个链接。",
      rerunTask: "重跑任务",
      rerunTitle: "确认重跑这个任务？",
      rerunDescription:
        "现有日志、会话目录和最终视频会被删除，然后使用同一个任务 ID 重新排队处理相同链接。",
      rerunning: "重跑中",
      confirmRerun: "确认重跑",
      deleteHelp: "删除这个任务、运行日志，以及对应的整个会话目录：",
      deleteTask: "删除任务",
      deleteTitle: "确认删除这个任务？",
      deleteDescription: "这会永久删除任务记录、日志文件和整个会话目录。此操作无法撤销。",
      deleting: "删除中",
      confirmDelete: "确认删除",
      runningLocked: "运行中的任务不能重跑或删除，请等待任务完成或失败。",
      loadError: "加载任务失败",
      deleteError: "删除任务失败",
      rerunError: "重跑任务失败",
      resumeError: "继续任务失败",
      stageInfoTitle: "阶段说明",
    },
    settings: {
      button: "设置",
      title: "运行设置",
      description: "设置会由 FastAPI 后端保存在本机。",
      language: "界面语言",
      cookie: "YouTube Cookie",
      savedCookie: "******** 已保存 YouTube Cookie ********",
      cookiePlaceholder: "粘贴 Netscape 格式 Cookie 内容",
      proxyPort: "yt-dlp 代理端口",
      baseUrl: "OpenAI Base URL",
      apiKey: "OpenAI API Key",
      apiKeyPlaceholder: "留空则保留现有 key",
      hideApiKey: "隐藏 API key",
      showApiKey: "显示 API key",
      model: "模型",
      selectModel: "选择模型",
      loading: "加载中",
      getModels: "获取模型",
      translateConcurrency: "翻译并发数",
      concurrencyHelp: "翻译阶段并行发起的 OpenAI 请求数。如果你的服务商允许，可以适当调高。",
      save: "保存设置",
      keySaved: "OpenAI API key 已保存。",
      saved: "设置已保存。",
      saveError: "保存设置失败",
      noModels: "没有返回可用模型。",
      loadModelsError: "加载模型失败",
      funasrUseVllm: "FunASR vLLM 引擎",
      funasrUseVllmHelp:
        "对 LLM-based 模型（如 Fun-ASR-Nano）使用 vLLM 推理引擎。auto：检测到 LLM 模型且 vllm 已装时自动启用；on：强制启用（需要 vllm）；off：强制禁用（Nano 会报 NotImplementedError）。",
      funasrVllmAuto: "自动（推荐）",
      funasrVllmOn: "始终启用",
      funasrVllmOff: "始终禁用",
    },
    status: {
      queued: "排队中",
      running: "运行中",
      succeeded: "已完成",
      failed: "失败",
      pending: "等待中",
    },
    stages: {
      download: "下载视频",
      separate: "分离人声与背景音",
      asr: "语音识别",
      asr_fix: "切分句子",
      translate: "翻译字幕",
      split_audio: "切分音频",
      tts: "生成配音",
      merge_audio: "混合音频",
      merge_video: "合成视频",
      done: "已完成",
    },
  },
}

export type StageInfo = {
  description: Record<UiLanguage, string>
  input: Record<UiLanguage, string>
  output: Record<UiLanguage, string>
}

export const STAGE_INFO: Record<string, StageInfo> = {
  download: {
    description: { en: "Download the video from YouTube/Bilibili, or import a local file.", zh: "从 YouTube/Bilibili 下载视频，或导入本地文件。" },
    input: { en: "Video URL or local file", zh: "视频链接或本地文件" },
    output: { en: "media/video_source.mp4\nmetadata/ytdlp_info.json or metadata/local_info.json", zh: "media/video_source.mp4\nmetadata/ytdlp_info.json or metadata/local_info.json" },
  },
  separate: {
    description: { en: "Use Demucs to separate the audio into vocals and background music (BGM).", zh: "使用 Demucs 将音频分离为人声和背景音乐（BGM）。" },
    input: { en: "media/video_source.mp4", zh: "media/video_source.mp4" },
    output: { en: "media/audio_vocals.wav\nmedia/audio_bgm.wav", zh: "media/audio_vocals.wav\nmedia/audio_bgm.wav" },
  },
  asr: {
    description: { en: "Use Whisper to recognize speech in the vocals and generate timestamps.", zh: "使用 Whisper 识别语音并生成时间戳。" },
    input: { en: "media/audio_vocals.wav", zh: "media/audio_vocals.wav" },
    output: { en: "metadata/asr.json", zh: "metadata/asr.json" },
  },
  asr_fix: {
    description: { en: "Re-segment the ASR output into clean, well-timed sentences.", zh: "将 ASR 输出重新切分为干净、时间合理的句子。" },
    input: { en: "metadata/asr.json", zh: "metadata/asr.json" },
    output: { en: "metadata/asr_fixed.json", zh: "metadata/asr_fixed.json" },
  },
  translate: {
    description: { en: "Translate the recognized text to the target language using OpenAI, and generate SRT subtitles.", zh: "使用 OpenAI 将识别的文本翻译为目标语言，并生成 SRT 字幕。" },
    input: { en: "metadata/asr_fixed.json", zh: "metadata/asr_fixed.json" },
    output: { en: "metadata/translation.{lang}.json\nmetadata/subtitles.{lang}.srt", zh: "metadata/translation.{lang}.json\nmetadata/subtitles.{lang}.srt" },
  },
  split_audio: {
    description: { en: "Split the vocals into per-segment reference clips for TTS.", zh: "将人声按句子切分为 TTS 参考音频片段。" },
    input: { en: "media/audio_vocals.wav\nmetadata/translation.{lang}.json", zh: "media/audio_vocals.wav\nmetadata/translation.{lang}.json" },
    output: { en: "segments/vocals/0001.wav\nsegments/vocals/0002.wav\n...", zh: "segments/vocals/0001.wav\nsegments/vocals/0002.wav\n..." },
  },
  tts: {
    description: { en: "Use VoxCPM2 to generate dubbed audio for each translated sentence.", zh: "使用 VoxCPM2 为每句翻译生成配音音频。" },
    input: { en: "metadata/translation.{lang}.json\nsegments/vocals/*.wav", zh: "metadata/translation.{lang}.json\nsegments/vocals/*.wav" },
    output: { en: "segments/tts/0001.wav\nsegments/tts/0002.wav\n...", zh: "segments/tts/0001.wav\nsegments/tts/0002.wav\n..." },
  },
  merge_audio: {
    description: { en: "Concatenate TTS clips, adjust speed to match original timing, and produce the final dubbing track.", zh: "拼接 TTS 片段，调整语速对齐原始时间轴，生成最终配音音轨。" },
    input: { en: "metadata/translation.{lang}.json\nsegments/tts/*.wav", zh: "metadata/translation.{lang}.json\nsegments/tts/*.wav" },
    output: { en: "tmp/audio_dubbing.wav\nmetadata/timings.json", zh: "tmp/audio_dubbing.wav\nmetadata/timings.json" },
  },
  merge_video: {
    description: { en: "Mix dubbing with BGM, optionally burn subtitles, and merge with the original video.", zh: "将配音与 BGM 混合，可选烧录字幕，与原始视频合并。" },
    input: { en: "media/video_source.mp4\ntmp/audio_dubbing.wav\nmedia/audio_bgm.wav\nmetadata/subtitles.{lang}.srt", zh: "media/video_source.mp4\ntmp/audio_dubbing.wav\nmedia/audio_bgm.wav\nmetadata/subtitles.{lang}.srt" },
    output: { en: "media/video_final.mp4", zh: "media/video_final.mp4" },
  },
}

type LanguageContextValue = {
  language: UiLanguage
  setLanguage: (language: UiLanguage) => void
  t: Messages
  activeTasksText: (count: number) => string
  loadedModelsText: (count: number) => string
  statusLabel: (status?: string | null) => string
  stageLabel: (name?: string | null, fallback?: string | null) => string
}

const LanguageContext = createContext<LanguageContextValue | null>(null)

function isLanguage(value: string | null): value is UiLanguage {
  return value === "en" || value === "zh"
}

function setDocumentLanguage(language: UiLanguage) {
  document.documentElement.lang = language === "zh" ? "zh-CN" : "en"
}

export function LanguageProvider({ children }: { children: ReactNode }) {
  const [language, setLanguageState] = useState<UiLanguage>("zh")

  useEffect(() => {
    const saved = window.localStorage.getItem(STORAGE_KEY)
    if (!isLanguage(saved)) return
    window.setTimeout(() => setLanguageState(saved), 0)
  }, [])

  useEffect(() => {
    setDocumentLanguage(language)
  }, [language])

  const value = useMemo<LanguageContextValue>(() => {
    const t = messages[language]
    return {
      language,
      setLanguage: (next) => {
        setLanguageState(next)
        window.localStorage.setItem(STORAGE_KEY, next)
        setDocumentLanguage(next)
      },
      t,
      activeTasksText: (count) =>
        language === "zh"
          ? `${count} 个任务正在排队或运行`
          : `${count} task${count > 1 ? "s" : ""} queued / running`,
      loadedModelsText: (count) =>
        language === "zh" ? `已加载 ${count} 个模型。` : `${count} models loaded.`,
      statusLabel: (status) => {
        if (!status) return t.common.loading
        return t.status[status as keyof typeof t.status] || status
      },
      stageLabel: (name, fallback) => {
        if (name && name in t.stages) return t.stages[name as keyof typeof t.stages]
        if (fallback && fallback in t.stages) return t.stages[fallback as keyof typeof t.stages]
        return fallback || name || t.common.waiting
      },
    }
  }, [language])

  return <LanguageContext.Provider value={value}>{children}</LanguageContext.Provider>
}

export function useI18n() {
  const context = useContext(LanguageContext)
  if (!context) {
    throw new Error("useI18n must be used inside LanguageProvider")
  }
  return context
}
