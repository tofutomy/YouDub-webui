function configuredApiBase(): string {
  const configured = process.env.NEXT_PUBLIC_API_BASE_URL?.trim()
  if (configured) return configured.replace(/\/$/, "")

  if (typeof window === "undefined") return "http://127.0.0.1:8000"
  return ""
}

export const API_BASE = configuredApiBase()

export type StageStatus = "pending" | "running" | "succeeded" | "failed"
export type TaskStatus = "queued" | "running" | "succeeded" | "failed"

export type TaskStage = {
  task_id: string
  name: string
  label: string
  status: StageStatus
  progress: number | null
  started_at: string | null
  completed_at: string | null
  last_message: string | null
  error_message: string | null
}

export type Task = {
  id: string
  url: string
  title: string | null
  status: TaskStatus
  current_stage: string | null
  session_path: string | null
  final_video_path: string | null
  error_message: string | null
  created_at: string
  started_at: string | null
  completed_at: string | null
  stages: TaskStage[]
  asr_model: string | null
  asr_language: string | null
  target_language: string | null
  add_subtitles: number | null
  stop_requested: number | null
  translate_mode: string | null
  validate_translation: number | null
  tts_mode: string | null
  translate_provider_id: string | null
  stop_after_translate: number | null
  demucs_model: string | null
  demucs_shifts: number | null
}

export type CookieInfo = {
  exists: boolean
  size: number
  updated_at: number | null
  content: string
}

export type OpenAISettings = {
  base_url: string
  api_key: string
  has_api_key: boolean
  model: string
  translate_concurrency: string
}

export type OpenAIModels = {
  models: string[]
}

export type YtdlpSettings = {
  proxy_port: string
}

export type FunasrSettings = {
  use_vllm: "auto" | "on" | "off"
}

export type TranslateProvider = {
  id: string
  name: string
  base_url: string
  api_key: string
  has_api_key: boolean
  model: string
  is_default: boolean
  created_at: string
  updated_at: string
}

export type LocalDirection = "en-zh" | "zh-en" | "ja-zh"

export type StageConfig = {
  asr_model: string
  asr_language: string
  target_language: string
  add_subtitles: boolean
  translate_mode: string
  validate_translation: boolean
  tts_mode: string
  translate_provider_id: string
  stop_after_translate: boolean
  filter_fillers: boolean
  demucs_model: string
  demucs_shifts: number
}

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...(options?.headers || {}),
    },
    cache: "no-store",
  })
  if (!response.ok) {
    const body = await response.json().catch(() => ({}))
    throw new Error(body.detail || `Request failed: ${response.status}`)
  }
  if (response.status === 204) {
    return undefined as T
  }
  return response.json()
}

export type TaskSummary = {
  id: string
  url: string
  title: string | null
  status: TaskStatus
  current_stage: string | null
  final_video_path: string | null
  error_message: string | null
  created_at: string
  started_at: string | null
  completed_at: string | null
}

export function getCurrentTask() {
  return request<Task | null>("/api/tasks/current")
}

export async function getTaskLog(taskId: string): Promise<string> {
  const response = await fetch(`${API_BASE}/api/tasks/${taskId}/log`, { cache: "no-store" })
  if (!response.ok) {
    throw new Error(`Failed to load log: ${response.status}`)
  }
  return response.text()
}

export function listTasks(limit = 10, offset = 0) {
  return request<{ tasks: TaskSummary[]; total: number }>(`/api/tasks?limit=${limit}&offset=${offset}`)
}

export function getTask(taskId: string) {
  return request<Task>(`/api/tasks/${taskId}`)
}

export function deleteTask(taskId: string) {
  return request<void>(`/api/tasks/${taskId}`, { method: "DELETE" })
}

export function rerunTask(taskId: string) {
  return request<Task>(`/api/tasks/${taskId}/rerun`, { method: "POST" })
}

export function resumeTask(taskId: string) {
  return request<Task>(`/api/tasks/${taskId}/resume`, { method: "POST" })
}

export function stopTask(taskId: string) {
  return request<Task>(`/api/tasks/${taskId}/stop`, { method: "POST" })
}

export function rerunStage(taskId: string, stageName: string) {
  return request<Task>(`/api/tasks/${taskId}/rerun-stage/${stageName}`, { method: "POST" })
}

export function rerunSingleStage(taskId: string, stageName: string) {
  return request<Task>(`/api/tasks/${taskId}/rerun-single-stage/${stageName}`, { method: "POST" })
}

export function updateTaskConfig(taskId: string, config: Partial<StageConfig>) {
  return request<Task>(`/api/tasks/${taskId}/config`, {
    method: "PATCH",
    body: JSON.stringify(config),
  })
}

export function clearStageOutput(taskId: string, stageName: string) {
  return request<Task>(`/api/tasks/${taskId}/clear-stage/${stageName}`, { method: "POST" })
}

export function createTask(url: string, config: StageConfig) {
  return request<Task>("/api/tasks", {
    method: "POST",
    body: JSON.stringify({ url, config }),
  })
}

export async function uploadLocalTask(file: File, config: StageConfig) {
  const form = new FormData()
  form.append("config", JSON.stringify(config))
  form.append("file", file)

  const response = await fetch(`${API_BASE}/api/tasks/upload`, {
    method: "POST",
    body: form,
    cache: "no-store",
  })
  if (!response.ok) {
    const body = await response.json().catch(() => ({}))
    throw new Error(body.detail || `Request failed: ${response.status}`)
  }
  return response.json() as Promise<Task>
}

export function createLocaldirTask(filePath: string, config: StageConfig) {
  return request<Task>("/api/tasks/localdir", {
    method: "POST",
    body: JSON.stringify({ file_path: filePath, config }),
  })
}

export function getCookieInfo() {
  return request<CookieInfo>("/api/cookies/youtube")
}

export function saveCookie(content: string) {
  return request<CookieInfo>("/api/cookies/youtube", {
    method: "POST",
    body: JSON.stringify({ content }),
  })
}

export function getOpenAISettings() {
  return request<OpenAISettings>("/api/settings/openai")
}

export function saveOpenAISettings(settings: {
  base_url: string
  api_key: string
  clear_api_key?: boolean
  model: string
  translate_concurrency: string
}) {
  return request<OpenAISettings>("/api/settings/openai", {
    method: "POST",
    body: JSON.stringify(settings),
  })
}

export function getOpenAIModels(settings: {
  base_url: string
  api_key: string
}) {
  return request<OpenAIModels>("/api/settings/openai/models", {
    method: "POST",
    body: JSON.stringify(settings),
  })
}

export function getYtdlpSettings() {
  return request<YtdlpSettings>("/api/settings/ytdlp")
}

export function saveYtdlpSettings(settings: YtdlpSettings) {
  return request<YtdlpSettings>("/api/settings/ytdlp", {
    method: "POST",
    body: JSON.stringify(settings),
  })
}

export function getFunasrSettings() {
  return request<FunasrSettings>("/api/settings/funasr")
}

export function saveFunasrSettings(settings: FunasrSettings) {
  return request<FunasrSettings>("/api/settings/funasr", {
    method: "POST",
    body: JSON.stringify(settings),
  })
}

// ---------------------------------------------------------------------------
// Translate Providers
// ---------------------------------------------------------------------------

export function getTranslateProviders() {
  return request<{ providers: TranslateProvider[] }>("/api/translate-providers")
}

export function createTranslateProvider(provider: {
  name: string
  base_url?: string
  api_key?: string
  model?: string
  is_default?: boolean
}) {
  return request<TranslateProvider>("/api/translate-providers", {
    method: "POST",
    body: JSON.stringify(provider),
  })
}

export function updateTranslateProvider(id: string, provider: {
  name?: string
  base_url?: string
  api_key?: string
  clear_api_key?: boolean
  model?: string
  is_default?: boolean
}) {
  return request<TranslateProvider>(`/api/translate-providers/${id}`, {
    method: "PATCH",
    body: JSON.stringify(provider),
  })
}

export function deleteTranslateProvider(id: string) {
  return request<void>(`/api/translate-providers/${id}`, { method: "DELETE" })
}

export function getTranslateProviderModels(id: string) {
  return request<{ models: string[] }>(`/api/translate-providers/${id}/models`, {
    method: "POST",
  })
}

export function finalVideoUrl(taskId: string) {
  return `${API_BASE}/api/tasks/${taskId}/artifact/final-video`
}

export function finalVideoDownloadUrl(taskId: string) {
  return `${API_BASE}/api/tasks/${taskId}/artifact/final-video?download=1`
}
