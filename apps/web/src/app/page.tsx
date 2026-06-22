"use client"

import Link from "next/link"
import { ChangeEvent, FormEvent, useEffect, useRef, useState } from "react"
import { ChevronRight, Play, Upload } from "lucide-react"

import {
  TaskSummary,
  TranslateProvider,
  createLocaldirTask,
  createTask,
  getTranslateProviders,
  listTasks,
  uploadLocalTask,
} from "@/lib/api"
import { useI18n } from "@/lib/i18n"
import { statusBadgeClass } from "@/lib/status"
import { AppHeader } from "@/components/app-header"
import {
  DEFAULT_STAGE_CONFIG,
  STAGE_FIELDS,
} from "@/lib/stage-config"
import type { StageConfig } from "@/lib/api"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Separator } from "@/components/ui/separator"

const CREATE_SECTION_STAGES = ["separate", "translate", "asr", "asr_fix", "tts", "merge_video"] as const

function isActive(status: string) {
  return status === "queued" || status === "running"
}

function formatTime(value: string | null) {
  if (!value) return ""
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return value
  return date.toLocaleString()
}

function shortUrl(url: string) {
  return url.replace(/^https?:\/\/(www\.)?/, "")
}

function activeCount(tasks: TaskSummary[]) {
  return tasks.filter((t) => isActive(t.status)).length
}

function getPageNumbers(current: number, totalPages: number): (number | "...")[] {
  if (totalPages <= 7) return Array.from({ length: totalPages }, (_, i) => i + 1)
  const pages: (number | "...")[] = [1]
  if (current > 3) pages.push("...")
  const start = Math.max(2, current - 1)
  const end = Math.min(totalPages - 1, current + 1)
  for (let i = start; i <= end; i++) pages.push(i)
  if (current < totalPages - 2) pages.push("...")
  pages.push(totalPages)
  return pages
}

export default function Home() {
  const { activeTasksText, stageLabel, statusLabel, t, paginationTotalText, paginationPageInfoText } = useI18n()
  const fileInputRef = useRef<HTMLInputElement>(null)
  const [youtubeUrl, setYoutubeUrl] = useState("")
  const [bilibiliUrl, setBilibiliUrl] = useState("")
  const [localFile, setLocalFile] = useState<File | null>(null)
  const [localPath, setLocalPath] = useState("")
  const [config, setConfig] = useState<StageConfig>(DEFAULT_STAGE_CONFIG)
  const [providerOptions, setProviderOptions] = useState<TranslateProvider[]>([])
  const [tasks, setTasks] = useState<TaskSummary[]>([])
  const [error, setError] = useState("")
  const [submitting, setSubmitting] = useState(false)
  const pageSize = 10
  const [page, setPage] = useState(1)
  const [total, setTotal] = useState(0)

  function updateConfig(patch: Partial<StageConfig>) {
    setConfig((prev) => ({ ...prev, ...patch }))
  }

  useEffect(() => {
    let cancelled = false

    const loadPage = async (p: number) => {
      const offset = (p - 1) * pageSize
      try {
        const { tasks: list, total: t } = await listTasks(pageSize, offset)
        if (!cancelled) { setTasks(list); setTotal(t) }
      } catch (err) {
        if (!cancelled) setError(err instanceof Error ? err.message : t.home.loadError)
      }
    }

    loadPage(page)
    getTranslateProviders()
      .then((resp) => { if (!cancelled) setProviderOptions(resp.providers) })
      .catch(() => {})
    if (page !== 1) return () => { cancelled = true }

    const interval = window.setInterval(() => loadPage(1), 2000)
    return () => {
      cancelled = true
      window.clearInterval(interval)
    }
  }, [page, t.home.loadError])

  function selectLocalFile(event: ChangeEvent<HTMLInputElement>) {
    setError("")
    setLocalFile(event.target.files?.[0] || null)
  }

  async function submitTask(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    setError("")
    const submittedUrl = youtubeUrl.trim() || bilibiliUrl.trim()
    const submittedPath = localPath.trim().replace(/^["']|["']$/g, "")
    if (!submittedUrl && !localFile && !submittedPath) return
    setSubmitting(true)
    try {
      if (localFile) {
        await uploadLocalTask(localFile, config)
      } else if (submittedPath) {
        await createLocaldirTask(submittedPath, config)
      } else {
        await createTask(submittedUrl, config)
      }
      const { tasks: list, total: t } = await listTasks(pageSize, 0)
      setTasks(list)
      setTotal(t)
      setPage(1)
    } catch (err) {
      setError(err instanceof Error ? err.message : t.home.createError)
    } finally {
      setSubmitting(false)
    }
  }

  const queued = activeCount(tasks)
  const totalPages = Math.max(1, Math.ceil(total / pageSize))
  const hasUrl = Boolean(youtubeUrl.trim() || bilibiliUrl.trim())
  const hasLocalFile = Boolean(localFile)
  const hasLocalPath = Boolean(localPath.trim())
  const canSubmit = Boolean((hasUrl || hasLocalFile || hasLocalPath) && !submitting)

  const sectionLabelKey: Record<string, string> = {
    separate: t.home.separateModeLabel,
    translate: t.home.groupTranslate,
    asr: t.home.groupAsr,
    asr_fix: t.home.groupAsrFix,
    tts: t.home.groupTts,
    merge_video: t.home.groupMergeVideo,
  }

  return (
    <main className="min-h-screen bg-[linear-gradient(135deg,#fff5f5_0%,#f2fbff_48%,#fff4fa_100%)] text-foreground">
      <div className="mx-auto flex w-full max-w-4xl flex-col gap-6 px-4 py-6 sm:px-6 lg:px-8">
        <AppHeader />

        <Card>
          <CardHeader>
            <CardTitle>{t.home.createTitle}</CardTitle>
          </CardHeader>
          <CardContent>
            <form onSubmit={submitTask} className="space-y-6">
              {/* Input source */}
              <div className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="youtube-url">{t.home.youtubeLabel}</Label>
                  <Input
                    id="youtube-url"
                    value={youtubeUrl}
                    onChange={(event) => setYoutubeUrl(event.target.value)}
                    placeholder="https://www.youtube.com/watch?v=..."
                    disabled={Boolean(bilibiliUrl.trim()) || hasLocalFile || hasLocalPath}
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="bilibili-url">{t.home.bilibiliLabel}</Label>
                  <Input
                    id="bilibili-url"
                    value={bilibiliUrl}
                    onChange={(event) => setBilibiliUrl(event.target.value)}
                    placeholder="https://www.bilibili.com/video/BV..."
                    disabled={Boolean(youtubeUrl.trim()) || hasLocalFile || hasLocalPath}
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="local-video">{t.home.localVideoLabel}</Label>
                  <Input
                    ref={fileInputRef}
                    id="local-video"
                    type="file"
                    accept="video/*,.mp4,.mov,.m4v,.mkv,.webm,.avi,.flv,.wmv"
                    onChange={selectLocalFile}
                    disabled={hasUrl || hasLocalPath}
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="local-path">{t.home.localDirLabel}</Label>
                  <Input
                    id="local-path"
                    value={localPath}
                    onChange={(event) => setLocalPath(event.target.value)}
                    placeholder={t.home.localDirPlaceholder}
                    disabled={hasUrl || hasLocalFile}
                  />
                </div>
              </div>

              {/* Stage config sections (from registry) */}
              {CREATE_SECTION_STAGES.map((stage) => {
                const fields = STAGE_FIELDS[stage]
                if (!fields || fields.length === 0) return null
                return (
                  <div key={stage} className="space-y-3">
                    <div className="flex items-center gap-2">
                      <Separator className="flex-1" />
                      <span className="text-xs font-medium text-muted-foreground">
                        {sectionLabelKey[stage]}
                      </span>
                      <Separator className="flex-1" />
                    </div>
                    {fields.map((field, i) => (
                      <div key={i}>
                        {field.render({ config, onChange: updateConfig, providerOptions })}
                      </div>
                    ))}
                  </div>
                )
              })}

              <div className="flex items-center justify-between gap-3">
                {queued > 0 ? (
                  <p className="text-xs text-muted-foreground">
                    {activeTasksText(queued)}
                  </p>
                ) : (
                  <span />
                )}
                <Button type="submit" disabled={!canSubmit}>
                  {(hasLocalFile || hasLocalPath) ? <Upload className="size-4" /> : <Play className="size-4" />}
                  {submitting ? t.home.submitting : t.home.createTask}
                </Button>
              </div>
            </form>

            {error ? (
              <div className="mt-4 rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">
                {error}
              </div>
            ) : null}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>{t.home.taskHistory} ({total})</CardTitle>
          </CardHeader>
          <CardContent className="px-0">
            {tasks.length === 0 ? (
              <div className="px-6 py-12 text-center text-sm text-muted-foreground">
                {t.home.empty}
              </div>
            ) : (
              <ul className="flex flex-col">
                {tasks.map((item) => (
                  <li key={item.id} className="border-b border-border/60 last:border-b-0">
                    <Link
                      href={`/tasks/${item.id}`}
                      className="flex w-full items-center gap-3 px-6 py-3 text-sm transition-colors hover:bg-muted/60"
                    >
                      <div className="min-w-0 flex-1">
                        <p className="truncate text-left font-medium text-zinc-900">
                          {item.title || shortUrl(item.url)}
                        </p>
                        <div className="mt-1 flex items-center gap-2 text-xs text-muted-foreground">
                          <Badge className={statusBadgeClass(item.status)}>{statusLabel(item.status)}</Badge>
                          <span>{formatTime(item.created_at)}</span>
                          {isActive(item.status) && item.current_stage ? (
                            <span>· {stageLabel(item.current_stage)}</span>
                          ) : null}
                        </div>
                      </div>
                      <ChevronRight className="size-4 shrink-0 text-muted-foreground" />
                    </Link>
                  </li>
                ))}
              </ul>
            )}
            {totalPages > 1 && (
              <div className="flex items-center justify-center gap-1 border-t border-border/60 px-6 py-3">
                <Button
                  variant="ghost"
                  size="sm"
                  disabled={page <= 1}
                  onClick={() => setPage(page - 1)}
                >
                  {t.home.paginationPrev}
                </Button>
                {getPageNumbers(page, totalPages).map((p, i) =>
                  p === "..." ? (
                    <span key={`dots-${i}`} className="px-2 text-xs text-muted-foreground">...</span>
                  ) : (
                    <Button
                      key={p}
                      variant={p === page ? "default" : "ghost"}
                      size="sm"
                      className="min-w-[2.25rem]"
                      onClick={() => setPage(p)}
                    >
                      {p}
                    </Button>
                  )
                )}
                <Button
                  variant="ghost"
                  size="sm"
                  disabled={page >= totalPages}
                  onClick={() => setPage(page + 1)}
                >
                  {t.home.paginationNext}
                </Button>
              </div>
            )}
            {total > 0 && (
              <div className="px-6 py-2 text-center text-xs text-muted-foreground">
                {paginationTotalText(total)} · {paginationPageInfoText(page, totalPages || 1)}
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </main>
  )
}
