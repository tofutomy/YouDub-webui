"use client"

import { useRouter } from "next/navigation"
import { use, useEffect, useMemo, useState } from "react"
import {
  CheckCircle2,
  Circle,
  Download,
  Eraser,
  FileText,
  Info,
  Loader2,
  Play,
  RotateCw,
  Settings,
  StopCircle,
  Trash2,
  XCircle,
} from "lucide-react"

import {
  StageStatus,
  Task,
  TranslateProvider,
  clearStageOutput,
  deleteTask,
  finalVideoDownloadUrl,
  finalVideoUrl,
  getTask,
  getTaskLog,
  getTranslateProviders,
  rerunSingleStage,
  rerunStage,
  rerunTask,
  resumeTask,
  stopTask,
  updateTaskConfig,
} from "@/lib/api"
import type { StageConfig } from "@/lib/api"
import { useI18n, STAGE_INFO } from "@/lib/i18n"
import { statusBadgeClass } from "@/lib/status"
import { AppHeader } from "@/components/app-header"
import {
  CONFIGURABLE_STAGES,
  DEFAULT_STAGE_CONFIG,
  STAGE_FIELDS,
  configForStage,
  configFromTask,
} from "@/lib/stage-config"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import {
  Dialog,
  DialogClose,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog"
import { Progress } from "@/components/ui/progress"
import { ScrollArea } from "@/components/ui/scroll-area"

function stageIcon(status: StageStatus) {
  if (status === "succeeded") return <CheckCircle2 className="size-5 text-[#00aeec]" />
  if (status === "failed") return <XCircle className="size-5 text-[#ff0033]" />
  if (status === "running") return <Loader2 className="size-5 animate-spin text-[#fb7299]" />
  return <Circle className="size-5 text-muted-foreground" />
}

function formatTime(value: string | null) {
  if (!value) return ""
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return value
  return date.toLocaleString()
}

function durationOf(start: string | null, end: string | null) {
  if (!start) return ""
  const startMs = new Date(start).getTime()
  const endMs = end ? new Date(end).getTime() : Date.now()
  if (!Number.isFinite(startMs) || !Number.isFinite(endMs)) return ""
  const seconds = Math.max(0, Math.round((endMs - startMs) / 1000))
  if (seconds < 60) return `${seconds}s`
  const minutes = Math.floor(seconds / 60)
  const rem = seconds % 60
  return `${minutes}m${rem.toString().padStart(2, "0")}s`
}

function normalizeProgress(value: number | null | undefined) {
  if (typeof value !== "number") return null
  return Math.max(0, Math.min(100, Math.round(value)))
}

export default function TaskDetailPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params)
  const router = useRouter()
  const { language, stageLabel, statusLabel, t } = useI18n()
  const [task, setTask] = useState<Task | null>(null)
  const [log, setLog] = useState("")
  const [error, setError] = useState("")
  const [deleteOpen, setDeleteOpen] = useState(false)
  const [deleting, setDeleting] = useState(false)
  const [deleteError, setDeleteError] = useState("")
  const [rerunOpen, setRerunOpen] = useState(false)
  const [rerunning, setRerunning] = useState(false)
  const [rerunError, setRerunError] = useState("")
  const [resuming, setResuming] = useState(false)
  const [resumeError, setResumeError] = useState("")
  const [stopOpen, setStopOpen] = useState(false)
  const [stopping, setStopping] = useState(false)
  const [stopError, setStopError] = useState("")
  const [rerunStageOpen, setRerunStageOpen] = useState(false)
  const [rerunStageTarget, setRerunStageTarget] = useState<string | null>(null)
  const [rerunStaging, setRerunStaging] = useState(false)
  const [rerunStageError, setRerunStageError] = useState("")
  const [rerunSingleOpen, setRerunSingleOpen] = useState(false)
  const [rerunSingleTarget, setRerunSingleTarget] = useState<string | null>(null)
  const [rerunSingleIng, setRerunSingleIng] = useState(false)
  const [rerunSingleError, setRerunSingleError] = useState("")
  const [infoStageTarget, setInfoStageTarget] = useState<string | null>(null)
  const [configStageTarget, setConfigStageTarget] = useState<string | null>(null)
  const [configSaving, setConfigSaving] = useState(false)
  const [configError, setConfigError] = useState("")
  const [configSuccess, setConfigSuccess] = useState(false)
  const [clearStageOpen, setClearStageOpen] = useState(false)
  const [clearStageTarget, setClearStageTarget] = useState<string | null>(null)
  const [clearingStage, setClearingStage] = useState(false)
  const [clearStageError, setClearStageError] = useState("")
  // Config form state
  const [cfgConfig, setCfgConfig] = useState<StageConfig>(DEFAULT_STAGE_CONFIG)
  const [providerOptions, setProviderOptions] = useState<TranslateProvider[]>([])

  const handleDelete = async () => {
    setDeleting(true)
    setDeleteError("")
    try {
      await deleteTask(id)
      router.replace("/")
    } catch (err) {
      setDeleteError(err instanceof Error ? err.message : t.task.deleteError)
      setDeleting(false)
    }
  }

  const handleRerun = async () => {
    setRerunning(true)
    setRerunError("")
    try {
      const next = await rerunTask(id)
      setRerunOpen(false)
      setTask(next)
      setLog("")
    } catch (err) {
      setRerunError(err instanceof Error ? err.message : t.task.rerunError)
    } finally {
      setRerunning(false)
    }
  }

  const handleResume = async () => {
    setResuming(true)
    setResumeError("")
    try {
      const next = await resumeTask(id)
      setTask(next)
    } catch (err) {
      setResumeError(err instanceof Error ? err.message : t.task.resumeError)
    } finally {
      setResuming(false)
    }
  }

  const handleStop = async () => {
    setStopping(true)
    setStopError("")
    try {
      await stopTask(id)
      setStopOpen(false)
      // The stop API only sets the flag for running tasks; the worker
      // asynchronously marks the task as failed.  Re-fetch after a short
      // delay so the UI reflects the final "failed" state immediately
      // instead of waiting up to 2 s for the next poll.
      await new Promise((r) => setTimeout(r, 500))
      const fresh = await getTask(id)
      setTask(fresh)
    } catch (err) {
      setStopError(err instanceof Error ? err.message : t.task.stopError)
    } finally {
      setStopping(false)
    }
  }

  const handleRerunStage = async () => {
    if (!rerunStageTarget) return
    setRerunStaging(true)
    setRerunStageError("")
    try {
      const next = await rerunStage(id, rerunStageTarget)
      setRerunStageOpen(false)
      setRerunStageTarget(null)
      setTask(next)
      setLog("")
    } catch (err) {
      setRerunStageError(err instanceof Error ? err.message : "Failed to rerun stage")
    } finally {
      setRerunStaging(false)
    }
  }

  const handleRerunSingleStage = async () => {
    if (!rerunSingleTarget) return
    setRerunSingleIng(true)
    setRerunSingleError("")
    try {
      const next = await rerunSingleStage(id, rerunSingleTarget)
      setRerunSingleOpen(false)
      setRerunSingleTarget(null)
      setTask(next)
      setLog("")
    } catch (err) {
      setRerunSingleError(err instanceof Error ? err.message : "Failed to rerun stage")
    } finally {
      setRerunSingleIng(false)
    }
  }

  const openConfigDialog = (stageName: string) => {
    setConfigStageTarget(stageName)
    setConfigError("")
    setConfigSuccess(false)
    if (task) {
      setCfgConfig(configFromTask(task))
    }
    if (stageName === "translate") {
      getTranslateProviders()
        .then((resp) => setProviderOptions(resp.providers))
        .catch(() => setProviderOptions([]))
    }
  }

  const handleSaveConfig = async () => {
    if (!configStageTarget || !task) return
    setConfigSaving(true)
    setConfigError("")
    setConfigSuccess(false)
    try {
      const patch = configForStage(configStageTarget, cfgConfig)
      const next = await updateTaskConfig(id, patch)
      setTask(next)
      setConfigSuccess(true)
    } catch (err) {
      setConfigError(err instanceof Error ? err.message : t.task.stageConfigError)
    } finally {
      setConfigSaving(false)
    }
  }

  const handleClearStage = async () => {
    if (!clearStageTarget) return
    setClearingStage(true)
    setClearStageError("")
    try {
      const next = await clearStageOutput(id, clearStageTarget)
      setClearStageOpen(false)
      setClearStageTarget(null)
      setTask(next)
    } catch (err) {
      setClearStageError(err instanceof Error ? err.message : t.task.clearStageError)
    } finally {
      setClearingStage(false)
    }
  }

  const isRunning = task?.status === "running"
  const isQueued = task?.status === "queued"
  const isActive = isRunning || isQueued
  const isFailed = task?.status === "failed"
  const isStopped = task?.stop_requested === 1

  useEffect(() => {
    let cancelled = false
    const load = async () => {
      try {
        const next = await getTask(id)
        if (cancelled) return
        setTask(next)
        const logText = await getTaskLog(id)
        if (cancelled) return
        setLog(logText)
      } catch (err) {
        if (!cancelled) setError(err instanceof Error ? err.message : t.task.loadError)
      }
    }
    load()
    const interval = window.setInterval(load, 2000)
    return () => {
      cancelled = true
      window.clearInterval(interval)
    }
  }, [id, t.task.loadError])

  const progress = useMemo(() => {
    if (!task?.stages?.length) return 0
    const completed = task.stages.filter((stage) => stage.status === "succeeded").length
    return Math.round((completed / task.stages.length) * 100)
  }, [task])

  if (error && !task) {
    return (
      <main className="min-h-screen bg-[linear-gradient(135deg,#fff5f5_0%,#f2fbff_48%,#fff4fa_100%)] text-foreground">
        <div className="mx-auto flex w-full max-w-4xl flex-col gap-6 px-4 py-6 sm:px-6 lg:px-8">
          <AppHeader backHref="/" />
          <Card>
            <CardContent className="px-6 py-10 text-sm text-red-600">{error}</CardContent>
          </Card>
        </div>
      </main>
    )
  }

  return (
    <main className="min-h-screen bg-[linear-gradient(135deg,#fff5f5_0%,#f2fbff_48%,#fff4fa_100%)] text-foreground">
      <div className="mx-auto flex w-full max-w-4xl flex-col gap-6 px-4 py-6 sm:px-6 lg:px-8">
        <AppHeader backHref="/" />

        <Card>
          <CardHeader className="gap-3">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <CardTitle>{t.task.overview}</CardTitle>
              <div className="flex flex-wrap items-center gap-2">
                <Badge className={statusBadgeClass(task?.status)}>{statusLabel(task?.status)}</Badge>
                {isActive ? (
                  <Dialog open={stopOpen} onOpenChange={setStopOpen}>
                    <DialogTrigger
                      render={
                        <Button variant="destructive" size="sm">
                          <StopCircle className="size-4" />
                          {t.task.stopTask}
                        </Button>
                      }
                    />
                    <DialogContent>
                      <DialogHeader>
                        <DialogTitle>{t.task.stopTitle}</DialogTitle>
                        <DialogDescription>
                          {t.task.stopDescription}
                        </DialogDescription>
                      </DialogHeader>
                      {stopError ? (
                        <div className="rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">
                          {stopError}
                        </div>
                      ) : null}
                      <DialogFooter>
                        <DialogClose render={<Button variant="outline" disabled={stopping} />}>
                          {t.common.cancel}
                        </DialogClose>
                        <Button variant="destructive" onClick={handleStop} disabled={stopping}>
                          {stopping ? <Loader2 className="size-4 animate-spin" /> : <StopCircle className="size-4" />}
                          {stopping ? t.task.stopping : t.task.confirmStop}
                        </Button>
                      </DialogFooter>
                    </DialogContent>
                  </Dialog>
                ) : null}
              </div>
            </div>
            <Progress value={progress} />
          </CardHeader>
          <CardContent>
            {task ? (
              <dl className="grid grid-cols-1 gap-x-6 gap-y-2 text-sm sm:grid-cols-[120px_1fr]">
                {task.title ? (
                  <>
                    <dt className="text-muted-foreground">{t.task.title}</dt>
                    <dd className="break-words font-medium">{task.title}</dd>
                  </>
                ) : null}
                <dt className="text-muted-foreground">URL</dt>
                <dd className="break-all">
                  <a href={task.url} target="_blank" rel="noreferrer" className="text-[#00aeec] hover:underline">
                    {task.url}
                  </a>
                </dd>
                <dt className="text-muted-foreground">{t.task.taskId}</dt>
                <dd className="font-mono text-xs">{task.id}</dd>
                <dt className="text-muted-foreground">{t.task.created}</dt>
                <dd>{formatTime(task.created_at)}</dd>
                <dt className="text-muted-foreground">{t.task.started}</dt>
                <dd>{formatTime(task.started_at)}</dd>
                <dt className="text-muted-foreground">{t.task.completed}</dt>
                <dd>{formatTime(task.completed_at) || "—"}</dd>
                {task.session_path ? (
                  <>
                    <dt className="text-muted-foreground">{t.task.session}</dt>
                    <dd className="break-all text-xs text-muted-foreground">{task.session_path}</dd>
                  </>
                ) : null}
              </dl>
            ) : (
              <div className="py-6 text-center text-sm text-muted-foreground">{t.task.loading}</div>
            )}
          </CardContent>
        </Card>

        {task?.status === "succeeded" && task.final_video_path ? (
          <Card>
            <CardHeader>
              <CardTitle>{t.task.finalVideo}</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <video
                key={task.id}
                src={finalVideoUrl(task.id)}
                controls
                preload="metadata"
                className="w-full rounded-md border border-emerald-200 bg-black"
              />
              <p className="break-all text-xs text-muted-foreground">{task.final_video_path}</p>
              <Button nativeButton={false} render={<a href={finalVideoDownloadUrl(task.id)} />}>
                <Download className="size-4" />
                {t.task.download}
              </Button>
            </CardContent>
          </Card>
        ) : null}

        <Card>
          <CardHeader>
            <CardTitle>{t.task.stages}</CardTitle>
          </CardHeader>
          <CardContent>
            {task ? (
              <ol className="grid gap-3">
                {task.stages.map((stage, index) => {
                  const stageProgress = normalizeProgress(stage.progress)
                  const canRerunStage = !isActive
                  const stageInfo = STAGE_INFO[stage.name]
                  return (
                    <li
                      key={stage.name}
                      className="flex items-start gap-3 rounded-lg border border-border bg-background px-4 py-3"
                    >
                      <div className="mt-0.5">{stageIcon(stage.status)}</div>
                      <div className="min-w-0 flex-1">
                        <div className="flex flex-wrap items-center gap-2">
                          <span className="text-xs text-muted-foreground">#{index + 1}</span>
                          <p className="font-medium">{stageLabel(stage.name, stage.label)}</p>
                          {stageInfo ? (
                            <button
                              type="button"
                              className="text-muted-foreground hover:text-foreground"
                              onClick={() => setInfoStageTarget(stage.name)}
                            >
                              <Info className="size-3.5" />
                            </button>
                          ) : null}
                          {CONFIGURABLE_STAGES.includes(stage.name) && canRerunStage ? (
                            <button
                              type="button"
                              className="text-muted-foreground hover:text-foreground"
                              onClick={() => openConfigDialog(stage.name)}
                            >
                              <Settings className="size-3.5" />
                            </button>
                          ) : null}
                          <Badge className={statusBadgeClass(stage.status)}>{statusLabel(stage.status)}</Badge>
                          {stage.started_at ? (
                            <span className="text-xs text-muted-foreground">
                              {durationOf(stage.started_at, stage.completed_at)}
                            </span>
                          ) : null}
                        </div>
                        <p className="mt-1 text-sm text-muted-foreground">
                          {stage.error_message || stage.last_message || t.common.waiting}
                        </p>
                        {stage.status === "running" && stageProgress !== null ? (
                          <div className="mt-2 flex items-center gap-3">
                            <Progress value={stageProgress} className="min-w-0 flex-1" />
                            <span className="w-10 text-right text-xs tabular-nums text-muted-foreground">
                              {stageProgress}%
                            </span>
                          </div>
                        ) : null}
                      </div>
                      {canRerunStage ? (
                        <div className="flex shrink-0 flex-row gap-1">
                          <Button
                            variant="ghost"
                            size="sm"
                            className="h-7 text-xs text-muted-foreground hover:text-foreground"
                            onClick={() => {
                              setRerunSingleTarget(stage.name)
                              setRerunSingleOpen(true)
                            }}
                          >
                            <RotateCw className="size-3" />
                            {t.task.rerunSingleStage}
                          </Button>
                          <Button
                            variant="ghost"
                            size="sm"
                            className="h-7 text-xs text-muted-foreground hover:text-foreground"
                            onClick={() => {
                              setRerunStageTarget(stage.name)
                              setRerunStageOpen(true)
                            }}
                          >
                            <RotateCw className="size-3" />
                            {t.task.rerunStage}
                          </Button>
                          <Button
                            variant="ghost"
                            size="sm"
                            className="h-7 text-xs text-muted-foreground hover:text-foreground"
                            onClick={() => {
                              setClearStageTarget(stage.name)
                              setClearStageOpen(true)
                            }}
                          >
                            <Eraser className="size-3" />
                            {t.task.clearStage}
                          </Button>
                        </div>
                      ) : null}
                    </li>
                  )
                })}
              </ol>
            ) : null}

            {task?.error_message ? (
              <div className="mt-4 rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">
                {task.error_message}
              </div>
            ) : null}
            {isFailed ? (
              <div className="mt-4 flex flex-col gap-2 rounded-lg border border-amber-200 bg-amber-50 px-3 py-3 sm:flex-row sm:items-center sm:justify-between">
                <p className="text-sm text-amber-800">
                  {isStopped ? t.task.stoppedNotice : t.task.resumeHelp}
                </p>
                <Button onClick={handleResume} disabled={resuming}>
                  {resuming ? <Loader2 className="size-4 animate-spin" /> : <Play className="size-4" />}
                  {resuming ? t.task.resuming : t.task.resumeTask}
                </Button>
              </div>
            ) : null}
            {resumeError ? (
              <div className="mt-2 rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">
                {resumeError}
              </div>
            ) : null}

            <Dialog open={rerunStageOpen} onOpenChange={setRerunStageOpen}>
              <DialogContent>
                <DialogHeader>
                  <DialogTitle>{t.task.rerunStageTitle}</DialogTitle>
                  <DialogDescription>
                    {t.task.rerunStageDescription}
                  </DialogDescription>
                </DialogHeader>
                {rerunStageError ? (
                  <div className="rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">
                    {rerunStageError}
                  </div>
                ) : null}
                <DialogFooter>
                  <DialogClose render={<Button variant="outline" disabled={rerunStaging} />}>
                    {t.common.cancel}
                  </DialogClose>
                  <Button onClick={handleRerunStage} disabled={rerunStaging}>
                    {rerunStaging ? <Loader2 className="size-4 animate-spin" /> : <RotateCw className="size-4" />}
                    {rerunStaging ? t.task.rerunningStage : t.task.confirmRerun}
                  </Button>
                </DialogFooter>
              </DialogContent>
            </Dialog>

            <Dialog open={rerunSingleOpen} onOpenChange={setRerunSingleOpen}>
              <DialogContent>
                <DialogHeader>
                  <DialogTitle>{t.task.rerunSingleStageTitle}</DialogTitle>
                  <DialogDescription>
                    {t.task.rerunSingleStageDescription}
                  </DialogDescription>
                </DialogHeader>
                {rerunSingleError ? (
                  <div className="rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">
                    {rerunSingleError}
                  </div>
                ) : null}
                <DialogFooter>
                  <DialogClose render={<Button variant="outline" disabled={rerunSingleIng} />}>
                    {t.common.cancel}
                  </DialogClose>
                  <Button onClick={handleRerunSingleStage} disabled={rerunSingleIng}>
                    {rerunSingleIng ? <Loader2 className="size-4 animate-spin" /> : <RotateCw className="size-4" />}
                    {rerunSingleIng ? t.task.rerunningSingleStage : t.task.confirmRerun}
                  </Button>
                </DialogFooter>
              </DialogContent>
            </Dialog>

            <Dialog open={clearStageOpen} onOpenChange={setClearStageOpen}>
              <DialogContent>
                <DialogHeader>
                  <DialogTitle>{t.task.clearStageTitle}</DialogTitle>
                  <DialogDescription>
                    {t.task.clearStageDescription}
                  </DialogDescription>
                </DialogHeader>
                {clearStageError ? (
                  <div className="rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">
                    {clearStageError}
                  </div>
                ) : null}
                <DialogFooter>
                  <DialogClose render={<Button variant="outline" disabled={clearingStage} />}>
                    {t.common.cancel}
                  </DialogClose>
                  <Button variant="destructive" onClick={handleClearStage} disabled={clearingStage}>
                    {clearingStage ? <Loader2 className="size-4 animate-spin" /> : <Eraser className="size-4" />}
                    {clearingStage ? t.common.loading : t.task.clearStage}
                  </Button>
                </DialogFooter>
              </DialogContent>
            </Dialog>

            <Dialog open={!!infoStageTarget} onOpenChange={(open) => { if (!open) setInfoStageTarget(null) }}>
              <DialogContent>
                <DialogHeader>
                  <DialogTitle>{t.task.stageInfoTitle}</DialogTitle>
                </DialogHeader>
                {infoStageTarget && STAGE_INFO[infoStageTarget] ? (
                  <div className="space-y-3 text-sm">
                    <div>
                      <p className="font-medium">{stageLabel(infoStageTarget)}</p>
                      <p className="mt-1 text-muted-foreground">{STAGE_INFO[infoStageTarget].description[language]}</p>
                    </div>
                    <div>
                      <p className="text-xs font-medium text-muted-foreground">Input</p>
                      <pre className="mt-1 rounded bg-muted p-2 font-mono text-xs">{STAGE_INFO[infoStageTarget].input[language].split("\n").map((line, i) => <span key={i}>{line}{"\n"}</span>)}</pre>
                    </div>
                    <div>
                      <p className="text-xs font-medium text-muted-foreground">Output</p>
                      <pre className="mt-1 rounded bg-muted p-2 font-mono text-xs">{STAGE_INFO[infoStageTarget].output[language].split("\n").map((line, i) => <span key={i}>{line}{"\n"}</span>)}</pre>
                    </div>
                  </div>
                ) : null}
                <DialogFooter>
                  <DialogClose render={<Button variant="outline" />}>
                    {t.common.close}
                  </DialogClose>
                </DialogFooter>
              </DialogContent>
            </Dialog>

            <Dialog open={!!configStageTarget} onOpenChange={(open) => { if (!open) { setConfigStageTarget(null); setConfigSuccess(false) } }}>
              <DialogContent>
                <DialogHeader>
                  <DialogTitle>{t.task.stageConfigTitle}</DialogTitle>
                </DialogHeader>
                {configStageTarget && STAGE_FIELDS[configStageTarget] ? (
                  <div className="space-y-4">
                    {STAGE_FIELDS[configStageTarget].map((field, i) => (
                      <div key={i}>
                        {field.render({
                          config: cfgConfig,
                          onChange: (patch) => setCfgConfig((prev) => ({ ...prev, ...patch })),
                          providerOptions,
                        })}
                      </div>
                    ))}
                  </div>
                ) : null}
                {configError ? (
                  <div className="rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">
                    {configError}
                  </div>
                ) : null}
                {configSuccess ? (
                  <div className="rounded-lg border border-green-200 bg-green-50 px-3 py-2 text-sm text-green-700">
                    {t.task.stageConfigSaved}
                  </div>
                ) : null}
                <DialogFooter>
                  <DialogClose render={<Button variant="outline" disabled={configSaving} />}>
                    {t.common.close}
                  </DialogClose>
                  <Button onClick={handleSaveConfig} disabled={configSaving}>
                    {configSaving ? <Loader2 className="size-4 animate-spin" /> : null}
                    {configSaving ? t.common.loading : t.settings.save}
                  </Button>
                </DialogFooter>
              </DialogContent>
            </Dialog>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between">
            <CardTitle>{t.task.runLog}</CardTitle>
            <FileText className="size-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <ScrollArea className="h-80 rounded-lg border bg-zinc-950 p-3 text-xs text-zinc-100">
              {log ? (
                <pre className="whitespace-pre-wrap break-words font-mono">{log}</pre>
              ) : (
                <p className="text-zinc-400">{t.task.emptyLog}</p>
              )}
            </ScrollArea>
          </CardContent>
        </Card>

        <Card className="border-red-200">
          <CardHeader>
            <CardTitle className="text-red-700">{t.task.dangerZone}</CardTitle>
          </CardHeader>
          <CardContent className="flex flex-col gap-4">
            <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
              <p className="text-sm text-muted-foreground">
                {t.task.rerunHelp}
              </p>
              <Dialog open={rerunOpen} onOpenChange={setRerunOpen}>
                <DialogTrigger
                  render={
                    <Button variant="outline" disabled={!task || isActive}>
                      <RotateCw className="size-4" />
                      {t.task.rerunTask}
                    </Button>
                  }
                />
                <DialogContent>
                  <DialogHeader>
                    <DialogTitle>{t.task.rerunTitle}</DialogTitle>
                    <DialogDescription>
                      {t.task.rerunDescription}
                    </DialogDescription>
                  </DialogHeader>
                  {rerunError ? (
                    <div className="rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">
                      {rerunError}
                    </div>
                  ) : null}
                  <DialogFooter>
                    <DialogClose render={<Button variant="outline" disabled={rerunning} />}>
                      {t.common.cancel}
                    </DialogClose>
                    <Button onClick={handleRerun} disabled={rerunning}>
                      {rerunning ? <Loader2 className="size-4 animate-spin" /> : <RotateCw className="size-4" />}
                      {rerunning ? t.task.rerunning : t.task.confirmRerun}
                    </Button>
                  </DialogFooter>
                </DialogContent>
              </Dialog>
            </div>
            <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
              <p className="text-sm text-muted-foreground">
                {t.task.deleteHelp} <code className="font-mono text-xs">workfolder/</code>
                {t.common.sentenceEnd}
              </p>
              <Dialog open={deleteOpen} onOpenChange={setDeleteOpen}>
                <DialogTrigger
                  render={
                    <Button variant="destructive" disabled={!task || isActive}>
                      <Trash2 className="size-4" />
                      {t.task.deleteTask}
                    </Button>
                  }
                />
                <DialogContent>
                  <DialogHeader>
                    <DialogTitle>{t.task.deleteTitle}</DialogTitle>
                    <DialogDescription>
                      {t.task.deleteDescription}
                    </DialogDescription>
                  </DialogHeader>
                  {deleteError ? (
                    <div className="rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">
                      {deleteError}
                    </div>
                  ) : null}
                  <DialogFooter>
                    <DialogClose render={<Button variant="outline" disabled={deleting} />}>
                      {t.common.cancel}
                    </DialogClose>
                    <Button variant="destructive" onClick={handleDelete} disabled={deleting}>
                      {deleting ? <Loader2 className="size-4 animate-spin" /> : <Trash2 className="size-4" />}
                      {deleting ? t.task.deleting : t.task.confirmDelete}
                    </Button>
                  </DialogFooter>
                </DialogContent>
              </Dialog>
            </div>
            {isActive ? (
              <p className="text-xs text-amber-600">{t.task.runningLocked}</p>
            ) : null}
          </CardContent>
        </Card>
      </div>
    </main>
  )
}
