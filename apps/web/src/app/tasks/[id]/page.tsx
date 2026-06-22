"use client"

import { useRouter } from "next/navigation"
import { use, useMemo, useState } from "react"
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
  TranslateProvider,
  clearStageOutput,
  deleteTask,
  finalVideoDownloadUrl,
  finalVideoUrl,
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
import { ConfirmDialog } from "@/components/confirm-dialog"
import { StageConfigDialog } from "@/components/stage-config-dialog"
import { StageInfoDialog } from "@/components/stage-info-dialog"
import {
  CONFIGURABLE_STAGES,
  DEFAULT_STAGE_CONFIG,
  configForStage,
  configFromTask,
} from "@/lib/stage-config"
import { useTaskPolling } from "@/hooks/use-task-polling"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { DialogTrigger } from "@/components/ui/dialog"
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
  const { stageLabel, statusLabel, t } = useI18n()
  const { task, log, error, setTask, setLog } = useTaskPolling(id)
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
                  <ConfirmDialog
                    open={stopOpen}
                    onOpenChange={setStopOpen}
                    title={t.task.stopTitle}
                    description={t.task.stopDescription}
                    confirmLabel={t.task.confirmStop}
                    busyLabel={t.task.stopping}
                    busy={stopping}
                    error={stopError}
                    variant="destructive"
                    confirmIcon={<StopCircle className="size-4" />}
                    onConfirm={handleStop}
                  >
                    <DialogTrigger
                      render={
                        <Button variant="destructive" size="sm">
                          <StopCircle className="size-4" />
                          {t.task.stopTask}
                        </Button>
                      }
                    />
                  </ConfirmDialog>
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

            <ConfirmDialog
              open={rerunStageOpen}
              onOpenChange={setRerunStageOpen}
              title={t.task.rerunStageTitle}
              description={t.task.rerunStageDescription}
              confirmLabel={t.task.confirmRerun}
              busyLabel={t.task.rerunningStage}
              busy={rerunStaging}
              error={rerunStageError}
              confirmIcon={<RotateCw className="size-4" />}
              onConfirm={handleRerunStage}
            />

            <ConfirmDialog
              open={rerunSingleOpen}
              onOpenChange={setRerunSingleOpen}
              title={t.task.rerunSingleStageTitle}
              description={t.task.rerunSingleStageDescription}
              confirmLabel={t.task.confirmRerun}
              busyLabel={t.task.rerunningSingleStage}
              busy={rerunSingleIng}
              error={rerunSingleError}
              confirmIcon={<RotateCw className="size-4" />}
              onConfirm={handleRerunSingleStage}
            />

            <ConfirmDialog
              open={clearStageOpen}
              onOpenChange={setClearStageOpen}
              title={t.task.clearStageTitle}
              description={t.task.clearStageDescription}
              confirmLabel={t.task.clearStage}
              busyLabel={t.common.loading}
              busy={clearingStage}
              error={clearStageError}
              variant="destructive"
              confirmIcon={<Eraser className="size-4" />}
              onConfirm={handleClearStage}
            />

            <StageInfoDialog
              stageName={infoStageTarget}
              onClose={() => setInfoStageTarget(null)}
            />

            <StageConfigDialog
              stageName={configStageTarget}
              config={cfgConfig}
              onConfigChange={(patch) => setCfgConfig((prev) => ({ ...prev, ...patch }))}
              providerOptions={providerOptions}
              saving={configSaving}
              success={configSuccess}
              error={configError}
              onClose={() => { setConfigStageTarget(null); setConfigSuccess(false) }}
              onSave={handleSaveConfig}
            />
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
              <ConfirmDialog
                open={rerunOpen}
                onOpenChange={setRerunOpen}
                title={t.task.rerunTitle}
                description={t.task.rerunDescription}
                confirmLabel={t.task.confirmRerun}
                busyLabel={t.task.rerunning}
                busy={rerunning}
                error={rerunError}
                confirmIcon={<RotateCw className="size-4" />}
                onConfirm={handleRerun}
              >
                <DialogTrigger
                  render={
                    <Button variant="outline" disabled={!task || isActive}>
                      <RotateCw className="size-4" />
                      {t.task.rerunTask}
                    </Button>
                  }
                />
              </ConfirmDialog>
            </div>
            <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
              <p className="text-sm text-muted-foreground">
                {t.task.deleteHelp} <code className="font-mono text-xs">workfolder/</code>
                {t.common.sentenceEnd}
              </p>
              <ConfirmDialog
                open={deleteOpen}
                onOpenChange={setDeleteOpen}
                title={t.task.deleteTitle}
                description={t.task.deleteDescription}
                confirmLabel={t.task.confirmDelete}
                busyLabel={t.task.deleting}
                busy={deleting}
                error={deleteError}
                variant="destructive"
                confirmIcon={<Trash2 className="size-4" />}
                onConfirm={handleDelete}
              >
                <DialogTrigger
                  render={
                    <Button variant="destructive" disabled={!task || isActive}>
                      <Trash2 className="size-4" />
                      {t.task.deleteTask}
                    </Button>
                  }
                />
              </ConfirmDialog>
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
