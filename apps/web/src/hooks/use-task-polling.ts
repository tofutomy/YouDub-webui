"use client"

import { useEffect, useState } from "react"

import { Task, getTask, getTaskLog } from "@/lib/api"

/**
 * Polls a task and its log every `intervalMs` milliseconds.
 *
 * Returns the latest task, log text, and any load error. The polling
 * auto-stops when the component unmounts.
 */
export function useTaskPolling(taskId: string, intervalMs = 2000) {
  const [task, setTask] = useState<Task | null>(null)
  const [log, setLog] = useState("")
  const [error, setError] = useState("")

  useEffect(() => {
    let cancelled = false
    const load = async () => {
      try {
        const next = await getTask(taskId)
        if (cancelled) return
        setTask(next)
        const logText = await getTaskLog(taskId)
        if (cancelled) return
        setLog(logText)
      } catch (err) {
        if (!cancelled) setError(err instanceof Error ? err.message : "Failed to load task")
      }
    }
    load()
    const interval = window.setInterval(load, intervalMs)
    return () => {
      cancelled = true
      window.clearInterval(interval)
    }
  }, [taskId, intervalMs])

  return { task, log, error, setTask, setLog, setError }
}
