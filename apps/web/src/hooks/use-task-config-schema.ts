"use client"

import { useCallback, useEffect, useState } from "react"

import { getTaskConfigSchema, type TaskConfigSchema } from "@/lib/api"

/** 拉取并缓存后端任务配置 Schema，支持手动刷新（例如打开弹窗时同步最新提供商）。 */
export function useTaskConfigSchema() {
  const [schema, setSchema] = useState<TaskConfigSchema | null>(null)

  const reload = useCallback(async () => {
    try {
      const next = await getTaskConfigSchema()
      setSchema(next)
      return next
    } catch {
      return null
    }
  }, [])

  useEffect(() => {
    let cancelled = false
    getTaskConfigSchema()
      .then((next) => {
        if (!cancelled) setSchema(next)
      })
      .catch(() => {})
    return () => {
      cancelled = true
    }
  }, [])

  return { schema, reload }
}
