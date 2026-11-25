"use client"

import { useId } from "react"

import type {
  LocalizedText,
  StageConfig,
  TaskConfigFieldSchema,
  TaskConfigOption,
} from "@/lib/api"
import { useI18n, type UiLanguage } from "@/lib/i18n"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"

/** 按当前界面语言提取多语言文案，缺失时回退中文或英文。 */
export function localizedText(
  text: LocalizedText | string | undefined,
  language: UiLanguage,
): string {
  if (!text) return ""
  if (typeof text === "string") return text
  return text[language] || text.zh || text.en || ""
}

/** 根据配置对象反推 composite_select 当前匹配的选项值。 */
export function resolveCompositeOption(
  field: TaskConfigFieldSchema,
  config: Record<string, unknown>,
): TaskConfigOption | undefined {
  const options = field.options || []
  // 优先做 patch 全字段精确匹配
  const exact = options.find((opt) => {
    if (!opt.patch) return false
    return Object.entries(opt.patch).every(([key, expected]) => config[key] === expected)
  })
  if (exact) return exact

  // 兼容 demucs_shifts 为非预设数字（例如 2 或 4）时按阈值归类
  if (field.id === "separate_mode") {
    const shifts = Number(config.demucs_shifts ?? 1)
    const fallbackValue = shifts >= 3 ? "high_quality" : "balanced"
    const matched = options.find((opt) => opt.value === fallbackValue)
    if (matched) return matched
  }

  const defaultVal = String(field.default ?? "")
  return options.find((opt) => opt.value === defaultVal) || options[0]
}

interface TaskConfigSchemaFieldProps {
  field: TaskConfigFieldSchema
  config: StageConfig
  onChange: (patch: Partial<StageConfig>) => void
}

/** 通用 Schema 字段渲染器：消费后端 /api/task-config/schema 自动渲染表单控件。 */
export function TaskConfigSchemaField({
  field,
  config,
  onChange,
}: TaskConfigSchemaFieldProps) {
  const { language } = useI18n()
  const domId = useId()
  const record = config as unknown as Record<string, unknown>
  const label = localizedText(field.label, language)
  const hint = localizedText(field.hint, language)

  if (field.type === "composite_select") {
    const options = field.options || []
    const current = resolveCompositeOption(field, record)
    const value = current?.value ?? String(field.default ?? "")
    const desc = localizedText(current?.description, language)
    const displayLabel = current ? localizedText(current.label, language) : value
    return (
      <div className="space-y-2">
        <Label htmlFor={domId}>{label}</Label>
        <Select
          value={value}
          onValueChange={(nextValue) => {
            const selected = options.find((opt) => opt.value === (nextValue ?? ""))
            if (selected?.patch) {
              onChange(selected.patch as Partial<StageConfig>)
            }
          }}
        >
          <SelectTrigger id={domId} className="h-10">
            <SelectValue>{displayLabel}</SelectValue>
          </SelectTrigger>
          <SelectContent>
            {options.map((opt) => (
              <SelectItem key={opt.value} value={opt.value}>
                {localizedText(opt.label, language)}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        {desc ? <p className="text-xs text-muted-foreground">{desc}</p> : null}
        {hint ? <p className="text-xs text-muted-foreground">{hint}</p> : null}
      </div>
    )
  }

  if (field.type === "select" && field.key) {
    const fieldKey = field.key
    const options = field.options || []
    const raw = record[fieldKey]
    const value = raw === undefined || raw === null ? String(field.default ?? "") : String(raw)
    const selected = options.find((opt) => opt.value === value)
    const displayLabel = selected ? localizedText(selected.label, language) : value
    const desc = localizedText(selected?.description, language)
    return (
      <div className="space-y-2">
        <Label htmlFor={domId}>{label}</Label>
        <Select
          value={value}
          onValueChange={(nextValue) => {
            onChange({ [fieldKey]: nextValue ?? String(field.default ?? "") } as Partial<StageConfig>)
          }}
        >
          <SelectTrigger id={domId} className="h-10">
            <SelectValue>{displayLabel}</SelectValue>
          </SelectTrigger>
          <SelectContent>
            {!selected && value ? (
              <SelectItem value={value}>{value}</SelectItem>
            ) : null}
            {options.map((opt) => (
              <SelectItem key={opt.value} value={opt.value}>
                {localizedText(opt.label, language)}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        {desc ? <p className="text-xs text-muted-foreground">{desc}</p> : null}
        {hint ? <p className="text-xs text-muted-foreground">{hint}</p> : null}
      </div>
    )
  }

  if (field.type === "boolean" && field.key) {
    const fieldKey = field.key
    const checked = Boolean(record[fieldKey] ?? field.default)
    return (
      <div className="space-y-1">
        <label className="flex cursor-pointer items-center gap-2 text-sm">
          <input
            type="checkbox"
            className="size-4 rounded border-gray-300"
            checked={checked}
            onChange={(event) =>
              onChange({ [fieldKey]: event.target.checked } as Partial<StageConfig>)
            }
          />
          {label}
        </label>
        {hint ? <p className="text-xs text-muted-foreground">{hint}</p> : null}
      </div>
    )
  }

  if (field.type === "textarea" && field.key) {
    const fieldKey = field.key
    const value = String(record[fieldKey] ?? field.default ?? "")
    return (
      <div className="space-y-2">
        <Label htmlFor={domId}>{label}</Label>
        <textarea
          id={domId}
          rows={field.rows ?? 3}
          maxLength={field.max_length}
          value={value}
          className="w-full rounded-md border bg-background p-2 text-sm"
          onChange={(event) =>
            onChange({ [fieldKey]: event.target.value } as Partial<StageConfig>)
          }
        />
        {hint ? <p className="text-xs text-muted-foreground">{hint}</p> : null}
      </div>
    )
  }

  if (field.type === "number" && field.key) {
    const fieldKey = field.key
    const value = record[fieldKey] ?? field.default ?? ""
    return (
      <div className="space-y-2">
        <Label htmlFor={domId}>{label}</Label>
        <Input
          id={domId}
          type="number"
          min={field.min}
          max={field.max}
          step={field.step}
          value={String(value)}
          onChange={(event) => {
            const parsed = Number(event.target.value)
            if (Number.isFinite(parsed)) {
              onChange({ [fieldKey]: parsed } as Partial<StageConfig>)
            }
          }}
        />
        {hint ? <p className="text-xs text-muted-foreground">{hint}</p> : null}
      </div>
    )
  }

  if (field.type === "text" && field.key) {
    const fieldKey = field.key
    const value = String(record[fieldKey] ?? field.default ?? "")
    return (
      <div className="space-y-2">
        <Label htmlFor={domId}>{label}</Label>
        <Input
          id={domId}
          maxLength={field.max_length}
          value={value}
          onChange={(event) =>
            onChange({ [fieldKey]: event.target.value } as Partial<StageConfig>)
          }
        />
        {hint ? <p className="text-xs text-muted-foreground">{hint}</p> : null}
      </div>
    )
  }

  return null
}
