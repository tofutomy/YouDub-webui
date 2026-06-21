"use client"

import type { ReactNode } from "react"
import type { StageConfig, Task, TranslateProvider } from "@/lib/api"
import {
  AsrModelSelect,
  DirectionSelect,
  TranslateModeSelect,
  TranslateProviderSelect,
  ValidateTranslationCheckbox,
  TtsModeSelect,
  AddSubtitlesCheckbox,
} from "@/components/stage-config-fields"
import type { Direction } from "@/components/stage-config-fields"

export type { StageConfig }

export const DEFAULT_STAGE_CONFIG: StageConfig = {
  asr_model: "",
  asr_language: "en",
  target_language: "zh",
  add_subtitles: true,
  translate_mode: "sentence",
  validate_translation: false,
  tts_mode: "controllable_clone",
  translate_provider_id: "",
}

export function configFromTask(task: Task): StageConfig {
  const al = task.asr_language || "en"
  const tl = task.target_language || "zh"
  return {
    asr_model: task.asr_model || "",
    asr_language: al,
    target_language: tl,
    add_subtitles: task.add_subtitles !== 0,
    translate_mode: task.translate_mode || "sentence",
    validate_translation: task.validate_translation === 1,
    tts_mode: task.tts_mode || "controllable_clone",
    translate_provider_id: task.translate_provider_id || "",
  }
}

export function configForStage(stage: string, config: StageConfig): Partial<StageConfig> {
  const defs = STAGE_FIELDS[stage] || []
  const patch: Partial<StageConfig> = {}
  for (const def of defs) {
    for (const key of def.keys) {
      ;(patch as Record<string, unknown>)[key] = config[key]
    }
  }
  return patch
}

function directionFromConfig(config: StageConfig): Direction {
  if (config.asr_language === "zh" && config.target_language === "en") return "zh-en"
  return "en-zh"
}

export interface FieldRenderProps {
  config: StageConfig
  onChange: (patch: Partial<StageConfig>) => void
  providerOptions: TranslateProvider[]
}

export interface StageFieldDef {
  keys: (keyof StageConfig)[]
  render: (props: FieldRenderProps) => ReactNode
}

export const STAGE_FIELDS: Record<string, StageFieldDef[]> = {
  asr: [
    {
      keys: ["asr_model"],
      render: ({ config, onChange }) => (
        <AsrModelSelect
          value={config.asr_model}
          onChange={(v) => onChange({ asr_model: v })}
        />
      ),
    },
  ],
  translate: [
    {
      keys: ["translate_provider_id"],
      render: ({ config, onChange, providerOptions }) => (
        <TranslateProviderSelect
          value={config.translate_provider_id}
          options={providerOptions}
          onChange={(v) => onChange({ translate_provider_id: v })}
        />
      ),
    },
    {
      keys: ["asr_language", "target_language"],
      render: ({ config, onChange }) => (
        <DirectionSelect
          value={directionFromConfig(config)}
          onChange={(d) => {
            const [al, tl] = d.split("-")
            onChange({ asr_language: al, target_language: tl })
          }}
        />
      ),
    },
    {
      keys: ["translate_mode"],
      render: ({ config, onChange }) => (
        <TranslateModeSelect
          value={config.translate_mode}
          onChange={(v) => onChange({ translate_mode: v })}
        />
      ),
    },
    {
      keys: ["validate_translation"],
      render: ({ config, onChange }) => (
        <ValidateTranslationCheckbox
          checked={config.validate_translation}
          onChange={(v) => onChange({ validate_translation: v })}
        />
      ),
    },
  ],
  tts: [
    {
      keys: ["tts_mode"],
      render: ({ config, onChange }) => (
        <TtsModeSelect
          value={config.tts_mode}
          onChange={(v) => onChange({ tts_mode: v })}
        />
      ),
    },
  ],
  merge_video: [
    {
      keys: ["add_subtitles"],
      render: ({ config, onChange }) => (
        <AddSubtitlesCheckbox
          checked={config.add_subtitles}
          onChange={(v) => onChange({ add_subtitles: v })}
        />
      ),
    },
  ],
}

export const CONFIGURABLE_STAGES = Object.keys(STAGE_FIELDS)
