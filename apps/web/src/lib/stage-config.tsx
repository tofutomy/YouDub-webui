"use client"

import type { ReactNode } from "react"
import type { StageConfig, Task, TranslateProvider } from "@/lib/api"
import {
  AsrModelSelect,
  DirectionSelect,
  SeparateModeSelect,
  TranslateModeSelect,
  TranslateProviderSelect,
  ValidateTranslationCheckbox,
  StopAfterTranslateCheckbox,
  TtsModeSelect,
  AddSubtitlesCheckbox,
  FilterFillersCheckbox,
} from "@/components/stage-config-fields"
import {
  DEFAULT_ASR_LANGUAGE,
  DEFAULT_TARGET_LANGUAGE,
  directionFromConfig,
  directionToLanguages,
} from "@/lib/directions"

export type { StageConfig }

export const DEFAULT_STAGE_CONFIG: StageConfig = {
  asr_model: "",
  asr_language: DEFAULT_ASR_LANGUAGE,
  target_language: DEFAULT_TARGET_LANGUAGE,
  add_subtitles: true,
  translate_mode: "sentence",
  validate_translation: false,
  stop_after_translate: false,
  filter_fillers: false,
  tts_mode: "controllable_clone",
  translate_provider_id: "",
  demucs_model: "",
  demucs_shifts: 1,
}

export function configFromTask(task: Task): StageConfig {
  const al = task.asr_language || DEFAULT_ASR_LANGUAGE
  const tl = task.target_language || DEFAULT_TARGET_LANGUAGE
  return {
    asr_model: task.asr_model || "",
    asr_language: al,
    target_language: tl,
    add_subtitles: task.add_subtitles !== 0,
    translate_mode: task.translate_mode || "sentence",
    validate_translation: task.validate_translation === 1,
    stop_after_translate: task.stop_after_translate === 1,
    filter_fillers: task.filter_fillers === 1,
    tts_mode: task.tts_mode || "controllable_clone",
    translate_provider_id: task.translate_provider_id || "",
    demucs_model: task.demucs_model || "",
    demucs_shifts: task.demucs_shifts ?? 1,
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
  separate: [
    {
      keys: ["demucs_model", "demucs_shifts"],
      render: ({ config, onChange }) => (
        <SeparateModeSelect
          demucs_shifts={config.demucs_shifts}
          onChange={onChange}
        />
      ),
    },
  ],
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
  asr_fix: [
    {
      keys: ["filter_fillers"],
      render: ({ config, onChange }) => (
        <FilterFillersCheckbox
          checked={config.filter_fillers}
          onChange={(v) => onChange({ filter_fillers: v })}
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
            const { asr_language, target_language } = directionToLanguages(d)
            onChange({ asr_language, target_language })
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
    {
      keys: ["stop_after_translate"],
      render: ({ config, onChange }) => (
        <StopAfterTranslateCheckbox
          checked={config.stop_after_translate}
          onChange={(v) => onChange({ stop_after_translate: v })}
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
