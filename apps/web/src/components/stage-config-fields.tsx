"use client"

import { useI18n } from "@/lib/i18n"
import { Label } from "@/components/ui/label"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"

/* ── ASR Model Select ── */

interface AsrModelSelectProps {
  id?: string
  value: string
  onChange: (value: string) => void
}

export function AsrModelSelect({ id = "asr-model", value, onChange }: AsrModelSelectProps) {
  const { t } = useI18n()
  return (
    <div className="space-y-2">
      <Label htmlFor={id}>{t.home.asrModelLabel}</Label>
      <Select value={value} onValueChange={(v) => onChange(v ?? "")}>
        <SelectTrigger id={id} className="h-10">
          <SelectValue placeholder={t.home.asrWhisperTurbo} />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value="">{t.home.asrWhisperTurbo}</SelectItem>
          <SelectItem value="whisper:large-v3">{t.home.asrWhisperLarge}</SelectItem>
          <SelectItem value="funasr:iic/SenseVoiceSmall">{t.home.asrSenseVoice}</SelectItem>
          <SelectItem value="funasr:FunAudioLLM/Fun-ASR-Nano-2512">{t.home.asrFunAsrNano}</SelectItem>
          <SelectItem value="qwen3asr:Qwen/Qwen3-ASR-1.7B">{t.home.asrQwen3Asr}</SelectItem>
        </SelectContent>
      </Select>
    </div>
  )
}

/* ── Direction Select ── */

type Direction = "en-zh" | "zh-en"

interface DirectionSelectProps {
  id?: string
  value: Direction
  onChange: (value: Direction) => void
}

export type { Direction }

export function DirectionSelect({ id = "direction", value, onChange }: DirectionSelectProps) {
  const { t } = useI18n()
  return (
    <div className="space-y-2">
      <Label htmlFor={id}>{t.home.localDirectionLabel}</Label>
      <Select value={value} onValueChange={(v) => onChange(v as Direction)}>
        <SelectTrigger id={id} className="h-10">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value="en-zh">{t.home.localEnZh}</SelectItem>
          <SelectItem value="zh-en">{t.home.localZhEn}</SelectItem>
        </SelectContent>
      </Select>
    </div>
  )
}

/* ── Translate Mode Select ── */

interface TranslateModeSelectProps {
  id?: string
  value: string
  onChange: (value: string) => void
}

export function TranslateModeSelect({ id = "translate-mode", value, onChange }: TranslateModeSelectProps) {
  const { t } = useI18n()
  return (
    <div className="space-y-2">
      <Label htmlFor={id}>{t.home.translateModeLabel}</Label>
      <Select value={value} onValueChange={(v) => onChange(v ?? "sentence")}>
        <SelectTrigger id={id} className="h-10">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value="sentence">{t.home.translateModeSentence}</SelectItem>
          <SelectItem value="batch">{t.home.translateModeBatch}</SelectItem>
        </SelectContent>
      </Select>
    </div>
  )
}

/* ── Validate Translation Checkbox ── */

interface ValidateTranslationCheckboxProps {
  checked: boolean
  onChange: (checked: boolean) => void
}

export function ValidateTranslationCheckbox({ checked, onChange }: ValidateTranslationCheckboxProps) {
  const { t } = useI18n()
  return (
    <label className="flex cursor-pointer items-center gap-2 text-sm">
      <input
        type="checkbox"
        className="size-4 rounded border-gray-300"
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
      />
      {t.home.validateTranslation}
    </label>
  )
}

/* ── TTS Mode Select ── */

interface TtsModeSelectProps {
  id?: string
  value: string
  onChange: (value: string) => void
}

export function TtsModeSelect({ id = "tts-mode", value, onChange }: TtsModeSelectProps) {
  const { t } = useI18n()
  return (
    <div className="space-y-2">
      <Label htmlFor={id}>{t.home.ttsModeLabel}</Label>
      <Select value={value} onValueChange={(v) => onChange(v ?? "controllable_clone")}>
        <SelectTrigger id={id} className="h-10">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value="controllable_clone">{t.home.ttsModeControllable}</SelectItem>
          <SelectItem value="hifi_clone">{t.home.ttsModeHifi}</SelectItem>
        </SelectContent>
      </Select>
      <p className="text-xs text-muted-foreground">
        {value === "hifi_clone" ? t.home.ttsModeHifiDesc : t.home.ttsModeControllableDesc}
      </p>
    </div>
  )
}

/* ── Add Subtitles Checkbox ── */

interface AddSubtitlesCheckboxProps {
  checked: boolean
  onChange: (checked: boolean) => void
}

export function AddSubtitlesCheckbox({ checked, onChange }: AddSubtitlesCheckboxProps) {
  const { t } = useI18n()
  return (
    <label className="flex cursor-pointer items-center gap-2 text-sm">
      <input
        type="checkbox"
        className="size-4 rounded border-gray-300"
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
      />
      {t.home.addSubtitles}
    </label>
  )
}
