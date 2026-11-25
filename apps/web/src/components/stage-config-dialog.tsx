"use client"

import { Loader2 } from "lucide-react"

import type { StageConfig, TaskConfigSchema, TranslateProvider } from "@/lib/api"
import { useI18n } from "@/lib/i18n"
import { STAGE_FIELDS } from "@/lib/stage-config"
import { TaskConfigSchemaField } from "@/components/task-config-schema-field"
import {
  Dialog,
  DialogClose,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { Button } from "@/components/ui/button"

interface StageConfigDialogProps {
  stageName: string | null
  config: StageConfig
  onConfigChange: (patch: Partial<StageConfig>) => void
  providerOptions: TranslateProvider[]
  schema?: TaskConfigSchema | null
  saving: boolean
  success: boolean
  error: string
  onClose: () => void
  onSave: () => void
}

export function StageConfigDialog({
  stageName,
  config,
  onConfigChange,
  providerOptions,
  schema,
  saving,
  success,
  error,
  onClose,
  onSave,
}: StageConfigDialogProps) {
  const { t } = useI18n()
  const schemaFields = stageName
    ? schema?.stages.find((item) => item.stage === stageName)?.fields
    : undefined
  const fields = stageName ? STAGE_FIELDS[stageName] : null
  return (
    <Dialog
      open={!!stageName}
      onOpenChange={(open) => { if (!open) { onClose() } }}
    >
      <DialogContent>
        <DialogHeader>
          <DialogTitle>{t.task.stageConfigTitle}</DialogTitle>
        </DialogHeader>
        {stageName && schemaFields && schemaFields.length > 0 ? (
          <div className="space-y-4">
            {schemaFields.map((field) => (
              <div key={field.id}>
                <TaskConfigSchemaField
                  field={field}
                  config={config}
                  onChange={onConfigChange}
                />
              </div>
            ))}
          </div>
        ) : stageName && fields ? (
          <div className="space-y-4">
            {fields.map((field, i) => (
              <div key={i}>
                {field.render({
                  config,
                  onChange: onConfigChange,
                  providerOptions,
                })}
              </div>
            ))}
          </div>
        ) : null}
        {error ? (
          <div className="rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">
            {error}
          </div>
        ) : null}
        {success ? (
          <div className="rounded-lg border border-green-200 bg-green-50 px-3 py-2 text-sm text-green-700">
            {t.task.stageConfigSaved}
          </div>
        ) : null}
        <DialogFooter>
          <DialogClose render={<Button variant="outline" disabled={saving} />}>
            {t.common.close}
          </DialogClose>
          <Button onClick={onSave} disabled={saving}>
            {saving ? <Loader2 className="size-4 animate-spin" /> : null}
            {saving ? t.common.loading : t.settings.save}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
