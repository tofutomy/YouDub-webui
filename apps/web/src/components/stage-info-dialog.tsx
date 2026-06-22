"use client"

import { useI18n, STAGE_INFO } from "@/lib/i18n"
import {
  Dialog,
  DialogClose,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { Button } from "@/components/ui/button"

interface StageInfoDialogProps {
  stageName: string | null
  onClose: () => void
}

export function StageInfoDialog({ stageName, onClose }: StageInfoDialogProps) {
  const { language, stageLabel, t } = useI18n()
  const info = stageName ? STAGE_INFO[stageName] : null
  return (
    <Dialog open={!!stageName} onOpenChange={(open) => { if (!open) onClose() }}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>{t.task.stageInfoTitle}</DialogTitle>
        </DialogHeader>
        {stageName && info ? (
          <div className="space-y-3 text-sm">
            <div>
              <p className="font-medium">{stageLabel(stageName)}</p>
              <p className="mt-1 text-muted-foreground">{info.description[language]}</p>
            </div>
            <div>
              <p className="text-xs font-medium text-muted-foreground">Input</p>
              <pre className="mt-1 rounded bg-muted p-2 font-mono text-xs">
                {info.input[language].split("\n").map((line, i) => <span key={i}>{line}{"\n"}</span>)}
              </pre>
            </div>
            <div>
              <p className="text-xs font-medium text-muted-foreground">Output</p>
              <pre className="mt-1 rounded bg-muted p-2 font-mono text-xs">
                {info.output[language].split("\n").map((line, i) => <span key={i}>{line}{"\n"}</span>)}
              </pre>
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
  )
}
