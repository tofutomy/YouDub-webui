"use client"

import { ReactNode } from "react"
import { Loader2 } from "lucide-react"

import { useI18n } from "@/lib/i18n"
import { Button } from "@/components/ui/button"
import {
  Dialog,
  DialogClose,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"

type ConfirmVariant = "default" | "destructive"

interface ConfirmDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  title: string
  description?: string
  confirmLabel: string
  cancelLabel?: string
  busyLabel?: string
  busy: boolean
  error?: string
  variant?: ConfirmVariant
  confirmIcon?: ReactNode
  onConfirm: () => void
  /** Optional trigger (e.g. DialogTrigger) rendered inside the Dialog. */
  children?: ReactNode
}

/**
 * Generic confirmation dialog used by the task detail page for
 * delete / rerun / stop / rerun-stage / clear-stage actions.
 */
export function ConfirmDialog({
  open,
  onOpenChange,
  title,
  description,
  confirmLabel,
  cancelLabel,
  busyLabel,
  busy,
  error,
  variant = "default",
  confirmIcon,
  onConfirm,
  children,
}: ConfirmDialogProps) {
  const { t } = useI18n()
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      {children}
      <DialogContent>
        <DialogHeader>
          <DialogTitle>{title}</DialogTitle>
          {description ? <DialogDescription>{description}</DialogDescription> : null}
        </DialogHeader>
        {error ? (
          <div className="rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">
            {error}
          </div>
        ) : null}
        <DialogFooter>
          <DialogClose render={<Button variant="outline" disabled={busy} />}>
            {cancelLabel ?? t.common.cancel}
          </DialogClose>
          <Button variant={variant} onClick={onConfirm} disabled={busy}>
            {busy ? <Loader2 className="size-4 animate-spin" /> : confirmIcon}
            {busy ? (busyLabel ?? t.common.loading) : confirmLabel}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
