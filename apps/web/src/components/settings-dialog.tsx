"use client"

import { FormEvent, useEffect, useState } from "react"
import { Eye, EyeOff, Plus, RefreshCw, Settings, Trash2, ChevronDown, ChevronRight, Star } from "lucide-react"

import {
  TranslateProvider,
  createTranslateProvider,
  deleteTranslateProvider,
  getCookieInfo,
  getFunasrSettings,
  getTranslateProviderModels,
  getTranslateProviders,
  getYtdlpSettings,
  saveCookie,
  saveFunasrSettings,
  saveOpenAISettings,
  saveYtdlpSettings,
  updateTranslateProvider,
} from "@/lib/api"
import { LANGUAGE_OPTIONS, useI18n } from "@/lib/i18n"
import { Button } from "@/components/ui/button"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { Textarea } from "@/components/ui/textarea"

type SettingsForm = {
  cookie: string
  translateConcurrency: string
  proxyPort: string
  useVllm: "auto" | "on" | "off"
}

type ProviderForm = {
  id: string | null  // null = new unsaved
  name: string
  baseUrl: string
  apiKey: string
  model: string
  isDefault: boolean
  hasApiKey: boolean
  expanded: boolean
  apiKeyDirty: boolean
  showApiKey: boolean
  modelsLoaded: boolean
  modelsLoading: boolean
  modelOptions: string[]
}

const SAVED_COOKIE_SENTINEL = "__YOUDUB_SAVED_COOKIE__"

type MessageKey = "keySaved" | "saved"

const defaultSettings: SettingsForm = {
  cookie: "",
  translateConcurrency: "50",
  useVllm: "auto",
  proxyPort: "",
}

function uniqueModels(models: string[]) {
  return Array.from(new Set(models.map((model) => model.trim()).filter(Boolean)))
}

function providerToForm(p: TranslateProvider): ProviderForm {
  return {
    id: p.id,
    name: p.name,
    baseUrl: p.base_url,
    apiKey: p.api_key || "",
    model: p.model,
    isDefault: p.is_default,
    hasApiKey: p.has_api_key,
    expanded: false,
    apiKeyDirty: false,
    showApiKey: false,
    modelsLoaded: false,
    modelsLoading: false,
    modelOptions: p.model ? [p.model] : [],
  }
}

function newProviderForm(baseUrl: string): ProviderForm {
  return {
    id: null,
    name: "",
    baseUrl: baseUrl || "https://api.openai.com/v1",
    apiKey: "",
    model: "",
    isDefault: false,
    hasApiKey: false,
    expanded: true,
    apiKeyDirty: false,
    showApiKey: false,
    modelsLoaded: false,
    modelsLoading: false,
    modelOptions: [],
  }
}

export function SettingsDialog() {
  const { language, loadedModelsText, setLanguage, t } = useI18n()
  const [open, setOpen] = useState(false)
  const [settings, setSettings] = useState(defaultSettings)
  const [providers, setProviders] = useState<ProviderForm[]>([])
  const [message, setMessage] = useState("")
  const [messageKey, setMessageKey] = useState<MessageKey | null>(null)
  const [cookieDirty, setCookieDirty] = useState(false)

  const visibleMessage =
    messageKey === "keySaved" ? t.settings.keySaved : messageKey === "saved" ? t.settings.saved : message

  const cookieValue =
    settings.cookie === SAVED_COOKIE_SENTINEL ? t.settings.savedCookie : settings.cookie

  useEffect(() => {
    if (!open) return
    Promise.all([getCookieInfo(), getTranslateProviders(), getYtdlpSettings(), getFunasrSettings()])
      .then(([cookie, providersResp, ytdlp, funasr]) => {
        setSettings({
          cookie: cookie.exists ? SAVED_COOKIE_SENTINEL : "",
          translateConcurrency: "50",
          proxyPort: ytdlp.proxy_port,
          useVllm: (funasr.use_vllm as "auto" | "on" | "off") || "auto",
        })
        setProviders(providersResp.providers.map(providerToForm))
        setCookieDirty(false)
        setMessage("")
        setMessageKey(null)
      })
      .catch((err) => {
        setMessageKey(null)
        setMessage(err.message)
      })
  }, [open])

  function updateProvider(index: number, updates: Partial<ProviderForm>) {
    setProviders((current) =>
      current.map((p, i) => (i === index ? { ...p, ...updates } : p))
    )
  }

  async function fetchProviderModels(index: number) {
    const provider = providers[index]
    updateProvider(index, { modelsLoading: true, modelsLoaded: false })
    try {
      const id = provider.id
      if (!id) {
        // New provider: use inline values
        setMessage(t.settings.saveProviderFirst)
        updateProvider(index, { modelsLoading: false })
        return
      }
      const resp = await getTranslateProviderModels(id)
      const models = uniqueModels([provider.model, ...resp.models])
      updateProvider(index, {
        modelOptions: models,
        modelsLoaded: true,
        modelsLoading: false,
        model: provider.model || models[0] || "",
      })
      setMessage(models.length ? loadedModelsText(models.length) : t.settings.noModels)
    } catch (err) {
      setMessage(err instanceof Error ? err.message : t.settings.loadModelsError)
      updateProvider(index, { modelsLoading: false })
    }
  }

  async function saveProvider(index: number) {
    const provider = providers[index]
    try {
      if (provider.id) {
        // Update existing
        const resp = await updateTranslateProvider(provider.id, {
          name: provider.name,
          base_url: provider.baseUrl,
          api_key: provider.apiKeyDirty ? provider.apiKey : "",
          clear_api_key: provider.apiKeyDirty && !provider.apiKey.trim(),
          model: provider.model,
          is_default: provider.isDefault,
        })
        updateProvider(index, {
          id: resp.id,
          hasApiKey: resp.has_api_key,
          apiKey: resp.api_key || "",
          apiKeyDirty: false,
        })
      } else {
        // Create new
        const resp = await createTranslateProvider({
          name: provider.name || provider.model || "Untitled",
          base_url: provider.baseUrl,
          api_key: provider.apiKey,
          model: provider.model,
          is_default: provider.isDefault,
        })
        updateProvider(index, {
          id: resp.id,
          hasApiKey: resp.has_api_key,
          apiKey: resp.api_key || "",
          apiKeyDirty: false,
          name: resp.name,
        })
      }
    } catch (err) {
      throw err
    }
  }

  async function removeProvider(index: number) {
    const provider = providers[index]
    if (provider.id) {
      await deleteTranslateProvider(provider.id)
    }
    setProviders((current) => current.filter((_, i) => i !== index))
  }

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    setMessage("")
    setMessageKey(null)
    try {
      if (cookieDirty) {
        await saveCookie(settings.cookie)
      }
      // Save all providers
      for (let i = 0; i < providers.length; i++) {
        await saveProvider(i)
      }
      // Save global translate concurrency
      await saveOpenAISettings({
        base_url: "",
        api_key: "",
        model: "",
        translate_concurrency: settings.translateConcurrency,
      })
      await saveYtdlpSettings({ proxy_port: settings.proxyPort })
      await saveFunasrSettings({ use_vllm: settings.useVllm })
      // Reload providers to get fresh state
      const providersResp = await getTranslateProviders()
      setProviders(providersResp.providers.map(providerToForm))
      setMessageKey("saved")
      setSettings((current) => ({
        ...current,
        cookie: cookieDirty ? (SAVED_COOKIE_SENTINEL) : current.cookie,
        proxyPort: settings.proxyPort,
        useVllm: settings.useVllm,
      }))
      setCookieDirty(false)
    } catch (err) {
      setMessageKey(null)
      setMessage(err instanceof Error ? err.message : t.settings.saveError)
    }
  }

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger render={<Button variant="outline" />}>
        <Settings className="size-4" />
        {t.settings.button}
      </DialogTrigger>
      <DialogContent className="max-h-[calc(100dvh-2rem)] overflow-hidden sm:max-w-2xl">
        <form onSubmit={submit} className="flex max-h-[calc(100dvh-4rem)] min-h-0 flex-col">
          <DialogHeader className="shrink-0 pr-8">
            <DialogTitle>{t.settings.title}</DialogTitle>
            <DialogDescription>{t.settings.description}</DialogDescription>
          </DialogHeader>
          <div className="mt-4 min-h-0 overflow-y-auto pr-1">
            <div className="grid gap-4 pb-4">
              <div className="grid gap-2">
                <Label htmlFor="uiLanguage">{t.settings.language}</Label>
                <Select
                  value={language}
                  onValueChange={(value) => {
                    if (value === "en" || value === "zh") setLanguage(value)
                  }}
                >
                  <SelectTrigger id="uiLanguage">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {LANGUAGE_OPTIONS.map((option) => (
                      <SelectItem key={option.value} value={option.value}>
                        {option.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <div className="grid gap-2">
                <Label htmlFor="cookie">{t.settings.cookie}</Label>
                <Textarea
                  id="cookie"
                  value={cookieValue}
                  onFocus={(event) => {
                    if (!cookieDirty && settings.cookie === SAVED_COOKIE_SENTINEL) {
                      event.currentTarget.select()
                    }
                  }}
                  onChange={(event) => {
                    setCookieDirty(true)
                    setSettings((current) => ({
                      ...current,
                      cookie:
                        current.cookie === SAVED_COOKIE_SENTINEL
                          ? event.target.value.replace(t.settings.savedCookie, "")
                          : event.target.value,
                    }))
                  }}
                  placeholder={t.settings.cookiePlaceholder}
                  className="min-h-44 max-h-[42dvh] overflow-auto font-mono text-xs leading-relaxed"
                />
              </div>
              <div className="grid gap-2">
                <Label htmlFor="proxyPort">{t.settings.proxyPort}</Label>
                <Input
                  id="proxyPort"
                  inputMode="numeric"
                  value={settings.proxyPort}
                  onChange={(event) =>
                    setSettings((current) => ({ ...current, proxyPort: event.target.value }))
                  }
                  placeholder="7890"
                />
              </div>
              <div className="grid gap-2">
                <Label htmlFor="useVllm">{t.settings.funasrUseVllm}</Label>
                <Select
                  value={settings.useVllm}
                  onValueChange={(value) => {
                    if (value === "auto" || value === "on" || value === "off") {
                      setSettings((current) => ({ ...current, useVllm: value }))
                    }
                  }}
                >
                  <SelectTrigger id="useVllm">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="auto">{t.settings.funasrVllmAuto}</SelectItem>
                    <SelectItem value="on">{t.settings.funasrVllmOn}</SelectItem>
                    <SelectItem value="off">{t.settings.funasrVllmOff}</SelectItem>
                  </SelectContent>
                </Select>
                <p className="text-xs text-muted-foreground">{t.settings.funasrUseVllmHelp}</p>
              </div>

              {/* ── Translate Providers ── */}
              <div className="grid gap-3">
                <div className="flex items-center justify-between">
                  <Label>{t.settings.translateProviders}</Label>
                  <Button
                    type="button"
                    variant="outline"
                    size="sm"
                    onClick={() => {
                      const defaultBase = providers.find((p) => p.isDefault)?.baseUrl || providers[0]?.baseUrl || providers[0]?.baseUrl || providers[0]?.baseUrl || ""
                      setProviders((current) => [...current, newProviderForm(defaultBase)])
                    }}
                  >
                    <Plus className="size-3" />
                    {t.settings.addProvider}
                  </Button>
                </div>
                {providers.length === 0 && (
                  <p className="text-xs text-muted-foreground">{t.settings.noProviders}</p>
                )}
                {providers.map((provider, index) => (
                  <div key={provider.id ?? `new-${index}`} className="rounded-lg border">
                    <button
                      type="button"
                      className="flex w-full items-center gap-2 px-3 py-2 text-left text-sm hover:bg-accent"
                      onClick={() => updateProvider(index, { expanded: !provider.expanded })}
                    >
                      {provider.expanded ? <ChevronDown className="size-4" /> : <ChevronRight className="size-4" />}
                      <span className="flex-1 font-medium truncate">
                        {provider.name || t.settings.unnamedProvider}
                      </span>
                      <span className="text-muted-foreground truncate">{provider.model}</span>
                      {provider.isDefault && (
                        <Star className="size-3 fill-yellow-400 text-yellow-400" />
                      )}
                    </button>
                    {provider.expanded && (
                      <div className="grid gap-3 border-t px-3 py-3">
                        <div className="grid gap-2">
                          <Label>{t.settings.providerName}</Label>
                          <Input
                            value={provider.name}
                            onChange={(e) => updateProvider(index, { name: e.target.value })}
                            placeholder={t.settings.providerNamePlaceholder}
                          />
                        </div>
                        <div className="grid gap-2">
                          <Label>{t.settings.baseUrl}</Label>
                          <Input
                            value={provider.baseUrl}
                            onChange={(e) => updateProvider(index, { baseUrl: e.target.value })}
                          />
                        </div>
                        <div className="grid gap-2">
                          <Label>{t.settings.apiKey}</Label>
                          <div className="relative">
                            <Input
                              key={`provider-key-${index}-${provider.showApiKey}`}
                              type={provider.showApiKey ? "text" : "password"}
                              value={provider.apiKey}
                              onFocus={(event) => {
                                if (!provider.apiKeyDirty) {
                                  event.currentTarget.select()
                                }
                              }}
                              onChange={(event) => {
                                updateProvider(index, {
                                  apiKeyDirty: true,
                                  apiKey: event.target.value,
                                })
                              }}
                              placeholder={t.settings.apiKeyPlaceholder}
                              className="pr-9"
                            />
                            <Button
                              type="button"
                              variant="ghost"
                              size="icon-sm"
                              className="absolute top-0.5 right-0.5"
                              onClick={() => updateProvider(index, { showApiKey: !provider.showApiKey })}
                            >
                              {provider.showApiKey ? <EyeOff className="size-4" /> : <Eye className="size-4" />}
                            </Button>
                          </div>
                        </div>
                        <div className="grid gap-2 sm:grid-cols-[1fr_auto]">
                          <div className="grid gap-2">
                            <Label>{t.settings.model}</Label>
                            {provider.modelsLoaded && provider.modelOptions.length > 0 ? (
                              <Select
                                value={provider.model}
                                onValueChange={(value) => updateProvider(index, { model: value || "" })}
                              >
                                <SelectTrigger>
                                  <SelectValue placeholder={t.settings.selectModel} />
                                </SelectTrigger>
                                <SelectContent>
                                  {provider.modelOptions.map((m) => (
                                    <SelectItem key={m} value={m}>{m}</SelectItem>
                                  ))}
                                </SelectContent>
                              </Select>
                            ) : (
                              <Input
                                value={provider.model}
                                onChange={(e) => updateProvider(index, { model: e.target.value })}
                              />
                            )}
                          </div>
                          <div className="grid gap-2 sm:self-end">
                            <Button
                              type="button"
                              variant="secondary"
                              onClick={() => fetchProviderModels(index)}
                              disabled={provider.modelsLoading || !provider.baseUrl.trim()}
                            >
                              <RefreshCw className="size-4" />
                              {provider.modelsLoading ? t.settings.loading : t.settings.getModels}
                            </Button>
                          </div>
                        </div>
                        <div className="flex items-center justify-between">
                          <label className="flex cursor-pointer items-center gap-2 text-sm">
                            <input
                              type="checkbox"
                              className="size-4 rounded border-gray-300"
                              checked={provider.isDefault}
                              onChange={(e) => {
                                const checked = e.target.checked
                                setProviders((current) =>
                                  current.map((p, i) => ({
                                    ...p,
                                    isDefault: i === index ? checked : (checked ? false : p.isDefault),
                                  }))
                                )
                              }}
                            />
                            {t.settings.setDefaultProvider}
                          </label>
                          <Button
                            type="button"
                            variant="destructive"
                            size="sm"
                            onClick={() => removeProvider(index)}
                          >
                            <Trash2 className="size-3" />
                            {t.settings.deleteProvider}
                          </Button>
                        </div>
                      </div>
                    )}
                  </div>
                ))}
              </div>

              <div className="grid gap-2">
                <Label htmlFor="translateConcurrency">{t.settings.translateConcurrency}</Label>
                <Input
                  id="translateConcurrency"
                  inputMode="numeric"
                  value={settings.translateConcurrency}
                  onChange={(event) =>
                    setSettings((current) => ({
                      ...current,
                      translateConcurrency: event.target.value.replace(/[^0-9]/g, ""),
                    }))
                  }
                  placeholder="50"
                />
                <p className="text-xs text-muted-foreground">
                  {t.settings.concurrencyHelp}
                </p>
              </div>
              {visibleMessage ? <p className="text-sm text-muted-foreground">{visibleMessage}</p> : null}
            </div>
          </div>
          <DialogFooter className="shrink-0">
            <Button type="submit">{t.settings.save}</Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  )
}
