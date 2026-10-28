import { useEffect, useMemo, useState } from 'react'
import { Loader2, Sparkles } from 'lucide-react'
import { useStore } from '../stores/useStore'
import { AnimatedTitleControls } from './AnimatedTitleControls'
import { EditorAnimatedTitleStatus } from './EditorAnimatedTitleDialog'
import { DEFAULT_TITLE_SETTINGS, TITLE_EFFECTS, titleFormState } from './editorAnimatedTitles'
import type { AnimatedTitleSettings } from './editorAnimatedTitles'
import { useEditorTitleStore } from './useEditorTitleStore'

const DRAFT = 'maestro-studio-title-draft-v1'
const CANVAS = { width: 1920, height: 1080 }
function normalizeDraft(raw: unknown): AnimatedTitleSettings {
  const defaults: AnimatedTitleSettings = { ...DEFAULT_TITLE_SETTINGS, aspect: '16:9' }
    if (!raw || typeof raw !== 'object' || Array.isArray(raw)) return defaults
    const value = raw as Partial<AnimatedTitleSettings>
    return {
      ...defaults,
      text: typeof value.text === 'string' ? value.text.slice(0, 160) : defaults.text,
      look: typeof value.look === 'string' ? value.look.slice(0, 1200) : defaults.look,
      direction: typeof value.direction === 'string' ? value.direction.slice(0, 1200) : defaults.direction,
      effect: typeof value.effect === 'string' && (value.effect === 'custom' || Object.hasOwn(TITLE_EFFECTS, value.effect)) ? value.effect : defaults.effect,
      seconds: typeof value.seconds === 'number' && Number.isFinite(value.seconds) ? Math.max(5, Math.min(14.5, value.seconds)) : defaults.seconds,
      aspect: value.aspect === '16:9' || value.aspect === '9:16' || value.aspect === '1:1' ? value.aspect : defaults.aspect,
      resolution: value.resolution === '480p' || value.resolution === '720p' ? value.resolution : defaults.resolution,
      background: value.background === 'black' ? 'black' : 'chroma',
    }
}
function loadDraft(): AnimatedTitleSettings {
  try { return normalizeDraft(JSON.parse(localStorage.getItem(DRAFT) || 'null')) }
  catch { return normalizeDraft(null) }
}

/** Studio titles use the same fields and isolated generation payload as Editor titles. */
export function StudioTitlesPanel() {
  const [draft, setDraft] = useState(loadDraft)
  const workspace = useStore(state => state.activeWorkspace)
  const restored = useStore(state => state.params._studio_title_settings)
  const busy = useEditorTitleStore(state => state.busy)
  const intent = useEditorTitleStore(state => state.intent)
  const error = useEditorTitleStore(state => state.error)
  const submit = useEditorTitleStore(state => state.submitStudio)
  const loadedSettings = useMemo(() => restored ? normalizeDraft(restored) : null, [restored])
  const settings = loadedSettings ?? draft
  useEffect(() => {
    try { localStorage.setItem(DRAFT, JSON.stringify(settings)) } catch { /* Submission separately requires durable storage. */ }
  }, [settings])
  const { validation } = titleFormState(settings, CANVAS)
  const blocked = busy || Boolean(intent && !['inserted', 'completed', 'failed'].includes(intent.status))
  const patch = (value: Partial<AnimatedTitleSettings>) => {
    setDraft({ ...settings, ...value })
    if (restored) useStore.setState(state => state.params._studio_title_settings === restored
      ? { params: { ...state.params, _studio_title_settings: undefined } } : state)
  }
  return <section aria-label="Studio Titles" className="mx-3 mb-3 min-w-0 space-y-4 rounded-2xl border border-border bg-bg-primary/30 p-3">
    <div className="flex items-center gap-2 text-sm font-semibold text-text-primary"><Sparkles size={16} className="text-accent-blue" />Titles</div>
    <p className="break-words text-xs text-text-secondary">Save to <strong>{workspace}</strong> · No Editor project needed.</p>
    <EditorAnimatedTitleStatus />
    <AnimatedTitleControls settings={settings} patch={patch} busy={busy} canvas={CANVAS} destination="studio" />
    {error && <p role="alert" className="text-xs text-red-400">{error}</p>}
    {blocked && !busy && <p className="text-xs text-text-secondary">A title is already being tracked. Its progress is shown above.</p>}
    <button type="button" disabled={blocked || Boolean(validation)} onClick={() => void submit(settings)} title={validation || undefined}
      className="flex min-h-11 w-full items-center justify-center gap-2 rounded-lg bg-accent-blue px-4 py-2 text-sm font-semibold text-white disabled:opacity-40">
      {busy ? <Loader2 size={15} className="animate-spin" /> : <Sparkles size={15} />}Generate title
    </button>
  </section>
}
