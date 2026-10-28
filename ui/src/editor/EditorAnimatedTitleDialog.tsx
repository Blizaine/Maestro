import { useEffect, useState } from 'react'
import { AlertCircle, CheckCircle2, ExternalLink, Loader2, Sparkles, X } from 'lucide-react'
import { DEFAULT_TITLE_SETTINGS, titleFormState } from './editorAnimatedTitles'
import { AnimatedTitleControls } from './AnimatedTitleControls'
import type { AnimatedTitleSettings } from './editorAnimatedTitles'
import { useEditorDialog } from './useEditorDialog'
import { useEditorStore } from './useEditorStore'
import { useEditorTitleStore } from './useEditorTitleStore'

export function EditorAnimatedTitleDialog({ open, onClose }: { open: boolean; onClose: () => void }) {
  const dialog = useEditorDialog(open, onClose)
  const project = useEditorStore(state => state.project)
  const playhead = useEditorStore(state => state.playhead)
  const busy = useEditorTitleStore(state => state.busy)
  const intent = useEditorTitleStore(state => state.intent)
  const error = useEditorTitleStore(state => state.error)
  const submit = useEditorTitleStore(state => state.submit)
  const [settings, setSettings] = useState<AnimatedTitleSettings>({ ...DEFAULT_TITLE_SETTINGS })
  if (!open || !project) return null
  const patch = (value: Partial<AnimatedTitleSettings>) => setSettings(current => ({ ...current, ...value }))
  const blocked = busy || Boolean(intent && !['inserted', 'completed', 'failed'].includes(intent.status))
  const { validation } = titleFormState(settings, project.canvas)
  return <div ref={dialog} className="fixed inset-0 z-[170] grid place-items-center overflow-y-auto bg-black/65 p-3 backdrop-blur-sm"
    onClick={event => { if (event.target === event.currentTarget) onClose() }}>
    <section role="dialog" aria-modal="true" aria-labelledby="animated-title-heading" className="flex max-h-[calc(100dvh-1.5rem)] w-full max-w-[620px] flex-col overflow-hidden rounded-2xl border border-border bg-bg-secondary shadow-2xl">
      <header className="flex items-center gap-2 border-b border-border px-4 py-3">
        <Sparkles size={17} className="text-accent-blue" /><h2 id="animated-title-heading" className="flex-1 text-sm font-semibold">Create animated title</h2>
        <button type="button" onClick={onClose} aria-label="Close animated title" className="rounded-lg p-2 text-text-secondary hover:bg-bg-hover"><X size={17} /></button>
      </header>
      <div className="space-y-4 overflow-y-auto p-4">
        <AnimatedTitleControls settings={settings} patch={patch} busy={busy} canvas={project.canvas} at={playhead} />
        {error && <p role="alert" className="rounded-lg border border-red-500/25 bg-red-500/5 p-3 text-xs text-red-400">{error}</p>}
        {blocked && !busy && <p className="text-xs text-text-secondary">A title is already being tracked. Close this dialog to view its progress.</p>}
      </div>
      <footer className="flex items-center justify-end gap-2 border-t border-border p-3">
        <button type="button" onClick={onClose} className="rounded-lg border border-border px-3 py-2 text-xs text-text-secondary">Close</button>
        <button type="button" disabled={blocked || Boolean(validation)} onClick={() => void submit(settings)} title={validation || undefined}
          className="flex items-center gap-2 rounded-lg bg-accent-blue px-4 py-2 text-xs font-semibold text-white disabled:opacity-40">
          {busy ? <Loader2 size={14} className="animate-spin" /> : <Sparkles size={14} />}Generate title
        </button>
      </footer>
    </section>
  </div>
}

export function EditorAnimatedTitleStatus() {
  const intent = useEditorTitleStore(state => state.intent)
  const reconcile = useEditorTitleStore(state => state.reconcile)
  const insertReady = useEditorTitleStore(state => state.insertReady)
  const dismiss = useEditorTitleStore(state => state.dismiss)
  const projectId = useEditorStore(state => state.project?.id)
  const workspace = useEditorStore(state => state.project?.workspace)
  useEffect(() => {
    void reconcile()
    const timer = window.setInterval(() => void reconcile(), 5000)
    return () => window.clearInterval(timer)
  }, [reconcile, projectId, workspace])
  if (!intent) return null
  const terminal = ['inserted', 'completed', 'failed'].includes(intent.status)
  const readyHere = intent.destination !== 'studio' && intent.status === 'ready' && projectId === intent.projectId && workspace === intent.workspace
  return <div role="status" className="flex shrink-0 flex-wrap items-center gap-2 border-b border-accent-blue/20 bg-accent-blue/5 px-4 py-2 text-xs">
    {intent.status === 'failed' ? <AlertCircle size={14} className="text-red-400" /> : terminal || intent.status === 'ready' ? <CheckCircle2 size={14} className="text-accent-blue" /> : <Loader2 size={14} className="animate-spin text-accent-blue" />}
    <span className="min-w-0 flex-1 break-words"><strong>{intent.settings.text}</strong> · {intent.error || intent.message || 'Preparing title'}
      {intent.status === 'ready' && !readyHere && ' · Open the original project to add it.'}</span>
    {intent.asset && <a href={intent.asset.url} target="_blank" rel="noreferrer" className="flex items-center gap-1 rounded-lg border border-border px-2 py-1.5"><ExternalLink size={12} />Preview</a>}
    {readyHere && <button type="button" onClick={() => void insertReady()} className="rounded-lg bg-accent-blue px-2 py-1.5 text-white">Add to project</button>}
    {intent.status === 'uncertain' && <button type="button" onClick={() => void reconcile()} className="rounded-lg border border-border px-2 py-1.5">Check receipt</button>}
    <button type="button" onClick={dismiss} title={terminal ? 'Dismiss title status' : 'Stop tracking only; the generation job is not cancelled'} className="rounded-lg border border-border px-2 py-1.5">{terminal ? 'Dismiss' : 'Stop tracking'}</button>
  </div>
}
