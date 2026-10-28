import { useEffect, useState } from 'react'
import { CheckCircle2, ExternalLink, Loader2, Sparkles, X } from 'lucide-react'
import { animatedTitlePrompt, titleDimensions, titleFrames } from './editorAnimatedTitles'
import type { AnimatedTitleSettings } from './editorAnimatedTitles'
import { useEditorDialog } from './useEditorDialog'
import { useEditorStore } from './useEditorStore'
import { useEditorTitleStore } from './useEditorTitleStore'

const DEFAULTS: AnimatedTitleSettings = {
  text: '', look: 'Bold metallic 3D lettering', effect: 'dust', direction: '',
  seconds: 5.2, aspect: 'project', resolution: '720p', keepAudio: false,
}
const FIELD = 'w-full rounded-lg border border-border bg-bg-tertiary px-3 py-2 text-xs text-text-primary outline-none focus:border-accent-blue/60'

export function EditorAnimatedTitleDialog({ open, onClose }: { open: boolean; onClose: () => void }) {
  const dialog = useEditorDialog(open, onClose)
  const project = useEditorStore(state => state.project)
  const playhead = useEditorStore(state => state.playhead)
  const busy = useEditorTitleStore(state => state.busy)
  const intent = useEditorTitleStore(state => state.intent)
  const error = useEditorTitleStore(state => state.error)
  const submit = useEditorTitleStore(state => state.submit)
  const [settings, setSettings] = useState<AnimatedTitleSettings>({ ...DEFAULTS })
  if (!open || !project) return null
  const patch = (value: Partial<AnimatedTitleSettings>) => setSettings(current => ({ ...current, ...value }))
  const blocked = busy || Boolean(intent && !['inserted', 'failed'].includes(intent.status))
  let geometry = { width: 0, height: 0 }, prompt = '', duration = 0, validation = ''
  try { geometry = titleDimensions(settings, project.canvas); duration = titleFrames(settings.seconds) / 24; prompt = animatedTitlePrompt(settings) }
  catch (reason) { validation = reason instanceof Error ? reason.message : 'Check the title settings.' }
  return <div ref={dialog} className="fixed inset-0 z-[170] grid place-items-center overflow-y-auto bg-black/65 p-3 backdrop-blur-sm"
    onClick={event => { if (event.target === event.currentTarget) onClose() }}>
    <section role="dialog" aria-modal="true" aria-labelledby="animated-title-heading" className="flex max-h-[calc(100dvh-1.5rem)] w-full max-w-[620px] flex-col overflow-hidden rounded-2xl border border-border bg-bg-secondary shadow-2xl">
      <header className="flex items-center gap-2 border-b border-border px-4 py-3">
        <Sparkles size={17} className="text-accent-blue" /><h2 id="animated-title-heading" className="flex-1 text-sm font-semibold">Create animated title</h2>
        <button type="button" onClick={onClose} aria-label="Close animated title" className="rounded-lg p-2 text-text-secondary hover:bg-bg-hover"><X size={17} /></button>
      </header>
      <div className="space-y-4 overflow-y-auto p-4">
        <p className="text-xs leading-relaxed text-text-secondary">H3 Fused 4-Step · Frames. Matching green start/end frames are prepared automatically. The result returns above your footage with chroma key enabled.</p>
        <label className="block space-y-1.5 text-xs text-text-secondary">Title text
          <textarea aria-label="Animated title text" value={settings.text} maxLength={160} rows={2} disabled={busy}
            onChange={event => patch({ text: event.target.value })} placeholder="Superman vs Thanos" className={FIELD} />
        </label>
        <label className="block space-y-1.5 text-xs text-text-secondary">Lettering and visual style
          <input aria-label="Title visual style" value={settings.look} maxLength={1200} disabled={busy}
            onChange={event => patch({ look: event.target.value })} placeholder="Describe any style: anime, hand-drawn, metallic, minimal…" className={FIELD} />
        </label>
        <div className="grid grid-cols-2 gap-3">
          <label className="space-y-1.5 text-xs text-text-secondary">Animation
            <select aria-label="Title animation" value={settings.effect} disabled={busy} onChange={event => patch({ effect: event.target.value as AnimatedTitleSettings['effect'] })} className={FIELD}>
              <option value="dust">Dust dissolve</option><option value="dissolve">Soft reveal and dissolve</option><option value="neon">Neon trace</option><option value="impact">Impact and shatter</option>
            </select>
          </label>
          <label className="space-y-1.5 text-xs text-text-secondary">Duration (seconds)
            <input aria-label="Title duration" type="number" min={5} max={14.5} step={.1} value={settings.seconds} disabled={busy}
              onChange={event => patch({ seconds: Number(event.target.value) })} className={FIELD} />
          </label>
          <label className="space-y-1.5 text-xs text-text-secondary">Aspect ratio
            <select aria-label="Title aspect ratio" value={settings.aspect} disabled={busy} onChange={event => patch({ aspect: event.target.value as AnimatedTitleSettings['aspect'] })} className={FIELD}>
              <option value="project">Match project</option><option value="16:9">16:9 · landscape</option><option value="9:16">9:16 · portrait</option><option value="1:1">1:1 · square</option>
            </select>
          </label>
          <label className="space-y-1.5 text-xs text-text-secondary">Resolution
            <select aria-label="Title resolution" value={settings.resolution} disabled={busy} onChange={event => patch({ resolution: event.target.value as AnimatedTitleSettings['resolution'] })} className={FIELD}>
              <option value="480p">480p · faster</option><option value="720p">720p</option>
            </select>
          </label>
        </div>
        <label className="block space-y-1.5 text-xs text-text-secondary">Additional direction (optional)
          <textarea aria-label="Title additional direction" value={settings.direction} maxLength={1200} rows={2} disabled={busy}
            onChange={event => patch({ direction: event.target.value })} placeholder="Hold the complete title for two seconds before it blows away." className={FIELD} />
        </label>
        <label className="flex items-center gap-2 text-xs text-text-secondary"><input type="checkbox" checked={settings.keepAudio} disabled={busy}
          onChange={event => patch({ keepAudio: event.target.checked })} className="accent-[var(--color-accent-blue)]" />Keep generated clip audio</label>
        <div className="rounded-lg border border-border bg-bg-tertiary p-3 text-[11px] leading-relaxed text-text-muted">
          {geometry.width}×{geometry.height} · {duration ? `${duration.toFixed(2)}s at 24 fps` : 'Check duration'} · inserts at {playhead.toFixed(2)}s.<br />
          No reference characters or Studio settings are reused. Review generated spelling and fine-particle edges; generative results can vary.
        </div>
        {prompt && <details className="text-xs text-text-secondary"><summary className="cursor-pointer">View prepared prompt</summary><pre className="mt-2 whitespace-pre-wrap rounded-lg bg-bg-tertiary p-3 text-[11px] leading-relaxed">{prompt}</pre></details>}
        {error && <p role="alert" className="rounded-lg border border-red-500/25 bg-red-500/5 p-3 text-xs text-red-400">{error}</p>}
        {validation && (!geometry.width || !duration) && <p role="alert" className="text-xs text-red-400">{validation}</p>}
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
  const terminal = ['inserted', 'failed'].includes(intent.status)
  const readyHere = intent.status === 'ready' && projectId === intent.projectId && workspace === intent.workspace
  return <div role="status" className="flex shrink-0 flex-wrap items-center gap-2 border-b border-accent-blue/20 bg-accent-blue/5 px-4 py-2 text-xs">
    {terminal || intent.status === 'ready' ? <CheckCircle2 size={14} className="text-accent-blue" /> : <Loader2 size={14} className="animate-spin text-accent-blue" />}
    <span className="min-w-0 flex-1 break-words"><strong>{intent.settings.text}</strong> · {intent.error || intent.message || 'Preparing title'}
      {intent.status === 'ready' && !readyHere && ' · Open the original project to add it.'}</span>
    {intent.asset && <a href={intent.asset.url} target="_blank" rel="noreferrer" className="flex items-center gap-1 rounded-lg border border-border px-2 py-1.5"><ExternalLink size={12} />Preview</a>}
    {readyHere && <button type="button" onClick={() => void insertReady()} className="rounded-lg bg-accent-blue px-2 py-1.5 text-white">Add to project</button>}
    {intent.status === 'uncertain' && <button type="button" onClick={() => void reconcile()} className="rounded-lg border border-border px-2 py-1.5">Check receipt</button>}
    <button type="button" onClick={dismiss} title={terminal ? 'Dismiss title status' : 'Stop tracking only; the generation job is not cancelled'} className="rounded-lg border border-border px-2 py-1.5">{terminal ? 'Dismiss' : 'Stop tracking'}</button>
  </div>
}
