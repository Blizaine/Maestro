import { useEffect, useRef } from 'react'
import { ArrowLeft, Loader2, Sparkles, X } from 'lucide-react'
import { Sidebar } from '../components/Sidebar/Sidebar'
import { GlobalQueuePopover } from '../components/GlobalQueuePopover'
import { useEditorStore } from './useEditorStore'
import { useStore } from '../stores/useStore'
import * as api from '../api/client'

const LABELS = { retake: 'Retake', edit_anything: 'Edit Anything', recast: 'Recast', repaint: 'Repaint', outpaint: 'Outpaint', upscale: 'Upscale', film_grain: 'Film Grain', revoice: 'Revoice' }

export function EditorAIWorkspace() {
  const open = useEditorStore(state => state.aiWorkspaceOpen)
  const operation = useEditorStore(state => state.roundTrip)
  const setOpen = useEditorStore(state => state.setAIWorkspaceOpen)
  const dismiss = useEditorStore(state => state.cancelAIRoundTrip)
  const job = useStore(state => state.jobs.find(candidate => candidate.id === operation?.claimedJobId))
  const close = useRef<HTMLButtonElement>(null)
  useEffect(() => {
    if (!open) return
    const previous = document.activeElement as HTMLElement | null
    close.current?.focus()
    return () => previous?.focus()
  }, [open])
  if (!operation) return null
  if (!open) return (
    <button type="button" onClick={() => setOpen(true)}
      className="absolute right-4 top-16 z-[80] flex max-w-[min(360px,90vw)] items-center gap-2 rounded-xl border border-accent-blue/30 bg-bg-secondary px-3 py-2 text-xs shadow-xl"
      aria-label="Open Editor AI operation"><Sparkles size={15} className="text-accent-blue" />
      <span className="truncate">{LABELS[operation.tool]} · {operation.status === 'armed' ? 'Configure edit' : job?.phase || operation.status}</span>
    </button>
  )
  const configuring = operation.status === 'armed'
  const failed = operation.status === 'failed'
  const completed = operation.status === 'completed'
  return <aside aria-label="Editor AI workspace" data-testid="editor-ai-workspace"
    className="absolute inset-y-0 right-0 z-[120] flex w-[min(460px,100%)] min-w-0 flex-col border-l border-border bg-bg-secondary shadow-2xl"
    onKeyDown={event => { if (event.key === 'Escape') { event.preventDefault(); event.stopPropagation(); setOpen(false) } }}>
    <header className="flex shrink-0 items-center gap-2 border-b border-border px-4 py-3">
      <Sparkles size={17} className="shrink-0 text-accent-blue" />
      <div className="min-w-0 flex-1"><h2 className="truncate text-sm font-semibold">{LABELS[operation.tool]}</h2>
        <p className="truncate text-xs text-text-muted">{operation.itemName}</p></div>
      <GlobalQueuePopover />
      <button ref={close} type="button" onClick={() => setOpen(false)} aria-label="Close AI configuration" className="rounded-lg p-2 text-text-secondary hover:bg-bg-hover"><X size={18} /></button>
    </header>
    <div className="shrink-0 space-y-1 border-b border-border px-4 py-2 text-[11px] text-text-secondary">
      <div>Source range: {operation.sourceRange.start.toFixed(2)}–{operation.sourceRange.end.toFixed(2)}s</div>
      <div>{operation.returnMode === 'alternate' ? 'Result becomes an alternate take.' : 'Replace the clip and keep its original take.'}</div>
    </div>
    {configuring ? <div className="min-h-0 flex-1"><Sidebar embedded /></div> : (
      <div className="flex min-h-0 flex-1 flex-col gap-4 overflow-y-auto p-4">
        <div role="status" className={`rounded-xl border p-4 ${failed ? 'border-red-500/30 bg-red-500/5' : 'border-border bg-bg-tertiary'}`}>
          {!failed && !completed && <Loader2 size={20} className="mb-3 animate-spin text-accent-blue" />}
          <h3 className="text-sm font-medium">{completed ? 'Take returned to the Editor' : failed ? 'This edit needs attention' : job?.phase || 'Queued for generation'}</h3>
          <p className="mt-2 break-words text-xs leading-relaxed text-text-secondary">{operation.error || job?.message || (completed ? 'Review the new take in the clip inspector. The original is still available.' : 'You can continue editing while this exact job is tracked.')}</p>
          {job && !failed && !completed && <progress max={100} value={job.progress} className="mt-3 h-2 w-full accent-[var(--color-accent-blue)]" />}
        </div>
        {operation.outputFile && <a className="rounded-lg border border-border px-3 py-2 text-xs text-accent-blue" href={api.getFileUrl(operation.outputFile, operation.workspace)} target="_blank" rel="noreferrer">Open returned media</a>}
        <button type="button" onClick={() => setOpen(false)} className="flex items-center justify-center gap-2 rounded-lg bg-accent-blue px-3 py-2.5 text-xs font-semibold text-white"><ArrowLeft size={14} />Continue editing</button>
        {(completed || failed) && <button type="button" onClick={dismiss} className="rounded-lg border border-border px-3 py-2 text-xs text-text-secondary">Dismiss operation</button>}
      </div>
    )}
  </aside>
}
