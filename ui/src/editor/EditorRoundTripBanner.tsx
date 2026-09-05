import { ArrowLeft, CheckCircle2, Loader2, Sparkles, X, XCircle } from 'lucide-react'
import { useStore } from '../stores/useStore'
import { useEditorStore } from './useEditorStore'

const TOOL_LABELS = {
  retake: 'Retake',
  edit_anything: 'Edit Anything',
  recast: 'Recast',
  repaint: 'Repaint',
  outpaint: 'Outpaint',
  upscale: 'Upscale',
  film_grain: 'Film Grain',
  revoice: 'Revoice',
} as const

export function EditorRoundTripBanner() {
  const sidebarMode = useStore(state => state.sidebarMode)
  const roundTrip = useEditorStore(state => state.roundTrip)
  const cancel = useEditorStore(state => state.cancelAIRoundTrip)
  const setSidebarMode = useStore(state => state.setSidebarMode)
  if (!roundTrip || sidebarMode === 'editor') return null

  const finished = roundTrip.status === 'completed' || roundTrip.status === 'failed'
  const failed = roundTrip.status === 'failed'
  return (
    <div className="fixed left-1/2 top-3 z-[260] flex w-[min(620px,calc(100vw-1rem))] -translate-x-1/2 items-center gap-2 rounded-xl border border-border bg-bg-secondary/95 px-3 py-2.5 shadow-2xl backdrop-blur-xl">
      {failed ? (
        <XCircle size={16} className="shrink-0 text-red-400" />
      ) : roundTrip.status === 'completed' ? (
        <CheckCircle2 size={16} className="shrink-0 text-emerald-400" />
      ) : roundTrip.status === 'armed' ? (
        <Sparkles size={16} className="shrink-0 text-accent-warm" />
      ) : (
        <Loader2 size={16} className="shrink-0 animate-spin text-accent-blue" />
      )}
      <div className="min-w-0 flex-1">
        <div className="truncate text-[11px] font-medium text-text-primary">
          {TOOL_LABELS[roundTrip.tool]} · {roundTrip.itemName}
        </div>
        <div className={`truncate text-[9px] ${failed ? 'text-red-300' : 'text-text-muted'}`}>
          {roundTrip.status === 'armed'
            ? 'Configure the edit and generate normally. The next job is linked to this Editor clip.'
            : roundTrip.status === 'queued'
              ? 'Queued — the result will return to Editor automatically.'
              : roundTrip.status === 'running'
                ? 'Generating — you can continue working while Maestro tracks the result.'
                : roundTrip.status === 'completed'
                  ? roundTrip.returnMode === 'replace'
                    ? 'The result replaced the timeline clip; the original remains available as a take.'
                    : 'The result was added as a non-destructive alternate take.'
                  : roundTrip.error || 'The linked generation did not complete.'}
        </div>
      </div>
      {finished && (
        <button type="button" onClick={() => setSidebarMode('editor')} className="flex shrink-0 items-center gap-1 rounded-lg bg-accent-blue/10 px-2.5 py-1.5 text-[9px] font-medium text-accent-blue hover:bg-accent-blue/20">
          <ArrowLeft size={11} /> Editor
        </button>
      )}
      <button type="button" onClick={cancel} className="shrink-0 rounded-lg p-1.5 text-text-muted hover:bg-bg-hover hover:text-text-primary" title={finished ? 'Dismiss' : 'Cancel automatic return'}>
        <X size={13} />
      </button>
    </div>
  )
}
