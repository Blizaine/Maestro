import { useState } from 'react'
import { Check, ChevronUp, Loader2, Sparkles } from 'lucide-react'
import { SidebarMenu } from '../components/Sidebar/SidebarMenu'

/** Title-only AI controls. The ordinary Studio enhancer intentionally does not apply to Titles. */
export function TitleEnhancementControls({ enabled, defaultEnabled, busy, canEnhance, onToggle, onDefaultChange, onEnhanceNow }: {
  enabled: boolean
  defaultEnabled: boolean
  busy: boolean
  canEnhance: boolean
  onToggle: (enabled: boolean) => void
  onDefaultChange: (enabled: boolean) => void
  onEnhanceNow: () => void
}) {
  const [open, setOpen] = useState(false)
  const [anchor, setAnchor] = useState<HTMLDivElement | null>(null)

  return <div ref={setAnchor} className="relative flex items-center">
    <button type="button" onClick={onEnhanceNow} disabled={busy || !canEnhance}
      aria-label="Enhance title now" title="Enhance the title description with AI"
      className="min-h-8 rounded-l-md p-2 text-text-muted hover:text-accent-blue hover:bg-bg-hover disabled:opacity-40">
      {busy ? <Loader2 size={14} className="animate-spin" /> : <Sparkles size={14} />}
    </button>
    <button type="button" onClick={() => setOpen(value => !value)} disabled={busy}
      aria-label="Title enhancement options" aria-haspopup="menu" aria-expanded={open}
      className="min-h-8 rounded-r-md border-l border-border px-1.5 text-text-muted hover:text-accent-blue hover:bg-bg-hover disabled:opacity-40">
      <ChevronUp size={12}/>
    </button>
    <SidebarMenu open={open} anchor={anchor} label="Title enhancement options" onClose={() => setOpen(false)} width={240}>
      <button type="button" role="menuitem" disabled={busy || !canEnhance}
        onClick={() => { setOpen(false); onEnhanceNow() }}
        className="flex min-h-11 w-full items-center rounded-lg px-3 text-left text-xs text-text-secondary hover:bg-bg-hover disabled:opacity-40">
        Enhance now
      </button>
      <button type="button" role="menuitemcheckbox" aria-checked={enabled} disabled={busy}
        onClick={() => { onToggle(!enabled); setOpen(false) }}
        className="flex min-h-11 w-full items-center justify-between gap-2 rounded-lg px-3 text-left text-xs text-text-secondary hover:bg-bg-hover disabled:opacity-40">
        Enhance on generation
        <Check size={14} aria-hidden="true" className={enabled ? 'text-accent-blue' : 'invisible'}/>
      </button>
      <button type="button" role="menuitemcheckbox" aria-label="Use by default" aria-checked={defaultEnabled} disabled={busy}
        onClick={() => onDefaultChange(!defaultEnabled)}
        className="flex min-h-11 w-full items-start gap-2 rounded-lg border-t border-border px-3 py-2.5 text-left text-xs text-text-secondary hover:bg-bg-hover disabled:opacity-40">
        <span aria-hidden="true" className={`mt-0.5 flex h-3.5 w-3.5 shrink-0 items-center justify-center rounded border ${defaultEnabled ? 'border-accent-blue bg-accent-blue text-white' : 'border-text-muted'}`}>
          {defaultEnabled && <Check size={12}/>}
        </span>
        <span>Use by default<span className="mt-1 block text-[10px] leading-relaxed text-text-muted">Automatically enhance new Studio submissions.</span></span>
      </button>
    </SidebarMenu>
  </div>
}
