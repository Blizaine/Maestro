import { Sparkles, X } from 'lucide-react'

/** Dismissing the enabled-only overlay changes the next submission, not the saved default. */
export function EnhancementBadge({ onRemove }: { onRemove: () => void }) {
  return <button type="button" onClick={onRemove} aria-label="Remove Enhance on generation"
    title="Skip enhancement for the next submission"
    className="studio-enhancement-badge absolute bottom-2 right-2 flex min-h-8 max-w-[calc(100%-16px)] items-center gap-1.5 rounded-lg border border-accent-blue/40 bg-bg-secondary px-2 text-[11px] text-accent-blue shadow-sm">
    <Sparkles size={13}/><span>Enhance on generation</span><X size={13}/>
  </button>
}
