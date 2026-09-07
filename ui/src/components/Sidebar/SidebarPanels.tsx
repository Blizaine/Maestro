/* eslint-disable react-refresh/only-export-components -- shared sidebar presentation contexts */
import { createContext, useContext, useEffect, useRef, useState, type ReactNode, type RefObject } from 'react'
import { createPortal } from 'react-dom'
import { Maximize2, Minimize2, X } from 'lucide-react'
import { H3MultiWindowControls } from './H3MultiWindowControls'

export const CharacterToolbarContext = createContext<HTMLElement | null>(null)
export function CharacterToolbarItem({ children }: { children: ReactNode }) {
  const target = useContext(CharacterToolbarContext)
  return target ? createPortal(children, target) : children
}

// Keep the same editor mounted when expanding, including its reviewed window
// plan, selection and enhancement state. Only the presentation changes.
export const ComposerContext = createContext<{ expanded: boolean; expand: () => void } | null>(null)
const ComposerToolbarContext = createContext<HTMLElement | null>(null)
export function ComposerToolbarItem({ children }: { children: ReactNode }) {
  const target = useContext(ComposerToolbarContext)
  return target ? createPortal(children, target) : children
}

export function usePanelFocus(open: boolean, ref: RefObject<HTMLDivElement | null>, close: () => void) {
  const closeRef = useRef(close)
  useEffect(() => { closeRef.current = close }, [close])
  useEffect(() => {
    if (!open) return
    const previous = document.activeElement as HTMLElement | null
    const panel = ref.current
    panel?.focus({ preventScroll: true })
    const handleKey = (event: KeyboardEvent) => {
      if (!panel?.contains(event.target as Node)) return
      if (event.key === 'Escape') { event.preventDefault(); event.stopPropagation(); closeRef.current() }
      if (event.key !== 'Tab') return
      const nodes = Array.from(panel.querySelectorAll<HTMLElement>('button:not(:disabled), input:not(:disabled), textarea:not(:disabled), select:not(:disabled), a[href], summary, [tabindex="0"]'))
        .filter(node => node.getClientRects().length > 0 && !node.closest('[inert]'))
      const first = nodes[0], last = nodes[nodes.length - 1]
      if (!first) { event.preventDefault(); return }
      if (event.shiftKey && (document.activeElement === first || document.activeElement === panel)) { event.preventDefault(); last.focus() }
      else if (!event.shiftKey && (document.activeElement === last || document.activeElement === panel)) { event.preventDefault(); first.focus() }
    }
    panel?.addEventListener('keydown', handleKey)
    return () => {
      panel?.removeEventListener('keydown', handleKey)
      if (previous?.isConnected) previous.focus({ preventScroll: true })
    }
  }, [open, ref])
}

export function SidebarDialog({ open, title, onClose, children }: { open: boolean; title: string; onClose: () => void; children: ReactNode }) {
  const ref = useRef<HTMLDivElement>(null)
  usePanelFocus(open, ref, onClose)
  return createPortal(
    <div hidden={!open} className={open ? 'fixed inset-0 z-[55] flex items-center justify-center bg-black/50 p-2 sm:p-6' : 'hidden'}
      onClick={event => { if (event.target === event.currentTarget) onClose() }}>
      <div ref={ref} role="dialog" aria-modal="true" aria-label={title} tabIndex={-1}
        className="flex max-h-[90dvh] w-full max-w-xl min-w-0 flex-col overflow-hidden rounded-2xl border border-border bg-bg-secondary shadow-2xl outline-none">
        <div className="flex shrink-0 items-center justify-between gap-3 border-b border-border px-4 py-3">
          <h2 className="text-sm font-semibold text-text-primary">{title}</h2>
          <button type="button" aria-label={`Close ${title}`} onClick={onClose} className="rounded-lg p-2 text-text-secondary hover:bg-bg-hover"><X size={18}/></button>
        </div>
        <div className="min-h-0 overflow-y-auto overscroll-contain p-3 sm:p-4">{children}</div>
      </div>
    </div>, document.body,
  )
}

export function PromptDock({ children }: { children: ReactNode }) {
  const [expanded, setExpanded] = useState(false)
  const [enhanceSlot, setEnhanceSlot] = useState<HTMLDivElement | null>(null)
  const panel = useRef<HTMLDivElement>(null)
  usePanelFocus(expanded, panel, () => setExpanded(false))
  return (
    <ComposerContext.Provider value={{ expanded, expand: () => setExpanded(true) }}>
    <ComposerToolbarContext.Provider value={enhanceSlot}>
      <div ref={panel} tabIndex={expanded ? -1 : undefined} role={expanded ? 'dialog' : undefined}
        aria-modal={expanded ? true : undefined} aria-label={expanded ? 'Expanded prompt editor' : 'Prompt composer'}
        data-expanded={expanded} className={`studio-composer outline-none ${expanded ? 'fixed inset-0 z-[100] flex flex-col bg-bg-secondary p-4 sm:p-8' : 'shrink-0 border-t border-border bg-bg-secondary px-3 pt-2 pb-2'}`}>
        <div className="mb-2 flex shrink-0 flex-wrap items-center gap-2">
          <span className="mr-auto text-xs font-medium text-text-primary">{expanded ? 'Prompt editor' : 'Prompt'}</span>
          <H3MultiWindowControls section="prompt" compact />
          <div ref={setEnhanceSlot} className="relative empty:hidden"/>
          <button type="button" aria-label={expanded ? 'Collapse prompt editor' : 'Expand prompt editor'}
            onMouseDown={event => { if (document.activeElement?.tagName === 'TEXTAREA') event.preventDefault() }}
            onClick={() => setExpanded(value => !value)} title={expanded ? 'Return to the sidebar (Escape)' : 'Write and review long scripts'}
            className="rounded-lg p-2 text-text-secondary hover:bg-bg-hover hover:text-text-primary">
            {expanded ? <Minimize2 size={15}/> : <Maximize2 size={15}/>}
          </button>
        </div>
        <div className={expanded ? 'min-h-0 flex-1 overflow-y-auto overscroll-contain' : 'studio-composer-content min-h-0'}>{children}</div>
        {expanded && <div className="flex shrink-0 justify-end border-t border-border pt-3 mt-3">
          <button type="button" onClick={() => setExpanded(false)} className="min-h-10 rounded-xl bg-accent-blue px-5 text-xs text-white">Done</button>
        </div>}
      </div>
    </ComposerToolbarContext.Provider>
    </ComposerContext.Provider>
  )
}
