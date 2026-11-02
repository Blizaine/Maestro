/* eslint-disable react-refresh/only-export-components -- shared sidebar presentation contexts */
import { createContext, useContext, useEffect, useLayoutEffect, useId, useRef, useState, type CSSProperties, type ReactNode, type RefObject } from 'react'
import { createPortal } from 'react-dom'
import { X } from 'lucide-react'
import { useIsMobileSidecar } from '../../lib/useIsMobile'
import { getSidebarDialogPosition } from './sidebarDialogGeometry'

export const CharacterToolbarContext = createContext<HTMLElement | null>(null)
export const SidebarLayoutContext = createContext<{ sidebar: HTMLElement | null; settings: HTMLElement | null } | null>(null)
export function CharacterToolbarItem({ children }: { children: ReactNode }) {
  const target = useContext(CharacterToolbarContext)
  return target ? createPortal(<div className="studio-character-trigger">{children}</div>, target) : children
}

// Keep the same editor mounted when expanding, including its reviewed window
// plan, selection and enhancement state. Only the presentation changes.
export const ComposerContext = createContext<{ expanded: boolean; expand: () => void } | null>(null)
const ComposerToolbarContext = createContext<HTMLElement | null>(null)
export function ComposerToolbarItem({ children }: { children: ReactNode }) {
  const target = useContext(ComposerToolbarContext)
  return target ? createPortal(children, target) : children
}

export function usePanelFocus(open: boolean, ref: RefObject<HTMLDivElement | null>, close: () => void, trapFocus = true) {
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
      if (event.key !== 'Tab' || !trapFocus) return
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
      if (previous?.isConnected && (panel?.contains(document.activeElement) || document.activeElement === document.body)) previous.focus({ preventScroll: true })
    }
  }, [open, ref, trapFocus])
}

export function SidebarDialog({ open, title, onClose, children, variant = 'center', id, headerStart, footer, closeLabel, anchor, fixedHeight, hideHeader = false }: {
  open: boolean; title: string; onClose: () => void; children: ReactNode
  variant?: 'center' | 'settings' | 'library' | 'workflow'; id?: string
  headerStart?: ReactNode; footer?: ReactNode; closeLabel?: string
  anchor?: HTMLElement | null
  /** Reserve space for changing controls without moving the anchored panel. */
  fixedHeight?: number
  hideHeader?: boolean
}) {
  const ref = useRef<HTMLDivElement>(null)
  const scrollRef = useRef<HTMLDivElement>(null)
  const layout = useContext(SidebarLayoutContext)
  const panelKey = useId()
  const closeRef = useRef(onClose)
  const [position, setPosition] = useState<CSSProperties>({})
  const isMobileSidecar = useIsMobileSidecar()
  const anchored = !!layout && variant !== 'center'
  const popover = anchored && variant !== 'library'
  useEffect(() => { closeRef.current = onClose }, [onClose])
  useEffect(() => {
    if (!open || !anchored) return
    // Settings and libraries share one overlay at a time. Nested media previews
    // use the centered variant and return to their parent panel when closed.
    const eventName = 'maestro-studio-panel-open'
    const dismiss = (event: Event) => {
      if ((event as CustomEvent<string>).detail !== panelKey) closeRef.current()
    }
    window.dispatchEvent(new CustomEvent(eventName, { detail: panelKey }))
    window.addEventListener(eventName, dismiss)
    return () => window.removeEventListener(eventName, dismiss)
  }, [open, anchored, panelKey])
  useEffect(() => {
    if (!open || !popover) return
    const outside = (event: PointerEvent) => {
      const target = event.target as Node
      if (!ref.current?.contains(target) && !layout?.settings?.contains(target) && !anchor?.contains(target)) closeRef.current()
    }
    document.addEventListener('pointerdown', outside)
    return () => document.removeEventListener('pointerdown', outside)
  }, [open, popover, layout, anchor])
  useLayoutEffect(() => {
    if (!open || !anchored) return
    const measure = () => {
      const viewport = window.visualViewport
      const top = viewport?.offsetTop || 0
      const height = viewport?.height || window.innerHeight
      const sidebar = layout?.sidebar?.getBoundingClientRect()
      const settingsBounds = layout?.settings?.getBoundingClientRect()
      const triggerBounds = anchor?.getBoundingClientRect()
      setPosition(getSidebarDialogPosition({
        variant,
        viewportTop: top,
        viewportHeight: height,
        windowWidth: window.innerWidth,
        windowHeight: window.innerHeight,
        mobile: isMobileSidecar,
        sidebar,
        settings: settingsBounds ? { top: settingsBounds.top } : null,
        trigger: triggerBounds,
        fixedHeight,
      }))
    }
    measure()
    // The drawer applies keyboard viewport changes in React. Measure after
    // that commit so an anchored popup follows its button's new position.
    let frame = 0
    const scheduleMeasure = () => {
      cancelAnimationFrame(frame)
      frame = requestAnimationFrame(measure)
    }
    const observer = new ResizeObserver(scheduleMeasure)
    for (const element of [anchor, layout?.sidebar, layout?.settings]) {
      if (element) observer.observe(element)
    }
    window.addEventListener('resize', scheduleMeasure)
    window.visualViewport?.addEventListener('resize', scheduleMeasure)
    window.visualViewport?.addEventListener('scroll', scheduleMeasure)
    return () => {
      cancelAnimationFrame(frame)
      observer.disconnect()
      window.removeEventListener('resize', scheduleMeasure)
      window.visualViewport?.removeEventListener('resize', scheduleMeasure)
      window.visualViewport?.removeEventListener('scroll', scheduleMeasure)
    }
  }, [open, anchored, isMobileSidecar, layout, variant, anchor, fixedHeight])
  useLayoutEffect(() => {
    if (!open || variant !== 'settings' || !isMobileSidecar) return
    const panel = ref.current
    const scroller = scrollRef.current
    if (!panel || !scroller) return

    let frame = 0
    const revealFocusedInput = () => {
      if (!isMobileSidecar) return
      const input = document.activeElement
      if (!(input instanceof HTMLElement) || !panel.contains(input)
        || !input.matches('textarea, input:not([type="range"]):not([type="checkbox"]):not([type="file"]), [contenteditable="true"]')) return

      const bounds = scroller.getBoundingClientRect()
      const field = input.getBoundingClientRect()
      const visibleHeight = Math.min(field.height, Math.max(0, scroller.clientHeight - 16))
      const delta = field.top < bounds.top + 8 ? field.top - bounds.top - 8
        : field.top + visibleHeight > bounds.bottom - 8 ? field.top + visibleHeight - bounds.bottom + 8 : 0
      if (Math.abs(delta) > 1) scroller.scrollTop += delta
    }
    const scheduleReveal = () => {
      cancelAnimationFrame(frame)
      frame = requestAnimationFrame(revealFocusedInput)
    }
    const observer = new ResizeObserver(scheduleReveal)
    observer.observe(scroller)

    scheduleReveal()
    panel.addEventListener('focusin', scheduleReveal)
    window.addEventListener('resize', scheduleReveal)
    window.visualViewport?.addEventListener('resize', scheduleReveal)
    window.visualViewport?.addEventListener('scroll', scheduleReveal)
    return () => {
      cancelAnimationFrame(frame)
      observer.disconnect()
      panel.removeEventListener('focusin', scheduleReveal)
      window.removeEventListener('resize', scheduleReveal)
      window.visualViewport?.removeEventListener('resize', scheduleReveal)
      window.visualViewport?.removeEventListener('scroll', scheduleReveal)
    }
  }, [open, variant, isMobileSidecar])
  usePanelFocus(open, ref, onClose, !popover)
  return createPortal(
    <div hidden={!open} className={open ? `fixed inset-0 z-[55] ${popover ? 'pointer-events-none' : anchored ? 'bg-black/25' : 'flex items-center justify-center bg-black/50 p-2 sm:p-6'}` : 'hidden'}
      onClick={event => { if (event.target === event.currentTarget) onClose() }}>
      <div ref={ref} id={id} role="dialog" aria-modal={popover ? undefined : true} aria-label={title} tabIndex={-1} style={anchored ? position : undefined}
        data-sidebar-overlay={anchored ? variant : undefined}
        className={`pointer-events-auto flex min-w-0 flex-col overflow-hidden rounded-2xl border border-border bg-bg-secondary shadow-2xl outline-none ${anchored ? 'fixed' : 'max-h-[90dvh] w-full max-w-xl'}`}>
        {!hideHeader && <div className="flex shrink-0 items-center justify-between gap-3 border-b border-border px-4 py-3">
          {headerStart}
          <h2 className="min-w-0 flex-1 break-words text-sm font-semibold text-text-primary">{title}</h2>
          <button type="button" aria-label={closeLabel || `Close ${title}`} onClick={onClose} className="shrink-0 rounded-lg p-2 text-text-secondary hover:bg-bg-hover"><X size={18}/></button>
        </div>}
        <div ref={scrollRef} className={`min-h-0 overflow-y-auto overscroll-contain p-3 sm:p-4 ${fixedHeight != null ? 'flex-1' : ''}`}
          style={fixedHeight != null ? { scrollbarGutter: 'stable', overflowAnchor: 'none' } : undefined}>{children}</div>
        {footer && <div className="shrink-0 border-t border-border p-3">{footer}</div>}
      </div>
    </div>, document.body,
  )
}

export function PromptDock({ children, toolbarHost }: { children: ReactNode; toolbarHost?: HTMLElement | null }) {
  const [expanded, setExpanded] = useState(false)
  const [enhanceSlot, setEnhanceSlot] = useState<HTMLDivElement | null>(null)
  const panel = useRef<HTMLDivElement>(null)
  usePanelFocus(expanded, panel, () => setExpanded(false))
  return (
    <ComposerContext.Provider value={{ expanded, expand: () => setExpanded(true) }}>
    <ComposerToolbarContext.Provider value={expanded || !toolbarHost ? enhanceSlot : toolbarHost}>
      <div ref={panel} tabIndex={expanded ? -1 : undefined} role={expanded ? 'dialog' : undefined}
        aria-modal={expanded ? true : undefined} aria-label={expanded ? 'H3 window prompts' : 'Prompt composer'}
        data-expanded={expanded} className={`studio-composer outline-none ${expanded ? 'fixed inset-0 z-[100] flex flex-col bg-bg-secondary p-4 sm:p-8' : 'flex grow shrink-0 basis-auto flex-col px-3 pb-3 pt-2'}`}>
        {expanded && <div className="mb-2 shrink-0 text-xs font-medium text-text-primary">H3 window prompts</div>}
        <div className={expanded ? 'min-h-0 flex-1 overflow-y-auto overscroll-contain' : 'studio-composer-content flex grow shrink-0 basis-auto flex-col'}>{children}</div>
        {(expanded || !toolbarHost) && <div role="group" aria-label="Prompt controls" className="studio-prompt-toolbar mt-2 flex shrink-0 items-center justify-end gap-2">
          <div ref={setEnhanceSlot} className="relative flex items-center justify-end gap-2 empty:hidden"/>
        </div>}
        {expanded && <div className="flex shrink-0 justify-end border-t border-border pt-3 mt-3">
          <button type="button" onClick={() => setExpanded(false)} className="min-h-10 rounded-xl bg-accent-blue px-5 text-xs text-white">Done</button>
        </div>}
      </div>
    </ComposerToolbarContext.Provider>
    </ComposerContext.Provider>
  )
}
