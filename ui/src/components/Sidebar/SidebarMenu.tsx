import { useEffect, useId, useLayoutEffect, useRef, useState, type CSSProperties, type ReactNode } from 'react'
import { createPortal } from 'react-dom'
import { usePanelFocus } from './SidebarPanels'

/** Compact choices above their trigger, within the visible keyboard viewport. */
export function SidebarMenu({ open, anchor, label, id, onClose, children, width = 200, align = 'end' }: {
  open: boolean; anchor: HTMLElement | null; label: string; id?: string
  onClose: () => void; children: ReactNode; width?: number | 'content'; align?: 'start' | 'end'
}) {
  const ref = useRef<HTMLDivElement>(null)
  const key = useId()
  const closeRef = useRef(onClose)
  const [position, setPosition] = useState<CSSProperties>({})
  useEffect(() => { closeRef.current = onClose }, [onClose])
  useLayoutEffect(() => {
    if (!open || !anchor) return
    const measure = () => {
      const viewport = window.visualViewport
      const top = viewport?.offsetTop || 0
      const left = viewport?.offsetLeft || 0
      const visibleWidth = viewport?.width || window.innerWidth
      const bounds = anchor.getBoundingClientRect()
      const sidebar = anchor.closest('.maestro-sidebar')?.getBoundingClientRect()
      const minLeft = Math.max(left + 8, sidebar ? sidebar.left + 8 : left + 8)
      const maxRight = Math.min(left + visibleWidth - 8, sidebar ? sidebar.right - 8 : left + visibleWidth - 8)
      const menuWidth = Math.min(width === 'content' ? Math.max(bounds.width, ref.current?.getBoundingClientRect().width || 96) : width, maxRight - minLeft)
      setPosition({
        width: width === 'content' ? 'max-content' : menuWidth,
        minWidth: width === 'content' ? bounds.width : undefined,
        maxWidth: maxRight - minLeft,
        left: Math.max(minLeft, Math.min(align === 'start' ? bounds.left : bounds.right - menuWidth, maxRight - menuWidth)),
        bottom: window.innerHeight - bounds.top + 6,
        maxHeight: Math.max(44, bounds.top - top - 14),
      })
    }
    measure()
    // The sidebar commits its viewport size/offset in React. Measure its
    // trigger after that commit, including offset-only iOS viewport scrolls.
    let frame = 0
    const scheduleMeasure = () => {
      cancelAnimationFrame(frame)
      frame = requestAnimationFrame(measure)
    }
    const observer = new ResizeObserver(scheduleMeasure)
    observer.observe(anchor)
    if (ref.current) observer.observe(ref.current)
    window.addEventListener('resize', scheduleMeasure)
    window.addEventListener('scroll', scheduleMeasure, true)
    window.visualViewport?.addEventListener('resize', scheduleMeasure)
    window.visualViewport?.addEventListener('scroll', scheduleMeasure)
    return () => {
      cancelAnimationFrame(frame)
      observer.disconnect()
      window.removeEventListener('resize', scheduleMeasure)
      window.removeEventListener('scroll', scheduleMeasure, true)
      window.visualViewport?.removeEventListener('resize', scheduleMeasure)
      window.visualViewport?.removeEventListener('scroll', scheduleMeasure)
    }
  }, [open, anchor, width, align])
  useEffect(() => {
    if (!open) return
    const eventName = 'maestro-studio-panel-open'
    const dismiss = (event: Event) => {
      if ((event as CustomEvent<string>).detail !== key) closeRef.current()
    }
    const outside = (event: PointerEvent) => {
      if (!ref.current?.contains(event.target as Node) && !anchor?.contains(event.target as Node)) closeRef.current()
    }
    window.dispatchEvent(new CustomEvent(eventName, { detail: key }))
    window.addEventListener(eventName, dismiss)
    document.addEventListener('pointerdown', outside)
    return () => {
      window.removeEventListener(eventName, dismiss)
      document.removeEventListener('pointerdown', outside)
    }
  }, [open, key, anchor])
  usePanelFocus(open, ref, onClose, false)
  return createPortal(<div ref={ref} id={id} role="menu" aria-label={label} hidden={!open} tabIndex={-1}
    style={{...position, ...(width === 'content' ? {width: 'max-content'} : {})}} className={open ? 'fixed z-[110] overflow-y-auto overscroll-contain rounded-xl border border-border bg-bg-secondary p-1 shadow-xl outline-none' : 'hidden'}
    onKeyDown={event => {
      if (!['ArrowDown', 'ArrowUp', 'Home', 'End'].includes(event.key)) return
      event.preventDefault()
      const items = Array.from(event.currentTarget.querySelectorAll<HTMLButtonElement>('button:not(:disabled)'))
      const current = items.indexOf(document.activeElement as HTMLButtonElement)
      const next = event.key === 'Home' ? 0 : event.key === 'End' ? items.length - 1
        : current < 0 ? (event.key === 'ArrowDown' ? 0 : items.length - 1)
        : (current + (event.key === 'ArrowDown' ? 1 : -1) + items.length) % items.length
      items[next]?.focus()
    }}>{children}</div>, document.body)
}
