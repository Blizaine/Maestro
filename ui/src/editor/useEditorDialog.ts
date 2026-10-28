import { useEffect, useRef } from 'react'

/** Keep modal keyboard gestures out of the timeline and return focus on close. */
export function useEditorDialog(open: boolean, onClose: () => void) {
  const root = useRef<HTMLDivElement>(null)
  const latestClose = useRef(onClose)
  useEffect(() => { latestClose.current = onClose }, [onClose])
  useEffect(() => {
    if (!open || !root.current) return
    const element = root.current
    const previous = document.activeElement as HTMLElement | null
    const controls = () => [...element.querySelectorAll<HTMLElement>('button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), a[href], [tabindex="0"]')]
      .filter(control => control.getClientRects().length > 0)
    controls()[0]?.focus()
    const keyboard = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        event.preventDefault()
        event.stopImmediatePropagation()
        latestClose.current()
      } else if (event.key === 'Tab') {
        const available = controls()
        const first = available[0], last = available.at(-1)
        if (!element.contains(document.activeElement)) { event.preventDefault(); first?.focus() }
        else if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last?.focus() }
        else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first?.focus() }
      }
    }
    document.addEventListener('keydown', keyboard, true)
    return () => {
      document.removeEventListener('keydown', keyboard, true)
      if (previous?.isConnected) previous.focus()
    }
  }, [open])
  return root
}
