import { useCallback, useEffect, useId, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import { Info } from 'lucide-react'

interface InfoTooltipProps {
  text: string
  label?: string
  placement?: 'auto' | 'top' | 'bottom'
}

type TooltipPosition = {
  top: number
  left: number
  below: boolean
}

/** Compact, portal-backed help for sidebar controls. */
export function InfoTooltip({
  text,
  label = 'More information',
  placement = 'auto',
}: InfoTooltipProps) {
  const [open, setOpen] = useState(false)
  const [position, setPosition] = useState<TooltipPosition>({
    top: 0,
    left: 0,
    below: false,
  })
  const triggerRef = useRef<HTMLButtonElement>(null)
  const tooltipId = useId()

  const updatePosition = useCallback(() => {
    if (!triggerRef.current) return
    const rect = triggerRef.current.getBoundingClientRect()
    const tooltipWidth = Math.min(272, window.innerWidth - 16)
    const below = placement === 'bottom'
      || (placement === 'auto' && rect.top < 150)
    const left = Math.min(
      Math.max(rect.left + (rect.width / 2), 8 + (tooltipWidth / 2)),
      window.innerWidth - 8 - (tooltipWidth / 2),
    )
    setPosition({
      top: below ? rect.bottom + 7 : rect.top - 7,
      left,
      below,
    })
  }, [placement])

  const showTooltip = useCallback(() => {
    updatePosition()
    setOpen(true)
  }, [updatePosition])

  useEffect(() => {
    if (!open) return

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') setOpen(false)
    }
    const handleOutsideClick = (event: PointerEvent) => {
      if (!triggerRef.current?.contains(event.target as Node)) setOpen(false)
    }

    document.addEventListener('keydown', handleKeyDown)
    document.addEventListener('pointerdown', handleOutsideClick)
    window.addEventListener('resize', updatePosition)
    window.addEventListener('scroll', updatePosition, true)
    return () => {
      document.removeEventListener('keydown', handleKeyDown)
      document.removeEventListener('pointerdown', handleOutsideClick)
      window.removeEventListener('resize', updatePosition)
      window.removeEventListener('scroll', updatePosition, true)
    }
  }, [open, updatePosition])

  return (
    <>
      <button
        ref={triggerRef}
        type="button"
        aria-label={label}
        aria-describedby={open ? tooltipId : undefined}
        onMouseEnter={showTooltip}
        onMouseLeave={() => setOpen(false)}
        onFocus={showTooltip}
        onBlur={() => setOpen(false)}
        onClick={event => {
          event.stopPropagation()
          showTooltip()
        }}
        className="inline-flex shrink-0 items-center justify-center rounded p-0.5 text-text-muted transition-colors hover:text-accent-blue focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-accent-blue"
      >
        <Info size={11} />
      </button>
      {open && createPortal(
        <div
          id={tooltipId}
          role="tooltip"
          className="pointer-events-none fixed z-[120] w-64 max-w-[calc(100vw-1rem)] rounded-lg border border-border bg-bg-secondary px-2.5 py-2 text-[10px] leading-relaxed text-text-secondary shadow-xl whitespace-pre-line"
          style={{
            top: position.top,
            left: position.left,
            transform: position.below
              ? 'translate(-50%, 0)'
              : 'translate(-50%, -100%)',
          }}
        >
          {text}
        </div>,
        document.body,
      )}
    </>
  )
}
