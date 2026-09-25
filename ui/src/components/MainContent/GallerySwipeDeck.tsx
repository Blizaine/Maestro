import {
  forwardRef,
  useCallback,
  useEffect,
  useImperativeHandle,
  useLayoutEffect,
  useRef,
  useState,
  type PointerEvent as ReactPointerEvent,
  type ReactNode,
  type TransitionEvent as ReactTransitionEvent,
  type WheelEvent as ReactWheelEvent,
} from 'react'

export interface GallerySwipeDeckHandle {
  navigate: (direction: -1 | 1) => void
  cancelGesture: () => void
}

export interface GallerySwipeDeckProps {
  activeId: string
  children: ReactNode
  /** Inert preview for the item before the active item. */
  previous?: ReactNode
  /** Inert preview for the item after the active item. */
  next?: ReactNode
  onNavigate: (direction: -1 | 1) => void
  /** Prepare one committed destination while its neighbor preview is centered. Return false to cancel it. */
  onPrepareNavigate?: (direction: -1 | 1) => Promise<boolean> | boolean
  /** Cancel an in-flight destination preparation when the deck itself is cancelled. */
  onCancelNavigation?: () => void
  /** A stationary tap, separate from a swipe or an interactive control. */
  onTap?: () => void
  enabled?: boolean
}

type Phase = 'idle' | 'dragging' | 'settling' | 'awaiting'
type PendingSettle = { kind: 'navigate'; direction: -1 | 1 } | { kind: 'reset' }

interface MotionState {
  activeId: string
  offset: number
  phase: Phase
}

interface PointerGesture {
  pointerId: number
  startX: number
  startY: number
  startedAt: number
  lastX: number
  lastY: number
  lastAt: number
  dragging: boolean
  rejected: boolean
  captured: boolean
}

const GESTURE_IGNORE = 'button,a,input,select,textarea,[contenteditable="true"],[role="slider"],[data-gallery-gesture-ignore]'
const POINTER_SLOP = 9
const WHEEL_THRESHOLD = 68
const WHEEL_QUIET_MS = 260
const WHEEL_COOLDOWN_MS = 420
const TRANSITION_MS = 250
const TRANSITION_FALLBACK_MS = TRANSITION_MS + 80

function hasPreview(node: ReactNode): boolean {
  return node !== null && node !== undefined && node !== false && node !== ''
}

function isInteractiveTarget(target: EventTarget | null): boolean {
  return target instanceof Element && Boolean(target.closest(GESTURE_IGNORE))
}

function isMediaControlBand(target: EventTarget | null, clientY: number): boolean {
  if (!(target instanceof Element)) return false
  const media = target.closest<HTMLMediaElement>('video, audio')
  if (!media?.controls) return false
  const bounds = media.getBoundingClientRect()
  return clientY >= bounds.bottom - 72
}

function reducedMotionPreference(): MediaQueryList | null {
  return typeof window !== 'undefined' && typeof window.matchMedia === 'function'
    ? window.matchMedia('(prefers-reduced-motion: reduce)')
    : null
}

export const GallerySwipeDeck = forwardRef<GallerySwipeDeckHandle, GallerySwipeDeckProps>(function GallerySwipeDeck({
  activeId,
  children,
  previous,
  next,
  onNavigate,
  onPrepareNavigate,
  onCancelNavigation,
  onTap,
  enabled = true,
}, ref) {
  const viewportRef = useRef<HTMLDivElement>(null)
  const currentPanelRef = useRef<HTMLDivElement>(null)
  const activeIdRef = useRef(activeId)
  const onNavigateRef = useRef(onNavigate)
  const onPrepareNavigateRef = useRef(onPrepareNavigate)
  const onCancelNavigationRef = useRef(onCancelNavigation)
  const availabilityRef = useRef({ enabled, previous: hasPreview(previous), next: hasPreview(next) })
  const pointerRef = useRef<PointerGesture | null>(null)
  const lockedRef = useRef(false)
  const pendingSettleRef = useRef<PendingSettle | null>(null)
  const settleTimerRef = useRef<number | null>(null)
  const navigationTokenRef = useRef(0)
  const navigationPreparationRef = useRef<{ token: number; promise: Promise<boolean> } | null>(null)
  const settleAnimationFinishedRef = useRef(false)
  const wheelTimerRef = useRef<number | null>(null)
  const wheelRef = useRef({ distance: 0, direction: 0, lastAt: 0, cooldownUntil: 0, triggered: false })
  const offsetRef = useRef(0)
  const viewportHeightRef = useRef(0)
  const [motion, setMotion] = useState<MotionState>({ activeId, offset: 0, phase: 'idle' })
  const [viewportHeight, setViewportHeight] = useState(0)
  const [reducedMotion, setReducedMotion] = useState(false)
  const offset = motion.activeId === activeId ? motion.offset : 0
  const phase = motion.activeId === activeId ? motion.phase : 'idle'

  useLayoutEffect(() => {
    onNavigateRef.current = onNavigate
    onPrepareNavigateRef.current = onPrepareNavigate
    onCancelNavigationRef.current = onCancelNavigation
    availabilityRef.current = { enabled, previous: hasPreview(previous), next: hasPreview(next) }
  }, [enabled, next, onCancelNavigation, onNavigate, onPrepareNavigate, previous])

  const updateOffset = useCallback((value: number) => {
    offsetRef.current = value
    setMotion(current => ({
      activeId,
      offset: value,
      phase: current.activeId === activeId ? current.phase : 'idle',
    }))
  }, [activeId])

  const updatePhase = useCallback((value: Phase) => {
    setMotion(current => ({
      activeId,
      offset: current.activeId === activeId ? current.offset : 0,
      phase: value,
    }))
  }, [activeId])

  const clearSettleTimer = useCallback(() => {
    if (settleTimerRef.current === null) return
    window.clearTimeout(settleTimerRef.current)
    settleTimerRef.current = null
  }, [])

  const clearWheelTimer = useCallback(() => {
    if (wheelTimerRef.current === null) return
    window.clearTimeout(wheelTimerRef.current)
    wheelTimerRef.current = null
  }, [])

  const releasePointerCapture = useCallback(() => {
    const gesture = pointerRef.current
    pointerRef.current = null
    if (!gesture?.captured) return
    const viewport = viewportRef.current
    if (!viewport?.hasPointerCapture(gesture.pointerId)) return
    try {
      viewport.releasePointerCapture(gesture.pointerId)
    } catch {
      // The browser may already have released capture during pointer cancellation.
    }
  }, [])

  const canNavigate = useCallback((direction: -1 | 1) => {
    const available = availabilityRef.current
    return available.enabled && (direction < 0 ? available.previous : available.next)
  }, [])

  const cancelPendingNavigation = useCallback(() => {
    if (pendingSettleRef.current?.kind !== 'navigate' && !navigationPreparationRef.current) return
    navigationTokenRef.current += 1
    navigationPreparationRef.current = null
    settleAnimationFinishedRef.current = false
    onCancelNavigationRef.current?.()
  }, [])

  const finishSettle = useCallback(() => {
    const pending = pendingSettleRef.current
    if (!pending || settleAnimationFinishedRef.current) return
    clearSettleTimer()
    settleAnimationFinishedRef.current = true
    if (pending.kind === 'navigate') {
      updatePhase('awaiting')
      const navigation = navigationPreparationRef.current
      const token = navigation?.token ?? navigationTokenRef.current
      const preparation = navigation?.promise ?? Promise.resolve(true)
      void preparation.then(canCommit => {
        if (navigationTokenRef.current !== token || activeIdRef.current !== activeId) return
        navigationPreparationRef.current = null
        pendingSettleRef.current = null
        settleAnimationFinishedRef.current = false
        if (!canCommit) {
          lockedRef.current = false
          updateOffset(0)
          updatePhase('idle')
          onCancelNavigationRef.current?.()
          return
        }
        onNavigateRef.current(pending.direction)
      }).catch(() => {
        if (navigationTokenRef.current !== token || activeIdRef.current !== activeId) return
        navigationPreparationRef.current = null
        pendingSettleRef.current = null
        settleAnimationFinishedRef.current = false
        onNavigateRef.current(pending.direction)
      })
      return
    }

    pendingSettleRef.current = null
    settleAnimationFinishedRef.current = false
    lockedRef.current = false
    updateOffset(0)
    updatePhase('idle')
  }, [activeId, clearSettleTimer, updateOffset, updatePhase])

  const beginSettle = useCallback((pending: PendingSettle, destination: number) => {
    if (pendingSettleRef.current || lockedRef.current) return
    lockedRef.current = true
    pendingSettleRef.current = pending
    settleAnimationFinishedRef.current = false
    if (pending.kind === 'navigate') {
      const token = ++navigationTokenRef.current
      let preparation: Promise<boolean>
      try {
        preparation = Promise.resolve(onPrepareNavigateRef.current?.(pending.direction) ?? true)
      } catch {
        preparation = Promise.resolve(true)
      }
      navigationPreparationRef.current = { token, promise: preparation }
    }
    updatePhase('settling')
    updateOffset(destination)
    settleTimerRef.current = window.setTimeout(finishSettle, reducedMotion ? 0 : TRANSITION_FALLBACK_MS)
  }, [finishSettle, reducedMotion, updateOffset, updatePhase])

  const startNavigation = useCallback((direction: -1 | 1) => {
    if (!canNavigate(direction) || lockedRef.current) return
    const height = viewportRef.current?.clientHeight || window.innerHeight || 1
    beginSettle({ kind: 'navigate', direction }, direction > 0 ? -height : height)
  }, [beginSettle, canNavigate])

  const resetAfterGesture = useCallback(() => {
    if (Math.abs(offsetRef.current) < 0.5) {
      lockedRef.current = false
      updateOffset(0)
      updatePhase('idle')
      return
    }
    beginSettle({ kind: 'reset' }, 0)
  }, [beginSettle, updateOffset, updatePhase])

  const cancelGesture = useCallback(() => {
    clearSettleTimer()
    cancelPendingNavigation()
    pendingSettleRef.current = null
    lockedRef.current = false
    releasePointerCapture()
    offsetRef.current = 0
    setMotion({ activeId: activeIdRef.current, offset: 0, phase: 'idle' })
  }, [cancelPendingNavigation, clearSettleTimer, releasePointerCapture])

  useImperativeHandle(ref, () => ({ navigate: startNavigation, cancelGesture }), [cancelGesture, startNavigation])

  useEffect(() => {
    const mediaQuery = reducedMotionPreference()
    if (!mediaQuery) return
    const update = () => setReducedMotion(mediaQuery.matches)
    update()
    mediaQuery.addEventListener?.('change', update)
    return () => mediaQuery.removeEventListener?.('change', update)
  }, [])

  useLayoutEffect(() => {
    const viewport = viewportRef.current
    if (!viewport) return
    const updateHeight = () => {
      const nextHeight = viewport.getBoundingClientRect().height
      const previousHeight = viewportHeightRef.current
      viewportHeightRef.current = nextHeight
      setViewportHeight(current => current === nextHeight ? current : nextHeight)
      if (previousHeight > 0 && nextHeight !== previousHeight
        && (pointerRef.current || pendingSettleRef.current)) {
        clearSettleTimer()
        cancelPendingNavigation()
        releasePointerCapture()
        pendingSettleRef.current = null
        lockedRef.current = false
        offsetRef.current = 0
        setMotion({ activeId: activeIdRef.current, offset: 0, phase: 'idle' })
      }
    }
    updateHeight()
    if (typeof ResizeObserver === 'undefined') {
      window.addEventListener('resize', updateHeight)
      return () => window.removeEventListener('resize', updateHeight)
    }
    const observer = new ResizeObserver(updateHeight)
    observer.observe(viewport)
    return () => observer.disconnect()
  }, [cancelPendingNavigation, clearSettleTimer, releasePointerCapture])

  useLayoutEffect(() => {
    if (enabled || (!pointerRef.current && !pendingSettleRef.current && !lockedRef.current)) return
    clearSettleTimer()
    cancelPendingNavigation()
    releasePointerCapture()
    pendingSettleRef.current = null
    lockedRef.current = false
    offsetRef.current = 0
    // This runs only when an in-flight gesture must be cancelled as its surface is disabled.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setMotion({ activeId, offset: 0, phase: 'idle' })
  }, [activeId, cancelPendingNavigation, clearSettleTimer, enabled, releasePointerCapture])

  useLayoutEffect(() => {
    if (activeIdRef.current === activeId) return
    activeIdRef.current = activeId
    clearSettleTimer()
    cancelPendingNavigation()
    releasePointerCapture()
    pendingSettleRef.current = null
    lockedRef.current = false
    offsetRef.current = 0
  }, [activeId, cancelPendingNavigation, clearSettleTimer, releasePointerCapture, updateOffset])

  useEffect(() => () => {
    clearSettleTimer()
    clearWheelTimer()
    navigationTokenRef.current += 1
    navigationPreparationRef.current = null
    pendingSettleRef.current = null
    releasePointerCapture()
  }, [clearSettleTimer, clearWheelTimer, releasePointerCapture])

  const startPointer = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (event.isPrimary === false) {
      // A second touch must stop an in-flight swipe, including after the deck
      // has captured the first pointer and begun moving the current panel.
      if ((event.pointerType === 'touch' || event.pointerType === 'pen') && pointerRef.current) cancelGesture()
      return
    }
    if (!availabilityRef.current.enabled || lockedRef.current || isInteractiveTarget(event.target)
      || isMediaControlBand(event.target, event.clientY)) return
    if (event.pointerType === 'mouse' && event.button !== 0) return
    if (event.pointerType !== 'touch' && event.pointerType !== 'pen' && event.pointerType !== 'mouse') return

    pointerRef.current = {
      pointerId: event.pointerId,
      startX: event.clientX,
      startY: event.clientY,
      startedAt: event.timeStamp,
      lastX: event.clientX,
      lastY: event.clientY,
      lastAt: event.timeStamp,
      dragging: false,
      rejected: false,
      captured: false,
    }
  }

  const movePointer = (event: ReactPointerEvent<HTMLDivElement>) => {
    const gesture = pointerRef.current
    if (!gesture || gesture.pointerId !== event.pointerId || gesture.rejected || lockedRef.current) return
    const deltaX = event.clientX - gesture.startX
    const deltaY = event.clientY - gesture.startY
    const absX = Math.abs(deltaX)
    const absY = Math.abs(deltaY)

    if (!gesture.dragging) {
      if (Math.max(absX, absY) < POINTER_SLOP) return
      if (absX >= absY * 1.15) {
        gesture.rejected = true
        pointerRef.current = null
        return
      }
      if (absY < absX * 1.15) return

      gesture.dragging = true
      const viewport = event.currentTarget
      try {
        viewport.setPointerCapture(event.pointerId)
        gesture.captured = true
      } catch {
        // Continue tracking on the viewport if capture is unavailable.
      }
      updatePhase('dragging')
    }

    event.preventDefault()
    const direction: -1 | 1 = deltaY < 0 ? 1 : -1
    const height = Math.max(1, event.currentTarget.clientHeight || viewportHeight || window.innerHeight)
    let nextOffset: number
    if (canNavigate(direction)) {
      nextOffset = Math.max(-height, Math.min(height, deltaY))
    } else {
      const maxOverscroll = Math.min(96, height * 0.16)
      nextOffset = Math.sign(deltaY) * maxOverscroll * (1 - Math.exp(-absY / Math.max(1, maxOverscroll)))
    }
    updateOffset(nextOffset)
    gesture.lastX = event.clientX
    gesture.lastY = event.clientY
    gesture.lastAt = event.timeStamp
  }

  const finishPointer = (event: ReactPointerEvent<HTMLDivElement>, cancelled = false) => {
    const gesture = pointerRef.current
    if (!gesture || gesture.pointerId !== event.pointerId) return
    const wasDragging = gesture.dragging
    const deltaX = event.clientX - gesture.startX
    const deltaY = event.clientY - gesture.startY
    const elapsed = Math.max(1, event.timeStamp - gesture.startedAt)
    const velocityY = deltaY / elapsed
    const direction: -1 | 1 = deltaY < 0 ? 1 : -1
    const height = Math.max(1, event.currentTarget.clientHeight || viewportHeight || window.innerHeight)
    releasePointerCapture()
    if (!wasDragging) {
      if (!cancelled && Math.max(Math.abs(deltaX), Math.abs(deltaY)) < POINTER_SLOP) onTap?.()
      return
    }

    if (!cancelled && canNavigate(direction) && Math.abs(deltaY) >= Math.abs(deltaX) * 1.15
      && (Math.abs(deltaY) >= Math.max(76, height * 0.22) || (Math.abs(deltaY) >= 42 && Math.abs(velocityY) >= 0.62))) {
      beginSettle({ kind: 'navigate', direction }, direction > 0 ? -height : height)
    } else {
      resetAfterGesture()
    }
  }

  const handleWheel = (event: ReactWheelEvent<HTMLDivElement>) => {
    if (!availabilityRef.current.enabled || event.ctrlKey || isInteractiveTarget(event.target)
      || isMediaControlBand(event.target, event.clientY)) return
    const scale = event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? (viewportHeight || window.innerHeight) : 1
    const deltaX = event.deltaX * scale
    const deltaY = event.deltaY * scale
    if (Math.abs(deltaY) <= Math.abs(deltaX) || deltaY === 0) return
    const now = event.timeStamp || performance.now()
    const wheel = wheelRef.current
    const direction = Math.sign(deltaY)
    if (now - wheel.lastAt > 180 || (wheel.direction !== 0 && wheel.direction !== direction)) wheel.distance = 0
    wheel.lastAt = now
    wheel.direction = direction

    if (wheel.triggered) {
      clearWheelTimer()
      wheelTimerRef.current = window.setTimeout(() => {
        wheelRef.current.triggered = false
        wheelRef.current.distance = 0
        wheelTimerRef.current = null
      }, WHEEL_QUIET_MS)
      return
    }
    if (lockedRef.current || now < wheel.cooldownUntil) return

    wheel.distance += deltaY
    if (Math.abs(wheel.distance) < WHEEL_THRESHOLD) return

    const navigateDirection: -1 | 1 = wheel.distance > 0 ? 1 : -1
    wheel.distance = 0
    wheel.cooldownUntil = now + WHEEL_COOLDOWN_MS
    wheel.triggered = true
    clearWheelTimer()
    wheelTimerRef.current = window.setTimeout(() => {
      wheelRef.current.triggered = false
      wheelRef.current.distance = 0
      wheelTimerRef.current = null
    }, WHEEL_QUIET_MS)
    startNavigation(navigateDirection)
  }

  const handleTransitionEnd = (event: ReactTransitionEvent<HTMLDivElement>) => {
    if (event.target !== currentPanelRef.current || event.propertyName !== 'transform') return
    finishSettle()
  }

  const fallbackHeight = typeof window === 'undefined' ? 1 : window.innerHeight
  const height = viewportHeight || fallbackHeight || 1
  const transition = phase === 'settling' || phase === 'awaiting'
    ? `transform ${reducedMotion ? 0 : TRANSITION_MS}ms cubic-bezier(0.22, 1, 0.36, 1)`
    : 'none'
  const currentStyle = {
    transform: `translate3d(0, ${offset}px, 0)`,
    transition,
    zIndex: 10,
    willChange: phase === 'idle' ? undefined : 'transform' as const,
  }

  return (
    <div
      ref={viewportRef}
      data-gallery-swipe-deck
      className="relative min-h-0 min-w-0 flex-1 overflow-hidden"
      style={{ touchAction: 'pan-x' }}
      onPointerDown={startPointer}
      onPointerMove={movePointer}
      onPointerUp={event => finishPointer(event)}
      onPointerCancel={event => finishPointer(event, true)}
      onLostPointerCapture={event => {
        // Touch starts with implicit capture on the image/video. Its lost-capture
        // event bubbles here when we take over; that is not a cancelled swipe.
        if (event.target === event.currentTarget
          && pointerRef.current?.pointerId === event.pointerId && pointerRef.current.dragging) finishPointer(event, true)
      }}
      onWheel={handleWheel}
    >
      {hasPreview(previous) && (
        <div
          aria-hidden="true"
          inert
          data-gallery-swipe-previous
          className="pointer-events-none absolute inset-0 min-h-0 min-w-0"
          style={{ transform: `translate3d(0, ${offset - height}px, 0)`, transition, zIndex: 0 }}
        >
          {previous}
        </div>
      )}
      {hasPreview(next) && (
        <div
          aria-hidden="true"
          inert
          data-gallery-swipe-next
          className="pointer-events-none absolute inset-0 min-h-0 min-w-0"
          style={{ transform: `translate3d(0, ${offset + height}px, 0)`, transition, zIndex: 0 }}
        >
          {next}
        </div>
      )}
      <div
        ref={currentPanelRef}
        data-gallery-swipe-current
        className="relative flex h-full min-h-0 min-w-0 flex-col"
        style={currentStyle}
        onTransitionEnd={handleTransitionEnd}
      >
        {children}
      </div>
    </div>
  )
})
