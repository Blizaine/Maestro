import { useCallback, useEffect, useLayoutEffect, useRef, useState, type MouseEvent as ReactMouseEvent, type PointerEvent as ReactPointerEvent } from 'react'

export interface GalleryZoomImageProps {
  src: string
  name: string
  onLoad?: () => void
  onError?: () => void
  /** True while a pinch or zoomed pan is in progress, so the swipe deck can stand down. */
  onInteractionChange?: (blocked: boolean) => void
}

interface ZoomView {
  scale: number
  x: number
  y: number
}

interface Point {
  x: number
  y: number
}

interface PointerStart {
  point: Point
  time: number
  pointerType: string
  moved: boolean
  doubleTap: boolean
}

interface PinchStart {
  distance: number
  scale: number
  anchor: Point
}

const MIN_SCALE = 1
const MAX_SCALE = 5
const POINTER_SLOP = 9
const DOUBLE_TAP_MS = 340
const DOUBLE_TAP_DISTANCE = 36
const INTERACTIVE_TARGET = 'button,a,input,select,textarea,[contenteditable="true"],[data-gallery-gesture-ignore]'

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value))
}

function distanceBetween(first: Point, second: Point): number {
  return Math.hypot(second.x - first.x, second.y - first.y)
}

function midpoint(first: Point, second: Point): Point {
  return { x: (first.x + second.x) / 2, y: (first.y + second.y) / 2 }
}

function isInteractiveTarget(target: EventTarget | null): boolean {
  return target instanceof Element && Boolean(target.closest(INTERACTIVE_TARGET))
}

export function GalleryZoomImage({ src, name, onLoad, onError, onInteractionChange }: GalleryZoomImageProps) {
  const rootRef = useRef<HTMLDivElement>(null)
  const imageRef = useRef<HTMLImageElement>(null)
  const viewRef = useRef<ZoomView>({ scale: 1, x: 0, y: 0 })
  const pointsRef = useRef(new Map<number, Point>())
  const startsRef = useRef(new Map<number, PointerStart>())
  const modeRef = useRef<'watching' | 'pan' | 'pinch'>('watching')
  const panStartRef = useRef<{ point: Point; x: number; y: number } | null>(null)
  const pinchStartRef = useRef<PinchStart | null>(null)
  const lastTapRef = useRef<{ point: Point; time: number } | null>(null)
  const lastTouchZoomAtRef = useRef(0)
  const blockedRef = useRef(false)
  const onInteractionChangeRef = useRef(onInteractionChange)
  const [view, setView] = useState<ZoomView>({ scale: 1, x: 0, y: 0 })

  useLayoutEffect(() => {
    onInteractionChangeRef.current = onInteractionChange
  }, [onInteractionChange])

  const notifyBlocked = useCallback((blocked: boolean) => {
    if (blockedRef.current === blocked) return
    blockedRef.current = blocked
    onInteractionChangeRef.current?.(blocked)
  }, [])

  const fittedSize = useCallback((): Point => {
    const root = rootRef.current
    const image = imageRef.current
    const width = root?.clientWidth ?? 0
    const height = root?.clientHeight ?? 0
    if (!root || !image || width <= 0 || height <= 0 || image.naturalWidth <= 0 || image.naturalHeight <= 0) {
      return { x: width, y: height }
    }
    const fitScale = Math.min(width / image.naturalWidth, height / image.naturalHeight)
    return { x: image.naturalWidth * fitScale, y: image.naturalHeight * fitScale }
  }, [])

  const boundedView = useCallback((next: ZoomView): ZoomView => {
    const scale = clamp(next.scale, MIN_SCALE, MAX_SCALE)
    if (scale <= MIN_SCALE + 0.001) return { scale: MIN_SCALE, x: 0, y: 0 }
    const root = rootRef.current
    const width = root?.clientWidth ?? 0
    const height = root?.clientHeight ?? 0
    const fit = fittedSize()
    const maxX = Math.max(0, (fit.x * scale - width) / 2)
    const maxY = Math.max(0, (fit.y * scale - height) / 2)
    return { scale, x: clamp(next.x, -maxX, maxX), y: clamp(next.y, -maxY, maxY) }
  }, [fittedSize])

  const applyView = useCallback((next: ZoomView) => {
    const bounded = boundedView(next)
    viewRef.current = bounded
    setView(bounded)
    notifyBlocked(bounded.scale > MIN_SCALE + 0.001 || pointsRef.current.size > 1)
  }, [boundedView, notifyBlocked])

  const resetZoom = useCallback(() => {
    pinchStartRef.current = null
    panStartRef.current = null
    applyView({ scale: MIN_SCALE, x: 0, y: 0 })
  }, [applyView])

  const pointFromEvent = useCallback((event: { clientX: number; clientY: number }): Point => {
    const bounds = rootRef.current?.getBoundingClientRect()
    return bounds ? { x: event.clientX - bounds.left, y: event.clientY - bounds.top } : { x: 0, y: 0 }
  }, [])

  const zoomAt = useCallback((point: Point, scale: number) => {
    const root = rootRef.current
    if (!root) return
    const nextScale = clamp(scale, MIN_SCALE, MAX_SCALE)
    if (nextScale <= MIN_SCALE + 0.001) {
      resetZoom()
      return
    }
    const current = viewRef.current
    const anchor = {
      x: (point.x - root.clientWidth / 2 - current.x) / current.scale,
      y: (point.y - root.clientHeight / 2 - current.y) / current.scale,
    }
    applyView({
      scale: nextScale,
      x: point.x - root.clientWidth / 2 - anchor.x * nextScale,
      y: point.y - root.clientHeight / 2 - anchor.y * nextScale,
    })
  }, [applyView, resetZoom])

  const toggleZoomAt = useCallback((point: Point) => {
    const current = viewRef.current
    if (current.scale > MIN_SCALE + 0.001) resetZoom()
    else zoomAt(point, 2.5)
  }, [resetZoom, zoomAt])

  const capturePointer = useCallback((pointerId: number) => {
    const root = rootRef.current
    if (!root) return
    try {
      root.setPointerCapture(pointerId)
    } catch {
      // Pointer capture can fail after a browser has already ended the pointer.
    }
  }, [])

  const startPinch = useCallback(() => {
    const root = rootRef.current
    const entries = Array.from(pointsRef.current.entries())
    if (!root || entries.length < 2) return
    const [firstId, first] = entries[0]
    const [secondId, second] = entries[1]
    const center = midpoint(first, second)
    const current = viewRef.current
    modeRef.current = 'pinch'
    pinchStartRef.current = {
      distance: Math.max(1, distanceBetween(first, second)),
      scale: current.scale,
      anchor: {
        x: (center.x - root.clientWidth / 2 - current.x) / current.scale,
        y: (center.y - root.clientHeight / 2 - current.y) / current.scale,
      },
    }
    panStartRef.current = null
    capturePointer(firstId)
    capturePointer(secondId)
    notifyBlocked(true)
  }, [capturePointer, notifyBlocked])

  const beginPan = useCallback((pointerId: number, point: Point) => {
    const current = viewRef.current
    if (current.scale <= MIN_SCALE + 0.001) {
      modeRef.current = 'watching'
      return
    }
    modeRef.current = 'pan'
    panStartRef.current = { point, x: current.x, y: current.y }
    capturePointer(pointerId)
    notifyBlocked(true)
  }, [capturePointer, notifyBlocked])

  const handlePointerDown = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (isInteractiveTarget(event.target) || (event.pointerType === 'mouse' && event.button !== 0)) return
    const point = pointFromEvent(event)
    const wasEmpty = pointsRef.current.size === 0
    pointsRef.current.set(event.pointerId, point)

    const now = event.timeStamp
    const previousTap = lastTapRef.current
    const isDoubleTap = wasEmpty && event.pointerType === 'touch' && previousTap !== null
      && now - previousTap.time <= DOUBLE_TAP_MS
      && distanceBetween(point, previousTap.point) <= DOUBLE_TAP_DISTANCE
    startsRef.current.set(event.pointerId, {
      point,
      time: now,
      pointerType: event.pointerType,
      moved: false,
      doubleTap: isDoubleTap,
    })
    if (isDoubleTap) lastTapRef.current = null

    if (pointsRef.current.size >= 2) {
      startPinch()
      return
    }
    if (viewRef.current.scale > MIN_SCALE + 0.001) beginPan(event.pointerId, point)
    else modeRef.current = 'watching'
  }

  const finishPointer = useCallback((event: PointerEvent, cancelled: boolean) => {
    const start = startsRef.current.get(event.pointerId)
    const currentPoint = pointsRef.current.get(event.pointerId)
    pointsRef.current.delete(event.pointerId)
    startsRef.current.delete(event.pointerId)
    const remaining = Array.from(pointsRef.current.entries())

    if (modeRef.current === 'pinch') {
      pinchStartRef.current = null
      if (remaining.length > 0 && viewRef.current.scale > MIN_SCALE + 0.001) {
        modeRef.current = 'pan'
        panStartRef.current = { point: remaining[0][1], x: viewRef.current.x, y: viewRef.current.y }
      } else if (remaining.length > 0) {
        modeRef.current = 'watching'
        panStartRef.current = null
      } else {
        modeRef.current = 'watching'
        panStartRef.current = null
      }
    } else if (modeRef.current === 'pan' && remaining.length > 0) {
      panStartRef.current = { point: remaining[0][1], x: viewRef.current.x, y: viewRef.current.y }
    } else if (remaining.length === 0) {
      modeRef.current = 'watching'
      panStartRef.current = null
    }

    notifyBlocked(viewRef.current.scale > MIN_SCALE + 0.001 || remaining.length > 1)

    if (cancelled || remaining.length > 0 || !start || !currentPoint) return
    const endedAt = pointFromEvent(event)
    if (distanceBetween(start.point, endedAt) >= POINTER_SLOP || start.moved) {
      lastTapRef.current = null
      return
    }
    if (start.pointerType === 'touch' && start.doubleTap) {
      lastTouchZoomAtRef.current = event.timeStamp
      toggleZoomAt(endedAt)
      lastTapRef.current = null
      return
    }
    if (start.pointerType === 'touch') lastTapRef.current = { point: endedAt, time: event.timeStamp }
  }, [notifyBlocked, pointFromEvent, toggleZoomAt])

  const handleDocumentPointerMove = useCallback((event: PointerEvent) => {
    if (!pointsRef.current.has(event.pointerId)) return
    const point = pointFromEvent(event)
    pointsRef.current.set(event.pointerId, point)
    const start = startsRef.current.get(event.pointerId)
    if (start && distanceBetween(start.point, point) >= POINTER_SLOP) start.moved = true

    if (modeRef.current === 'pinch' && pointsRef.current.size >= 2) {
      const [first, second] = Array.from(pointsRef.current.values())
      const pinchStart = pinchStartRef.current
      const root = rootRef.current
      if (!pinchStart || !root) return
      const center = midpoint(first, second)
      const scale = clamp(pinchStart.scale * distanceBetween(first, second) / pinchStart.distance, MIN_SCALE, MAX_SCALE)
      applyView({
        scale,
        x: center.x - root.clientWidth / 2 - pinchStart.anchor.x * scale,
        y: center.y - root.clientHeight / 2 - pinchStart.anchor.y * scale,
      })
      event.preventDefault()
      event.stopPropagation()
      return
    }

    if (modeRef.current === 'pan' && pointsRef.current.size === 1) {
      const panStart = panStartRef.current
      if (!panStart) return
      applyView({
        scale: viewRef.current.scale,
        x: panStart.x + point.x - panStart.point.x,
        y: panStart.y + point.y - panStart.point.y,
      })
      event.preventDefault()
      event.stopPropagation()
    }
  }, [applyView, pointFromEvent])

  const handleNativeWheel = useCallback((event: WheelEvent) => {
    if (!event.ctrlKey || !rootRef.current || isInteractiveTarget(event.target)) return
    event.preventDefault()
    event.stopPropagation()
    const point = pointFromEvent(event)
    zoomAt(point, viewRef.current.scale * Math.exp(-event.deltaY * 0.002))
  }, [pointFromEvent, zoomAt])

  const handleDocumentPointerUp = useCallback((event: PointerEvent) => {
    finishPointer(event, false)
  }, [finishPointer])

  const handleDocumentPointerCancel = useCallback((event: PointerEvent) => {
    finishPointer(event, true)
  }, [finishPointer])

  const handleDoubleClick = (event: ReactMouseEvent<HTMLDivElement>) => {
    if (isInteractiveTarget(event.target)) return
    event.preventDefault()
    // Some mobile browsers synthesize dblclick after the touch double-tap above.
    if (lastTouchZoomAtRef.current > 0 && event.timeStamp - lastTouchZoomAtRef.current < 650) return
    toggleZoomAt(pointFromEvent(event))
  }

  const reclampCurrentView = useCallback(() => {
    const current = viewRef.current
    const bounded = boundedView(current)
    if (bounded.scale === current.scale && bounded.x === current.x && bounded.y === current.y) return
    viewRef.current = bounded
    setView(bounded)
  }, [boundedView])

  useEffect(() => {
    const root = rootRef.current
    if (!root) return
    const observer = typeof ResizeObserver === 'undefined' ? null : new ResizeObserver(reclampCurrentView)
    observer?.observe(root)
    window.addEventListener('resize', reclampCurrentView)
    document.addEventListener('pointermove', handleDocumentPointerMove, true)
    document.addEventListener('pointerup', handleDocumentPointerUp, true)
    document.addEventListener('pointercancel', handleDocumentPointerCancel, true)
    root.addEventListener('wheel', handleNativeWheel, { passive: false })
    return () => {
      observer?.disconnect()
      window.removeEventListener('resize', reclampCurrentView)
      document.removeEventListener('pointermove', handleDocumentPointerMove, true)
      document.removeEventListener('pointerup', handleDocumentPointerUp, true)
      document.removeEventListener('pointercancel', handleDocumentPointerCancel, true)
      root.removeEventListener('wheel', handleNativeWheel)
    }
  }, [handleDocumentPointerCancel, handleDocumentPointerMove, handleDocumentPointerUp, handleNativeWheel, reclampCurrentView])

  useEffect(() => () => {
    pointsRef.current.clear()
    startsRef.current.clear()
    if (blockedRef.current) {
      blockedRef.current = false
      onInteractionChangeRef.current?.(false)
    }
  }, [])

  useEffect(() => {
    // Parent instances are keyed by media identity, and this also covers callers that reuse an instance.
    lastTapRef.current = null
    pointsRef.current.clear()
    startsRef.current.clear()
    modeRef.current = 'watching'
    // eslint-disable-next-line react-hooks/set-state-in-effect
    resetZoom()
  }, [resetZoom, src])

  return (
    <div
      ref={rootRef}
      data-gallery-image-zoom
      data-zoom-scale={view.scale}
      className="relative flex h-full w-full min-h-0 min-w-0 flex-1 items-center justify-center overflow-hidden bg-black"
      style={{ touchAction: 'none' }}
      onPointerDown={handlePointerDown}
      onDoubleClick={handleDoubleClick}
    >
      <img
        ref={imageRef}
        src={src}
        alt={name}
        draggable={false}
        className="absolute inset-0 h-full w-full select-none object-contain"
        style={{ transform: `translate3d(${view.x}px, ${view.y}px, 0) scale(${view.scale})`, transformOrigin: 'center' }}
        onLoad={() => {
          reclampCurrentView()
          onLoad?.()
        }}
        onError={onError}
      />
      {view.scale > MIN_SCALE + 0.001 && (
        <button
          type="button"
          aria-label="Reset zoom"
          title="Reset zoom"
          onClick={resetZoom}
          style={{
            top: 'max(0.75rem, env(safe-area-inset-top))',
            left: 'max(0.75rem, env(safe-area-inset-left))',
          }}
          className="absolute left-3 top-3 z-20 rounded-full bg-black/65 px-3 py-2 text-xs font-medium text-white shadow-md hover:bg-black/85 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-white"
        >
          Reset zoom
        </button>
      )}
    </div>
  )
}
