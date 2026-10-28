import { outputIdentity } from '../../lib/galleryIdentity'
import { useState, useEffect, useLayoutEffect, useRef, useMemo, useCallback } from 'react'
import type { KeyboardEvent as ReactKeyboardEvent, PointerEvent as ReactPointerEvent } from 'react'
import { Film, Music, X } from 'lucide-react'
import { useStore } from '../../stores/useStore'
import { requestThumbnail } from '../../lib/thumbnailCache'
import { useMediaDuration } from '../../lib/mediaDuration'
import { useIsMobileSidecar } from '../../lib/useIsMobile'
import { getMediaTimestamp } from '../../lib/mediaTimestamp'
import type { OutputFile } from '../../types'
import { MediaDurationBadge } from './MediaDurationBadge'

const MIN_PANEL_WIDTH = 80
const MIN_MOBILE_PANEL_WIDTH = 100
const MAX_PANEL_WIDTH = 360
const MIN_GALLERY_WIDTH = 240
const THUMB_GAP = 6
const THUMB_OVERSCAN = 10
const PANEL_WIDTH_KEY = 'maestro-thumbnail-panel-width'

interface Props {
  activeIndex: number
  onThumbnailClick: (id: string) => void
  open: boolean
  onClose: () => void
}

interface ResizeDrag {
  pointerId: number
  startX: number
  startWidth: number
  latestWidth: number
  originalCursor: string
  originalCursorPriority: string
  originalUserSelect: string
  originalUserSelectPriority: string
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value))
}

function readPreferredWidth(): number {
  try {
    const stored = Number(window.localStorage.getItem(PANEL_WIDTH_KEY))
    if (Number.isFinite(stored) && stored > 0) return clamp(stored, MIN_PANEL_WIDTH, MAX_PANEL_WIDTH)
  } catch {
    // Storage may be unavailable in a restricted browser context.
  }
  return MIN_PANEL_WIDTH
}

function persistPreferredWidth(width: number): void {
  try {
    window.localStorage.setItem(PANEL_WIDTH_KEY, String(Math.round(width)))
  } catch {
    // The panel remains usable when storage is unavailable.
  }
}

function VideoThumbnail({ file }: { file: OutputFile }) {
  const [thumbUrl, setThumbUrl] = useState<string | null>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const duration = useMediaDuration(containerRef, file)

  useEffect(() => {
    let cancelled = false

    requestThumbnail(file.url, file.url).then((dataUrl) => {
      if (!cancelled && dataUrl) setThumbUrl(dataUrl)
    })

    return () => { cancelled = true }
  }, [file.url, file.name])

  return (
    <div ref={containerRef} className="relative w-full h-full">
      {thumbUrl ? (
        <img src={thumbUrl} alt={file.name} className="w-full h-full object-cover" />
      ) : (
        <div className="w-full h-full bg-bg-active flex items-center justify-center">
          <Film size={14} className="text-text-muted animate-pulse" />
        </div>
      )}
      <MediaDurationBadge seconds={duration} className="bottom-1 right-1 px-1 py-0.5 text-[9px]" />
    </div>
  )
}

function VirtualizedThumbnailList({
  activeIndex,
  onThumbnailClick,
  onMobileClick,
}: Pick<Props, 'activeIndex' | 'onThumbnailClick'> & { onMobileClick?: () => void }) {
  const outputs = useStore(s => s.filteredOutputs())
  const containerRef = useRef<HTMLDivElement>(null)
  const [scrollTop, setScrollTop] = useState(0)
  const [viewHeight, setViewHeight] = useState(400)
  const [thumbHeight, setThumbHeight] = useState(45)

  useEffect(() => {
    const el = containerRef.current
    if (!el) return
    const measure = () => {
      setViewHeight(el.clientHeight)
      const nextHeight = el.clientWidth * 9 / 16
      if (nextHeight > 0) setThumbHeight(nextHeight)
    }
    const ro = new ResizeObserver(measure)
    ro.observe(el)
    measure()
    return () => ro.disconnect()
  }, [])

  const loadingMore = useRef(false)

  const handleScroll = useCallback(() => {
    const el = containerRef.current
    if (!el) return
    setScrollTop(el.scrollTop)
    const distanceToBottom = el.scrollHeight - el.scrollTop - el.clientHeight
    if (distanceToBottom < el.clientHeight && !loadingMore.current) {
      const store = useStore.getState()
      if (store.outputs.length < store.outputsTotal) {
        loadingMore.current = true
        store.loadMoreOutputs().finally(() => { loadingMore.current = false })
      }
    }
  }, [])

  // Width changes alter every row's height. Reposition only when the selected
  // item would otherwise be outside the viewport, and use an instant scroll.
  useEffect(() => {
    const container = containerRef.current
    if (!container || outputs.length === 0) return
    const itemSize = thumbHeight + THUMB_GAP
    const itemTop = activeIndex * itemSize
    const itemBottom = itemTop + thumbHeight
    const viewTop = container.scrollTop
    const viewBottom = viewTop + container.clientHeight

    if (itemTop < viewTop || itemBottom > viewBottom) {
      const maxScrollTop = Math.max(0, outputs.length * itemSize - THUMB_GAP - container.clientHeight)
      const nextScrollTop = clamp(itemTop - (container.clientHeight - thumbHeight) / 2, 0, maxScrollTop)
      container.scrollTop = nextScrollTop
      setScrollTop(nextScrollTop)
    }
  }, [activeIndex, outputs.length, thumbHeight, viewHeight])

  const itemSize = thumbHeight + THUMB_GAP
  const totalHeight = outputs.length * itemSize - (outputs.length > 0 ? THUMB_GAP : 0)

  const { startIdx, endIdx } = useMemo(() => {
    const start = Math.max(0, Math.floor(scrollTop / itemSize) - THUMB_OVERSCAN)
    const visibleCount = Math.ceil(viewHeight / itemSize)
    const end = Math.min(outputs.length, Math.floor(scrollTop / itemSize) + visibleCount + THUMB_OVERSCAN)
    return { startIdx: start, endIdx: end }
  }, [scrollTop, viewHeight, outputs.length, itemSize])

  return (
    <div
      ref={containerRef}
      className="flex-1 min-h-0 w-full overflow-y-auto"
      style={{ scrollbarGutter: 'stable' }}
      onScroll={handleScroll}
    >
      <div className="relative" style={{ height: totalHeight }}>
        {outputs.slice(startIdx, endIdx).map((file, i) => {
          const idx = startIdx + i
          const timestamp = getMediaTimestamp(undefined, file.created_at)
          const thumbLabel = timestamp ? `${file.name} · ${timestamp.compact}` : file.name
          return (
            <button
              key={outputIdentity(file)}
              title={thumbLabel}
              aria-label={`Show ${thumbLabel}`}
              aria-current={activeIndex === idx ? 'true' : undefined}
              data-thumb-index={idx}
              onClick={() => {
                onThumbnailClick(outputIdentity(file))
                onMobileClick?.()
              }}
              className={`absolute left-0 right-0 rounded-lg border overflow-hidden transition-colors ${
                activeIndex === idx
                  ? 'border-accent-blue ring-1 ring-accent-blue shadow-active-ring'
                  : 'border-border hover:border-border-light'
              }`}
              style={{
                top: idx * itemSize,
                height: thumbHeight,
              }}
            >
              {file.type === 'video' ? (
                <VideoThumbnail file={file} />
              ) : file.type === 'audio' ? (
                <div className="w-full h-full bg-bg-active flex items-center justify-center">
                  <Music size={14} className="text-text-muted" />
                </div>
              ) : (
                <img src={file.url} alt={file.name} className="w-full h-full object-cover" />
              )}
            </button>
          )
        })}
      </div>
    </div>
  )
}

export function ThumbnailGallery({ activeIndex, onThumbnailClick, open, onClose }: Props) {
  const outputs = useStore(s => s.filteredOutputs())
  const outputsLoading = useStore(s => s.outputsLoading)
  const isMobile = useIsMobileSidecar()
  const hasContent = outputs.length > 0 || outputsLoading
  const [preferredWidth, setPreferredWidth] = useState(readPreferredWidth)
  const [parentWidth, setParentWidth] = useState(0)
  const [viewportWidth, setViewportWidth] = useState(() => typeof window === 'undefined' ? 0 : window.innerWidth)
  const panelRef = useRef<HTMLElement>(null)
  const dragRef = useRef<ResizeDrag | null>(null)

  useLayoutEffect(() => {
    if (!open || !hasContent) return
    const panel = panelRef.current
    const parent = panel?.parentElement
    if (!panel || !parent) return
    const measure = () => {
      setParentWidth(parent.clientWidth)
      setViewportWidth(window.innerWidth)
    }
    const ro = new ResizeObserver(measure)
    ro.observe(parent)
    window.addEventListener('resize', measure)
    measure()
    return () => {
      ro.disconnect()
      window.removeEventListener('resize', measure)
    }
  }, [open, isMobile, hasContent])

  const maximumWidth = isMobile
    ? Math.min(MAX_PANEL_WIDTH, Math.max(1, viewportWidth))
    : Math.min(MAX_PANEL_WIDTH, Math.max(MIN_PANEL_WIDTH, parentWidth - MIN_GALLERY_WIDTH))
  const minimumWidth = Math.min(isMobile ? MIN_MOBILE_PANEL_WIDTH : MIN_PANEL_WIDTH, maximumWidth)
  const effectiveWidth = clamp(preferredWidth, minimumWidth, maximumWidth)

  const restoreDragStyles = useCallback(() => {
    const drag = dragRef.current
    if (!drag) return
    const style = document.body.style
    if (drag.originalCursor) style.setProperty('cursor', drag.originalCursor, drag.originalCursorPriority)
    else style.removeProperty('cursor')
    if (drag.originalUserSelect) style.setProperty('user-select', drag.originalUserSelect, drag.originalUserSelectPriority)
    else style.removeProperty('user-select')
  }, [])

  const finishResize = useCallback((persist: boolean) => {
    const drag = dragRef.current
    if (!drag) return
    restoreDragStyles()
    dragRef.current = null
    if (persist) {
      const width = clamp(drag.latestWidth, minimumWidth, maximumWidth)
      setPreferredWidth(width)
      persistPreferredWidth(width)
    }
  }, [maximumWidth, minimumWidth, restoreDragStyles])

  useEffect(() => () => {
    const drag = dragRef.current
    restoreDragStyles()
    dragRef.current = null
    if (drag) persistPreferredWidth(drag.latestWidth)
  }, [restoreDragStyles])

  useEffect(() => {
    if (!open || !hasContent) finishResize(true)
  }, [open, hasContent, finishResize])

  const handleResizePointerDown = useCallback((event: ReactPointerEvent<HTMLDivElement>) => {
    if (dragRef.current) return
    if (event.button !== 0 && event.pointerType !== 'touch') return
    event.preventDefault()
    const style = document.body.style
    dragRef.current = {
      pointerId: event.pointerId,
      startX: event.clientX,
      startWidth: effectiveWidth,
      latestWidth: effectiveWidth,
      originalCursor: style.getPropertyValue('cursor'),
      originalCursorPriority: style.getPropertyPriority('cursor'),
      originalUserSelect: style.getPropertyValue('user-select'),
      originalUserSelectPriority: style.getPropertyPriority('user-select'),
    }
    style.setProperty('cursor', 'col-resize')
    style.setProperty('user-select', 'none')
    try { event.currentTarget.setPointerCapture(event.pointerId) } catch { /* Capture may be unavailable in a synthetic event. */ }
  }, [effectiveWidth])

  const handleResizePointerMove = useCallback((event: ReactPointerEvent<HTMLDivElement>) => {
    const drag = dragRef.current
    if (!drag || drag.pointerId !== event.pointerId) return
    event.preventDefault()
    const width = clamp(drag.startWidth + drag.startX - event.clientX, minimumWidth, maximumWidth)
    drag.latestWidth = width
    setPreferredWidth(width)
  }, [maximumWidth, minimumWidth])

  const handleResizePointerEnd = useCallback((event: ReactPointerEvent<HTMLDivElement>) => {
    if (dragRef.current?.pointerId !== event.pointerId) return
    finishResize(true)
  }, [finishResize])

  const handleResizeKeys = useCallback((event: ReactKeyboardEvent<HTMLDivElement>) => {
    let nextWidth: number | undefined
    if (event.key === 'ArrowLeft') nextWidth = effectiveWidth + 20
    else if (event.key === 'ArrowRight') nextWidth = effectiveWidth - 20
    else if (event.key === 'Home') nextWidth = minimumWidth
    else if (event.key === 'End') nextWidth = maximumWidth
    else return
    event.preventDefault()
    const width = clamp(nextWidth, minimumWidth, maximumWidth)
    setPreferredWidth(width)
    persistPreferredWidth(width)
  }, [effectiveWidth, maximumWidth, minimumWidth])

  if (!open || (outputs.length === 0 && !outputsLoading)) return null

  const resizeHandle = (
    <div
      role="separator"
      aria-label="Resize thumbnails"
      aria-orientation="vertical"
      aria-valuemin={minimumWidth}
      aria-valuemax={maximumWidth}
      aria-valuenow={Math.round(effectiveWidth)}
      aria-valuetext={`${Math.round(effectiveWidth)} pixels`}
      tabIndex={0}
      onKeyDown={handleResizeKeys}
      onPointerDown={handleResizePointerDown}
      onPointerMove={handleResizePointerMove}
      onPointerUp={handleResizePointerEnd}
      onPointerCancel={handleResizePointerEnd}
      onLostPointerCapture={() => finishResize(true)}
      className="group absolute left-0 top-0 bottom-0 z-10 w-2 cursor-col-resize touch-none select-none hover:bg-bg-hover/70 focus:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-accent-blue"
      style={{ touchAction: 'none', userSelect: 'none' }}
    >
      <span aria-hidden="true" className="absolute left-1/2 top-0 bottom-0 w-px -translate-x-1/2 bg-border group-hover:bg-accent-blue group-focus-visible:bg-accent-blue" />
    </div>
  )

  if (isMobile) {
    return (
      <>
        <div className="fixed inset-0 bg-black/40 z-40" onClick={onClose} aria-hidden="true" />
        <aside
          ref={panelRef}
          id="gallery-thumbnail-panel"
          data-testid="thumbnail-panel"
          aria-label="Gallery thumbnails"
          className="fixed top-0 right-0 h-full bg-bg-secondary border-l border-border z-50 flex flex-col overflow-hidden"
          style={{ width: effectiveWidth, maxWidth: '100vw' }}
        >
          {resizeHandle}
          <div className="px-2 py-2 border-b border-border flex items-center justify-between min-w-0">
            <span className="min-w-0 truncate text-[10px] text-text-muted uppercase tracking-wider">History</span>
            <button
              type="button"
              aria-label="Close thumbnails"
              title="Close thumbnails"
              onClick={onClose}
              className="shrink-0 p-1 rounded hover:bg-bg-hover text-text-secondary"
            >
              <X size={14} />
            </button>
          </div>
          <VirtualizedThumbnailList
            activeIndex={activeIndex}
            onThumbnailClick={onThumbnailClick}
            onMobileClick={onClose}
          />
        </aside>
      </>
    )
  }

  return (
    <aside
      ref={panelRef}
      id="gallery-thumbnail-panel"
      data-testid="thumbnail-panel"
      aria-label="Gallery thumbnails"
      className="relative shrink-0 min-h-0 border-l border-border flex flex-col overflow-hidden"
      style={{ width: effectiveWidth, flexBasis: effectiveWidth }}
    >
      {resizeHandle}
      <VirtualizedThumbnailList activeIndex={activeIndex} onThumbnailClick={onThumbnailClick} />
    </aside>
  )
}
