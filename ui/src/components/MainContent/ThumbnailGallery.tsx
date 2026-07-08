import { useState, useEffect, useRef, useMemo, useCallback } from 'react'
import { Film, Music, PanelRightClose, PanelRightOpen, X } from 'lucide-react'
import { useStore } from '../../stores/useStore'
import { requestThumbnail } from '../../lib/thumbnailCache'
import { useIsMobile } from '../../lib/useIsMobile'

// Thumbnail dimensions: 80px wide, 16:9 aspect = 45px tall, 6px gap
const THUMB_HEIGHT = 45
const THUMB_GAP = 6
const THUMB_OVERSCAN = 10

function VideoThumbnail({ src, name }: { src: string; name: string }) {
  const [thumbUrl, setThumbUrl] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false

    requestThumbnail(src, name).then((dataUrl) => {
      if (!cancelled && dataUrl) setThumbUrl(dataUrl)
    })

    return () => { cancelled = true }
  }, [src, name])

  if (thumbUrl) {
    return <img src={thumbUrl} alt={name} className="w-full h-full object-cover" />
  }

  return (
    <div className="w-full h-full bg-bg-active flex items-center justify-center">
      <Film size={14} className="text-text-muted animate-pulse" />
    </div>
  )
}

interface Props {
  activeIndex: number
  onThumbnailClick: (index: number) => void
}

function VirtualizedThumbnailList({ activeIndex, onThumbnailClick, onMobileClick }: Props & { onMobileClick?: () => void }) {
  const outputs = useStore(s => s.filteredOutputs())
  const containerRef = useRef<HTMLDivElement>(null)
  const [scrollTop, setScrollTop] = useState(0)
  const [viewHeight, setViewHeight] = useState(400)
  const isAutoScrolling = useRef(false)

  // Measure container height
  useEffect(() => {
    const el = containerRef.current
    if (!el) return
    const ro = new ResizeObserver(entries => {
      setViewHeight(entries[0].contentRect.height)
    })
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  const loadingMore = useRef(false)

  const handleScroll = useCallback(() => {
    const el = containerRef.current
    if (!el) return
    setScrollTop(el.scrollTop)
    // Infinite scroll: load more when near the bottom
    const distanceToBottom = el.scrollHeight - el.scrollTop - el.clientHeight
    if (distanceToBottom < el.clientHeight && !loadingMore.current) {
      const store = useStore.getState()
      if (store.outputs.length < store.outputsTotal) {
        loadingMore.current = true
        store.loadMoreOutputs().finally(() => { loadingMore.current = false })
      }
    }
  }, [])

  // Auto-scroll to keep active thumbnail visible
  useEffect(() => {
    const container = containerRef.current
    if (!container) return
    const itemTop = activeIndex * (THUMB_HEIGHT + THUMB_GAP)
    const itemBottom = itemTop + THUMB_HEIGHT
    const viewTop = container.scrollTop
    const viewBottom = viewTop + viewHeight

    if (itemTop < viewTop || itemBottom > viewBottom) {
      isAutoScrolling.current = true
      container.scrollTo({ top: Math.max(0, itemTop - viewHeight / 2 + THUMB_HEIGHT / 2), behavior: 'smooth' })
      setTimeout(() => { isAutoScrolling.current = false }, 400)
    }
  }, [activeIndex, viewHeight])

  const totalHeight = outputs.length * (THUMB_HEIGHT + THUMB_GAP) - (outputs.length > 0 ? THUMB_GAP : 0)

  const { startIdx, endIdx } = useMemo(() => {
    const itemSize = THUMB_HEIGHT + THUMB_GAP
    const start = Math.max(0, Math.floor(scrollTop / itemSize) - THUMB_OVERSCAN)
    const visibleCount = Math.ceil(viewHeight / itemSize)
    const end = Math.min(outputs.length, Math.floor(scrollTop / itemSize) + visibleCount + THUMB_OVERSCAN)
    return { startIdx: start, endIdx: end }
  }, [scrollTop, viewHeight, outputs.length])

  return (
    <div
      ref={containerRef}
      className="flex-1 overflow-y-auto w-[80px]"
      onScroll={handleScroll}
    >
      <div className="relative" style={{ height: totalHeight }}>
        {outputs.slice(startIdx, endIdx).map((file, i) => {
          const idx = startIdx + i
          return (
            <button
              key={file.name}
              data-thumb-index={idx}
              onClick={() => {
                onThumbnailClick(idx)
                onMobileClick?.()
              }}
              className={`absolute left-0 right-0 rounded-lg border overflow-hidden transition-all ${
                activeIndex === idx
                  // shadow-active-ring is empty in the default theme
                  // (no bloom — the existing border+ring covers it).
                  // In Golden Hour it adds an amber halo for the
                  // "alive/dynamic" feel from the reference render.
                  ? 'border-accent-blue ring-1 ring-accent-blue shadow-active-ring'
                  : 'border-border hover:border-border-light'
              }`}
              style={{
                top: idx * (THUMB_HEIGHT + THUMB_GAP),
                height: THUMB_HEIGHT,
              }}
            >
              {file.type === 'video' ? (
                <VideoThumbnail src={file.url} name={file.name} />
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

export function ThumbnailGallery({ activeIndex, onThumbnailClick }: Props) {
  const outputs = useStore(s => s.filteredOutputs())
  const outputsLoading = useStore(s => s.outputsLoading)
  const isMobile = useIsMobile()
  const [collapsed, setCollapsed] = useState(false)
  const [mobileOpen, setMobileOpen] = useState(false)

  if (outputs.length === 0 && !outputsLoading) return null

  // Mobile: overlay from right
  if (isMobile) {
    return (
      <>
        {/* Toggle button - fixed in bottom-right */}
        <button
          onClick={() => setMobileOpen(true)}
          className="fixed bottom-4 right-4 z-30 w-10 h-10 rounded-full bg-bg-secondary border border-border shadow-lg flex items-center justify-center text-text-secondary hover:text-text-primary transition-colors"
          title="Show thumbnails"
        >
          <PanelRightOpen size={18} />
        </button>

        {/* Overlay backdrop */}
        {mobileOpen && (
          <div
            className="fixed inset-0 bg-black/40 z-40"
            onClick={() => setMobileOpen(false)}
          />
        )}

        {/* Slide-in panel from right */}
        <aside className={`fixed top-0 right-0 h-full w-[100px] bg-bg-secondary border-l border-border z-50 flex flex-col transform transition-transform duration-300 ease-in-out ${
          mobileOpen ? 'translate-x-0' : 'translate-x-full'
        }`}>
          <div className="px-2 py-2 border-b border-border flex items-center justify-between">
            <span className="text-[10px] text-text-muted uppercase tracking-wider">History</span>
            <button
              onClick={() => setMobileOpen(false)}
              className="p-1 rounded hover:bg-bg-hover text-text-secondary"
            >
              <X size={14} />
            </button>
          </div>
          {mobileOpen && (
            <VirtualizedThumbnailList
              activeIndex={activeIndex}
              onThumbnailClick={onThumbnailClick}
              onMobileClick={() => setMobileOpen(false)}
            />
          )}
        </aside>
      </>
    )
  }

  // Desktop: collapsible sidebar
  return (
    <div className={`shrink-0 border-l border-border flex flex-col transition-all duration-200 ${
      collapsed ? 'w-0 overflow-hidden' : 'w-[80px]'
    }`}>
      {/* Collapse toggle */}
      <button
        onClick={() => setCollapsed(!collapsed)}
        className="absolute right-0 top-2 z-10 w-5 h-10 bg-bg-secondary border border-border border-r-0 rounded-l-md flex items-center justify-center text-text-muted hover:text-text-primary transition-colors"
        style={{ right: collapsed ? 0 : 80 }}
        title={collapsed ? 'Show thumbnails' : 'Hide thumbnails'}
      >
        {collapsed ? <PanelRightOpen size={12} /> : <PanelRightClose size={12} />}
      </button>

      {!collapsed && (
        <VirtualizedThumbnailList
          activeIndex={activeIndex}
          onThumbnailClick={onThumbnailClick}
        />
      )}
    </div>
  )
}
