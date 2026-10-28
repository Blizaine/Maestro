import { useCallback, useEffect, useRef, useState } from 'react'
import {
  Film,
  FolderOpen,
  RotateCcw,
  SlidersHorizontal,
  X,
} from 'lucide-react'
import { EditorInspector } from './EditorInspector'
import { EditorLeftPanel } from './EditorLeftPanel'
import { EditorPreview } from './EditorPreview'
import { EditorTimeline } from './EditorTimeline'
import { useStore } from '../stores/useStore'
import { useEditorStore } from './useEditorStore'

const LAYOUT_STORAGE_KEY = 'maestro-editor-layout-v1'
const LEFT_WIDTH_MIN = 232
const LEFT_WIDTH_MAX = 420
const RIGHT_WIDTH_MIN = 264
const RIGHT_WIDTH_MAX = 480
const TIMELINE_HEIGHT_MIN = 180
const TIMELINE_HEIGHT_MAX = 520
const RESIZE_HANDLE_SIZE = 8

interface EditorLayoutPreferences {
  leftWidth: number
  rightWidth: number
  timelineHeight: number
}

type ResizeKind = 'left' | 'right' | 'timeline'

const DEFAULT_LAYOUT: EditorLayoutPreferences = {
  leftWidth: 292,
  rightWidth: 318,
  timelineHeight: 290,
}

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.max(minimum, Math.min(maximum, value))
}

function readLayoutPreferences(): EditorLayoutPreferences {
  if (typeof window === 'undefined') return DEFAULT_LAYOUT
  try {
    const raw = window.localStorage.getItem(LAYOUT_STORAGE_KEY)
    if (!raw) return DEFAULT_LAYOUT
    const parsed = JSON.parse(raw) as Partial<EditorLayoutPreferences>
    return {
      leftWidth: clamp(Number(parsed.leftWidth) || DEFAULT_LAYOUT.leftWidth, LEFT_WIDTH_MIN, LEFT_WIDTH_MAX),
      rightWidth: clamp(Number(parsed.rightWidth) || DEFAULT_LAYOUT.rightWidth, RIGHT_WIDTH_MIN, RIGHT_WIDTH_MAX),
      timelineHeight: clamp(Number(parsed.timelineHeight) || DEFAULT_LAYOUT.timelineHeight, TIMELINE_HEIGHT_MIN, TIMELINE_HEIGHT_MAX),
    }
  } catch {
    return DEFAULT_LAYOUT
  }
}

function useWideEditorLayout(): boolean {
  const subscribe = useCallback((notify: () => void) => {
    const query = window.matchMedia('(min-width: 1280px)')
    query.addEventListener('change', notify)
    return () => query.removeEventListener('change', notify)
  }, [])
  const getSnapshot = useCallback(() => window.matchMedia('(min-width: 1280px)').matches, [])
  return useSyncExternalStoreCompat(subscribe, getSnapshot)
}

// Keeping this tiny wrapper local avoids adding another shared breakpoint hook
// for a single Editor-specific layout threshold.
function useSyncExternalStoreCompat(subscribe: (notify: () => void) => () => void, getSnapshot: () => boolean): boolean {
  const [value, setValue] = useState(getSnapshot)
  useEffect(() => {
    const update = () => setValue(getSnapshot())
    const unsubscribe = subscribe(update)
    update()
    return unsubscribe
  }, [getSnapshot, subscribe])
  return value
}

function ResizeSeparator({
  kind,
  value,
  minimum,
  maximum,
  onPointerDown,
  onPointerMove,
  onPointerUp,
  onKeyDown,
  inert = false,
}: {
  kind: ResizeKind
  value: number
  minimum: number
  maximum: number
  onPointerDown: (event: React.PointerEvent<HTMLDivElement>, kind: ResizeKind) => void
  onPointerMove: (event: React.PointerEvent<HTMLDivElement>, kind: ResizeKind) => void
  onPointerUp: (event: React.PointerEvent<HTMLDivElement>, kind: ResizeKind) => void
  onKeyDown: (event: React.KeyboardEvent<HTMLDivElement>, kind: ResizeKind) => void
  inert?: boolean
}) {
  const vertical = kind !== 'timeline'
  const label = kind === 'left' ? 'Resize Media panel' : kind === 'right' ? 'Resize Inspector panel' : 'Resize timeline height'
  return (
    <div
      role="separator"
      aria-label={label}
      aria-orientation={vertical ? 'vertical' : 'horizontal'}
      aria-valuemin={minimum}
      aria-valuemax={maximum}
      aria-valuenow={Math.round(value)}
      aria-valuetext={`${Math.round(value)} pixels`}
      tabIndex={0}
      inert={inert}
      data-testid={`editor-resize-${kind}`}
      className={`group relative z-20 shrink-0 touch-none outline-none focus-visible:bg-accent-blue/15 ${vertical ? 'w-2 cursor-col-resize' : 'h-2 cursor-row-resize'}`}
      onPointerDown={event => onPointerDown(event, kind)}
      onPointerMove={event => onPointerMove(event, kind)}
      onPointerUp={event => onPointerUp(event, kind)}
      onPointerCancel={event => onPointerUp(event, kind)}
      onLostPointerCapture={event => onPointerUp(event, kind)}
      onKeyDown={event => onKeyDown(event, kind)}
    >
      <span className={`absolute rounded-full bg-border transition-colors group-hover:bg-accent-blue/60 group-focus-visible:bg-accent-blue ${vertical ? 'inset-y-0 left-[3px] w-px' : 'inset-x-0 top-[3px] h-px'}`} />
      <span className={`absolute rounded-full bg-transparent transition-colors group-hover:bg-accent-blue/10 ${vertical ? 'inset-y-0 inset-x-0' : 'inset-x-0 inset-y-0'}`} />
    </div>
  )
}

export function EditorLayout({ isMobile }: { isMobile: boolean }) {
  const isWide = useWideEditorLayout()
  const [layout, setLayout] = useState(readLayoutPreferences)
  const [inspectorOpen, setInspectorOpen] = useState(false)
  const [lastMobileMode, setLastMobileMode] = useState(isMobile)
  const [layoutRootHeight, setLayoutRootHeight] = useState(640)
  const layoutRootRef = useRef<HTMLDivElement>(null)
  const inspectorTriggerRef = useRef<HTMLButtonElement>(null)
  const inspectorCloseRef = useRef<HTMLButtonElement>(null)
  const wasInspectorOpen = useRef(false)
  const activePointer = useRef<{ kind: ResizeKind; pointerId: number } | null>(null)
  const mobilePanel = useEditorStore(state => state.mobilePanel)
  const setMobilePanel = useEditorStore(state => state.setMobilePanel)
  const setSidebarOpen = useStore(state => state.setSidebarOpen)
  const compactInspector = !isMobile && !isWide
  const minimumTimelineHeight = Math.min(TIMELINE_HEIGHT_MIN, Math.max(90, layoutRootHeight * 0.4))
  const maxTimelineHeight = Math.max(minimumTimelineHeight, Math.min(TIMELINE_HEIGHT_MAX, layoutRootHeight - 170))
  const visibleTimelineHeight = Math.min(layout.timelineHeight, maxTimelineHeight)

  useEffect(() => {
    try {
      window.localStorage.setItem(LAYOUT_STORAGE_KEY, JSON.stringify(layout))
    } catch {
      // The layout still works in sessions where browser storage is disabled.
    }
  }, [layout])

  useEffect(() => {
    const root = layoutRootRef.current
    if (!root) return
    const updateHeight = () => setLayoutRootHeight(root.clientHeight || window.innerHeight)
    updateHeight()
    const observer = new ResizeObserver(updateHeight)
    observer.observe(root)
    return () => observer.disconnect()
  }, [isMobile])

  // Reset this local drawer when the layout changes, before rendering children.
  if (lastMobileMode !== isMobile) {
    setLastMobileMode(isMobile)
    if (isMobile) setInspectorOpen(false)
  }

  useEffect(() => {
    if (isWide || isMobile) return
    if (inspectorOpen) {
      wasInspectorOpen.current = true
      inspectorCloseRef.current?.focus()
    } else if (wasInspectorOpen.current) {
      wasInspectorOpen.current = false
      inspectorTriggerRef.current?.focus()
    }
  }, [inspectorOpen, isMobile, isWide])

  useEffect(() => {
    if (!inspectorOpen || isWide || isMobile) return
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        event.preventDefault()
        event.stopPropagation()
        setInspectorOpen(false)
        return
      }
      if (event.key !== 'Tab') return
      const dialog = document.querySelector<HTMLElement>('[data-testid="editor-inspector-drawer"]')
      if (!dialog) return
      const controls = Array.from(dialog.querySelectorAll<HTMLElement>(
        'a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])',
      )).filter(element => element.offsetParent !== null)
      if (!controls.length) return
      const first = controls[0]
      const last = controls[controls.length - 1]
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault()
        last.focus()
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault()
        first.focus()
      }
    }
    document.addEventListener('keydown', handleKeyDown)
    return () => document.removeEventListener('keydown', handleKeyDown)
  }, [inspectorOpen, isMobile, isWide])

  const updateSize = (kind: ResizeKind, value: number) => {
    setLayout(current => {
      if (kind === 'left') return { ...current, leftWidth: clamp(value, LEFT_WIDTH_MIN, LEFT_WIDTH_MAX) }
      if (kind === 'right') return { ...current, rightWidth: clamp(value, RIGHT_WIDTH_MIN, RIGHT_WIDTH_MAX) }
      return { ...current, timelineHeight: clamp(value, minimumTimelineHeight, maxTimelineHeight) }
    })
  }

  const handlePointerDown = (event: React.PointerEvent<HTMLDivElement>, kind: ResizeKind) => {
    if (event.button !== 0) return
    event.preventDefault()
    activePointer.current = { kind, pointerId: event.pointerId }
    event.currentTarget.setPointerCapture(event.pointerId)
  }

  const handlePointerMove = (event: React.PointerEvent<HTMLDivElement>, kind: ResizeKind) => {
    const active = activePointer.current
    if (!active || active.kind !== kind || active.pointerId !== event.pointerId) return
    const bounds = layoutRootRef.current?.getBoundingClientRect()
    if (!bounds) return
    if (kind === 'left') {
      updateSize('left', event.clientX - bounds.left - RESIZE_HANDLE_SIZE / 2)
    } else if (kind === 'right') {
      updateSize('right', bounds.right - event.clientX - RESIZE_HANDLE_SIZE / 2)
    } else {
      updateSize('timeline', bounds.bottom - event.clientY - RESIZE_HANDLE_SIZE / 2)
    }
  }

  const handlePointerUp = (event: React.PointerEvent<HTMLDivElement>, kind: ResizeKind) => {
    if (activePointer.current?.kind === kind && activePointer.current.pointerId === event.pointerId) {
      activePointer.current = null
    }
  }

  const handleSeparatorKeyDown = (event: React.KeyboardEvent<HTMLDivElement>, kind: ResizeKind) => {
    const step = event.shiftKey ? 32 : 16
    let value: number | null = null
    if (kind === 'left') {
      if (event.key === 'ArrowLeft') value = layout.leftWidth - step
      if (event.key === 'ArrowRight') value = layout.leftWidth + step
      if (event.key === 'Home') value = LEFT_WIDTH_MIN
      if (event.key === 'End') value = LEFT_WIDTH_MAX
    } else if (kind === 'right') {
      if (event.key === 'ArrowLeft') value = layout.rightWidth + step
      if (event.key === 'ArrowRight') value = layout.rightWidth - step
      if (event.key === 'Home') value = RIGHT_WIDTH_MIN
      if (event.key === 'End') value = RIGHT_WIDTH_MAX
    } else {
      if (event.key === 'ArrowUp') value = layout.timelineHeight + step
      if (event.key === 'ArrowDown') value = layout.timelineHeight - step
      if (event.key === 'Home') value = TIMELINE_HEIGHT_MIN
      if (event.key === 'End') value = maxTimelineHeight
    }
    if (value === null) return
    event.preventDefault()
    updateSize(kind, value)
  }

  const resetLayout = () => {
    setLayout(DEFAULT_LAYOUT)
    setInspectorOpen(false)
  }

  const openMobilePanel = (panel: 'media' | 'timeline' | 'inspector') => {
    setMobilePanel(panel)
    setSidebarOpen(panel !== 'timeline')
  }

  return (
    <div className="relative flex min-h-0 min-w-0 flex-1 flex-col overflow-hidden bg-bg-primary" data-testid="editor-layout" data-layout-mode={isMobile ? 'mobile' : isWide ? 'desktop' : 'intermediate'}>
      <div className="flex h-9 shrink-0 items-center justify-between border-b border-border/80 bg-bg-primary px-3">
        <div className="flex min-w-0 items-center gap-2">
          <Film size={13} className="shrink-0 text-accent-blue" />
          <span className="truncate text-[10px] font-semibold uppercase tracking-[0.12em] text-text-secondary">Editing workspace</span>
          {!isMobile && <span className="hidden text-[9px] text-text-muted sm:inline">Media · Preview · Inspector</span>}
        </div>
        <div className="flex shrink-0 items-center gap-1.5">
          {compactInspector && (
            <button
              ref={inspectorTriggerRef}
              type="button"
              onClick={() => setInspectorOpen(true)}
              aria-haspopup="dialog"
              aria-expanded={inspectorOpen}
              data-testid="editor-inspector-trigger"
              className="flex h-7 items-center gap-1.5 rounded-md border border-border px-2 text-[9px] font-medium text-text-secondary transition-colors hover:border-accent-blue/40 hover:bg-bg-hover hover:text-text-primary focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent-blue"
            >
              <SlidersHorizontal size={11} /> Adjust
            </button>
          )}
          {!isMobile && (
            <button
              type="button"
              onClick={resetLayout}
              aria-label="Reset Editor layout"
              title="Reset panel sizes to the default layout"
              data-testid="editor-layout-reset"
              className="flex h-7 items-center gap-1.5 rounded-md px-2 text-[9px] text-text-muted transition-colors hover:bg-bg-hover hover:text-text-primary focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent-blue"
            >
              <RotateCcw size={11} /> <span className="hidden sm:inline">Reset layout</span>
            </button>
          )}
        </div>
      </div>

      {isMobile ? (
        <>
          <div className="flex min-h-0 flex-1 flex-col [@media(orientation:landscape)_and_(max-height:500px)]:flex-row">
            <section className="relative min-h-[190px] min-w-0 flex-[0.9] [@media(orientation:landscape)_and_(max-height:500px)]:min-h-0" data-testid="editor-preview-pane">
              <EditorPreview />
            </section>
            <section className="min-h-[150px] min-w-0 flex-[1.1] border-t border-border [@media(orientation:landscape)_and_(max-height:500px)]:min-h-0 [@media(orientation:landscape)_and_(max-height:500px)]:border-l" data-testid="editor-timeline-pane">
              <EditorTimeline compact fill />
            </section>
          </div>
          <nav aria-label="Editor workspace panels" className="grid h-14 shrink-0 grid-cols-3 border-t border-border bg-bg-secondary px-2" style={{ paddingBottom: 'env(safe-area-inset-bottom)' }}>
            {([
              ['media', FolderOpen, 'Media'],
              ['timeline', Film, 'Timeline'],
              ['inspector', SlidersHorizontal, 'Adjust'],
            ] as const).map(([panel, Icon, label]) => (
              <button
                key={panel}
                type="button"
                onClick={() => openMobilePanel(panel)}
                aria-pressed={mobilePanel === panel}
                data-testid={`editor-mobile-${panel}`}
                className={`flex min-h-11 flex-col items-center justify-center gap-0.5 rounded-lg px-3 text-[10px] font-medium focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent-blue ${mobilePanel === panel ? 'text-accent-blue' : 'text-text-secondary hover:bg-bg-hover hover:text-text-primary'}`}
              >
                <Icon size={16} />
                {label}
              </button>
            ))}
          </nav>
          <EditorLeftPanel mobile />
        </>
      ) : (
        <div ref={layoutRootRef} className="relative flex min-h-0 flex-1 flex-col" data-testid="editor-layout-resizable">
          <div className="flex min-h-0 flex-1" inert={inspectorOpen && compactInspector}>
            <div className="min-h-0 shrink-0 overflow-hidden" style={{ width: layout.leftWidth }} data-testid="editor-media-pane">
              <EditorLeftPanel />
            </div>
            <ResizeSeparator
              kind="left"
              value={layout.leftWidth}
              minimum={LEFT_WIDTH_MIN}
              maximum={LEFT_WIDTH_MAX}
              onPointerDown={handlePointerDown}
              onPointerMove={handlePointerMove}
              onPointerUp={handlePointerUp}
              onKeyDown={handleSeparatorKeyDown}
            />
            <section className="relative flex min-h-0 min-w-0 flex-1 flex-col" data-testid="editor-preview-pane">
              <EditorPreview />
            </section>
            {isWide && (
              <>
                <ResizeSeparator
                  kind="right"
                  value={layout.rightWidth}
                  minimum={RIGHT_WIDTH_MIN}
                  maximum={RIGHT_WIDTH_MAX}
                  onPointerDown={handlePointerDown}
                  onPointerMove={handlePointerMove}
                  onPointerUp={handlePointerUp}
                  onKeyDown={handleSeparatorKeyDown}
                />
                <div className="min-h-0 shrink-0 overflow-hidden border-l border-border" style={{ width: layout.rightWidth }} data-testid="editor-inspector-pane">
                  <EditorInspector compact />
                </div>
              </>
            )}
          </div>

          <ResizeSeparator
            kind="timeline"
            value={visibleTimelineHeight}
            minimum={minimumTimelineHeight}
            maximum={maxTimelineHeight}
            onPointerDown={handlePointerDown}
            onPointerMove={handlePointerMove}
            onPointerUp={handlePointerUp}
            onKeyDown={handleSeparatorKeyDown}
            inert={inspectorOpen && compactInspector}
          />
          <section className="min-h-0 shrink-0 overflow-hidden" style={{ height: visibleTimelineHeight }} data-testid="editor-timeline-pane" inert={inspectorOpen && compactInspector}>
            <EditorTimeline fill />
          </section>

          {compactInspector && inspectorOpen && (
            <div className="absolute inset-0 z-50" data-testid="editor-inspector-overlay">
              <button
                type="button"
                aria-label="Close Adjust panel"
                className="absolute inset-0 cursor-default bg-black/45"
                onClick={() => setInspectorOpen(false)}
              />
              <aside
                role="dialog"
                aria-modal="true"
                aria-labelledby="editor-inspector-drawer-title"
                data-testid="editor-inspector-drawer"
                className="absolute inset-y-0 right-0 flex flex-col overflow-hidden border-l border-border bg-bg-secondary shadow-2xl"
                style={{ width: 'min(360px, calc(100vw - 1rem))', bottom: visibleTimelineHeight + RESIZE_HANDLE_SIZE }}
              >
                <div className="flex h-10 shrink-0 items-center justify-between border-b border-border px-3">
                  <h2 id="editor-inspector-drawer-title" className="text-[10px] font-semibold uppercase tracking-[0.12em] text-text-secondary">Adjust</h2>
                  <button
                    ref={inspectorCloseRef}
                    type="button"
                    onClick={() => setInspectorOpen(false)}
                    aria-label="Close Adjust panel"
                    className="grid h-8 w-8 place-items-center rounded-md text-text-secondary hover:bg-bg-hover hover:text-text-primary focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent-blue"
                  >
                    <X size={14} />
                  </button>
                </div>
                <div className="min-h-0 flex-1 overflow-hidden">
                  <EditorInspector compact />
                </div>
              </aside>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
