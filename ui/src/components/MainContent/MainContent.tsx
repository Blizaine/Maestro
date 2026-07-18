import { useRef, useCallback, useState, useEffect, useMemo, type JSX } from 'react'
import { Film, Play, Square, FolderOpen, Plus, Check, Loader2, X, BookMarked, Upload, Trash2 } from 'lucide-react'
import { TabFilter } from './TabFilter'
import { ThumbnailGallery } from './ThumbnailGallery'
import { MediaFeedItem } from './MediaFeedItem'
import { useStore } from '../../stores/useStore'
import type { GenerationJob } from '../../types'

function WorkspaceSelector() {
  const workspaces = useStore(s => s.workspaces)
  const activeWorkspace = useStore(s => s.activeWorkspace)
  const browsingUploads = useStore(s => s.browsingUploads)
  const switchWorkspace = useStore(s => s.switchWorkspace)
  const createWorkspace = useStore(s => s.createWorkspace)
  const deleteWorkspace = useStore(s => s.deleteWorkspace)
  const [open, setOpen] = useState(false)
  const [creating, setCreating] = useState(false)
  const [newName, setNewName] = useState('')
  const [confirmDelete, setConfirmDelete] = useState<string | null>(null)
  const [deleting, setDeleting] = useState<string | null>(null)
  const [deleteError, setDeleteError] = useState<string | null>(null)
  const dropdownRef = useRef<HTMLDivElement>(null)

  const handleDelete = async (name: string, e: React.MouseEvent) => {
    e.stopPropagation()
    if (confirmDelete !== name) {
      setConfirmDelete(name)
      setTimeout(() => setConfirmDelete(c => (c === name ? null : c)), 4000)
      return
    }
    setConfirmDelete(null)
    setDeleting(name)
    setDeleteError(null)
    try {
      await deleteWorkspace(name)
    } catch (err) {
      setDeleteError(err instanceof Error ? err.message : String(err))
      setTimeout(() => setDeleteError(null), 6000)
    } finally {
      setDeleting(null)
    }
  }

  // Close on outside click
  useEffect(() => {
    if (!open) return
    const handler = (e: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target as Node)) {
        setOpen(false)
        setCreating(false)
      }
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [open])

  const handleCreate = async () => {
    const name = newName.trim().replace(/\s+/g, '-')
    if (!name) return
    try {
      await createWorkspace(name)
      setNewName('')
      setCreating(false)
      setOpen(false)
    } catch {
      // error logged in store
    }
  }

  return (
    <div className="relative" ref={dropdownRef}>
      <button
        onClick={() => setOpen(!open)}
        className="flex items-center gap-1.5 px-2 py-1 rounded-md text-xs text-text-secondary hover:text-text-primary hover:bg-bg-hover transition-colors border border-border"
        title="Switch workspace"
      >
        <FolderOpen size={12} />
        <span className="max-w-[120px] truncate">{browsingUploads ? 'Uploads' : activeWorkspace}</span>
      </button>

      {open && (
        <div className="absolute right-0 top-full mt-1 w-56 bg-bg-secondary border border-border rounded-lg shadow-lg z-50 overflow-hidden">
          <div className="px-2 py-1.5 border-b border-border">
            <span className="text-[10px] text-text-muted uppercase tracking-wider">Workspaces</span>
          </div>
          <div className="max-h-[200px] overflow-y-auto">
            {workspaces.map(ws => (
              <div key={ws.name} className="flex items-center group hover:bg-bg-hover transition-colors">
                <button
                  onClick={() => { switchWorkspace(ws.name); setOpen(false) }}
                  className={`flex-1 min-w-0 text-left px-3 py-2 text-xs flex items-center justify-between ${
                    ws.name === activeWorkspace && !browsingUploads ? 'text-accent-blue' : 'text-text-secondary'
                  }`}
                >
                  <span className="truncate">{ws.name}</span>
                  {ws.name === activeWorkspace && !browsingUploads && <Check size={12} className="shrink-0" />}
                </button>
                {/* default IS the outputs folder itself — not deletable */}
                {ws.name !== 'default' && (
                  <button
                    onClick={e => handleDelete(ws.name, e)}
                    disabled={deleting === ws.name}
                    className={`px-2 py-2 shrink-0 transition-colors ${
                      confirmDelete === ws.name
                        ? 'text-red-400 bg-red-500/15'
                        : deleting === ws.name
                          ? 'text-text-muted cursor-wait'
                          : 'text-text-muted opacity-0 group-hover:opacity-100 focus-visible:opacity-100 hover:text-red-400'
                    }`}
                    title={confirmDelete === ws.name
                      ? `Click again to permanently delete "${ws.name}" and its ${ws.file_count ?? 0} files`
                      : `Delete workspace (${ws.file_count ?? 0} files)`}
                  >
                    {deleting === ws.name ? <Loader2 size={12} className="animate-spin" /> : <Trash2 size={12} />}
                  </button>
                )}
              </div>
            ))}
          </div>
          {deleteError && (
            <div className="px-3 py-1.5 text-[10px] text-red-400 border-t border-border leading-snug">{deleteError}</div>
          )}
          {/* Virtual Uploads view — browse user-uploaded media (read-only;
              generations keep saving to the real active workspace). */}
          <div className="border-t border-border">
            <button
              onClick={() => { switchWorkspace('__uploads__'); setOpen(false) }}
              className={`w-full text-left px-3 py-2 text-xs flex items-center justify-between hover:bg-bg-hover transition-colors ${
                browsingUploads ? 'text-accent-blue' : 'text-text-secondary'
              }`}
              title="Browse media you've uploaded — reuse as inputs"
            >
              <span className="flex items-center gap-1.5"><Upload size={12} /> Uploads</span>
              {browsingUploads && <Check size={12} />}
            </button>
          </div>
          <div className="border-t border-border p-2">
            {creating ? (
              <div className="flex gap-1.5">
                <input
                  type="text"
                  value={newName}
                  onChange={e => setNewName(e.target.value)}
                  onKeyDown={e => e.key === 'Enter' && handleCreate()}
                  placeholder="workspace-name"
                  className="flex-1 bg-bg-tertiary border border-border rounded px-2 py-1 text-xs text-text-primary focus:outline-none focus:border-accent-blue"
                  autoFocus
                />
                <button
                  onClick={handleCreate}
                  disabled={!newName.trim()}
                  className="px-2 py-1 text-xs bg-accent-blue text-white rounded hover:bg-accent-blue-hover disabled:opacity-50"
                >
                  Create
                </button>
              </div>
            ) : (
              <button
                onClick={() => setCreating(true)}
                className="w-full text-left px-1 py-1 text-xs text-accent-blue hover:text-accent-blue-hover flex items-center gap-1"
              >
                <Plus size={12} /> New Workspace
              </button>
            )}
          </div>
        </div>
      )}
    </div>
  )
}

// How many items to render beyond the viewport in each direction
const OVERSCAN = 5
// Info bar height + border/padding
const INFO_BAR_HEIGHT = 48
// aspect-video = 56.25% of width (16:9)
const ASPECT_RATIO = 0.5625
// Gap between items (tailwind space-y-3 = 12px)
const GAP = 12

function stripTimeSuffix(msg: string): string {
  return msg.replace(/\s*\|\s*\d+:\d+.*$/, '').trim()
}

function JobPlaceholder({ job, onStop, onDismiss }: { job: GenerationJob; onStop: () => void; onDismiss: () => void }) {
  const hasSteps = job.totalSteps > 0
  const progressPct = hasSteps ? (job.step / job.totalSteps) * 100 : job.progress * 100
  const phase = stripTimeSuffix(job.phase || job.message)
  const isFailed = job.status === 'failed' || job.status === 'cancelled'
  const errorText = job.error || job.message || (job.status === 'cancelled' ? 'Cancelled' : 'Generation failed')

  return (
    <div className={`rounded-xl border overflow-hidden ${
      isFailed ? 'border-red-500/30 bg-bg-tertiary' : 'border-accent-blue/30 bg-bg-tertiary'
    }`}>
      <div className="w-full aspect-video flex items-center justify-center relative">
        {/* Dismiss button (top-right, failed only) */}
        {isFailed && (
          <button
            onClick={onDismiss}
            className="absolute top-2 right-2 p-1.5 rounded-full bg-bg-active text-text-secondary hover:bg-red-600 hover:text-white transition-colors z-10"
            title="Dismiss"
          >
            <X size={14} />
          </button>
        )}
        <div className="flex flex-col items-center gap-3 text-text-muted w-full max-w-md px-4">
          <Film size={40} className={isFailed ? 'text-red-400' : 'animate-pulse'} />

          <div className="text-center w-full">
            <p className={`text-sm font-medium ${isFailed ? 'text-red-400' : 'text-text-secondary'}`}>
              {isFailed ? (job.status === 'cancelled' ? 'Cancelled' : 'Generation Failed') : job.status === 'queued' ? 'Queued...' : 'Generating...'}
            </p>
            {!isFailed && phase && (
              <p className="text-xs mt-1 truncate">{phase}</p>
            )}
            {hasSteps && !isFailed && (
              <p className="text-[10px] text-text-muted mt-0.5">
                Step {job.step}/{job.totalSteps}
              </p>
            )}
            {isFailed && (
              <p className="text-[11px] text-text-secondary mt-2 max-h-24 overflow-y-auto px-2 leading-relaxed whitespace-pre-wrap break-words">
                {errorText}
              </p>
            )}
          </div>

          {/* Progress bar — hidden when failed */}
          {!isFailed && (
            <div className="w-full bg-bg-active rounded-full h-1.5 overflow-hidden">
              {progressPct > 0 ? (
                <div
                  className="h-full bg-accent-green rounded-full transition-all duration-300"
                  style={{ width: `${progressPct}%` }}
                />
              ) : (
                <div className="h-full bg-accent-green/60 rounded-full animate-pulse w-full" />
              )}
            </div>
          )}
        </div>
      </div>

      {/* Bottom bar */}
      <div className="px-3 py-2 min-h-[40px] flex items-center justify-between">
        <div className="text-[11px] text-text-muted truncate flex-1">
          {isFailed ? 'Click × to dismiss — the tile stays so you can see what failed' : phase || 'Preparing...'}
        </div>
        {!isFailed && (
          <button
            onClick={onStop}
            className="flex items-center gap-1 text-xs text-red-400 hover:text-red-300 transition-colors shrink-0 ml-2"
          >
            <Square size={11} />
            Stop
          </button>
        )}
      </div>
    </div>
  )
}

function PipelinePlaceholder() {
  const pipelineStatus = useStore(s => s.pipelineStatus)
  const pipelineId = useStore(s => s.pipelineId)
  const stopPipeline = useStore(s => s.stopPipeline)

  if (!pipelineId || !pipelineStatus) return null
  if (pipelineStatus.status === 'completed' || pipelineStatus.status === 'failed' || pipelineStatus.status === 'cancelled') return null

  const phase = pipelineStatus.phase || 'planning'
  const progress = pipelineStatus.progress
  const message = progress?.message || phase

  const hasSteps = (progress?.total_steps ?? 0) > 0
  const progressPct = hasSteps
    ? ((progress?.step ?? 0) / progress!.total_steps) * 100
    : progress && progress.total > 0
      ? (progress.current / progress.total) * 100
      : 0
  const phaseLabel = stripTimeSuffix(message)

  return (
    <div className="rounded-xl overflow-hidden border border-accent-blue/30 bg-bg-tertiary">
      <div className="w-full aspect-video flex items-center justify-center">
        <div className="flex flex-col items-center gap-3 text-text-muted w-full max-w-xs px-4">
          <Film size={40} className="animate-pulse" />

          <div className="text-center w-full">
            <p className="text-sm font-medium text-text-secondary">
              {pipelineStatus?.status === 'paused' ? 'Paused — Review' : 'Director'}
            </p>
            <p className="text-xs mt-1 truncate">{phaseLabel}</p>
            {hasSteps && (
              <p className="text-[10px] text-text-muted mt-0.5">
                Step {progress!.step}/{progress!.total_steps}
              </p>
            )}
          </div>

          {/* Progress bar */}
          <div className="w-full bg-bg-active rounded-full h-1.5 overflow-hidden">
            {progressPct > 0 ? (
              <div
                className="h-full bg-accent-green rounded-full transition-all duration-300"
                style={{ width: `${progressPct}%` }}
              />
            ) : (
              <div className="h-full bg-accent-green/60 rounded-full animate-pulse w-full" />
            )}
          </div>
        </div>
      </div>

      {/* Bottom bar with stop button */}
      <div className="px-3 py-2 min-h-[40px] flex items-center justify-between">
        <div className="text-[11px] text-text-muted truncate flex-1">
          {phaseLabel || 'Preparing...'}
        </div>
        <button
          onClick={() => stopPipeline()}
          className="flex items-center gap-1 text-xs text-red-400 hover:text-red-300 transition-colors shrink-0 ml-2"
        >
          <Square size={11} />
          Stop
        </button>
      </div>
    </div>
  )
}

export function MainContent() {
  const outputs = useStore(s => s.filteredOutputs())
  const outputsTotal = useStore(s => s.outputsTotal)
  const outputsLoading = useStore(s => s.outputsLoading)
  const jobs = useStore(s => s.jobs)
  const generationMode = useStore(s => s.generationMode)
  const stopGeneration = useStore(s => s.stopGeneration)
  const dismissJob = useStore(s => s.dismissJob)
  const setSelectedOutput = useStore(s => s.setSelectedOutput)

  const feedRef = useRef<HTMLDivElement>(null)
  const [activeIndex, setActiveIndex] = useState(0)
  const isUserScrolling = useRef(false)
  const scrollTargetIndex = useRef<number | null>(null)

  // Virtualization state
  const [scrollTop, setScrollTop] = useState(0)
  const [containerHeight, setContainerHeight] = useState(800)
  const [containerWidth, setContainerWidth] = useState(800)
  const measuredHeights = useRef<Map<number, number>>(new Map())

  // Dynamic estimated item height based on actual container width
  const estimatedItemHeight = Math.round(containerWidth * ASPECT_RATIO) + INFO_BAR_HEIGHT

  // Total height of all job placeholders at top
  const placeholderTotalHeight = jobs.length > 0
    ? jobs.length * estimatedItemHeight + (jobs.length - 1) * GAP + GAP
    : 0

  // Measure container on mount and resize; clear stale heights on width change
  useEffect(() => {
    const el = feedRef.current
    if (!el) return
    let prevWidth = 0
    const ro = new ResizeObserver((entries) => {
      const rect = entries[0].contentRect
      setContainerHeight(rect.height)
      const newWidth = rect.width
      setContainerWidth(newWidth)
      if (prevWidth && Math.abs(newWidth - prevWidth) > 2) {
        measuredHeights.current.clear()
      }
      prevWidth = newWidth
    })
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  const getItemHeight = useCallback((index: number) => {
    return measuredHeights.current.get(index) ?? estimatedItemHeight
  }, [estimatedItemHeight])

  const { startIndex, endIndex, totalHeight, itemOffsets } = useMemo(() => {
    const count = outputs.length
    const offsets: number[] = new Array(count)
    let cumulative = placeholderTotalHeight

    for (let i = 0; i < count; i++) {
      offsets[i] = cumulative
      cumulative += getItemHeight(i) + GAP
    }
    const total = cumulative - (count > 0 ? GAP : 0)

    let lo = 0, hi = count - 1
    const viewStart = scrollTop - OVERSCAN * estimatedItemHeight
    while (lo < hi) {
      const mid = (lo + hi) >>> 1
      if (offsets[mid] + getItemHeight(mid) < viewStart) lo = mid + 1
      else hi = mid
    }
    const start = Math.max(0, lo)

    const viewEnd = scrollTop + containerHeight + OVERSCAN * estimatedItemHeight
    let end = start
    while (end < count && offsets[end] < viewEnd) end++

    return {
      startIndex: start,
      endIndex: Math.min(end, count),
      totalHeight: Math.max(total, placeholderTotalHeight),
      itemOffsets: offsets,
    }
  }, [outputs.length, scrollTop, containerHeight, getItemHeight, placeholderTotalHeight, estimatedItemHeight])

  const [, setMeasureEpoch] = useState(0)
  const handleItemMeasured = useCallback((index: number, height: number) => {
    const prev = measuredHeights.current.get(index)
    if (prev !== height) {
      measuredHeights.current.set(index, height)
      setMeasureEpoch(e => e + 1)
    }
  }, [])

  const handleItemVisible = useCallback((index: number) => {
    if (scrollTargetIndex.current !== null) return
    setActiveIndex(index)
    if (isUserScrolling.current) {
      setSelectedOutput(index)
    }
  }, [setSelectedOutput])

  const handleThumbnailClick = useCallback((index: number) => {
    setSelectedOutput(index)
    setActiveIndex(index)
    scrollTargetIndex.current = index
    isUserScrolling.current = false
    const feedEl = feedRef.current
    if (!feedEl) return

    // ── Why this is two phases ──
    // The virtualizer only renders items inside [startIndex, endIndex].
    // Items outside that window have NEVER been measured — their height
    // is an estimate. Summing the estimates to compute an offset for a
    // distant target accumulates error linearly with distance: a click
    // 200 items away can land hundreds of px off.
    //
    // The previous implementation did a single smooth scrollTo to the
    // estimated offset. As items entered the viewport mid-animation,
    // they got measured and the total height shifted under the
    // animation, so the smooth scroll landed on the wrong item. The
    // 800ms guard then expired and the IntersectionObserver picked up
    // a wrong-active item → thumbnail strip auto-scrolled away from
    // what the user clicked → infinite oscillation.
    //
    // The fix:
    //   Phase 1: INSTANT jump to the estimated offset. This is allowed
    //            to be slightly wrong; its only job is to bring the
    //            target item into the virtualizer's render window so
    //            it actually mounts in the DOM.
    //   Phase 2: requestAnimationFrame wait until the DOM contains an
    //            element with `data-feed-index="${index}"`, then call
    //            scrollIntoView on it for pixel-precise alignment.
    //            By the time the element exists, its height has been
    //            measured, so this final align is accurate.
    //   Guard:   scrollTargetIndex.current is held until phase 2
    //            finishes (not a fixed timeout). handleItemVisible
    //            ignores intersection events while this is non-null,
    //            so no wrong-active leak through.
    //   Re-entrancy: a stale align loop checks scrollTargetIndex
    //            against its captured target on every frame and bails
    //            if a newer click overrode it.

    const estimatedOffset = placeholderTotalHeight +
      Array.from({ length: index }, (_, i) => getItemHeight(i) + GAP).reduce((a, b) => a + b, 0)
    feedEl.scrollTo({ top: estimatedOffset, behavior: 'auto' })

    const targetIndexAtStart = index
    let attempts = 0
    const MAX_ATTEMPTS = 30 // ~500ms at 60fps
    const align = () => {
      // Newer click overrode our target — bail.
      if (scrollTargetIndex.current !== targetIndexAtStart) return
      attempts++
      const targetEl = feedEl.querySelector(`[data-feed-index="${index}"]`) as HTMLElement | null
      if (targetEl) {
        targetEl.scrollIntoView({ behavior: 'auto', block: 'start' })
        // One more frame so any post-mount measurement settles
        // before we release the guard.
        requestAnimationFrame(() => {
          if (scrollTargetIndex.current === targetIndexAtStart) {
            scrollTargetIndex.current = null
          }
        })
      } else if (attempts < MAX_ATTEMPTS) {
        requestAnimationFrame(align)
      } else {
        // Item didn't mount within the budget — release the guard so
        // the user isn't stuck. Rare; happens if outputs.length changed
        // mid-flight or the index is out of range.
        if (scrollTargetIndex.current === targetIndexAtStart) {
          scrollTargetIndex.current = null
        }
      }
    }
    requestAnimationFrame(align)
  }, [setSelectedOutput, getItemHeight, placeholderTotalHeight])

  // Infinite scroll: load more when near the bottom
  const loadingMore = useRef(false)
  const handleFeedScroll = useCallback(() => {
    const el = feedRef.current
    if (!el) return
    setScrollTop(el.scrollTop)
    if (scrollTargetIndex.current === null) {
      isUserScrolling.current = true
    }
    // Trigger load-more when within 2 screens of the bottom
    const distanceToBottom = el.scrollHeight - el.scrollTop - el.clientHeight
    if (distanceToBottom < el.clientHeight * 2 && !loadingMore.current) {
      const store = useStore.getState()
      if (store.outputs.length < store.outputsTotal) {
        loadingMore.current = true
        store.loadMoreOutputs().finally(() => { loadingMore.current = false })
      }
    }
  }, [])

  useEffect(() => {
    measuredHeights.current.clear()
  }, [outputs.length])

  const visibleItems = useMemo(() => {
    const items: JSX.Element[] = []
    for (let i = startIndex; i < endIndex; i++) {
      const file = outputs[i]
      if (!file) continue
      items.push(
        <MediaFeedItem
          key={file.name}
          file={file}
          index={i}
          isActive={activeIndex === i}
          onVisible={handleItemVisible}
          onMeasured={handleItemMeasured}
          style={{
            position: 'absolute',
            top: itemOffsets[i],
            left: 0,
            right: 0,
          }}
        />
      )
    }
    return items
  }, [startIndex, endIndex, outputs, activeIndex, handleItemVisible, handleItemMeasured, itemOffsets])

  return (
    <main className="flex-1 flex flex-col h-full overflow-hidden">
      {/* Top bar */}
      <div className="px-2 md:px-6 py-2 md:py-3 border-b border-border flex items-center justify-between gap-2">
        <TabFilter />
        <div className="flex items-center gap-2 shrink-0">
          <div className="text-[10px] md:text-xs text-text-muted hidden md:block">
            {outputsTotal > outputs.length
              ? `${outputs.length} / ${outputsTotal} items`
              : `${outputs.length} ${outputs.length === 1 ? 'item' : 'items'}`}
          </div>
          <WorkspaceSelector />
        </div>
      </div>

      {/* Content area: feed + thumbnails */}
      <div className="flex-1 flex flex-row gap-0 overflow-hidden relative">
        {/* Scrollable media feed */}
        <div
          ref={feedRef}
          className="flex-1 overflow-y-auto p-3 md:p-4"
          onScroll={handleFeedScroll}
        >
          {/* Pipeline + Job placeholders at top (not virtualized — small count) */}
          <div className="space-y-3 mb-3">
            <PipelinePlaceholder />
            {jobs.map((j, i) => (
              <JobPlaceholder
                key={j.id || `pending-${i}`}
                job={j}
                onStop={() => stopGeneration(j.id)}
                onDismiss={() => dismissJob(j.id)}
              />
            ))}
          </div>

          {/* Position container for virtualized output items */}
          <div className="relative" style={{ height: totalHeight - placeholderTotalHeight }}>
            {visibleItems.map(item => {
              // Adjust top positions to be relative to this container (subtract placeholder height)
              const adjustedStyle = {
                ...item.props.style,
                top: (item.props.style?.top as number) - placeholderTotalHeight,
              }
              return { ...item, props: { ...item.props, style: adjustedStyle } }
            })}
          </div>

          {/* Loading state */}
          {outputsLoading && outputs.length === 0 && (
            <div className="flex items-center justify-center min-h-[300px]">
              <div className="flex flex-col items-center gap-3 text-text-muted">
                <Loader2 size={24} className="animate-spin text-accent-blue" />
                <p className="text-sm">Indexing workspace...</p>
              </div>
            </div>
          )}

          {/* Empty state — first-run quick start. Teaches the three steps
              to a first generation and sets the one expectation that most
              surprises new users: the first run of each model downloads
              its weights (tens of GB) before anything appears. */}
          {!outputsLoading && outputs.length === 0 && jobs.length === 0 && (() => {
            const noun = generationMode === 'image' ? 'images'
              : generationMode === 'audio' ? 'audio' : 'videos'
            const example = generationMode === 'image'
              ? 'a neon city street at night, cinematic'
              : generationMode === 'audio'
              ? 'a dreamy synthwave track about the ocean'
              : 'a golden retriever surfing a big wave, slow motion'
            return (
              <div className="flex items-center justify-center min-h-[300px] px-6">
                <div className="flex flex-col items-center gap-4 text-center max-w-sm">
                  <div className="w-16 h-16 rounded-2xl bg-bg-active flex items-center justify-center text-text-muted">
                    <Play size={24} />
                  </div>
                  <p className="text-sm text-text-secondary">Your generated {noun} will appear here.</p>
                  <ol className="text-xs text-text-muted space-y-1.5 text-left">
                    <li><span className="text-accent-blue font-medium">1.</span> Pick a model in the sidebar (a good default is already selected).</li>
                    <li><span className="text-accent-blue font-medium">2.</span> Type a prompt — e.g. <span className="text-text-secondary italic">“{example}”</span></li>
                    <li><span className="text-accent-blue font-medium">3.</span> Hit Generate.</li>
                  </ol>
                  <p className="text-[11px] text-text-muted leading-snug">
                    Heads up: the first time you use a model, its weights download
                    once (often tens of GB) before generation starts — later runs
                    are fast. Progress shows at the bottom-right.
                  </p>
                  <button
                    onClick={() => useStore.getState().setRecipesOpen(true)}
                    className="mt-1 flex items-center gap-1.5 px-3 py-1.5 text-xs bg-accent-blue/10 border border-accent-blue/30 rounded-lg text-accent-blue hover:bg-accent-blue/20 transition-colors"
                  >
                    <BookMarked size={13} /> Browse recipes
                  </button>
                </div>
              </div>
            )
          })()}
        </div>

        {/* Thumbnail sidebar */}
        <ThumbnailGallery
          activeIndex={activeIndex}
          onThumbnailClick={handleThumbnailClick}
        />
      </div>
    </main>
  )
}
