import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import {
  AudioLines,
  ClipboardCopy,
  ClipboardPaste,
  Copy,
  Film,
  Lock,
  Link2,
  Magnet,
  MapPin,
  Plus,
  Scissors,
  ScanLine,
  StepBack,
  StepForward,
  TextCursorInput,
  Trash2,
  Unlink2,
  Unlock,
  Volume2,
  VolumeX,
  Waves,
  ZoomIn,
  ZoomOut,
} from 'lucide-react'
import type { EditorAsset, EditorTimelineItem, EditorTrack } from '../types'
import {
  closestAvailableEditorStart,
  editorProjectDuration,
  formatEditorTime,
  nextEditorItemStart,
  previousEditorItemEnd,
} from './editorUtils'
import { EditorClipMedia } from './EditorClipMedia'
import { useEditorStore } from './useEditorStore'

const TRACK_HEADER_WIDTH = 132
const TRACK_HEIGHT = 46

type DragMode = 'move' | 'trim-start' | 'trim-end' | 'fade-in' | 'fade-out'
interface DragState {
  itemId: string
  mode: DragMode
  sourceTrackId: string
  previewTrackId: string
  pointerStart: number
  originalStart: number
  originalEnd: number
  previewStart: number
  previewEnd: number
  originalFadeIn: number
  originalFadeOut: number
  previewFadeIn: number
  previewFadeOut: number
}

interface MarqueeState {
  startX: number
  startY: number
  currentX: number
  currentY: number
  baselineItemIds: string[]
  trackId: string
  laneLeft: number
}

interface TimelinePinchState {
  startDistance: number
  startZoom: number
  anchorTime: number
  anchorViewportX: number
}

function TrackIcon({ type }: { type: EditorTrack['type'] }) {
  if (type === 'audio') return <AudioLines size={12} />
  if (type === 'text') return <TextCursorInput size={12} />
  return <Film size={12} />
}

function clipClasses(type: EditorTrack['type']): string {
  if (type === 'audio') return 'border-chip-purple/50 bg-chip-purple/20 text-chip-purple'
  if (type === 'text') return 'border-chip-orange/50 bg-chip-orange/20 text-chip-orange'
  return 'border-accent-blue/50 bg-accent-blue/20 text-accent-blue'
}

export function EditorTimeline({ compact = false }: { compact?: boolean }) {
  const scrollRef = useRef<HTMLDivElement>(null)
  const timelineContentRef = useRef<HTMLDivElement>(null)
  const trackMenuButtonRef = useRef<HTMLButtonElement>(null)
  const touchPointersRef = useRef(new Map<number, { x: number; y: number }>())
  const pinchRef = useRef<TimelinePinchState | null>(null)
  const [drag, setDrag] = useState<DragState | null>(null)
  const [marquee, setMarquee] = useState<MarqueeState | null>(null)
  const [playheadPointerId, setPlayheadPointerId] = useState<number | null>(null)
  const [trackMenuOpen, setTrackMenuOpen] = useState(false)
  const [trackMenuPosition, setTrackMenuPosition] = useState({ left: 8, top: 8 })
  const project = useEditorStore(state => state.project)
  const selectedItemId = useEditorStore(state => state.selectedItemId)
  const selectedItemIds = useEditorStore(state => state.selectedItemIds)
  const selectedTrackId = useEditorStore(state => state.selectedTrackId)
  const playhead = useEditorStore(state => state.playhead)
  const pixelsPerSecond = useEditorStore(state => state.pixelsPerSecond)
  const snapping = useEditorStore(state => state.snapping)
  const ripple = useEditorStore(state => state.ripple)
  const setPlayhead = useEditorStore(state => state.setPlayhead)
  const setPlaying = useEditorStore(state => state.setPlaying)
  const selectItem = useEditorStore(state => state.selectItem)
  const selectItems = useEditorStore(state => state.selectItems)
  const setPixelsPerSecond = useEditorStore(state => state.setPixelsPerSecond)
  const setSnapping = useEditorStore(state => state.setSnapping)
  const setRipple = useEditorStore(state => state.setRipple)
  const splitSelected = useEditorStore(state => state.splitSelected)
  const duplicateSelected = useEditorStore(state => state.duplicateSelected)
  const copySelected = useEditorStore(state => state.copySelected)
  const pasteClipboard = useEditorStore(state => state.pasteClipboard)
  const clipboard = useEditorStore(state => state.clipboard)
  const deleteSelected = useEditorStore(state => state.deleteSelected)
  const jumpToEdit = useEditorStore(state => state.jumpToEdit)
  const moveItem = useEditorStore(state => state.moveItem)
  const trimItem = useEditorStore(state => state.trimItem)
  const toggleTrackMute = useEditorStore(state => state.toggleTrackMute)
  const toggleTrackLock = useEditorStore(state => state.toggleTrackLock)
  const addTrack = useEditorStore(state => state.addTrack)
  const removeTrack = useEditorStore(state => state.removeTrack)
  const addMedia = useEditorStore(state => state.addMedia)
  const updateItem = useEditorStore(state => state.updateItem)
  const linkSelected = useEditorStore(state => state.linkSelected)
  const unlinkSelected = useEditorStore(state => state.unlinkSelected)
  const addMarker = useEditorStore(state => state.addMarker)
  const updateMarker = useEditorStore(state => state.updateMarker)
  const removeMarker = useEditorStore(state => state.removeMarker)

  const zoomToFit = () => {
    const viewportWidth = Math.max(200, (scrollRef.current?.clientWidth || 900) - TRACK_HEADER_WIDTH - 24)
    const fitted = viewportWidth / Math.max(12, duration + 2)
    setPixelsPerSecond(Math.max(20, Math.min(180, fitted)))
    if (scrollRef.current) scrollRef.current.scrollLeft = 0
  }

  const duration = editorProjectDuration(project)
  const contentDuration = Math.max(12, duration + 4)
  const contentWidth = Math.max(compact ? 620 : 900, contentDuration * pixelsPerSecond)
  const trackHeight = compact ? 58 : TRACK_HEIGHT
  const tickStep = pixelsPerSecond >= 90 ? 1 : pixelsPerSecond >= 45 ? 2 : 5
  const ticks = useMemo(() => (
    Array.from({ length: Math.min(1200, Math.ceil(contentDuration / tickStep) + 1) }, (_, index) => index * tickStep)
  ), [contentDuration, tickStep])
  const scrubPlayhead = useCallback((clientX: number) => {
    const rect = timelineContentRef.current?.getBoundingClientRect()
    if (!rect) return
    const seconds = (clientX - rect.left - TRACK_HEADER_WIDTH) / pixelsPerSecond
    setPlayhead(Math.max(0, Math.min(contentDuration, seconds)))
  }, [contentDuration, pixelsPerSecond, setPlayhead])
  const displayTracks = useMemo(() => {
    if (!project) return []
    const groupOrder: Record<EditorTrack['type'], number> = { text: 2, video: 1, audio: 0 }
    const originalOrder = new Map(project.tracks.map((track, index) => [track.id, index]))
    return [...project.tracks].sort((left, right) => (
      groupOrder[right.type] - groupOrder[left.type]
      || right.z_index - left.z_index
      || (originalOrder.get(left.id) || 0) - (originalOrder.get(right.id) || 0)
    ))
  }, [project])

  useEffect(() => {
    if (!trackMenuOpen) return
    const closeMenu = (event: PointerEvent) => {
      const target = event.target as Node | null
      if (target && trackMenuButtonRef.current?.contains(target)) return
      if (target instanceof Element && target.closest('[data-editor-track-menu]')) return
      setTrackMenuOpen(false)
    }
    const closeForViewportChange = () => setTrackMenuOpen(false)
    window.addEventListener('pointerdown', closeMenu)
    window.addEventListener('resize', closeForViewportChange)
    window.addEventListener('scroll', closeForViewportChange, true)
    return () => {
      window.removeEventListener('pointerdown', closeMenu)
      window.removeEventListener('resize', closeForViewportChange)
      window.removeEventListener('scroll', closeForViewportChange, true)
    }
  }, [trackMenuOpen])

  const snapTime = useCallback((raw: number, itemId: string, mode: DragMode, span: number): number => {
    if (!project) return Math.max(0, raw)
    const frame = 1 / Math.max(1, project.canvas.fps)
    let value = Math.round(Math.max(0, raw) / frame) * frame
    if (!snapping) return value
    const points = [0, playhead, ...(project.markers || []).map(marker => marker.time)]
    project.tracks.forEach(track => track.items.forEach(item => {
      if (item.id !== itemId) points.push(item.start, item.start + item.duration)
    }))
    const candidates = mode === 'move'
      ? points.flatMap(point => [point, point - span])
      : points
    const nearest = candidates.reduce<{ value: number; distance: number } | null>((best, point) => {
      const distance = Math.abs(point - value)
      return !best || distance < best.distance ? { value: point, distance } : best
    }, null)
    if (nearest && nearest.distance <= 9 / pixelsPerSecond) value = Math.max(0, nearest.value)
    return value
  }, [pixelsPerSecond, playhead, project, snapping])

  useEffect(() => {
    if (!drag) return
    const move = (event: PointerEvent) => {
      const delta = (event.clientX - drag.pointerStart) / pixelsPerSecond
      const span = drag.originalEnd - drag.originalStart
      if (drag.mode === 'move') {
        const snappedStart = snapTime(drag.originalStart + delta, drag.itemId, drag.mode, span)
        const groupMove = selectedItemIds.includes(drag.itemId) && selectedItemIds.length > 1
        const hoveredRow = document.elementFromPoint(event.clientX, event.clientY)
          ?.closest<HTMLElement>('[data-editor-track-id]')
        const hoveredTrack = project?.tracks.find(track => track.id === hoveredRow?.dataset.editorTrackId)
        const sourceTrack = project?.tracks.find(track => track.id === drag.sourceTrackId)
        const previewTrack = groupMove
          ? sourceTrack
          : hoveredTrack && sourceTrack
          && hoveredTrack.type === sourceTrack.type
          && !hoveredTrack.locked
            ? hoveredTrack
            : sourceTrack
        const previewTrackId = previewTrack?.id || drag.sourceTrackId
        const previewStart = previewTrack
          ? closestAvailableEditorStart(previewTrack, snappedStart, span, drag.itemId)
          : snappedStart
        setDrag(current => current ? {
          ...current,
          previewStart,
          previewEnd: previewStart + span,
          previewTrackId,
        } : null)
      } else if (drag.mode === 'fade-in') {
        const previewFadeIn = Math.max(0, Math.min(
          drag.originalEnd - drag.originalStart,
          drag.originalFadeIn + delta,
        ))
        setDrag(current => current ? { ...current, previewFadeIn } : null)
      } else if (drag.mode === 'fade-out') {
        const previewFadeOut = Math.max(0, Math.min(
          drag.originalEnd - drag.originalStart,
          drag.originalFadeOut - delta,
        ))
        setDrag(current => current ? { ...current, previewFadeOut } : null)
      } else if (drag.mode === 'trim-start') {
        const sourceTrack = project?.tracks.find(track => track.id === drag.sourceTrackId)
        const previousEnd = sourceTrack
          ? previousEditorItemEnd(sourceTrack, drag.itemId, drag.originalStart)
          : 0
        const previewStart = Math.min(
          drag.originalEnd - 1 / Math.max(1, project?.canvas.fps || 30),
          Math.max(
            previousEnd,
            snapTime(drag.originalStart + delta, drag.itemId, drag.mode, span),
          ),
        )
        setDrag(current => current ? { ...current, previewStart, previewEnd: drag.originalEnd } : null)
      } else {
        const sourceTrack = project?.tracks.find(track => track.id === drag.sourceTrackId)
        const nextStart = sourceTrack
          ? nextEditorItemStart(sourceTrack, drag.itemId, drag.originalEnd)
          : null
        const requestedEnd = Math.max(
          drag.originalStart + 1 / Math.max(1, project?.canvas.fps || 30),
          snapTime(drag.originalEnd + delta, drag.itemId, drag.mode, span),
        )
        const previewEnd = nextStart === null ? requestedEnd : Math.min(requestedEnd, nextStart)
        setDrag(current => current ? { ...current, previewStart: drag.originalStart, previewEnd } : null)
      }
    }
    const up = () => {
      if (drag.mode === 'move') moveItem(drag.itemId, drag.previewStart, drag.previewTrackId)
      else if (drag.mode === 'fade-in') updateItem(drag.itemId, {
        fade_in: drag.previewFadeIn,
        transition_in: drag.previewFadeIn > 0
          ? project?.tracks.flatMap(track => track.items).find(item => item.id === drag.itemId)?.transition_in === 'fade_black'
            ? 'fade_black'
            : 'dissolve'
          : 'none',
      })
      else if (drag.mode === 'fade-out') updateItem(drag.itemId, {
        fade_out: drag.previewFadeOut,
        transition_out: drag.previewFadeOut > 0
          ? project?.tracks.flatMap(track => track.items).find(item => item.id === drag.itemId)?.transition_out === 'fade_black'
            ? 'fade_black'
            : 'dissolve'
          : 'none',
      })
      else trimItem(drag.itemId, drag.mode === 'trim-start' ? 'start' : 'end', drag.mode === 'trim-start' ? drag.previewStart : drag.previewEnd)
      setDrag(null)
    }
    window.addEventListener('pointermove', move)
    window.addEventListener('pointerup', up, { once: true })
    return () => {
      window.removeEventListener('pointermove', move)
      window.removeEventListener('pointerup', up)
    }
  }, [drag, moveItem, pixelsPerSecond, project?.canvas.fps, project?.tracks, selectedItemIds, snapTime, trimItem, updateItem])

  useEffect(() => {
    if (playheadPointerId === null) return
    const move = (event: PointerEvent) => {
      if (event.pointerId !== playheadPointerId) return
      event.preventDefault()
      scrubPlayhead(event.clientX)
    }
    const finish = (event: PointerEvent) => {
      if (event.pointerId !== playheadPointerId) return
      scrubPlayhead(event.clientX)
      setPlayheadPointerId(null)
    }
    window.addEventListener('pointermove', move)
    window.addEventListener('pointerup', finish)
    window.addEventListener('pointercancel', finish)
    return () => {
      window.removeEventListener('pointermove', move)
      window.removeEventListener('pointerup', finish)
      window.removeEventListener('pointercancel', finish)
    }
  }, [playheadPointerId, scrubPlayhead])

  useEffect(() => {
    if (!marquee) return
    const move = (event: PointerEvent) => {
      const bounds = timelineContentRef.current?.getBoundingClientRect()
      if (!bounds) return
      setMarquee(current => current ? {
        ...current,
        currentX: event.clientX - bounds.left,
        currentY: event.clientY - bounds.top,
      } : null)
    }
    const finish = (event: PointerEvent) => {
      const bounds = timelineContentRef.current?.getBoundingClientRect()
      if (!bounds) {
        setMarquee(null)
        return
      }
      const currentX = event.clientX - bounds.left
      const currentY = event.clientY - bounds.top
      const left = Math.min(marquee.startX, currentX)
      const right = Math.max(marquee.startX, currentX)
      const top = Math.min(marquee.startY, currentY)
      const bottom = Math.max(marquee.startY, currentY)
      const moved = Math.hypot(currentX - marquee.startX, currentY - marquee.startY) >= 4
      if (!moved) {
        selectItem(null, marquee.trackId)
        setPlayhead(Math.max(0, (event.clientX - marquee.laneLeft) / pixelsPerSecond))
      } else {
        const hits = Array.from(
          timelineContentRef.current?.querySelectorAll<HTMLElement>('[data-editor-item-id]') || [],
        ).flatMap(element => {
          const rect = element.getBoundingClientRect()
          const elementLeft = rect.left - bounds.left
          const elementRight = rect.right - bounds.left
          const elementTop = rect.top - bounds.top
          const elementBottom = rect.bottom - bounds.top
          return elementLeft < right && elementRight > left && elementTop < bottom && elementBottom > top
            ? [element.dataset.editorItemId || '']
            : []
        }).filter(Boolean)
        const selected = Array.from(new Set([...marquee.baselineItemIds, ...hits]))
        selectItems(selected, hits.at(-1) || marquee.baselineItemIds.at(-1) || null, marquee.trackId)
      }
      setMarquee(null)
    }
    window.addEventListener('pointermove', move)
    window.addEventListener('pointerup', finish, { once: true })
    window.addEventListener('pointercancel', finish, { once: true })
    return () => {
      window.removeEventListener('pointermove', move)
      window.removeEventListener('pointerup', finish)
      window.removeEventListener('pointercancel', finish)
    }
  }, [marquee, pixelsPerSecond, selectItem, selectItems, setPlayhead])

  if (!project) return null

  const selectedItems = project.tracks.flatMap(track => track.items)
    .filter(item => selectedItemIds.includes(item.id))
  const selectionHasLink = selectedItems.some(item => Boolean(item.link_group_id))

  const trackMenuPortal = trackMenuOpen && typeof document !== 'undefined'
    ? createPortal(
        <div
          data-editor-track-menu
          className="fixed z-[240] w-28 rounded-lg border border-border bg-bg-secondary p-1 shadow-2xl"
          style={trackMenuPosition}
        >
          {(['video', 'audio', 'text'] as const).map(type => (
            <button key={type} type="button" onClick={() => { addTrack(type); setTrackMenuOpen(false) }} className="flex w-full items-center gap-1.5 rounded px-2 py-1.5 text-[9px] capitalize text-text-secondary hover:bg-bg-hover">
              <TrackIcon type={type} /> {type}
            </button>
          ))}
        </div>,
        document.body,
      )
    : null

  const beginDrag = (event: React.PointerEvent, item: EditorTimelineItem, mode: DragMode) => {
    event.preventDefault()
    event.stopPropagation()
    const track = project.tracks.find(candidate => candidate.items.some(entry => entry.id === item.id))
    if (!track || track.locked) return
    selectItem(item.id, track.id, mode === 'move' && (event.metaKey || event.ctrlKey || event.shiftKey))
    setDrag({
      itemId: item.id,
      mode,
      sourceTrackId: track.id,
      previewTrackId: track.id,
      pointerStart: event.clientX,
      originalStart: item.start,
      originalEnd: item.start + item.duration,
      previewStart: item.start,
      previewEnd: item.start + item.duration,
      originalFadeIn: item.fade_in || 0,
      originalFadeOut: item.fade_out || 0,
      previewFadeIn: item.fade_in || 0,
      previewFadeOut: item.fade_out || 0,
    })
  }

  const timeFromPointer = (event: React.PointerEvent<HTMLElement>): number => {
    const rect = event.currentTarget.getBoundingClientRect()
    return Math.max(0, (event.clientX - rect.left) / pixelsPerSecond)
  }

  const dropAsset = (event: React.DragEvent<HTMLElement>, trackId: string) => {
    event.preventDefault()
    try {
      let payload = event.dataTransfer.getData('application/x-maestro-editor-asset')
      if (!payload) {
        const plainText = event.dataTransfer.getData('text/plain')
        if (plainText.startsWith('maestro-editor-asset:')) {
          payload = plainText.slice('maestro-editor-asset:'.length)
        }
      }
      if (!payload) return
      const asset = JSON.parse(payload) as EditorAsset
      if (!asset || typeof asset.name !== 'string' || typeof asset.type !== 'string') return
      const rect = event.currentTarget.getBoundingClientRect()
      const at = Math.max(0, (event.clientX - rect.left) / pixelsPerSecond)
      void addMedia(asset, at, trackId)
    } catch { /* Ignore unrelated drags. */ }
  }

  const confirmTrackRemoval = (track: EditorTrack) => {
    if (track.items.length > 0 && !window.confirm(
      `Remove “${track.name}” and its ${track.items.length} clip${track.items.length === 1 ? '' : 's'}? This can be undone with Ctrl+Z.`,
    )) return
    removeTrack(track.id)
  }

  const beginTimelineTouch = (event: React.PointerEvent<HTMLDivElement>) => {
    if (event.pointerType !== 'touch') return
    touchPointersRef.current.set(event.pointerId, { x: event.clientX, y: event.clientY })
    if (touchPointersRef.current.size !== 2 || !scrollRef.current) return
    const points = Array.from(touchPointersRef.current.values())
    const centerX = (points[0].x + points[1].x) / 2
    const distance = Math.max(1, Math.hypot(points[0].x - points[1].x, points[0].y - points[1].y))
    const rect = scrollRef.current.getBoundingClientRect()
    const contentX = scrollRef.current.scrollLeft + centerX - rect.left - TRACK_HEADER_WIDTH
    pinchRef.current = {
      startDistance: distance,
      startZoom: pixelsPerSecond,
      anchorTime: Math.max(0, contentX / pixelsPerSecond),
      anchorViewportX: centerX - rect.left,
    }
    setDrag(null)
    setMarquee(null)
    setPlayheadPointerId(null)
    event.preventDefault()
    event.stopPropagation()
    touchPointersRef.current.forEach((_point, pointerId) => {
      try { scrollRef.current?.setPointerCapture(pointerId) } catch { /* Browser owns this pointer. */ }
    })
  }

  const updateTimelineTouch = (event: React.PointerEvent<HTMLDivElement>) => {
    if (event.pointerType !== 'touch' || !touchPointersRef.current.has(event.pointerId)) return
    touchPointersRef.current.set(event.pointerId, { x: event.clientX, y: event.clientY })
    const pinch = pinchRef.current
    if (!pinch || touchPointersRef.current.size < 2 || !scrollRef.current) return
    const points = Array.from(touchPointersRef.current.values())
    const distance = Math.max(1, Math.hypot(points[0].x - points[1].x, points[0].y - points[1].y))
    const nextZoom = Math.max(20, Math.min(180, pinch.startZoom * distance / pinch.startDistance))
    event.preventDefault()
    event.stopPropagation()
    setPixelsPerSecond(nextZoom)
    const scroll = scrollRef.current
    requestAnimationFrame(() => {
      scroll.scrollLeft = Math.max(
        0,
        TRACK_HEADER_WIDTH + pinch.anchorTime * nextZoom - pinch.anchorViewportX,
      )
    })
  }

  const finishTimelineTouch = (event: React.PointerEvent<HTMLDivElement>) => {
    if (event.pointerType !== 'touch') return
    touchPointersRef.current.delete(event.pointerId)
    if (touchPointersRef.current.size < 2) pinchRef.current = null
  }

  return (
    <section className={`flex min-h-0 flex-col border-t border-border bg-bg-secondary ${compact ? 'h-full' : 'h-[310px] shrink-0'}`}>
      <div className="flex h-10 shrink-0 items-center gap-1.5 overflow-x-auto border-b border-border px-2.5 [scrollbar-width:none] [&::-webkit-scrollbar]:hidden">
        <button type="button" onClick={splitSelected} disabled={!selectedItemId} className="flex items-center gap-1 rounded-md px-2 py-1 text-[9px] text-text-secondary hover:bg-bg-hover hover:text-text-primary disabled:opacity-25" title="Split at playhead (S)">
          <Scissors size={11} /> <span className="hidden sm:inline">Split</span>
        </button>
        <button type="button" onClick={duplicateSelected} disabled={!selectedItemId} className="rounded-md p-1.5 text-text-muted hover:bg-bg-hover hover:text-text-primary disabled:opacity-25" title="Duplicate clip">
          <Copy size={11} />
        </button>
        <button type="button" onClick={copySelected} disabled={!selectedItemId} className="hidden rounded-md p-1.5 text-text-muted hover:bg-bg-hover hover:text-text-primary disabled:opacity-25 sm:block" title="Copy clip (Ctrl+C)">
          <ClipboardCopy size={11} />
        </button>
        <button type="button" onClick={pasteClipboard} disabled={!clipboard} className="hidden rounded-md p-1.5 text-text-muted hover:bg-bg-hover hover:text-text-primary disabled:opacity-25 sm:block" title="Paste at playhead (Ctrl+V)">
          <ClipboardPaste size={11} />
        </button>
        <button type="button" onClick={deleteSelected} disabled={!selectedItemId} className="rounded-md p-1.5 text-text-muted hover:bg-red-500/10 hover:text-red-400 disabled:opacity-25" title="Delete selected">
          <Trash2 size={11} />
        </button>
        <div className="mx-1 h-4 w-px bg-border" />
        <button type="button" onClick={() => jumpToEdit(-1)} className="rounded-md p-1.5 text-text-muted hover:bg-bg-hover hover:text-text-primary" title="Previous edit (Up arrow)">
          <StepBack size={11} />
        </button>
        <button type="button" onClick={() => jumpToEdit(1)} className="rounded-md p-1.5 text-text-muted hover:bg-bg-hover hover:text-text-primary" title="Next edit (Down arrow)">
          <StepForward size={11} />
        </button>
        <button type="button" onClick={() => setSnapping(!snapping)} className={`rounded-md p-1.5 ${snapping ? 'bg-accent-blue/15 text-accent-blue' : 'text-text-muted hover:bg-bg-hover'}`} title="Snapping">
          <Magnet size={11} />
        </button>
        <button type="button" onClick={() => setRipple(!ripple)} className={`flex items-center gap-1 rounded-md px-2 py-1 text-[9px] ${ripple ? 'bg-accent-blue/15 text-accent-blue' : 'text-text-muted hover:bg-bg-hover'}`} title="Close gaps when deleting clips">
          <Waves size={11} /> <span className="hidden sm:inline">Ripple</span>
        </button>
        <button type="button" onClick={() => addMarker(playhead)} className="rounded-md p-1.5 text-text-muted hover:bg-bg-hover hover:text-accent-warm" title="Add marker at playhead (M)">
          <MapPin size={11} />
        </button>
        {selectedItemIds.length >= 2 && !selectionHasLink && (
          <button type="button" onClick={linkSelected} className="rounded-md p-1.5 text-text-muted hover:bg-bg-hover hover:text-accent-blue" title="Link selected clips">
            <Link2 size={11} />
          </button>
        )}
        {selectionHasLink && (
          <button type="button" onClick={unlinkSelected} className="rounded-md bg-accent-blue/10 p-1.5 text-accent-blue hover:bg-accent-blue/20" title="Unlink selected clips">
            <Unlink2 size={11} />
          </button>
        )}
        <div>
          <button
            ref={trackMenuButtonRef}
            type="button"
            onClick={event => {
              if (trackMenuOpen) {
                setTrackMenuOpen(false)
                return
              }
              const rect = event.currentTarget.getBoundingClientRect()
              setTrackMenuPosition({
                left: Math.max(8, Math.min(window.innerWidth - 120, rect.left)),
                top: rect.bottom + 6 + 96 <= window.innerHeight
                  ? rect.bottom + 6
                  : Math.max(8, rect.top - 102),
              })
              setTrackMenuOpen(true)
            }}
            className="flex items-center gap-1 rounded-md px-2 py-1 text-[9px] text-text-muted hover:bg-bg-hover hover:text-text-primary"
          >
            <Plus size={11} /> Track
          </button>
        </div>
        <div className="ml-auto flex items-center gap-1">
          <span className="hidden font-mono text-[9px] text-text-muted sm:block">{formatEditorTime(playhead, true, project.canvas.fps)}</span>
          <button type="button" onClick={zoomToFit} className="rounded p-1 text-text-muted hover:bg-bg-hover hover:text-text-primary" title="Fit timeline">
            <ScanLine size={11} />
          </button>
          <button type="button" onClick={() => setPixelsPerSecond(pixelsPerSecond - 10)} className="rounded p-1 text-text-muted hover:bg-bg-hover"><ZoomOut size={11} /></button>
          <input type="range" min={20} max={180} value={pixelsPerSecond} onChange={event => setPixelsPerSecond(Number(event.target.value))} className="hidden w-16 accent-[var(--color-accent-blue)] sm:block md:w-24" aria-label="Timeline zoom" />
          <button type="button" onClick={() => setPixelsPerSecond(pixelsPerSecond + 10)} className="rounded p-1 text-text-muted hover:bg-bg-hover"><ZoomIn size={11} /></button>
        </div>
      </div>

      <div
        ref={scrollRef}
        className="min-h-0 flex-1 overflow-auto overscroll-contain"
        style={{ touchAction: 'pan-x pan-y' }}
        onPointerDownCapture={beginTimelineTouch}
        onPointerMoveCapture={updateTimelineTouch}
        onPointerUpCapture={finishTimelineTouch}
        onPointerCancelCapture={finishTimelineTouch}
      >
        <div ref={timelineContentRef} className="relative" style={{ width: TRACK_HEADER_WIDTH + contentWidth, minHeight: 28 + project.tracks.length * trackHeight }}>
          <div className="sticky top-0 z-30 flex h-7 border-b border-border bg-bg-secondary/95 backdrop-blur">
            <div className="sticky left-0 z-40 w-[132px] shrink-0 border-r border-border bg-bg-secondary px-2 py-1 text-[8px] uppercase tracking-wider text-text-muted">Timeline</div>
            <div
              className="relative h-full"
              style={{ width: contentWidth }}
              onPointerDown={event => setPlayhead(timeFromPointer(event))}
            >
              {ticks.map(tick => (
                <div key={tick} className="absolute bottom-0 top-0 border-l border-border/80" style={{ left: tick * pixelsPerSecond }}>
                  <span className="absolute left-1 top-0.5 font-mono text-[7px] text-text-muted">{formatEditorTime(tick)}</span>
                </div>
              ))}
              {(project.markers || []).map(marker => (
                <button
                  key={marker.id}
                  type="button"
                  className="absolute top-0 z-20 h-4 w-4 -translate-x-1/2"
                  style={{ left: marker.time * pixelsPerSecond, color: marker.color }}
                  title={`${marker.label} · ${formatEditorTime(marker.time, true, project.canvas.fps)} · double-click to rename · right-click to remove`}
                  onPointerDown={event => { event.stopPropagation(); setPlayhead(marker.time) }}
                  onDoubleClick={event => {
                    event.stopPropagation()
                    const label = window.prompt('Marker name', marker.label)
                    if (label !== null) updateMarker(marker.id, { label })
                  }}
                  onContextMenu={event => { event.preventDefault(); event.stopPropagation(); removeMarker(marker.id) }}
                >
                  <span className="absolute left-1/2 top-0 h-2.5 w-2.5 -translate-x-1/2 rotate-45 rounded-[2px] bg-current shadow" />
                </button>
              ))}
            </div>
          </div>

          {displayTracks.map(track => {
            const draggedItem = drag?.mode === 'move'
              ? project.tracks.flatMap(candidate => candidate.items).find(item => item.id === drag.itemId)
              : undefined
            const items = drag?.mode === 'move' && drag.previewTrackId !== drag.sourceTrackId
              ? [
                  ...track.items.filter(item => item.id !== drag.itemId),
                  ...(track.id === drag.previewTrackId && draggedItem ? [draggedItem] : []),
                ]
              : track.items
            const isMoveTarget = drag?.mode === 'move'
              && drag.previewTrackId === track.id
              && drag.previewTrackId !== drag.sourceTrackId
            return (
            <div
              key={track.id}
              className={`group/track flex border-b border-border/70 ${isMoveTarget ? 'bg-accent-blue/10' : selectedTrackId === track.id ? 'bg-white/[0.025]' : ''}`}
              style={{ height: trackHeight }}
              data-editor-track-id={track.id}
              data-editor-track-type={track.type}
              data-editor-track-locked={track.locked ? 'true' : 'false'}
            >
              <div
                className="sticky left-0 z-20 flex w-[132px] shrink-0 items-center gap-0.5 border-r border-border bg-bg-secondary px-2"
                onPointerDown={() => selectItem(null, track.id)}
              >
                <TrackIcon type={track.type} />
                <span className="min-w-0 flex-1 truncate text-[8px] text-text-secondary" title={track.name}>{track.name}</span>
                <button type="button" onPointerDown={event => event.stopPropagation()} onClick={() => toggleTrackMute(track.id)} className={`rounded p-1 ${track.muted ? 'text-red-400' : 'text-text-muted hover:text-text-primary'}`} title={track.muted ? 'Unmute track' : 'Mute track'}>
                  {track.muted ? <VolumeX size={9} /> : <Volume2 size={9} />}
                </button>
                <button type="button" onPointerDown={event => event.stopPropagation()} onClick={() => toggleTrackLock(track.id)} className={`rounded p-1 ${track.locked ? 'text-accent-blue' : 'text-text-muted hover:text-text-primary'}`} title={track.locked ? 'Unlock track' : 'Lock track'}>
                  {track.locked ? <Lock size={9} /> : <Unlock size={9} />}
                </button>
                <button
                  type="button"
                  onPointerDown={event => event.stopPropagation()}
                  onClick={() => confirmTrackRemoval(track)}
                  className="rounded p-1 text-text-muted opacity-45 hover:bg-red-500/10 hover:text-red-400 group-hover/track:opacity-100"
                  title={track.items.length ? `Remove track and ${track.items.length} clip${track.items.length === 1 ? '' : 's'}` : 'Remove track'}
                >
                  <Trash2 size={9} />
                </button>
              </div>
              <div
                className="relative h-full bg-bg-primary/30"
                style={{ width: contentWidth }}
                onPointerDown={event => {
                  if (event.button !== 0) return
                  event.preventDefault()
                  event.stopPropagation()
                  setPlaying(false)
                  const bounds = timelineContentRef.current?.getBoundingClientRect()
                  const lane = event.currentTarget.getBoundingClientRect()
                  if (!bounds) return
                  setMarquee({
                    startX: event.clientX - bounds.left,
                    startY: event.clientY - bounds.top,
                    currentX: event.clientX - bounds.left,
                    currentY: event.clientY - bounds.top,
                    baselineItemIds: event.ctrlKey || event.metaKey || event.shiftKey ? selectedItemIds : [],
                    trackId: track.id,
                    laneLeft: lane.left,
                  })
                }}
                onDragOver={event => { event.preventDefault(); event.dataTransfer.dropEffect = 'copy' }}
                onDrop={event => dropAsset(event, track.id)}
              >
                {ticks.map(tick => <div key={tick} className="pointer-events-none absolute inset-y-0 border-l border-border/25" style={{ left: tick * pixelsPerSecond }} />)}
                {items.map(item => {
                  const preview = drag?.itemId === item.id ? drag : null
                  const participatesInLinkedPreview = Boolean(
                    drag
                    && selectedItemIds.includes(drag.itemId)
                    && selectedItemIds.includes(item.id),
                  )
                  const dragStartDelta = drag ? drag.previewStart - drag.originalStart : 0
                  const dragEndDelta = drag ? drag.previewEnd - drag.originalEnd : 0
                  const start = preview?.previewStart ?? (
                    participatesInLinkedPreview && (drag?.mode === 'move' || drag?.mode === 'trim-start')
                      ? item.start + dragStartDelta
                      : item.start
                  )
                  const end = preview?.previewEnd ?? (
                    participatesInLinkedPreview && drag?.mode === 'move'
                      ? item.start + item.duration + dragStartDelta
                      : participatesInLinkedPreview && drag?.mode === 'trim-end'
                        ? item.start + item.duration + dragEndDelta
                        : item.start + item.duration
                  )
                  const width = Math.max(8, (end - start) * pixelsPerSecond)
                  const fadeIn = preview?.previewFadeIn ?? item.fade_in ?? 0
                  const fadeOut = preview?.previewFadeOut ?? item.fade_out ?? 0
                  const asset = item.asset_id ? project.assets[item.asset_id] : undefined
                  const isSelected = selectedItemIds.includes(item.id)
                  return (
                    <div
                      key={item.id}
                      data-editor-item-id={item.id}
                      className={`group absolute touch-none cursor-grab overflow-hidden rounded-md border shadow-sm active:cursor-grabbing ${clipClasses(track.type)} ${isSelected ? 'ring-2 ring-accent-warm ring-offset-1 ring-offset-bg-primary' : ''} ${track.locked ? 'cursor-not-allowed opacity-60' : ''} ${item.disabled ? 'opacity-35 saturate-0' : ''}`}
                      style={{ left: start * pixelsPerSecond, width, top: compact ? 8 : 6, height: compact ? 42 : 34 }}
                      onPointerDown={event => beginDrag(event, item, 'move')}
                      title={`${item.name} · ${item.duration.toFixed(2)}s`}
                    >
                      <EditorClipMedia asset={asset} item={item} trackType={track.type} width={width} workspace={project.workspace} />
                      <div className="relative z-10 flex h-full items-center gap-1 overflow-hidden bg-gradient-to-r from-black/35 via-transparent to-black/20 px-2">
                        <TrackIcon type={track.type} />
                        <span className="truncate text-[8px] font-medium">{item.text || item.name}</span>
                        {item.link_group_id && <Link2 size={9} className="ml-auto shrink-0 opacity-75" />}
                      </div>
                      {fadeIn > 0 && <span className="pointer-events-none absolute inset-y-0 left-0 z-10 bg-gradient-to-r from-black/70 to-transparent" style={{ width: Math.min(width, fadeIn * pixelsPerSecond) }} />}
                      {fadeOut > 0 && <span className="pointer-events-none absolute inset-y-0 right-0 z-10 bg-gradient-to-l from-black/70 to-transparent" style={{ width: Math.min(width, fadeOut * pixelsPerSecond) }} />}
                      {!track.locked && (
                        <>
                          <button type="button" onPointerDown={event => beginDrag(event, item, 'trim-start')} className={`absolute inset-y-0 left-0 cursor-ew-resize bg-current/10 hover:bg-current/25 group-hover:opacity-100 ${compact ? 'w-4' : 'w-2'} ${compact && isSelected ? 'opacity-100' : 'opacity-0'}`} aria-label="Trim clip start" />
                          <button type="button" onPointerDown={event => beginDrag(event, item, 'trim-end')} className={`absolute inset-y-0 right-0 cursor-ew-resize bg-current/10 hover:bg-current/25 group-hover:opacity-100 ${compact ? 'w-4' : 'w-2'} ${compact && isSelected ? 'opacity-100' : 'opacity-0'}`} aria-label="Trim clip end" />
                          <button
                            type="button"
                            onPointerDown={event => beginDrag(event, item, 'fade-in')}
                            className={`absolute top-1 z-20 -translate-x-1/2 rotate-45 cursor-ew-resize border border-current bg-bg-primary shadow group-hover:opacity-100 ${compact ? 'h-4 w-4' : 'h-2.5 w-2.5'} ${compact && isSelected ? 'opacity-100' : 'opacity-0'}`}
                            style={{ left: Math.max(6, Math.min(width - 6, fadeIn * pixelsPerSecond)) }}
                            aria-label="Adjust clip fade in"
                            title={`Fade in ${fadeIn.toFixed(2)}s`}
                          />
                          <button
                            type="button"
                            onPointerDown={event => beginDrag(event, item, 'fade-out')}
                            className={`absolute top-1 z-20 -translate-x-1/2 rotate-45 cursor-ew-resize border border-current bg-bg-primary shadow group-hover:opacity-100 ${compact ? 'h-4 w-4' : 'h-2.5 w-2.5'} ${compact && isSelected ? 'opacity-100' : 'opacity-0'}`}
                            style={{ left: Math.max(6, Math.min(width - 6, width - fadeOut * pixelsPerSecond)) }}
                            aria-label="Adjust clip fade out"
                            title={`Fade out ${fadeOut.toFixed(2)}s`}
                          />
                        </>
                      )}
                    </div>
                  )
                })}
              </div>
            </div>
            )
          })}

          {(project.markers || []).map(marker => (
            <button
              key={`line-${marker.id}`}
              type="button"
              aria-label={`Go to ${marker.label}`}
              className="absolute bottom-0 top-7 z-10 w-px opacity-45 hover:w-0.5 hover:opacity-100"
              style={{ left: TRACK_HEADER_WIDTH + marker.time * pixelsPerSecond, backgroundColor: marker.color }}
              onClick={() => setPlayhead(marker.time)}
            />
          ))}

          {marquee && (
            <div
              className="pointer-events-none absolute z-40 border border-accent-blue bg-accent-blue/15 shadow-[0_0_0_1px_rgba(0,0,0,0.25)]"
              style={{
                left: Math.min(marquee.startX, marquee.currentX),
                top: Math.min(marquee.startY, marquee.currentY),
                width: Math.abs(marquee.currentX - marquee.startX),
                height: Math.abs(marquee.currentY - marquee.startY),
              }}
            />
          )}

          <div
            role="slider"
            aria-label="Timeline playhead"
            aria-valuemin={0}
            aria-valuemax={contentDuration}
            aria-valuenow={playhead}
            tabIndex={0}
            className={`group absolute bottom-0 top-0 z-50 touch-none cursor-ew-resize outline-none ${compact ? 'w-7' : 'w-3'} ${playheadPointerId !== null ? 'cursor-grabbing' : ''}`}
            style={{ left: TRACK_HEADER_WIDTH + playhead * pixelsPerSecond - (compact ? 14 : 6) }}
            onPointerDown={event => {
              event.preventDefault()
              event.stopPropagation()
              setPlaying(false)
              setPlayheadPointerId(event.pointerId)
              scrubPlayhead(event.clientX)
            }}
            onKeyDown={event => {
              const frame = 1 / Math.max(1, project.canvas.fps)
              if (event.key === 'ArrowLeft') {
                event.preventDefault()
                setPlayhead(Math.max(0, playhead - frame))
              } else if (event.key === 'ArrowRight') {
                event.preventDefault()
                setPlayhead(Math.min(contentDuration, playhead + frame))
              }
            }}
          >
            <div className="pointer-events-none absolute bottom-0 left-1/2 top-7 w-px -translate-x-1/2 bg-accent-warm shadow-[0_0_5px_rgba(245,158,11,0.55)] group-hover:w-0.5" />
            <div className={`pointer-events-none absolute left-1/2 -translate-x-1/2 rotate-45 bg-accent-warm ${compact ? 'top-[20px] h-3 w-3' : 'top-[23px] h-2 w-2'}`} />
          </div>
        </div>
      </div>
      {trackMenuPortal}
    </section>
  )
}
