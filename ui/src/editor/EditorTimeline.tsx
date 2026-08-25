import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  AudioLines,
  Copy,
  Film,
  Lock,
  Magnet,
  Plus,
  Scissors,
  TextCursorInput,
  Trash2,
  Unlock,
  Volume2,
  VolumeX,
  Waves,
  ZoomIn,
  ZoomOut,
} from 'lucide-react'
import type { EditorAsset, EditorTimelineItem, EditorTrack } from '../types'
import { editorProjectDuration, formatEditorTime } from './editorUtils'
import { useEditorStore } from './useEditorStore'

const TRACK_HEADER_WIDTH = 116
const TRACK_HEIGHT = 46

type DragMode = 'move' | 'trim-start' | 'trim-end'
interface DragState {
  itemId: string
  mode: DragMode
  pointerStart: number
  originalStart: number
  originalEnd: number
  previewStart: number
  previewEnd: number
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
  const [drag, setDrag] = useState<DragState | null>(null)
  const [trackMenuOpen, setTrackMenuOpen] = useState(false)
  const project = useEditorStore(state => state.project)
  const selectedItemId = useEditorStore(state => state.selectedItemId)
  const playhead = useEditorStore(state => state.playhead)
  const pixelsPerSecond = useEditorStore(state => state.pixelsPerSecond)
  const snapping = useEditorStore(state => state.snapping)
  const ripple = useEditorStore(state => state.ripple)
  const setPlayhead = useEditorStore(state => state.setPlayhead)
  const selectItem = useEditorStore(state => state.selectItem)
  const setPixelsPerSecond = useEditorStore(state => state.setPixelsPerSecond)
  const setSnapping = useEditorStore(state => state.setSnapping)
  const setRipple = useEditorStore(state => state.setRipple)
  const splitSelected = useEditorStore(state => state.splitSelected)
  const duplicateSelected = useEditorStore(state => state.duplicateSelected)
  const deleteSelected = useEditorStore(state => state.deleteSelected)
  const moveItem = useEditorStore(state => state.moveItem)
  const trimItem = useEditorStore(state => state.trimItem)
  const toggleTrackMute = useEditorStore(state => state.toggleTrackMute)
  const toggleTrackLock = useEditorStore(state => state.toggleTrackLock)
  const addTrack = useEditorStore(state => state.addTrack)
  const addMedia = useEditorStore(state => state.addMedia)

  const duration = editorProjectDuration(project)
  const contentDuration = Math.max(12, duration + 4)
  const contentWidth = Math.max(compact ? 620 : 900, contentDuration * pixelsPerSecond)
  const tickStep = pixelsPerSecond >= 90 ? 1 : pixelsPerSecond >= 45 ? 2 : 5
  const ticks = useMemo(() => (
    Array.from({ length: Math.min(1200, Math.ceil(contentDuration / tickStep) + 1) }, (_, index) => index * tickStep)
  ), [contentDuration, tickStep])

  const snapTime = useCallback((raw: number, itemId: string, mode: DragMode, span: number): number => {
    if (!project) return Math.max(0, raw)
    const frame = 1 / Math.max(1, project.canvas.fps)
    let value = Math.round(Math.max(0, raw) / frame) * frame
    if (!snapping) return value
    const points = [0, playhead]
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
        const previewStart = snapTime(drag.originalStart + delta, drag.itemId, drag.mode, span)
        setDrag(current => current ? { ...current, previewStart, previewEnd: previewStart + span } : null)
      } else if (drag.mode === 'trim-start') {
        const previewStart = Math.min(
          drag.originalEnd - 1 / Math.max(1, project?.canvas.fps || 30),
          snapTime(drag.originalStart + delta, drag.itemId, drag.mode, span),
        )
        setDrag(current => current ? { ...current, previewStart, previewEnd: drag.originalEnd } : null)
      } else {
        const previewEnd = Math.max(
          drag.originalStart + 1 / Math.max(1, project?.canvas.fps || 30),
          snapTime(drag.originalEnd + delta, drag.itemId, drag.mode, span),
        )
        setDrag(current => current ? { ...current, previewStart: drag.originalStart, previewEnd } : null)
      }
    }
    const up = () => {
      if (drag.mode === 'move') moveItem(drag.itemId, drag.previewStart)
      else trimItem(drag.itemId, drag.mode === 'trim-start' ? 'start' : 'end', drag.mode === 'trim-start' ? drag.previewStart : drag.previewEnd)
      setDrag(null)
    }
    window.addEventListener('pointermove', move)
    window.addEventListener('pointerup', up, { once: true })
    return () => {
      window.removeEventListener('pointermove', move)
      window.removeEventListener('pointerup', up)
    }
  }, [drag, moveItem, pixelsPerSecond, project?.canvas.fps, snapTime, trimItem])

  if (!project) return null

  const beginDrag = (event: React.PointerEvent, item: EditorTimelineItem, mode: DragMode) => {
    event.preventDefault()
    event.stopPropagation()
    const track = project.tracks.find(candidate => candidate.items.some(entry => entry.id === item.id))
    if (track?.locked) return
    selectItem(item.id, track?.id)
    setDrag({
      itemId: item.id,
      mode,
      pointerStart: event.clientX,
      originalStart: item.start,
      originalEnd: item.start + item.duration,
      previewStart: item.start,
      previewEnd: item.start + item.duration,
    })
  }

  const timeFromPointer = (event: React.PointerEvent<HTMLElement>): number => {
    const rect = event.currentTarget.getBoundingClientRect()
    return Math.max(0, (event.clientX - rect.left) / pixelsPerSecond)
  }

  const dropAsset = (event: React.DragEvent<HTMLElement>, trackId: string) => {
    event.preventDefault()
    try {
      const asset = JSON.parse(event.dataTransfer.getData('application/x-maestro-editor-asset')) as EditorAsset
      const rect = event.currentTarget.getBoundingClientRect()
      const at = Math.max(0, (event.clientX - rect.left) / pixelsPerSecond)
      void addMedia(asset, at, trackId)
    } catch { /* Ignore unrelated drags. */ }
  }

  return (
    <section className={`flex min-h-0 flex-col border-t border-border bg-bg-secondary ${compact ? 'h-full' : 'h-[310px] shrink-0'}`}>
      <div className="flex h-10 shrink-0 items-center gap-1.5 border-b border-border px-2.5">
        <button type="button" onClick={splitSelected} disabled={!selectedItemId} className="flex items-center gap-1 rounded-md px-2 py-1 text-[9px] text-text-secondary hover:bg-bg-hover hover:text-text-primary disabled:opacity-25" title="Split at playhead (S)">
          <Scissors size={11} /> <span className="hidden sm:inline">Split</span>
        </button>
        <button type="button" onClick={duplicateSelected} disabled={!selectedItemId} className="rounded-md p-1.5 text-text-muted hover:bg-bg-hover hover:text-text-primary disabled:opacity-25" title="Duplicate clip">
          <Copy size={11} />
        </button>
        <button type="button" onClick={deleteSelected} disabled={!selectedItemId} className="rounded-md p-1.5 text-text-muted hover:bg-red-500/10 hover:text-red-400 disabled:opacity-25" title="Delete selected">
          <Trash2 size={11} />
        </button>
        <div className="mx-1 h-4 w-px bg-border" />
        <button type="button" onClick={() => setSnapping(!snapping)} className={`rounded-md p-1.5 ${snapping ? 'bg-accent-blue/15 text-accent-blue' : 'text-text-muted hover:bg-bg-hover'}`} title="Snapping">
          <Magnet size={11} />
        </button>
        <button type="button" onClick={() => setRipple(!ripple)} className={`flex items-center gap-1 rounded-md px-2 py-1 text-[9px] ${ripple ? 'bg-accent-blue/15 text-accent-blue' : 'text-text-muted hover:bg-bg-hover'}`} title="Close gaps when deleting clips">
          <Waves size={11} /> <span className="hidden sm:inline">Ripple</span>
        </button>
        <div className="relative">
          <button type="button" onClick={() => setTrackMenuOpen(open => !open)} className="flex items-center gap-1 rounded-md px-2 py-1 text-[9px] text-text-muted hover:bg-bg-hover hover:text-text-primary">
            <Plus size={11} /> Track
          </button>
          {trackMenuOpen && (
            <div className="absolute bottom-full left-0 z-50 mb-1 w-28 rounded-lg border border-border bg-bg-secondary p-1 shadow-xl">
              {(['video', 'audio', 'text'] as const).map(type => (
                <button key={type} type="button" onClick={() => { addTrack(type); setTrackMenuOpen(false) }} className="flex w-full items-center gap-1.5 rounded px-2 py-1.5 text-[9px] capitalize text-text-secondary hover:bg-bg-hover">
                  <TrackIcon type={type} /> {type}
                </button>
              ))}
            </div>
          )}
        </div>
        <div className="ml-auto flex items-center gap-1">
          <span className="hidden font-mono text-[9px] text-text-muted sm:block">{formatEditorTime(playhead, true, project.canvas.fps)}</span>
          <button type="button" onClick={() => setPixelsPerSecond(pixelsPerSecond - 10)} className="rounded p-1 text-text-muted hover:bg-bg-hover"><ZoomOut size={11} /></button>
          <input type="range" min={20} max={180} value={pixelsPerSecond} onChange={event => setPixelsPerSecond(Number(event.target.value))} className="w-16 accent-[var(--color-accent-blue)] md:w-24" aria-label="Timeline zoom" />
          <button type="button" onClick={() => setPixelsPerSecond(pixelsPerSecond + 10)} className="rounded p-1 text-text-muted hover:bg-bg-hover"><ZoomIn size={11} /></button>
        </div>
      </div>

      <div ref={scrollRef} className="min-h-0 flex-1 overflow-auto overscroll-contain">
        <div className="relative" style={{ width: TRACK_HEADER_WIDTH + contentWidth, minHeight: 28 + project.tracks.length * TRACK_HEIGHT }}>
          <div className="sticky top-0 z-30 flex h-7 border-b border-border bg-bg-secondary/95 backdrop-blur">
            <div className="sticky left-0 z-40 w-[116px] shrink-0 border-r border-border bg-bg-secondary px-2 py-1 text-[8px] uppercase tracking-wider text-text-muted">Timeline</div>
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
            </div>
          </div>

          {project.tracks.map(track => (
            <div key={track.id} className="flex h-[46px] border-b border-border/70">
              <div className="sticky left-0 z-20 flex w-[116px] shrink-0 items-center gap-1 border-r border-border bg-bg-secondary px-2">
                <TrackIcon type={track.type} />
                <span className="min-w-0 flex-1 truncate text-[8px] text-text-secondary" title={track.name}>{track.name}</span>
                <button type="button" onClick={() => toggleTrackMute(track.id)} className={`rounded p-1 ${track.muted ? 'text-red-400' : 'text-text-muted hover:text-text-primary'}`} title={track.muted ? 'Unmute track' : 'Mute track'}>
                  {track.muted ? <VolumeX size={9} /> : <Volume2 size={9} />}
                </button>
                <button type="button" onClick={() => toggleTrackLock(track.id)} className={`rounded p-1 ${track.locked ? 'text-accent-blue' : 'text-text-muted hover:text-text-primary'}`} title={track.locked ? 'Unlock track' : 'Lock track'}>
                  {track.locked ? <Lock size={9} /> : <Unlock size={9} />}
                </button>
              </div>
              <div
                className="relative h-full bg-bg-primary/30"
                style={{ width: contentWidth }}
                onPointerDown={event => {
                  selectItem(null, track.id)
                  setPlayhead(timeFromPointer(event))
                }}
                onDragOver={event => { event.preventDefault(); event.dataTransfer.dropEffect = 'copy' }}
                onDrop={event => dropAsset(event, track.id)}
              >
                {ticks.map(tick => <div key={tick} className="pointer-events-none absolute inset-y-0 border-l border-border/25" style={{ left: tick * pixelsPerSecond }} />)}
                {track.items.map(item => {
                  const preview = drag?.itemId === item.id ? drag : null
                  const start = preview?.previewStart ?? item.start
                  const end = preview?.previewEnd ?? item.start + item.duration
                  const width = Math.max(8, (end - start) * pixelsPerSecond)
                  return (
                    <div
                      key={item.id}
                      className={`group absolute top-1.5 h-[34px] cursor-grab overflow-hidden rounded-md border shadow-sm active:cursor-grabbing ${clipClasses(track.type)} ${selectedItemId === item.id ? 'ring-2 ring-accent-warm ring-offset-1 ring-offset-bg-primary' : ''} ${track.locked ? 'cursor-not-allowed opacity-60' : ''}`}
                      style={{ left: start * pixelsPerSecond, width }}
                      onPointerDown={event => beginDrag(event, item, 'move')}
                      title={`${item.name} · ${item.duration.toFixed(2)}s`}
                    >
                      <div className="flex h-full items-center gap-1 overflow-hidden px-2">
                        <TrackIcon type={track.type} />
                        <span className="truncate text-[8px] font-medium">{item.text || item.name}</span>
                        {track.type === 'audio' && <span className="ml-auto flex h-4 items-center gap-px opacity-60">{[4, 8, 5, 10, 6, 9, 3].map((height, index) => <i key={index} className="w-px bg-current" style={{ height }} />)}</span>}
                      </div>
                      {!track.locked && (
                        <>
                          <button type="button" onPointerDown={event => beginDrag(event, item, 'trim-start')} className="absolute inset-y-0 left-0 w-2 cursor-ew-resize bg-current/10 opacity-0 hover:bg-current/25 group-hover:opacity-100" aria-label="Trim clip start" />
                          <button type="button" onPointerDown={event => beginDrag(event, item, 'trim-end')} className="absolute inset-y-0 right-0 w-2 cursor-ew-resize bg-current/10 opacity-0 hover:bg-current/25 group-hover:opacity-100" aria-label="Trim clip end" />
                        </>
                      )}
                    </div>
                  )
                })}
              </div>
            </div>
          ))}

          <div className="pointer-events-none absolute bottom-0 top-7 z-20 w-px bg-accent-warm shadow-[0_0_5px_rgba(245,158,11,0.55)]" style={{ left: TRACK_HEADER_WIDTH + playhead * pixelsPerSecond }}>
            <div className="absolute -left-[4px] -top-1 h-2 w-2 rotate-45 bg-accent-warm" />
          </div>
        </div>
      </div>
    </section>
  )
}
