import { useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import { Maximize2, Pause, Play, RotateCcw, SkipBack, SkipForward } from 'lucide-react'
import type { EditorAsset, EditorTimelineItem, EditorTrack, EditorTransform } from '../types'
import { useIsMobile } from '../lib/useIsMobile'
import { activeEditorItems, editorProjectDuration, fitEditorCanvasToViewport, formatEditorTime } from './editorUtils'
import { DEFAULT_EDITOR_FONT, editorFontStack } from './editorFonts'
import { useEditorMediaPreview } from './editorMediaPreview'
import { useEditorStore } from './useEditorStore'

type CanvasInteractionMode = 'move' | 'resize'

interface CanvasInteraction {
  itemId: string
  pointerId: number
  mode: CanvasInteractionMode
  startClientX: number
  startClientY: number
  centerClientX: number
  centerClientY: number
  startDistance: number
  canvasWidth: number
  canvasHeight: number
  startTransform: EditorTransform
  previewTransform: EditorTransform
}

interface CanvasGuides {
  x: number[]
  y: number[]
}

interface CanvasPinchInteraction {
  itemId: string
  startDistance: number
  startAngle: number
  startCenterX: number
  startCenterY: number
  canvasWidth: number
  canvasHeight: number
  startTransform: EditorTransform
  previewTransform: EditorTransform
}

interface VisualBounds {
  centerX: number
  centerY: number
  halfWidth: number
  halfHeight: number
}

const CANVAS_SNAP_SCREEN_PX = 9

function visualFrameSize(
  asset: EditorAsset,
  item: EditorTimelineItem,
  canvasWidth: number,
  canvasHeight: number,
): { width: number; height: number } {
  const sourceWidth = Math.max(1, asset.width || canvasWidth)
  const sourceHeight = Math.max(1, asset.height || canvasHeight)
  const sourceAspect = sourceWidth / sourceHeight
  const canvasAspect = canvasWidth / Math.max(1, canvasHeight)
  if (item.fit !== 'contain' || !asset.width || !asset.height) {
    return { width: canvasWidth, height: canvasHeight }
  }
  return sourceAspect >= canvasAspect
    ? { width: canvasWidth, height: canvasWidth / sourceAspect }
    : { width: canvasHeight * sourceAspect, height: canvasHeight }
}

function visualBounds(
  asset: EditorAsset,
  item: EditorTimelineItem,
  canvasWidth: number,
  canvasHeight: number,
  transform: EditorTransform = item.transform,
): VisualBounds {
  const frame = visualFrameSize(asset, item, canvasWidth, canvasHeight)
  const scale = Math.max(0.05, transform.scale || 1)
  const radians = (transform.rotation || 0) * Math.PI / 180
  const cosine = Math.abs(Math.cos(radians))
  const sine = Math.abs(Math.sin(radians))
  const scaledWidth = frame.width * scale
  const scaledHeight = frame.height * scale
  return {
    centerX: transform.x || 0,
    centerY: transform.y || 0,
    halfWidth: (scaledWidth * cosine + scaledHeight * sine) / 2,
    halfHeight: (scaledWidth * sine + scaledHeight * cosine) / 2,
  }
}

function uniqueGuideValues(values: number[]): number[] {
  return values.reduce<number[]>((result, value) => {
    if (!result.some(existing => Math.abs(existing - value) < 0.01)) result.push(value)
    return result
  }, [])
}

function nearestGuideOffset(
  movingAnchors: number[],
  targets: number[],
  threshold: number,
): { offset: number; guide: number } | null {
  let best: { offset: number; guide: number; distance: number } | null = null
  for (const anchor of movingAnchors) {
    for (const target of targets) {
      const offset = target - anchor
      const distance = Math.abs(offset)
      if (distance <= threshold && (!best || distance < best.distance)) {
        best = { offset, guide: target, distance }
      }
    }
  }
  return best ? { offset: best.offset, guide: best.guide } : null
}

function clipEnvelope(item: EditorTimelineItem, playhead: number): number {
  const elapsed = Math.max(0, playhead - item.start)
  const remaining = Math.max(0, item.start + item.duration - playhead)
  const fadeIn = Math.max(0, Math.min(item.duration, item.fade_in || 0))
  const fadeOut = Math.max(0, Math.min(item.duration, item.fade_out || 0))
  const fadeInLevel = fadeIn > 0 ? Math.min(1, elapsed / fadeIn) : 1
  const fadeOutLevel = fadeOut > 0 ? Math.min(1, remaining / fadeOut) : 1
  return Math.max(0, Math.min(1, fadeInLevel, fadeOutLevel))
}

function usesBlackTransition(item: EditorTimelineItem, playhead: number): boolean {
  const elapsed = Math.max(0, playhead - item.start)
  const remaining = Math.max(0, item.start + item.duration - playhead)
  return (
    item.transition_in === 'fade_black'
    && (item.fade_in || 0) > 0
    && elapsed < (item.fade_in || 0)
  ) || (
    item.transition_out === 'fade_black'
    && (item.fade_out || 0) > 0
    && remaining < (item.fade_out || 0)
  )
}

function hexToRgba(hex: string, opacity: number): string {
  const value = /^#[0-9a-f]{6}$/i.test(hex) ? hex.slice(1) : '000000'
  const red = Number.parseInt(value.slice(0, 2), 16)
  const green = Number.parseInt(value.slice(2, 4), 16)
  const blue = Number.parseInt(value.slice(4, 6), 16)
  return `rgba(${red}, ${green}, ${blue}, ${Math.max(0, Math.min(1, opacity))})`
}

function syncMediaElement(
  element: HTMLMediaElement,
  item: EditorTimelineItem,
  playhead: number,
  playing: boolean,
  volume: number,
): void {
  const wantedTime = Math.max(0, item.source_in + (playhead - item.start) * item.speed)
  if (Number.isFinite(element.duration)) {
    const bounded = Math.min(Math.max(0, element.duration - 0.03), wantedTime)
    if (Math.abs(element.currentTime - bounded) > (playing ? 0.2 : 0.045)) element.currentTime = bounded
  } else if (Math.abs(element.currentTime - wantedTime) > 0.045) {
    element.currentTime = wantedTime
  }
  element.playbackRate = Math.max(0.1, Math.min(8, item.speed))
  element.volume = Math.max(0, Math.min(1, volume))
  if (playing) {
    if (element.paused) void element.play().catch(() => {})
  } else if (!element.paused) {
    element.pause()
  }
}

function PreviewVisual({
  asset,
  item,
  track,
  playhead,
  playing,
  canvasWidth,
  canvasHeight,
  selected,
  locked,
  workspace,
  mobilePlayback,
  previewTransform,
  onTransformStart,
}: {
  asset: EditorAsset
  item: EditorTimelineItem
  track: EditorTrack
  playhead: number
  playing: boolean
  canvasWidth: number
  canvasHeight: number
  selected: boolean
  locked: boolean
  workspace: string
  mobilePlayback: boolean
  previewTransform?: EditorTransform
  onTransformStart: (
    event: React.PointerEvent<HTMLElement>,
    item: EditorTimelineItem,
    track: EditorTrack,
    mode: CanvasInteractionMode,
  ) => void
}) {
  const videoRef = useRef<HTMLVideoElement>(null)
  const mediaPreview = useEditorMediaPreview(
    asset.type === 'video' ? asset : undefined,
    workspace,
    true,
    mobilePlayback ? 'mobile' : 'auto',
  )
  const sourceUrl = mediaPreview.data?.proxy_url || asset.url
  const transform = previewTransform || item.transform
  const sourceWidth = Math.max(1, asset.width || canvasWidth)
  const sourceHeight = Math.max(1, asset.height || canvasHeight)
  const sourceAspect = sourceWidth / sourceHeight
  const canvasAspect = canvasWidth / Math.max(1, canvasHeight)
  let frameWidth = 100
  let frameHeight = 100
  if (item.fit === 'contain' && asset.width > 0 && asset.height > 0) {
    if (sourceAspect >= canvasAspect) {
      frameHeight = (canvasWidth / sourceAspect) / canvasHeight * 100
    } else {
      frameWidth = (canvasHeight * sourceAspect) / canvasWidth * 100
    }
  }
  const scale = Math.max(0.05, transform?.scale || 1)
  const inverseHandleScale = 1 / scale
  const envelope = clipEnvelope(item, playhead)
  const fadeThroughBlack = usesBlackTransition(item, playhead)

  useEffect(() => {
    const video = videoRef.current
    if (!video || asset.type !== 'video') return
    // Preview the same mix FFmpeg exports: an added music/voice track does not
    // implicitly silence audio embedded in visible video clips.
    video.muted = Boolean(item.muted) || track.muted
    syncMediaElement(
      video,
      item,
      playhead,
      playing,
      item.volume * (track.volume ?? 1) * clipEnvelope(item, playhead),
    )
  }, [asset.type, item, playhead, playing, sourceUrl, track.muted, track.volume])

  return (
    <div
      data-editor-canvas-item-id={item.id}
      className={`group absolute touch-none ${locked ? 'cursor-not-allowed' : 'cursor-move'}`}
      style={{
        left: `calc(50% + ${(transform?.x || 0) / Math.max(1, canvasWidth) * 100}%)`,
        top: `calc(50% + ${(transform?.y || 0) / Math.max(1, canvasHeight) * 100}%)`,
        width: `${frameWidth}%`,
        height: `${frameHeight}%`,
        opacity: item.opacity * (fadeThroughBlack ? 1 : envelope),
        filter: fadeThroughBlack ? `brightness(${envelope})` : undefined,
        transform: `translate(-50%, -50%) scale(${scale}) rotate(${transform?.rotation || 0}deg)`,
        transformOrigin: 'center',
      }}
      onPointerDown={event => onTransformStart(event, item, track, 'move')}
    >
      {asset.type === 'image' ? (
        <img src={asset.url} alt="" draggable={false} className="pointer-events-none h-full w-full select-none" style={{ objectFit: item.fit }} />
      ) : asset.missing ? (
        <div className="grid h-full w-full place-items-center bg-red-950/70 text-[10px] font-semibold uppercase tracking-wider text-red-100">Media offline</div>
      ) : (
        <video
          ref={videoRef}
          src={sourceUrl}
          className="pointer-events-none h-full w-full"
          style={{ objectFit: item.fit }}
          playsInline
          preload="auto"
        />
      )}
      {selected && (
        <div className={`pointer-events-none absolute inset-0 border ${locked ? 'border-dashed border-white/65' : 'border-accent-warm'} shadow-[0_0_0_1px_rgba(0,0,0,0.45)]`}>
          {!locked && [
            '-left-1.5 -top-1.5 cursor-nwse-resize',
            '-right-1.5 -top-1.5 cursor-nesw-resize',
            '-bottom-1.5 -left-1.5 cursor-nesw-resize',
            '-bottom-1.5 -right-1.5 cursor-nwse-resize',
          ].map((position, index) => (
            <button
              key={position}
              type="button"
              aria-label={`Resize visual layer from corner ${index + 1}`}
              className={`pointer-events-auto absolute h-5 w-5 rounded-sm border border-black/50 bg-accent-warm shadow sm:h-3 sm:w-3 ${position}`}
              style={{ transform: `scale(${inverseHandleScale})` }}
              onPointerDown={event => onTransformStart(event, item, track, 'resize')}
            />
          ))}
        </div>
      )}
    </div>
  )
}

function PreviewAudio({
  asset,
  item,
  track,
  playhead,
  playing,
}: {
  asset: EditorAsset
  item: EditorTimelineItem
  track: EditorTrack
  playhead: number
  playing: boolean
}) {
  const ref = useRef<HTMLAudioElement>(null)
  useEffect(() => {
    if (!ref.current) return
    const level = item.muted ? 0 : item.volume * (track.volume ?? 1) * clipEnvelope(item, playhead)
    syncMediaElement(ref.current, item, playhead, playing, level)
  }, [item, playhead, playing, track.volume])
  return <audio ref={ref} src={asset.url} preload="auto" />
}

export function EditorPreview() {
  const isMobile = useIsMobile()
  const project = useEditorStore(state => state.project)
  const playhead = useEditorStore(state => state.playhead)
  const playing = useEditorStore(state => state.playing)
  const selectedItemId = useEditorStore(state => state.selectedItemId)
  const setPlayhead = useEditorStore(state => state.setPlayhead)
  const setPlaying = useEditorStore(state => state.setPlaying)
  const selectItem = useEditorStore(state => state.selectItem)
  const updateItem = useEditorStore(state => state.updateItem)
  const snapping = useEditorStore(state => state.snapping)
  const previewViewportRef = useRef<HTMLDivElement>(null)
  const canvasRef = useRef<HTMLDivElement>(null)
  const canvasTouchPointersRef = useRef(new Map<number, { x: number; y: number }>())
  const playheadRef = useRef(playhead)
  const [canvasInteraction, setCanvasInteraction] = useState<CanvasInteraction | null>(null)
  const [canvasPinch, setCanvasPinch] = useState<CanvasPinchInteraction | null>(null)
  const [canvasGuides, setCanvasGuides] = useState<CanvasGuides>({ x: [], y: [] })
  const [previewCanvasSize, setPreviewCanvasSize] = useState({ width: 0, height: 0 })
  const hasProject = Boolean(project)
  const projectCanvasWidth = project?.canvas.width || 1
  const projectCanvasHeight = project?.canvas.height || 1
  const duration = editorProjectDuration(project)
  const active = useMemo(
    () => project ? activeEditorItems(project, playhead) : [],
    [playhead, project],
  )
  const activeAudio = active.filter(entry => entry.track.type === 'audio')
  const activeVisual = active.filter(entry => entry.track.type === 'video')
  const activeText = active.filter(entry => entry.track.type === 'text')

  useEffect(() => {
    playheadRef.current = playhead
  }, [playhead])

  useLayoutEffect(() => {
    const viewport = previewViewportRef.current
    if (!viewport || !hasProject) return
    const updateSize = () => {
      const next = fitEditorCanvasToViewport(
        viewport.clientWidth,
        viewport.clientHeight,
        projectCanvasWidth,
        projectCanvasHeight,
      )
      setPreviewCanvasSize(current => (
        Math.abs(current.width - next.width) < 0.5
        && Math.abs(current.height - next.height) < 0.5
          ? current
          : next
      ))
    }
    updateSize()
    const observer = typeof ResizeObserver === 'undefined'
      ? null
      : new ResizeObserver(updateSize)
    observer?.observe(viewport)
    window.addEventListener('resize', updateSize)
    return () => {
      observer?.disconnect()
      window.removeEventListener('resize', updateSize)
    }
  }, [hasProject, projectCanvasHeight, projectCanvasWidth])

  useEffect(() => {
    if (!playing || !project) return
    const startedAt = performance.now()
    const originalPlayhead = playheadRef.current
    // The iOS fallback must publish at the same rate as Maestro's mobile
    // proxy. If Safari declines a delayed play() call, timeline-driven seeks
    // still present a real 30 fps preview instead of a 15 fps slideshow.
    const publishInterval = 1000 / 30
    let latestPlayhead = originalPlayhead
    let lastPublishedAt = startedAt - publishInterval
    let frame = 0
    const tick = (now: number) => {
      const next = originalPlayhead + (now - startedAt) / 1000
      latestPlayhead = next
      if (next >= duration) {
        setPlayhead(duration)
        setPlaying(false)
        return
      }
      // Media elements render natively between clock publications. Keeping the
      // global timeline aligned to the 30 fps mobile proxy avoids visible
      // stepping when Safari falls back to timeline-driven seeking.
      if (now - lastPublishedAt >= publishInterval) {
        setPlayhead(next)
        lastPublishedAt = now
      }
      frame = requestAnimationFrame(tick)
    }
    frame = requestAnimationFrame(tick)
    return () => {
      cancelAnimationFrame(frame)
      if (latestPlayhead < duration) setPlayhead(latestPlayhead)
    }
  }, [duration, isMobile, playing, project, setPlayhead, setPlaying])

  if (!project) return <div className="flex min-h-0 flex-1 items-center justify-center text-xs text-text-muted">Opening Editor…</div>

  const seek = (value: number) => setPlayhead(Math.max(0, Math.min(duration, value)))
  const frameDuration = 1 / project.canvas.fps
  const startPlaybackFromGesture = () => {
    // iOS Safari requires unmuted media playback to begin inside the original
    // tap handler. Calling play() only from PreviewVisual's React effect can
    // be rejected, after which the old 15 Hz playhead sync looked like slow
    // playback. Unlock every currently visible media element synchronously.
    canvasRef.current?.querySelectorAll<HTMLMediaElement>('video, audio').forEach(element => {
      void element.play().catch(() => {})
    })
    setPlaying(true)
  }

  const snapTargets = (excludedItemId: string): CanvasGuides => {
    const x = [-project.canvas.width / 2, 0, project.canvas.width / 2]
    const y = [-project.canvas.height / 2, 0, project.canvas.height / 2]
    activeVisual.forEach(({ item }) => {
      if (item.id === excludedItemId) return
      const asset = item.asset_id ? project.assets[item.asset_id] : null
      if (!asset) return
      const bounds = visualBounds(
        asset,
        item,
        project.canvas.width,
        project.canvas.height,
      )
      x.push(
        bounds.centerX - bounds.halfWidth,
        bounds.centerX,
        bounds.centerX + bounds.halfWidth,
      )
      y.push(
        bounds.centerY - bounds.halfHeight,
        bounds.centerY,
        bounds.centerY + bounds.halfHeight,
      )
    })
    return { x: uniqueGuideValues(x), y: uniqueGuideValues(y) }
  }

  const beginCanvasTransform = (
    event: React.PointerEvent<HTMLElement>,
    item: EditorTimelineItem,
    track: EditorTrack,
    mode: CanvasInteractionMode,
  ) => {
    event.preventDefault()
    event.stopPropagation()
    selectItem(item.id, track.id)
    if (track.locked) return
    const canvas = canvasRef.current
    if (!canvas) return
    const rect = canvas.getBoundingClientRect()
    const startTransform = { ...item.transform }
    const centerClientX = rect.left + rect.width / 2 + startTransform.x / project.canvas.width * rect.width
    const centerClientY = rect.top + rect.height / 2 + startTransform.y / project.canvas.height * rect.height
    setPlaying(false)
    setCanvasGuides({ x: [], y: [] })
    canvas.setPointerCapture?.(event.pointerId)
    setCanvasInteraction({
      itemId: item.id,
      pointerId: event.pointerId,
      mode,
      startClientX: event.clientX,
      startClientY: event.clientY,
      centerClientX,
      centerClientY,
      startDistance: Math.max(1, Math.hypot(event.clientX - centerClientX, event.clientY - centerClientY)),
      canvasWidth: rect.width,
      canvasHeight: rect.height,
      startTransform,
      previewTransform: startTransform,
    })
  }

  const beginCanvasTouch = (event: React.PointerEvent<HTMLDivElement>) => {
    if (event.pointerType !== 'touch') return
    canvasTouchPointersRef.current.set(event.pointerId, { x: event.clientX, y: event.clientY })
    if (canvasTouchPointersRef.current.size !== 2 || !selectedItemId) return
    const activeEntry = activeVisual.find(entry => entry.item.id === selectedItemId)
    if (!activeEntry || activeEntry.track.locked) return
    const canvas = canvasRef.current
    if (!canvas) return
    const rect = canvas.getBoundingClientRect()
    const points = Array.from(canvasTouchPointersRef.current.values())
    const deltaX = points[1].x - points[0].x
    const deltaY = points[1].y - points[0].y
    const startTransform = { ...activeEntry.item.transform }
    setPlaying(false)
    setCanvasInteraction(null)
    setCanvasGuides({ x: [], y: [] })
    setCanvasPinch({
      itemId: activeEntry.item.id,
      startDistance: Math.max(1, Math.hypot(deltaX, deltaY)),
      startAngle: Math.atan2(deltaY, deltaX),
      startCenterX: (points[0].x + points[1].x) / 2,
      startCenterY: (points[0].y + points[1].y) / 2,
      canvasWidth: rect.width,
      canvasHeight: rect.height,
      startTransform,
      previewTransform: startTransform,
    })
    event.preventDefault()
    event.stopPropagation()
    canvasTouchPointersRef.current.forEach((_point, pointerId) => {
      try { canvas.setPointerCapture(pointerId) } catch { /* Browser owns this pointer. */ }
    })
  }

  const updateCanvasTransform = (event: React.PointerEvent<HTMLDivElement>) => {
    if (event.pointerType === 'touch' && canvasTouchPointersRef.current.has(event.pointerId)) {
      canvasTouchPointersRef.current.set(event.pointerId, { x: event.clientX, y: event.clientY })
    }
    if (canvasPinch && canvasTouchPointersRef.current.size >= 2) {
      const points = Array.from(canvasTouchPointersRef.current.values())
      const deltaX = points[1].x - points[0].x
      const deltaY = points[1].y - points[0].y
      const centerX = (points[0].x + points[1].x) / 2
      const centerY = (points[0].y + points[1].y) / 2
      const distance = Math.max(1, Math.hypot(deltaX, deltaY))
      const angle = Math.atan2(deltaY, deltaX)
      event.preventDefault()
      setCanvasPinch(current => current ? {
        ...current,
        previewTransform: {
          ...current.startTransform,
          x: Math.max(-project.canvas.width, Math.min(
            project.canvas.width,
            current.startTransform.x
              + (centerX - current.startCenterX) / Math.max(1, current.canvasWidth) * project.canvas.width,
          )),
          y: Math.max(-project.canvas.height, Math.min(
            project.canvas.height,
            current.startTransform.y
              + (centerY - current.startCenterY) / Math.max(1, current.canvasHeight) * project.canvas.height,
          )),
          scale: Math.max(0.05, Math.min(
            4,
            current.startTransform.scale * distance / current.startDistance,
          )),
          rotation: Math.max(-360, Math.min(
            360,
            current.startTransform.rotation + (angle - current.startAngle) * 180 / Math.PI,
          )),
        },
      } : null)
      return
    }
    if (!canvasInteraction || event.pointerId !== canvasInteraction.pointerId) return
    event.preventDefault()
    if (canvasInteraction.mode === 'move') {
      let x = canvasInteraction.startTransform.x
        + (event.clientX - canvasInteraction.startClientX) / Math.max(1, canvasInteraction.canvasWidth) * project.canvas.width
      let y = canvasInteraction.startTransform.y
        + (event.clientY - canvasInteraction.startClientY) / Math.max(1, canvasInteraction.canvasHeight) * project.canvas.height
      const guides: CanvasGuides = { x: [], y: [] }
      const activeEntry = activeVisual.find(entry => entry.item.id === canvasInteraction.itemId)
      const asset = activeEntry?.item.asset_id ? project.assets[activeEntry.item.asset_id] : null
      if (snapping && activeEntry && asset) {
        const rawTransform = { ...canvasInteraction.startTransform, x, y }
        const bounds = visualBounds(
          asset,
          activeEntry.item,
          project.canvas.width,
          project.canvas.height,
          rawTransform,
        )
        const targets = snapTargets(canvasInteraction.itemId)
        const xSnap = nearestGuideOffset(
          [bounds.centerX - bounds.halfWidth, bounds.centerX, bounds.centerX + bounds.halfWidth],
          targets.x,
          CANVAS_SNAP_SCREEN_PX / Math.max(1, canvasInteraction.canvasWidth) * project.canvas.width,
        )
        const ySnap = nearestGuideOffset(
          [bounds.centerY - bounds.halfHeight, bounds.centerY, bounds.centerY + bounds.halfHeight],
          targets.y,
          CANVAS_SNAP_SCREEN_PX / Math.max(1, canvasInteraction.canvasHeight) * project.canvas.height,
        )
        if (xSnap) {
          x += xSnap.offset
          guides.x = [xSnap.guide]
        }
        if (ySnap) {
          y += ySnap.offset
          guides.y = [ySnap.guide]
        }
      }
      setCanvasGuides(guides)
      setCanvasInteraction({
        ...canvasInteraction,
        previewTransform: {
          ...canvasInteraction.startTransform,
          x: Math.max(-project.canvas.width, Math.min(project.canvas.width, x)),
          y: Math.max(-project.canvas.height, Math.min(project.canvas.height, y)),
        },
      })
      return
    }
    const distance = Math.hypot(
      event.clientX - canvasInteraction.centerClientX,
      event.clientY - canvasInteraction.centerClientY,
    )
    let scale = Math.max(
      0.05,
      Math.min(4, canvasInteraction.startTransform.scale * distance / canvasInteraction.startDistance),
    )
    const guides: CanvasGuides = { x: [], y: [] }
    const activeEntry = activeVisual.find(entry => entry.item.id === canvasInteraction.itemId)
    const asset = activeEntry?.item.asset_id ? project.assets[activeEntry.item.asset_id] : null
    if (snapping && activeEntry && asset) {
      const unitBounds = visualBounds(
        asset,
        activeEntry.item,
        project.canvas.width,
        project.canvas.height,
        { ...canvasInteraction.startTransform, scale: 1 },
      )
      const targets = snapTargets(canvasInteraction.itemId)
      const candidates: Array<{ scale: number; guide: number; axis: 'x' | 'y'; distance: number }> = []
      targets.x.forEach(target => [-1, 1].forEach(side => {
        const candidateScale = (target - unitBounds.centerX) / (side * unitBounds.halfWidth)
        if (candidateScale < 0.05 || candidateScale > 4) return
        const edge = unitBounds.centerX + side * unitBounds.halfWidth * scale
        const screenDistance = Math.abs(target - edge) / project.canvas.width * canvasInteraction.canvasWidth
        if (screenDistance <= CANVAS_SNAP_SCREEN_PX) {
          candidates.push({ scale: candidateScale, guide: target, axis: 'x', distance: screenDistance })
        }
      }))
      targets.y.forEach(target => [-1, 1].forEach(side => {
        const candidateScale = (target - unitBounds.centerY) / (side * unitBounds.halfHeight)
        if (candidateScale < 0.05 || candidateScale > 4) return
        const edge = unitBounds.centerY + side * unitBounds.halfHeight * scale
        const screenDistance = Math.abs(target - edge) / project.canvas.height * canvasInteraction.canvasHeight
        if (screenDistance <= CANVAS_SNAP_SCREEN_PX) {
          candidates.push({ scale: candidateScale, guide: target, axis: 'y', distance: screenDistance })
        }
      }))
      candidates.sort((left, right) => left.distance - right.distance)
      const snapped = candidates[0]
      if (snapped) {
        scale = snapped.scale
        guides[snapped.axis] = [snapped.guide]
      }
    }
    setCanvasGuides(guides)
    setCanvasInteraction({
      ...canvasInteraction,
      previewTransform: {
        ...canvasInteraction.startTransform,
        scale,
      },
    })
  }

  const finishCanvasTransform = (event: React.PointerEvent<HTMLDivElement>) => {
    if (event.pointerType === 'touch') canvasTouchPointersRef.current.delete(event.pointerId)
    if (canvasPinch) {
      if (canvasTouchPointersRef.current.size < 2) {
        const { startTransform, previewTransform } = canvasPinch
        const changed = (Object.keys(startTransform) as Array<keyof EditorTransform>).some(key => (
          Math.abs(startTransform[key] - previewTransform[key]) > 1e-4
        ))
        if (changed) updateItem(canvasPinch.itemId, { transform: previewTransform })
        setCanvasPinch(null)
        setCanvasGuides({ x: [], y: [] })
      }
      return
    }
    if (!canvasInteraction || event.pointerId !== canvasInteraction.pointerId) return
    const { startTransform, previewTransform } = canvasInteraction
    const changed = (Object.keys(startTransform) as Array<keyof EditorTransform>).some(key => (
      Math.abs(startTransform[key] - previewTransform[key]) > 1e-4
    ))
    if (changed) updateItem(canvasInteraction.itemId, { transform: previewTransform })
    setCanvasInteraction(null)
    setCanvasGuides({ x: [], y: [] })
    canvasRef.current?.releasePointerCapture?.(event.pointerId)
  }

  return (
    <section className="flex h-full min-h-0 flex-1 flex-col bg-bg-primary">
      <div className="min-h-0 flex-1 overflow-hidden p-3 md:p-5">
        <div ref={previewViewportRef} className="flex h-full min-h-0 w-full min-w-0 items-center justify-center">
          <div
            ref={canvasRef}
            className="relative shrink-0 overflow-hidden rounded-md bg-black shadow-2xl ring-1 ring-white/5"
            style={{
              aspectRatio: `${project.canvas.width} / ${project.canvas.height}`,
              width: `${previewCanvasSize.width || 1}px`,
              height: `${previewCanvasSize.height || 1}px`,
              visibility: previewCanvasSize.width > 0 ? 'visible' : 'hidden',
              background: project.canvas.background,
              touchAction: 'none',
            }}
          onDoubleClick={() => void canvasRef.current?.requestFullscreen?.()}
          onPointerDownCapture={beginCanvasTouch}
          onPointerMove={updateCanvasTransform}
          onPointerUp={finishCanvasTransform}
          onPointerCancel={() => {
            setCanvasInteraction(null)
            setCanvasPinch(null)
            canvasTouchPointersRef.current.clear()
            setCanvasGuides({ x: [], y: [] })
          }}
        >
          {activeVisual.map(({ item, track }) => {
            const asset = item.asset_id ? project.assets[item.asset_id] : null
            return asset ? (
              <PreviewVisual
                key={item.id}
                asset={asset}
                item={item}
                track={track}
                playhead={playhead}
                playing={playing}
                canvasWidth={project.canvas.width}
                canvasHeight={project.canvas.height}
                selected={selectedItemId === item.id}
                locked={track.locked}
                workspace={project.workspace}
                mobilePlayback={isMobile}
                previewTransform={canvasPinch?.itemId === item.id
                  ? canvasPinch.previewTransform
                  : canvasInteraction?.itemId === item.id
                    ? canvasInteraction.previewTransform
                    : undefined}
                onTransformStart={beginCanvasTransform}
              />
            ) : null
          })}
          {activeText.map(({ item }) => {
            const style = item.style || {
              x: 0,
              y: 0,
              font_size: 64,
              font_family: DEFAULT_EDITOR_FONT,
              color: '#ffffff',
              background_color: '#000000',
              background_opacity: 0.32,
              text_align: 'center' as const,
            }
            const textAlign = style.text_align || 'center'
            const translateX = textAlign === 'left' ? 0 : textAlign === 'right' ? -100 : -50
            return (
              <div
                key={item.id}
                className="pointer-events-none absolute max-w-[88%] whitespace-pre-wrap rounded px-3 py-1.5 font-semibold leading-tight drop-shadow-lg"
                style={{
                  color: style.color,
                  fontFamily: editorFontStack(style.font_family),
                  fontSize: `clamp(12px, ${Math.max(1.1, style.font_size / project.canvas.height * 75)}vw, 92px)`,
                  left: `calc(50% + ${style.x / project.canvas.width * 100}%)`,
                  top: `calc(50% + ${style.y / project.canvas.height * 100}%)`,
                  transform: `translate(${translateX}%, -50%)`,
                  textAlign,
                  background: hexToRgba(style.background_color || '#000000', style.background_opacity ?? 0.32),
                  opacity: item.opacity * clipEnvelope(item, playhead),
                }}
              >
                {item.text || 'Title'}
              </div>
            )
          })}
          {activeAudio.map(({ item, track }) => {
            const asset = item.asset_id ? project.assets[item.asset_id] : null
            return asset ? <PreviewAudio key={item.id} asset={asset} item={item} track={track} playhead={playhead} playing={playing} /> : null
          })}
          {canvasGuides.x.map(value => (
            <div
              key={`x-${value}`}
              className="pointer-events-none absolute inset-y-0 z-[80] w-px bg-accent-warm shadow-[0_0_5px_rgba(245,158,11,0.8)]"
              style={{ left: `calc(50% + ${value / project.canvas.width * 100}%)` }}
            />
          ))}
          {canvasGuides.y.map(value => (
            <div
              key={`y-${value}`}
              className="pointer-events-none absolute inset-x-0 z-[80] h-px bg-accent-warm shadow-[0_0_5px_rgba(245,158,11,0.8)]"
              style={{ top: `calc(50% + ${value / project.canvas.height * 100}%)` }}
            />
          ))}
          {activeVisual.length === 0 && activeText.length === 0 && (
            <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 text-center text-white/25">
              <Play size={28} />
              <span className="text-[10px]">Drop media onto the timeline</span>
            </div>
          )}
          </div>
        </div>
      </div>

      <div className="flex h-11 shrink-0 items-center gap-2 border-t border-border bg-bg-secondary px-3 md:px-4">
        <button type="button" onClick={() => seek(0)} className="rounded p-1.5 text-text-muted hover:bg-bg-hover hover:text-text-primary" title="Go to start">
          <SkipBack size={14} />
        </button>
        <button type="button" onClick={() => seek(playhead - frameDuration)} className="hidden rounded p-1.5 text-text-muted hover:bg-bg-hover hover:text-text-primary sm:block" title="Previous frame">
          <RotateCcw size={13} />
        </button>
        <button
          type="button"
          onClick={() => {
            if (!playing && playhead >= duration && duration > 0) seek(0)
            if (playing) setPlaying(false)
            else startPlaybackFromGesture()
          }}
          disabled={duration <= 0}
          className="flex h-7 w-7 items-center justify-center rounded-full bg-text-primary text-bg-primary hover:opacity-90 disabled:opacity-30"
          title={playing ? 'Pause (Space)' : 'Play (Space)'}
        >
          {playing ? <Pause size={13} fill="currentColor" /> : <Play size={13} fill="currentColor" className="ml-0.5" />}
        </button>
        <button type="button" onClick={() => seek(playhead + frameDuration)} className="hidden rounded p-1.5 text-text-muted hover:bg-bg-hover hover:text-text-primary sm:block" title="Next frame">
          <SkipForward size={13} />
        </button>
        <span className="w-[88px] font-mono text-[10px] text-text-secondary">
          {formatEditorTime(playhead, true, project.canvas.fps)}
        </span>
        <input
          type="range"
          min={0}
          max={Math.max(frameDuration, duration)}
          step={frameDuration}
          value={Math.min(playhead, Math.max(frameDuration, duration))}
          onChange={event => seek(Number(event.target.value))}
          className="min-w-0 flex-1 accent-[var(--color-accent-blue)]"
          aria-label="Playhead"
        />
        <span className="hidden w-11 text-right font-mono text-[9px] text-text-muted sm:block">{formatEditorTime(duration)}</span>
        <button type="button" onClick={() => void canvasRef.current?.requestFullscreen?.()} className="rounded p-1.5 text-text-muted hover:bg-bg-hover hover:text-text-primary" title="Fullscreen preview">
          <Maximize2 size={13} />
        </button>
      </div>
    </section>
  )
}
