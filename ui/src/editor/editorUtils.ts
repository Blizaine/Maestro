import type { EditorExportSettings, EditorProject, EditorTimelineItem, EditorTrack } from '../types'

const EDITOR_TIME_EPSILON = 1e-6

function normalizedSpan(start: number, duration: number): { start: number; end: number } {
  const safeStart = Math.max(0, Number.isFinite(start) ? start : 0)
  const safeDuration = Math.max(0, Number.isFinite(duration) ? duration : 0)
  return { start: safeStart, end: safeStart + safeDuration }
}

export function canPlaceEditorItem(
  track: EditorTrack,
  start: number,
  duration: number,
  ignoreItemId?: string,
): boolean {
  const candidate = normalizedSpan(start, duration)
  return !track.items.some(item => {
    if (item.id === ignoreItemId) return false
    const existing = normalizedSpan(item.start, item.duration)
    return candidate.start < existing.end - EDITOR_TIME_EPSILON
      && candidate.end > existing.start + EDITOR_TIME_EPSILON
  })
}

/** Find the closest legal start on a track. Clips may touch, but never overlap. */
export function closestAvailableEditorStart(
  track: EditorTrack,
  requestedStart: number,
  duration: number,
  ignoreItemId?: string,
  forwardOnly = false,
): number {
  const requested = Math.max(0, Number.isFinite(requestedStart) ? requestedStart : 0)
  const safeDuration = Math.max(0, Number.isFinite(duration) ? duration : 0)
  if (canPlaceEditorItem(track, requested, safeDuration, ignoreItemId)) return requested

  const candidates = new Set<number>([0])
  track.items.forEach(item => {
    if (item.id === ignoreItemId) return
    const span = normalizedSpan(item.start, item.duration)
    candidates.add(span.end)
    candidates.add(Math.max(0, span.start - safeDuration))
  })
  const legal = [...candidates].filter(candidate => (
    canPlaceEditorItem(track, candidate, safeDuration, ignoreItemId)
  ))
  const forward = legal
    .filter(candidate => candidate >= requested - EDITOR_TIME_EPSILON)
    .sort((left, right) => left - right)
  if (forwardOnly && forward.length > 0) return forward[0]

  legal.sort((left, right) => {
    const distance = Math.abs(left - requested) - Math.abs(right - requested)
    if (Math.abs(distance) > EDITOR_TIME_EPSILON) return distance
    const leftIsForward = left >= requested - EDITOR_TIME_EPSILON
    const rightIsForward = right >= requested - EDITOR_TIME_EPSILON
    if (leftIsForward !== rightIsForward) return leftIsForward ? -1 : 1
    return left - right
  })
  return legal[0] ?? Math.max(0, ...track.items.map(item => item.start + item.duration))
}

export function previousEditorItemEnd(
  track: EditorTrack,
  itemId: string,
  currentStart: number,
): number {
  return track.items.reduce((boundary, item) => (
    item.id !== itemId && item.start < currentStart - EDITOR_TIME_EPSILON
      ? Math.max(boundary, item.start + item.duration)
      : boundary
  ), 0)
}

export function nextEditorItemStart(
  track: EditorTrack,
  itemId: string,
  currentEnd: number,
): number | null {
  const starts = track.items
    .filter(item => item.id !== itemId && item.start >= currentEnd - EDITOR_TIME_EPSILON)
    .map(item => item.start)
  return starts.length > 0 ? Math.min(...starts) : null
}

export function formatEditorTime(seconds: number, includeFrames = false, fps = 30): string {
  const safe = Math.max(0, Number.isFinite(seconds) ? seconds : 0)
  const hours = Math.floor(safe / 3600)
  const minutes = Math.floor((safe % 3600) / 60)
  const wholeSeconds = Math.floor(safe % 60)
  const frames = Math.floor((safe - Math.floor(safe)) * fps)
  const base = hours > 0
    ? `${hours}:${minutes.toString().padStart(2, '0')}:${wholeSeconds.toString().padStart(2, '0')}`
    : `${minutes}:${wholeSeconds.toString().padStart(2, '0')}`
  return includeFrames ? `${base}:${frames.toString().padStart(2, '0')}` : base
}

export function activeEditorItems(project: EditorProject, time: number): Array<{
  track: EditorTrack
  item: EditorTimelineItem
}> {
  return project.tracks
    .filter(track => !track.muted)
    .flatMap(track => track.items.map(item => ({ track, item })))
    .filter(({ item }) => (
      !item.disabled && time >= item.start && time < item.start + item.duration
    ))
    .sort((left, right) => left.track.z_index - right.track.z_index)
}

export function editorProjectDuration(project: EditorProject | null): number {
  if (!project) return 0
  return project.tracks.reduce((maximum, track) => (
    track.items.reduce((trackMaximum, item) => (
      item.disabled ? trackMaximum : Math.max(trackMaximum, item.start + item.duration)
    ), maximum)
  ), 0)
}

export function editorCanvasLabel(width: number, height: number): string {
  const ratio = width / Math.max(1, height)
  if (Math.abs(ratio - 16 / 9) < 0.03) return '16:9'
  if (Math.abs(ratio - 9 / 16) < 0.03) return '9:16'
  if (Math.abs(ratio - 1) < 0.03) return '1:1'
  if (Math.abs(ratio - 4 / 3) < 0.03) return '4:3'
  return `${width}×${height}`
}

export interface EditorPreviewCanvasSize {
  width: number
  height: number
}

/**
 * Fit the edit canvas inside the preview viewport without ever distorting its
 * aspect ratio. CSS max-width/max-height constraints can independently clamp
 * the two axes, which made a selected layer's resize outline stretch on wide,
 * short browser windows.
 */
export function fitEditorCanvasToViewport(
  viewportWidth: number,
  viewportHeight: number,
  canvasWidth: number,
  canvasHeight: number,
): EditorPreviewCanvasSize {
  const safeViewportWidth = Math.max(0, Number.isFinite(viewportWidth) ? viewportWidth : 0)
  const safeViewportHeight = Math.max(0, Number.isFinite(viewportHeight) ? viewportHeight : 0)
  const safeCanvasWidth = Math.max(1, Number.isFinite(canvasWidth) ? canvasWidth : 1)
  const safeCanvasHeight = Math.max(1, Number.isFinite(canvasHeight) ? canvasHeight : 1)
  if (safeViewportWidth <= 0 || safeViewportHeight <= 0) return { width: 0, height: 0 }

  const aspect = safeCanvasWidth / safeCanvasHeight
  const widthCap = safeCanvasWidth < safeCanvasHeight ? 520 : 980
  let width = Math.min(safeViewportWidth, widthCap)
  let height = width / aspect
  if (height > safeViewportHeight) {
    height = safeViewportHeight
    width = height * aspect
  }
  return {
    width: Math.max(1, width),
    height: Math.max(1, height),
  }
}

function evenDimension(value: number): number {
  return Math.max(2, Math.round(value / 2) * 2)
}

/** Resolve a named delivery preset without changing the project's edit canvas. */
export function editorExportDimensions(
  project: Pick<EditorProject, 'canvas' | 'export'>,
  settings: EditorExportSettings = project.export,
): { width: number; height: number } {
  const { width, height } = project.canvas
  if (settings.resolution === 'canvas') return { width, height }
  const edge = Number.parseInt(settings.resolution, 10)
  if (!Number.isFinite(edge) || edge <= 0) return { width, height }
  if (width === height) return { width: edge, height: edge }
  if (width > height) {
    return { width: evenDimension(edge * width / height), height: edge }
  }
  return { width: edge, height: evenDimension(edge * height / width) }
}

export function editorExportFps(
  project: Pick<EditorProject, 'canvas' | 'export'>,
  settings: EditorExportSettings = project.export,
): number {
  return settings.frame_rate === 'project' ? project.canvas.fps : settings.frame_rate
}
