import type {
  EditorAnimatedProperty,
  EditorKeyframe,
  EditorTimelineItem,
  EditorTrack,
  EditorMediaType,
} from '../types'

export interface EditorAnimatedPropertySettings {
  property: EditorAnimatedProperty
  label: string
  min: number
  max: number
  step: number
  unit?: string
}

export const EDITOR_ANIMATED_PROPERTY_SETTINGS: Record<EditorAnimatedProperty, EditorAnimatedPropertySettings> = {
  x: { property: 'x', label: 'Position X', min: -10000, max: 10000, step: 1, unit: 'px' },
  y: { property: 'y', label: 'Position Y', min: -10000, max: 10000, step: 1, unit: 'px' },
  scale: { property: 'scale', label: 'Scale', min: 0.1, max: 4, step: 0.01, unit: '×' },
  rotation: { property: 'rotation', label: 'Rotation', min: -180, max: 180, step: 1, unit: '°' },
  opacity: { property: 'opacity', label: 'Opacity', min: 0, max: 1, step: 0.01 },
  volume: { property: 'volume', label: 'Volume', min: 0, max: 4, step: 0.01, unit: '×' },
}

export const EDITOR_KEYFRAME_EASINGS: Array<{ value: EditorKeyframe['easing']; label: string }> = [
  { value: 'linear', label: 'Linear' },
  { value: 'hold', label: 'Hold' },
  { value: 'ease-in', label: 'Ease in' },
  { value: 'ease-out', label: 'Ease out' },
  { value: 'ease-in-out', label: 'Ease in and out' },
]

export function getApplicableEditorProperties(
  trackType: EditorTrack['type'],
  assetType?: EditorMediaType,
  hasAudio = false,
): EditorAnimatedProperty[] {
  if (trackType === 'audio') return ['volume']
  if (trackType === 'text') return ['x', 'y', 'scale', 'opacity']
  return [
    'x',
    'y',
    'scale',
    'rotation',
    'opacity',
    ...(assetType === 'video' && hasAudio ? ['volume' as const] : []),
  ]
}

function isTextItem(item: EditorTimelineItem): boolean {
  return item.text !== undefined
}

export function getEditorAnimatedBaseValue(
  item: EditorTimelineItem,
  property: EditorAnimatedProperty,
): number {
  switch (property) {
    case 'x': return isTextItem(item) ? item.style?.x ?? 0 : item.transform.x
    case 'y': return isTextItem(item) ? item.style?.y ?? 0 : item.transform.y
    case 'scale': return item.transform.scale
    case 'rotation': return item.transform.rotation
    case 'opacity': return item.opacity
    case 'volume': return item.volume
  }
  return 0
}

export function clampEditorAnimatedValue(
  property: EditorAnimatedProperty,
  value: number,
  fallback = 0,
): number {
  const settings = EDITOR_ANIMATED_PROPERTY_SETTINGS[property]
  const safeValue = Number.isFinite(value) ? value : fallback
  const safeFallback = Number.isFinite(fallback) ? fallback : settings.min
  return Math.min(settings.max, Math.max(settings.min, Number.isFinite(safeValue) ? safeValue : safeFallback))
}

export function clampEditorKeyframeTime(
  item: EditorTimelineItem,
  time: number,
): number {
  const duration = Number.isFinite(item.duration) ? Math.max(0, item.duration) : 0
  const safeTime = Number.isFinite(time) ? time : 0
  return Math.min(duration, Math.max(0, safeTime))
}

export function editorClipRelativeTime(
  item: EditorTimelineItem,
  absolutePlayhead: number,
): number {
  const time = Number.isFinite(absolutePlayhead) && Number.isFinite(item.start)
    ? absolutePlayhead - item.start
    : 0
  return clampEditorKeyframeTime(item, time)
}

export function getEditorKeyframes(
  item: EditorTimelineItem,
  property: EditorAnimatedProperty,
): EditorKeyframe[] {
  return (item.keyframes?.[property] || [])
    .filter(keyframe => Number.isFinite(keyframe.time) && Number.isFinite(keyframe.value))
    .map(keyframe => ({ ...keyframe, time: Math.max(0, keyframe.time) }))
    .sort((left, right) => left.time - right.time)
}

function easingProgress(easing: EditorKeyframe['easing'], progress: number): number {
  const t = Math.min(1, Math.max(0, progress))
  switch (easing) {
    case 'hold': return t < 1 ? 0 : 1
    case 'ease-in': return t * t
    case 'ease-out': return 1 - (1 - t) * (1 - t)
    case 'ease-in-out': return 3 * t * t - 2 * t * t * t
    case 'linear':
    default: return t
  }
}

function valueAtRelativeTime(
  item: EditorTimelineItem,
  property: EditorAnimatedProperty,
  relativeTime: number,
): number {
  const frames = getEditorKeyframes(item, property)
  if (frames.length === 0) return getEditorAnimatedBaseValue(item, property)
  if (relativeTime <= frames[0].time) return frames[0].value
  const last = frames[frames.length - 1]
  if (relativeTime >= last.time) return last.value

  for (let index = 0; index < frames.length - 1; index += 1) {
    const left = frames[index]
    const right = frames[index + 1]
    if (relativeTime > right.time) continue
    const interval = right.time - left.time
    if (interval <= 0) return right.value
    const progress = (relativeTime - left.time) / interval
    const eased = easingProgress(left.easing, progress)
    return left.value + ((right.value - left.value) * eased)
  }
  return last.value
}

export function evaluateEditorProperty(
  item: EditorTimelineItem,
  property: EditorAnimatedProperty,
  absolutePlayhead: number,
): number {
  const relativeTime = Number.isFinite(absolutePlayhead) && Number.isFinite(item.start)
    ? absolutePlayhead - item.start
    : 0
  const duration = Number.isFinite(item.duration) ? Math.max(0, item.duration) : 0
  return valueAtRelativeTime(item, property, Math.min(duration, Math.max(0, relativeTime)))
}

/** Return an evaluated view of an item while leaving its authored data intact. */
export function evaluateEditorItem(
  item: EditorTimelineItem,
  absolutePlayhead: number,
): EditorTimelineItem {
  const x = evaluateEditorProperty(item, 'x', absolutePlayhead)
  const y = evaluateEditorProperty(item, 'y', absolutePlayhead)
  const style = item.style || (isTextItem(item)
    ? { x: 0, y: 0, font_size: 64, color: '#ffffff' }
    : undefined)
  return {
    ...item,
    opacity: evaluateEditorProperty(item, 'opacity', absolutePlayhead),
    volume: evaluateEditorProperty(item, 'volume', absolutePlayhead),
    ...(style ? {
      style: {
        ...style,
        x: isTextItem(item) ? x : style.x,
        y: isTextItem(item) ? y : style.y,
      },
    } : {}),
    transform: {
      ...item.transform,
      x,
      y,
      scale: evaluateEditorProperty(item, 'scale', absolutePlayhead),
      rotation: evaluateEditorProperty(item, 'rotation', absolutePlayhead),
    },
  }
}

function updateKeyframes(
  item: EditorTimelineItem,
  property: EditorAnimatedProperty,
  frames: EditorKeyframe[],
): EditorTimelineItem {
  const keyframes = { ...(item.keyframes || {}) }
  if (frames.length) keyframes[property] = frames
  else delete keyframes[property]
  return {
    ...item,
    keyframes: Object.keys(keyframes).length ? keyframes : undefined,
  }
}

export function upsertEditorKeyframe(
  item: EditorTimelineItem,
  property: EditorAnimatedProperty,
  time: number,
  value: number,
  easing?: EditorKeyframe['easing'],
): EditorTimelineItem {
  const frames = getEditorKeyframes(item, property)
  const keyTime = clampEditorKeyframeTime(item, time)
  const existing = frames.find(frame => Math.abs(frame.time - keyTime) < 1e-4)
  const nextFrames = frames.filter(frame => Math.abs(frame.time - keyTime) >= 1e-4)
  if (frames.length === 0 && keyTime > 1e-4) {
    nextFrames.push({
      time: 0,
      value: clampEditorAnimatedValue(property, getEditorAnimatedBaseValue(item, property)),
      easing: 'linear',
    })
  }
  nextFrames.push({
    time: keyTime,
    value: clampEditorAnimatedValue(property, value, getEditorAnimatedBaseValue(item, property)),
    easing: easing || existing?.easing || 'linear',
  })
  nextFrames.sort((left, right) => left.time - right.time)
  return updateKeyframes(item, property, nextFrames)
}

export function updateEditorKeyframe(
  item: EditorTimelineItem,
  property: EditorAnimatedProperty,
  index: number,
  patch: Partial<EditorKeyframe>,
): EditorTimelineItem {
  const frames = getEditorKeyframes(item, property)
  const current = frames[index]
  if (!current) return item
  const next = {
    ...current,
    ...patch,
    time: clampEditorKeyframeTime(item, patch.time ?? current.time),
    value: clampEditorAnimatedValue(
      property,
      patch.value ?? current.value,
      getEditorAnimatedBaseValue(item, property),
    ),
  }
  const nextFrames = frames
    .filter((frame, frameIndex) => frameIndex !== index && Math.abs(frame.time - next.time) >= 1e-4)
    .concat(next)
    .sort((left, right) => left.time - right.time)
  return updateKeyframes(item, property, nextFrames)
}

export function deleteEditorKeyframe(
  item: EditorTimelineItem,
  property: EditorAnimatedProperty,
  index: number,
): EditorTimelineItem {
  const frames = getEditorKeyframes(item, property)
  if (!frames[index]) return item
  return updateKeyframes(item, property, frames.filter((_, frameIndex) => frameIndex !== index))
}
