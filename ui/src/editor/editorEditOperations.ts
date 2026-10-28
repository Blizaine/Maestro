import type { EditorProject, EditorTimelineItem, EditorTrack } from '../types'
import { evaluateEditorProperty, getEditorKeyframes } from './editorAnimation'
import type { CaptionCue, CaptionTimelineItem } from './editorCaptions'

export type EditorEditOperation =
  | { type: 'slip'; itemId: string; seconds: number }
  | { type: 'roll'; itemId: string; seconds: number }
  | { type: 'close-gap'; itemId: string }
  | { type: 'crossfade'; itemId: string; seconds: number }
  | { type: 'import-captions'; cues: CaptionCue[]; offset?: number; name?: string }
  | { type: 'duck-audio'; itemId: string; narrationTrackId: string; duckDb: number; attack: number; release: number }
  | { type: 'set-volume'; itemId: string; volume: number; envelopeGain?: number }

function locate(project: EditorProject, itemId: string): { item: EditorTimelineItem; track: EditorTrack } {
  for (const track of project.tracks) {
    const item = track.items.find(candidate => candidate.id === itemId)
    if (item) return { item, track }
  }
  throw new Error('The selected clip is no longer in this project.')
}

function linkedIds(project: EditorProject, seeds: string[]): Set<string> {
  const ids = new Set(seeds)
  const links = new Set(project.tracks.flatMap(track => track.items)
    .filter(item => ids.has(item.id) && item.link_group_id).map(item => item.link_group_id))
  project.tracks.forEach(track => track.items.forEach(item => {
    if (item.link_group_id && links.has(item.link_group_id)) ids.add(item.id)
  }))
  return ids
}

function hasSourceRange(project: EditorProject, item: EditorTimelineItem, sourceIn: number, duration: number): boolean {
  const asset = item.asset_id ? project.assets[item.asset_id] : undefined
  return sourceIn >= -0.000001 && duration >= 1 / project.canvas.fps - 0.000001
    && (!asset || asset.type === 'image' || asset.duration <= 0
      || sourceIn + duration * item.speed <= asset.duration + 0.000001)
}

function editorId(): string {
  return globalThis.crypto?.randomUUID?.() || `caption-${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`
}

function importCaptions(project: EditorProject, operation: Extract<EditorEditOperation, { type: 'import-captions' }>): void {
  const offset = operation.offset ?? 0
  const name = operation.name?.trim() || 'Captions'
  const minimumDuration = 1 / Math.max(1, project.canvas.fps || 24)
  if (!Number.isFinite(offset)) throw new Error('Choose a finite caption offset.')
  if (!name) throw new Error('Choose a caption track name.')
  if (!Array.isArray(operation.cues) || operation.cues.length === 0) throw new Error('Add at least one caption cue.')
  if (operation.cues.length > 5000) throw new Error('Import up to 5,000 caption cues per track.')
  const cues = operation.cues.map((cue, index) => {
    const start = Number(cue.start) + offset
    const end = Number(cue.end) + offset
    const text = typeof cue.text === 'string' ? cue.text.trim() : ''
    if (!Number.isFinite(start) || start < 0) throw new Error(`Caption ${index + 1} starts before the project.`)
    if (!Number.isFinite(end) || end - start + 1e-6 < minimumDuration) {
      throw new Error(`Caption ${index + 1} is shorter than one project frame.`)
    }
    if (end > 86400) throw new Error(`Caption ${index + 1} extends past the 24-hour project limit.`)
    if (!text) throw new Error(`Caption ${index + 1} needs text.`)
    return { start, end, text }
  }).sort((left, right) => left.start - right.start || left.end - right.end)

  const existing = project.tracks.find(track => track.type === 'text' && track.name === name)
  if (existing?.locked) throw new Error('Unlock the caption track before importing cues.')
  if ((existing?.items.length || 0) + cues.length > 5000 || (!existing && project.tracks.length >= 100)) throw new Error('This import would exceed the project track or clip limit.')
  for (let index = 0; index < cues.length; index++) {
    if (index > 0 && cues[index].start < cues[index - 1].end - 0.000001) throw new Error('Caption cues overlap. Adjust their timing or import them on separate tracks.')
    if (existing?.items.some(item => cues[index].start < item.start + item.duration - 0.000001 && cues[index].end > item.start + 0.000001)) {
      throw new Error('These captions overlap cues already on this track. Choose a new track name or adjust the offset.')
    }
  }
  const track: EditorTrack = existing || {
    id: editorId(),
    name,
    type: 'text',
    z_index: Math.min(100, Math.max(0, ...project.tracks.map(candidate => Number.isFinite(candidate.z_index) ? candidate.z_index : 0)) + 1),
    muted: false,
    locked: false,
    volume: 1,
    items: [],
  }
  const fontSize = Math.min(88, Math.max(36, Math.round(project.canvas.height * 0.06)))
  const newItems = cues.map((cue, index) => ({
    id: editorId(),
    name: `Caption ${index + 1}`,
    start: cue.start,
    duration: cue.end - cue.start,
    source_in: 0,
    speed: 1,
    volume: 1,
    opacity: 1,
    fit: 'contain' as const,
    transform: { x: 0, y: 0, scale: 1, rotation: 0 },
    text: cue.text,
    caption: true,
    style: {
      x: 0,
      y: Math.round(project.canvas.height * 0.32),
      font_family: 'Arial',
      font_size: fontSize,
      color: '#ffffff',
      background_color: '#000000',
      background_opacity: 0.5,
      text_align: 'center' as const,
    },
  } satisfies CaptionTimelineItem))
  if (!existing) project.tracks.push(track)
  track.items.push(...newItems)
  track.items.sort((left, right) => left.start - right.start)
}

function duckAudio(project: EditorProject, operation: Extract<EditorEditOperation, { type: 'duck-audio' }>): void {
  const { item: music, track: musicTrack } = locate(project, operation.itemId)
  const narrationTrack = project.tracks.find(track => track.id === operation.narrationTrackId)
  if (!narrationTrack) throw new Error('Choose a narration track in this project.')
  if (musicTrack.locked || narrationTrack.locked) throw new Error('Unlock the music and narration tracks before ducking audio.')
  if (musicTrack.id === narrationTrack.id) throw new Error('Choose a separate narration track from the music clip.')
  if (narrationTrack.type !== 'audio' && narrationTrack.type !== 'video') throw new Error('Choose an audio or video narration track.')
  const musicAsset = music.asset_id ? project.assets[music.asset_id] : undefined
  if (musicTrack.type === 'text' || (musicTrack.type !== 'audio' && !(musicAsset?.type === 'video' && musicAsset.has_audio))) {
    throw new Error('Select a music or other clip with an audio source.')
  }
  const { duckDb, attack, release } = operation
  if (![duckDb, attack, release].every(Number.isFinite) || duckDb < 0 || duckDb > 36 || attack < 0 || attack > 10 || release < 0 || release > 10) {
    throw new Error('Duck depth must be 0–36 dB and attack/release must be 0–10 seconds.')
  }
  if (!Number.isFinite(music.start) || !Number.isFinite(music.duration) || music.duration <= 0) {
    throw new Error('The selected music clip has an invalid timeline range.')
  }
  const overlaps = narrationTrack.items
    .filter(() => !narrationTrack.muted && (narrationTrack.volume ?? 1) > 0)
    .filter(item => item.id !== music.id && !item.muted && !item.disabled)
    .filter(item => narrationTrack.type === 'audio' || Boolean(project.assets[item.asset_id || '']?.has_audio))
    .map(item => ({ start: Math.max(0, item.start - music.start), end: Math.min(music.duration, item.start + item.duration - music.start) }))
    .filter(range => range.end > range.start)
    .sort((left, right) => left.start - right.start)
  if (!overlaps.length) throw new Error('There are no narration clips overlapping this music clip.')
  const ranges = overlaps.reduce<Array<{ start: number; end: number }>>((result, range) => {
    const previous = result[result.length - 1]
    if (previous && range.start <= previous.end + 1e-6) previous.end = Math.max(previous.end, range.end)
    else result.push({ ...range })
    return result
  }, [])
  const gain = 10 ** (-duckDb / 20)
  const originalFrames = getEditorKeyframes(music, 'volume')
  const sampleTimes = new Set<number>([0, music.duration, ...originalFrames.map(frame => Math.min(music.duration, frame.time))])
  const transitions = ranges.map(range => {
    const span = range.end - range.start
    const transitionTotal = attack + release
    const attackDuration = transitionTotal > span && transitionTotal > 0 ? span * attack / transitionTotal : Math.min(attack, span)
    const releaseDuration = transitionTotal > span && transitionTotal > 0 ? span * release / transitionTotal : Math.min(release, span - attackDuration)
    return { ...range, attackEnd: range.start + attackDuration, releaseStart: range.end - releaseDuration, releaseDuration }
  })
  transitions.forEach(range => {
    sampleTimes.add(range.start)
    sampleTimes.add(range.attackEnd)
    sampleTimes.add(range.releaseStart)
    sampleTimes.add(range.end)
  })
  const times = [...sampleTimes].sort((left, right) => left - right)
  if (times.length > 1000) throw new Error('This ducking edit needs more than 1,000 volume keys. Split the music clip into shorter sections first.')
  const nextFrames = times.map((time, index) => {
    let multiplier = 1
    for (const range of transitions) {
      if (time < range.start || time > range.end) continue
      if (time === range.end) {
        multiplier = 1
        break
      }
      if (range.attackEnd > range.start && time < range.attackEnd) {
        const progress = (time - range.start) / (range.attackEnd - range.start)
        multiplier = 1 + (gain - 1) * progress
      } else if (range.releaseDuration > 0 && time >= range.releaseStart) {
        const progress = (time - range.releaseStart) / (range.end - range.releaseStart)
        multiplier = gain + (1 - gain) * progress
      } else multiplier = gain
      break
    }
    const base = evaluateEditorProperty(music, 'volume', music.start + time)
    const stepAtNext = transitions.some(range => (
      (range.attackEnd === range.start && Math.abs(range.start - times[index + 1]) < 1e-6)
      || (range.releaseDuration === 0 && Math.abs(range.end - times[index + 1]) < 1e-6)
    ))
    return { time, value: Math.min(4, Math.max(0, base * multiplier)), easing: stepAtNext ? 'hold' as const : 'linear' as const }
  })
  music.keyframes = {
    ...(music.keyframes || {}),
    volume: nextFrames,
  }
}

/** Apply to an isolated project draft. Validation completes before each mutation. */
export function applyEditorEditOperation(project: EditorProject, operation: EditorEditOperation): void {
  if (operation.type === 'import-captions') return importCaptions(project, operation)
  if (operation.type === 'duck-audio') return duckAudio(project, operation)
  const { item, track } = locate(project, operation.itemId)
  if (track.locked) throw new Error('Unlock this track before editing its clips.')
  if (operation.type === 'set-volume') {
    if (!Number.isFinite(operation.volume) || operation.volume < 0 || operation.volume > 4) {
      throw new Error('Clip volume must be between 0 and 4.')
    }
    if (operation.envelopeGain !== undefined
      && (!Number.isFinite(operation.envelopeGain) || operation.envelopeGain < 0 || operation.envelopeGain > 10000)) {
      throw new Error('The volume-envelope gain is invalid.')
    }
    item.volume = operation.volume
    if (operation.envelopeGain !== undefined && item.keyframes?.volume) {
      item.keyframes = {
        ...item.keyframes,
        volume: item.keyframes.volume.map(frame => ({
          ...frame,
          value: Math.min(4, Math.max(0, frame.value * operation.envelopeGain!)),
        })),
      }
    }
    return
  }
  if ('seconds' in operation && !Number.isFinite(operation.seconds)) throw new Error('Choose a finite edit offset.')
  if (operation.type === 'crossfade') {
    if (track.type !== 'video') throw new Error('Crossfades require a visual track.')
    const previous = [...track.items].filter(candidate => candidate.id !== item.id && candidate.start < item.start)
      .sort((a, b) => b.start - a.start)[0]
    if (!previous) throw new Error('Select a clip with a preceding clip on this track.')
    const seconds = operation.seconds
    if (seconds < 0 || seconds > Math.min(previous.duration, item.duration)) throw new Error('The crossfade must fit inside both clips.')
    const following = track.items.filter(candidate => candidate.start >= item.start)
    if (previous.link_group_id || following.some(candidate => candidate.link_group_id)) {
      throw new Error('Unlink these clips before changing overlap, to preserve linked audio timing.')
    }
    const nextStart = previous.start + previous.duration - seconds
    const delta = nextStart - item.start
    following.forEach(candidate => { candidate.start += delta })
    item.transition_overlap = seconds
    item.transition_in = seconds > 0 ? 'dissolve' : 'none'
    item.fade_in = seconds
    previous.fade_out = 0
    previous.transition_out = 'none'
    return
  }
  if (operation.type === 'slip') {
    const ids = linkedIds(project, [item.id])
    const entries = [...ids].map(id => locate(project, id))
    if (entries.some(entry => entry.track.locked)) throw new Error('A linked clip is on a locked track.')
    if (entries.some(entry => entry.track.type === 'text' || project.assets[entry.item.asset_id || '']?.type === 'image')) {
      throw new Error('Slip editing requires video or audio source media.')
    }
    if (entries.some(({ item: candidate }) => !hasSourceRange(project, candidate,
      candidate.source_in + operation.seconds * candidate.speed, candidate.duration))) {
      throw new Error('There is not enough source media for this slip edit.')
    }
    entries.forEach(entry => { entry.item.source_in = Math.max(0, entry.item.source_in + operation.seconds * entry.item.speed) })
    return
  }
  if (operation.type === 'roll') {
    const end = item.start + item.duration
    const next = track.items.find(candidate => candidate.id !== item.id && Math.abs(candidate.start - end) < 0.00001)
    if (!next) throw new Error('Roll editing needs an adjoining clip at this clip’s end.')
    if (item.link_group_id || next.link_group_id) throw new Error('Unlink these clips before rolling this cut, to preserve linked audio timing.')
    const delta = operation.seconds
    const nextSource = next.source_in + delta * next.speed
    if (!hasSourceRange(project, item, item.source_in, item.duration + delta)
      || !hasSourceRange(project, next, nextSource, next.duration - delta)) {
      throw new Error('There is not enough source media on both sides of this cut.')
    }
    item.duration += delta
    next.start += delta
    next.duration -= delta
    next.source_in = Math.max(0, nextSource)
    ;[item, next].forEach(candidate => {
      candidate.fade_in = Math.min(candidate.duration, candidate.fade_in || 0)
      candidate.fade_out = Math.min(candidate.duration, candidate.fade_out || 0)
    })
    return
  }
  if (operation.type !== 'close-gap') throw new Error('This edit operation is not supported.')
  const previousEnd = track.items.filter(candidate => candidate.id !== item.id
    && candidate.start + candidate.duration <= item.start + 0.000001)
    .reduce((maximum, candidate) => Math.max(maximum, candidate.start + candidate.duration), 0)
  const gap = item.start - previousEnd
  if (gap < 0.000001) throw new Error('There is no gap immediately before this clip.')
  const moving = linkedIds(project, track.items.filter(candidate => candidate.start >= item.start).map(candidate => candidate.id))
  const entries = [...moving].map(id => locate(project, id))
  if (entries.some(entry => entry.track.locked)) throw new Error('A linked clip is on a locked track.')
  for (const entry of entries) {
    const start = entry.item.start - gap
    if (start < -0.000001 || entry.track.items.some(other => !moving.has(other.id)
      && start < other.start + other.duration - 0.000001
      && start + entry.item.duration > other.start + 0.000001)) {
      throw new Error('Closing this gap would overlap another linked clip.')
    }
  }
  entries.forEach(entry => { entry.item.start = Math.max(0, entry.item.start - gap) })
  project.tracks.forEach(candidate => candidate.items.sort((a, b) => a.start - b.start))
}
