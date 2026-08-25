import type { EditorProject, EditorTimelineItem, EditorTrack } from '../types'

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
