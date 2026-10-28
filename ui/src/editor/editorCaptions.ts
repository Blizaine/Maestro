import type { EditorProject, EditorTimelineItem } from '../types'

export interface CaptionCue {
  start: number
  end: number
  text: string
}

export type CaptionFormat = 'srt' | 'vtt'
export type CaptionTimelineItem = EditorTimelineItem & { caption?: boolean }

function parseTimestamp(value: string, format: CaptionFormat): number {
  const timestamp = value.trim()
  if (format === 'srt') {
    const match = timestamp.match(/^(\d{2,}):([0-5]\d):([0-5]\d)[,.](\d{3})$/)
    if (!match) throw new Error(`Invalid SRT timestamp “${value}”. Use HH:MM:SS,mmm.`)
    const [, hours, minutes, seconds, milliseconds] = match
    return Number(hours) * 3600 + Number(minutes) * 60 + Number(seconds) + Number(milliseconds) / 1000
  }

  const match = timestamp.match(/^(?:(\d{2,}):)?(\d{2,}):([0-5]\d)\.(\d{3})$/)
  if (!match) throw new Error(`Invalid WebVTT timestamp “${value}”. Use MM:SS.mmm or HH:MM:SS.mmm.`)
  const [, hours, minutes, seconds, milliseconds] = match
  if (hours && Number(minutes) > 59) throw new Error(`Invalid WebVTT timestamp “${value}”. Minutes must be below 60 when hours are present.`)
  return Number(hours || 0) * 3600 + Number(minutes) * 60 + Number(seconds) + Number(milliseconds) / 1000
}

function cueFromBlock(lines: string[], format: CaptionFormat, cueNumber: number): CaptionCue {
  const timingIndex = lines.findIndex(line => line.includes('-->'))
  if (timingIndex < 0 || timingIndex > 1) {
    throw new Error(`Cue ${cueNumber} is missing a valid timestamp line.`)
  }
  const timingLine = lines[timingIndex].trim()
  const timingMatch = timingLine.match(/^(.+?)\s+-->\s+(\S+)(?:\s+.*)?$/)
  if (!timingMatch) throw new Error(`Cue ${cueNumber} has an invalid timestamp line.`)
  const start = parseTimestamp(timingMatch[1], format)
  const end = parseTimestamp(timingMatch[2], format)
  const text = lines.slice(timingIndex + 1).join('\n').trim()
  if (!text) throw new Error(`Cue ${cueNumber} has no subtitle text.`)
  if (end <= start) throw new Error(`Cue ${cueNumber} must end after it starts.`)
  return { start, end, text }
}

function splitBlocks(source: string): string[][] {
  return source
    .replace(/^\uFEFF/, '')
    .replace(/\r/g, '')
    .split(/\n\s*\n/)
    .map(block => block.split('\n').map(line => line.trimEnd()))
    .filter(lines => lines.some(line => line.trim()))
}

function sortedValidatedCues(cues: CaptionCue[]): CaptionCue[] {
  if (!cues.length) throw new Error('Add at least one caption cue.')
  return cues.map((cue, index) => {
    const start = Number(cue.start)
    const end = Number(cue.end)
    const text = typeof cue.text === 'string' ? cue.text.trim() : ''
    if (!Number.isFinite(start) || start < 0) throw new Error(`Cue ${index + 1} start time must be zero or later.`)
    if (!Number.isFinite(end) || end <= start) throw new Error(`Cue ${index + 1} end time must be later than its start.`)
    if (!text) throw new Error(`Cue ${index + 1} needs subtitle text.`)
    return { start, end, text }
  }).sort((left, right) => left.start - right.start || left.end - right.end)
}

export function parseSrt(source: string): CaptionCue[] {
  return sortedValidatedCues(splitBlocks(source).map((lines, index) => cueFromBlock(lines, 'srt', index + 1)))
}

export function parseWebVtt(source: string): CaptionCue[] {
  const lines = source.replace(/^\uFEFF/, '').replace(/\r/g, '').split('\n')
  const header = lines[0]?.trim()
  if (!header || !/^WEBVTT(?:\s|$)/.test(header)) throw new Error('WebVTT files must begin with a WEBVTT header.')
  // The blank line after the required header belongs to the header block. If
  // left in place, splitBlocks keeps that leading empty line with a following
  // NOTE/STYLE block, so the block filter mistakes the metadata for a cue.
  const body = lines.slice(1).join('\n').replace(/^(?:[ \t]*\n)+/, '')
  const blocks = splitBlocks(body).filter(block => {
    const first = block.find(line => line.trim())?.trim() || ''
    return !/^(NOTE|STYLE|REGION)(?:\s|$)/.test(first)
  })
  return sortedValidatedCues(blocks.map((block, index) => cueFromBlock(block, 'vtt', index + 1)))
}

export function parseCaptionFile(source: string, filename = ''): CaptionCue[] {
  const isVtt = filename.toLowerCase().endsWith('.vtt') || /^\uFEFF?\s*WEBVTT(?:\s|$)/i.test(source)
  return isVtt ? parseWebVtt(source) : parseSrt(source)
}

function millisecondsToTimestamp(seconds: number, decimalSeparator: ',' | '.'): string {
  const totalMilliseconds = Math.round(Math.max(0, seconds) * 1000)
  const hours = Math.floor(totalMilliseconds / 3_600_000)
  const minutes = Math.floor(totalMilliseconds / 60_000) % 60
  const wholeSeconds = Math.floor(totalMilliseconds / 1000) % 60
  const milliseconds = totalMilliseconds % 1000
  return `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}:${String(wholeSeconds).padStart(2, '0')}${decimalSeparator}${String(milliseconds).padStart(3, '0')}`
}

function webVttTimestamp(seconds: number): string {
  const timestamp = millisecondsToTimestamp(seconds, '.')
  return timestamp.startsWith('00:') ? timestamp.slice(3) : timestamp
}

export function exportSrt(cues: CaptionCue[]): string {
  return sortedValidatedCues(cues).map((cue, index) => (
    `${index + 1}\n${millisecondsToTimestamp(cue.start, ',')} --> ${millisecondsToTimestamp(cue.end, ',')}\n${cue.text}`
  )).join('\n\n') + '\n'
}

export function exportWebVtt(cues: CaptionCue[]): string {
  const body = sortedValidatedCues(cues).map(cue => (
    `${webVttTimestamp(cue.start)} --> ${webVttTimestamp(cue.end)}\n${cue.text}`
  )).join('\n\n')
  return `WEBVTT\n\n${body}\n`
}

export function collectCaptionTrackCues(project: EditorProject, trackId?: string): CaptionCue[] {
  const cues: CaptionCue[] = []
  project.tracks
    .filter(track => track.type === 'text' && (!trackId || track.id === trackId))
    .forEach(track => {
      track.items.forEach(item => {
        if (!(item as CaptionTimelineItem).caption || !item.text?.trim()) return
        cues.push({ start: item.start, end: item.start + item.duration, text: item.text })
      })
    })
  return cues.sort((left, right) => left.start - right.start || left.end - right.end)
}

export function captionTracks(project: EditorProject): Array<{ id: string; name: string; count: number }> {
  return project.tracks
    .filter(track => track.type === 'text')
    .map(track => ({
      id: track.id,
      name: track.name,
      count: track.items.filter(item => (item as CaptionTimelineItem).caption && Boolean(item.text?.trim())).length,
    }))
    .filter(track => track.count > 0)
}
