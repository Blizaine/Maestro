import { formatDuration, formatTimecode } from './durationPlanning'

/** Bounded footer label. Full duration and frame counts remain in the duration panel. */
export function composerDuration(seconds: number): string {
  if (seconds < 60) return formatDuration(seconds, true)
  return formatTimecode(seconds).replace(/^00:/, '').replace(/^0(?=\d:)/, '')
}
