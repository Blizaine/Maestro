import type { OutputTimestamp } from '../types'

const timestampLabels: Record<OutputTimestamp['kind'], string> = {
  generated: 'Generated',
  processed: 'Processed',
  uploaded: 'Uploaded',
  file_modified: 'File date',
}

export function getMediaTimestamp(timestamp?: OutputTimestamp, fallbackCreatedAt?: number) {
  const value = typeof timestamp?.value === 'number' && Number.isFinite(timestamp.value) && timestamp.value > 0
    ? timestamp.value
    : typeof fallbackCreatedAt === 'number' && Number.isFinite(fallbackCreatedAt) && fallbackCreatedAt > 0
      ? fallbackCreatedAt
      : null
  if (value == null) return null

  // The API uses epoch seconds; accept milliseconds too for older local fixtures.
  const date = new Date(value < 1_000_000_000_000 ? value * 1000 : value)
  if (!Number.isFinite(date.getTime())) return null
  const compact = new Intl.DateTimeFormat(undefined, { dateStyle: 'medium', timeStyle: 'short' }).format(date)
  const exact = new Intl.DateTimeFormat(undefined, {
    year: 'numeric', month: 'long', day: 'numeric', hour: 'numeric', minute: '2-digit', second: '2-digit', timeZoneName: 'short',
  }).format(date)
  const kind = timestamp?.value === value ? timestamp.kind : 'file_modified'
  return { value, date, compact, exact, kind, label: timestampLabels[kind] }
}
