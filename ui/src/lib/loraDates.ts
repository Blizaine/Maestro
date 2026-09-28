export type LoraDateSource = 'released' | 'downloaded'

export type ResolvedLoraDate = {
  value: string
  timestamp: number
  source: LoraDateSource
}

/** Parse only usable date strings so malformed release metadata can fall back
 *  to a real download date instead of masking it. */
export function parseLoraDate(value: string | null | undefined): number | null {
  if (!value?.trim()) return null
  const timestamp = Date.parse(value)
  return Number.isFinite(timestamp) ? timestamp : null
}

/** The date shown in LoRA age labels and used by newest-first sorting. */
export function resolveLoraDate(
  released: string | null | undefined,
  downloaded: string | null | undefined,
): ResolvedLoraDate | null {
  const releasedTimestamp = parseLoraDate(released)
  if (releasedTimestamp !== null) {
    return { value: released!, timestamp: releasedTimestamp, source: 'released' }
  }

  const downloadedTimestamp = parseLoraDate(downloaded)
  if (downloadedTimestamp !== null) {
    return { value: downloaded!, timestamp: downloadedTimestamp, source: 'downloaded' }
  }
  return null
}

export function formatResolvedLoraDate(date: ResolvedLoraDate): string {
  return new Date(date.timestamp).toLocaleDateString()
}
