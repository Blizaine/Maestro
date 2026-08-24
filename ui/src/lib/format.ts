/** Shared byte formatting — one rounding rule everywhere a size shows
 *  (LoRA cards, CivitAI detail pane, future storage views). */
export function formatBytes(bytes: number): string {
  if (bytes >= 1073741824) return `${(bytes / 1073741824).toFixed(1)} GB`
  if (bytes >= 1048576) return `${(bytes / 1048576).toFixed(0)} MB`
  if (bytes >= 1024) return `${(bytes / 1024).toFixed(0)} KB`
  return `${bytes} B`
}

/** Compact age for dense list rows: "today", "3d", "2w", "5mo", "1y".
 *  Returns '' for null/invalid input so call sites can render-and-forget. */
export function formatAge(iso: string | null | undefined): string {
  if (!iso) return ''
  const t = Date.parse(iso)
  if (Number.isNaN(t)) return ''
  const days = Math.max(0, Math.floor((Date.now() - t) / 86400000))
  if (days < 1) return 'today'
  if (days < 7) return `${days}d`
  if (days < 30) return `${Math.floor(days / 7)}w`
  // Clamp at 11mo: days 360-364 would otherwise floor to "12mo" while
  // still failing the < 365 year cutoff.
  if (days < 365) return `${Math.min(11, Math.floor(days / 30))}mo`
  return `${Math.floor(days / 365)}y`
}

/** Gallery generation duration as whole minutes + seconds: "2m 8s". */
export function formatGenerationDuration(seconds: number): string {
  const rounded = Number.isFinite(seconds) ? Math.max(0, Math.round(seconds)) : 0
  const minutes = Math.floor(rounded / 60)
  return `${minutes}m ${rounded % 60}s`
}

/** Human-scale ETA without false precision: "~50s", "~12m", "~1h 25m". */
export function formatEtaDuration(seconds: number | null | undefined): string {
  if (seconds == null || !Number.isFinite(seconds)) return ''
  const safe = Math.max(0, seconds)
  if (safe < 90) {
    return `~${Math.max(10, Math.round(safe / 10) * 10)}s`
  }
  if (safe < 3600) {
    return `~${Math.max(1, Math.round(safe / 60))}m`
  }
  const roundedMinutes = Math.max(5, Math.round(safe / 300) * 5)
  const hours = Math.floor(roundedMinutes / 60)
  const minutes = roundedMinutes % 60
  return `~${hours}h${minutes ? ` ${minutes}m` : ''}`
}

/** Local wall-clock completion time from an epoch-seconds prediction. */
export function formatEstimatedClock(epochSeconds: number | null | undefined): string {
  if (epochSeconds == null || !Number.isFinite(epochSeconds)) return ''
  return new Date(epochSeconds * 1000).toLocaleTimeString([], {
    hour: 'numeric',
    minute: '2-digit',
  })
}
