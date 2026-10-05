import { useEffect, useMemo, useState } from 'react'
import { Check, Loader2, AlertCircle } from 'lucide-react'
import { useStore } from '../../stores/useStore'
import { cancelDownload } from '../../api/client'
import type { CivitAIDownload } from '../../types'

const COMPLETED_VISIBLE_MS = 30_000
type CancelStatus = 'pending' | 'cancelling' | 'cancelled' | 'completed' | 'failed'

function timestampMs(value: unknown): number | null {
  const timestamp = Number(value)
  if (!Number.isFinite(timestamp) || timestamp <= 0) return null
  // The backend uses Unix seconds, but accepting milliseconds keeps this
  // tolerant of persisted/older records from alternate download sources.
  return timestamp < 1_000_000_000_000 ? timestamp * 1000 : timestamp
}

function displayProgress(download: CivitAIDownload): number {
  const downloaded = Number(download.bytes_downloaded)
  const total = Number(download.bytes_total)
  let progress = Number(download.progress)

  // Byte counts are the least ambiguous source. Otherwise trust the
  // backend's normalized 0..100 contract and clamp corrupt values.
  if (Number.isFinite(downloaded) && Number.isFinite(total) && total > 0) {
    progress = (Math.max(0, downloaded) / total) * 100
  }

  if (!Number.isFinite(progress)) return 0
  return Math.min(100, Math.max(0, progress))
}

export function DownloadBar() {
  const downloads = useStore(s => s.civitDownloads)
  const [now, setNow] = useState<number | null>(null)
  const [cancelStatuses, setCancelStatuses] = useState<Record<string, CancelStatus>>({})
  const [cancelErrors, setCancelErrors] = useState<Record<string, string>>({})
  const [cancelledAt, setCancelledAt] = useState<Record<string, number>>({})

  const visible = useMemo(() => downloads.filter(download => {
    const cancelId = download.cancel_id || null
    const acknowledgedCancelled = download.status === 'cancelled'
      || (cancelId !== null && cancelStatuses[cancelId] === 'cancelled')
    if (acknowledgedCancelled) {
      if (now === null) return false
      const completedAt = timestampMs(download.completed_at)
        ?? (cancelId ? cancelledAt[cancelId] ?? null : null)
      return completedAt !== null && now - completedAt < COMPLETED_VISIBLE_MS
    }
    if (download.status !== 'completed') return true
    if (now === null) return false
    const completedAt = timestampMs(download.completed_at)
    return completedAt !== null && now - completedAt < COMPLETED_VISIBLE_MS
  }), [cancelStatuses, cancelledAt, downloads, now])

  const hasTerminal = downloads.some(download => download.status === 'completed' || download.status === 'cancelled')
  useEffect(() => {
    if (!hasTerminal) return
    // Initialize outside the effect body so react-hooks/set-state-in-effect
    // remains satisfied while the first render safely hides historical rows.
    const timer = window.setTimeout(() => setNow(Date.now()), 0)
    return () => window.clearTimeout(timer)
  }, [downloads, hasTerminal])

  const hasVisibleTerminal = visible.some(download => download.status === 'completed'
    || download.status === 'cancelled'
    || (download.cancel_id ? cancelStatuses[download.cancel_id] === 'cancelled' : false))
  useEffect(() => {
    if (!hasVisibleTerminal) return
    const timer = window.setInterval(() => setNow(Date.now()), 1000)
    return () => window.clearInterval(timer)
  }, [hasVisibleTerminal])

  const handleCancel = async (download: CivitAIDownload) => {
    const cancelId = download.cancel_id
    if (download.cancellable !== true || !cancelId) return

    setCancelErrors(prev => { const next = { ...prev }; delete next[cancelId]; return next })
    setCancelStatuses(prev => ({ ...prev, [cancelId]: 'pending' }))
    try {
      const result = await cancelDownload(cancelId)
      setCancelStatuses(prev => ({ ...prev, [cancelId]: result.status }))
      if (result.status === 'cancelled') {
        const acknowledgedAt = Date.now()
        setCancelledAt(prev => ({ ...prev, [cancelId]: acknowledgedAt }))
        setNow(acknowledgedAt)
      }
    } catch (error) {
      setCancelStatuses(prev => { const next = { ...prev }; delete next[cancelId]; return next })
      setCancelErrors(prev => ({
        ...prev,
        [cancelId]: error instanceof Error ? error.message : String(error),
      }))
    }
  }

  if (visible.length === 0) return null

  return (
    <div className="border-t border-border bg-bg-tertiary px-4 py-2 space-y-1.5 shrink-0">
      {visible.map(download => {
        const progress = displayProgress(download)
        const cancelId = download.cancel_id || null
        const cancelStatus = cancelId ? cancelStatuses[cancelId] : undefined
        const cancelError = cancelId ? cancelErrors[cancelId] : undefined
        const isCancelled = download.status === 'cancelled' || cancelStatus === 'cancelled'
        const isCancelling = !isCancelled && (download.status === 'cancelling'
          || cancelStatus === 'pending' || cancelStatus === 'cancelling')
        const isCompleted = download.status === 'completed' || cancelStatus === 'completed'
        const isFailed = download.status === 'failed' || cancelStatus === 'failed'
        const canCancel = download.cancellable === true && !!cancelId
          && download.status === 'downloading' && !isCancelling && !isCancelled
          && !isCompleted && !isFailed
        return (
          <div key={download.id} className="flex flex-wrap items-center gap-2">
            {isCancelled ? (
              <span aria-hidden="true" className="text-text-muted">×</span>
            ) : isCancelling ? (
              <Loader2 size={12} className="animate-spin text-text-secondary shrink-0" />
            ) : isCompleted ? (
              <Check size={12} className="text-accent-green shrink-0" />
            ) : isFailed ? (
              <AlertCircle size={12} className="text-red-400 shrink-0" />
            ) : (
              <Loader2 size={12} className="animate-spin text-accent-blue shrink-0" />
            )}
            <span className="text-[11px] text-text-secondary truncate flex-1">{download.filename}</span>
            {isCancelled ? (
              <span className="text-[10px] text-text-muted shrink-0" role="status">Cancelled</span>
            ) : isCancelling ? (
              <span className="text-[10px] text-text-secondary shrink-0" role="status" aria-live="polite">Cancelling</span>
            ) : isCompleted ? (
              <span className="text-[10px] text-text-muted shrink-0" role="status">Complete</span>
            ) : isFailed ? (
              <span className="text-[10px] text-red-400 truncate max-w-48" role="status">{download.error || 'Download failed'}</span>
            ) : (
              <>
                <div className="w-24 bg-bg-active rounded-full h-1 overflow-hidden shrink-0">
                  <div className="h-full bg-accent-blue rounded-full transition-all" style={{ width: `${progress}%` }} />
                </div>
                <span className="text-[10px] text-text-muted w-8 text-right shrink-0">{Math.round(progress)}%</span>
                {canCancel ? (
                  <button
                    type="button"
                    onClick={() => { void handleCancel(download) }}
                    className="min-h-[40px] min-w-[72px] px-3 py-2 text-[11px] leading-tight border border-border rounded text-text-secondary hover:text-red-300 hover:border-red-400/60 transition-colors shrink-0"
                    title={cancelError || `Cancel download of ${download.filename}`}
                    aria-label={`Cancel download of ${download.filename}`}
                  >
                    Cancel
                  </button>
                ) : (
                  <span className="text-[10px] text-text-muted shrink-0" role="status" aria-label={`Downloading ${download.filename}`}>
                    Downloading
                  </span>
                )}
              </>
            )}
            {cancelError && !isCancelled && !isCancelling && (
              <span className="w-full text-[10px] text-red-400 truncate" role="alert" title={cancelError}>
                Cancel failed: {cancelError}
              </span>
            )}
          </div>
        )
      })}
    </div>
  )
}
