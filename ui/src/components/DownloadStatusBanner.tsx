import { useEffect, useState } from 'react'
import { AlertTriangle, Download, Loader2 } from 'lucide-react'
import { cancelDownload, fetchActiveDownloads, type ActiveDownload } from '../api/client'
import { useLongPoll } from '../lib/useLongPoll'

type CancelStatus = 'pending' | 'cancelling' | 'cancelled' | 'completed' | 'failed'
const STALLED_AFTER_SECONDS = 30
const NO_DOWNLOADS: { downloads: ActiveDownload[]; receivedAt: number } = { downloads: [], receivedAt: 0 }

/**
 * DownloadStatusBanner — fixed-position list of active model transfers.
 *
 * Long-polls /api/v1/downloads/active. Cancellation is offered only when the
 * server marks one row cancellable and supplies its opaque cancel_id.
 */
export function DownloadStatusBanner() {
  const [snapshot, setSnapshot] = useState(NO_DOWNLOADS)
  const [cancelStatuses, setCancelStatuses] = useState<Record<string, CancelStatus>>({})
  const [cancelErrors, setCancelErrors] = useState<Record<string, string>>({})
  const [now, setNow] = useState(0)
  const { downloads, receivedAt } = snapshot

  useLongPoll(fetchActiveDownloads, result => {
    setSnapshot(prev => ({
      downloads: result.downloads,
      receivedAt: result.downloads.length === 0 && prev.downloads.length === 0
        ? prev.receivedAt
        : Date.now(),
    }))
    const currentIds = new Set(result.downloads.flatMap(download => download.cancel_id ? [download.cancel_id] : []))
    setCancelStatuses(prev => {
      const next: Record<string, CancelStatus> = {}
      for (const id of currentIds) {
        const download = result.downloads.find(item => item.cancel_id === id)
        if (download?.status === 'cancelling') next[id] = 'cancelling'
        else if (download?.status === 'cancelled') next[id] = 'cancelled'
        else if (download?.status === 'done') next[id] = 'completed'
        else if (download?.status === 'incomplete') next[id] = 'failed'
        else if (prev[id]) next[id] = prev[id]
      }
      return next
    })
    setCancelErrors(prev => Object.fromEntries(
      Object.entries(prev).filter(([id]) => currentIds.has(id)),
    ))
  })

  const handleCancel = async (download: ActiveDownload) => {
    const cancelId = download.cancel_id
    if (download.cancellable !== true || !cancelId) return

    setCancelErrors(prev => { const next = { ...prev }; delete next[cancelId]; return next })
    setCancelStatuses(prev => ({ ...prev, [cancelId]: 'pending' }))
    try {
      const result = await cancelDownload(cancelId)
      setCancelStatuses(prev => ({ ...prev, [cancelId]: result.status }))
    } catch (error) {
      setCancelStatuses(prev => { const next = { ...prev }; delete next[cancelId]; return next })
      setCancelErrors(prev => ({
        ...prev,
        [cancelId]: error instanceof Error ? error.message : String(error),
      }))
    }
  }

  const secondsSinceProgress = (download: ActiveDownload) =>
    download.seconds_since_progress + Math.max(0, now - receivedAt) / 1000
  const incomplete = downloads.some(download => download.status === 'incomplete'
    && (!download.cancel_id || cancelStatuses[download.cancel_id] !== 'cancelled'))
  const stalled = downloads.some(download => secondsSinceProgress(download) > STALLED_AFTER_SECONDS
    && !['incomplete', 'done', 'cancelled', 'cancelling'].includes(download.status)
    && (!download.cancel_id
      || !['pending', 'cancelling', 'cancelled', 'completed', 'failed'].includes(cancelStatuses[download.cancel_id] || '')))
  useEffect(() => {
    if (!stalled) return
    const id = setInterval(() => setNow(Date.now()), 1000)
    return () => clearInterval(id)
  }, [stalled])

  if (downloads.length === 0) return null

  return (
    <div className="fixed bottom-4 right-4 z-40 max-w-md w-[calc(100vw-2rem)] sm:w-auto">
      <div className={`bg-bg-secondary rounded-lg border shadow-2xl overflow-hidden ${
        incomplete ? 'border-red-500/60' : stalled ? 'border-amber-500/60' : 'border-border'
      }`}>
        {incomplete && (
          <div className="px-4 py-2 bg-red-500/15 border-b border-red-500/30 flex items-center gap-2">
            <AlertTriangle size={14} className="text-red-400 shrink-0" />
            <div className="text-xs font-medium text-text-primary">
              A download was interrupted — re-run to finish it
            </div>
          </div>
        )}
        {stalled && !incomplete && (
          <div className="px-4 py-2 bg-amber-500/15 border-b border-amber-500/30 flex items-center gap-2">
            <AlertTriangle size={14} className="text-indicator-warning shrink-0" />
            <div className="text-xs font-medium text-text-primary">
              A download is slow — waiting for retry
            </div>
          </div>
        )}

        <div className="px-4 py-3">
          <div className="flex items-center justify-between gap-3 mb-2">
            <div className="flex items-center gap-2 text-xs font-medium text-text-primary">
              <Download size={14} className="text-accent-blue shrink-0" />
              Downloads
            </div>
            <div className="text-[10px] text-text-muted shrink-0">
              {downloads.length} {downloads.length === 1 ? 'file' : 'files'}
            </div>
          </div>

          <div className="space-y-3 max-h-[50vh] overflow-y-auto pr-0.5">
            {downloads.map(download => {
              const cancelId = download.cancel_id || null
              const cancelStatus = cancelId ? cancelStatuses[cancelId] : undefined
              const cancelError = cancelId ? cancelErrors[cancelId] : undefined
              const isCancelled = download.status === 'cancelled' || cancelStatus === 'cancelled'
              const isCancelling = !isCancelled && (download.status === 'cancelling'
                || cancelStatus === 'pending' || cancelStatus === 'cancelling')
              const isCompleted = download.status === 'done' || cancelStatus === 'completed'
              const isFailed = download.status === 'incomplete' || cancelStatus === 'failed'
              const rowStalled = !isCancelled && !isCancelling && !isCompleted && !isFailed
                && secondsSinceProgress(download) > STALLED_AFTER_SECONDS
              const canCancel = download.cancellable === true && !!cancelId
                && !isCancelled && !isCancelling && !isCompleted && !isFailed

              return (
                <div key={download.file_id} className="border-t border-border/70 pt-2 first:border-t-0 first:pt-0">
                  <div className="flex items-start gap-2">
                    {isCancelled ? (
                      <span aria-hidden="true" className="mt-0.5 text-text-muted">×</span>
                    ) : isCancelling ? (
                      <Loader2 size={13} className="mt-0.5 animate-spin text-text-secondary shrink-0" />
                    ) : isFailed ? (
                      <AlertTriangle size={13} className="mt-0.5 text-indicator-warning shrink-0" />
                    ) : (
                      <Download size={13} className={`mt-0.5 shrink-0 ${rowStalled ? 'text-indicator-warning' : 'text-accent-blue'}`} />
                    )}
                    <div className="flex-1 min-w-0">
                      <div className="flex items-start justify-between gap-2">
                        <div className="text-[10px] text-text-secondary truncate" title={download.filename}>
                          {download.filename}
                        </div>
                        {isCancelled ? (
                          <span className="text-[10px] text-text-muted shrink-0" role="status">Cancelled</span>
                        ) : isCancelling ? (
                          <span className="text-[10px] text-text-secondary shrink-0" role="status" aria-live="polite">Cancelling</span>
                        ) : isCompleted ? (
                          <span className="text-[10px] text-text-muted shrink-0" role="status">Complete</span>
                        ) : canCancel ? (
                          <button
                            type="button"
                            onClick={() => { void handleCancel(download) }}
                            className="min-h-[40px] min-w-[72px] px-3 py-2 text-[11px] leading-tight border border-border rounded text-text-secondary hover:text-red-300 hover:border-red-400/60 transition-colors shrink-0"
                            title={cancelError || `Cancel download of ${download.filename}`}
                            aria-label={`Cancel download of ${download.filename}`}
                          >
                            Cancel
                          </button>
                        ) : null}
                      </div>
                      {cancelError && !isCancelled && !isCancelling && (
                        <div className="text-[10px] text-red-400 mt-1 truncate" role="alert" title={cancelError}>
                          Cancel failed: {cancelError}
                        </div>
                      )}
                      {!isCancelled && !isCancelling && !isCompleted && (
                        <DownloadProgressBar download={download} stalled={rowStalled} />
                      )}
                      {isFailed && !isCancelled && (
                        <div className="text-[10px] text-text-muted mt-1" role="status">Download interrupted</div>
                      )}
                      {rowStalled && (
                        <div className="text-[10px] text-text-secondary mt-1 leading-snug">
                          No progress for {Math.round(secondsSinceProgress(download))}s. The download will retry automatically.
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      </div>
    </div>
  )
}

function _formatBytes(n: number): string {
  if (n < 1024) return `${n} B`
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`
  if (n < 1024 * 1024 * 1024) return `${(n / (1024 * 1024)).toFixed(1)} MB`
  return `${(n / (1024 * 1024 * 1024)).toFixed(2)} GB`
}

function DownloadProgressBar({
  download,
  stalled,
}: {
  download: ActiveDownload
  stalled: boolean
}) {
  const pct = download.total_bytes
    ? Math.round((download.downloaded_bytes / download.total_bytes) * 100)
    : null

  return (
    <div className="mt-1.5">
      <div className="h-1 rounded-full bg-bg-tertiary overflow-hidden">
        <div
          className={`h-full transition-all duration-500 ${
            stalled ? 'bg-indicator-warning' : 'bg-accent-blue'
          }`}
          style={{ width: pct !== null ? `${pct}%` : '15%' }}
        />
      </div>
      <div className="flex items-center justify-between mt-1">
        <span className="text-[10px] text-text-muted">
          {_formatBytes(download.downloaded_bytes)}
          {download.total_bytes !== null && (
            <> / {_formatBytes(download.total_bytes)}</>
          )}
        </span>
        {pct !== null && (
          <span className={`text-[10px] tabular-nums ${
            stalled ? 'text-indicator-warning' : 'text-text-secondary'
          }`}>
            {pct}%
          </span>
        )}
      </div>
    </div>
  )
}
