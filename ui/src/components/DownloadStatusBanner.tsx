import { useEffect, useState } from 'react'
import { Download, AlertTriangle } from 'lucide-react'
import { fetchActiveDownloads, type ActiveDownload } from '../api/client'

/**
 * DownloadStatusBanner — fixed-position overlay shown while
 * model files are being downloaded from HuggingFace (or other CDNs).
 *
 * Polls /api/v1/downloads/active every 2s and renders nothing when
 * the list is empty. When downloads are active, shows a compact
 * banner with the current file's progress + a "stalled / retrying"
 * badge if the byte counter hasn't advanced in >15s.
 *
 * Polling is unconditional (vs gated on "is a job running") because
 * model downloads can fire from several paths in Maestro: job
 * submission, model selection, etc. Polling
 * is cheap (a 2s GET every 2s) and only ever returns data when the
 * banner needs to be visible.
 *
 * Pairs with services/safe_download.py — that module patches HF
 * downloads to detect mid-stream stalls and recover automatically.
 * This banner just makes the recovery visible to the user instead
 * of the previous "frozen progress bar in console" UX.
 */
export function DownloadStatusBanner() {
  const [downloads, setDownloads] = useState<ActiveDownload[]>([])

  useEffect(() => {
    let cancelled = false

    const tick = async () => {
      try {
        const result = await fetchActiveDownloads()
        if (!cancelled) setDownloads(result.downloads)
      } catch {
        // Endpoint not available (older backend) or transient — ignore
        if (!cancelled) setDownloads([])
      }
    }

    tick()
    const interval = setInterval(tick, 2000)
    return () => {
      cancelled = true
      clearInterval(interval)
    }
  }, [])

  if (downloads.length === 0) return null

  // Pick the highest-priority download to feature: any "stalled"
  // first, then the one with the most progress (closest to done).
  // Show a count when there are multiple.
  //
  // Threshold for "stalled" is 30s — covers genuinely-broken
  // connections without false-positiving on the brief lulls that
  // show up routinely in healthy CDN downloads (especially during
  // chunk boundary handoffs).
  // An interrupted download (byte count fell short of the expected size)
  // takes top priority — the file is likely truncated and the user should
  // know rather than hit a mysterious load failure next generation.
  const incomplete = downloads.find(d => d.status === 'incomplete')
  const stalled = downloads.find(d => d.seconds_since_progress > 30 && d.status !== 'incomplete')
  const featured = incomplete ?? stalled ?? downloads.reduce((best, cur) => {
    const bestPct = best.total_bytes ? best.downloaded_bytes / best.total_bytes : 0
    const curPct = cur.total_bytes ? cur.downloaded_bytes / cur.total_bytes : 0
    return curPct > bestPct ? cur : best
  }, downloads[0])

  return (
    <div className="fixed bottom-4 right-4 z-40 max-w-md w-[calc(100vw-2rem)] sm:w-auto">
      {/* Outer container is always solid bg-bg-secondary so text
          stays readable over images/videos in the main feed.
          Border color switches to amber on stall to draw the eye
          without sacrificing contrast. */}
      <div className={`bg-bg-secondary rounded-lg border shadow-2xl overflow-hidden ${
        incomplete ? 'border-red-500/60' : stalled ? 'border-amber-500/60' : 'border-border'
      }`}>
        {/* Interrupted download — red strip. The file is probably truncated;
            re-running the download/generation fetches the rest. */}
        {incomplete && (
          <div className="px-4 py-2 bg-red-500/15 border-b border-red-500/30 flex items-center gap-2">
            <AlertTriangle size={14} className="text-red-400 shrink-0" />
            <div className="text-xs font-medium text-red-100">
              A download was interrupted — re-run to finish it
            </div>
          </div>
        )}
        {/* Optional amber accent strip on stall — semi-transparent
            tint over the solid backdrop, same pattern as OomRecoveryBanner. */}
        {stalled && !incomplete && (
          <div className="px-4 py-2 bg-amber-500/15 border-b border-amber-500/30 flex items-center gap-2">
            <AlertTriangle size={14} className="text-amber-400 shrink-0" />
            <div className="text-xs font-medium text-amber-100">
              Download is slow — waiting for retry
            </div>
          </div>
        )}

        <div className="px-4 py-3">
          <div className="flex items-start gap-2.5">
            {!stalled && (
              <Download size={16} className="text-accent-blue shrink-0 mt-0.5 animate-pulse" />
            )}
            <div className="flex-1 min-w-0">
              {!stalled && (
                <div className="flex items-center justify-between gap-2">
                  <div className="text-xs font-medium text-text-primary truncate">
                    Downloading model files
                  </div>
                  {downloads.length > 1 && (
                    <div className="text-[10px] text-text-muted shrink-0">
                      {downloads.length} files
                    </div>
                  )}
                </div>
              )}
              {stalled && downloads.length > 1 && (
                <div className="text-[10px] text-text-muted text-right -mt-0.5 mb-0.5">
                  {downloads.length} files
                </div>
              )}
              <div className="text-[10px] text-text-muted truncate" title={featured.filename}>
                {featured.filename}
              </div>
              <DownloadProgressBar download={featured} stalled={!!stalled} />
              {stalled && (
                <div className="text-[11px] text-text-secondary mt-1.5 leading-snug">
                  No progress for {Math.round(featured.seconds_since_progress)}s.
                  The download will resume from where it left off as soon as
                  the connection recovers — no action needed from you.
                </div>
              )}
            </div>
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
            stalled ? 'bg-amber-400/60' : 'bg-accent-blue'
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
            stalled ? 'text-amber-300' : 'text-text-secondary'
          }`}>
            {pct}%
          </span>
        )}
      </div>
    </div>
  )
}
