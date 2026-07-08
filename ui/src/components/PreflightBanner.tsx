import { useEffect, useState } from 'react'
import { AlertTriangle, X } from 'lucide-react'
import { fetchPreflight, type PreflightCheck } from '../api/client'

/**
 * PreflightBanner — one-time environment sanity check shown at the top
 * of the app on first load. Surfaces the three failures that otherwise
 * only appear as a cryptic traceback mid-generation: ffmpeg missing, no
 * CUDA GPU, and low disk on the output drive.
 *
 * Renders nothing when everything is fine. Dismissible; stays dismissed
 * for the session (sessionStorage) so it doesn't nag after the user has
 * acknowledged it, but returns next launch if the problem persists.
 */
export function PreflightBanner() {
  const [checks, setChecks] = useState<PreflightCheck[]>([])
  const [dismissed, setDismissed] = useState(
    () => sessionStorage.getItem('maestro_preflight_dismissed') === '1'
  )

  useEffect(() => {
    let cancelled = false
    fetchPreflight()
      .then(r => { if (!cancelled) setChecks(r.checks || []) })
      .catch(() => { /* older backend / transient — say nothing */ })
    return () => { cancelled = true }
  }, [])

  if (dismissed || checks.length === 0) return null

  const hasError = checks.some(c => c.level === 'error')

  return (
    <div className={`fixed top-0 inset-x-0 z-50 px-4 py-2 flex items-start gap-2.5 border-b backdrop-blur-sm ${
      hasError
        ? 'bg-red-500/20 border-red-500/40'
        : 'bg-amber-500/20 border-amber-500/40'
    }`}>
      <AlertTriangle
        size={15}
        className={`shrink-0 mt-0.5 ${hasError ? 'text-red-400' : 'text-amber-400'}`}
      />
      <div className="flex-1 min-w-0 space-y-0.5">
        {checks.map(c => (
          <div key={c.id} className={`text-[11px] leading-snug ${
            c.level === 'error' ? 'text-red-100' : 'text-amber-100'
          }`}>
            {c.message}
          </div>
        ))}
      </div>
      <button
        onClick={() => {
          sessionStorage.setItem('maestro_preflight_dismissed', '1')
          setDismissed(true)
        }}
        className="shrink-0 p-0.5 rounded text-text-muted hover:text-text-primary transition-colors"
        aria-label="Dismiss"
      >
        <X size={13} />
      </button>
    </div>
  )
}
