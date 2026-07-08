import { Check, Loader2, AlertCircle } from 'lucide-react'
import { useStore } from '../../stores/useStore'

export function DownloadBar() {
  const downloads = useStore(s => s.civitDownloads)
  const active = downloads.filter(d => d.status !== 'completed' || Date.now() < 30000) // show completed briefly

  if (active.length === 0) return null

  return (
    <div className="border-t border-border bg-bg-tertiary px-4 py-2 space-y-1.5 shrink-0">
      {active.map(dl => (
        <div key={dl.id} className="flex items-center gap-2">
          {dl.status === 'downloading' ? (
            <Loader2 size={12} className="animate-spin text-accent-blue shrink-0" />
          ) : dl.status === 'completed' ? (
            <Check size={12} className="text-accent-green shrink-0" />
          ) : (
            <AlertCircle size={12} className="text-red-400 shrink-0" />
          )}
          <span className="text-[11px] text-text-secondary truncate flex-1">{dl.filename}</span>
          {dl.status === 'downloading' && (
            <>
              <div className="w-24 bg-bg-active rounded-full h-1 overflow-hidden shrink-0">
                <div className="h-full bg-accent-blue rounded-full transition-all" style={{ width: `${dl.progress}%` }} />
              </div>
              <span className="text-[10px] text-text-muted w-8 text-right shrink-0">{dl.progress}%</span>
            </>
          )}
          {dl.status === 'failed' && (
            <span className="text-[10px] text-red-400 truncate">{dl.error}</span>
          )}
        </div>
      ))}
    </div>
  )
}
