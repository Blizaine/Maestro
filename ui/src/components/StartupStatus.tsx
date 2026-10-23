import { Loader2, RefreshCw } from 'lucide-react'
import { useStore } from '../stores/useStore'
import { RETRY_STARTUP_EVENT } from '../lib/appStartup'

export function StartupStatus() {
  const loaded = useStore(state => state.modelsLoaded)
  const loading = useStore(state => state.modelsLoading)
  const error = useStore(state => state.modelsLoadError)
  if (loaded && !loading && !error) return null
  return <div role={error ? 'alert' : 'status'} className="mx-3 my-2 flex shrink-0 items-center gap-2 rounded-lg border border-border px-3 py-2 text-xs text-text-secondary">
    {loading ? <Loader2 size={14} className="shrink-0 animate-spin"/> : <RefreshCw size={14} className="shrink-0"/>}
    <span className="min-w-0 flex-1">{error ? `Could not load controls: ${error}` : 'Loading model controls…'}</span>
    {error && <button type="button" className="shrink-0 text-accent-blue hover:underline"
      onClick={() => window.dispatchEvent(new Event(RETRY_STARTUP_EVENT))}>Retry</button>}
  </div>
}
