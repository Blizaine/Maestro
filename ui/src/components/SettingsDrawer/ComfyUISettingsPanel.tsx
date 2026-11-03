import { useEffect, useMemo, useState } from 'react'
import { AlertTriangle, Check, ChevronDown, ChevronRight, FolderOpen, Loader2, Plus, RefreshCw, Save } from 'lucide-react'
import * as api from '../../api/comfyui'
import { cancelJob } from '../../api/client'
import { useStore } from '../../stores/useStore'
import { ComfyUIWorkflowButton } from '../Sidebar/ComfyUIWorkflowDialog'

type RootOption = { path: string; category: string; installation?: string }

function compatibilityLabel(value: string | boolean): string {
  if (value === true) return 'Compatible'
  if (value === false) return 'Incompatible'
  return value ? value.replaceAll('_', ' ') : 'Unknown'
}

function modelSize(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes <= 0) return 'Size unknown'
  const units = ['B', 'KB', 'MB', 'GB', 'TB']
  let value = bytes
  let unit = 0
  while (value >= 1024 && unit < units.length - 1) { value /= 1024; unit += 1 }
  return `${value >= 10 || unit === 0 ? value.toFixed(0) : value.toFixed(1)} ${units[unit]}`
}

export function ComfyUISettingsPanel() {
  const [open, setOpen] = useState(false)
  const reconnectJobs = useStore(state => state.reconnectJobs)
  const [status, setStatus] = useState<api.ComfyUIStatus | null>(null)
  const [enabled, setEnabled] = useState(false)
  const [dedicated, setDedicated] = useState(false)
  const [endpoint, setEndpoint] = useState('http://127.0.0.1:8188')
  const [loading, setLoading] = useState(false)
  const [saving, setSaving] = useState(false)
  const [testing, setTesting] = useState(false)
  const [discovering, setDiscovering] = useState(false)
  const [scanning, setScanning] = useState(false)
  const [linking, setLinking] = useState(false)
  const [receiptAction, setReceiptAction] = useState<string | null>(null)
  const [connection, setConnection] = useState<api.ComfyUIConnectionTest | null>(null)
  const [installations, setInstallations] = useState<api.ComfyUIInstallation[]>([])
  const [roots, setRoots] = useState<RootOption[]>([])
  const [selectedPaths, setSelectedPaths] = useState<Set<string>>(new Set())
  const [manualRoot, setManualRoot] = useState('')
  const [scan, setScan] = useState<api.ComfyUIModelScan | null>(null)
  const [linkedPaths, setLinkedPaths] = useState<string[]>([])
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!open || status) return
    let cancelled = false
    setLoading(true)
    setError(null)
    api.fetchComfyUIStatus().then(result => {
      if (cancelled) return
      setStatus(result)
      setEnabled(result.enabled === true)
      setDedicated(result.dedicated === true)
      setEndpoint(result.endpoint || 'http://127.0.0.1:8188')
      setInstallations(result.installations ?? [])
    }).catch(reason => {
      if (!cancelled) setError(reason instanceof Error ? reason.message : 'Could not load ComfyUI settings.')
    }).finally(() => {
      if (!cancelled) setLoading(false)
    })
    return () => { cancelled = true }
  }, [open, status])

  const selectedRoots = useMemo(() => roots.filter(root => selectedPaths.has(root.path)), [roots, selectedPaths])
  const settingsDirty = !status || status.enabled !== enabled || status.endpoint !== endpoint.trim() || (status.dedicated === true) !== dedicated

  const saveSettings = async () => {
    setSaving(true); setError(null)
    try {
      const result = await api.saveComfyUISettings({ enabled, endpoint: endpoint.trim(), dedicated })
      setStatus(result)
      setEnabled(result.enabled === true)
      setDedicated(result.dedicated === true)
      setEndpoint(result.endpoint || endpoint.trim())
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'Could not save ComfyUI settings.')
    } finally { setSaving(false) }
  }

  const testConnection = async () => {
    setTesting(true); setError(null); setConnection(null)
    try {
      const result = await api.testComfyUIConnection()
      setConnection(result)
      setStatus(current => current ? { ...current, connected: result.connected, connection_error: result.connection_error ?? result.error ?? null } : current)
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'Could not test the ComfyUI connection.')
    } finally { setTesting(false) }
  }

  const discover = async () => {
    setDiscovering(true); setError(null); setScan(null)
    try {
      const result = await api.discoverComfyUIModels()
      setInstallations(result.installations ?? [])
      const discoveredRoots = (result.installations ?? []).flatMap(installation =>
        (installation.model_paths ?? []).map(item => ({
          path: item.path,
          category: item.category,
          installation: installation.name,
        })),
      )
      setRoots(current => {
        const byPath = new Map<string, RootOption>()
        for (const item of [...discoveredRoots, ...current.filter(root => !root.installation)]) {
          if (item.path?.trim()) byPath.set(item.path, item)
        }
        return [...byPath.values()]
      })
      setSelectedPaths(current => {
        const next = new Set(current)
        for (const root of discoveredRoots) if (root.path?.trim()) next.add(root.path)
        return next
      })
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'Could not discover ComfyUI model folders.')
    } finally { setDiscovering(false) }
  }

  const addManualRoot = () => {
    const path = manualRoot.trim()
    if (!path) return
    setRoots(current => current.some(root => root.path === path)
      ? current
      : [...current, { path, category: 'manual root' }])
    setSelectedPaths(current => new Set(current).add(path))
    setManualRoot('')
    setScan(null)
  }

  const togglePath = (path: string) => {
    setSelectedPaths(current => {
      const next = new Set(current)
      if (next.has(path)) next.delete(path)
      else next.add(path)
      return next
    })
  }

  const scanRoots = async () => {
    if (!selectedRoots.length) return
    setScanning(true); setError(null)
    try { setScan(await api.scanComfyUIModels(selectedRoots.map(root => root.path))) }
    catch (reason) { setError(reason instanceof Error ? reason.message : 'Could not scan the selected model folders.') }
    finally { setScanning(false) }
  }

  const linkRoots = async () => {
    if (!selectedRoots.length) return
    setLinking(true); setError(null)
    try {
      const result = await api.linkComfyUIModels(selectedRoots.map(root => root.path))
      setLinkedPaths(result.linked_paths ?? [])
    } catch (reason) { setError(reason instanceof Error ? reason.message : 'Could not link these model folders.') }
    finally { setLinking(false) }
  }

  const refreshStatus = async () => {
    const result = await api.fetchComfyUIStatus()
    setStatus(result)
    setEnabled(result.enabled === true)
    setDedicated(result.dedicated === true)
    setEndpoint(result.endpoint || 'http://127.0.0.1:8188')
    setInstallations(result.installations ?? [])
  }

  const resumeSubmission = async (submissionId: string) => {
    setReceiptAction(submissionId); setError(null)
    try {
      await api.resumeComfyUIJobBySubmission(submissionId)
      await reconnectJobs().catch(() => undefined)
      await refreshStatus()
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'Could not resume ComfyUI job monitoring.')
    } finally { setReceiptAction(null) }
  }

  const dismissPrepared = async (submission: api.ComfyUIPendingSubmission) => {
    setReceiptAction(submission.submission_id); setError(null)
    try {
      if (submission.job_id) await cancelJob(submission.job_id)
      else await api.abandonComfyUISubmission(submission.submission_id)
      await reconnectJobs().catch(() => undefined)
      await refreshStatus()
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'Could not dismiss this unsubmitted workflow intent.')
    } finally { setReceiptAction(null) }
  }

  return (
    <section className="rounded-xl border border-border bg-bg-secondary">
      <button
        type="button"
        aria-expanded={open}
        aria-controls="comfyui-settings-content"
        onClick={() => setOpen(value => !value)}
        className="flex w-full items-center gap-2 px-3 py-2.5 text-left"
      >
        {open ? <ChevronDown size={13} /> : <ChevronRight size={13} />}
        <span className="min-w-0 flex-1">
          <span className="block text-xs font-medium text-text-primary">ComfyUI integration</span>
          <span className="mt-0.5 block text-[10px] text-text-muted">Optional workflows and read-only model folders</span>
        </span>
        <span className="rounded-full border border-border px-2 py-0.5 text-[9px] text-text-muted">{enabled ? 'Enabled' : 'Off'}</span>
      </button>

      {open && <div id="comfyui-settings-content" className="space-y-3 border-t border-border px-3 py-3">
        {loading && <p role="status" className="flex items-center gap-1.5 text-[10px] text-text-muted"><Loader2 size={12} className="animate-spin" /> Loading settings…</p>}
        {error && <div role="alert" className="rounded-lg border border-red-500/30 bg-red-500/10 px-2.5 py-2 text-[10px] text-chip-red">{error}</div>}

        <label className="flex cursor-pointer items-start gap-2 text-[11px] text-text-primary">
          <input type="checkbox" checked={enabled} onChange={event => setEnabled(event.target.checked)} className="mt-0.5 accent-[var(--color-accent-blue)]" />
          <span><span className="font-medium">Enable ComfyUI integration</span><span className="mt-0.5 block text-[10px] text-text-muted">Off by default. Maestro's native engine remains the default generation path.</span></span>
        </label>

        <label className="flex cursor-pointer items-start gap-2 text-[11px] text-text-primary">
          <input type="checkbox" checked={dedicated} onChange={event => setDedicated(event.target.checked)} className="mt-0.5 accent-[var(--color-accent-blue)]" />
          <span><span className="font-medium">Dedicated ComfyUI server</span><span className="mt-0.5 block text-[10px] text-text-muted">Use only for a ComfyUI instance reserved for Maestro. When switching to Native, Maestro can release ComfyUI's cached models after its queue is idle. Shared servers are never interrupted.</span></span>
        </label>

        <div>
          <label htmlFor="comfyui-endpoint" className="mb-1 block text-[10px] font-medium text-text-secondary">ComfyUI endpoint</label>
          <input id="comfyui-endpoint" type="url" value={endpoint} onChange={event => setEndpoint(event.target.value)} placeholder="http://127.0.0.1:8188" className="w-full rounded-lg border border-border bg-bg-primary px-2.5 py-2 text-xs text-text-primary outline-none placeholder:text-text-muted focus:border-accent-blue" />
          <p className="mt-1 text-[9px] leading-relaxed text-text-muted">Use an existing local or LAN ComfyUI server. Maestro will not install custom nodes.</p>
        </div>

        <div className="flex flex-wrap gap-2">
          <button type="button" onClick={() => void saveSettings()} disabled={saving || loading || !endpoint.trim()} className="inline-flex items-center gap-1.5 rounded-lg bg-accent-blue px-2.5 py-2 text-[10px] font-medium text-white hover:brightness-110 disabled:opacity-50">
            {saving ? <Loader2 size={12} className="animate-spin" /> : <Save size={12} />} Save settings
          </button>
          <button type="button" onClick={() => void testConnection()} disabled={testing || loading || settingsDirty} className="inline-flex items-center gap-1.5 rounded-lg border border-border bg-bg-primary px-2.5 py-2 text-[10px] text-text-secondary hover:text-text-primary disabled:opacity-50">
            {testing ? <Loader2 size={12} className="animate-spin" /> : <RefreshCw size={12} />} Test connection
          </button>
          <ComfyUIWorkflowButton label="Open workflow library" className="rounded-lg border border-border bg-bg-primary px-2.5 py-2 text-[10px] text-text-secondary hover:text-text-primary" />
        </div>
        {settingsDirty && <p className="-mt-1 text-[9px] text-text-muted">Save these settings before testing the connection.</p>}

        {connection && <div role="status" className={`rounded-lg border px-2.5 py-2 text-[10px] ${connection.connected ? 'border-emerald-500/30 bg-emerald-500/10 text-text-primary' : 'border-amber-500/30 bg-amber-500/10 text-text-primary'}`}>
          <div className="flex items-center gap-1.5 font-medium">{connection.connected ? <Check size={12} className="text-indicator-success" /> : <AlertTriangle size={12} className="text-indicator-warning" />}{connection.connected ? 'ComfyUI is reachable' : 'Connection failed'}</div>
          {(connection.error || connection.connection_error) && <p className="mt-1 break-words text-text-secondary">{connection.error || connection.connection_error}</p>}
          {connection.missing?.length ? <p className="mt-1 text-text-secondary">Missing custom nodes: {connection.missing.join(', ')}</p> : null}
        </div>}

        {(status?.pending_submissions?.length ?? 0) > 0 && <div className="space-y-2 border-t border-border pt-3">
          <div className="text-[10px] font-semibold text-text-primary">Pending ComfyUI submissions</div>
          <p className="text-[9px] text-text-muted">Resume monitoring reconnects to existing owned work and never submits another prompt.</p>
          <div className="max-h-40 space-y-1.5 overflow-y-auto">
            {status?.pending_submissions?.map(submission => {
              const state = submission.state.toLowerCase()
              const prepared = state === 'prepared'
              const resumable = ['dispatching', 'running', 'uncertain'].includes(state)
              return <div key={submission.submission_id} className="rounded-lg border border-border bg-bg-primary px-2.5 py-2">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="min-w-0 flex-1 break-all font-mono text-[9px] text-text-secondary">{submission.job_id || submission.submission_id}</span>
                  <span className="shrink-0 rounded border border-border px-1.5 py-0.5 text-[8px] text-text-muted">{submission.state}</span>
                </div>
                {submission.error && <p className="mt-1 break-words text-[9px] text-indicator-warning">{submission.error}</p>}
                {prepared
                  ? <button type="button" onClick={() => void dismissPrepared(submission)} disabled={receiptAction !== null} className="mt-1.5 rounded border border-border px-2 py-1 text-[9px] text-text-secondary hover:text-text-primary disabled:opacity-50">{receiptAction === submission.submission_id ? 'Dismissing…' : 'Dismiss unsubmitted intent'}</button>
                  : resumable
                    ? <button type="button" onClick={() => void resumeSubmission(submission.submission_id)} disabled={receiptAction !== null} className="mt-1.5 rounded border border-border px-2 py-1 text-[9px] text-text-secondary hover:text-text-primary disabled:opacity-50">{receiptAction === submission.submission_id ? 'Resuming…' : 'Resume monitoring'}</button>
                    : null}
              </div>
            })}
          </div>
        </div>}

        <div className="space-y-2 border-t border-border pt-3">
          <div className="flex items-center gap-2">
            <FolderOpen size={13} className="text-accent-blue" />
            <h3 className="flex-1 text-[11px] font-semibold text-text-primary">Model folders</h3>
            <button type="button" onClick={() => void discover()} disabled={discovering} className="inline-flex items-center gap-1 rounded border border-border px-2 py-1 text-[9px] text-text-secondary hover:text-text-primary disabled:opacity-50">
              {discovering ? <Loader2 size={10} className="animate-spin" /> : <RefreshCw size={10} />} Discover
            </button>
          </div>
          <p className="text-[9px] leading-relaxed text-text-muted">Select existing folders to scan or link them as read-only references. No model weights are moved or deleted.</p>

          {installations.map(installation => <div key={installation.id} className="rounded-lg border border-border bg-bg-primary px-2.5 py-2">
            <div className="flex items-start justify-between gap-2 text-[10px]">
              <span className="font-medium text-text-primary">{installation.name}</span>
              <span className={installation.available ? 'text-indicator-success' : 'text-text-muted'}>{installation.available ? 'Available' : 'Unavailable'}</span>
            </div>
            <p className="mt-0.5 break-all font-mono text-[9px] text-text-muted">{installation.path}</p>
          </div>)}

          <div className="flex gap-1.5">
            <input value={manualRoot} onChange={event => setManualRoot(event.target.value)} onKeyDown={event => { if (event.key === 'Enter') { event.preventDefault(); addManualRoot() } }} aria-label="Manual ComfyUI model root" placeholder="Add a model root path" className="min-w-0 flex-1 rounded-md border border-border bg-bg-primary px-2 py-1.5 text-[10px] text-text-primary outline-none placeholder:text-text-muted focus:border-accent-blue" />
            <button type="button" onClick={addManualRoot} disabled={!manualRoot.trim()} className="inline-flex items-center gap-1 rounded-md border border-border px-2 py-1.5 text-[9px] text-text-secondary hover:text-text-primary disabled:opacity-40"><Plus size={10} /> Add root</button>
          </div>

          {roots.length > 0 && <div className="max-h-40 space-y-1 overflow-y-auto rounded-lg border border-border bg-bg-primary p-2">
            {roots.map(root => <label key={root.path} className="flex cursor-pointer items-start gap-1.5 text-[9px] text-text-secondary">
              <input type="checkbox" checked={selectedPaths.has(root.path)} onChange={() => togglePath(root.path)} className="mt-0.5" />
              <span className="min-w-0 flex-1"><span className="font-medium text-text-primary">{root.category}</span>{root.installation && <span className="text-text-muted"> · {root.installation}</span>}<span className="block break-all font-mono text-text-muted">{root.path}</span></span>
            </label>)}
          </div>}

          <div className="flex flex-wrap gap-2">
            <button type="button" onClick={() => void scanRoots()} disabled={!selectedRoots.length || scanning} className="rounded-lg border border-border bg-bg-primary px-2.5 py-2 text-[9px] text-text-secondary hover:text-text-primary disabled:opacity-40">{scanning ? 'Scanning…' : `Scan selected (${selectedRoots.length})`}</button>
            <button type="button" onClick={() => void linkRoots()} disabled={!selectedRoots.length || linking} className="rounded-lg border border-border bg-bg-primary px-2.5 py-2 text-[9px] text-text-secondary hover:text-text-primary disabled:opacity-40">{linking ? 'Linking…' : 'Link selected read-only'}</button>
          </div>

          {linkedPaths.length > 0 && <div role="status" className="rounded-lg border border-emerald-500/30 bg-emerald-500/10 px-2.5 py-2 text-[9px] text-text-primary">
            <div className="mb-1 font-medium">Linked read-only folders</div>
            {linkedPaths.map(path => <div key={path} className="break-all font-mono text-text-secondary">{path}</div>)}
          </div>}

          {scan && <div className="rounded-lg border border-border bg-bg-primary p-2.5">
            <div className="mb-1.5 flex items-center justify-between gap-2 text-[9px]">
              <span className="font-medium text-text-primary">Found {scan.files.length} model file{scan.files.length === 1 ? '' : 's'}</span>
              {scan.truncated && <span className="text-indicator-warning">Results truncated</span>}
            </div>
            <div className="max-h-56 space-y-1.5 overflow-y-auto">
              {scan.files.map(file => <div key={file.id} className="border-t border-border/70 pt-1.5 text-[9px]">
                <div className="flex items-start gap-2"><span className="min-w-0 flex-1 break-all text-text-primary">{file.filename}</span><span className="shrink-0 text-text-muted">{modelSize(file.size)}</span></div>
                <div className="mt-0.5 text-text-muted">{file.category} · {compatibilityLabel(file.compatibility)}</div>
                {file.reason && <p className="mt-0.5 break-words text-indicator-warning">{file.reason}</p>}
                {file.relative_path && <p className="mt-0.5 break-all font-mono text-text-muted">{file.relative_path}</p>}
              </div>)}
              {scan.files.length === 0 && <p className="text-[9px] text-text-muted">No model files were found in the selected folders.</p>}
            </div>
          </div>}
        </div>
      </div>}
    </section>
  )
}
