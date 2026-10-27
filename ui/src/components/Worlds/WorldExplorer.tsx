import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { ArrowDown, ArrowLeft, ArrowRight, ArrowUp, Camera, Download, Film, Loader2, RefreshCw, X } from 'lucide-react'
import * as api from '../../api/worlds'
import type { WorldRecord, WorldRequest, WorldRequestFields, WorldRequestReceipt } from '../../api/worlds'

const PENDING_KEY = 'maestro.worlds.pending-request.v1'
const MOVE_ACTIONS = [
  { action: 'Forward', label: 'Forward', icon: ArrowUp },
  { action: 'Backward', label: 'Backward', icon: ArrowDown },
  { action: 'Left', label: 'Step left', icon: ArrowLeft },
  { action: 'Right', label: 'Step right', icon: ArrowRight },
  { action: 'Look_Left', label: 'Look left' },
  { action: 'Look_Right', label: 'Look right' },
  { action: 'Look_Up', label: 'Look up' },
  { action: 'Look_Down', label: 'Look down' },
  { action: 'Wait', label: 'Wait' },
] as const

type StoredPending = { request: WorldRequest; cancelAttempted?: boolean }
type OwnedRequest = {
  request: WorldRequest
  receipt: WorldRequestReceipt
  cancelAttempted: boolean
}

export interface WorldExplorerProps {
  open?: boolean
  onClose?: () => void
}

function readPending(): StoredPending | null {
  try {
    const value = JSON.parse(window.localStorage.getItem(PENDING_KEY) || 'null') as StoredPending | null
    if (!value?.request?.client_request_id || !value.request.request_digest) return null
    return value
  } catch {
    return null
  }
}

function savePending(value: StoredPending): void {
  window.localStorage.setItem(PENDING_KEY, JSON.stringify(value))
}

function clearPending(): void {
  window.localStorage.removeItem(PENDING_KEY)
}

function receiptIsFinal(receipt: WorldRequestReceipt): boolean {
  return ['published', 'failed', 'cancelled', 'verification_failed', 'ownership_conflict'].includes(receipt.state)
}

export function WorldExplorer({ open = true, onClose }: WorldExplorerProps) {
  const [worlds, setWorlds] = useState<WorldRecord[]>([])
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [owned, setOwned] = useState<OwnedRequest | null>(null)
  const [description, setDescription] = useState('')
  const [eventText, setEventText] = useState('')
  const [durationMs, setDurationMs] = useState(800)
  const [acknowledged, setAcknowledged] = useState(false)
  const [loadingWorlds, setLoadingWorlds] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [notice, setNotice] = useState<string | null>(null)
  const [fileBusy, setFileBusy] = useState(false)
  const [captureUrl, setCaptureUrl] = useState<string | null>(null)
  const startLock = useRef(false)
  const pollLock = useRef(false)
  const ownedRef = useRef<OwnedRequest | null>(null)
  const panelRef = useRef<HTMLElement | null>(null)
  const closeButtonRef = useRef<HTMLButtonElement | null>(null)
  const onCloseRef = useRef(onClose)
  ownedRef.current = owned
  onCloseRef.current = onClose

  const selected = useMemo(() => worlds.find(world => world.world_id === selectedId) || null, [worlds, selectedId])
  const inFlight = Boolean(owned && !receiptIsFinal(owned.receipt))
  const activeRequestId = owned?.request.client_request_id
  const shouldPollReceipt = Boolean(owned && !receiptIsFinal(owned.receipt))
  const generationDisabled = inFlight || !acknowledged

  const refreshWorlds = useCallback(async () => {
    setLoadingWorlds(true)
    try {
      const values = await api.listWorlds()
      setWorlds(values)
      setSelectedId(current => current && values.some(world => world.world_id === current)
        ? current
        : values[0]?.world_id || null)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Could not load saved worlds.')
    } finally {
      setLoadingWorlds(false)
    }
  }, [])

  const finishReceipt = useCallback((receipt: WorldRequestReceipt) => {
    setOwned(current => current ? { ...current, receipt } : current)
    if (receipt.state === 'published' && receipt.world) {
      setSelectedId(receipt.world.world_id)
      setWorlds(current => [receipt.world!, ...current.filter(world => world.world_id !== receipt.world!.world_id)])
      clearPending()
      setOwned(null)
      setNotice('World saved. You can continue exploring when the queued job is complete.')
      void refreshWorlds()
      return
    }
    if (['failed', 'cancelled', 'verification_failed'].includes(receipt.state)) {
      clearPending()
      setOwned(null)
      setError(receipt.detail || `World job ${receipt.state}.`)
      return
    }
    if (receipt.state === 'ownership_conflict') {
      setError(receipt.detail || 'The saved request could not be matched to its own generation job. It will not be repeated or cancelled.')
    }
  }, [refreshWorlds])

  useEffect(() => {
    if (!open) return
    setError(null)
    void refreshWorlds()
  }, [open, refreshWorlds])

  useEffect(() => {
    if (!open) return
    const pending = readPending()
    if (!pending) return
    setAcknowledged(true)
    const receipt: WorldRequestReceipt = {
      client_request_id: pending.request.client_request_id,
      request_digest: pending.request.request_digest,
      state: 'reconciling',
    }
    setOwned({ request: pending.request, receipt, cancelAttempted: pending.cancelAttempted === true })
    // A reloaded browser only reads the durable receipt. It never repeats POST.
    void api.getWorldRequest(pending.request.client_request_id)
      .then(finishReceipt)
      .catch(() => setNotice('Checking the saved request receipt. No generation was resubmitted.'))
  }, [open, finishReceipt])

  useEffect(() => {
    if (!open || !activeRequestId || !shouldPollReceipt) return
    const poll = async () => {
      if (pollLock.current) return
      pollLock.current = true
      try {
        const receipt = await api.getWorldRequest(activeRequestId)
        if (ownedRef.current?.request.client_request_id === receipt.client_request_id) finishReceipt(receipt)
      } catch {
        // The durable request remains stored; the next GET retries reconciliation.
      } finally {
        pollLock.current = false
      }
    }
    const timer = window.setInterval(() => { void poll() }, 2500)
    return () => window.clearInterval(timer)
  }, [open, activeRequestId, shouldPollReceipt, finishReceipt])

  useEffect(() => {
    return () => {
      if (captureUrl) URL.revokeObjectURL(captureUrl)
    }
  }, [captureUrl])

  useEffect(() => {
    if (!open) return
    const previousFocus = document.activeElement instanceof HTMLElement ? document.activeElement : null
    const panel = panelRef.current
    const focusableSelector = 'a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])'
    const focusInitial = () => {
      const first = closeButtonRef.current || panel?.querySelector<HTMLElement>(focusableSelector)
      ;(first || panel)?.focus()
    }
    const animationFrame = window.requestAnimationFrame(focusInitial)
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape' && onCloseRef.current) {
        event.stopPropagation()
        onCloseRef.current()
        return
      }
      if (event.key !== 'Tab' || !panel) return
      const items = Array.from(panel.querySelectorAll<HTMLElement>(focusableSelector))
      if (items.length === 0) {
        event.preventDefault()
        panel.focus()
        return
      }
      const first = items[0]
      const last = items[items.length - 1]
      if (event.shiftKey && (document.activeElement === first || !panel.contains(document.activeElement))) {
        event.preventDefault(); last.focus()
      } else if (!event.shiftKey && (document.activeElement === last || !panel.contains(document.activeElement))) {
        event.preventDefault(); first.focus()
      }
    }
    document.addEventListener('keydown', onKeyDown, true)
    return () => {
      window.cancelAnimationFrame(animationFrame)
      document.removeEventListener('keydown', onKeyDown, true)
      if (previousFocus?.isConnected) previousFocus.focus()
    }
  }, [open])

  const startRequest = async (fields: WorldRequestFields) => {
    if (startLock.current || inFlight) return
    startLock.current = true
    setError(null)
    setNotice(null)
    try {
      const request = await api.prepareWorldRequest(fields)
      const initial: OwnedRequest = {
        request,
        receipt: {
          client_request_id: request.client_request_id,
          request_digest: request.request_digest,
          state: 'submitting',
        },
        cancelAttempted: false,
      }
      // Durably remember both the exact UUID and digest before the one allowed POST.
      savePending({ request, cancelAttempted: false })
      setOwned(initial)
      try {
        const receipt = await api.submitWorldRequest(request)
        finishReceipt(receipt)
      } catch (postError) {
        // POST outcome may be ambiguous. Reconcile with GET only; never retry POST.
        try {
          finishReceipt(await api.getWorldRequest(request.client_request_id))
        } catch {
          setNotice('The request receipt is not available yet. This request will be checked again without resubmitting.')
          setError(postError instanceof Error ? postError.message : 'Request status is still being reconciled.')
        }
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Could not prepare the world request.')
    } finally {
      startLock.current = false
    }
  }

  const startMove = (action: typeof MOVE_ACTIONS[number]['action']) => {
    if (!selected) return
    void startRequest({
      operation: 'move', world_id: selected.world_id, action,
      duration_ms: durationMs, acknowledge_pixelumm_noncommercial: true,
    })
  }

  const startEvent = () => {
    if (!selected || !eventText.trim()) return
    void startRequest({
      operation: 'continue', world_id: selected.world_id, text: eventText.trim(),
      acknowledge_pixelumm_noncommercial: true,
    })
  }

  const cancelOwnedRequest = async () => {
    const current = ownedRef.current
    if (!current || current.cancelAttempted || !current.receipt.job_id || !inFlight) return
    const updated = { ...current, cancelAttempted: true }
    try {
      savePending({ request: current.request, cancelAttempted: true })
      setOwned(updated)
      const receipt = await api.cancelWorldRequest(current.request.client_request_id)
      finishReceipt(receipt)
      setNotice('Cancellation was sent for this request. Checking its final status…')
    } catch (e) {
      // A cancel POST can also have an ambiguous result. Persist the one-shot
      // marker and only reconcile status from here.
      setOwned(updated)
      setError(e instanceof Error ? e.message : 'Cancel status is being reconciled.')
      try {
        finishReceipt(await api.getWorldRequest(current.request.client_request_id))
      } catch {
        setNotice('Cancel status is unknown. This request will be checked by GET; cancel will not be repeated.')
      }
    }
  }

  const selectWorld = async (worldId: string) => {
    setError(null)
    try {
      const world = await api.getWorld(worldId)
      setWorlds(current => [world, ...current.filter(item => item.world_id !== world.world_id)])
      setSelectedId(world.world_id)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Could not reopen this world.')
    }
  }

  const downloadSavedWorld = async () => {
    if (!selected || fileBusy || !acknowledged) return
    setFileBusy(true); setError(null)
    try {
      const blob = await api.downloadWorld(selected.world_id)
      api.triggerBrowserDownload(blob, `${selected.world_id}.wgpworld`)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Could not save the world file.')
    } finally {
      setFileBusy(false)
    }
  }

  const recordWorld = async () => {
    if (!selected || fileBusy || !selected.recording_available || !acknowledged) return
    setFileBusy(true); setError(null)
    try {
      const blob = await api.recordWorld(selected.world_id)
      api.triggerBrowserDownload(blob, `${selected.world_id}-recording.mp4`)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Recording is unavailable.')
    } finally {
      setFileBusy(false)
    }
  }

  const captureFrame = async () => {
    if (!selected || fileBusy) return
    setFileBusy(true); setError(null)
    try {
      const blob = await api.captureWorld(selected.world_id)
      setCaptureUrl(current => {
        if (current) URL.revokeObjectURL(current)
        return URL.createObjectURL(blob)
      })
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Capture is unavailable.')
    } finally {
      setFileBusy(false)
    }
  }

  if (!open) return null

  return (
    <section ref={panelRef} role="dialog" aria-modal="true" aria-labelledby="world-explorer-title" tabIndex={-1} className="fixed inset-y-0 right-0 z-[85] flex w-full max-w-5xl flex-col border-l border-border bg-bg-primary shadow-2xl">
      <header className="flex shrink-0 items-center gap-3 border-b border-border px-4 py-3">
        <div className="min-w-0 flex-1">
          <h2 id="world-explorer-title" className="text-sm font-semibold text-text-primary">World Explorer <span className="ml-1 text-[10px] font-normal uppercase tracking-wide text-indicator-warning">Experimental</span></h2>
          <p className="mt-0.5 text-[10px] text-text-muted">Each action is a queued generation and can take time; this is not real-time movement.</p>
        </div>
        <button onClick={() => void refreshWorlds()} disabled={loadingWorlds} className="rounded-md border border-border p-2 text-text-secondary hover:text-text-primary disabled:opacity-50" title="Refresh saved worlds">
          {loadingWorlds ? <Loader2 size={15} className="animate-spin" /> : <RefreshCw size={15} />}
        </button>
        {onClose && <button ref={closeButtonRef} onClick={onClose} className="rounded-md border border-border p-2 text-text-secondary hover:text-text-primary" title="Close Worlds" aria-label="Close World Explorer"><X size={15} /></button>}
      </header>

      <div className="shrink-0 border-b border-amber-500/30 bg-amber-500/10 px-4 py-2.5 text-[11px] text-text-primary">
        <div className="font-medium">PixelUMM decoder terms</div>
        <p className="mt-0.5 text-text-secondary">Hy-Nomas Worlds uses NVIDIA PixelUMM fine-tuned decoder weights restricted to non-commercial research or evaluation.</p>
        <div className="mt-1.5 flex flex-wrap items-center gap-2">
          <a className="underline underline-offset-2 hover:text-text-primary" href={api.PIXELUMM_LICENSE_URL} target="_blank" rel="noreferrer">Read the NVIDIA license</a>
          <label className="flex cursor-pointer items-center gap-1.5 text-text-primary">
            <input type="checkbox" checked={acknowledged} disabled={inFlight} onChange={event => setAcknowledged(event.target.checked)} />
            I will use the decoder only under these terms.
          </label>
        </div>
      </div>

      {error && <div role="alert" className="shrink-0 border-b border-red-500/30 bg-red-500/10 px-4 py-2 text-[11px] text-chip-red">{error}</div>}
      {notice && <div className="shrink-0 border-b border-border bg-bg-secondary px-4 py-2 text-[11px] text-text-secondary">{notice}</div>}
      {owned && inFlight && (
        <div className="flex shrink-0 flex-wrap items-center gap-2 border-b border-border bg-bg-secondary px-4 py-2 text-[11px] text-text-secondary">
          <Loader2 size={13} className="animate-spin" />
          <span className="flex-1">Owned request: {owned.receipt.job_status || owned.receipt.state}. This job is reconciled by its saved request ID.</span>
          <button onClick={() => void cancelOwnedRequest()} disabled={!owned.receipt.job_id || owned.cancelAttempted || !['held', 'queued', 'running'].includes(owned.receipt.job_status || '')} className="rounded border border-border px-2 py-1 text-text-primary hover:border-red-500/60 disabled:opacity-40">
            {owned.cancelAttempted ? 'Cancel sent' : 'Cancel this job'}
          </button>
        </div>
      )}

      <div className="grid min-h-0 flex-1 grid-cols-1 md:grid-cols-[250px_minmax(0,1fr)]">
        <aside className="min-h-0 overflow-y-auto border-b border-border p-3 md:border-b-0 md:border-r">
          <div className="mb-2 flex items-center justify-between">
            <h3 className="text-xs font-semibold text-text-primary">Saved worlds</h3>
            <span className="text-[10px] text-text-muted">{worlds.length}</span>
          </div>
          {loadingWorlds && worlds.length === 0 ? (
            <div className="flex items-center gap-2 py-4 text-xs text-text-muted"><Loader2 size={14} className="animate-spin" /> Loading worlds…</div>
          ) : worlds.length === 0 ? (
            <p className="py-3 text-xs leading-relaxed text-text-muted">No saved worlds yet. Describe a starting place to create one.</p>
          ) : (
            <div className="space-y-1.5">
              {worlds.map(world => (
                <button key={world.world_id} onClick={() => void selectWorld(world.world_id)} disabled={inFlight} className={`w-full rounded-lg border px-2.5 py-2 text-left transition-colors disabled:opacity-50 ${selectedId === world.world_id ? 'border-accent-blue/70 bg-accent-blue/10' : 'border-border bg-bg-secondary hover:border-border-light'}`}>
                  <span className="block truncate text-xs font-medium text-text-primary">{world.title || world.description || world.world_id}</span>
                  <span className="mt-0.5 block text-[10px] text-text-muted">{world.segment_count ?? 1} saved segment{world.segment_count === 1 ? '' : 's'}</span>
                </button>
              ))}
            </div>
          )}

          <div className="mt-5 border-t border-border pt-3">
            <label htmlFor="world-description" className="mb-1.5 block text-xs font-semibold text-text-primary">Create a world</label>
            <textarea id="world-description" value={description} onChange={event => setDescription(event.target.value)} disabled={generationDisabled} maxLength={5000} rows={4} placeholder="Describe the place and what can be seen…" className="w-full resize-y rounded-lg border border-border bg-bg-secondary px-2.5 py-2 text-xs text-text-primary outline-none placeholder:text-text-muted focus:border-accent-blue disabled:opacity-50" />
            <button onClick={() => void startRequest({ operation: 'create', description: description.trim(), acknowledge_pixelumm_noncommercial: true })} disabled={generationDisabled || !description.trim()} className="mt-2 w-full rounded-lg bg-accent-blue px-3 py-2 text-xs font-medium text-white hover:brightness-110 disabled:cursor-not-allowed disabled:opacity-45">
              {owned && inFlight ? <span className="inline-flex items-center gap-1.5"><Loader2 size={13} className="animate-spin" /> {owned.receipt.job_status || owned.receipt.state}</span> : 'Create world'}
            </button>
          </div>
        </aside>

        <main className="min-h-0 overflow-y-auto p-4">
          {!selected ? (
            <div className="mx-auto flex min-h-64 max-w-xl flex-col items-center justify-center text-center">
              <Film size={28} className="text-text-muted" />
              <h3 className="mt-3 text-sm font-medium text-text-primary">Explore a saved world</h3>
              <p className="mt-1 max-w-md text-xs leading-relaxed text-text-muted">Create a starting scene or reopen a world from your saved list. Each move continues from its saved state.</p>
            </div>
          ) : (
            <div className="mx-auto max-w-3xl space-y-4">
              <div>
                <h3 className="truncate text-base font-semibold text-text-primary">{selected.title || selected.description || 'Saved world'}</h3>
                {selected.description && <p className="mt-1 text-xs leading-relaxed text-text-secondary">{selected.description}</p>}
              </div>
              <div className="overflow-hidden rounded-xl border border-border bg-black">
                {selected.video_url ? (
                  <video className="max-h-[45vh] w-full bg-black object-contain" src={selected.video_url} poster={selected.poster_url} controls playsInline />
                ) : selected.poster_url ? (
                  <img className="max-h-[45vh] w-full object-contain" src={selected.poster_url} alt="Saved world poster" />
                ) : (
                  <div className="flex aspect-video items-center justify-center text-xs text-white/70">No preview is available.</div>
                )}
              </div>

              <div className="grid gap-4 lg:grid-cols-2">
                <section className="rounded-xl border border-border bg-bg-secondary p-3">
                  <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
                    <h4 className="text-xs font-semibold text-text-primary">Move and look</h4>
                    <label className="flex items-center gap-1.5 text-[10px] text-text-secondary">Duration
                      <select value={durationMs} onChange={event => setDurationMs(Number(event.target.value))} disabled={generationDisabled} className="rounded border border-border bg-bg-primary px-1.5 py-1 text-text-primary disabled:opacity-50">
                        <option value={800}>0.8 s</option><option value={1500}>1.5 s</option><option value={2500}>2.5 s</option><option value={5000}>5 s</option><option value={10000}>10 s</option>
                      </select>
                    </label>
                  </div>
                  <div className="grid grid-cols-3 gap-1.5">
                    {MOVE_ACTIONS.map(item => {
                      const Icon = 'icon' in item ? item.icon : undefined
                      return (
                        <button key={item.action} onClick={() => startMove(item.action)} disabled={generationDisabled} className="flex min-h-9 items-center justify-center gap-1.5 rounded-md border border-border bg-bg-primary px-2 py-1.5 text-[10px] text-text-secondary hover:border-accent-blue/60 hover:text-text-primary disabled:cursor-not-allowed disabled:opacity-45">
                          {Icon && <Icon size={12} />}{item.label}
                        </button>
                      )
                    })}
                  </div>
                  <label htmlFor="world-event" className="mb-1.5 mt-3 block text-[10px] font-medium text-text-secondary">Event in the world</label>
                  <div className="flex gap-1.5">
                    <input id="world-event" value={eventText} onChange={event => setEventText(event.target.value)} onKeyDown={event => { if (event.key === 'Enter') startEvent() }} disabled={generationDisabled} maxLength={2000} placeholder="The fox steps onto the trail…" className="min-w-0 flex-1 rounded-md border border-border bg-bg-primary px-2 py-1.5 text-xs text-text-primary outline-none placeholder:text-text-muted focus:border-accent-blue disabled:opacity-50" />
                    <button onClick={startEvent} disabled={generationDisabled || !eventText.trim()} className="rounded-md bg-accent-blue px-3 text-[10px] font-medium text-white disabled:opacity-45">Event</button>
                  </div>
                </section>

                <section className="rounded-xl border border-border bg-bg-secondary p-3">
                  <h4 className="mb-2 text-xs font-semibold text-text-primary">World files</h4>
                  <div className="grid grid-cols-2 gap-1.5">
                    <button onClick={() => void downloadSavedWorld()} disabled={fileBusy || !acknowledged || inFlight} className="flex items-center justify-center gap-1.5 rounded-md border border-border bg-bg-primary px-2 py-2 text-[10px] text-text-secondary hover:text-text-primary disabled:opacity-45"><Download size={12} /> Save .wgpworld</button>
                    <button onClick={() => void recordWorld()} disabled={fileBusy || !selected.recording_available || !acknowledged || inFlight} title={selected.recording_available ? 'Concatenate saved clips' : 'No saved clip chain is available'} className="flex items-center justify-center gap-1.5 rounded-md border border-border bg-bg-primary px-2 py-2 text-[10px] text-text-secondary hover:text-text-primary disabled:opacity-45"><Film size={12} /> Record clips</button>
                    <button onClick={() => void captureFrame()} disabled={fileBusy || inFlight} className="col-span-2 flex items-center justify-center gap-1.5 rounded-md border border-border bg-bg-primary px-2 py-2 text-[10px] text-text-secondary hover:text-text-primary disabled:opacity-45"><Camera size={12} /> Capture saved frame</button>
                  </div>
                  {!selected.recording_available && <p className="mt-2 text-[10px] leading-relaxed text-text-muted">Recording is unavailable until this world has a saved creation and move clip chain.</p>}
                  {captureUrl && <img className="mt-3 max-h-48 w-full rounded-md border border-border object-contain" src={captureUrl} alt="Captured saved world frame" />}
                </section>
              </div>

            </div>
          )}
        </main>
      </div>
    </section>
  )
}
