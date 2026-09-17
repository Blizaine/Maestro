import { useEffect, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import { X } from 'lucide-react'
import * as api from '../api/client'
import { useStore } from '../stores/useStore'
import type { GenerationJob } from '../types'

export function EnhancedJobReview({job, onClose, onSubmitted}: {job: GenerationJob; onClose: () => void; onSubmitted?: () => void}) {
  const dialog = useRef<HTMLDialogElement>(null)
  const [data, setData] = useState<Awaited<ReturnType<typeof api.fetchJobEnhancement>> | null>(null)
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  useEffect(() => {
    dialog.current?.showModal()
    let alive = true
    const refresh = () => api.fetchJobEnhancement(job.id).then(result => { if (alive) setData(result) })
      .catch(reason => { if (alive) setError(String(reason.message || reason)) })
    void refresh()
    const timer = ['held', 'queued', 'running'].includes(job.status) ? setInterval(() => void refresh(), 3000) : undefined
    return () => { alive = false; clearInterval(timer) }
  }, [job.id, job.status])
  const act = async (action: 'retry' | 'refresh' | 'as_written' | 'accept_draft') => {
    if (busy) return
    setBusy(true); setError('')
    try {
      const accepted = await api.retryEnhancedJob(job.id, action)
      await useStore.getState().reconnectJobs({
        ...job, id: accepted.job_id, status: accepted.status, progress: 0,
        step: 0, totalSteps: 0, phase: '', message: 'Queued for generation',
        error: null, outputFiles: [], showInGallery: true,
        enhancement: {...job.enhancement!, error: undefined,
          state: action === 'accept_draft' || action === 'as_written' ? 'complete'
            : action === 'refresh' ? 'pending' : job.enhancement!.state},
      })
      onClose()
      onSubmitted?.()
    } catch (reason) { setError(reason instanceof Error ? reason.message : String(reason)) }
    finally { setBusy(false) }
  }
  const openDraft = async (original: boolean) => {
    if (!data || busy) return
    setBusy(true)
    const previousMetadata = useStore.getState().selectedOutputMeta
    let loadedMetadata: typeof previousMetadata = null
    try {
      const params = original ? data.original_params : data.prepared?.params
      if (!params) return
      const metadata = {source: 'sidecar' as const, params}
      loadedMetadata = metadata
      useStore.setState({selectedOutputMeta: metadata})
      await useStore.getState().loadSettingsFromOutput()
      useStore.getState().setEnhanceOnGeneration(false)
      useStore.getState().setSidebarOpen(true)
      onClose()
    } catch (reason) { setError(reason instanceof Error ? reason.message : String(reason)) }
    finally {
      if (loadedMetadata && useStore.getState().selectedOutputMeta === loadedMetadata) {
        useStore.setState({selectedOutputMeta: previousMetadata})
      }
      setBusy(false)
    }
  }
  const terminal = ['failed', 'cancelled', 'completed'].includes(job.status)
  const button = 'min-h-9 rounded-lg border border-border px-3 py-2 text-xs hover:bg-bg-hover disabled:opacity-40'
  return createPortal(<dialog ref={dialog} onCancel={onClose} aria-labelledby="enhanced-job-title"
    className="m-auto w-[min(680px,calc(100vw-24px))] max-h-[calc(100dvh-24px)] overflow-y-auto rounded-xl border border-border bg-bg-secondary p-4 text-text-primary backdrop:bg-black/60">
    <div className="mb-3 flex items-center justify-between gap-3">
      <h2 id="enhanced-job-title" className="text-sm font-medium">Job prompts</h2>
      <button type="button" aria-label="Close job prompts" onClick={onClose} className="rounded p-2 hover:bg-bg-hover"><X size={16}/></button>
    </div>
    {error && <p role="alert" className="mb-3 text-xs text-indicator-warning">{error}</p>}
    {!data ? <p className="text-xs text-text-muted">Loading saved prompts…</p> : <>
      {(data.enhancement.error || job.error) && <p role="alert" className="mb-3 text-xs text-indicator-warning">{data.enhancement.error || job.error}</p>}
      <details className="mb-3" open={!data.enhancement.enhanced_prompt}>
        <summary className="cursor-pointer text-xs font-medium">Original prompt</summary>
        <p className="mt-2 whitespace-pre-wrap break-words text-xs leading-relaxed">{data.enhancement.original_prompt}</p>
      </details>
      {data.prepared?.h3_window_plan?.windows?.length ? data.prepared.h3_window_plan.windows.map((window, index) =>
        <details key={index} className="mb-3" open={index === 0}>
          <summary className="cursor-pointer text-xs font-medium">Window {index + 1}</summary>
          <p className="mt-2 whitespace-pre-wrap break-words text-xs leading-relaxed">{window.prompt}</p>
        </details>) : data.enhancement.enhanced_prompt && <div className="mb-3">
          <h3 className="mb-2 text-xs font-medium">Enhanced prompt</h3>
          <p className="whitespace-pre-wrap break-words text-xs leading-relaxed">{data.enhancement.enhanced_prompt}</p>
        </div>}
      {data.enhancement.warnings?.map((warning, index) => <p key={index} className="mb-2 text-xs text-indicator-warning">{warning}</p>)}
      <div className="mt-4 flex flex-wrap gap-2">
        <button className={button} disabled={busy} onClick={() => void openDraft(true)}>Use original in Studio</button>
        {data.prepared && <button className={button} disabled={busy} onClick={() => void openDraft(false)}>Edit draft in Studio</button>}
        {terminal && <>
          <button className={button} disabled={busy} onClick={() => void act('retry')}>{data.enhancement.state === 'complete' ? 'Retry generation' : 'Retry enhancement'}</button>
          {data.enhancement.state === 'complete' && <button className={button} disabled={busy} onClick={() => void act('refresh')}>Create a new draft</button>}
          {data.enhancement.state === 'review' && <button className={button} disabled={busy} onClick={() => void act('accept_draft')}>Generate reviewed draft</button>}
          {data.enhancement.state !== 'complete' && <button className={button} disabled={busy} onClick={() => void act('as_written')}>Generate as written</button>}
        </>}
      </div>
    </>}
  </dialog>, document.body)
}
