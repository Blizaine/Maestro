import { useState } from 'react'
import { useStore } from '../../stores/useStore'
import * as api from '../../api/client'
import { MediaFinishingControls, dlssSpatialOptions } from './MediaFinishingControls'
import type { GenerationJob } from '../../types'

export function trackMediaFlowJobs(ids: string[]) {
  const jobs: GenerationJob[] = ids.map(id => ({ id, status: 'queued', progress: 0,
    step: 0, totalSteps: 0, phase: '', message: 'Queued (Media Flow)',
    outputFiles: [], error: null, oomInfo: null }))
  useStore.setState(state => ({ jobs: [...jobs, ...state.jobs], isGenerating: true }))
  const pending = new Set(ids)
  let polling = false
  const timer = setInterval(async () => {
    if (polling) return
    polling = true
    try {
      await Promise.all([...pending].map(async id => {
        try {
          const status = await api.fetchJobStatus(id)
          const terminal = ['completed', 'cancelled', 'failed'].includes(status.status)
          useStore.setState(state => {
            const updated = state.jobs.map(job => job.id === id ? { ...job, status: status.status,
              progress: status.progress / 100, step: status.step, totalSteps: status.total_steps,
              message: status.message, phase: status.phase, outputFiles: status.output_files,
              error: status.error, oomInfo: status.oom_info ?? null } : job)
            return { jobs: updated, isGenerating: updated.some(job => ['running', 'queued'].includes(job.status)) }
          })
          if (terminal) {
            pending.delete(id)
            if (status.status === 'completed') void useStore.getState().loadOutputs()
          }
        } catch { /* A transient connection failure does not cancel server jobs. */ }
      }))
    } finally {
      polling = false
      if (!pending.size) clearInterval(timer)
    }
  }, 2000)
}

export function MediaFlowPanel({ image = false }: { image?: boolean }) {
  const [files, setFiles] = useState<Array<{ path: string; name: string }>>([])
  const [spatial, setSpatial] = useState(image ? 'dlss5*1' : '')
  const [temporal, setTemporal] = useState(image ? '' : 'rife2')
  const [options, setOptions] = useState<Record<string, unknown>>({})
  const [busy, setBusy] = useState(false)
  const [message, setMessage] = useState('')
  const outputs = useStore(state => state.outputs)
  const selected = useStore(state => state.selectedOutput)
  const current = outputs[selected]
  const addFiles = async (selectedFiles: FileList | null) => {
    if (!selectedFiles) return
    setBusy(true)
    setMessage('')
    try {
      if (files.length + selectedFiles.length > 256) throw new Error('A batch can contain up to 256 files.')
      for (const file of Array.from(selectedFiles)) {
        const result = await api.uploadImage(file)
        setFiles(previous => [...previous, { path: result.path, name: file.name }])
      }
    } catch (error) { setMessage(error instanceof Error ? error.message : 'Upload failed') }
    finally { setBusy(false) }
  }
  const run = async () => {
    setBusy(true)
    setMessage('')
    try {
      const response = await fetch('/api/v1/media-flow', { method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ files, spatial_upsampling: spatial, temporal_upsampling: image ? '' : temporal,
          ...options, workspace: useStore.getState().activeWorkspace }) })
      const result = await response.json()
      if (!response.ok) throw new Error(result.detail || 'Could not queue Media Flow')
      trackMediaFlowJobs(result.job_ids)
      setMessage(`Queued ${result.job_ids.length} files. Progress and cancellation are in the job queue.`)
    } catch (error) { setMessage(error instanceof Error ? error.message : 'Could not start Media Flow') }
    finally { setBusy(false) }
  }
  return <details className="border-t border-border pt-3 text-xs">
    <summary className="cursor-pointer text-text-primary font-medium">Media Flow — batch processing</summary>
    <div className="space-y-3 mt-3">
      <p className="text-text-muted">Process a collection with the same settings. Each result is saved as a new file; video audio is retained.</p>
      <input type="file" multiple accept={image ? 'image/*' : 'video/*'} disabled={busy}
        onChange={event => { void addFiles(event.target.files); event.target.value = '' }} className="w-full text-xs" />
      <button className="text-accent-blue disabled:opacity-40" disabled={!current || current.type !== (image ? 'image' : 'video') || busy}
        onClick={() => { if (current) setFiles(previous => [...previous, { path: current.name, name: current.name }]) }}>Add selected gallery {image ? 'image' : 'video'}</button>
      {files.length > 0 && <div className="max-h-40 overflow-y-auto space-y-1">
        {files.map((file, index) => <div className="flex gap-2" key={`${file.path}-${index}`}>
          <span className="truncate flex-1">{file.name}</span>
          <button aria-label={`Remove ${file.name}`} disabled={busy} onClick={() => setFiles(previous => previous.filter((_, i) => i !== index))}>×</button>
        </div>)}
      </div>}
      <label className="block">Neural Rendering / spatial scaling
        <select className="w-full bg-bg-tertiary border border-border rounded-lg p-2" value={spatial} onChange={event => setSpatial(event.target.value)}>
          {!image && <option value="">Original resolution</option>}
          {dlssSpatialOptions.map(option => <option key={option.value} value={option.value}>{option.label}</option>)}
          {[1.5, 2, 3, 4].map(scale => <option key={scale} value={`lanczos${scale}`}>Lanczos ×{scale}</option>)}
        </select>
      </label>
      <MediaFinishingControls spatial={spatial} temporal={temporal} onTemporal={setTemporal}
        options={options} onOptions={setOptions} image={image} />
      <button disabled={busy || !files.length || (!spatial && !temporal)} onClick={() => void run()}
        className="w-full rounded-lg bg-cta p-2 text-white disabled:opacity-40">{busy ? 'Preparing…' : `Queue ${files.length} files`}</button>
      {message && <p className="text-text-secondary break-words">{message}</p>}
    </div>
  </details>
}
