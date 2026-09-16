import { useState } from 'react'
import { useStore } from '../../stores/useStore'
import * as api from '../../api/client'
import { trackMediaFlowJobs } from '../../lib/mediaFlow'

export function OutpaintBatchPanel() {
  const [files, setFiles] = useState<Array<{ path: string; name: string }>>([])
  const [busy, setBusy] = useState(false)
  const [message, setMessage] = useState('')
  const add = async (list: FileList | null) => {
    if (!list) return
    setBusy(true)
    setMessage('')
    try {
      if (files.length + list.length > 256) throw new Error('A batch supports up to 256 videos.')
      for (const file of Array.from(list)) {
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
      const state = useStore.getState()
      const { x, y, w, h } = state.outpaintVideoBox
      if (w <= 0 || h <= 0) throw new Error('Set the source position in the canvas first.')
      const margins = [100 * y / h, 100 * Math.max(0, 1 - y - h) / h,
        100 * x / w, 100 * Math.max(0, 1 - x - w) / w]
      const response = await fetch('/api/v1/media-flow/outpaint', { method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ files, margins_percent: margins, model_type: state.params.model_type,
          prompt: state.params.prompt || 'Extend the scene naturally', resolution_preset: state.outpaintResolutionPreset,
          preserve_source_audio: state.outpaintPreserveSourceAudio, mask_preserving_outpaint: state.outpaintMaskPreserving,
          num_inference_steps: state.params.num_inference_steps, guidance_scale: state.params.guidance_scale,
          activated_loras: state.params.activated_loras, loras_multipliers: state.params.loras_multipliers,
          custom_settings: state.params.custom_settings, seed: state.params.seed,
          sliding_window_size: Math.round(state.slidingWindowSeconds * (state.modelOptions?.fps || 24)),
          sliding_window_overlap: state.slidingWindowOverlap, workspace: state.activeWorkspace }) })
      const result = await response.json()
      if (!response.ok) throw new Error(result.detail || 'Could not submit batch')
      if (result.job_ids.length) trackMediaFlowJobs(result.job_ids)
      setMessage(`Queued ${result.job_ids.length} videos. ${result.errors.map((error: { path: string; error: string }) => `${error.path}: ${error.error}`).join('; ')}`)
    } catch (error) { setMessage(error instanceof Error ? error.message : 'Could not submit batch') }
    finally { setBusy(false) }
  }
  return <details className="border-t border-border pt-3 text-xs">
    <summary className="cursor-pointer text-text-primary">Media Flow — batch Outpaint</summary>
    <div className="space-y-2 mt-3">
      <p className="text-text-muted">Uses this canvas's proportional margins, prompt, model and quality for each full video.</p>
      <input type="file" multiple accept="video/*" disabled={busy} onChange={event => { void add(event.target.files); event.target.value = '' }} className="w-full" />
      <div className="max-h-36 overflow-auto">{files.map((file, index) => <div key={`${file.path}-${index}`} className="flex gap-2">
        <span className="truncate flex-1">{file.name}</span><button disabled={busy} aria-label={`Remove ${file.name}`} onClick={() => setFiles(previous => previous.filter((_, i) => i !== index))}>×</button>
      </div>)}</div>
      <button className="bg-cta text-white rounded p-2 w-full disabled:opacity-40" disabled={busy || !files.length} onClick={() => void run()}>{busy ? 'Preparing…' : `Queue ${files.length} Outpaint videos`}</button>
      {message && <p className="text-text-secondary break-words">{message}</p>}
    </div>
  </details>
}
