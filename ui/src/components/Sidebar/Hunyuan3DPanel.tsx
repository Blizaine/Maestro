import { useEffect, useMemo, useRef, useState } from 'react'
import { Box, ChevronDown, Cpu, Layers3, Loader2, Play, Square, Upload, X } from 'lucide-react'
import { useStore } from '../../stores/useStore'
import { ModelSelector } from './ModelSelector'
import {
  cancelHunyuan3DJob,
  fetchHunyuan3DCapabilities,
  fetchHunyuan3DJob,
  startHunyuan3DJob,
  uploadImage,
  type Hunyuan3DCapabilities,
  type Hunyuan3DJob,
} from '../../api/client'

type ViewName = 'front' | 'left' | 'right' | 'back'
type UploadedView = { path: string; name: string; url: string }

const viewLabels: Record<ViewName, string> = {
  front: 'Front',
  left: 'Left',
  right: 'Right',
  back: 'Back',
}

function ViewUpload({ view, value, busy, required, onUpload, onRemove }: {
  view: ViewName
  value?: UploadedView
  busy: boolean
  required?: boolean
  onUpload: (file: File) => void
  onRemove: () => void
}) {
  const inputRef = useRef<HTMLInputElement>(null)
  return (
    <div className="min-w-0">
      <div className="text-[10px] text-text-muted uppercase tracking-wider mb-1">
        {viewLabels[view]}{required ? ' *' : ''}
      </div>
      {value ? (
        <div className="relative aspect-square rounded-lg overflow-hidden border border-border bg-bg-primary group">
          <img src={value.url} alt={viewLabels[view]} className="w-full h-full object-cover" />
          <button onClick={onRemove} className="absolute top-1 right-1 p-1 rounded-full bg-black/60 text-white hover:bg-red-600 transition-colors" title={`Remove ${viewLabels[view]} view`}>
            <X size={11} />
          </button>
          <div className="absolute inset-x-0 bottom-0 bg-black/60 px-1.5 py-1 text-[9px] text-white truncate">{value.name}</div>
        </div>
      ) : (
        <button type="button" disabled={busy} onClick={() => inputRef.current?.click()} className="w-full aspect-square rounded-lg border border-dashed border-border bg-bg-primary hover:border-accent-blue text-text-muted hover:text-accent-blue transition-colors flex flex-col items-center justify-center gap-1 disabled:opacity-50">
          {busy ? <Loader2 size={16} className="animate-spin" /> : <Upload size={16} />}
          <span className="text-[9px]">{busy ? 'Uploading' : 'Add image'}</span>
        </button>
      )}
      <input ref={inputRef} type="file" accept="image/*" className="hidden" onChange={event => {
        const file = event.target.files?.[0]
        if (file) onUpload(file)
        event.target.value = ''
      }} />
    </div>
  )
}

export function Hunyuan3DPanel() {
  const loadOutputs = useStore(state => state.loadOutputs)
  const setMediaFilter = useStore(state => state.setMediaFilter)
  const enabledModels = useStore(state => state.enabledModels)
  const toggleModelEnabled = useStore(state => state.toggleModelEnabled)
  const modelId = useStore(state => state.params.model_type)
  const prompt = useStore(state => state.params.prompt)
  const setParam = useStore(state => state.setParam)
  const selectMaestroModel = useStore(state => state.selectModel)
  const [capabilities, setCapabilities] = useState<Hunyuan3DCapabilities | null>(null)
  const [capabilityError, setCapabilityError] = useState<string | null>(null)
  const [views, setViews] = useState<Partial<Record<ViewName, UploadedView>>>({})
  const [uploadingView, setUploadingView] = useState<ViewName | null>(null)
  const [preset, setPreset] = useState('balanced')
  const [textureMode, setTextureMode] = useState('v2-turbo')
  const [steps, setSteps] = useState(5)
  const [guidance, setGuidance] = useState(5)
  const [octree, setOctree] = useState(256)
  const [chunks, setChunks] = useState(12000)
  const [seed, setSeed] = useState(1234)
  const [outputFormat, setOutputFormat] = useState('glb')
  const [textureResolution, setTextureResolution] = useState(512)
  const [cpuOffload, setCpuOffload] = useState(true)
  const [flashvdm, setFlashvdm] = useState(true)
  const [removeBackground, setRemoveBackground] = useState(true)
  const [compile, setCompile] = useState(false)
  const [reduceFace, setReduceFace] = useState(false)
  const [targetFaces, setTargetFaces] = useState(40000)
  const [mcAlgo, setMcAlgo] = useState('dmc')
  const [advancedOpen, setAdvancedOpen] = useState(false)
  const [job, setJob] = useState<Hunyuan3DJob | null>(null)
  const [error, setError] = useState<string | null>(null)
  const completedJobRef = useRef<string | null>(null)

  useEffect(() => {
    fetchHunyuan3DCapabilities().then(setCapabilities).catch(err => {
      setCapabilityError(err instanceof Error ? err.message : 'Could not load Hunyuan3D capabilities')
    })
  }, [])

  const selectedModel = useMemo(() => capabilities?.models.find(model => model.id === modelId), [capabilities, modelId])
  const isMultiview = !!selectedModel?.multiview
  const isRunning = job?.status === 'queued' || job?.status === 'running'
  const installed = !!capabilities?.runtime.installed
  const hasInput = isMultiview ? !!views.front : !!views.front || !!prompt.trim()

  const applyPreset = (presetId: string) => {
    const next = capabilities?.presets.find(item => item.id === presetId)
    if (!next) return
    setPreset(presetId)
    selectMaestroModel(next.model_id)
    setSteps(next.num_inference_steps)
    setGuidance(next.guidance_scale)
    setOctree(next.octree_resolution)
    setChunks(next.num_chunks)
    setTextureMode(next.texture_mode)
    setCpuOffload(next.cpu_offload)
    setFlashvdm(next.flashvdm)
  }

  useEffect(() => {
    if (!capabilities || capabilities.models.some(model => model.id === modelId)) return
    selectMaestroModel('hunyuan3d-2-turbo')
  }, [capabilities, modelId, selectMaestroModel])

  useEffect(() => {
    if (!selectedModel) return
    if (selectedModel.engine !== 'v21' && textureMode === 'pbr') setTextureMode('v2-turbo')
    if (selectedModel.engine === 'v21' && textureMode === 'v2-turbo') setTextureMode('pbr')
  }, [selectedModel, textureMode])

  // The backend only accepts PBR when exporting GLB (all material maps must
  // stay embedded), so coerce the format no matter how PBR was activated:
  // preset click, model switch, or manual texture selection.
  useEffect(() => {
    if (textureMode === 'pbr' && outputFormat !== 'glb') setOutputFormat('glb')
  }, [textureMode, outputFormat])

  const uploadView = async (view: ViewName, file: File) => {
    setUploadingView(view)
    setError(null)
    try {
      const result = await uploadImage(file)
      setViews(current => ({ ...current, [view]: { path: result.path, name: file.name, url: result.url } }))
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Image upload failed')
    } finally {
      setUploadingView(null)
    }
  }

  useEffect(() => {
    if (!job || (job.status !== 'queued' && job.status !== 'running')) return
    let disposed = false
    let failures = 0
    const poll = async () => {
      try {
        const next = await fetchHunyuan3DJob(job.job_id)
        failures = 0
        if (!disposed) setJob(next)
      } catch (err) {
        if (disposed) return
        failures += 1
        const message = err instanceof Error ? err.message : 'Could not read 3D job status'
        setError(message)
        // A 404 (job registry lost — backend restart) or repeated failures
        // will never recover; mark the job failed locally so the interval
        // stops and the Generate button becomes usable again.
        const lost = (err as Error & { status?: number }).status === 404
        if (lost || failures >= 4) {
          setJob(current => current && { ...current, status: 'failed', error: lost ? 'The 3D job was lost — the backend probably restarted.' : message })
        }
      }
    }
    const timer = window.setInterval(poll, 1500)
    void poll()
    return () => {
      disposed = true
      window.clearInterval(timer)
    }
  }, [job?.job_id, job?.status])

  useEffect(() => {
    if (job?.status === 'completed' && completedJobRef.current !== job.job_id) {
      completedJobRef.current = job.job_id
      void loadOutputs()
      setMediaFilter('model3d')
    }
    if (job?.status === 'failed') setError(job.error || job.message)
  }, [job, loadOutputs, setMediaFilter])

  const run = async () => {
    setError(null)
    try {
      const images = Object.fromEntries(Object.entries(views).filter(([, value]) => !!value).map(([name, value]) => [name, value!.path]))
      const nextJob = await startHunyuan3DJob({
        preset,
        model_id: modelId,
        prompt: prompt.trim(),
        images,
        texture_mode: textureMode,
        num_inference_steps: steps,
        guidance_scale: guidance,
        octree_resolution: octree,
        num_chunks: chunks,
        seed,
        output_format: outputFormat,
        texture_resolution: textureResolution,
        cpu_offload: cpuOffload,
        flashvdm,
        remove_background: removeBackground,
        compile,
        reduce_face: reduceFace,
        target_face_num: targetFaces,
        mc_algo: mcAlgo,
      })
      // A 3D model is enabled when the user actually starts using it. This
      // also happens to be the point where the isolated runtime fetches its
      // weights on first use.
      if (modelId && !enabledModels.has(modelId)) toggleModelEnabled(modelId)
      setJob(nextJob)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Hunyuan3D generation failed')
    }
  }

  const cancel = async () => {
    if (!job) return
    try {
      setJob(await cancelHunyuan3DJob(job.job_id))
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not cancel 3D generation')
    }
  }

  if (capabilityError) return <div className="rounded-lg border border-red-500/30 bg-red-500/10 p-3 text-xs text-red-300">{capabilityError}</div>

  return (
    <div className="flex flex-col gap-4">
      <div className="flex items-start justify-between gap-3">
        <div>
          <div className="flex items-center gap-1.5 text-xs font-medium text-text-primary"><Box size={15} className="text-accent-blue" /> Hunyuan3D Studio</div>
          <p className="text-[10px] text-text-muted mt-1">Native geometry, multi-view reconstruction and PBR materials.</p>
        </div>
        <div className="flex items-center gap-1 text-[9px] text-accent-green bg-accent-green/10 border border-accent-green/20 rounded-full px-2 py-1 whitespace-nowrap"><Cpu size={10} /> VRAM released after each job</div>
      </div>

      {!capabilities ? (
        <div className="flex items-center justify-center py-8 text-xs text-text-muted"><Loader2 size={15} className="animate-spin mr-2" /> Loading models...</div>
      ) : !installed ? (
        <div className="rounded-lg border border-amber-500/30 bg-amber-500/10 p-3">
          <p className="text-xs font-medium text-amber-300">Hunyuan3D runtime is not installed</p>
          <p className="text-[10px] text-text-muted mt-1 leading-relaxed">Stop Maestro and run its standard <strong>Install</strong> or <strong>Update</strong> action. The runtime stays inside Maestro; model weights download on first use.</p>
        </div>
      ) : (
        <>
          <div>
            <label className="text-[10px] text-text-muted uppercase tracking-wider mb-1.5 block">Performance profile</label>
            <div className="grid grid-cols-2 gap-1.5">
              {capabilities.presets.map(item => (
                <button key={item.id} onClick={() => applyPreset(item.id)} className={`text-left rounded-lg border px-2.5 py-2 transition-colors ${preset === item.id ? 'border-accent-blue bg-accent-blue/10' : 'border-border bg-bg-tertiary hover:border-border-light'}`}>
                  <div className="text-[11px] font-medium text-text-primary">{item.label}</div>
                  <div className="text-[9px] text-text-muted mt-0.5 line-clamp-2">{item.description}</div>
                </button>
              ))}
            </div>
          </div>

          <div>
            <label className="text-[10px] text-text-muted uppercase tracking-wider mb-1.5 block">Hunyuan model</label>
            <ModelSelector />
            {selectedModel && <p className="text-[9px] text-text-muted mt-1">{selectedModel.description} Recommended: {selectedModel.recommended_vram_gb}GB+ VRAM.</p>}
          </div>

          {!isMultiview && (
            <div>
              <label className="text-[10px] text-text-muted uppercase tracking-wider mb-1.5 block">Prompt or reference image</label>
              <textarea value={prompt} onChange={event => setParam('prompt', event.target.value)} rows={3} placeholder="Describe a single object, or upload a reference below..." className="w-full bg-bg-tertiary border border-border rounded-lg px-3 py-2 text-xs text-text-primary resize-none focus:outline-none focus:border-accent-blue" />
            </div>
          )}

          <div>
            <div className="flex items-center justify-between mb-1.5">
              <label className="text-[10px] text-text-muted uppercase tracking-wider">{isMultiview ? 'Reference views' : 'Reference image'}</label>
              {isMultiview && <span className="text-[9px] text-text-muted">Front required; others optional</span>}
            </div>
            <div className={`grid gap-2 ${isMultiview ? 'grid-cols-4' : 'grid-cols-1 max-w-[92px]'}`}>
              {(isMultiview ? (['front', 'left', 'right', 'back'] as ViewName[]) : (['front'] as ViewName[])).map(view => (
                <ViewUpload key={view} view={view} value={views[view]} busy={uploadingView === view} required={isMultiview && view === 'front'} onUpload={file => void uploadView(view, file)} onRemove={() => setViews(current => ({ ...current, [view]: undefined }))} />
              ))}
            </div>
          </div>

          <button onClick={() => setAdvancedOpen(value => !value)} className="flex items-center justify-between w-full rounded-lg bg-bg-tertiary border border-border px-3 py-2 text-[11px] text-text-secondary hover:text-text-primary">
            <span className="flex items-center gap-1.5"><Layers3 size={12} /> Advanced 3D settings</span>
            <ChevronDown size={13} className={`transition-transform ${advancedOpen ? 'rotate-180' : ''}`} />
          </button>

          {advancedOpen && (
            <div className="rounded-lg border border-border bg-bg-tertiary p-3 space-y-3">
              <div className="grid grid-cols-2 gap-2">
                <label className="text-[10px] text-text-muted">Texture
                  <select value={textureMode} onChange={event => setTextureMode(event.target.value)} className="mt-1 w-full bg-bg-primary border border-border rounded px-2 py-1.5 text-[11px] text-text-primary">
                    {capabilities.texture_modes.filter(mode => mode.id !== 'pbr' || selectedModel?.engine === 'v21').map(mode => <option key={mode.id} value={mode.id}>{mode.label}</option>)}
                  </select>
                </label>
                <label className="text-[10px] text-text-muted">Output
                  <select value={outputFormat} onChange={event => setOutputFormat(event.target.value)} className="mt-1 w-full bg-bg-primary border border-border rounded px-2 py-1.5 text-[11px] text-text-primary">
                    {capabilities.output_formats.map(format => <option key={format} value={format} disabled={textureMode === 'pbr' && format !== 'glb'}>{format.toUpperCase()}{textureMode === 'pbr' && format !== 'glb' ? ' (PBR requires GLB)' : ''}</option>)}
                  </select>
                </label>
                <label className="text-[10px] text-text-muted">Steps<input type="number" min={1} max={100} value={steps} onChange={event => setSteps(Number(event.target.value))} className="mt-1 w-full bg-bg-primary border border-border rounded px-2 py-1.5 text-[11px] text-text-primary" /></label>
                <label className="text-[10px] text-text-muted">Guidance<input type="number" min={0} max={30} step={0.1} value={guidance} onChange={event => setGuidance(Number(event.target.value))} className="mt-1 w-full bg-bg-primary border border-border rounded px-2 py-1.5 text-[11px] text-text-primary" /></label>
                <label className="text-[10px] text-text-muted">Octree resolution
                  <select value={octree} onChange={event => setOctree(Number(event.target.value))} className="mt-1 w-full bg-bg-primary border border-border rounded px-2 py-1.5 text-[11px] text-text-primary">{[64, 128, 256, 384, 512].map(value => <option key={value} value={value}>{value}</option>)}</select>
                </label>
                <label className="text-[10px] text-text-muted">Processing chunks<input type="number" min={1000} max={500000} step={1000} value={chunks} onChange={event => setChunks(Number(event.target.value))} className="mt-1 w-full bg-bg-primary border border-border rounded px-2 py-1.5 text-[11px] text-text-primary" /></label>
                <label className="text-[10px] text-text-muted">Seed<input type="number" min={0} value={seed} onChange={event => setSeed(Number(event.target.value))} className="mt-1 w-full bg-bg-primary border border-border rounded px-2 py-1.5 text-[11px] text-text-primary" /></label>
                <label className="text-[10px] text-text-muted">Surface algorithm<select value={mcAlgo} onChange={event => setMcAlgo(event.target.value)} className="mt-1 w-full bg-bg-primary border border-border rounded px-2 py-1.5 text-[11px] text-text-primary"><option value="dmc">DMC</option><option value="mc">Marching Cubes</option></select></label>
                {textureMode !== 'none' && <label className="text-[10px] text-text-muted">Texture resolution<select value={textureResolution} onChange={event => setTextureResolution(Number(event.target.value))} className="mt-1 w-full bg-bg-primary border border-border rounded px-2 py-1.5 text-[11px] text-text-primary"><option value={512}>512</option><option value={768}>768</option><option value={1024}>1024</option></select></label>}
                {reduceFace && <label className="text-[10px] text-text-muted">Target faces<input type="number" min={100} max={1000000} step={1000} value={targetFaces} onChange={event => setTargetFaces(Number(event.target.value))} className="mt-1 w-full bg-bg-primary border border-border rounded px-2 py-1.5 text-[11px] text-text-primary" /></label>}
              </div>
              <div className="grid grid-cols-2 gap-x-3 gap-y-2 text-[10px] text-text-secondary">
                <label className="flex items-center gap-1.5"><input type="checkbox" checked={cpuOffload} onChange={event => setCpuOffload(event.target.checked)} /> CPU offload</label>
                <label className="flex items-center gap-1.5"><input type="checkbox" checked={flashvdm} onChange={event => setFlashvdm(event.target.checked)} /> FlashVDM</label>
                <label className="flex items-center gap-1.5"><input type="checkbox" checked={removeBackground} onChange={event => setRemoveBackground(event.target.checked)} /> Remove background</label>
                <label className="flex items-center gap-1.5"><input type="checkbox" checked={compile} onChange={event => setCompile(event.target.checked)} /> Torch compile</label>
                <label className="flex items-center gap-1.5"><input type="checkbox" checked={reduceFace} onChange={event => setReduceFace(event.target.checked)} /> Simplify mesh</label>
              </div>
              <p className="text-[9px] text-text-muted">Higher octree/texture resolutions and PBR consume substantially more VRAM. CPU offload is recommended while other Maestro models are in use.</p>
            </div>
          )}

          {job && (
            <div className={`rounded-lg border p-3 ${job.status === 'failed' ? 'border-red-500/30 bg-red-500/10' : 'border-border bg-bg-tertiary'}`}>
              <div className="flex items-center justify-between text-[10px]"><span className="text-text-secondary">{job.message}</span><span className="text-text-muted">{Math.round(job.progress * 100)}%</span></div>
              <div className="h-1.5 bg-bg-primary rounded-full overflow-hidden mt-2"><div className="h-full bg-accent-green transition-all" style={{ width: `${Math.max(2, job.progress * 100)}%` }} /></div>
              {job.error && <p className="text-[10px] text-red-300 mt-2 whitespace-pre-wrap max-h-24 overflow-y-auto">{job.error}</p>}
            </div>
          )}
          {error && <p className="text-[10px] text-red-400 whitespace-pre-wrap">{error}</p>}

          {isRunning ? (
            <button onClick={() => void cancel()} className="w-full px-4 py-2.5 rounded-lg flex items-center justify-center gap-1.5 bg-red-500/15 border border-red-500/30 text-red-300 hover:bg-red-500/25 text-xs font-medium"><Square size={13} /> Cancel 3D generation</button>
          ) : (
            <button disabled={!hasInput} onClick={() => void run()} className={`w-full px-4 py-2.5 rounded-lg flex items-center justify-center gap-1.5 text-xs font-medium transition-all ${hasInput ? 'bg-cta hover:brightness-110 shadow-accent-glow text-white' : 'bg-bg-tertiary border border-border text-text-muted cursor-not-allowed'}`}><Play size={13} fill={hasInput ? 'currentColor' : 'none'} /> Generate 3D asset</button>
          )}
          <p className="text-[9px] text-text-muted text-center">First use downloads only the selected checkpoint. The isolated worker exits after export.</p>
        </>
      )}
    </div>
  )
}
