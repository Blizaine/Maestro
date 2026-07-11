import { useRef, useState } from 'react'
import { Wrench, Upload, X, Film, Mic, Play, Box } from 'lucide-react'
import { useStore } from '../../stores/useStore'
import * as api from '../../api/client'

// Upscale methods — same set as Post Processing's Spatial Upsampling, minus the
// VAE options (those are tied to the generation pipeline, not a standalone clip).
const upscaleMethods = [
  { value: 'flashvsr2', label: 'FlashVSR 2x' },
  { value: 'flashvsr3', label: 'FlashVSR 3x' },
  { value: 'flashvsr4', label: 'FlashVSR 4x' },
  { value: 'flashvsr2pass2', label: 'FlashVSR Two Pass 2x' },
  { value: 'flashvsr2pass4', label: 'FlashVSR Two Pass 4x' },
  { value: 'lanczos1.5', label: 'Lanczos 1.5x (fast)' },
  { value: 'lanczos2', label: 'Lanczos 2x (fast)' },
]

export function ToolsPanel() {
  const tool = useStore(s => s.toolsTool)
  const setTool = useStore(s => s.setToolsTool)
  const sourcePath = useStore(s => s.toolsSourcePath)
  const sourceName = useStore(s => s.toolsSourceName)
  const sourceUrl = useStore(s => s.toolsSourceUrl)
  const setSource = useStore(s => s.setToolsSource)
  const method = useStore(s => s.toolsUpscaleMethod)
  const setMethod = useStore(s => s.setToolsUpscaleMethod)
  const revoiceMode = useStore(s => s.toolsRevoiceMode)
  const setRevoiceMode = useStore(s => s.setToolsRevoiceMode)
  const revoiceRefs = useStore(s => s.toolsRevoiceRefs)
  const setRevoiceRef = useStore(s => s.setToolsRevoiceRef)
  const runTool = useStore(s => s.runTool)
  const servicesConfig = useStore(s => s.servicesConfig)
  const loadOutputs = useStore(s => s.loadOutputs)

  const outputs = useStore(s => s.outputs)
  const selectedOutput = useStore(s => s.selectedOutput)
  const flashvsrMode = useStore(s => s.servicesConfig?.flashvsr_mode ?? 1)
  const current = outputs[selectedOutput]
  const currentIsVideo = !!current && current.type === 'video'

  const fileRef = useRef<HTMLInputElement>(null)
  const model3dFileRef = useRef<HTMLInputElement>(null)
  const vcFileRefs = [useRef<HTMLInputElement>(null), useRef<HTMLInputElement>(null)]
  const [uploading, setUploading] = useState(false)
  const [vcUploading, setVcUploading] = useState<number | null>(null)
  const [model3dPrompt, setModel3dPrompt] = useState('')
  const [model3dImage, setModel3dImage] = useState<{ path: string; name: string; url: string } | null>(null)
  const [model3dUploading, setModel3dUploading] = useState(false)
  const [model3dRunning, setModel3dRunning] = useState(false)
  const [model3dError, setModel3dError] = useState<string | null>(null)

  const handleSourceUpload = async (file: File) => {
    setUploading(true)
    try {
      const r = await api.uploadImage(file)  // /api/v1/upload handles video too
      setSource({ path: r.path, name: file.name, url: r.url })
    } catch (e) {
      console.error('Source upload failed:', e)
    } finally {
      setUploading(false)
    }
  }

  const handleModel3dImageUpload = async (file: File) => {
    setModel3dUploading(true)
    setModel3dError(null)
    try {
      const r = await api.uploadImage(file)
      setModel3dImage({ path: r.path, name: file.name, url: r.url })
    } catch (e) {
      setModel3dError(e instanceof Error ? e.message : 'Image upload failed')
    } finally {
      setModel3dUploading(false)
    }
  }

  const runModel3d = async () => {
    setModel3dRunning(true)
    setModel3dError(null)
    try {
      await api.generateModel3D({
        provider: servicesConfig?.model3d_provider || 'hunyuan3d',
        remote_url: servicesConfig?.model3d_remote_url || '',
        endpoint: servicesConfig?.model3d_endpoint || '/generate',
        prompt: model3dPrompt,
        image_path: model3dImage?.path,
        output_format: 'glb',
        texture: true,
        num_inference_steps: 30,
        guidance_scale: 5,
      })
      await loadOutputs()
    } catch (e) {
      setModel3dError(e instanceof Error ? e.message : '3D generation failed')
    } finally {
      setModel3dRunning(false)
    }
  }

  const useCurrentClip = () => {
    if (currentIsVideo) setSource({ path: current.name, name: current.name, url: current.url })
  }

  const handleVcUpload = async (index: number, file: File) => {
    setVcUploading(index)
    try {
      const r = await api.uploadAudio(file)
      setRevoiceRef(index, { filename: file.name, path: r.path })
    } catch (e) {
      console.error('Voice ref upload failed:', e)
    } finally {
      setVcUploading(null)
    }
  }

  const hasRefs = revoiceRefs.some(r => r && r.path)
  const canRun = !!sourcePath && (tool === 'upscale' || hasRefs)
  const flashvsrOff = flashvsrMode === 0 && method.startsWith('flashvsr')

  return (
    <div className="flex flex-col gap-4">
      <div>
        <div className="flex items-center gap-1.5 text-[11px] text-text-muted uppercase tracking-wider mb-2">
          <Wrench size={12} /> Tools — post-process any clip
        </div>
        <div className="bg-bg-tertiary border border-border rounded-lg p-3 space-y-3 mb-4">
          <div className="flex items-center justify-between gap-2">
            <div className="flex items-center gap-1.5 text-[11px] text-text-muted uppercase tracking-wider">
              <Box size={12} /> 3D Model
            </div>
            <span className="text-[10px] text-text-muted truncate">
              {servicesConfig?.model3d_provider || 'hunyuan3d'}
            </span>
          </div>
          <textarea
            value={model3dPrompt}
            onChange={e => setModel3dPrompt(e.target.value)}
            placeholder="Text prompt for text-to-3D providers..."
            rows={3}
            className="w-full bg-bg-primary border border-border rounded-lg px-3 py-2 text-xs text-text-primary resize-none focus:outline-none focus:border-accent-blue"
          />
          {model3dImage ? (
            <div className="flex items-center gap-2 bg-bg-primary border border-border rounded-lg px-2 py-1.5">
              <img src={model3dImage.url} alt="" className="w-8 h-8 rounded object-cover border border-border" />
              <span className="flex-1 min-w-0 truncate text-[11px] text-text-primary">{model3dImage.name}</span>
              <button onClick={() => setModel3dImage(null)} className="p-0.5 text-text-muted hover:text-red-400 transition-colors" title="Remove">
                <X size={12} />
              </button>
            </div>
          ) : (
            <div
              onClick={() => model3dFileRef.current?.click()}
              className={`border border-dashed border-border rounded-lg p-2 text-center cursor-pointer hover:border-accent-blue transition-colors ${model3dUploading ? 'opacity-50 pointer-events-none' : ''}`}
            >
              <Upload size={14} className="mx-auto mb-1 text-text-muted" />
              <p className="text-[11px] text-text-secondary">{model3dUploading ? 'Uploading...' : 'Optional image reference'}</p>
              <input
                ref={model3dFileRef}
                type="file"
                accept="image/*"
                className="hidden"
                onChange={e => { const f = e.target.files?.[0]; if (f) handleModel3dImageUpload(f) }}
              />
            </div>
          )}
          {model3dError && <p className="text-[10px] text-red-400 leading-snug">{model3dError}</p>}
          <button
            onClick={runModel3d}
            disabled={model3dRunning || (!model3dPrompt.trim() && !model3dImage) || !servicesConfig?.model3d_remote_url}
            className={`w-full px-3 py-2 rounded-lg flex items-center justify-center gap-1.5 font-medium text-xs transition-all ${
              model3dRunning || (!model3dPrompt.trim() && !model3dImage) || !servicesConfig?.model3d_remote_url
                ? 'bg-bg-primary text-text-muted cursor-not-allowed border border-border'
                : 'bg-cta hover:brightness-110 shadow-accent-glow text-white'
            }`}
          >
            <Play size={13} fill="currentColor" />
            {model3dRunning ? 'Generating 3D...' : 'Generate 3D Model'}
          </button>
        </div>
        {/* Tool selector */}
        <div className="flex bg-bg-tertiary rounded-lg p-0.5 border border-border">
          {([['upscale', 'Upscale'], ['revoice', 'Revoice']] as const).map(([val, label]) => (
            <button
              key={val}
              onClick={() => setTool(val)}
              className={`flex-1 text-xs py-2 rounded-md transition-all ${
                tool === val ? 'bg-bg-active text-text-primary' : 'text-text-secondary hover:text-text-primary'
              }`}
            >
              {label}
            </button>
          ))}
        </div>
      </div>

      {/* Source clip — upload, or use the clip currently selected in the gallery */}
      <div>
        <label className="text-[11px] text-text-muted uppercase tracking-wider mb-1.5 block">Source Clip</label>
        {sourcePath ? (
          <div className="bg-bg-tertiary border border-border rounded-lg p-2 space-y-2">
            {sourceUrl && (
              <video src={sourceUrl} className="w-full rounded-md max-h-44 bg-black" muted controls playsInline />
            )}
            <div className="flex items-center gap-2">
              <Film size={12} className="text-accent-blue shrink-0" />
              <span className="flex-1 min-w-0 truncate text-[11px] text-text-primary">{sourceName}</span>
              <button onClick={() => setSource(null)} className="p-0.5 text-text-muted hover:text-red-400 transition-colors" title="Clear">
                <X size={12} />
              </button>
            </div>
          </div>
        ) : (
          <div className="space-y-2">
            <div
              onClick={() => fileRef.current?.click()}
              className={`border-2 border-dashed border-border rounded-lg p-3 text-center cursor-pointer hover:border-accent-blue transition-colors ${uploading ? 'opacity-50 pointer-events-none' : ''}`}
            >
              <Upload size={16} className="mx-auto mb-1 text-text-muted" />
              <p className="text-[11px] text-text-secondary">{uploading ? 'Uploading...' : 'Upload a video clip'}</p>
              <input
                ref={fileRef}
                type="file"
                accept="video/*"
                className="hidden"
                onChange={e => { const f = e.target.files?.[0]; if (f) handleSourceUpload(f) }}
              />
            </div>
            <button
              onClick={useCurrentClip}
              disabled={!currentIsVideo}
              className="w-full text-[11px] py-1.5 rounded-md border border-border bg-bg-tertiary text-text-secondary hover:text-text-primary disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
            >
              {currentIsVideo ? 'Use selected gallery clip' : 'Select a video in the gallery first'}
            </button>
          </div>
        )}
      </div>

      {/* Tool params */}
      {tool === 'upscale' ? (
        <div>
          <label className="text-[11px] text-text-muted uppercase tracking-wider mb-1.5 block">Upscale Method</label>
          <select
            value={method}
            onChange={e => setMethod(e.target.value)}
            className="w-full bg-bg-tertiary border border-border rounded-lg px-3 py-2 text-sm text-text-primary focus:outline-none focus:border-accent-blue"
          >
            {upscaleMethods.map(o => <option key={o.value} value={o.value}>{o.label}</option>)}
          </select>
          {flashvsrOff && (
            <p className="text-[10px] text-amber-400 mt-1.5 leading-snug">
              FlashVSR is disabled in Settings → Services. Enable it, or pick a Lanczos method.
            </p>
          )}
          <p className="text-[10px] text-text-muted mt-1.5 leading-snug">
            FlashVSR is model-based super-resolution (sharper, slower; weights download on first use). Lanczos is a fast classic resize. The clip's audio is preserved.
          </p>
        </div>
      ) : (
        <div className="space-y-2">
          <label className="text-[11px] text-text-muted uppercase tracking-wider block">Replace Voice (SeedVC)</label>
          <div className="flex gap-1.5 text-xs">
            {([['single', 'Single Voice'], ['two', 'Two Voices']] as const).map(([val, label]) => (
              <button
                key={val}
                onClick={() => setRevoiceMode(val)}
                className={`flex-1 py-1.5 rounded-md border transition-colors ${
                  revoiceMode === val
                    ? 'bg-accent-blue/10 border-accent-blue text-text-primary'
                    : 'bg-bg-tertiary border-border text-text-secondary hover:text-text-primary'
                }`}
              >
                {label}
              </button>
            ))}
          </div>
          <p className="text-[10px] text-text-muted leading-snug">
            {revoiceMode === 'single'
              ? 'Replaces every voice in the clip with the reference voice.'
              : 'Auto-detects 2 speakers; preserves background music & silence. First detected → Voice A, second → Voice B.'}
          </p>
          {[0, ...(revoiceMode === 'two' ? [1] : [])].map(idx => {
            const ref = revoiceRefs[idx]
            const label = revoiceMode === 'two' ? (idx === 0 ? 'Voice A' : 'Voice B') : 'Reference Voice'
            return (
              <div key={idx}>
                <label className="text-[10px] text-text-muted uppercase tracking-wider mb-1 block">{label}</label>
                {!ref || !ref.path ? (
                  <div
                    onClick={() => vcFileRefs[idx].current?.click()}
                    className={`border-2 border-dashed border-border rounded-lg p-2 text-center cursor-pointer hover:border-accent-blue transition-colors ${vcUploading === idx ? 'opacity-50 pointer-events-none' : ''}`}
                  >
                    <p className="text-[11px] text-text-secondary">{vcUploading === idx ? 'Uploading...' : `Upload ${label.toLowerCase()} sample`}</p>
                    <input
                      ref={vcFileRefs[idx]}
                      type="file"
                      accept="audio/*,video/*"
                      className="hidden"
                      onChange={e => { const f = e.target.files?.[0]; if (f) handleVcUpload(idx, f) }}
                    />
                  </div>
                ) : (
                  <div className="flex items-center gap-2 bg-bg-tertiary border border-border rounded-lg px-2 py-1.5">
                    <Mic size={12} className="text-accent-blue shrink-0" />
                    <span className="flex-1 min-w-0 truncate text-[11px] text-text-primary">{ref.filename}</span>
                    <button onClick={() => setRevoiceRef(idx, null)} className="p-0.5 text-text-muted hover:text-red-400 transition-colors" title="Remove">
                      <X size={12} />
                    </button>
                  </div>
                )}
              </div>
            )
          })}
        </div>
      )}

      {/* Run */}
      <button
        onClick={() => runTool()}
        disabled={!canRun}
        className={`w-full px-4 py-2.5 rounded-lg flex items-center justify-center gap-1.5 font-medium text-xs transition-all ${
          canRun
            ? 'bg-cta hover:brightness-110 shadow-accent-glow text-white'
            : 'bg-bg-tertiary text-text-muted cursor-not-allowed border border-border'
        }`}
      >
        <Play size={13} fill={canRun ? 'white' : 'currentColor'} />
        {tool === 'upscale' ? 'Upscale Clip' : 'Replace Voice'}
      </button>
    </div>
  )
}
