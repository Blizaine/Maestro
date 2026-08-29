import { useEffect, useRef, useState } from 'react'
import { Wrench, Upload, X, Film, Image as ImageIcon, Mic, Play } from 'lucide-react'
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

const imageUpscaleMethods = [
  { value: 'flashvsr2', label: 'FlashVSR 2x (AI detail)' },
  { value: 'lanczos1.5', label: 'Lanczos 1.5x (fast)' },
  { value: 'lanczos2', label: 'Lanczos 2x (fast)' },
  { value: 'lanczos3', label: 'Lanczos 3x (fast)' },
  { value: 'lanczos4', label: 'Lanczos 4x (fast)' },
]

export function ToolsPanel({
  forcedTool,
  mediaKind = 'video',
  embedded = false,
}: {
  forcedTool?: 'upscale' | 'film_grain' | 'revoice'
  mediaKind?: 'image' | 'video'
  embedded?: boolean
}) {
  const storedTool = useStore(s => s.toolsTool)
  const setTool = useStore(s => s.setToolsTool)
  const storedUpscaleMedia = useStore(s => s.toolsUpscaleMedia)
  const setUpscaleMedia = useStore(s => s.setToolsUpscaleMedia)
  const sourcePath = useStore(s => s.toolsSourcePath)
  const sourceName = useStore(s => s.toolsSourceName)
  const sourceUrl = useStore(s => s.toolsSourceUrl)
  const setSource = useStore(s => s.setToolsSource)
  const method = useStore(s => s.toolsUpscaleMethod)
  const setMethod = useStore(s => s.setToolsUpscaleMethod)
  const grainIntensity = useStore(s => s.filmGrainIntensity)
  const setGrainIntensity = useStore(s => s.setFilmGrainIntensity)
  const grainSaturation = useStore(s => s.filmGrainSaturation)
  const setGrainSaturation = useStore(s => s.setFilmGrainSaturation)
  const revoiceMode = useStore(s => s.toolsRevoiceMode)
  const setRevoiceMode = useStore(s => s.setToolsRevoiceMode)
  const revoiceRefs = useStore(s => s.toolsRevoiceRefs)
  const setRevoiceRef = useStore(s => s.setToolsRevoiceRef)

  const outputs = useStore(s => s.outputs)
  const selectedOutput = useStore(s => s.selectedOutput)
  const flashvsrMode = useStore(s => s.servicesConfig?.flashvsr_mode ?? 1)
  const current = outputs[selectedOutput]
  const currentMatchesMedia = !!current && current.type === mediaKind

  const fileRef = useRef<HTMLInputElement>(null)
  const vcFileRefs = [useRef<HTMLInputElement>(null), useRef<HTMLInputElement>(null)]
  const [uploading, setUploading] = useState(false)
  const [vcUploading, setVcUploading] = useState<number | null>(null)
  const tool = forcedTool || storedTool

  useEffect(() => {
    if (forcedTool && storedTool !== forcedTool) setTool(forcedTool)
  }, [forcedTool, setTool, storedTool])

  useEffect(() => {
    if (tool === 'upscale' && storedUpscaleMedia !== mediaKind) {
      setUpscaleMedia(mediaKind)
      setSource(null)
    }
  }, [mediaKind, setSource, setUpscaleMedia, storedUpscaleMedia, tool])

  // Video exposes several temporal/two-pass FlashVSR variants that do not
  // apply to a still image. If the user moves from Video Upscale to Image
  // Upscale, select the nearest valid still-image method instead of leaving
  // a hidden, invalid value behind the dropdown.
  useEffect(() => {
    if (tool !== 'upscale') return
    const choices = mediaKind === 'image' ? imageUpscaleMethods : upscaleMethods
    if (!choices.some(choice => choice.value === method)) {
      setMethod('flashvsr2')
    }
  }, [mediaKind, method, setMethod, tool])

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

  const useCurrentClip = () => {
    if (currentMatchesMedia && current) setSource({ path: current.name, name: current.name, url: current.url })
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
  const canRun = !!sourcePath && (
    tool === 'upscale'
    || (tool === 'film_grain' && grainIntensity > 0)
    || (tool === 'revoice' && hasRefs)
  )
  const flashvsrOff = flashvsrMode === 0 && method.startsWith('flashvsr')

  const handleRun = () => {
    // setToolsTool is synchronous; reading from the store immediately after
    // it guarantees runTool dispatches the workflow displayed by this panel.
    if (storedTool !== tool) setTool(tool)
    void useStore.getState().runTool()
  }

  return (
    <div className="flex flex-col gap-4">
      {!embedded && (
      <div>
        <div className="flex items-center gap-1.5 text-[11px] text-text-muted uppercase tracking-wider mb-2">
          <Wrench size={12} /> Tools — post-process any clip
        </div>
        {/* Tool selector */}
        <div className="flex bg-bg-tertiary rounded-lg p-0.5 border border-border">
          {([['upscale', 'Upscale'], ['film_grain', 'Film Grain'], ['revoice', 'Revoice']] as const).map(([val, label]) => (
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
      )}

      {/* Source media — upload, or use the matching gallery selection. */}
      <div>
        <label className="text-[11px] text-text-muted uppercase tracking-wider mb-1.5 block">
          Source {mediaKind === 'image' ? 'Image' : 'Clip'}
        </label>
        {sourcePath ? (
          <div className="bg-bg-tertiary border border-border rounded-lg p-2 space-y-2">
            {sourceUrl && mediaKind === 'video' && (
              <video src={sourceUrl} className="w-full rounded-md max-h-44 bg-black" muted controls playsInline />
            )}
            {sourceUrl && mediaKind === 'image' && (
              <img src={sourceUrl} alt="Upscale source" className="w-full rounded-md max-h-56 object-contain bg-black" />
            )}
            <div className="flex items-center gap-2">
              {mediaKind === 'image'
                ? <ImageIcon size={12} className="text-accent-blue shrink-0" />
                : <Film size={12} className="text-accent-blue shrink-0" />}
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
              <p className="text-[11px] text-text-secondary">
                {uploading ? 'Uploading...' : `Upload ${mediaKind === 'image' ? 'an image' : 'a video clip'}`}
              </p>
              <input
                ref={fileRef}
                type="file"
                accept={mediaKind === 'image' ? 'image/*' : 'video/*'}
                className="hidden"
                onChange={e => { const f = e.target.files?.[0]; if (f) handleSourceUpload(f) }}
              />
            </div>
            <button
              onClick={useCurrentClip}
              disabled={!currentMatchesMedia}
              className="w-full text-[11px] py-1.5 rounded-md border border-border bg-bg-tertiary text-text-secondary hover:text-text-primary disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
            >
              {currentMatchesMedia
                ? `Use selected gallery ${mediaKind}`
                : `Select ${mediaKind === 'image' ? 'an image' : 'a video'} in the gallery first`}
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
            {(mediaKind === 'image' ? imageUpscaleMethods : upscaleMethods).map(o => (
              <option key={o.value} value={o.value}>{o.label}</option>
            ))}
          </select>
          {flashvsrOff && (
            <p className="text-[10px] text-indicator-warning mt-1.5 leading-snug">
              FlashVSR is disabled in Settings → Services. Enable it, or pick a Lanczos method.
            </p>
          )}
          <p className="text-[10px] text-text-muted mt-1.5 leading-snug">
            FlashVSR is model-based super-resolution (sharper, slower; weights download on first use). Lanczos is a fast classic resize.
            {mediaKind === 'video' ? " The clip's audio is preserved." : ''}
          </p>
        </div>
      ) : tool === 'film_grain' ? (
        <div className="space-y-4">
          <div>
            <div className="flex items-center justify-between mb-1.5">
              <label className="text-[11px] text-text-muted uppercase tracking-wider">Grain Intensity</label>
              <span className="text-xs text-text-secondary">{grainIntensity.toFixed(2)}</span>
            </div>
            <input
              type="range"
              min={0}
              max={1}
              step={0.01}
              value={grainIntensity}
              onChange={event => setGrainIntensity(Number(event.target.value))}
            />
          </div>
          <div>
            <div className="flex items-center justify-between mb-1.5">
              <label className="text-[11px] text-text-muted uppercase tracking-wider">Grain Saturation</label>
              <span className="text-xs text-text-secondary">{grainSaturation.toFixed(2)}</span>
            </div>
            <input
              type="range"
              min={0}
              max={1}
              step={0.01}
              value={grainSaturation}
              onChange={event => setGrainSaturation(Number(event.target.value))}
            />
          </div>
          <p className="text-[10px] text-text-muted leading-snug">
            Adds film grain to the full clip as a new finished copy. The original video and its audio are preserved.
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
        onClick={handleRun}
        disabled={!canRun}
        className={`w-full px-4 py-2.5 rounded-lg flex items-center justify-center gap-1.5 font-medium text-xs transition-all ${
          canRun
            ? 'bg-cta hover:brightness-110 shadow-accent-glow text-white'
            : 'bg-bg-tertiary text-text-muted cursor-not-allowed border border-border'
        }`}
      >
        <Play size={13} fill={canRun ? 'white' : 'currentColor'} />
        {tool === 'upscale'
          ? `Upscale ${mediaKind === 'image' ? 'Image' : 'Clip'}`
          : tool === 'film_grain'
            ? 'Apply Film Grain'
            : 'Replace Voice'}
      </button>
    </div>
  )
}
