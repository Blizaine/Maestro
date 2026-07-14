import { useState, useMemo, useRef } from 'react'
import { ChevronDown, ChevronRight, X, Mic } from 'lucide-react'
import { useStore } from '../../stores/useStore'
import * as api from '../../api/client'

const baseOptions = [
  { value: '', label: 'None' },
  { value: 'lanczos1.5', label: 'Lanczos 1.5x' },
  { value: 'lanczos2', label: 'Lanczos 2x' },
]

const vaeOptions = [
  { value: 'vae1', label: 'VAE 1x (Refine)' },
  { value: 'vae2', label: 'VAE 2x' },
]

// FlashVSR (DiT super-resolution) spatial upsampling — ported from upstream
// Wan2GP. These values route to the FlashVSR bridge in wgp.py's
// perform_spatial_upsampling(); the model auto-downloads (~GB) on first use.
// "Two Pass" = higher quality, ~2x slower.
const flashvsrOptions = [
  { value: 'flashvsr2', label: 'FlashVSR 2x' },
  { value: 'flashvsr3', label: 'FlashVSR 3x' },
  { value: 'flashvsr4', label: 'FlashVSR 4x' },
  { value: 'flashvsr2pass2', label: 'FlashVSR Two Pass 2x' },
  { value: 'flashvsr2pass4', label: 'FlashVSR Two Pass 4x' },
]

export function PostProcessing() {
  const [open, setOpen] = useState(false)
  const spatialUpsampling = useStore(s => s.spatialUpsampling)
  const setSpatialUpsampling = useStore(s => s.setSpatialUpsampling)
  const filmGrainIntensity = useStore(s => s.filmGrainIntensity)
  const setFilmGrainIntensity = useStore(s => s.setFilmGrainIntensity)
  const filmGrainSaturation = useStore(s => s.filmGrainSaturation)
  const setFilmGrainSaturation = useStore(s => s.setFilmGrainSaturation)
  // Voice clone (SeedVC) — postprocessing voice replacement on the
  // finished video. Only shown for video generation modes (the audio-only
  // generation modes have their own voice cloning paths built into the
  // model handlers themselves, e.g. Scenema's per-character voice refs).
  const voiceCloneEnabled = useStore(s => s.voiceCloneEnabled)
  const setVoiceCloneEnabled = useStore(s => s.setVoiceCloneEnabled)
  const voiceCloneMode = useStore(s => s.voiceCloneMode)
  const setVoiceCloneMode = useStore(s => s.setVoiceCloneMode)
  const voiceCloneRefs = useStore(s => s.voiceCloneRefs)
  const setVoiceCloneRef = useStore(s => s.setVoiceCloneRef)
  const generationMode = useStore(s => s.generationMode)
  const showVoiceClone = generationMode === 'video' || generationMode === 'avatar'
  const [vcUploading, setVcUploading] = useState<number | null>(null)
  const vcFileRefs = [useRef<HTMLInputElement>(null), useRef<HTMLInputElement>(null)]

  const handleVcUpload = async (index: number, file: File) => {
    setVcUploading(index)
    try {
      const result = await api.uploadAudio(file)
      setVoiceCloneRef(index, { filename: file.name, path: result.path })
    } catch (e) {
      console.error('Voice ref upload failed:', e)
    } finally {
      setVcUploading(null)
    }
  }
  // Compute showVae as a primitive boolean directly in the selector to avoid
  // unstable array references that cause infinite re-renders (React error #185)
  const showVae = useStore(s => {
    const modes = s.modelOptions?.vae_upsampler_modes
    if (!modes?.length) return false
    const effectiveMode = s.generationMode === 'image' ? 1 : s.params.image_mode
    return modes.includes(effectiveMode)
  })

  const upsamplingOptions = useMemo(
    () => [...baseOptions, ...flashvsrOptions, ...(showVae ? vaeOptions : [])],
    [showVae],
  )

  const hasVoiceClone = voiceCloneEnabled && voiceCloneRefs.some(r => r && r.path)
  const hasAny = spatialUpsampling !== '' || filmGrainIntensity > 0 || hasVoiceClone

  return (
    <div>
      <button
        onClick={() => setOpen(!open)}
        className="flex items-center gap-1.5 text-[11px] text-text-muted uppercase tracking-wider w-full hover:text-text-primary transition-colors"
      >
        {open ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
        <span className="flex-1 text-left">Post Processing</span>
        {hasAny && <span className="w-1.5 h-1.5 rounded-full bg-accent-blue" />}
      </button>

      {open && (
        <div className="mt-3 space-y-4">
          {/* Spatial Upsampling */}
          <div>
            <label className="text-[11px] text-text-muted uppercase tracking-wider mb-1.5 block">
              Spatial Upsampling
            </label>
            <select
              value={spatialUpsampling}
              onChange={e => setSpatialUpsampling(e.target.value)}
              className="w-full bg-bg-tertiary border border-border rounded-lg px-3 py-2 text-sm text-text-primary focus:outline-none focus:border-accent-blue"
            >
              {upsamplingOptions.map(opt => (
                <option key={opt.value} value={opt.value}>{opt.label}</option>
              ))}
            </select>
          </div>

          {/* Film Grain Intensity */}
          <div>
            <div className="flex items-center justify-between mb-1.5">
              <label className="text-[11px] text-text-muted uppercase tracking-wider">Film Grain Intensity</label>
              <span className="text-xs text-text-secondary">{filmGrainIntensity.toFixed(2)}</span>
            </div>
            <input
              type="range"
              min={0}
              max={1}
              step={0.01}
              value={filmGrainIntensity}
              onChange={e => setFilmGrainIntensity(parseFloat(e.target.value))}
            />
          </div>

          {/* Film Grain Saturation */}
          {filmGrainIntensity > 0 && (
            <div>
              <div className="flex items-center justify-between mb-1.5">
                <label className="text-[11px] text-text-muted uppercase tracking-wider">Film Grain Saturation</label>
                <span className="text-xs text-text-secondary">{filmGrainSaturation.toFixed(2)}</span>
              </div>
              <input
                type="range"
                min={0}
                max={1}
                step={0.01}
                value={filmGrainSaturation}
                onChange={e => setFilmGrainSaturation(parseFloat(e.target.value))}
              />
            </div>
          )}

          {/* Voice Clone (SeedVC) — only for video / avatar modes.
              Replaces 1 or 2 voices in the generated video's audio
              with user-supplied reference voice(s). Backend pipeline:
              app/postprocessing/voice_clone.py. */}
          {showVoiceClone && (
            <div className="border-t border-border pt-3 space-y-2">
              <div className="flex items-center justify-between">
                <label className="text-[11px] text-text-muted uppercase tracking-wider flex items-center gap-1.5">
                  <Mic size={11} /> Voice Clone (SeedVC)
                </label>
                <button
                  onClick={() => setVoiceCloneEnabled(!voiceCloneEnabled)}
                  className={`relative w-9 h-5 rounded-full transition-colors ${
                    voiceCloneEnabled ? 'bg-accent-blue' : 'bg-bg-tertiary border border-border'
                  }`}
                >
                  <span className={`absolute top-0.5 left-0.5 w-4 h-4 rounded-full bg-white border border-border transition-transform ${
                    voiceCloneEnabled ? 'translate-x-4' : ''
                  }`} />
                </button>
              </div>
              {voiceCloneEnabled && (
                <>
                  {/* Mode selector */}
                  <div className="flex gap-1.5 text-xs">
                    <button
                      onClick={() => setVoiceCloneMode('single')}
                      className={`flex-1 py-1.5 rounded-md border transition-colors ${
                        voiceCloneMode === 'single'
                          ? 'bg-accent-blue/10 border-accent-blue text-text-primary'
                          : 'bg-bg-tertiary border-border text-text-secondary hover:text-text-primary'
                      }`}
                    >
                      Single Voice
                    </button>
                    <button
                      onClick={() => setVoiceCloneMode('two')}
                      className={`flex-1 py-1.5 rounded-md border transition-colors ${
                        voiceCloneMode === 'two'
                          ? 'bg-accent-blue/10 border-accent-blue text-text-primary'
                          : 'bg-bg-tertiary border-border text-text-secondary hover:text-text-primary'
                      }`}
                    >
                      Two Voices
                    </button>
                  </div>
                  <p className="text-[10px] text-text-muted leading-snug">
                    {voiceCloneMode === 'single'
                      ? 'Replaces every voice in the audio with the reference voice. Affects the whole audio track.'
                      : 'Auto-detects 2 speakers; preserves background music & silence. First detected speaker → Voice A, second → Voice B.'}
                  </p>

                  {/* Voice reference upload(s) */}
                  {[0, ...(voiceCloneMode === 'two' ? [1] : [])].map(idx => {
                    const ref = voiceCloneRefs[idx]
                    const label = voiceCloneMode === 'two' ? (idx === 0 ? 'Voice A' : 'Voice B') : 'Reference Voice'
                    return (
                      <div key={idx}>
                        <label className="text-[10px] text-text-muted uppercase tracking-wider mb-1 block">{label}</label>
                        {!ref || !ref.path ? (
                          <div
                            onClick={() => vcFileRefs[idx].current?.click()}
                            className={`border-2 border-dashed border-border rounded-lg p-2 text-center cursor-pointer hover:border-accent-blue transition-colors ${
                              vcUploading === idx ? 'opacity-50 pointer-events-none' : ''
                            }`}
                          >
                            <p className="text-[11px] text-text-secondary">
                              {vcUploading === idx ? 'Uploading...' : `Upload ${label.toLowerCase()} sample`}
                            </p>
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
                            <button
                              onClick={() => setVoiceCloneRef(idx, null)}
                              className="p-0.5 text-text-muted hover:text-red-400 transition-colors"
                            >
                              <X size={12} />
                            </button>
                          </div>
                        )}
                      </div>
                    )
                  })}
                </>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  )
}
