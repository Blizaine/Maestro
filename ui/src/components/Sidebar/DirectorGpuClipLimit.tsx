import { useEffect, useState } from 'react'
import { fetchDirectorMusicClipLimits, type DirectorMusicClipLimits } from '../../api/client'
import { resolveResolution, useStore } from '../../stores/useStore'
import type { ModelOptions } from '../../types'
import { formatSeconds } from './DurationSlider'

export function DirectorGpuClipLimit({model, options}: {model: string; options: ModelOptions | null}) {
  const resolution = useStore(s => s.directorResolution)
  const aspect = useStore(s => s.directorAspectRatio)
  const frames = useStore(s => s.directorVideoMaxShotFramesByModel[model])
  const setFrames = useStore(s => s.setDirectorVideoMaxShotFrames)
  const [result, setResult] = useState<{key: string; limits?: DirectorMusicClipLimits; error?: string} | null>(null)
  const canvas = resolveResolution(options, resolution, aspect)
  const key = JSON.stringify([model, canvas, frames])
  const bounded = Boolean(options?.frames_maximum && (options?.architecture?.startsWith('minimax_h3') || options?.omni_reference
    || options?.director_memory_policy || options?.sliding_window_memory_policy))
  useEffect(() => {
    if (!bounded) return
    let cancelled = false
    fetchDirectorMusicClipLimits({video_model: model, video_params: {resolution: canvas},
      director_max_shot_frames: frames, director_music_clip_seconds: null})
      .then(limits => {if (!cancelled) setResult({key, limits})})
      .catch(error => {if (!cancelled) setResult({key, error: error.message})})
    return () => {cancelled = true}
  }, [bounded, model, canvas, frames, key])
  if (!bounded || !options) return null
  const limits = result?.key === key ? result.limits : undefined
  const fps = limits?.fps || options.fps || 24
  const minimum = limits?.frames_minimum || options.frames_minimum || 1
  const maximum = limits?.hard_max_frames || options.frames_maximum || minimum
  const step = Math.max(1, limits?.frame_step || options.frames_steps || 1)
  const choices = Array.from({length: Math.floor((maximum - minimum) / step) + 1}, (_, i) => minimum + i * step)
  const recommended = limits?.recommended_frames
  return <div role="group" aria-label="GPU clip limit" className="space-y-1">
    <div className="flex items-center justify-between gap-2">
      <label htmlFor="director-gpu-clip-limit" className="text-[11px] text-text-secondary">GPU clip limit</label>
      <select id="director-gpu-clip-limit" value={frames ?? ''}
        onChange={event => setFrames(model, event.target.value ? Number(event.target.value) : null)}
        className="bg-bg-tertiary border border-border rounded px-1.5 py-0.5 text-[11px] text-text-primary focus:outline-none focus:border-accent-blue">
        <option value="">Auto{recommended ? ` · ${formatSeconds(recommended / fps)}` : ''}</option>
        {choices.map(value => <option key={value} value={value}>{formatSeconds(value / fps)}</option>)}
      </select>
    </div>
    <p className="text-[10px] leading-relaxed text-text-muted">Maximum generated duration per shot. Music-led cuts can be shorter. Your choice is remembered for this model.</p>
    {frames && recommended && frames > recommended
      ? <p className="text-[10px] text-amber-400">Above Auto’s {formatSeconds(recommended / fps)} recommendation; uses more GPU memory.</p>
      : result?.key === key && result.error && <p className="text-[10px] text-amber-400">{result.error}</p>}
  </div>
}
