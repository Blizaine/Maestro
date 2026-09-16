import { useEffect, useState } from 'react'
import { useStore, resolveResolution } from '../../stores/useStore'
import { fetchDirectorMusicClipLimits, fetchModelOptions, type DirectorMusicClipLimits } from '../../api/client'
import { formatSeconds } from './DurationSlider'

export function DirectorMusicClipLength({disabled = false}: {disabled?: boolean}) {
  const model = useStore(s => s.selectedModelPerMode.video || 'ltx2_22B_distilled_1_1')
  const resolution = useStore(s => s.directorResolution)
  const aspect = useStore(s => s.directorAspectRatio)
  const manualFrames = useStore(s => s.directorVideoMaxShotFramesByModel[model])
  const seconds = useStore(s => s.directorMusicClipSeconds)
  const setSeconds = useStore(s => s.setDirectorMusicClipSeconds)
  const analysis = useStore(s => s.directorAnalysis)
  const duration = useStore(s => s.directorSongDuration)
  const seamless = useStore(s => s.directorSeamless)
  const replan = useStore(s => s.directorSetEnergyBias)
  const bias = useStore(s => s.directorEnergyBias)
  const loading = useStore(s => s.directorLoading)
  const [appliedSeconds, setAppliedSeconds] = useState(seconds)
  const limitsKey = JSON.stringify([model, resolution, aspect, manualFrames])
  const [result, setResult] = useState<{key: string; limits?: DirectorMusicClipLimits; error?: string} | null>(null)
  const limits = result?.key === limitsKey ? result.limits : undefined
  const error = result?.key === limitsKey ? result.error : ''

  useEffect(() => {
    let cancelled = false
    fetchModelOptions(model).then(options => fetchDirectorMusicClipLimits({
      video_model: model, video_params: {resolution: resolveResolution(options, resolution, aspect)},
      director_max_shot_frames: manualFrames, director_music_clip_seconds: null,
    })).then(value => {if (!cancelled) setResult({key: limitsKey, limits: value})})
      .catch(reason => {if (!cancelled) setResult({key: limitsKey, error: reason.message})})
    return () => {cancelled = true}
  }, [model, resolution, aspect, manualFrames, limitsKey])

  const automatic = seconds === null
  const maximumFrames = limits ? Math.min(limits.hard_max_frames, manualFrames || limits.hard_max_frames) : 0
  const frames = !limits ? 0 : automatic ? limits.max_frames
    : limits.frames_minimum + Math.max(0, Math.floor((Math.min(seconds * limits.fps, maximumFrames) - limits.frames_minimum + 1e-6) / limits.frame_step)) * limits.frame_step
  const effective = limits ? frames / limits.fps : 0
  const locked = disabled || seamless || !limits
  return (
    <div role="group" aria-label="Music video clip length" className="space-y-1.5 border-t border-border/50 pt-3">
      <div className="flex items-center justify-between gap-2">
        <span className="text-[11px] text-text-secondary">Clip length</span>
        <div className="flex items-center gap-2">
          <button type="button" aria-pressed={automatic} disabled={locked}
            onClick={() => setSeconds(automatic ? effective : null)}
            className={`rounded-md border px-2 py-1 text-[11px] disabled:opacity-50 ${automatic ? 'border-accent-blue bg-accent-blue/15 text-text-primary' : 'border-border text-text-muted'}`}>Auto</button>
          <span className="min-w-12 text-right text-xs text-text-secondary">{effective ? formatSeconds(effective) : '…'}</span>
        </div>
      </div>
      {limits && <input type="range" aria-label="Maximum music video clip length"
        min={limits.frames_minimum} max={maximumFrames} step={limits.frame_step} value={frames}
        disabled={locked} className={`w-full ${automatic || locked ? 'opacity-40' : ''}`}
        onPointerDown={() => {if (automatic && !locked) setSeconds(effective)}}
        onKeyDown={event => {if (automatic && !locked && ['ArrowLeft', 'ArrowRight', 'Home', 'End', 'PageUp', 'PageDown'].includes(event.key)) setSeconds(effective)}}
        onChange={event => setSeconds(Number(event.target.value) / limits.fps)} />}
      <p className="text-[10px] leading-relaxed text-text-muted">
        {seamless ? 'Separate-clip length is available when Seamless is off.' : effective
          ? <>Maximum per clip; cuts follow the music. At least {Math.ceil((analysis?.duration || duration) / effective)} clips.
            {!automatic && limits && <> Recommended {formatSeconds(limits.recommended_seconds)}.</>}</>
          : error || 'Reading model limits…'}
      </p>
      {!seamless && <p className="text-[10px] leading-relaxed text-text-muted">Shorter clips can reduce peak memory but need more preparation.</p>}
      {analysis && seconds !== appliedSeconds && !disabled && !seamless && (
        <button type="button" disabled={loading}
          className="text-[11px] text-accent-blue hover:underline disabled:opacity-50"
          onClick={async () => {await replan(bias); if (!useStore.getState().directorError) setAppliedSeconds(seconds)}}>
          {loading ? 'Updating clips…' : 'Update clip layout'}
        </button>
      )}
    </div>
  )
}
