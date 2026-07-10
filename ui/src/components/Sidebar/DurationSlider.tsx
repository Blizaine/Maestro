import { useEffect } from 'react'
import { Lock, Unlock } from 'lucide-react'
import { useStore } from '../../stores/useStore'

export function DurationSlider() {
  const duration = useStore(s => s.durationSeconds)
  const setDuration = useStore(s => s.setDurationSeconds)
  const windowSize = useStore(s => s.slidingWindowSeconds)
  const setWindowSize = useStore(s => s.setSlidingWindowSeconds)
  const overlap = useStore(s => s.slidingWindowOverlap)
  const locked = useStore(s => s.slidingWindowLocked)
  const modelOptions = useStore(s => s.modelOptions)

  const fps = modelOptions?.fps ?? 16
  const swDefaults = (modelOptions as Record<string, unknown> | null)?.sliding_window_defaults as Record<string, number> | undefined
  const discardFrames = swDefaults?.discard_last_frames ?? 0
  const overlapSeconds = Math.round((overlap / fps) * 10) / 10
  const discardSeconds = Math.round((discardFrames / fps) * 10) / 10
  const stride = windowSize - discardSeconds - overlapSeconds
  const windowCount = stride > 0 && duration > windowSize
    ? 1 + Math.ceil((duration - windowSize + discardSeconds) / stride)
    : 1
  const showSlidingWindow = duration > windowSize

  // Auto-track: window size follows duration with a +1s buffer until
  // duration exceeds 20s (unless locked).
  //
  // The +1s buffer is the fix for an observed bug: when duration was
  // set EXACTLY equal to sliding window size, wgp's internal latent-
  // step quantization could land video_length one step ABOVE
  // sliding_window_size after rounding, causing a single-window clip
  // to split into two windows and produce a stutter at the boundary.
  // Adding a small buffer guarantees sliding_window stays comfortably
  // above video_length after quantization. The cost — user sees
  // "Window: 20s" for a 19s clip — is trivial; the benefit is
  // single-window generation always works as intended.
  useEffect(() => {
    if (!locked) {
      if (duration <= 20) {
        setWindowSize(duration + 1)
      } else if (windowSize < 10) {
        setWindowSize(20)
      }
    }
  }, [duration, locked]) // eslint-disable-line react-hooks/exhaustive-deps

  const imageMode = useStore(s => s.params.image_mode)
  const isMultiClip = imageMode === 2
  const promptLineCount = useStore(s => s.params.prompt.split('\n').filter((l: string) => l.trim()).length)

  return (
    <div>
      <div className="flex items-center justify-between mb-1.5">
        <label className="text-[11px] text-text-muted uppercase tracking-wider">Duration</label>
        <span className="text-xs text-text-secondary">
          {duration >= 60 ? `${Math.floor(duration / 60)}m${duration % 60 ? ` ${duration % 60}s` : ''}` : `${duration}s`}
          {showSlidingWindow && (
            <span className="text-text-muted ml-1">({windowCount} win)</span>
          )}
        </span>
      </div>
      <input
        type="range"
        min={1}
        max={300}
        step={1}
        value={duration}
        onChange={e => setDuration(Number(e.target.value))}
      />
      {showSlidingWindow && !isMultiClip && (
        <div className="text-[10px] text-text-muted mt-1">
          {windowCount} windows of {windowSize}s &middot; {promptLineCount}/{windowCount} prompts
          {promptLineCount < windowCount && ' (last reused)'}
        </div>
      )}
    </div>
  )
}

/** Exposed for Advanced Settings popup */
export function WindowSettings() {
  const duration = useStore(s => s.durationSeconds)
  const windowSize = useStore(s => s.slidingWindowSeconds)
  const setWindowSize = useStore(s => s.setSlidingWindowSeconds)
  const overlap = useStore(s => s.slidingWindowOverlap)
  const setOverlap = useStore(s => s.setSlidingWindowOverlap)
  const locked = useStore(s => s.slidingWindowLocked)
  const setLocked = useStore(s => s.setSlidingWindowLocked)
  const modelOptions = useStore(s => s.modelOptions)

  const fps = modelOptions?.fps ?? 16
  const swDefaults = (modelOptions as Record<string, unknown> | null)?.sliding_window_defaults as Record<string, number> | undefined
  const overlapMin = swDefaults?.overlap_min ?? 1
  const overlapMax = swDefaults?.overlap_max ?? 97
  const overlapStep = swDefaults?.overlap_step ?? 4
  const discardFrames = swDefaults?.discard_last_frames ?? 0
  const overlapSeconds = Math.round((overlap / fps) * 10) / 10
  const discardSeconds = Math.round((discardFrames / fps) * 10) / 10
  const stride = windowSize - discardSeconds - overlapSeconds
  const windowCount = stride > 0 && duration > windowSize
    ? 1 + Math.ceil((duration - windowSize + discardSeconds) / stride)
    : 1
  const showSlidingWindow = duration > windowSize

  return (
    <div className="space-y-3">
      <div>
        <div className="flex items-center justify-between mb-1.5">
          <div className="flex items-center gap-1.5">
            <label className="text-[11px] text-text-muted uppercase tracking-wider">Window Size</label>
            <button
              onClick={() => {
                if (locked) {
                  // Unlocking — let auto-track resume
                  setLocked(false)
                } else {
                  // Locking — freeze current window size
                  setLocked(true)
                }
              }}
              className={`p-0.5 rounded transition-colors ${
                locked
                  ? 'text-accent-blue hover:text-accent-blue/70'
                  : 'text-text-muted hover:text-text-secondary'
              }`}
              title={locked ? 'Window size locked — click to unlock (auto-track)' : 'Click to lock window size'}
            >
              {locked ? <Lock size={10} /> : <Unlock size={10} />}
            </button>
          </div>
          <span className="text-xs text-text-secondary">
            {windowSize}s
            {locked && <span className="text-accent-blue/60 ml-1 text-[9px]">locked</span>}
          </span>
        </div>
        <input
          type="range"
          min={3}
          max={40}
          step={1}
          value={windowSize}
          onChange={e => {
            setWindowSize(Number(e.target.value))
            // Any manual change to window size automatically locks it
            if (!locked) setLocked(true)
          }}
        />
        {showSlidingWindow && (
          <div className="text-[10px] text-text-muted mt-1">
            {windowCount} window{windowCount > 1 ? 's' : ''} of {windowSize}s
          </div>
        )}
      </div>

      {showSlidingWindow && overlapStep > 0 && (
        <div>
          <div className="flex items-center justify-between mb-1.5">
            <label className="text-[11px] text-text-muted uppercase tracking-wider">Window Overlap</label>
            <span className="text-xs text-text-secondary">{overlap}f ({overlapSeconds}s)</span>
          </div>
          <input
            type="range"
            min={overlapMin}
            max={overlapMax}
            step={overlapStep || 1}
            value={overlap}
            onChange={e => setOverlap(Number(e.target.value))}
          />
        </div>
      )}
    </div>
  )
}
