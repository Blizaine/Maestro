import { useEffect } from 'react'
import { Lock, Unlock } from 'lucide-react'
import { useStore } from '../../stores/useStore'

const formatSeconds = (seconds: number) => {
  const rounded = Math.round(seconds * 10) / 10
  return Number.isInteger(rounded) ? `${rounded}s` : `${rounded.toFixed(1)}s`
}

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
  const supportsSlidingWindows = modelOptions?.sliding_window === true
  const nativeMinSeconds = modelOptions?.frames_minimum
    ? modelOptions.frames_minimum / fps
    : 1
  const nativeMaxSeconds = modelOptions?.frames_maximum
    ? modelOptions.frames_maximum / fps
    : null
  const minDuration = Math.max(1, nativeMinSeconds)
  const maxDuration = !supportsSlidingWindows && nativeMaxSeconds
    ? nativeMaxSeconds
    : 300
  const durationStep = nativeMaxSeconds ? 0.1 : 1
  const discardFrames = swDefaults?.discard_last_frames ?? 0
  const overlapSeconds = overlap / fps
  const discardSeconds = discardFrames / fps
  const stride = windowSize - discardSeconds - overlapSeconds
  const windowCount = stride > 0 && duration > windowSize
    ? 1 + Math.ceil((duration - windowSize + discardSeconds) / stride)
    : 1
  const showSlidingWindow = supportsSlidingWindows && duration > windowSize

  // Auto-track: window size follows duration with a small model-native
  // buffer until it reaches that model's declared per-window ceiling.
  //
  // A one-native-step buffer fixes an observed bug: when duration was
  // set EXACTLY equal to sliding window size, wgp's internal latent-
  // step quantization could land video_length one step ABOVE
  // sliding_window_size after rounding, causing a single-window clip
  // to split into two windows and produce a stutter at the boundary.
  // The small buffer guarantees sliding_window stays comfortably
  // above video_length after quantization. The cost — user sees
  // "Window: 20s" for a 19s clip — is trivial; the benefit is
  // single-window generation always works as intended.
  useEffect(() => {
    if (duration > maxDuration) {
      setDuration(maxDuration)
      return
    }
    if (!supportsSlidingWindows || locked) return

    let nextWindowSize: number
    if (swDefaults) {
      const windowMin = (swDefaults.window_min ?? Math.round(3 * fps)) / fps
      const windowMax = (swDefaults.window_max ?? Math.round(40 * fps)) / fps
      const nativeBuffer = (swDefaults.window_step ?? fps) / fps
      nextWindowSize = Math.min(
        windowMax,
        Math.max(windowMin, duration + nativeBuffer),
      )
    } else if (duration <= 20) {
      nextWindowSize = duration + 1
    } else if (windowSize < 10) {
      nextWindowSize = 20
    } else {
      return
    }
    if (Math.abs(nextWindowSize - windowSize) > 0.0001) {
      setWindowSize(nextWindowSize)
    }
  }, [duration, locked, supportsSlidingWindows, maxDuration, fps, swDefaults, windowSize, setDuration, setWindowSize])

  const imageMode = useStore(s => s.params.image_mode)
  const isMultiClip = imageMode === 2
  const promptLineCount = useStore(s => s.params.prompt.split('\n').filter((l: string) => l.trim()).length)

  return (
    <div>
      <div className="flex items-center justify-between mb-1.5">
        <label className="text-[11px] text-text-muted uppercase tracking-wider">Duration</label>
        <span className="text-xs text-text-secondary">
          {duration >= 60 ? `${Math.floor(duration / 60)}m${duration % 60 ? ` ${Math.round(duration % 60)}s` : ''}` : formatSeconds(duration)}
          {showSlidingWindow && (
            <span className="text-text-muted ml-1">({windowCount} win)</span>
          )}
        </span>
      </div>
      <input
        type="range"
        min={minDuration}
        max={maxDuration}
        step={durationStep}
        value={duration}
        onChange={e => setDuration(Number(e.target.value))}
      />
      {showSlidingWindow && !isMultiClip && (
        <div className="text-[10px] text-text-muted mt-1">
          {windowCount} windows of {formatSeconds(windowSize)} &middot; {promptLineCount}/{windowCount} prompts
          {promptLineCount < windowCount && ' (last reused)'}
        </div>
      )}
    </div>
  )
}

/** Exposed for Advanced Settings popup */
export function WindowSettings() {
  const studioDuration = useStore(s => s.durationSeconds)
  const generationMode = useStore(s => s.generationMode)
  const editSubMode = useStore(s => s.editSubMode)
  const outpaintTrimStart = useStore(s => s.outpaintTrimStart)
  const outpaintTrimEnd = useStore(s => s.outpaintTrimEnd)
  const editVideoDuration = useStore(s => s.editVideoDuration)
  const windowSize = useStore(s => s.slidingWindowSeconds)
  const setWindowSize = useStore(s => s.setSlidingWindowSeconds)
  const overlap = useStore(s => s.slidingWindowOverlap)
  const setOverlap = useStore(s => s.setSlidingWindowOverlap)
  const locked = useStore(s => s.slidingWindowLocked)
  const setLocked = useStore(s => s.setSlidingWindowLocked)
  const modelOptions = useStore(s => s.modelOptions)
  const isOutpaint = generationMode === 'avatar' && editSubMode === 'outpaint'
  const trimmedOutpaintDuration = outpaintTrimEnd > outpaintTrimStart
    ? outpaintTrimEnd - outpaintTrimStart
    : editVideoDuration
  const duration = isOutpaint ? trimmedOutpaintDuration : studioDuration

  const fps = modelOptions?.fps ?? 16
  const swDefaults = (modelOptions as Record<string, unknown> | null)?.sliding_window_defaults as Record<string, number> | undefined
  const supportsSlidingWindows = modelOptions?.sliding_window === true
  const windowMinSeconds = (swDefaults?.window_min ?? Math.round(3 * fps)) / fps
  const windowMaxSeconds = (swDefaults?.window_max ?? Math.round(40 * fps)) / fps
  const windowStepSeconds = Math.max(1, swDefaults?.window_step ?? fps) / fps
  const overlapMin = swDefaults?.overlap_min ?? 1
  const overlapMax = swDefaults?.overlap_max ?? 97
  const overlapStep = swDefaults?.overlap_step ?? 4
  const discardFrames = swDefaults?.discard_last_frames ?? 0
  const overlapSeconds = overlap / fps
  const discardSeconds = discardFrames / fps
  const stride = windowSize - discardSeconds - overlapSeconds
  const windowCount = stride > 0 && duration > windowSize
    ? 1 + Math.ceil((duration - windowSize + discardSeconds) / stride)
    : 1
  const showSlidingWindow = duration > windowSize

  if (!supportsSlidingWindows) return null

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
            {formatSeconds(windowSize)}
            {locked && <span className="text-accent-blue/60 ml-1 text-[9px]">locked</span>}
          </span>
        </div>
        <input
          type="range"
          min={windowMinSeconds}
          max={windowMaxSeconds}
          step={windowStepSeconds}
          value={windowSize}
          onChange={e => {
            setWindowSize(Number(e.target.value))
            // Any manual change to window size automatically locks it
            if (!locked) setLocked(true)
          }}
        />
        {showSlidingWindow && (
          <div className="text-[10px] text-text-muted mt-1">
            {windowCount} window{windowCount > 1 ? 's' : ''} of {formatSeconds(windowSize)}
          </div>
        )}
      </div>

      {showSlidingWindow && overlapStep > 0 && (
        <div>
          <div className="flex items-center justify-between mb-1.5">
            <label className="text-[11px] text-text-muted uppercase tracking-wider">Window Overlap</label>
            <span className="text-xs text-text-secondary">{overlap}f ({formatSeconds(overlapSeconds)})</span>
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
