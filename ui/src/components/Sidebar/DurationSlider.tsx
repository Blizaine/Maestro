import { useEffect } from 'react'
import { Lock, Unlock } from 'lucide-react'
import { useStore } from '../../stores/useStore'
import {
  effectiveH3OmniSequenceFrames,
  recommendedH3PassProfile,
  recommendedH3OmniSequenceProfile,
} from '../../lib/h3Memory'

export const formatSeconds = (seconds: number) => {
  const rounded = Math.round(seconds * 10) / 10
  return Number.isInteger(rounded) ? `${rounded}s` : `${rounded.toFixed(1)}s`
}

// Kept as a public alias because Director shares the same pass-level table.
export const recommendedWindowProfile = recommendedH3PassProfile

export function DurationSlider() {
  const duration = useStore(s => s.durationSeconds)
  const setDuration = useStore(s => s.setDurationSeconds)
  const windowSize = useStore(s => s.slidingWindowSeconds)
  const setWindowSize = useStore(s => s.setSlidingWindowSeconds)
  const overlap = useStore(s => s.slidingWindowOverlap)
  const locked = useStore(s => s.slidingWindowLocked)
  const modelOptions = useStore(s => s.modelOptions)
  const omniReferenceSequence = useStore(s => (
    s.modelOptions?.omni_reference === true
    && s.params.minimax_h3_reference_sequence === true
  ))
  const resolution = useStore(s => s.params.resolution)
  const totalVramGb = useStore(s => s.systemStats?.gpu.vram_total_gb ?? 0)
  const fps = modelOptions?.fps ?? 16
  const swDefaults = (modelOptions as Record<string, unknown> | null)?.sliding_window_defaults as Record<string, number> | undefined
  const supportsSlidingWindows = modelOptions?.sliding_window === true
  const minimumFrames = modelOptions?.frames_minimum ?? Math.round(fps)
  const maximumFrames = modelOptions?.frames_maximum ?? Math.round(300 * fps)
  const frameStep = modelOptions?.frames_steps ?? Math.round(fps)
  const memoryPolicy = omniReferenceSequence
    ? modelOptions?.omni_sequence_memory_policy
    : modelOptions?.sliding_window_memory_policy
  const windowRecommendation = omniReferenceSequence
    ? recommendedH3OmniSequenceProfile(
        memoryPolicy,
        resolution,
        totalVramGb,
        minimumFrames,
        maximumFrames,
        frameStep,
      )
    : recommendedWindowProfile(memoryPolicy, resolution, totalVramGb)
  const safeWindowFrames = windowRecommendation?.frames ?? null
  const unsupportedAutoResolution = windowRecommendation?.supported === false
  const nativeMinSeconds = modelOptions?.frames_minimum
    ? modelOptions.frames_minimum / fps
    : 1
  const nativeMaxSeconds = modelOptions?.frames_maximum
    ? modelOptions.frames_maximum / fps
    : null
  const minDuration = Math.max(1, nativeMinSeconds)
  const maxDuration = omniReferenceSequence
    ? 120
    : (!supportsSlidingWindows && nativeMaxSeconds ? nativeMaxSeconds : 300)
  const durationStep = nativeMaxSeconds ? 0.1 : 1
  const discardFrames = swDefaults?.discard_last_frames ?? 0
  const overlapSeconds = overlap / fps
  const discardSeconds = discardFrames / fps
  const stride = windowSize - discardSeconds - overlapSeconds
  const windowCount = stride > 0 && duration > windowSize
    ? 1 + Math.ceil((duration - windowSize + discardSeconds) / stride)
    : 1
  const showSlidingWindow = supportsSlidingWindows && duration > windowSize
  const { frames: omniSequenceClipFrames } = effectiveH3OmniSequenceFrames({
    policy: modelOptions?.omni_sequence_memory_policy,
    resolution,
    totalVramGb,
    minimumFrames,
    maximumFrames,
    frameStep,
    selectedFrames: Math.round(windowSize * fps),
    manualOverride: locked,
  })
  const totalFrames = Math.max(1, Math.round(duration * fps))
  const omniSequenceClipCount = omniReferenceSequence
    ? Math.max(1, Math.ceil(totalFrames / Math.max(1, omniSequenceClipFrames)))
    : 1
  const showOmniSequence = omniReferenceSequence && omniSequenceClipCount > 1

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
    if (omniReferenceSequence) {
      if (locked || safeWindowFrames == null) return
      const nextWindowSize = safeWindowFrames / fps
      if (Math.abs(nextWindowSize - windowSize) > 0.0001) {
        setWindowSize(nextWindowSize)
      }
      return
    }
    if (!supportsSlidingWindows || locked) return

    let nextWindowSize: number
    if (swDefaults) {
      const windowMin = (swDefaults.window_min ?? Math.round(3 * fps)) / fps
      const windowMax = (swDefaults.window_max ?? Math.round(40 * fps)) / fps
      const automaticWindowMax = Math.min(
        windowMax,
        unsupportedAutoResolution
          ? windowMin
          : (safeWindowFrames != null ? safeWindowFrames / fps : windowMax),
      )
      const nativeBuffer = (swDefaults.window_step ?? fps) / fps
      nextWindowSize = Math.min(
        automaticWindowMax,
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
  }, [duration, locked, supportsSlidingWindows, omniReferenceSequence, maxDuration, fps, swDefaults, safeWindowFrames, unsupportedAutoResolution, windowSize, setDuration, setWindowSize])

  const imageMode = useStore(s => s.params.image_mode)
  const isMultiClip = imageMode === 2
  const promptLineCount = useStore(s => s.params.prompt.split('\n').filter((l: string) => l.trim()).length)
  const automaticPromptPacing = (
    modelOptions?.sliding_window_auto_prompt_pacing === true
  )

  return (
    <div>
      <div className="flex items-center justify-between mb-1.5">
        <label className="text-[11px] text-text-muted uppercase tracking-wider">Duration</label>
        <span className="text-xs text-text-secondary">
          {duration >= 60 ? `${Math.floor(duration / 60)}m${duration % 60 ? ` ${Math.round(duration % 60)}s` : ''}` : formatSeconds(duration)}
          {showSlidingWindow && (
            <span className="text-text-muted ml-1">({windowCount} win)</span>
          )}
          {showOmniSequence && (
            <span className="text-text-muted ml-1">({omniSequenceClipCount} clips)</span>
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
          {windowCount} windows of {formatSeconds(windowSize)} &middot;{' '}
          {automaticPromptPacing
            ? 'full prompt auto-paced'
            : <>{promptLineCount}/{windowCount} prompts{promptLineCount < windowCount && ' (last reused)'}</>}
        </div>
      )}
      {showOmniSequence && (
        <div className="text-[10px] text-text-muted mt-1">
          {omniSequenceClipCount} independent Omni clips &middot;{' '}
          {locked ? 'manual' : 'Auto'} max {formatSeconds(omniSequenceClipFrames / fps)} &middot; joined automatically
        </div>
      )}
      {unsupportedAutoResolution && (
        <div className="text-[10px] text-amber-400 mt-1">
          {locked
            ? `Manual VRAM override: ${resolution} may run out of memory on this ${totalVramGb.toFixed(0)} GB GPU.`
            : `For ${totalVramGb.toFixed(0)} GB, H3 Auto recommends ${windowRecommendation?.fallbackResolution ?? 'a lower resolution'} instead of ${resolution}. Lock ${omniReferenceSequence ? 'Sequence Clip Length' : 'Window Size'} in Advanced to override.`}
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
  const omniReferenceSequence = useStore(s => (
    s.modelOptions?.omni_reference === true
    && s.params.minimax_h3_reference_sequence === true
  ))
  const resolution = useStore(s => s.params.resolution)
  const totalVramGb = useStore(s => s.systemStats?.gpu.vram_total_gb ?? 0)
  const isOutpaint = generationMode === 'avatar' && editSubMode === 'outpaint'
  const trimmedOutpaintDuration = outpaintTrimEnd > outpaintTrimStart
    ? outpaintTrimEnd - outpaintTrimStart
    : editVideoDuration
  const duration = isOutpaint ? trimmedOutpaintDuration : studioDuration

  const fps = modelOptions?.fps ?? 16
  const swDefaults = (modelOptions as Record<string, unknown> | null)?.sliding_window_defaults as Record<string, number> | undefined
  const supportsSlidingWindows = modelOptions?.sliding_window === true
  const minimumFrames = omniReferenceSequence
    ? (modelOptions?.frames_minimum ?? Math.round(3 * fps))
    : (swDefaults?.window_min ?? Math.round(3 * fps))
  const maximumFrames = omniReferenceSequence
    ? (modelOptions?.frames_maximum ?? Math.round(15 * fps))
    : (swDefaults?.window_max ?? Math.round(40 * fps))
  const frameStep = omniReferenceSequence
    ? (modelOptions?.frames_steps ?? fps)
    : (swDefaults?.window_step ?? fps)
  const windowMinSeconds = minimumFrames / fps
  const windowMaxSeconds = maximumFrames / fps
  const windowStepSeconds = Math.max(1, frameStep) / fps
  const overlapMin = swDefaults?.overlap_min ?? 1
  const overlapMax = swDefaults?.overlap_max ?? 97
  const overlapStep = swDefaults?.overlap_step ?? 4
  const discardFrames = swDefaults?.discard_last_frames ?? 0
  const overlapSeconds = overlap / fps
  const discardSeconds = discardFrames / fps
  const stride = windowSize - discardSeconds - overlapSeconds
  const windowCount = omniReferenceSequence
    ? Math.max(1, Math.ceil(
        Math.max(1, Math.round(duration * fps))
        / Math.max(1, Math.round(windowSize * fps)),
      ))
    : (stride > 0 && duration > windowSize
        ? 1 + Math.ceil((duration - windowSize + discardSeconds) / stride)
        : 1)
  const showSlidingWindow = duration > windowSize
  const memoryPolicy = omniReferenceSequence
    ? modelOptions?.omni_sequence_memory_policy
    : modelOptions?.sliding_window_memory_policy
  const windowRecommendation = omniReferenceSequence
    ? recommendedH3OmniSequenceProfile(
        memoryPolicy,
        resolution,
        totalVramGb,
        minimumFrames,
        maximumFrames,
        frameStep,
      )
    : recommendedWindowProfile(memoryPolicy, resolution, totalVramGb)
  const safeWindowFrames = windowRecommendation?.frames ?? null
  const safeWindowSeconds = safeWindowFrames != null
    ? safeWindowFrames / fps
    : null
  const unsupportedAutoResolution = windowRecommendation?.supported === false
  const exceedsSafeRecommendation = (
    locked
    && safeWindowSeconds != null
    && windowSize > safeWindowSeconds + 0.0001
  )

  if (!supportsSlidingWindows && !omniReferenceSequence) return null

  return (
    <div className="space-y-3">
      <div>
        <div className="flex items-center justify-between mb-1.5">
          <div className="flex items-center gap-1.5">
            <label className="text-[11px] text-text-muted uppercase tracking-wider">
              {omniReferenceSequence ? 'Sequence Clip Length' : 'Window Size'}
            </label>
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
              title={locked
                ? `${omniReferenceSequence ? 'Sequence clip length' : 'Window size'} locked - click to resume Auto`
                : `Click to override Auto ${omniReferenceSequence ? 'clip length' : 'window size'}`}
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
            {windowCount} {omniReferenceSequence ? 'independent clip' : 'window'}{windowCount > 1 ? 's' : ''} of up to {formatSeconds(windowSize)}
          </div>
        )}
        {windowRecommendation != null && (
          <div className={`text-[10px] mt-1 ${unsupportedAutoResolution || exceedsSafeRecommendation ? 'text-amber-400' : 'text-text-muted'}`}>
            {unsupportedAutoResolution
              ? (locked
                ? `Manual override enabled: ${resolution} is above the automatic profile for ${totalVramGb.toFixed(0)} GB and may run out of VRAM.`
                : `Auto does not recommend ${resolution} on ${totalVramGb.toFixed(0)} GB. Choose ${windowRecommendation.fallbackResolution ?? 'a lower resolution'}, or lock ${omniReferenceSequence ? 'Sequence Clip Length' : 'Window Size'} to try it experimentally.`)
              : (exceedsSafeRecommendation
                ? `Manual override exceeds the ${formatSeconds(safeWindowSeconds!)} recommendation for ${totalVramGb.toFixed(0)} GB at this resolution and may run out of VRAM.`
                : `Auto max: ${formatSeconds(safeWindowSeconds!)} for ${totalVramGb.toFixed(0)} GB at this resolution.${omniReferenceSequence && (windowRecommendation.referenceMarginFrames ?? 0) > 0 ? ' Includes Ref2VA reference headroom.' : ''}`)}
          </div>
        )}
      </div>

      {supportsSlidingWindows && showSlidingWindow && overlapStep > 0 && (
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
          {modelOptions?.sliding_window_audio_history === true && (
            <div className="text-[10px] text-text-muted mt-1">
              Carries recent motion and matching stereo audio into each new window. 18 frames is recommended.
            </div>
          )}
        </div>
      )}
    </div>
  )
}
