/* eslint-disable react-refresh/only-export-components -- Director imports the shared H3 duration helpers */
import { useEffect, useRef } from 'react'
import { Lock, Save, Unlock } from 'lucide-react'
import { useStore } from '../../stores/useStore'
import {
  continuationFirstWindowSeconds,
  durationWindowPlan,
  LONG_FORM_MAX_SECONDS,
} from '../../lib/durationPlanning'
import {
  effectiveH3OmniSequenceFrames,
  h3OmniSequenceWindowCount,
  h3TimelineFrames,
  h3WindowOverrideKey,
  normalizeH3NativeFrames,
  recommendedH3PassProfile,
  recommendedH3OmniSequenceProfile,
} from '../../lib/h3Memory'
import { DurationPresetControl } from './DurationPresetControl'

export const formatSeconds = (seconds: number) => {
  const rounded = Math.round(seconds * 10) / 10
  return Number.isInteger(rounded) ? `${rounded}s` : `${rounded.toFixed(1)}s`
}

// Kept as a public alias because Director shares the same pass-level table.
export const recommendedWindowProfile = recommendedH3PassProfile

export function DurationSlider() {
  const duration = useStore(s => s.durationSeconds)
  const setDuration = useStore(s => s.setDurationSeconds)
  const setParam = useStore(s => s.setParam)
  const windowSize = useStore(s => s.slidingWindowSeconds)
  const setWindowSize = useStore(s => s.setSlidingWindowSeconds)
  const overlap = useStore(s => s.slidingWindowOverlap)
  const locked = useStore(s => s.slidingWindowLocked)
  const setLocked = useStore(s => s.setSlidingWindowLocked)
  const modelOptions = useStore(s => s.modelOptions)
  const omniReferenceSequence = useStore(s => (
    s.modelOptions?.omni_reference === true
    && s.params.minimax_h3_reference_sequence === true
  ))
  const nativeOmniContinuation = useStore(s => (
    s.params.minimax_h3_sequence_continuity !== false
  ))
  const manualOmniPrompts = useStore(s => (
    s.params.minimax_h3_sequence_prompt_mode === 'manual'
  ))
  const resolution = useStore(s => s.params.resolution)
  const modelType = useStore(s => s.params.model_type)
  const studioVideoWorkflow = useStore(s => s.studioVideoWorkflow)
  const prompt = useStore(s => s.params.prompt)
  const h3OriginalPrompt = useStore(s => s.params._h3_original_prompt)
  const ltxOriginalPrompt = useStore(s => s.params._ltx_original_prompt)
  const durationPlanningMode = useStore(s => s.params._duration_planning_mode ?? 'auto')
  // Keep the selector snapshot referentially stable when no references exist.
  // Returning a new [] here on every Zustand getSnapshot call can make React
  // repeatedly re-render the Studio shell before it ever becomes visible.
  const h3References = useStore(s => s.params.minimax_h3_references)
  const audioGuide = useStore(s => s.params.audio_guide)
  const videoGuide = useStore(s => s.params.video_guide)
  const h3SequencePromptMode = useStore(s => s.params.minimax_h3_sequence_prompt_mode)
  const ltxWindowPromptMode = useStore(s => s.params.ltx_window_prompt_mode)
  const h3FirstLastMultiWindow = useStore(s => s.params.minimax_h3_multi_window === true)
  const manualFirstLastPrompts = useStore(s => s.params.minimax_h3_window_storyboard === false)
  const ltxMultiWindow = useStore(s => s.params.ltx_multi_window === true)
  const manualLtxPrompts = useStore(s => s.params.ltx_window_prompt_mode === 'manual')
  const h3WindowOverrides = useStore(s => s.h3WindowOverrides)
  const totalVramGb = useStore(s => s.systemStats?.gpu.vram_total_gb ?? 0)
  const fps = modelOptions?.fps ?? 16
  const swDefaults = (modelOptions as Record<string, unknown> | null)?.sliding_window_defaults as Record<string, number> | undefined
  const supportsSlidingWindows = modelOptions?.sliding_window === true
  const minimumFrames = modelOptions?.frames_minimum ?? Math.round(fps)
  const maximumFrames = modelOptions?.frames_maximum ?? Math.round(300 * fps)
  const frameStep = modelOptions?.frames_steps ?? Math.round(fps)
  const isOmniReference = modelOptions?.omni_reference === true
  const isH3 = String(modelOptions?.architecture || '').startsWith('minimax_h3')
  const isLtx = modelOptions?.multi_window_sequence_controls === true
  // Prompt Enhance replaces the visible runtime prompt with a much longer
  // model-native document. Auto duration must keep sizing the user's story,
  // not count Context-IR field prose (or its tag instructions) as new action
  // and dialogue. The immutable source is retained until the user edits the
  // prompt, at which point setParam('prompt') deliberately clears it.
  const durationPlanningPrompt = String(
    (isH3 && typeof h3OriginalPrompt === 'string' && h3OriginalPrompt.trim())
      ? h3OriginalPrompt
      : (isLtx && typeof ltxOriginalPrompt === 'string' && ltxOriginalPrompt.trim())
        ? ltxOriginalPrompt
        : prompt,
  )
  const h3MultiWindowEnabled = isOmniReference
    ? omniReferenceSequence
    : h3FirstLastMultiWindow
  const rollingSequenceEnabled = isH3
    ? h3MultiWindowEnabled
    : isLtx
      ? ltxMultiWindow
      : supportsSlidingWindows
  const overrideKey = h3WindowOverrideKey(modelType, resolution)
  const savedOverrideFrames = isH3 ? h3WindowOverrides[overrideKey] : undefined
  const directOmni = isOmniReference && !omniReferenceSequence
  const memoryPolicy = isOmniReference
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
  const isVideoExtend = studioVideoWorkflow === 'extend' && supportsSlidingWindows
  const minDuration = Math.max(
    1,
    isVideoExtend
      ? continuationFirstWindowSeconds(nativeMinSeconds, overlap, fps)
      : nativeMinSeconds,
  )
  const maxDuration = isH3 || isLtx || supportsSlidingWindows
    ? LONG_FORM_MAX_SECONDS
    : Math.max(minDuration, nativeMaxSeconds ?? LONG_FORM_MAX_SECONDS)
  const discardFrames = swDefaults?.discard_last_frames ?? 0
  const overlapSeconds = overlap / fps
  const discardSeconds = discardFrames / fps
  const firstWindowSeconds = isVideoExtend
    ? continuationFirstWindowSeconds(windowSize, overlap, fps)
    : windowSize
  const durationPlan = durationWindowPlan(
    duration,
    windowSize,
    overlapSeconds,
    discardSeconds,
    firstWindowSeconds,
  )
  const windowCount = rollingSequenceEnabled ? durationPlan.windowCount : 1
  const showSlidingWindow = supportsSlidingWindows
    && rollingSequenceEnabled
    && windowCount > 1
    && !omniReferenceSequence
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
  const totalFrames = h3TimelineFrames(
    duration,
    fps,
    modelOptions?.frames_maximum,
  )
  const omniSequenceClipCount = omniReferenceSequence
    ? h3OmniSequenceWindowCount({
        totalFrames,
        windowFrames: omniSequenceClipFrames,
        overlapFrames: overlap,
        nativeContinuation: nativeOmniContinuation,
      })
    : 1
  const showOmniSequence = omniReferenceSequence && omniSequenceClipCount > 1
  const previousH3DefaultSelection = useRef<string | null>(null)
  const previousH3Recommendation = useRef<number | null>(null)
  const h3DefaultAppliedThisPass = useRef(false)

  // Sequence mode is a consequence of timeline length, not a separate user
  // choice. This also repairs older sidecars whose saved duration exceeded a
  // native pass but whose legacy multi-window checkbox was off.
  useEffect(() => {
    const shouldSequence = durationPlan.windowCount > 1
    if (isLtx && ltxMultiWindow !== shouldSequence) {
      setParam('ltx_multi_window', shouldSequence)
    } else if (isH3 && isOmniReference && omniReferenceSequence !== shouldSequence) {
      setParam('minimax_h3_reference_sequence', shouldSequence)
    } else if (isH3 && !isOmniReference && h3FirstLastMultiWindow !== shouldSequence) {
      setParam('minimax_h3_multi_window', shouldSequence)
    }
  }, [durationPlan.windowCount, h3FirstLastMultiWindow, isH3, isLtx, isOmniReference, ltxMultiWindow, omniReferenceSequence, setParam])

  // A saved override belongs to one exact model/canvas pair. Switching model
  // or resolution starts both the visible Duration and native Window Length
  // at that saved value, or at Auto's VRAM recommendation. The recommendation
  // is a starting point, not a ceiling; subsequent user movement does not
  // retrigger this initialization.
  useEffect(() => {
    h3DefaultAppliedThisPass.current = false
    if (!isH3) {
      previousH3DefaultSelection.current = null
      previousH3Recommendation.current = null
      return
    }
    const selectionChanged = previousH3DefaultSelection.current !== overrideKey
    const recommendationChanged = (
      savedOverrideFrames == null
      && previousH3Recommendation.current !== safeWindowFrames
    )
    previousH3DefaultSelection.current = overrideKey
    previousH3Recommendation.current = safeWindowFrames
    if (!selectionChanged && !recommendationChanged) return

    const preferredFrames = savedOverrideFrames ?? safeWindowFrames
    if (preferredFrames != null) {
      const preferredSeconds = preferredFrames / fps
      if (Math.abs(preferredSeconds - windowSize) > 0.0001) {
        setWindowSize(preferredSeconds)
      }
      // Enabling a long-form H3 sequence changes the memory recommendation,
      // but that recommendation is the per-window size—not a replacement for
      // an established total timeline (for example, a drive-audio track).
      // Model/resolution changes still initialize both values as before.
      const shouldInitializeTotalDuration = selectionChanged || !h3MultiWindowEnabled
      if (
        shouldInitializeTotalDuration
        && Math.abs(preferredSeconds - duration) > 0.0001
      ) {
        setDuration(preferredSeconds)
      }
    }
    setLocked(savedOverrideFrames != null)
    h3DefaultAppliedThisPass.current = true
  }, [
    isH3,
    overrideKey,
    savedOverrideFrames,
    safeWindowFrames,
    fps,
    duration,
    windowSize,
    setDuration,
    setWindowSize,
    setLocked,
    h3MultiWindowEnabled,
  ])

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
    if (h3DefaultAppliedThisPass.current) {
      h3DefaultAppliedThisPass.current = false
      return
    }
    if (duration > maxDuration) {
      setDuration(maxDuration)
      return
    }
    if (isH3) {
      if (locked || savedOverrideFrames != null || safeWindowFrames == null) return
      const nextWindowSize = safeWindowFrames / fps
      if (Math.abs(nextWindowSize - windowSize) > 0.0001) {
        setWindowSize(nextWindowSize)
      }
      return
    }
    // LTX long-form Auto follows Duration up to the model's native window
    // ceiling. This keeps a 58-second timeline at roughly three 20-second
    // passes instead of preserving a short one-window value and creating a
    // dozen tiny passes. Moving Window Length in Advanced locks it, so an
    // intentionally shorter user-selected pass still remains untouched.
    if (omniReferenceSequence) {
      if (locked || safeWindowFrames == null) return
      const nextWindowSize = safeWindowFrames / fps
      if (Math.abs(nextWindowSize - windowSize) > 0.0001) {
        setWindowSize(nextWindowSize)
      }
      return
    }
    if (!supportsSlidingWindows || locked) return

    // Auto Duration already derives its recommendation from the current
    // native window. Feeding a one-window recommendation back into LTX's
    // duration-following window control creates a circular update:
    //
    //   duration = window -> window = duration + one step -> repeat
    //
    // LTX-2.5's 10.0s default needs roughly 33 synchronous updates to reach
    // its 20.9s ceiling, which trips React's maximum-update-depth guard and
    // blanks the entire UI during startup (GitHub #97). A one-window Auto plan
    // therefore stays at its stable native default. A genuine multi-window
    // Auto plan jumps directly to the safe ceiling below, retaining the
    // efficient long-window behavior without a chain of nested updates.
    if (
      durationPlanningMode === 'auto'
      && duration <= windowSize + 0.05
    ) return

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
      nextWindowSize = durationPlanningMode === 'auto'
        ? automaticWindowMax
        : Math.min(
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
  }, [duration, durationPlanningMode, locked, supportsSlidingWindows, omniReferenceSequence, maxDuration, fps, swDefaults, safeWindowFrames, unsupportedAutoResolution, windowSize, setDuration, setWindowSize, isH3, isLtx, ltxMultiWindow, savedOverrideFrames, overrideKey])

  const imageMode = useStore(s => s.params.image_mode)
  const isMultiClip = imageMode === 2
  const promptLineCount = prompt.split('\n').filter((line: string) => line.trim()).length
  const automaticPromptPacing = (
    (modelOptions?.sliding_window_auto_prompt_pacing === true
      && !manualFirstLastPrompts)
    || (isLtx && ltxMultiWindow && !manualLtxPrompts)
  )
  const manualWindowPrompts = (
    (isOmniReference && manualOmniPrompts)
    || (!isOmniReference && isH3 && manualFirstLastPrompts)
    || (isLtx && manualLtxPrompts)
  )
  const creativeWindowPlanning = (
    (isH3 && h3SequencePromptMode === 'creative')
    || (isLtx && ltxWindowPromptMode === 'creative')
  )
  const driveReference = h3References?.find(reference => (
    reference.type === 'audio' && reference.audio_intent === 'drive'
  ))
  const driveDuration = Number(driveReference?.duration_seconds)
  const hasTimedGuide = Boolean(audioGuide || videoGuide)
  const autoSourceSeconds = Number.isFinite(driveDuration) && driveDuration > 0
    ? driveDuration
    : hasTimedGuide ? duration : null
  const autoSourceLabel = Number.isFinite(driveDuration) && driveDuration > 0
    ? 'music / performance timeline'
    : videoGuide ? 'control video' : audioGuide ? 'audio track' : undefined

  return (
    <div>
      <DurationPresetControl
        value={duration}
        onChange={setDuration}
        minSeconds={minDuration}
        maxSeconds={maxDuration}
        windowSeconds={windowSize}
        firstWindowSeconds={firstWindowSeconds}
        overlapSeconds={overlapSeconds}
        discardSeconds={discardSeconds}
        showSingleWindow={isH3 || isLtx || supportsSlidingWindows}
        enablePlanningModes={isH3 || isLtx || supportsSlidingWindows}
        planningMode={durationPlanningMode}
        onPlanningModeChange={mode => setParam('_duration_planning_mode', mode)}
        autoPrompt={durationPlanningPrompt}
        autoPlanningStyle={creativeWindowPlanning ? 'creative' : 'faithful'}
        autoSourceSeconds={autoSourceSeconds}
        autoSourceLabel={autoSourceLabel}
        autoManualWindowCount={manualWindowPrompts ? Math.max(1, promptLineCount) : null}
      />
      {showSlidingWindow && !isMultiClip && (
        <div className="text-[10px] text-text-muted mt-1">
          {windowCount} windows of {formatSeconds(windowSize)} &middot;{' '}
          {automaticPromptPacing
            ? (isLtx ? 'AI-planned window prompts' : 'full prompt auto-paced')
            : <span className={promptLineCount === windowCount ? '' : 'text-amber-400'}>
                {promptLineCount}/{windowCount} prompts
              </span>}
        </div>
      )}
      {showOmniSequence && (
        <div className="text-[10px] text-text-muted mt-1">
          {omniSequenceClipCount} {nativeOmniContinuation ? 'native Omni windows' : 'independent Omni clips'} &middot;{' '}
          {locked ? 'manual' : 'Auto'} max {formatSeconds(omniSequenceClipFrames / fps)} &middot;{' '}
          {nativeOmniContinuation ? 'motion + audio carried' : 'hard cuts joined'} &middot;{' '}
          {manualOmniPrompts
            ? <span className={promptLineCount === omniSequenceClipCount ? '' : 'text-amber-400'}>
                {promptLineCount}/{omniSequenceClipCount} manual prompts
              </span>
            : 'AI-planned prompts'}
        </div>
      )}
      {unsupportedAutoResolution && (
        <div className="text-[10px] text-amber-400 mt-1">
          {directOmni
            ? `For ${totalVramGb.toFixed(0)} GB, H3 Omni Auto recommends ${windowRecommendation?.fallbackResolution ?? 'a lower resolution'} instead of ${resolution}. Multi-window sequence can divide longer output into VRAM-aware windows.`
            : locked
            ? `Manual VRAM override: ${resolution} may run out of memory on this ${totalVramGb.toFixed(0)} GB GPU.`
            : `For ${totalVramGb.toFixed(0)} GB, H3 Auto recommends ${windowRecommendation?.fallbackResolution ?? 'a lower resolution'} instead of ${resolution}. Lock Window Length in Advanced to override.`}
        </div>
      )}
      {directOmni && !unsupportedAutoResolution && safeWindowFrames != null && nativeMaxSeconds != null && safeWindowFrames / fps < nativeMaxSeconds && (
        <div className="text-[10px] text-text-muted mt-1">
          VRAM-aware default: {formatSeconds(safeWindowFrames / fps)}. You can manually raise the native pass to {formatSeconds(nativeMaxSeconds)}; longer timelines use Multi-window sequence.
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
  const h3WindowOverrides = useStore(s => s.h3WindowOverrides)
  const saveH3WindowOverride = useStore(s => s.saveH3WindowOverride)
  const clearH3WindowOverride = useStore(s => s.clearH3WindowOverride)
  const modelOptions = useStore(s => s.modelOptions)
  const omniReferenceSequence = useStore(s => (
    s.modelOptions?.omni_reference === true
    && s.params.minimax_h3_reference_sequence === true
  ))
  const nativeOmniContinuation = useStore(s => (
    s.params.minimax_h3_sequence_continuity !== false
  ))
  const h3FirstLastMultiWindow = useStore(s => s.params.minimax_h3_multi_window === true)
  const ltxMultiWindow = useStore(s => s.params.ltx_multi_window === true)
  const modelType = useStore(s => s.params.model_type)
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
  const isH3 = String(modelOptions?.architecture || '').startsWith('minimax_h3')
  const isLtx = modelOptions?.multi_window_sequence_controls === true
  const isOmniReference = modelOptions?.omni_reference === true
  const h3MultiWindowEnabled = isOmniReference
    ? omniReferenceSequence
    : h3FirstLastMultiWindow
  const rollingSequenceEnabled = isH3
    ? h3MultiWindowEnabled
    : isLtx
      ? ltxMultiWindow
      : false
  const minimumFrames = isH3
    ? (modelOptions?.frames_minimum ?? 124)
    : omniReferenceSequence
    ? (modelOptions?.frames_minimum ?? Math.round(3 * fps))
    : (swDefaults?.window_min ?? Math.round(3 * fps))
  const maximumFrames = isH3
    ? (modelOptions?.frames_maximum ?? 345)
    : omniReferenceSequence
    ? (modelOptions?.frames_maximum ?? Math.round(15 * fps))
    : (swDefaults?.window_max ?? Math.round(40 * fps))
  const frameStep = isH3
    ? (modelOptions?.frames_steps ?? 17)
    : omniReferenceSequence
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
  const windowCount = !rollingSequenceEnabled && (isH3 || isLtx)
    ? 1
    : omniReferenceSequence
    ? h3OmniSequenceWindowCount({
        totalFrames: h3TimelineFrames(
          duration,
          fps,
          modelOptions?.frames_maximum,
        ),
        windowFrames: Math.max(1, Math.round(windowSize * fps)),
        overlapFrames: overlap,
        nativeContinuation: nativeOmniContinuation,
      })
    : (stride > 0 && duration > windowSize
        ? 1 + Math.ceil((duration - windowSize + discardSeconds) / stride)
        : 1)
  const showSlidingWindow = rollingSequenceEnabled && duration > windowSize
  const memoryPolicy = isOmniReference
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
  const overrideKey = h3WindowOverrideKey(modelType, resolution)
  const savedOverrideFrames = isH3 ? h3WindowOverrides[overrideKey] : undefined
  const currentWindowFrames = normalizeH3NativeFrames(
    Math.round(windowSize * fps),
    minimumFrames,
    maximumFrames,
    frameStep,
  )
  const exceedsSafeRecommendation = (
    locked
    && safeWindowSeconds != null
    && windowSize > safeWindowSeconds + 0.0001
  )

  if (
    !isH3 && !supportsSlidingWindows && !omniReferenceSequence
  ) return null

  return (
    <div className="space-y-3">
      <div>
        <div className="flex items-center justify-between mb-1.5">
          <div className="flex items-center gap-1.5">
            <label className="text-[11px] text-text-muted uppercase tracking-wider">
              {isH3 || isLtx ? 'Window Length' : 'Window Size'}
            </label>
            {isH3 && safeWindowSeconds != null && (
              <span className="text-[9px] text-text-muted normal-case">
                Recommended {formatSeconds(safeWindowSeconds)}
              </span>
            )}
            <button
              onClick={() => {
                if (locked) {
                  if (savedOverrideFrames != null) {
                    clearH3WindowOverride(modelType, resolution)
                  }
                  setLocked(false)
                } else {
                  setLocked(true)
                }
              }}
              className={`p-0.5 rounded transition-colors ${
                locked
                  ? 'text-accent-blue hover:text-accent-blue/70'
                  : 'text-text-muted hover:text-text-secondary'
              }`}
              title={locked
                ? `${savedOverrideFrames != null ? 'Saved override' : 'Temporary override'} - click to resume Auto`
                : 'Click to temporarily override the recommended window length'}
            >
              {locked ? <Lock size={10} /> : <Unlock size={10} />}
            </button>
            {isH3 && (
              <button
                onClick={() => {
                  saveH3WindowOverride(modelType, resolution, currentWindowFrames)
                  setLocked(true)
                }}
                disabled={savedOverrideFrames === currentWindowFrames}
                className={`p-0.5 rounded transition-colors disabled:cursor-default ${
                  savedOverrideFrames === currentWindowFrames
                    ? 'text-emerald-400/70'
                    : 'text-text-muted hover:text-accent-blue'
                }`}
                title={savedOverrideFrames === currentWindowFrames
                  ? `Saved for ${modelType} at ${resolution}`
                  : `Save ${formatSeconds(currentWindowFrames / fps)} for this model and resolution`}
              >
                <Save size={10} />
              </button>
            )}
          </div>
          <span className="text-xs text-text-secondary">
            {formatSeconds(windowSize)}
            {savedOverrideFrames === currentWindowFrames
              ? <span className="text-emerald-400/70 ml-1 text-[9px]">saved</span>
              : locked && <span className="text-accent-blue/60 ml-1 text-[9px]">manual</span>}
          </span>
        </div>
        <input
          type="range"
          min={isH3 ? minimumFrames : windowMinSeconds}
          max={isH3 ? maximumFrames : windowMaxSeconds}
          step={isH3 ? frameStep : windowStepSeconds}
          value={isH3 ? currentWindowFrames : windowSize}
          onChange={e => {
            // Any manual change to window size automatically locks it
            if (!locked) setLocked(true)
            const sliderValue = Number(e.target.value)
            setWindowSize(isH3 ? sliderValue / fps : sliderValue)
          }}
        />
        {showSlidingWindow && (
          <div className="text-[10px] text-text-muted mt-1">
            {windowCount} {omniReferenceSequence && !nativeOmniContinuation ? 'independent clip' : 'window'}{windowCount > 1 ? 's' : ''} of up to {formatSeconds(windowSize)}
          </div>
        )}
        {windowRecommendation != null && (
          <div className={`text-[10px] mt-1 ${unsupportedAutoResolution || exceedsSafeRecommendation ? 'text-amber-400' : 'text-text-muted'}`}>
            {unsupportedAutoResolution
              ? (locked
                ? `Manual override enabled: ${resolution} is above the automatic profile for ${totalVramGb.toFixed(0)} GB and may run out of VRAM.`
                : `Auto does not recommend ${resolution} on ${totalVramGb.toFixed(0)} GB. Choose ${windowRecommendation.fallbackResolution ?? 'a lower resolution'}, or manually set Window Length to try it experimentally.`)
              : (exceedsSafeRecommendation
                ? `Manual override exceeds the ${formatSeconds(safeWindowSeconds!)} recommendation for ${totalVramGb.toFixed(0)} GB at this resolution and may run out of VRAM.`
                : `Recommended: ${formatSeconds(safeWindowSeconds!)} for ${totalVramGb.toFixed(0)} GB at this resolution. The slider remains available through ${formatSeconds(maximumFrames / fps)}.${omniReferenceSequence && (windowRecommendation.referenceMarginFrames ?? 0) > 0 ? ' Includes Ref2VA reference headroom.' : ''}`)}
          </div>
        )}
      </div>

      {supportsSlidingWindows && showSlidingWindow && overlapStep > 0 && (!omniReferenceSequence || nativeOmniContinuation) && (
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
