import { normalizeH3ClipFrames, h3SlidingWindowCount } from './h3Memory'

export interface SlidingWindowDefaults {
  window_min?: number
  window_max?: number
  window_step?: number
  overlap_min?: number
  overlap_max?: number
  overlap_step?: number
  overlap_offset?: number
  discard_last_frames?: number
}

export interface H3SlidingWindowPlanTiming {
  source_prompt: string
  model_type: string
  resolution: string
  total_frames: number
  window_frames: number
  effective_window_frames?: number
  overlap_frames?: number
  discard_frames?: number
  window_count: number
  camera_coverage?: string
  plan_kind?: string
  windows: Array<{ start_frame: number; end_frame: number }>
}

export interface H3SlidingWindowTimingExpectation {
  sourcePrompt: string
  modelType: string
  resolution: string
  totalFrames: number
  windowFrames: number
  overlapFrames: number
  discardFrames: number
  cameraCoverage?: string
}

/** Match the backend's model-specific frame lattice for sliding windows. */
export function normalizeSlidingWindowFrames(
  value: number,
  defaults?: SlidingWindowDefaults | null,
): number {
  if (!defaults) return Math.max(1, Math.round(value))
  const minimum = defaults.window_min ?? 1
  const maximum = defaults.window_max ?? Math.max(minimum, value)
  const step = Math.max(1, defaults.window_step ?? 1)
  const normalized = minimum + Math.round((value - minimum) / step) * step
  return Math.max(minimum, Math.min(maximum, normalized))
}

/** Match the backend's model-specific overlap lattice, including its offset. */
export function normalizeSlidingWindowOverlap(
  value: number,
  defaults?: SlidingWindowDefaults | null,
): number {
  if (!defaults) return Math.max(0, Math.round(value))
  const minimum = defaults.overlap_min ?? 1
  const maximum = defaults.overlap_max ?? Math.max(minimum, value)
  const step = Math.max(1, defaults.overlap_step ?? 1)
  const offset = defaults.overlap_offset ?? minimum
  const normalized = offset + Math.round((value - offset) / step) * step
  return Math.max(minimum, Math.min(maximum, normalized))
}

/**
 * Apply the single-pass native-length repair used at submission. A rolling
 * H3 Frames timeline is a schedule of native windows, so its final short
 * boundary is valid and must keep the user's requested total frame count.
 */
export function normalizeH3TimelineFramesForSubmission({
  requestedFrames,
  minimumFrames,
  maximumFrames,
  frameStep,
  continuationContextFrames = 0,
  preserveRollingFramesTimeline = false,
}: {
  requestedFrames: number
  minimumFrames: number
  maximumFrames: number | null | undefined
  frameStep: number
  continuationContextFrames?: number
  preserveRollingFramesTimeline?: boolean
}): number {
  const requested = Math.max(1, Math.round(requestedFrames))
  const context = Math.max(0, Math.round(continuationContextFrames))
  if (preserveRollingFramesTimeline || maximumFrames == null) return requested
  if (requested + context > maximumFrames + 1) return requested
  return normalizeH3ClipFrames(
    requested + context,
    minimumFrames,
    maximumFrames,
    frameStep,
  ) - context
}

/** Preserve the visible timeline only when Frames will really span passes. */
export function isH3RollingFramesTimeline({
  framesWorkflow,
  multiWindowRequested,
  omniReference,
  requestedFrames,
  continuationContextFrames = 0,
  selectedWindowFrames,
}: {
  framesWorkflow: boolean
  multiWindowRequested: boolean
  omniReference: boolean
  requestedFrames: number
  continuationContextFrames?: number
  selectedWindowFrames: number
}): boolean {
  return framesWorkflow
    && multiWindowRequested
    && !omniReference
    && Math.max(0, Math.round(requestedFrames))
      + Math.max(0, Math.round(continuationContextFrames))
      > Math.max(1, Math.round(selectedWindowFrames))
}

/**
 * The H3 clean-tail experiment discards 17 boundary frames before selecting
 * the next continuation tail. Keep this timing decision shared by planning,
 * stale-plan checks, and the generation payload.
 */
export function resolveH3StoryboardDiscardFrames(
  _defaults?: SlidingWindowDefaults | null,
  customSettings?: unknown,
): number {
  return customSettings != null
    && typeof customSettings === 'object'
    && !Array.isArray(customSettings)
    && (customSettings as Record<string, unknown>).h3_long_sequence_clean_tail === true
    ? 17 : 0
}

function expectedH3StoryboardBoundaries({
  totalFrames,
  windowFrames,
  overlapFrames,
  discardFrames,
}: H3SlidingWindowTimingExpectation): Array<[number, number]> {
  const total = Math.max(1, Math.round(totalFrames))
  const window = Math.max(1, Math.round(windowFrames))
  if (total <= window) return [[0, total]]
  const overlap = Math.max(0, Math.min(window - 1, Math.round(overlapFrames)))
  const discard = Math.max(0, Math.min(window - overlap - 1, Math.round(discardFrames)))
  const stride = Math.max(1, window - overlap - discard)
  const count = h3SlidingWindowCount({ totalFrames: total, windowFrames: window, overlapFrames: overlap, discardFrames: discard })
  const boundaries: Array<[number, number]> = []
  let start = 0
  let end = Math.min(total, window - (count > 1 ? discard : 0))
  boundaries.push([start, end])
  while (end < total) {
    start = end
    end = Math.min(total, start + stride)
    boundaries.push([start, end])
  }
  return boundaries
}

/** Verify that the exact reviewed plan still describes the current schedule. */
export function h3SlidingWindowPlanMatchesTiming(
  plan: H3SlidingWindowPlanTiming | null | undefined,
  expected: H3SlidingWindowTimingExpectation,
): boolean {
  if (!plan || plan.plan_kind === 'reference_sequence') return false
  const windowFrames = plan.effective_window_frames ?? plan.window_frames
  const expectedCount = h3SlidingWindowCount({
    totalFrames: expected.totalFrames,
    windowFrames: expected.windowFrames,
    overlapFrames: expected.overlapFrames,
    discardFrames: expected.discardFrames,
  })
  const expectedBoundaries = expectedH3StoryboardBoundaries(expected)
  const boundariesMatch = plan.windows.length === expectedBoundaries.length
    && plan.windows.every((window, index) => (
      window.start_frame === expectedBoundaries[index][0]
      && window.end_frame === expectedBoundaries[index][1]
    ))
  return plan.source_prompt.trim() === expected.sourcePrompt.trim()
    && plan.model_type === expected.modelType
    && plan.resolution === expected.resolution
    && plan.total_frames === expected.totalFrames
    && windowFrames === expected.windowFrames
    && (plan.overlap_frames == null || plan.overlap_frames === expected.overlapFrames)
    && (plan.discard_frames == null || plan.discard_frames === expected.discardFrames)
    && plan.window_count === expectedCount
    && plan.windows.length === expectedCount
    && boundariesMatch
    && (!expected.cameraCoverage || (plan.camera_coverage || 'auto') === expected.cameraCoverage)
}
