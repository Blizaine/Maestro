import type { SlidingWindowMemoryPolicy } from '../types'

export type H3MemoryRecommendation = {
  supported: boolean
  frames: number | null
  baseFrames?: number | null
  referenceMarginFrames?: number
  fallbackResolution?: string
}

/** Stable key for a user-saved H3 native-pass override. */
export function h3WindowOverrideKey(modelType: string, resolution: string): string {
  return `${String(modelType || '').trim()}::${String(resolution || '').trim().toLowerCase()}`
}

export function recommendedH3PassProfile(
  policy: SlidingWindowMemoryPolicy | null | undefined,
  resolution: string,
  totalVramGb: number,
): H3MemoryRecommendation | null {
  if (!policy || !Number.isFinite(totalVramGb) || totalVramGb <= 0) return null
  const normalizedResolution = String(resolution || '').trim().toLowerCase()
  let pixels = policy.auto_resolution_pixels?.[normalizedResolution]
  if (!pixels) {
    const match = normalizedResolution.match(/^(\d+)x(\d+)$/)
    if (match) pixels = Number(match[1]) * Number(match[2])
  }
  if (!pixels) return null
  const band = policy.resolution_bands.find(item => pixels! >= item.min_pixels)
  const tier = band?.vram_tiers.find(item => (
    item.max_vram_gb == null || totalVramGb <= item.max_vram_gb
  ))
  if (!tier) return null
  const frames = tier.frames != null && tier.frames > 0 ? tier.frames : null
  return {
    supported: frames != null,
    frames,
    fallbackResolution: tier.fallback_resolution,
  }
}

export function normalizeH3NativeFrames(
  requestedFrames: number,
  minimumFrames: number,
  maximumFrames: number,
  frameStep: number,
): number {
  const minimum = Math.max(1, Math.round(minimumFrames))
  const maximum = Math.max(minimum, Math.round(maximumFrames))
  const step = Math.max(1, Math.round(frameStep))
  const requested = Math.max(minimum, Math.min(maximum, Math.round(requestedFrames)))
  return Math.min(
    maximum,
    minimum + Math.floor((requested - minimum) / step) * step,
  )
}

/**
 * Repair an independently generated H3 clip without shortening source media.
 *
 * H3's native clips start at 124 frames and advance in 17-frame steps. Media
 * durations and pre-H3 sidecars can instead contain ordinary 24-fps counts
 * such as 120 (exactly five seconds), so round upward to the next legal clip.
 */
export function normalizeH3ClipFrames(
  requestedFrames: number,
  minimumFrames: number,
  maximumFrames: number,
  frameStep: number,
): number {
  const minimum = Math.max(1, Math.round(minimumFrames))
  const step = Math.max(1, Math.round(frameStep))
  const rawMaximum = Math.max(minimum, Math.round(maximumFrames))
  const maximum = minimum + Math.floor((rawMaximum - minimum) / step) * step
  const requested = Number.isFinite(requestedFrames)
    ? Math.round(requestedFrames)
    : minimum
  if (requested <= minimum) return minimum
  const latticeSteps = Math.ceil((requested - minimum) / step)
  return Math.min(maximum, minimum + latticeSteps * step)
}

/** Repair a multi-clip H3 schedule without accumulating timing drift. */
export function normalizeH3ClipFrameSchedule(
  requestedFrames: number[],
  minimumFrames: number,
  maximumFrames: number,
  frameStep: number,
): number[] {
  const minimum = Math.max(1, Math.round(minimumFrames))
  const step = Math.max(1, Math.round(frameStep))
  const rawMaximum = Math.max(minimum, Math.round(maximumFrames))
  const maximum = minimum + Math.floor((rawMaximum - minimum) / step) * step
  const valid: number[] = []
  for (let value = minimum; value <= maximum; value += step) valid.push(value)
  let residual = 0
  return requestedFrames.map(rawValue => {
    const requested = Number.isFinite(rawValue) ? Math.round(rawValue) : minimum
    const target = Math.max(1, requested) + residual
    let chosen = valid[0]
    for (const candidate of valid) {
      const candidateDistance = Math.abs(candidate - target)
      const chosenDistance = Math.abs(chosen - target)
      if (candidateDistance < chosenDistance) chosen = candidate
    }
    residual = target - chosen
    return chosen
  })
}

export function recommendedH3OmniSequenceProfile(
  policy: SlidingWindowMemoryPolicy | null | undefined,
  resolution: string,
  totalVramGb: number,
  minimumFrames: number,
  maximumFrames: number,
  frameStep: number,
): H3MemoryRecommendation | null {
  const profile = recommendedH3PassProfile(policy, resolution, totalVramGb)
  if (!profile || !profile.supported || profile.frames == null) return profile
  const baseFrames = normalizeH3NativeFrames(
    profile.frames,
    minimumFrames,
    maximumFrames,
    frameStep,
  )
  const marginSteps = Math.max(0, Math.round(policy?.reference_margin_steps ?? 1))
  const safeFrames = normalizeH3NativeFrames(
    baseFrames - marginSteps * Math.max(1, frameStep),
    minimumFrames,
    maximumFrames,
    frameStep,
  )
  return {
    ...profile,
    frames: safeFrames,
    baseFrames,
    referenceMarginFrames: baseFrames - safeFrames,
  }
}

export function effectiveH3OmniSequenceFrames({
  policy,
  resolution,
  totalVramGb,
  minimumFrames,
  maximumFrames,
  frameStep,
  selectedFrames,
  manualOverride,
}: {
  policy: SlidingWindowMemoryPolicy | null | undefined
  resolution: string
  totalVramGb: number
  minimumFrames: number
  maximumFrames: number
  frameStep: number
  selectedFrames: number
  manualOverride: boolean
}): { frames: number; recommendation: H3MemoryRecommendation | null } {
  const selected = normalizeH3NativeFrames(
    selectedFrames,
    minimumFrames,
    maximumFrames,
    frameStep,
  )
  const recommendation = recommendedH3OmniSequenceProfile(
    policy,
    resolution,
    totalVramGb,
    minimumFrames,
    maximumFrames,
    frameStep,
  )
  return {
    frames: manualOverride || recommendation?.frames == null
      ? selected
      : recommendation.frames,
    recommendation,
  }
}

/**
 * Return the number of Ref2VA passes needed for an Omni sequence.
 *
 * Native continuation owns an overlap prefix on every pass after the first,
 * so each later window advances by `windowFrames - overlapFrames`. Hard-cut
 * sequences are independent clips and therefore advance by the full pass.
 * Keep this small helper shared by the duration UI, prompt editor, and submit
 * validation so "3 windows" always means the same thing everywhere.
 */
export function h3OmniSequenceWindowCount({
  totalFrames,
  windowFrames,
  overlapFrames,
  nativeContinuation,
}: {
  totalFrames: number
  windowFrames: number
  overlapFrames: number
  nativeContinuation: boolean
}): number {
  const total = Math.max(1, Math.round(totalFrames))
  const window = Math.max(1, Math.round(windowFrames))
  if (total <= window) return 1
  if (!nativeContinuation) return Math.max(1, Math.ceil(total / window))
  return h3SlidingWindowCount({
    totalFrames: total,
    windowFrames: window,
    overlapFrames,
  })
}

/** Match the backend's committed-output geometry for rolling H3 windows. */
export function h3SlidingWindowCount({
  totalFrames,
  windowFrames,
  overlapFrames,
  discardFrames = 0,
}: {
  totalFrames: number
  windowFrames: number
  overlapFrames: number
  discardFrames?: number
}): number {
  const total = Math.max(1, Math.round(totalFrames))
  const window = Math.max(1, Math.round(windowFrames))
  if (total <= window) return 1
  const overlap = Math.max(0, Math.min(window - 1, Math.round(overlapFrames)))
  const discard = Math.max(
    0,
    Math.min(window - overlap - 1, Math.round(discardFrames)),
  )
  const stride = Math.max(1, window - overlap - discard)
  return 1 + Math.ceil((total - window + discard) / stride)
}

/** Treat H3's UI-rounded 14.4s endpoint as the native 345-frame pass. */
export function h3TimelineFrames(
  durationSeconds: number,
  fps: number,
  nativeMaximumFrames?: number | null,
): number {
  const raw = Math.max(1, Math.round(durationSeconds * Math.max(1, fps)))
  if (
    nativeMaximumFrames != null
    && raw <= Math.max(1, Math.round(nativeMaximumFrames)) + 1
  ) {
    return Math.min(Math.max(1, Math.round(nativeMaximumFrames)), raw)
  }
  return raw
}
