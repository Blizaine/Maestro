import type { SlidingWindowMemoryPolicy } from '../types'

export type H3MemoryRecommendation = {
  supported: boolean
  frames: number | null
  baseFrames?: number | null
  referenceMarginFrames?: number
  fallbackResolution?: string
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
