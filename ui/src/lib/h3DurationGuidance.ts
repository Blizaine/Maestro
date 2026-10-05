import type {
  MiniMaxH3Reference,
  ModelOptions,
  SlidingWindowMemoryPolicy,
} from '../types'
import {
  normalizeH3NativeFrames,
  recommendedH3OmniSequenceProfile,
  recommendedH3PassProfile,
} from './h3Memory'

const H3_FPS = 24
const H3_FRAME_CHUNK = 17
const H3_FRAME_OFFSET = 5
const H3_LATENT_CHUNK = 5
const H3_LATENT_OFFSET = 2
const H3_AUDIO_LATENTS_PER_SECOND = 40
const H3_AUDIO_CHANNELS = 2
const H3_CANVAS_MULTIPLE = 32
const H3_OUTPUT_MAX_PIXELS = 768 * 1344
const H3_MAX_REFERENCE_IMAGE_SHORT_EDGE = 2048
const H3_MAX_REFERENCE_ASPECT_RATIO = 4
const H3_MAX_NATIVE_FRAMES = 345
const H3_MAX_EXTENDED_FRAMES = 719
const H3_DEFAULT_ACTIVATION_CHUNK_TOKENS = 8192
const H3_LARGE_SEQUENCE_TOKENS = 50000
export const H3_RMS_NORM_NATIVE_MAX_ROWS = 75000
const QWEN_MERGED_IMAGE_TOKEN_AREA = 16 * 16 * 4
const QWEN_MIN_IMAGE_PIXELS = 65536
const QWEN_MIN_VIDEO_PIXELS = 4096
const QWEN_MAX_IMAGE_PIXELS = 16_777_216
const QWEN_MAX_VIDEO_PIXELS = 25_165_824
const H3_QWEN_VIDEO_SAMPLE_FPS = 2

export type H3DurationGuidance = {
  supported: boolean
  path: 'fast' | 'chunked' | 'uncertain'
  rowsLow: number | null
  rowsHigh: number | null
  fastMaxFrames: number | null
  risk: 'within-profile' | 'caution' | 'high' | 'unknown'
  recommendationFrames: number | null
  beyondRecommended: boolean
  notes: string[]
}

export type H3DurationGuidanceInput = {
  options: ModelOptions | null
  resolution: string
  frames: number
  totalVramGb: number
  references?: MiniMaxH3Reference[]
  referenceDetail?: 'match' | 'max'
  /** Already-conditioned history frames, excluding the regenerated boundary frame. */
  continuationFrames?: number
  /** Explicit first/last or timed keyframe images. Continuation boundary is counted separately. */
  firstLastImageCount?: number
  videoGuide?: boolean
  audioGuide?: boolean
  prompt?: string
}

type RowRange = { low: number; high: number | null }
type Geometry = { rowsPerFrame: number; pixels: number }
type ReferenceDurations = { low: number[]; high: number[] }

function finitePositive(value: unknown): number | null {
  const parsed = Number(value)
  return Number.isFinite(parsed) && parsed > 0 ? parsed : null
}

function normalizeH3Frames(
  rawFrames: number,
  minimumFrames = 124,
  maximumFrames = H3_MAX_EXTENDED_FRAMES,
): number {
  const minimum = Math.max(H3_FRAME_OFFSET, Math.round(minimumFrames))
  const maximum = Math.min(
    H3_MAX_EXTENDED_FRAMES,
    Math.max(minimum, Math.round(maximumFrames)),
  )
  const legalMinimum = H3_FRAME_OFFSET + Math.ceil(
    (minimum - H3_FRAME_OFFSET) / H3_FRAME_CHUNK,
  ) * H3_FRAME_CHUNK
  const legalMaximum = H3_FRAME_OFFSET + Math.floor(
    (maximum - H3_FRAME_OFFSET) / H3_FRAME_CHUNK,
  ) * H3_FRAME_CHUNK
  const requested = Number.isFinite(rawFrames) ? Math.round(rawFrames) : legalMinimum
  const aligned = H3_FRAME_OFFSET + Math.ceil(
    (Math.max(legalMinimum, requested) - H3_FRAME_OFFSET) / H3_FRAME_CHUNK,
  ) * H3_FRAME_CHUNK
  return Math.min(legalMaximum, Math.max(legalMinimum, aligned))
}

function h3VideoLatents(frames: number): number {
  const aligned = normalizeH3Frames(frames, H3_FRAME_OFFSET)
  return Math.floor((aligned - H3_FRAME_OFFSET) / H3_FRAME_CHUNK) * H3_LATENT_CHUNK
    + H3_LATENT_OFFSET
}

function referenceVideoLatents(frameCount: number): number {
  const usableFrames = Math.max(
    22,
    Math.floor((frameCount - H3_FRAME_OFFSET) / H3_FRAME_CHUNK) * H3_FRAME_CHUNK
      + H3_FRAME_OFFSET,
  )
  return Math.floor((usableFrames - H3_FRAME_OFFSET) / H3_FRAME_CHUNK) * H3_LATENT_CHUNK
    + H3_LATENT_OFFSET
}

function getResolutionGeometry(options: ModelOptions | null, resolution: string): Geometry | null {
  const normalized = String(resolution || '').trim().toLowerCase()
  const dimensions = normalized.match(/^(\d{2,5})x(\d{2,5})$/)
  if (dimensions) {
    const width = Number(dimensions[1])
    const height = Number(dimensions[2])
    if (width > 0 && height > 0 && width % H3_CANVAS_MULTIPLE === 0 && height % H3_CANVAS_MULTIPLE === 0) {
      const rowsPerFrame = (width / H3_CANVAS_MULTIPLE) * (height / H3_CANVAS_MULTIPLE)
      return { rowsPerFrame, pixels: width * height }
    }
    return null
  }

  // Auto-resolution pixel budgets do not specify the realized aspect ratio or
  // rounded canvas. Keep them for VRAM recommendations, but do not turn them
  // into an exact row count near a runtime cutoff.
  void options
  return null
}

function memoryPolicy(options: ModelOptions | null): SlidingWindowMemoryPolicy | null {
  return options?.omni_reference === true
    ? options.omni_sequence_memory_policy ?? null
    : options?.sliding_window_memory_policy ?? null
}

function utf8ByteLength(value: string): number {
  let bytes = 0
  for (const character of value) {
    const codePoint = character.codePointAt(0) || 0
    bytes += codePoint <= 0x7f ? 1 : codePoint <= 0x7ff ? 2 : codePoint <= 0xffff ? 3 : 4
  }
  return bytes
}

function promptRows(prompt: string | undefined): RowRange {
  if (typeof prompt !== 'string') return { low: 0, high: null }
  const codePoints = Array.from(prompt).length
  const bytes = utf8ByteLength(prompt)
  // The browser does not have H3's tokenizer. UTF-8 bytes give a deliberately
  // loose upper bound for multilingual text; the lower side is only a guide.
  return {
    low: Math.max(8, Math.floor(codePoints / 8)),
    high: Math.max(64, bytes + 64),
  }
}

function addRows(total: RowRange, addition: RowRange): RowRange {
  return {
    low: total.low + Math.max(0, addition.low),
    high: total.high == null || addition.high == null
      ? null
      : total.high + Math.max(0, addition.high),
  }
}

function maxReferenceRowsPerFrame(detail: 'match' | 'max', geometry: Geometry): number {
  if (detail === 'max') {
    // Ref2VA image references keep a 2048px short edge and accept up to 4:1.
    // Video references use resolve_canvas_size's 768px/area-capped canvas.
    const maxImageRows = (H3_MAX_REFERENCE_IMAGE_SHORT_EDGE / H3_CANVAS_MULTIPLE)
      * (H3_MAX_REFERENCE_IMAGE_SHORT_EDGE * H3_MAX_REFERENCE_ASPECT_RATIO / H3_CANVAS_MULTIPLE)
    return Math.max(
      maxImageRows,
      Math.ceil(roundedPixelAreaUpper(H3_OUTPUT_MAX_PIXELS) / (H3_CANVAS_MULTIPLE ** 2)),
    )
  }
  // Match detail scales the source to the target area, then rounds each axis
  // to the nearest 32px. A 4:1 aspect bound gives x+y <= 2.5*sqrt(area), and
  // rounding can add at most 16px on each axis.
  const area = geometry.pixels
  const roundedPixelBound = roundedPixelAreaUpper(area)
  return Math.max(1, Math.ceil(roundedPixelBound / (H3_CANVAS_MULTIPLE ** 2)))
}

function roundedPixelAreaUpper(area: number): number {
  // For an aspect ratio up to 4:1, width + height <= 2.5*sqrt(area).
  // Rounding both axes up to the next 32px boundary adds at most 16px each.
  return area + 16 * 2.5 * Math.sqrt(area) + 256
}

function maxVideoReferenceRowsPerFrame(detail: 'match' | 'max', geometry: Geometry): number {
  if (detail === 'max') {
    return Math.max(1, Math.ceil(roundedPixelAreaUpper(H3_OUTPUT_MAX_PIXELS) / (H3_CANVAS_MULTIPLE ** 2)))
  }
  return maxReferenceRowsPerFrame('match', geometry)
}

function maxImageVisionTokens(detail: 'match' | 'max', geometry: Geometry): number {
  const pixelBound = detail === 'max'
    ? Math.min(
      H3_MAX_REFERENCE_IMAGE_SHORT_EDGE ** 2 * H3_MAX_REFERENCE_ASPECT_RATIO,
      QWEN_MAX_IMAGE_PIXELS,
    )
    : Math.min(QWEN_MAX_IMAGE_PIXELS, roundedPixelAreaUpper(geometry.pixels))
  return Math.ceil(Math.max(QWEN_MIN_IMAGE_PIXELS, pixelBound) / QWEN_MERGED_IMAGE_TOKEN_AREA) + 2
}

function maxVideoVisionTokens(detail: 'match' | 'max', geometry: Geometry): number {
  const pixelBound = detail === 'max'
    ? roundedPixelAreaUpper(H3_OUTPUT_MAX_PIXELS)
    : roundedPixelAreaUpper(geometry.pixels)
  return Math.ceil(
    Math.max(QWEN_MIN_VIDEO_PIXELS, Math.min(QWEN_MAX_VIDEO_PIXELS, pixelBound))
      / QWEN_MERGED_IMAGE_TOKEN_AREA,
  ) + 2
}

function minImageVisionTokens(): number {
  return Math.ceil(QWEN_MIN_IMAGE_PIXELS / QWEN_MERGED_IMAGE_TOKEN_AREA)
}

function minVideoVisionTokens(): number {
  return Math.ceil(QWEN_MIN_VIDEO_PIXELS / QWEN_MERGED_IMAGE_TOKEN_AREA)
}

function referenceDuration(reference: MiniMaxH3Reference): number | null {
  return finitePositive(reference.effective_duration_seconds)
    ?? finitePositive(reference.source_duration_seconds)
    ?? finitePositive(reference.audio_duration_seconds)
    ?? finitePositive(reference.duration_seconds)
}

function allocateDurations(durations: number[]): number[] {
  const values = durations.map(duration => Math.max(2, Math.min(15, duration)))
  if (values.reduce((sum, value) => sum + value, 0) <= 15.000001) return values

  const result = values.map(() => 0)
  const remaining = new Set(values.map((_value, index) => index))
  let budget = 15
  while (remaining.size) {
    const share = budget / remaining.size
    const short = [...remaining].filter(index => values[index] <= share + 1e-9)
    if (!short.length) {
      for (const index of remaining) result[index] = share
      break
    }
    for (const index of short) {
      result[index] = values[index]
      budget -= values[index]
      remaining.delete(index)
    }
  }
  return result
}

function applyPassBudget(durations: number[], passFrames: number, kind: 'audio' | 'video'): number[] {
  if (!durations.length) return []
  const passSeconds = passFrames / H3_FPS
  const budgets = allocateDurations(durations).map(duration => Math.min(passSeconds, duration))
  const allowedSeconds = Math.min(15, passSeconds * durations.length)
  let excess = Math.max(0, budgets.reduce((sum, value) => sum + value, 0) - allowedSeconds)
  for (const index of budgets.map((_value, i) => i).sort((a, b) => budgets[b] - budgets[a])) {
    const reduction = Math.min(excess, budgets[index])
    budgets[index] -= reduction
    excess -= reduction
    if (excess <= 1e-9) break
  }

  if (kind === 'audio') return budgets

  const frameCounts = budgets.map(value => Math.max(1, Math.round(value * H3_FPS)))
  const maxTotalFrames = Math.min(
    Math.round(15 * H3_FPS),
    Math.round(passSeconds * H3_FPS) * durations.length,
  )
  while (frameCounts.reduce((sum, value) => sum + value, 0) > maxTotalFrames) {
    const longest = frameCounts.indexOf(Math.max(...frameCounts))
    if (frameCounts[longest] <= 1) break
    frameCounts[longest] -= 1
  }
  return frameCounts
}

function getDurationBounds(references: MiniMaxH3Reference[]): ReferenceDurations {
  return {
    low: references.map(reference => referenceDuration(reference) ?? 2),
    high: references.map(reference => referenceDuration(reference) ?? 15),
  }
}

function estimateReferenceRows(
  references: MiniMaxH3Reference[],
  passFrames: number,
  detail: 'match' | 'max',
  geometry: Geometry | null,
): { rows: RowRange; visionTokens: RowRange; unknownRefmod: boolean; notes: string[] } {
  let rows: RowRange = { low: 0, high: 0 }
  let visionTokens: RowRange = { low: 0, high: 0 }
  let unknownRefmod = false
  const notes: string[] = []
  const images = references.filter(reference => reference.type === 'image')
  const videos = references.filter(reference => reference.type === 'video')
  const audios = references.filter(reference => reference.type === 'audio')
  const videoDurations = getDurationBounds(videos)
  const audioDurations = getDurationBounds(audios)
  const videoFramesLow = applyPassBudget(videoDurations.low, passFrames, 'video')
  const videoFramesHigh = applyPassBudget(videoDurations.high, passFrames, 'video')
  const audioSecondsLow = applyPassBudget(audioDurations.low, passFrames, 'audio')
  const audioSecondsHigh = applyPassBudget(audioDurations.high, passFrames, 'audio')
  const maxDetailGeometry: Geometry = { rowsPerFrame: 1, pixels: 0 }
  const referenceGeometry = geometry ?? maxDetailGeometry
  const detailHasKnownCap = geometry != null || detail === 'max'
  const maxVideoRows = detailHasKnownCap ? maxVideoReferenceRowsPerFrame(detail, referenceGeometry) : null
  const maxImageRows = detailHasKnownCap ? maxReferenceRowsPerFrame(detail, referenceGeometry) : null
  const maxImageTokens = detailHasKnownCap ? maxImageVisionTokens(detail, referenceGeometry) : null
  const maxVideoTokens = detailHasKnownCap ? maxVideoVisionTokens(detail, referenceGeometry) : null

  for (const reference of images) {
    if (reference.refmod_path) {
      unknownRefmod = true
      rows = addRows(rows, { low: 0, high: null })
      notes.push('A .refmod reference has stored latent dimensions the browser cannot inspect, so its packed rows remain unknown.')
    } else {
      rows = addRows(rows, { low: 1, high: maxImageRows })
    }
    visionTokens = addRows(visionTokens, { low: minImageVisionTokens(), high: maxImageTokens })
  }

  for (let index = 0; index < videos.length; index += 1) {
    const reference = videos[index]
    const lowFrames = videoFramesLow[index] ?? 1
    const highFrames = videoFramesHigh[index] ?? lowFrames
    const lowLatents = referenceVideoLatents(lowFrames)
    const highLatents = referenceVideoLatents(highFrames)
    if (reference.refmod_path) {
      unknownRefmod = true
      rows = addRows(rows, { low: 0, high: null })
      notes.push('A .refmod reference has stored latent dimensions the browser cannot inspect, so its packed rows remain unknown.')
    } else {
      rows = addRows(rows, {
        low: lowLatents,
        high: maxVideoRows == null ? null : highLatents * maxVideoRows,
      })
    }

    const lowSamples = Math.max(1, Math.ceil(lowFrames / (H3_FPS / H3_QWEN_VIDEO_SAMPLE_FPS)))
    const highSamples = Math.max(1, Math.ceil(highFrames / (H3_FPS / H3_QWEN_VIDEO_SAMPLE_FPS)) + 1)
    visionTokens = addRows(visionTokens, {
      low: lowSamples * minVideoVisionTokens(),
      high: maxVideoTokens == null ? null : highSamples * (maxVideoTokens + 8),
    })
  }

  if (audios.length) {
    const lowAudioSeconds = audioSecondsLow.map((budget, index) => {
      const reference = audios[index]
      if (reference.audio_intent === 'drive') return budget
      const duration = referenceDuration(reference)
      return duration == null ? 0 : Math.min(budget, duration)
    })
    const highAudioSeconds = audioSecondsHigh.map((budget, index) => {
      const reference = audios[index]
      if (reference.audio_intent === 'drive') return budget
      const duration = referenceDuration(reference)
      return duration == null ? budget : Math.min(budget, duration)
    })
    rows = addRows(rows, {
      low: lowAudioSeconds.reduce((sum, value) => sum + Math.round(value * H3_AUDIO_LATENTS_PER_SECOND), 0)
        * H3_AUDIO_CHANNELS,
      high: highAudioSeconds.reduce((sum, value) => sum + Math.round(value * H3_AUDIO_LATENTS_PER_SECOND), 0)
        * H3_AUDIO_CHANNELS,
    })
  }

  const videoAudioMin = videos.reduce((sum, reference, index) => {
    const definitelyHasAudio = reference.include_audio !== false
      && (Boolean(reference.audio_path) || reference.has_audio === true)
    if (!definitelyHasAudio) return sum
    const windowSeconds = (videoFramesLow[index] ?? 0) / H3_FPS
    const audioSeconds = reference.follow_timeline
      ? windowSeconds
      : Math.min(windowSeconds, finitePositive(reference.audio_duration_seconds) ?? 0)
    return sum + Math.round(audioSeconds * H3_AUDIO_LATENTS_PER_SECOND)
  }, 0)
  const videoAudioMax = videos.reduce((sum, reference, index) => {
    const mayHaveAudio = reference.include_audio !== false
      && (Boolean(reference.audio_path) || reference.has_audio !== false)
    if (!mayHaveAudio) return sum
    const windowSeconds = (videoFramesHigh[index] ?? 0) / H3_FPS
    const knownAudioDuration = finitePositive(reference.audio_duration_seconds)
    const audioSeconds = reference.follow_timeline
      ? windowSeconds
      : knownAudioDuration == null ? windowSeconds : Math.min(windowSeconds, knownAudioDuration)
    return sum + Math.round(audioSeconds * H3_AUDIO_LATENTS_PER_SECOND)
  }, 0)
  if (videoAudioMax > 0) {
    rows = addRows(rows, {
      low: videoAudioMin * H3_AUDIO_CHANNELS,
      high: videoAudioMax * H3_AUDIO_CHANNELS,
    })
  }

  const textLabelsLow = references.length * 4
  const textLabelsHigh = references.length * 24 + videos.length * 8
  visionTokens = addRows(visionTokens, { low: textLabelsLow, high: textLabelsHigh })

  const hasUnknownMediaDuration = [...videos, ...audios].some(reference => referenceDuration(reference) == null)
  if (hasUnknownMediaDuration) {
    rows = { ...rows, high: null }
    notes.push('An audio or video reference has no known duration; H3’s per-pass allocation can shift between references, so the upper row estimate remains unknown.')
  }
  if (references.length && !geometry) {
    notes.push('Reference detail rows need output geometry; this resolution has no exact size or pixel-area policy.')
  }
  if (videos.length || audios.length) {
    notes.push('H3 allocates audio and video reference durations separately, with a 15-second total budget for each type per pass.')
  }
  if (detail === 'max' && references.length) {
    notes.push('Maximum detail can add many visual rows; H3 image references use a 2048px short edge and video references use the 768px area-capped canvas.')
  }
  if (references.some(reference => Boolean(reference.refmod_path)) && !unknownRefmod) {
    unknownRefmod = true
    rows = addRows(rows, { low: 0, high: null })
    notes.push('A .refmod reference has stored latent dimensions the browser cannot inspect, so its packed rows remain unknown.')
  }
  return { rows, visionTokens, unknownRefmod, notes: [...new Set(notes)] }
}

function estimateRows(
  input: H3DurationGuidanceInput,
  frames: number,
  geometry: Geometry | null,
  prompt: RowRange,
  detail: 'match' | 'max',
): { range: RowRange; unknownRefmod: boolean; hasUnknownGuide: boolean; notes: string[] } {
  const passFrames = normalizeH3Frames(frames)
  const rawHistoryFrames = finitePositive(input.continuationFrames) ?? 0
  const historyFrames = Math.min(Math.max(0, Math.round(rawHistoryFrames)), Math.max(0, passFrames - 5))
  const targetFrames = normalizeH3Frames(Math.max(5, passFrames - historyFrames), 5)
  const targetAudioRows = Math.round(targetFrames / H3_FPS * H3_AUDIO_LATENTS_PER_SECOND)
    * H3_AUDIO_CHANNELS
  const firstLastCount = Math.max(0, Math.round(finitePositive(input.firstLastImageCount) ?? 0))
  const hasContinuation = historyFrames > 0
  const continuationGroupsLow = Math.floor(historyFrames / H3_FRAME_CHUNK)
  const continuationGroupsHigh = Math.ceil(historyFrames / H3_FRAME_CHUNK)
  const rowsPerFrame = geometry?.rowsPerFrame ?? null
  const references = input.references ?? []
  const referenceRows = estimateReferenceRows(references, passFrames, detail, geometry)
  let range: RowRange = addRows(prompt, referenceRows.visionTokens)
  range = addRows(range, { low: targetAudioRows, high: targetAudioRows })

  if (rowsPerFrame != null) {
    range = addRows(range, {
      low: h3VideoLatents(targetFrames) * rowsPerFrame
        + continuationGroupsLow * H3_LATENT_CHUNK * rowsPerFrame
        + (firstLastCount + (hasContinuation ? 1 : 0)) * rowsPerFrame,
      high: h3VideoLatents(targetFrames) * rowsPerFrame
        + continuationGroupsHigh * H3_LATENT_CHUNK * rowsPerFrame
        + (firstLastCount + (hasContinuation ? 1 : 0)) * rowsPerFrame,
    })
    range = addRows(range, referenceRows.rows)
  } else {
    range = addRows(range, { low: 0, high: null })
    range = addRows(range, referenceRows.rows)
  }

  if (firstLastCount || hasContinuation) {
    if (geometry) {
      const keyframeVision = firstLastCount + (hasContinuation ? 1 : 0)
      const maxTokens = Math.ceil(
        Math.max(
          QWEN_MIN_IMAGE_PIXELS,
          Math.min(QWEN_MAX_IMAGE_PIXELS, roundedPixelAreaUpper(geometry.pixels)),
        ) / QWEN_MERGED_IMAGE_TOKEN_AREA,
      ) + 2
      range = addRows(range, { low: keyframeVision, high: keyframeVision * maxTokens })
    } else {
      range = addRows(range, { low: 0, high: null })
    }
  }

  if (hasContinuation && input.options?.sliding_window_audio_history) {
    // The audio stream may be absent from the continuation source. Bound the
    // optional carried stereo history without assuming every video has audio.
    const audioHistoryHigh = Math.round(historyFrames / H3_FPS * H3_AUDIO_LATENTS_PER_SECOND)
      * H3_AUDIO_CHANNELS
    range = addRows(range, { low: 0, high: audioHistoryHigh })
  }

  const hasUnknownGuide = Boolean(input.videoGuide || input.audioGuide)
  if (hasUnknownGuide) {
    range = addRows(range, { low: 0, high: null })
  }

  const notes = [...referenceRows.notes]
  if (geometry == null) {
    const size = String(input.resolution || '').trim().match(/^(\d{2,5})x(\d{2,5})$/i)
    notes.push(size
      ? 'Target dimensions are not multiples of H3’s 32px canvas grid, so the browser leaves target rows unknown.'
      : 'Target video row count is unknown because an auto/preset resolution does not identify the realized canvas dimensions.')
  }
  if (typeof input.prompt !== 'string') {
    notes.push('Prompt text was not provided; its tokenizer rows are unknown.')
  }
  if (historyFrames > 0) {
    notes.push('Continuation history is included as conditioning rows inside this pass; it is not counted as a full extra output duration.')
  }
  if (historyFrames && historyFrames % H3_FRAME_CHUNK !== 0) {
    notes.push('Continuation history is not on H3’s 17-frame overlap lattice, so its latent-row range is widened.')
  }
  if (hasUnknownGuide) {
    notes.push('A video or audio guide is active, but its source dimensions and duration are not available; no full extra stream is assumed.')
  }
  return {
    range,
    unknownRefmod: referenceRows.unknownRefmod,
    hasUnknownGuide,
    notes: [...new Set(notes)],
  }
}

function supportsH3Video(options: ModelOptions | null): boolean {
  if (!options || options.audio_only) return false
  const model = String(options.model_type || '').toLowerCase()
  const architecture = String(options.architecture || '').toLowerCase()
  return model !== 'viggle_animate'
    && architecture !== 'viggle_animate'
    && !model.includes('viggle')
    && !architecture.includes('viggle')
    && (model.startsWith('minimax_h3') || architecture.startsWith('minimax_h3'))
}

function estimateFastMaxFrames(
  input: H3DurationGuidanceInput,
  geometry: Geometry | null,
  prompt: RowRange,
  detail: 'match' | 'max',
  minFrames: number,
): number | null {
  if (!geometry || prompt.high == null || input.videoGuide || input.audioGuide) return null
  if ((input.references ?? []).some(reference => Boolean(reference.refmod_path))) return null

  let fastMax: number | null = null
  for (let candidate = minFrames; candidate <= H3_MAX_NATIVE_FRAMES; candidate += H3_FRAME_CHUNK) {
    const estimated = estimateRows(input, candidate, geometry, prompt, detail).range
    if (estimated.high != null && estimated.high <= H3_RMS_NORM_NATIVE_MAX_ROWS) {
      fastMax = candidate
    }
  }
  return fastMax
}

/**
 * Estimate the packed H3 transformer rows for one pass and compare them with
 * the loaded model's existing VRAM/resolution duration profile. This describes
 * sequence shape and runtime eligibility; it cannot predict render time or
 * guarantee that a render will fit in memory.
 */
export function estimateH3DurationGuidance(
  input: H3DurationGuidanceInput,
): H3DurationGuidance {
  const notes: string[] = [
    'Rows are an estimate of H3’s packed transformer sequence; this does not guarantee speed or memory fit.',
    `H3 keeps RMSNorm native through ${H3_RMS_NORM_NATIVE_MAX_ROWS} rows. Its default ${H3_DEFAULT_ACTIVATION_CHUNK_TOKENS}-row activation chunks become fixed above ${H3_LARGE_SEQUENCE_TOKENS} rows, so the path label describes RMSNorm eligibility only.`,
  ]
  if (!supportsH3Video(input.options)) {
    return {
      supported: false,
      path: 'uncertain',
      rowsLow: null,
      rowsHigh: null,
      fastMaxFrames: null,
      risk: 'unknown',
      recommendationFrames: null,
      beyondRecommended: false,
      notes: ['Duration guidance is available for H3 video models; audio-only and Viggle models are excluded.'],
    }
  }

  const options = input.options!
  const detail: 'match' | 'max' = input.referenceDetail === 'max' ? 'max' : 'match'
  const configuredMinimum = Math.max(124, Number(options.frames_minimum) || 124)
  const minimumFrames = normalizeH3Frames(configuredMinimum, 124, H3_MAX_NATIVE_FRAMES)
  const requestedFrames = normalizeH3Frames(input.frames, minimumFrames, H3_MAX_EXTENDED_FRAMES)
  const geometry = getResolutionGeometry(options, input.resolution)
  const prompt = promptRows(input.prompt)
  const estimated = estimateRows(input, requestedFrames, geometry, prompt, detail)
  notes.push(...estimated.notes)
  if (requestedFrames >= H3_FRAME_OFFSET && input.frames > H3_MAX_NATIVE_FRAMES) {
    notes.push('This request is beyond H3’s 345-frame (about 14.4-second) native pass and is in the experimental extended range.')
  }
  if (input.frames > H3_MAX_EXTENDED_FRAMES) {
    notes.push('The row estimate is capped at H3’s 719-frame extended lattice ceiling.')
  }
  if (typeof input.prompt === 'string') {
    notes.push('Text rows use a broad UTF-8 byte range because the H3 tokenizer is not run in the browser.')
  }

  const rowsLow = estimated.range.low
  const rowsHigh = estimated.range.high
  const path: H3DurationGuidance['path'] = rowsLow > H3_RMS_NORM_NATIVE_MAX_ROWS
    ? 'chunked'
    : rowsHigh != null && rowsHigh <= H3_RMS_NORM_NATIVE_MAX_ROWS
      ? 'fast'
      : 'uncertain'

  const policy = memoryPolicy(options)
  const useOmniProfile = options.omni_reference === true
  const memoryProfile = useOmniProfile
    ? recommendedH3OmniSequenceProfile(
      policy,
      input.resolution,
      input.totalVramGb,
      minimumFrames,
      H3_MAX_NATIVE_FRAMES,
      H3_FRAME_CHUNK,
    )
    : recommendedH3PassProfile(policy, input.resolution, input.totalVramGb)
  const recommendationFrames = memoryProfile?.supported && memoryProfile.frames != null
    ? normalizeH3NativeFrames(
      memoryProfile.frames,
      minimumFrames,
      H3_MAX_NATIVE_FRAMES,
      H3_FRAME_CHUNK,
    )
    : null
  const beyondRecommended = requestedFrames > H3_MAX_NATIVE_FRAMES
  const exceedsMemoryRecommendation = recommendationFrames != null
    && requestedFrames > recommendationFrames
  const unsupportedMemoryProfile = memoryProfile?.supported === false
  const extended = requestedFrames > H3_MAX_NATIVE_FRAMES
  const hasReferences = (input.references ?? []).length > 0
  const hasKeyframes = Boolean(input.firstLastImageCount || input.continuationFrames)
  const hasUnknownContext = estimated.range.high == null || estimated.unknownRefmod || estimated.hasUnknownGuide
  let risk: H3DurationGuidance['risk']
  if (extended || exceedsMemoryRecommendation || unsupportedMemoryProfile) {
    risk = 'high'
  } else if (recommendationFrames == null) {
    risk = 'unknown'
  } else if (hasReferences || hasKeyframes || hasUnknownContext || detail === 'max') {
    risk = 'caution'
  } else {
    risk = 'within-profile'
  }

  if (!Number.isFinite(input.totalVramGb) || input.totalVramGb <= 0) {
    notes.push('VRAM is unknown, so no model-specific frame recommendation is available.')
  } else if (unsupportedMemoryProfile) {
    notes.push('The model has no supported memory recommendation for this resolution and VRAM tier; treat memory risk as high.')
  } else if (recommendationFrames == null) {
    notes.push('The loaded model has no applicable recommendation for this exact resolution and VRAM value.')
  }
  if (exceedsMemoryRecommendation && !extended) {
    notes.push('The selected pass exceeds the loaded model’s VRAM/resolution recommendation, although it remains within the native 345-frame duration envelope.')
  }
  if (hasReferences && recommendationFrames != null && !beyondRecommended) {
    notes.push('The VRAM profile includes a conservative reference margin, but unusually large references can still raise memory use.')
  }
  if (path === 'chunked') {
    notes.push('The estimated lower bound is above the native RMSNorm cutoff; this predicts extra chunking work, not an out-of-memory result.')
  } else if (path === 'uncertain') {
    notes.push('The row range crosses the RMSNorm cutoff or contains context the browser cannot measure.')
  }

  return {
    supported: true,
    path,
    rowsLow,
    rowsHigh,
    fastMaxFrames: estimateFastMaxFrames(input, geometry, prompt, detail, minimumFrames),
    risk,
    recommendationFrames,
    beyondRecommended,
    notes: [...new Set(notes)],
  }
}
