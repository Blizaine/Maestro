export const LONG_FORM_MAX_SECONDS = 60 * 60

export type DurationPlanningMode = 'duration' | 'windows' | 'auto'
export type AutoDurationPlanningStyle = 'faithful' | 'creative'

export const LONG_FORM_DURATION_PRESETS = [
  { label: '30s', seconds: 30 },
  { label: '1m', seconds: 60 },
  { label: '2m', seconds: 2 * 60 },
  { label: '3m', seconds: 3 * 60 },
  { label: '4m', seconds: 4 * 60 },
  { label: '5m', seconds: 5 * 60 },
  { label: '10m', seconds: 10 * 60 },
  { label: '15m', seconds: 15 * 60 },
  { label: '30m', seconds: 30 * 60 },
  { label: '60m', seconds: LONG_FORM_MAX_SECONDS },
] as const

export function formatDuration(seconds: number, includeTenths = false): string {
  const safe = Math.max(0, Number.isFinite(seconds) ? seconds : 0)
  const rounded = includeTenths ? Math.round(safe * 10) / 10 : Math.round(safe)
  const hours = Math.floor(rounded / 3600)
  const minutes = Math.floor((rounded % 3600) / 60)
  const remaining = rounded % 60
  const secondsText = includeTenths && !Number.isInteger(remaining)
    ? remaining.toFixed(1)
    : String(Math.round(remaining))
  if (hours > 0) return `${hours}h ${minutes}m ${secondsText}s`
  if (minutes > 0) return `${minutes}m ${secondsText}s`
  return `${secondsText}s`
}

export function formatTimecode(seconds: number): string {
  const safe = Math.max(0, Math.round((Number.isFinite(seconds) ? seconds : 0) * 10) / 10)
  const hours = Math.floor(safe / 3600)
  const minutes = Math.floor((safe % 3600) / 60)
  const remaining = safe % 60
  const secondsText = Number.isInteger(remaining)
    ? String(remaining).padStart(2, '0')
    : remaining.toFixed(1).padStart(4, '0')
  return `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}:${secondsText}`
}

export function parseTimecode(value: string): number | null {
  const text = value.trim()
  if (!text) return null
  if (/^\d+(?:\.\d+)?$/.test(text)) return Number(text)
  const parts = text.split(':')
  if (parts.length < 2 || parts.length > 3 || parts.some(part => !/^\d+(?:\.\d+)?$/.test(part))) {
    return null
  }
  const numbers = parts.map(Number)
  if (numbers.some(number => !Number.isFinite(number))) return null
  if (parts.length === 2) return numbers[0] * 60 + numbers[1]
  return numbers[0] * 3600 + numbers[1] * 60 + numbers[2]
}

export interface DurationWindowPlan {
  windowCount: number
  generatedSeconds: number
  requestedSeconds: number
  trimSeconds: number
  strideSeconds: number
}

export interface AutoDurationPlan extends DurationWindowPlan {
  reason: string
  source: 'media' | 'manual_prompts' | 'explicit_prompt' | 'story_scope' | 'empty_prompt'
  inferredWindowLimit: number
}

export interface AutoDurationOptions {
  prompt?: string
  planningStyle?: AutoDurationPlanningStyle
  sourceSeconds?: number | null
  sourceLabel?: string
  manualWindowCount?: number | null
  minimumSeconds?: number
  maximumSeconds?: number
  maximumInferredWindows?: number
}

export function durationWindowPlan(
  requestedSeconds: number,
  windowSeconds: number,
  overlapSeconds = 0,
  discardSeconds = 0,
): DurationWindowPlan {
  const requested = Math.max(0, requestedSeconds)
  const window = Math.max(0.1, windowSeconds)
  const stride = Math.max(0.1, window - Math.max(0, overlapSeconds) - Math.max(0, discardSeconds))
  if (requested <= window + 0.0001) {
    return {
      windowCount: 1,
      generatedSeconds: window,
      requestedSeconds: requested,
      trimSeconds: Math.max(0, window - requested),
      strideSeconds: stride,
    }
  }
  const count = 1 + Math.ceil((requested - window + Math.max(0, discardSeconds)) / stride)
  const generated = window + (count - 1) * stride - Math.max(0, discardSeconds)
  return {
    windowCount: count,
    generatedSeconds: generated,
    requestedSeconds: requested,
    trimSeconds: Math.max(0, generated - requested),
    strideSeconds: stride,
  }
}

export function nearestWholeWindowDuration(
  targetSeconds: number,
  windowSeconds: number,
  overlapSeconds = 0,
  discardSeconds = 0,
  maximumSeconds = LONG_FORM_MAX_SECONDS,
): DurationWindowPlan {
  const window = Math.max(0.1, windowSeconds)
  const stride = Math.max(0.1, window - Math.max(0, overlapSeconds) - Math.max(0, discardSeconds))
  const target = Math.min(maximumSeconds, Math.max(window, targetSeconds))
  const estimate = Math.max(1, Math.round((target - window + Math.max(0, discardSeconds)) / stride) + 1)
  const candidates = [estimate - 1, estimate, estimate + 1]
    .filter(count => count >= 1)
    .map(count => {
      const generated = count === 1
        ? window
        : window + (count - 1) * stride - Math.max(0, discardSeconds)
      return { count, generated }
    })
    .filter(candidate => candidate.generated <= maximumSeconds + 0.0001)
  const best = candidates.sort((a, b) => (
    Math.abs(a.generated - target) - Math.abs(b.generated - target)
  ))[0] || { count: 1, generated: Math.min(window, maximumSeconds) }
  return {
    windowCount: best.count,
    generatedSeconds: best.generated,
    requestedSeconds: best.generated,
    trimSeconds: 0,
    strideSeconds: stride,
  }
}

/** Exact generated timeline for a requested number of native passes.
 *
 * This deliberately shares the same overlap/discard geometry as
 * `durationWindowPlan` and `nearestWholeWindowDuration`; the Windows UI is
 * therefore a different way to select the existing timeline, not a second
 * renderer-specific duration implementation.
 */
export function wholeWindowDuration(
  requestedWindowCount: number,
  windowSeconds: number,
  overlapSeconds = 0,
  discardSeconds = 0,
  maximumSeconds = LONG_FORM_MAX_SECONDS,
): DurationWindowPlan {
  const window = Math.max(0.1, windowSeconds)
  const stride = Math.max(0.1, window - Math.max(0, overlapSeconds) - Math.max(0, discardSeconds))
  const maximumCount = maximumWholeWindowCount(
    window,
    overlapSeconds,
    discardSeconds,
    maximumSeconds,
  )
  const count = Math.max(1, Math.min(maximumCount, Math.round(requestedWindowCount || 1)))
  const generated = count === 1
    ? Math.min(window, maximumSeconds)
    : Math.min(
        maximumSeconds,
        window + (count - 1) * stride - Math.max(0, discardSeconds),
      )
  return {
    windowCount: count,
    generatedSeconds: generated,
    requestedSeconds: generated,
    trimSeconds: 0,
    strideSeconds: stride,
  }
}

export function maximumWholeWindowCount(
  windowSeconds: number,
  overlapSeconds = 0,
  discardSeconds = 0,
  maximumSeconds = LONG_FORM_MAX_SECONDS,
): number {
  const window = Math.max(0.1, windowSeconds)
  const discard = Math.max(0, discardSeconds)
  const stride = Math.max(0.1, window - Math.max(0, overlapSeconds) - discard)
  if (maximumSeconds <= window + 0.0001) return 1
  return Math.max(1, Math.floor((maximumSeconds + discard - window) / stride) + 1)
}

function wordCount(value: string): number {
  return (value.match(/[\p{L}\p{N}]+(?:['’][\p{L}\p{N}]+)*/gu) || []).length
}

const dialogueAttributionOnly = /^(?:(?:then|and|but)\s+)?(?:[\p{L}\p{N}'’.-]+\s+)+(?:says?|asks?|responds?|replies?|answers?|adds?|shouts?|yells?|whispers?|murmurs?|exclaims?|calls?|announces?)(?:\s+(?:in|with)\s+.+)?$/iu
const sceneRequestOnly = /^(?:please\s+)?(?:make|create|generate|write|produce)\s+(?:me\s+)?(?:a|an|the)?\s*(?:scene|video|clip|film|movie|sequence)\b(?:\s+(?:from|in|for|with|using)\b.*)?$/iu

function visibleActionBeatCount(prompt: string): number {
  // Dialogue already receives its own word-based timing below. Replace every
  // quoted turn with a sentence boundary so adjacent speaker attributions do
  // not collapse together, then count only independently visible action.
  const nonDialoguePrompt = prompt.replace(/["“](.*?)["”]/gs, '. ')
  const actionSentences = nonDialoguePrompt
    .split(/(?:[.!?]+|\n+)/)
    .map(part => part.trim().replace(/^[,;:\s]+|[,;:\s]+$/g, ''))
    .filter(Boolean)
    .filter(part => !dialogueAttributionOnly.test(part))
    .filter(part => !sceneRequestOnly.test(part))
  const actionText = actionSentences.join(' ')
  const transitions = (actionText.match(/\b(?:then|next|after(?:ward|wards)?|suddenly|finally|meanwhile|before|until|followed by)\b/gi) || []).length
  return Math.max(1, actionSentences.length, transitions + 1)
}

function explicitDurationFromPrompt(prompt: string): number | null {
  const text = String(prompt || '')
  const timecode = text.match(/\b(?:duration|length|runtime|for)\s*(?:of|is|:)?\s*(\d{1,2}):(\d{2})(?::(\d{2}(?:\.\d+)?))?\b/i)
  if (timecode) {
    const first = Number(timecode[1])
    const second = Number(timecode[2])
    const third = timecode[3] == null ? null : Number(timecode[3])
    return third == null ? first * 60 + second : first * 3600 + second * 60 + third
  }

  // Require duration language so a camera cue such as "at 30s" does not
  // accidentally resize the whole project.
  const unit = text.match(
    /(?:\bfor\s+|\b(?:duration|length|runtime)\s*(?:of|is|:)?\s*|\bmake\s+(?:it|this)\s+)(\d+(?:\.\d+)?)\s*(hours?|hrs?|hr|h|minutes?|mins?|min|m|seconds?|secs?|sec|s)\b/i,
  )
  const adjectival = unit ? null : text.match(
    /\b(\d+(?:\.\d+)?)\s*[- ]\s*(hour|hr|minute|min|second|sec)s?\s+(?:video|clip|scene|sequence|film|movie|song|speech|conversation)\b/i,
  )
  const match = unit || adjectival
  if (!match) return null
  const amount = Number(match[1])
  const name = match[2].toLowerCase()
  if (!Number.isFinite(amount) || amount <= 0) return null
  if (name.startsWith('h')) return amount * 3600
  if (name.startsWith('m')) return amount * 60
  return amount
}

/** Deterministic, bounded duration recommendation.
 *
 * Exact media, an explicit duration in the prompt, and manual prompt lines
 * are user-owned constraints and may use the full one-hour ceiling. Only an
 * inferred story scope is capped (eight windows by default), preventing a
 * short concept from silently becoming an all-day render.
 */
export function recommendAutoDuration(
  windowSeconds: number,
  overlapSeconds = 0,
  discardSeconds = 0,
  options: AutoDurationOptions = {},
): AutoDurationPlan {
  const minimum = Math.max(0.1, options.minimumSeconds ?? 1)
  const maximum = Math.max(minimum, options.maximumSeconds ?? LONG_FORM_MAX_SECONDS)
  const inferredLimit = Math.max(1, Math.round(options.maximumInferredWindows ?? 8))
  const boundedPlan = (seconds: number) => durationWindowPlan(
    Math.min(maximum, Math.max(minimum, seconds)),
    windowSeconds,
    overlapSeconds,
    discardSeconds,
  )
  const result = (
    seconds: number,
    reason: string,
    source: AutoDurationPlan['source'],
  ): AutoDurationPlan => ({
    ...boundedPlan(seconds),
    requestedSeconds: Math.min(maximum, Math.max(minimum, seconds)),
    reason,
    source,
    inferredWindowLimit: inferredLimit,
  })

  const sourceSeconds = Number(options.sourceSeconds)
  if (Number.isFinite(sourceSeconds) && sourceSeconds > 0) {
    return result(
      sourceSeconds,
      `Matches the complete ${options.sourceLabel || 'timed media input'}.`,
      'media',
    )
  }

  const manualCount = Math.round(Number(options.manualWindowCount))
  if (Number.isFinite(manualCount) && manualCount > 0) {
    const plan = wholeWindowDuration(
      manualCount,
      windowSeconds,
      overlapSeconds,
      discardSeconds,
      maximum,
    )
    return {
      ...plan,
      reason: `${plan.windowCount} manual prompt ${plan.windowCount === 1 ? 'line' : 'lines'} define the exact window count.`,
      source: 'manual_prompts',
      inferredWindowLimit: inferredLimit,
    }
  }

  const prompt = String(options.prompt || '').trim()
  const explicitSeconds = explicitDurationFromPrompt(prompt)
  if (explicitSeconds != null) {
    return result(
      explicitSeconds,
      'Uses the duration explicitly requested in the prompt.',
      'explicit_prompt',
    )
  }

  if (!prompt) {
    const plan = wholeWindowDuration(1, windowSeconds, overlapSeconds, discardSeconds, maximum)
    return {
      ...plan,
      reason: 'No timed media or story scope yet, so Auto starts with one window.',
      source: 'empty_prompt',
      inferredWindowLimit: inferredLimit,
    }
  }

  const quoted = [...prompt.matchAll(/["“](.*?)["”]/gs)].map(match => match[1])
  const dialogueWords = quoted.reduce((sum, line) => sum + wordCount(line), 0)
  const dialogueTurns = quoted.filter(line => line.trim()).length
  const visibleBeats = visibleActionBeatCount(prompt)
  const speakingSeconds = dialogueWords / 2.15 + dialogueTurns * 0.8
  const actionSeconds = visibleBeats * 3.5
  const scopedSeconds = Math.max(windowSeconds, speakingSeconds + actionSeconds)
  const scopedPlan = durationWindowPlan(
    scopedSeconds,
    windowSeconds,
    overlapSeconds,
    discardSeconds,
  )
  const creativeFloor = options.planningStyle === 'creative' ? 3 : 1
  const dialogueFloor = dialogueTurns >= 2 ? 2 : 1
  const inferredCount = Math.max(
    creativeFloor,
    dialogueFloor,
    Math.min(inferredLimit, scopedPlan.windowCount),
  )
  const plan = wholeWindowDuration(
    Math.min(inferredLimit, inferredCount),
    windowSeconds,
    overlapSeconds,
    discardSeconds,
    maximum,
  )
  return {
    ...plan,
    reason: options.planningStyle === 'creative'
      ? `Creative Auto gives the idea room for ${plan.windowCount} paced story windows.`
      : `Auto sized ${plan.windowCount} ${plan.windowCount === 1 ? 'window' : 'windows'} from the visible beats and exact dialogue.`,
    source: 'story_scope',
    inferredWindowLimit: inferredLimit,
  }
}
