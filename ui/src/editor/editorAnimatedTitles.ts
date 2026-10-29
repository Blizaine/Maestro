import type { EditorAsset, EditorProject } from '../types'
import { DEFAULT_EDITOR_KEYING } from './editorKeying'

export const TITLE_MODEL = 'minimax_h3_fused_turbo'
/** randomUUID is secure-context-only; LAN HTTP still provides getRandomValues. */
export function titleSubmissionId(): string {
  const crypto = globalThis.crypto
  if (typeof crypto?.randomUUID === 'function') return crypto.randomUUID()
  if (typeof crypto?.getRandomValues !== 'function') throw new Error('This browser cannot create a reliable title submission identity.')
  const bytes = crypto.getRandomValues(new Uint8Array(16))
  bytes[6] = (bytes[6] & 15) | 64
  bytes[8] = (bytes[8] & 63) | 128
  const hex = Array.from(bytes, byte => byte.toString(16).padStart(2, '0')).join('')
  return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`
}
export const TITLE_EFFECTS = {
  dust: 'The complete title appears, holds clearly long enough to read, then breaks into fine dust and blows away.',
  dissolve: 'The complete title fades into view, holds clearly long enough to read, then gently dissolves away.',
  neon: 'The title traces on with luminous neon strokes, holds clearly long enough to read, then switches off and disappears.',
  impact: 'The complete title assembles with a bold energetic impact, holds clearly long enough to read, then shatters and clears the frame.',
  smoke: 'Soft smoke reveals the complete title. Hold it clearly long enough to read, then let the lettering dissolve into smoke and clear the frame.',
  typewriter: 'Type the title on one character at a time. Hold the complete wording clearly, then erase the characters until the frame is clear.',
  brush: 'Draw the lettering with expressive brush strokes. Hold the complete title clearly, then let the strokes fade away.',
  glitch: 'Assemble the title through brief digital glitch fragments. Settle into crisp readable lettering, hold, then glitch out and clear the frame.',
  liquid: 'Liquid metal flows together into the exact title. Hold the solid readable lettering, then let it flow apart and disappear.',
  fire: 'Sparks trace and reveal the title with fiery edges. Hold the complete lettering clearly, then let the embers fade and clear the frame.',
  frost: 'Frost crystals form the title. Hold the complete readable lettering, then let it melt into a faint mist and disappear.',
  paper: 'Folded paper shapes unfold into the exact title. Hold it clearly long enough to read, then fold the lettering away and clear the frame.',
  sand: 'Fine sand gathers into the exact title. Hold the complete lettering clearly, then let it scatter away until the frame is clear.',
  electric: 'Electric arcs trace the exact title. Hold the completed readable lettering, then let the arcs flicker out and disappear.',
  hologram: 'A holographic scan reveals the exact title. Stabilize the complete wording for readability, then scan it away and clear the frame.',
  paint: 'Paint splashes assemble into the exact title. Hold the complete readable lettering, then let it break into droplets and disappear.',
  kinetic: 'Animate the exact title with bold changes of scale, rotation, compression and word placement. Let the lettering itself carry the motion. Settle into a complete readable composition, hold, then clear the frame.',
  collision: 'Bring the title words in from opposing sides so they collide and settle into the exact readable title. Hold the complete wording, then scatter the letters and clear the frame.',
  rebuild: 'Break the exact title into letter fragments, then regroup them into a crisp, correctly spelled composition. Hold it clearly long enough to read, then let the lettering disappear.',
  offset: 'Reveal the exact title, briefly offset and overlap copies of its lettering, then bring every copy back into one clean readable composition. Hold the complete title, then clear the frame.',
} as const
export const TITLE_EFFECT_LABELS: Record<keyof typeof TITLE_EFFECTS, string> = {
  dust: 'Dust dissolve', dissolve: 'Soft reveal and dissolve', neon: 'Neon trace', impact: 'Impact and shatter',
  smoke: 'Smoke reveal', typewriter: 'Typewriter', brush: 'Brush lettering', glitch: 'Digital glitch',
  liquid: 'Liquid metal', fire: 'Fire and sparks', frost: 'Frost and melt', paper: 'Paper unfolding',
  sand: 'Sand assembly', electric: 'Electric trace', hologram: 'Hologram scan', paint: 'Paint splash',
  kinetic: 'Kinetic typography', collision: 'Word collision', rebuild: 'Deconstruct and rebuild', offset: 'Offset and overlap',
}

export type TitleBackground = 'chroma' | 'black'
export interface TitlePhrase { id: string; text: string; seconds: number; animation: string }
export const TITLE_PHRASE_ANIMATIONS = {
  fade: { label: 'Fade', direction: 'Fade this phrase in, hold clearly, then fade it out completely.' },
  slide: { label: 'Slide', direction: 'Slide this phrase in, hold clearly, then slide every letter out of frame.' },
  scale: { label: 'Scale', direction: 'Scale this phrase into view, hold clearly, then shrink it away completely.' },
  rotate: { label: 'Rotate', direction: 'Rotate this phrase into place, hold clearly, then rotate it away completely.' },
  shatter: { label: 'Shatter', direction: 'Assemble this phrase, hold clearly, then shatter its letters into fragments that disappear completely.' },
  typewriter: { label: 'Typewriter', direction: 'Type this phrase one character at a time, hold clearly, then erase every character.' },
} as const

export interface AnimatedTitleSettings {
  brief?: string
  stylePreset?: 'custom' | keyof typeof TITLE_EFFECTS
  plannedKey?: string
  plannerWarnings?: string[]
  rawTitleDraft?: string
  titleDraftHistoryId?: string
  titleReviewRequired?: boolean
  manualPlanEdited?: boolean
  /** Missing on older saved titles, which use one title and a free-form direction. */
  mode?: 'single' | 'sequence'
  phrases?: TitlePhrase[]
  sequenceDirection?: string
  /** Whole-clip duration; phrase slots are individual video frames within it. */
  sequenceFrames?: number
  text: string
  look: string
  effect: 'custom' | keyof typeof TITLE_EFFECTS
  direction: string
  seconds: number
  aspect: 'project' | '16:9' | '9:16' | '1:1'
  resolution: '480p' | '720p'
  keepAudio: boolean
  /** Missing on older saved title intents, which always used green keying. */
  background?: TitleBackground
}

export interface TitleCreativePlan {
  phrases: { text: string; animation: string; weight: number }[]
  look: string; direction: string; sequence: boolean
}

export function titleBrief(settings: AnimatedTitleSettings): string {
  if (settings.brief !== undefined) return settings.brief
  if (settings.mode === 'sequence' && settings.phrases?.length) return [
    `Show ${settings.phrases.map(phrase => JSON.stringify(phrase.text)).join(', then ')} one at a time.`,
    ...settings.phrases.map(phrase => phrase.animation ? `${JSON.stringify(phrase.text)}: ${phrase.animation}` : ''),
    settings.look, settings.sequenceDirection,
  ].filter(Boolean).join('\n')
  return settings.text.trim() ? [`Animate ${JSON.stringify(settings.text.trim())}.`, settings.look, settings.direction].filter(Boolean).join('\n') : ''
}

export function titlePlanningKey(settings: AnimatedTitleSettings): string {
  return JSON.stringify([titleBrief(settings).trim(), settings.stylePreset ?? 'custom',
    titleFrames(titleSeconds(settings)), settings.background ?? 'chroma'])
}

export function titlePlanIsCurrent(settings: AnimatedTitleSettings): boolean {
  return settings.brief === undefined || settings.plannedKey === titlePlanningKey(settings)
}

/** Creative descriptions are allowed before a concrete render prompt exists. */
export function titleComposerValidation(settings: AnimatedTitleSettings, canvas: { width: number; height: number }): string {
  try {
    if (settings.manualPlanEdited) return 'Choose Use manual title plan to keep your Advanced edits, or Enhance to create a new AI plan.'
    titleDimensions(settings, canvas); titleFrames(titleSeconds(settings)); titleBackground(settings)
    if (settings.brief !== undefined && !titlePlanIsCurrent(settings)) {
      if (!settings.brief.trim() || settings.brief.length > 4000) return 'Describe your title animation in 1–4,000 characters.'
      return ''
    }
    return titleFormState(settings, canvas).validation
  } catch (reason) { return reason instanceof Error ? reason.message : 'Check the title settings.' }
}

export function applyTitleCreativePlan(settings: AnimatedTitleSettings, plan: TitleCreativePlan): AnimatedTitleSettings {
  const frames = titleFrames(titleSeconds(settings))
  const phrases = plan.phrases.map(phrase => ({ id: titleSubmissionId(), text: phrase.text,
    seconds: Number.isFinite(phrase.weight) && phrase.weight > 0 ? Math.min(10, phrase.weight) : 1, animation: phrase.animation }))
  let result: AnimatedTitleSettings = { ...settings, brief: titleBrief(settings), manualPlanEdited: false, effect: 'custom', look: plan.look,
    text: phrases.map(phrase => phrase.text).join(plan.sequence ? '\n' : ' '),
    direction: phrases[0]?.animation || plan.direction || 'Reveal the title, hold it clearly, then clear the frame.',
    sequenceDirection: plan.direction, seconds: frames / 24, mode: plan.sequence && phrases.length > 1 ? 'sequence' : 'single',
    phrases: plan.sequence && phrases.length > 1 ? phrases : undefined, sequenceFrames: frames }
  if (result.mode === 'sequence') result = alignTitleSequence(result, frames)
  result.plannedKey = titlePlanningKey(result)
  animatedTitlePrompt(result)
  return result
}

export const TITLE_SEQUENCE_DURATIONS = Array.from({ length: 14 }, (_, index) => {
  const frames = 124 + index * 17
  return { frames, seconds: frames / 24, label: `${(frames / 24).toFixed(3)}s · ${frames} frames` }
})

function nativeSequenceFrames(frames: number): number {
  if (!TITLE_SEQUENCE_DURATIONS.some(choice => choice.frames === frames)) throw new Error('Choose an H3 native sequence duration.')
  return frames
}

/** Proportional integer-frame allocation with a one-second minimum per phrase. */
function allocatePhraseFrames(weights: number[], total: number): number[] {
  if (total < weights.length * 24 || total > weights.length * 240) throw new Error('The phrases do not fit the selected clip length.')
  const result = weights.map(() => 0)
  const safeWeights = weights.map(weight => Number.isFinite(weight) && weight > 0 ? Math.min(10, weight) : 1)
  let remaining = total
  let active = weights.map((_, index) => index)
  while (active.length) {
    const sum = active.reduce((value, index) => value + safeWeights[index], 0)
    const fixed = active.find(index => remaining * safeWeights[index] / sum < 24 || remaining * safeWeights[index] / sum > 240)
    if (fixed !== undefined) {
      result[fixed] = remaining * safeWeights[fixed] / sum < 24 ? 24 : 240
      remaining -= result[fixed]
      active = active.filter(index => index !== fixed)
      continue
    }
    const shares = active.map(index => ({ index, exact: remaining * safeWeights[index] / sum }))
    for (const share of shares) result[share.index] = Math.floor(share.exact)
    const extra = remaining - shares.reduce((value, share) => value + result[share.index], 0)
    shares.sort((a, b) => (b.exact - Math.floor(b.exact)) - (a.exact - Math.floor(a.exact)) || a.index - b.index)
    for (let index = 0; index < extra; index++) result[shares[index].index]++
    break
  }
  return result
}

/** Used by editable drafts only; submitted legacy intents retain their original plan. */
export function alignTitleSequence(settings: AnimatedTitleSettings, frames?: number): AnimatedTitleSettings {
  const phrases = settings.phrases ?? []
  if (phrases.length < 2 || phrases.length > 6) throw new Error('Use 2–6 phrases in a title sequence.')
  const requested = frames ?? settings.sequenceFrames ?? titleFrames(Math.max(5, Math.min(14.5,
    phrases.reduce((sum, phrase) => sum + (Number.isFinite(phrase.seconds) ? phrase.seconds : 1), 0))))
  nativeSequenceFrames(requested)
  // Adding a sixth phrase may need the next longer native clip to retain 1s slots.
  const choices = TITLE_SEQUENCE_DURATIONS.filter(choice => choice.frames >= phrases.length * 24)
  const selected = choices.reduce((best, choice) => Math.abs(choice.frames - requested) < Math.abs(best.frames - requested) ? choice : best)
  const counts = allocatePhraseFrames(phrases.map(phrase => phrase.seconds), selected.frames)
  return { ...settings, sequenceFrames: selected.frames, phrases: phrases.map((phrase, index) => ({ ...phrase, seconds: counts[index] / 24 })) }
}

/** Keep the edited slot; redistribute the rest without changing the native clip length. */
export function setTitlePhraseSeconds(settings: AnimatedTitleSettings, id: string, seconds: number): AnimatedTitleSettings {
  const aligned = alignTitleSequence(settings)
  const phrases = aligned.phrases!
  const index = phrases.findIndex(phrase => phrase.id === id)
  if (index < 0 || !Number.isFinite(seconds)) return aligned
  const frames = aligned.sequenceFrames!
  const selected = Math.max(24, frames - (phrases.length - 1) * 240,
    Math.min(240, frames - (phrases.length - 1) * 24, Math.round(seconds * 24)))
  const others = phrases.filter((_, position) => position !== index)
  const counts = allocatePhraseFrames(others.map(phrase => phrase.seconds), frames - selected)
  let otherIndex = 0
  return { ...aligned, phrases: phrases.map((phrase, position) => ({ ...phrase,
    seconds: (position === index ? selected : counts[otherIndex++]) / 24 })) }
}

/** Converting pasted lines is explicit; ordinary multiline titles remain supported. */
export function newTitlePhrases(text: string, seconds = 10): TitlePhrase[] {
  const lines = text.split(/\r?\n/).map(line => line.trim()).filter(Boolean)
  if (lines.length > 6) throw new Error('Paste at most 6 phrases; no lines have been discarded.')
  while (lines.length < 2) lines.push('')
  const totalTenths = Math.max(lines.length * 10, Math.round((Number.isFinite(seconds) ? Math.max(5, Math.min(14.5, seconds)) : 10) * 10))
  const each = Math.floor(totalTenths / lines.length)
  return lines.map((line, index) => ({ id: titleSubmissionId(), text: line,
    seconds: (each + (index < totalTenths % lines.length ? 1 : 0)) / 10, animation: '' }))
}

export function titleText(settings: AnimatedTitleSettings): string {
  return settings.mode === 'sequence' ? (settings.phrases ?? []).map(phrase => phrase.text.trim()).join(' → ') : settings.text.trim()
}

export function titleSeconds(settings: AnimatedTitleSettings): number {
  return settings.mode === 'sequence' ? settings.sequenceFrames !== undefined ? settings.sequenceFrames / 24
    : Math.round((settings.phrases ?? []).reduce((sum, phrase) => sum + phrase.seconds, 0) * 1000) / 1000 : settings.seconds
}

/** Slot lengths include the entrance, readable hold and exit, within one GPU clip. */
export function titleSequenceTimeline(settings: AnimatedTitleSettings) {
  const phrases = settings.phrases
  if (!Array.isArray(phrases) || phrases.length < 2 || phrases.length > 6) throw new Error('Use 2–6 phrases in a title sequence.')
  if (phrases.some(phrase => !phrase || typeof phrase.id !== 'string' || !phrase.id
    || typeof phrase.text !== 'string' || !phrase.text.trim() || phrase.text.length > 80
    || !Number.isFinite(phrase.seconds) || phrase.seconds < 1 || phrase.seconds > 10
    || typeof phrase.animation !== 'string' || phrase.animation.length > 800)) {
    throw new Error('Each phrase needs 1–80 characters, 1–10 seconds and an animation under 800 characters.')
  }
  if (new Set(phrases.map(phrase => phrase.id)).size !== phrases.length) throw new Error('Each phrase needs a unique identity.')
  if (phrases.map(phrase => phrase.text.trim()).join('\n').length > 160) throw new Error('Keep the entire sequence within 160 characters.')
  const requestedSeconds = Math.round(phrases.reduce((sum, phrase) => sum + phrase.seconds, 0) * 1000) / 1000
  const frames = settings.sequenceFrames === undefined ? titleFrames(requestedSeconds) : nativeSequenceFrames(settings.sequenceFrames)
  const counts = settings.sequenceFrames === undefined ? null : allocatePhraseFrames(phrases.map(phrase => phrase.seconds), frames)
  let elapsed = 0
  let elapsedFrames = 0
  return { requestedSeconds, duration: frames / 24, phrases: phrases.map((phrase, index) => {
    const start = counts ? elapsedFrames / 24 : Math.round(elapsed / requestedSeconds * frames) / 24
    elapsed += phrase.seconds
    elapsedFrames += counts?.[index] ?? 0
    const end = counts ? elapsedFrames / 24 : index === phrases.length - 1 ? frames / 24 : Math.round(elapsed / requestedSeconds * frames) / 24
    return { ...phrase, seconds: counts ? counts[index] / 24 : phrase.seconds, text: phrase.text.trim(), start, end }
  }) }
}

export const DEFAULT_TITLE_SETTINGS: AnimatedTitleSettings = {
  text: '', look: '', effect: 'custom', direction: '', seconds: 5.2,
  aspect: 'project', resolution: '720p', keepAudio: false, background: 'chroma',
}

export function titleFormState(settings: AnimatedTitleSettings, canvas: { width: number; height: number }) {
  let geometry = { width: 0, height: 0 }, prompt = '', duration = 0, validation = ''
  try {
    geometry = titleDimensions(settings, canvas)
    duration = titleFrames(titleSeconds(settings)) / 24
    prompt = animatedTitlePrompt(settings)
  } catch (reason) { validation = reason instanceof Error ? reason.message : 'Check the title settings.' }
  return { geometry, prompt, duration, validation }
}

/** One choice governs the boundary images, prompt and inserted clip's keying. */
export function titleBackground(settings: { background?: TitleBackground }) {
  const mode = settings.background === undefined ? 'chroma' : settings.background
  if (mode === 'chroma') return { mode, color: '#00FF00', fileLabel: 'green', chromaKey: true } as const
  if (mode === 'black') return { mode, color: '#000000', fileLabel: 'black', chromaKey: false } as const
  throw new Error('Choose a supported title background.')
}

export function titleDimensions(settings: AnimatedTitleSettings, canvas: { width: number; height: number }) {
  const ratio = settings.aspect === 'project' ? canvas.width / canvas.height
    : settings.aspect === '1:1' ? 1 : settings.aspect === '9:16' ? 9 / 16 : 16 / 9
  if (!Number.isFinite(ratio) || ratio <= 0 || ratio < .25 || ratio > 4) throw new Error('Choose a supported title aspect ratio.')
  const area = settings.resolution === '480p' ? 832 * 480 : 1280 * 704
  return {
    width: Math.max(32, Math.round(Math.sqrt(area * ratio) / 32) * 32),
    height: Math.max(32, Math.round(Math.sqrt(area / ratio) / 32) * 32),
  }
}

export function titleFrames(seconds: number): number {
  if (!Number.isFinite(seconds) || seconds < 5 || seconds > 14.5) throw new Error('Title duration must be between 5 and 14.5 seconds.')
  return Math.max(124, Math.min(345, Math.round((seconds * 24 - 5) / 17) * 17 + 5))
}

export function animatedTitlePrompt(settings: AnimatedTitleSettings): string {
  if (settings.manualPlanEdited) throw new Error('Choose Use manual title plan before generating with your Advanced edits.')
  if (!titlePlanIsCurrent(settings)) throw new Error('The description or style changed. Enhance it or review the manual plan before generating.')
  const background = titleBackground(settings)
  if (settings.mode !== undefined && settings.mode !== 'single' && settings.mode !== 'sequence') throw new Error('Choose a supported title layout.')
  const sequence = settings.mode === 'sequence' ? titleSequenceTimeline(settings) : null
  const text = settings.text.trim()
  if (!sequence && (!text || text.length > 160)) throw new Error('Enter a title of 1–160 characters.')
  if (settings.effect !== 'custom' && !Object.hasOwn(TITLE_EFFECTS, settings.effect)) throw new Error('Choose an animation effect.')
  if (settings.look.length > 1200 || settings.direction.length > 1200 || (settings.sequenceDirection?.length ?? 0) > 1200) throw new Error('Keep each direction under 1,200 characters.')
  const animation = settings.direction.trim() || (settings.effect === 'custom' ? '' : TITLE_EFFECTS[settings.effect])
  if (!sequence && !animation) throw new Error('Describe the animation or choose an optional preset.')
  return [
    sequence ? 'Animated typography sequence. Show exactly ONE phrase at a time in the order below. Never combine phrases into a complete title, list, stacked layout or final recap. Preserve the spelling, punctuation and language of each phrase; no extra lettering.'
      : `Animated title VFX. The ONLY visible lettering is exactly ${JSON.stringify(text)}. Preserve every word, spelling and punctuation; no extra text.`,
    settings.look.trim() ? `Lettering and visual style: ${settings.look.trim()}` : 'Clear, readable lettering.',
    sequence ? [
      `Phrase schedule for this ${sequence.duration.toFixed(3)}-second clip (each slot includes entrance, readable hold and complete exit):`,
      ...sequence.phrases.map((phrase, index) => `${index + 1}. ${phrase.start.toFixed(3)}–${phrase.end.toFixed(3)}s: The ONLY readable text is ${JSON.stringify(phrase.text)}. ${phrase.animation.trim() || 'Reveal this phrase, hold it legibly, then fade it out completely.'} Clear every letter and recognizable fragment before the next phrase appears.`),
      `Final slot: show ONLY ${JSON.stringify(sequence.phrases.at(-1)!.text)}; no earlier phrase returns. Fade all lettering and fragments to the supplied background before the final frame.`,
      settings.sequenceDirection?.trim() ? `Additional motion direction, applied separately to each phrase within its slot: ${settings.sequenceDirection.trim()}` : '',
      'The phrase schedule and single-phrase visibility take priority over any style or animation direction suggesting repeated, overlapping or combined titles.',
    ].filter(Boolean).join('\n') : animation,
    background.chromaKey
      ? 'Start and end on the supplied perfectly uniform bright green background (#00FF00). Keep that same flat green backdrop throughout, with no texture, horizon, floor, shadows or lighting changes.'
      : 'Start and end on the supplied perfectly uniform pure black background (#000000). Keep that same flat black backdrop throughout, with no texture, horizon, floor, shadows or lighting changes. The black background is part of the finished title card.',
    'Only title lettering and its animation effect are visible. No people, faces, characters, bodies, scenery, logos or additional objects. Do not visualize people or objects mentioned by the title words.',
    sequence ? 'The typography itself is the visual subject. Give the current phrase a readable hold within its own slot. Transitions must clear the previous phrase completely; never overlap readable words from different slots.'
      : 'The typography itself is the visual subject, not captions over a scene. Animate the lettering according to the requested motion; give each word or requested text group a clear readable moment. Include a complete-title hold unless the animation description explicitly requests otherwise. Repeated or rearranged text uses only the supplied title words.',
    'Keep important readable lettering inside the frame with safe margins; follow any explicit request for intentional edge cropping. Use a locked camera unless the animation description requests a camera move. Preserve the requested lettering style and language. No speech or music.',
    background.chromaKey ? 'No green lettering, green particles or green reflections; reserve bright green for the removable backdrop.' : '',
  ].filter(Boolean).join('\n\n')
}

export function animatedTitleParams(settings: AnimatedTitleSettings, canvas: { width: number; height: number }, backgroundPath: string, workspace: string, clientId: string) {
  const dimensions = titleDimensions(settings, canvas)
  const frames = titleFrames(titleSeconds(settings))
  if (!backgroundPath || !clientId) throw new Error('The title background or submission identity is missing.')
  return {
    model_type: TITLE_MODEL, generation_mode: 'video', workspace,
    prompt: animatedTitlePrompt(settings), resolution: `${dimensions.width}x${dimensions.height}`,
    video_length: frames, num_inference_steps: 4, guidance_scale: 1,
    image_start: backgroundPath, image_end: backgroundPath, image_prompt_type: 'SE',
    image_refs: [], video_prompt_type: '', audio_prompt_type: '', image_mode: 0,
    activated_loras: [], loras_multipliers: '', override_attention: 'sla',
    minimax_h3_multi_window: false, minimax_h3_reference_sequence: false,
    minimax_h3_window_storyboard: false, sliding_window_size: frames,
    sliding_window_overlap: 18, sliding_window_discard_last_frames: 0,
    multi_prompts_gen_type: 2, repeat_generation: 1, seed: -1,
    spatial_upsampling: '', film_grain_intensity: 0, custom_settings: {},
    _client_submission_id: clientId,
  }
}

/** One undoable, idempotent insertion above all existing visual layers. */
export function insertAnimatedTitle(project: EditorProject, asset: EditorAsset, options: {
  id: string; at: number; text: string; keepAudio: boolean; itemId: string; trackId: string; background?: TitleBackground
}) {
  const existingTrack = project.tracks.find(track => track.items.some(item => item.animated_title_id === options.id))
  const existing = existingTrack?.items.find(item => item.animated_title_id === options.id)
  if (existingTrack && existing) return { itemId: existing.id, trackId: existingTrack.id, start: existing.start }
  if (asset.type !== 'video' || !Number.isFinite(asset.duration) || asset.duration <= 0) throw new Error('The generated title has no usable video duration.')
  if (!Number.isFinite(options.at) || options.at < 0 || options.at + asset.duration > 86400) throw new Error('The title placement is outside the supported timeline.')
  const background = titleBackground(options)
  const start = options.at
  project.assets[asset.id] = asset
  project.tracks.push({
    id: options.trackId, name: 'Title FX', type: 'video',
    z_index: Math.max(-1, ...project.tracks.filter(track => track.type !== 'audio').map(track => track.z_index)) + 1,
    muted: false, locked: false, volume: 1,
    items: [{ id: options.itemId, asset_id: asset.id, name: options.text,
      animated_title_id: options.id, start, duration: asset.duration, source_in: 0,
      speed: 1, opacity: 1, volume: 1, muted: !options.keepAudio, fit: 'contain',
      transform: { x: 0, y: 0, scale: 1, rotation: 0 }, fade_in: 0, fade_out: 0,
      keying: { ...DEFAULT_EDITOR_KEYING, enabled: background.chromaKey, similarity: .23, softness: .12, spill: .65 },
    }],
  })
  return { itemId: options.itemId, trackId: options.trackId, start }
}
