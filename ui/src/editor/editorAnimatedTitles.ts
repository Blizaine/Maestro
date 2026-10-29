import type { EditorAsset, EditorProject, ModelDef, ModelOptions } from '../types'
import { DEFAULT_EDITOR_KEYING } from './editorKeying'

export const TITLE_MODEL = 'minimax_h3_fused_turbo'
export const MAX_TITLE_BRIEF_CHARS = 16_000
// JSON can escape one input character into six characters in a saved plan key.
export const MAX_TITLE_PLAN_KEY_CHARS = MAX_TITLE_BRIEF_CHARS * 6 + 8_000
export const TITLE_BRIEF_LENGTH_ERROR = `Describe your title animation in 1–${MAX_TITLE_BRIEF_CHARS.toLocaleString('en-US')} characters.`
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

export type TitleBackground = 'chroma' | 'blue' | 'black' | 'white' | 'custom'
export interface TitleModelProfile {
  modelType: string
  name: string
  fps: number
  minimumFrames: number
  maximumFrames: number
  frameStep: number
  nativeH3: boolean
  startFrame: boolean
  endFrame: boolean
}

const LEGACY_H3_TITLE_PROFILE: TitleModelProfile = {
  modelType: TITLE_MODEL, name: 'H3 Fused 4-Step — Frames', fps: 24,
  minimumFrames: 124, maximumFrames: 345, frameStep: 17,
  nativeH3: true, startFrame: true, endFrame: true,
}

function profileChoices(profile: TitleModelProfile) {
  const frames: number[] = []
  const minimum = Math.max(profile.minimumFrames, Math.ceil(5 * profile.fps))
  const maximum = Math.min(profile.maximumFrames, Math.floor(14.5 * profile.fps))
  const first = profile.minimumFrames + Math.max(0, Math.ceil((minimum - profile.minimumFrames) / profile.frameStep)) * profile.frameStep
  for (let value = first; value <= maximum; value += profile.frameStep) frames.push(value)
  return frames.map(value => ({ frames: value, seconds: value / profile.fps,
    label: `${(value / profile.fps).toFixed(3)}s · ${value} frames` }))
}

/** Build a bounded title profile from the selected model's current metadata. */
export function titleModelProfile(model: ModelDef, options: ModelOptions): TitleModelProfile {
  const isLegacyH3 = model.model_type === TITLE_MODEL
  const nativeH3 = model.model_type.startsWith('minimax_h3')
  const fps = Number.isFinite(options?.fps) && options.fps > 0 ? options.fps
    : isLegacyH3 && Number.isFinite(model?.fps) && model.fps > 0 ? model.fps : isLegacyH3 ? 24 : 0
  const minimumFrames = Number.isInteger(options?.frames_minimum) && options.frames_minimum > 0
    ? options.frames_minimum : isLegacyH3 ? LEGACY_H3_TITLE_PROFILE.minimumFrames : 0
  const frameStep = Number.isInteger(options?.frames_steps) && options.frames_steps > 0
    ? options.frames_steps : isLegacyH3 ? LEGACY_H3_TITLE_PROFILE.frameStep : 0
  const maximumFrames = options?.frames_maximum === null || options?.frames_maximum === undefined
    ? (isLegacyH3 ? LEGACY_H3_TITLE_PROFILE.maximumFrames
      : minimumFrames > 0 && frameStep > 0 && fps > 0
        ? minimumFrames + Math.max(0, Math.floor((14.5 * fps - minimumFrames) / frameStep)) * frameStep : 0)
    : options.frames_maximum
  if (!model?.model_type || !model.name?.trim() || !Number.isFinite(fps) || fps < 1 || fps > 120
    || !Number.isInteger(minimumFrames) || minimumFrames <= 0 || minimumFrames > 1_000_000
    || !Number.isInteger(maximumFrames) || maximumFrames < minimumFrames || maximumFrames > 1_000_000
    || !Number.isInteger(frameStep) || frameStep <= 0 || frameStep > 1_000_000) {
    throw new Error('The selected video model has invalid title timing settings.')
  }
  const profile: TitleModelProfile = {
    modelType: model.model_type, name: model.name, fps, minimumFrames, maximumFrames, frameStep,
    nativeH3,
    startFrame: isLegacyH3 || model.is_i2v === true,
    endFrame: isLegacyH3 || (model.is_i2v === true && (model.supports_end_frame === true || options?.supports_end_frame === true)),
  }
  if (!profileChoices(profile).length) throw new Error('The selected video model has no native title duration between 5 and 14.5 seconds.')
  return profile
}

/** Validate a persisted profile before using it to restore a Studio draft. */
export function normalizeTitleModelProfile(raw: unknown): TitleModelProfile | undefined {
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) return undefined
  const value = raw as Record<string, unknown>
  if (typeof value.modelType !== 'string' || !value.modelType.trim() || value.modelType.length > 160
    || typeof value.name !== 'string' || !value.name.trim() || value.name.length > 240
    || typeof value.fps !== 'number' || !Number.isFinite(value.fps) || value.fps < 1 || value.fps > 120
    || typeof value.minimumFrames !== 'number' || !Number.isInteger(value.minimumFrames) || value.minimumFrames <= 0 || value.minimumFrames > 1_000_000
    || typeof value.maximumFrames !== 'number' || !Number.isInteger(value.maximumFrames) || value.maximumFrames < value.minimumFrames || value.maximumFrames > 1_000_000
    || typeof value.frameStep !== 'number' || !Number.isInteger(value.frameStep) || value.frameStep <= 0 || value.frameStep > 1_000_000
    || typeof value.nativeH3 !== 'boolean' || value.nativeH3 !== (value.modelType.startsWith('minimax_h3'))
    || typeof value.startFrame !== 'boolean' || typeof value.endFrame !== 'boolean'
    || (value.endFrame && !value.startFrame)) return undefined
  const profile = value as unknown as TitleModelProfile
  return profileChoices(profile).length ? profile : undefined
}

function settingsProfile(settings?: Pick<AnimatedTitleSettings, 'model'>): TitleModelProfile {
  if (!settings?.model) return LEGACY_H3_TITLE_PROFILE
  const profile = normalizeTitleModelProfile(settings.model)
  if (!profile) throw new Error('Choose a supported video model before generating this title.')
  return profile
}

export function titleModelType(settings?: Pick<AnimatedTitleSettings, 'model'>): string {
  return settings?.model ? settingsProfile(settings).modelType : TITLE_MODEL
}

export function titleFps(settings?: Pick<AnimatedTitleSettings, 'model'>): number {
  return settings?.model ? settingsProfile(settings).fps : 24
}

export function titleDurationChoices(settings?: Pick<AnimatedTitleSettings, 'model'>) {
  return profileChoices(settings?.model ? settingsProfile(settings) : LEGACY_H3_TITLE_PROFILE)
}

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
  directDescription?: boolean
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
  aspect: 'project' | '21:9' | '16:9' | '9:16' | '1:1' | '4:3' | '3:4'
  resolution: '480p' | '540p' | '720p' | '768p' | '1080p'
  keepAudio: boolean
  /** Missing on older saved title intents, which always used green keying. */
  background?: TitleBackground
  backgroundColor?: string
  /** Missing on old Editor intents, which always used the native H3 Frames recipe. */
  model?: TitleModelProfile
  generation?: { steps?: number; guidance?: number; seed?: number }
  /** Frozen standard Studio Advanced controls; title inputs remain owned by the composer. */
  generationParams?: Record<string, unknown>
}

const TITLE_COMPOSER_INPUT_KEYS = new Set([
  'prompt', 'alt_prompt', 'model_type', 'resolution', 'video_length', 'image_mode', 'generation_mode',
  'workspace', 'settings_version', 'image_start', 'image_end', 'image_refs', 'image_guide', 'image_mask',
  'image_prompt_type', 'video_prompt_type', 'audio_prompt_type', 'video_guide', 'video_mask', 'video_source',
  'video_frame_offset', 'video_guide_outpainting', 'voice_reference', 'frames_positions', 'per_clip_frames',
  'duration_seconds', 'remove_background_images_ref', 'minimax_h3_references', 'minimax_h3_control_visual_mode',
  'minimax_h3_multi_window', 'minimax_h3_reference_sequence', 'minimax_h3_window_storyboard',
  'minimax_h3_extended_duration', 'h3_window_prompts', 'h3_window_plan', 'h3_window_plan_signature',
  'ltx_multi_window', 'ltx_window_prompt_mode', 'ltx_window_prompts',
  'sliding_window_size', 'sliding_window_overlap', 'sliding_window_discard_last_frames', 'sliding_window_memory_override',
  'multi_prompts_gen_type', 'input_video_strength', 'speakers_locations', 'viggle_character', 'audio_mixer_tracks',
])

/** Keep the ordinary generation controls without importing scene inputs or recursive UI metadata. */
export function titleAdvancedParams(value: unknown): Record<string, unknown> {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return {}
  const result = Object.fromEntries(Object.entries(value).filter(([key, item]) => item !== undefined
    && !key.startsWith('_') && !TITLE_COMPOSER_INPUT_KEYS.has(key) && !/^audio_guide\d*$/.test(key)))
  return JSON.parse(JSON.stringify(result)) as Record<string, unknown>
}

export interface TitleCreativePlan {
  phrases: { text: string; animation: string; weight: number }[]
  look: string; direction: string; sequence: boolean
}

/** Titles own their planner; never route through the ordinary video enhancer. */
export function titleEnhancementEnabled(state: { enhanceOnGeneration: boolean | null; enhanceOnGenerationDefault: boolean }): boolean {
  return state.enhanceOnGeneration ?? state.enhanceOnGenerationDefault
}

/** The ordinary no-enhancement path: send the authored description to H3. */
export function directTitleDescription(settings: AnimatedTitleSettings): AnimatedTitleSettings {
  const frames = titleFrames(titleSeconds(settings), settings)
  const brief = titleBrief(settings)
  const quoted = [...brief.matchAll(/“([^”]+)”|‘([^’]+)’|"([^"\n]+)"/g)].map(match => match.slice(1).find(value => value !== undefined) ?? '').join('\n')
  const direct: AnimatedTitleSettings = { ...settings, brief, directDescription: true,
    text: quoted.length <= 160 ? quoted : '', mode: 'single', phrases: undefined, look: '', direction: '', sequenceDirection: '',
    seconds: frames / titleFps(settings), sequenceFrames: frames, effect: 'custom', titleReviewRequired: false,
    plannerWarnings: [], rawTitleDraft: undefined, titleDraftHistoryId: undefined }
  direct.plannedKey = titlePlanningKey(direct)
  animatedTitlePrompt(direct)
  return direct
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
    titleFrames(titleSeconds(settings), settings), titleModelType(settings), titleFps(settings),
    settings.model ? [settings.model.minimumFrames, settings.model.maximumFrames, settings.model.frameStep,
      settings.model.nativeH3, settings.model.startFrame, settings.model.endFrame] : null,
    titleBackground(settings).plannerToken])
}

function legacyTitlePlanningKey(settings: AnimatedTitleSettings): string {
  return JSON.stringify([titleBrief(settings).trim(), settings.stylePreset ?? 'custom',
    titleFrames(titleSeconds(settings)), settings.background ?? 'chroma'])
}

export function titlePlanIsCurrent(settings: AnimatedTitleSettings): boolean {
  if (settings.brief === undefined) return true
  if (settings.plannedKey === titlePlanningKey(settings)) return true
  // Older H3 drafts and durable Editor receipts used this compact key. Keep
  // those plans valid when the settings have no selected-model snapshot.
  return !settings.model && (settings.background === undefined || settings.background === 'chroma' || settings.background === 'black')
    && settings.plannedKey === legacyTitlePlanningKey(settings)
}

/** Creative descriptions are allowed before a concrete render prompt exists. */
export function titleComposerValidation(settings: AnimatedTitleSettings, canvas: { width: number; height: number }): string {
  try {
    if (settings.manualPlanEdited) return 'Choose Use manual title plan to keep your Advanced edits, or Enhance to create a new AI plan.'
    titleDimensions(settings, canvas); titleFrames(titleSeconds(settings), settings); titleBackground(settings)
    if (settings.brief !== undefined && !titlePlanIsCurrent(settings)) {
      if (!settings.brief.trim() || settings.brief.length > MAX_TITLE_BRIEF_CHARS) return TITLE_BRIEF_LENGTH_ERROR
      return ''
    }
    return titleFormState(settings, canvas).validation
  } catch (reason) { return reason instanceof Error ? reason.message : 'Check the title settings.' }
}

export function applyTitleCreativePlan(settings: AnimatedTitleSettings, plan: TitleCreativePlan): AnimatedTitleSettings {
  const frames = titleFrames(titleSeconds(settings), settings)
  const fps = titleFps(settings)
  const phrases = plan.phrases.map(phrase => ({ id: titleSubmissionId(), text: phrase.text,
    seconds: Number.isFinite(phrase.weight) && phrase.weight > 0 ? Math.min(10, phrase.weight) : 1, animation: phrase.animation }))
  let result: AnimatedTitleSettings = { ...settings, brief: titleBrief(settings), directDescription: false, manualPlanEdited: false, effect: 'custom', look: plan.look,
    text: phrases.map(phrase => phrase.text).join(plan.sequence ? '\n' : ' '),
    direction: phrases[0]?.animation || plan.direction || 'Reveal the title, hold it clearly, then clear the frame.',
    sequenceDirection: plan.direction, seconds: frames / fps, mode: plan.sequence && phrases.length > 1 ? 'sequence' : 'single',
    phrases: plan.sequence && phrases.length > 1 ? phrases : undefined, sequenceFrames: frames }
  if (result.mode === 'sequence') result = alignTitleSequence(result, frames)
  result.plannedKey = titlePlanningKey(result)
  animatedTitlePrompt(result)
  return result
}

/** Legacy H3 choices remain exported for Editor controls and old drafts. */
export const TITLE_SEQUENCE_DURATIONS = profileChoices(LEGACY_H3_TITLE_PROFILE)

function nativeSequenceFrames(frames: number, settings?: Pick<AnimatedTitleSettings, 'model'>): number {
  if (!titleDurationChoices(settings).some(choice => choice.frames === frames)) throw new Error('Choose a native duration supported by the selected video model.')
  return frames
}

/** Proportional integer-frame allocation with a one-second minimum per phrase. */
function allocatePhraseFrames(weights: number[], total: number, fps: number): number[] {
  const minimum = Math.ceil(fps)
  const maximum = Math.floor(10 * fps)
  if (total < weights.length * minimum || total > weights.length * maximum) throw new Error('The phrases do not fit the selected clip length.')
  const result = weights.map(() => 0)
  const safeWeights = weights.map(weight => Number.isFinite(weight) && weight > 0 ? Math.min(10, weight) : 1)
  let remaining = total
  let active = weights.map((_, index) => index)
  while (active.length) {
    const sum = active.reduce((value, index) => value + safeWeights[index], 0)
    const fixed = active.find(index => remaining * safeWeights[index] / sum < minimum || remaining * safeWeights[index] / sum > maximum)
    if (fixed !== undefined) {
      result[fixed] = remaining * safeWeights[fixed] / sum < minimum ? minimum : maximum
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
  const fps = titleFps(settings)
  const requested = frames ?? settings.sequenceFrames ?? titleFrames(Math.max(5, Math.min(14.5,
    phrases.reduce((sum, phrase) => sum + (Number.isFinite(phrase.seconds) ? phrase.seconds : 1), 0))), settings)
  if (frames !== undefined || settings.sequenceFrames !== undefined) nativeSequenceFrames(requested, settings)
  // Longer clips may be needed to retain one second per phrase.
  const choices = titleDurationChoices(settings).filter(choice => choice.frames >= phrases.length * Math.ceil(fps)
    && choice.frames <= phrases.length * Math.floor(10 * fps))
  if (!choices.length) throw new Error('The phrases do not fit a native clip length for the selected video model.')
  const selected = choices.reduce((best, choice) => Math.abs(choice.frames - requested) < Math.abs(best.frames - requested) ? choice : best)
  const counts = allocatePhraseFrames(phrases.map(phrase => phrase.seconds), selected.frames, fps)
  return { ...settings, sequenceFrames: selected.frames,
    phrases: phrases.map((phrase, index) => ({ ...phrase, seconds: counts[index] / fps })) }
}

/** Keep the edited slot; redistribute the rest without changing the native clip length. */
export function setTitlePhraseSeconds(settings: AnimatedTitleSettings, id: string, seconds: number): AnimatedTitleSettings {
  const aligned = alignTitleSequence(settings)
  const fps = titleFps(aligned)
  const phrases = aligned.phrases!
  const index = phrases.findIndex(phrase => phrase.id === id)
  if (index < 0 || !Number.isFinite(seconds)) return aligned
  const frames = aligned.sequenceFrames!
  const minimum = Math.ceil(fps), maximum = Math.floor(10 * fps)
  const selected = Math.max(minimum, frames - (phrases.length - 1) * maximum,
    Math.min(maximum, frames - (phrases.length - 1) * minimum, Math.round(seconds * fps)))
  const others = phrases.filter((_, position) => position !== index)
  const counts = allocatePhraseFrames(others.map(phrase => phrase.seconds), frames - selected, fps)
  let otherIndex = 0
  return { ...aligned, phrases: phrases.map((phrase, position) => ({ ...phrase,
    seconds: (position === index ? selected : counts[otherIndex++]) / fps })) }
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
  if (settings.directDescription && !settings.text.trim()) return 'Title animation'
  return settings.mode === 'sequence' ? (settings.phrases ?? []).map(phrase => phrase.text.trim()).join(' → ') : settings.text.trim()
}

export function titleSeconds(settings: AnimatedTitleSettings): number {
  const fps = titleFps(settings)
  return settings.mode === 'sequence' ? settings.sequenceFrames !== undefined ? settings.sequenceFrames / fps
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
  const fps = titleFps(settings)
  const requestedSeconds = Math.round(phrases.reduce((sum, phrase) => sum + phrase.seconds, 0) * 1000) / 1000
  const frames = settings.sequenceFrames === undefined ? titleFrames(requestedSeconds, settings) : nativeSequenceFrames(settings.sequenceFrames, settings)
  const counts = settings.sequenceFrames === undefined ? null : allocatePhraseFrames(phrases.map(phrase => phrase.seconds), frames, fps)
  let elapsed = 0
  let elapsedFrames = 0
  return { requestedSeconds, duration: frames / fps, phrases: phrases.map((phrase, index) => {
    const start = counts ? elapsedFrames / fps : Math.round(elapsed / requestedSeconds * frames) / fps
    elapsed += phrase.seconds
    elapsedFrames += counts?.[index] ?? 0
    const end = counts ? elapsedFrames / fps : index === phrases.length - 1 ? frames / fps : Math.round(elapsed / requestedSeconds * frames) / fps
    return { ...phrase, seconds: counts ? counts[index] / fps : phrase.seconds, text: phrase.text.trim(), start, end }
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
    duration = titleFrames(titleSeconds(settings), settings) / titleFps(settings)
    prompt = animatedTitlePrompt(settings)
  } catch (reason) { validation = reason instanceof Error ? reason.message : 'Check the title settings.' }
  return { geometry, prompt, duration, validation }
}

/** One choice governs the prompt, boundary image and Editor keying. */
export interface ResolvedTitleBackground {
  mode: TitleBackground
  color: string
  fileLabel: string
  label: string
  chromaKey: boolean
  plannerToken: string
}

export function titleBackground(settings: { background?: TitleBackground; backgroundColor?: string }): ResolvedTitleBackground {
  const mode = settings.background === undefined ? 'chroma' : settings.background
  if (mode === 'chroma') return { mode, color: '#00FF00', fileLabel: 'green', label: 'Green', chromaKey: true, plannerToken: mode }
  if (mode === 'blue') return { mode, color: '#0000FF', fileLabel: 'blue', label: 'Blue', chromaKey: true, plannerToken: mode }
  if (mode === 'black') return { mode, color: '#000000', fileLabel: 'black', label: 'Black', chromaKey: false, plannerToken: mode }
  if (mode === 'white') return { mode, color: '#FFFFFF', fileLabel: 'white', label: 'White', chromaKey: false, plannerToken: mode }
  if (mode === 'custom') {
    if (typeof settings.backgroundColor !== 'string' || !/^#[0-9a-f]{6}$/i.test(settings.backgroundColor)) {
      throw new Error('Enter a custom title background as a six-digit hex color, such as #123ABC.')
    }
    const color = settings.backgroundColor.toUpperCase()
    return { mode, color, fileLabel: `custom-${color.slice(1).toLowerCase()}`, label: `Custom ${color}`,
      chromaKey: false, plannerToken: color }
  }
  throw new Error('Choose a supported title background.')
}

type TitleCanvas = { width: number; height: number }
type StandardTitleAspect = Exclude<AnimatedTitleSettings['aspect'], 'project'>
const H3_TITLE_DIMENSIONS: Record<AnimatedTitleSettings['resolution'], Record<StandardTitleAspect, TitleCanvas>> = {
  '480p': { '21:9': { width: 1120, height: 480 }, '16:9': { width: 864, height: 480 }, '9:16': { width: 480, height: 864 }, '1:1': { width: 640, height: 640 }, '4:3': { width: 640, height: 480 }, '3:4': { width: 480, height: 640 } },
  '540p': { '21:9': { width: 1280, height: 544 }, '16:9': { width: 960, height: 544 }, '9:16': { width: 544, height: 960 }, '1:1': { width: 736, height: 736 }, '4:3': { width: 736, height: 544 }, '3:4': { width: 544, height: 736 } },
  '720p': { '21:9': { width: 1632, height: 704 }, '16:9': { width: 1280, height: 704 }, '9:16': { width: 704, height: 1280 }, '1:1': { width: 704, height: 704 }, '4:3': { width: 928, height: 704 }, '3:4': { width: 704, height: 928 } },
  '768p': { '21:9': { width: 1792, height: 768 }, '16:9': { width: 1344, height: 768 }, '9:16': { width: 768, height: 1344 }, '1:1': { width: 768, height: 768 }, '4:3': { width: 1024, height: 768 }, '3:4': { width: 768, height: 1024 } },
  '1080p': { '21:9': { width: 2528, height: 1088 }, '16:9': { width: 1920, height: 1088 }, '9:16': { width: 1088, height: 1920 }, '1:1': { width: 1088, height: 1088 }, '4:3': { width: 1440, height: 1088 }, '3:4': { width: 1088, height: 1440 } },
}

export function titleDimensions(settings: AnimatedTitleSettings, canvas: { width: number; height: number }) {
  const preset = H3_TITLE_DIMENSIONS[settings.resolution]
  if (!preset || (settings.aspect !== 'project' && !Object.hasOwn(preset, settings.aspect))) throw new Error('Choose a supported title resolution and aspect ratio.')
  const ratio = settings.aspect === 'project' ? canvas.width / canvas.height
    : settings.aspect === '1:1' ? 1 : settings.aspect === '9:16' ? 9 / 16
      : settings.aspect === '21:9' ? 21 / 9 : settings.aspect === '4:3' ? 4 / 3
        : settings.aspect === '3:4' ? 3 / 4 : 16 / 9
  if (!Number.isFinite(ratio) || ratio <= 0 || ratio < .25 || ratio > 4) throw new Error('Choose a supported title aspect ratio.')
  // Keep the dimensions used by saved 480p / 720p title settings. New format
  // choices use the same native canvases as MiniMax H3 Video.
  const legacyResolution = settings.resolution === '480p' || settings.resolution === '720p'
  const legacyAspect = settings.aspect === 'project' || settings.aspect === '16:9'
    || settings.aspect === '9:16' || settings.aspect === '1:1'
  if (legacyResolution && legacyAspect) {
    const area = settings.resolution === '480p' ? 832 * 480 : 1280 * 704
    return {
      width: Math.max(32, Math.round(Math.sqrt(area * ratio) / 32) * 32),
      height: Math.max(32, Math.round(Math.sqrt(area / ratio) / 32) * 32),
    }
  }

  if (settings.aspect !== 'project') return preset[settings.aspect]

  const base = preset['16:9']
  const area = base.width * base.height
  return {
    width: Math.max(32, Math.round(Math.sqrt(area * ratio) / 32) * 32),
    height: Math.max(32, Math.round(Math.sqrt(area / ratio) / 32) * 32),
  }
}

export function titleFrames(seconds: number, settings?: Pick<AnimatedTitleSettings, 'model'>): number {
  if (!Number.isFinite(seconds) || seconds < 5 || seconds > 14.5) throw new Error('Title duration must be between 5 and 14.5 seconds.')
  const profile = settings?.model ? settingsProfile(settings) : LEGACY_H3_TITLE_PROFILE
  if (profile.nativeH3 && profile.fps === 24 && profile.minimumFrames === 124
    && profile.maximumFrames === 345 && profile.frameStep === 17) {
    return Math.max(124, Math.min(345, Math.round((seconds * 24 - 5) / 17) * 17 + 5))
  }
  const choices = titleDurationChoices(settings)
  if (!choices.length) throw new Error('The selected video model has no native title duration between 5 and 14.5 seconds.')
  return choices.reduce((best, choice) => Math.abs(choice.seconds - seconds) <= Math.abs(best.seconds - seconds) ? choice : best).frames
}

function titleBackdropDirection(background: ResolvedTitleBackground, settings: AnimatedTitleSettings) {
  const profile = settingsProfile(settings)
  if (profile.nativeH3 && background.mode === 'chroma') {
    return 'Start and end on the supplied perfectly uniform bright green background (#00FF00). Keep that same flat green backdrop throughout, with no texture, horizon, floor, shadows or lighting changes.'
  }
  if (profile.nativeH3 && background.mode === 'black') {
    return 'Start and end on the supplied perfectly uniform pure black background (#000000). Keep that same flat black backdrop throughout, with no texture, horizon, floor, shadows or lighting changes. The black background is part of the finished title card.'
  }
  const hue = background.mode === 'chroma' ? 'bright green' : background.mode === 'blue' ? 'bright blue'
    : background.mode === 'black' ? 'pure black' : background.mode === 'white' ? 'pure white' : `solid ${background.color}`
  const uniform = `uniform ${hue} background (${background.color})`
  const treatment = background.chromaKey
    ? `Keep that same flat ${background.label.toLowerCase()} backdrop throughout, with no texture, horizon, floor, shadows or lighting changes.`
    : `Keep that same flat ${background.label.toLowerCase()} backdrop throughout, with no texture, horizon, floor, shadows or lighting changes. The ${background.label.toLowerCase()} background is part of the finished title card.`
  if (profile.startFrame) {
    const boundary = profile.endFrame ? 'Start and end on the supplied perfectly ' : 'Start with the supplied perfectly '
    return `${boundary}${uniform}. ${treatment}`
  }
  return `Use a perfectly ${uniform} throughout the clip. ${treatment}`
}

function backgroundGlyphConstraint(background: ResolvedTitleBackground) {
  if (background.mode === 'chroma') return 'No green lettering, green particles or green reflections; reserve bright green for the removable backdrop.'
  if (background.mode === 'blue') return 'No blue lettering, blue particles or blue reflections; reserve bright blue for the removable backdrop.'
  if (background.mode === 'custom') return `Do not use ${background.color} for the lettering or its effects; reserve it for the flat background.`
  return ''
}

export function animatedTitlePrompt(settings: AnimatedTitleSettings): string {
  if (settings.manualPlanEdited) throw new Error('Choose Use manual title plan before generating with your Advanced edits.')
  if (!titlePlanIsCurrent(settings)) throw new Error('The description or style changed. Enhance it or review the manual plan before generating.')
  const background = titleBackground(settings)
  if (settings.mode !== undefined && settings.mode !== 'single' && settings.mode !== 'sequence') throw new Error('Choose a supported title layout.')
  if (settings.directDescription) {
    const brief = titleBrief(settings).trim()
    if (!brief || brief.length > MAX_TITLE_BRIEF_CHARS) throw new Error(TITLE_BRIEF_LENGTH_ERROR)
    const preset = settings.stylePreset ?? 'custom'
    if (preset !== 'custom' && !Object.hasOwn(TITLE_EFFECTS, preset)) throw new Error('Choose a supported title style.')
    return [
      'Animated title typography. Follow the user\'s requested title wording, language, order and motion. Preserve exact quoted on-screen text and punctuation; no extra lettering.',
      `User description:\n${brief}`,
      preset === 'custom' ? '' : `Optional animation treatment: ${TITLE_EFFECTS[preset]} Apply this separately to the current phrase within the user's requested sequence; the user's wording and order take priority.`,
      titleBackdropDirection(background, settings),
      'Only title lettering and its animation effect are visible. No people, faces, characters, bodies, scenery, logos or additional objects. Do not visualize people or objects mentioned by the title words.',
      'Keep readable lettering inside the frame with safe margins unless the description requests intentional cropping. Preserve the requested lettering style and language. No speech or music.',
      backgroundGlyphConstraint(background),
    ].filter(Boolean).join('\n\n')
  }
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
    titleBackdropDirection(background, settings),
    'Only title lettering and its animation effect are visible. No people, faces, characters, bodies, scenery, logos or additional objects. Do not visualize people or objects mentioned by the title words.',
    sequence ? 'The typography itself is the visual subject. Give the current phrase a readable hold within its own slot. Transitions must clear the previous phrase completely; never overlap readable words from different slots.'
      : 'The typography itself is the visual subject, not captions over a scene. Animate the lettering according to the requested motion; give each word or requested text group a clear readable moment. Include a complete-title hold unless the animation description explicitly requests otherwise. Repeated or rearranged text uses only the supplied title words.',
    'Keep important readable lettering inside the frame with safe margins; follow any explicit request for intentional edge cropping. Use a locked camera unless the animation description requests a camera move. Preserve the requested lettering style and language. No speech or music.',
    backgroundGlyphConstraint(background),
  ].filter(Boolean).join('\n\n')
}

export interface TitleGenerationContext {
  options?: Pick<ModelOptions, 'model_type' | 'default_num_inference_steps' | 'default_guidance_scale'
    | 'inference_steps_min' | 'inference_steps_max' | 'lock_inference_steps' | 'lock_guidance_scale'>
  defaults?: Record<string, unknown>
}

function defaultNumber(defaults: Record<string, unknown>, key: string, fallback?: number): number | undefined {
  const value = defaults[key]
  if (typeof value === 'number' && Number.isFinite(value)) return value
  return fallback !== undefined && Number.isFinite(fallback) ? fallback : undefined
}

function generationOverrides(settings: AnimatedTitleSettings, context: TitleGenerationContext, legacyH3: boolean) {
  const defaults = context.defaults ?? {}
  const options = context.options
  const stepsDefault = defaultNumber(defaults, 'num_inference_steps', options?.default_num_inference_steps ?? undefined)
    ?? (legacyH3 ? 4 : undefined)
  const guidanceDefault = defaultNumber(defaults, 'guidance_scale', options?.default_guidance_scale ?? undefined)
    ?? (legacyH3 ? 1 : undefined)
  const steps = settings.generation?.steps ?? stepsDefault
  const guidance = settings.generation?.guidance ?? guidanceDefault
  const seed = settings.generation?.seed ?? defaultNumber(defaults, 'seed', -1)
  if (settings.generation?.steps !== undefined && (!Number.isInteger(steps) || steps! < 1
    || (options?.inference_steps_min != null && steps! < options.inference_steps_min)
    || (options?.inference_steps_max != null && steps! > options.inference_steps_max))) {
    throw new Error('Choose an inference step count supported by the selected video model.')
  }
  if (settings.generation?.guidance !== undefined && (!Number.isFinite(guidance) || guidance! < 0)) {
    throw new Error('Choose a valid guidance scale for the selected video model.')
  }
  if (settings.generation?.seed !== undefined && (!Number.isInteger(seed) || seed! < -1)) {
    throw new Error('Choose a valid integer seed or -1 for a random seed.')
  }
  if (options?.lock_inference_steps && settings.generation?.steps !== undefined && steps !== stepsDefault) {
    throw new Error('The selected video model locks its inference step count.')
  }
  if (options?.lock_guidance_scale && settings.generation?.guidance !== undefined && guidance !== guidanceDefault) {
    throw new Error('The selected video model locks its guidance scale.')
  }
  return { steps, guidance, seed }
}

export function animatedTitleParams(settings: AnimatedTitleSettings, canvas: { width: number; height: number }, backgroundPath: string, workspace: string, clientId: string, context: TitleGenerationContext = {}) {
  const dimensions = titleDimensions(settings, canvas)
  const profile = settingsProfile(settings)
  const frames = titleFrames(titleSeconds(settings), settings)
  if ((profile.startFrame && !backgroundPath) || !clientId) throw new Error('The title background or submission identity is missing.')
  const defaults = context.defaults ?? {}
  const params: Record<string, unknown> = {
    model_type: titleModelType(settings), generation_mode: 'video', workspace,
    prompt: animatedTitlePrompt(settings), resolution: `${dimensions.width}x${dimensions.height}`,
    video_length: frames,
    image_refs: [], video_prompt_type: '', audio_prompt_type: '', image_mode: 0,
    activated_loras: [], loras_multipliers: '', repeat_generation: 1, multi_prompts_gen_type: 2, custom_settings: {},
    _client_submission_id: clientId,
  }
  const advanced = titleAdvancedParams(settings.generationParams)
  const generation = { ...settings.generation,
    ...(typeof advanced.num_inference_steps === 'number' ? { steps: advanced.num_inference_steps } : {}),
    ...(typeof advanced.guidance_scale === 'number' ? { guidance: advanced.guidance_scale } : {}),
    ...(typeof advanced.seed === 'number' ? { seed: advanced.seed } : {}),
  }
  const { steps, guidance, seed } = generationOverrides({ ...settings, generation }, context, profile.modelType === TITLE_MODEL)
  if (steps !== undefined) params.num_inference_steps = steps
  if (guidance !== undefined) params.guidance_scale = guidance
  if (seed !== undefined) params.seed = seed
  for (const field of ['sample_solver', 'flow_shift', 'embedded_guidance_scale', 'audio_guidance_scale',
    'sliding_window_size', 'sliding_window_overlap', 'sliding_window_discard_last_frames'] as const) {
    const value = defaults[field]
    if ((typeof value === 'string' && value.length > 0) || (typeof value === 'number' && Number.isFinite(value))) params[field] = value
  }
  if (profile.startFrame) {
    params.image_start = backgroundPath
    params.image_prompt_type = profile.endFrame ? 'SE' : 'S'
    if (profile.endFrame) params.image_end = backgroundPath
  }
  if (!profile.nativeH3) {
    // Use one window matching the chosen clip length so model defaults cannot
    // stretch a title into a longer multi-window generation.
    params.sliding_window_size = frames
    params.sliding_window_overlap = 0
    params.sliding_window_discard_last_frames = 0
  }
  if (profile.nativeH3) {
    params.minimax_h3_multi_window = false
    params.minimax_h3_reference_sequence = false
    params.minimax_h3_window_storyboard = false
    params.sliding_window_size = frames
    if (profile.modelType === TITLE_MODEL) {
      params.override_attention = 'sla'
      params.sliding_window_overlap = 18
      params.sliding_window_discard_last_frames = 0
      params.spatial_upsampling = ''
      params.film_grain_intensity = 0
    } else {
      const attention = defaults.override_attention
      if (attention === 'sol' || attention === 'sla' || attention === 'veda' || attention === 'sdpa' || attention === '') {
        params.override_attention = attention
      }
    }
  } else {
    params.negative_prompt = ''
  }
  if (settings.generationParams) {
    Object.assign(params, advanced)
    // These are the mode-specific inputs: Advanced never changes a title into
    // a source edit, independent prompt batch, or rolling continuation.
    params.multi_prompts_gen_type = 2
    params.sliding_window_size = frames
    if (profile.nativeH3) {
      params.minimax_h3_multi_window = false
      params.minimax_h3_reference_sequence = false
      params.minimax_h3_window_storyboard = false
    } else {
      for (const key of Object.keys(params)) if (key.startsWith('minimax_h3_')) delete params[key]
      if (params.override_attention !== '' && params.override_attention !== 'sdpa') delete params.override_attention
    }
    if (steps !== undefined) params.num_inference_steps = steps
    if (guidance !== undefined) params.guidance_scale = guidance
    if (seed !== undefined) params.seed = seed
  }
  return params
}

/** One undoable, idempotent insertion above all existing visual layers. */
export function insertAnimatedTitle(project: EditorProject, asset: EditorAsset, options: {
  id: string; at: number; text: string; keepAudio: boolean; itemId: string; trackId: string; background?: TitleBackground; backgroundColor?: string
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
      keying: { ...DEFAULT_EDITOR_KEYING, enabled: background.chromaKey,
        color: background.chromaKey ? background.color.toLowerCase() : DEFAULT_EDITOR_KEYING.color,
        similarity: .23, softness: .12, spill: .65 },
    }],
  })
  return { itemId: options.itemId, trackId: options.trackId, start }
}
