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
} as const

export interface AnimatedTitleSettings {
  text: string
  look: string
  effect: keyof typeof TITLE_EFFECTS
  direction: string
  seconds: number
  aspect: 'project' | '16:9' | '9:16' | '1:1'
  resolution: '480p' | '720p'
  keepAudio: boolean
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
  const text = settings.text.trim()
  if (!text || text.length > 160) throw new Error('Enter a title of 1–160 characters.')
  if (!Object.hasOwn(TITLE_EFFECTS, settings.effect)) throw new Error('Choose an animation effect.')
  if (settings.look.length > 1200 || settings.direction.length > 1200) throw new Error('Keep each direction under 1,200 characters.')
  return [
    `Animated title VFX. The ONLY visible lettering is exactly ${JSON.stringify(text)}. Preserve every word, spelling and punctuation; no extra text.`,
    settings.look.trim() ? `Lettering and visual style: ${settings.look.trim()}` : 'Bold, clear, readable lettering.',
    TITLE_EFFECTS[settings.effect],
    settings.direction.trim(),
    'Start and end on the supplied perfectly uniform bright green background (#00FF00). Keep that same flat green backdrop throughout, with no texture, horizon, floor, shadows or lighting changes.',
    'Only title lettering and its animation effect are visible. No people, faces, characters, bodies, scenery, logos or additional objects. Do not visualize people or objects mentioned by the title words.',
    'Keep all lettering and particles within a generous safe margin, centered in the frame. Locked camera. No green lettering, green particles or green reflections. No speech or music.',
  ].filter(Boolean).join('\n\n')
}

export function animatedTitleParams(settings: AnimatedTitleSettings, canvas: { width: number; height: number }, backgroundPath: string, workspace: string, clientId: string) {
  const dimensions = titleDimensions(settings, canvas)
  const frames = titleFrames(settings.seconds)
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
  id: string; at: number; text: string; keepAudio: boolean; itemId: string; trackId: string
}) {
  const existingTrack = project.tracks.find(track => track.items.some(item => item.animated_title_id === options.id))
  const existing = existingTrack?.items.find(item => item.animated_title_id === options.id)
  if (existingTrack && existing) return { itemId: existing.id, trackId: existingTrack.id, start: existing.start }
  if (asset.type !== 'video' || !Number.isFinite(asset.duration) || asset.duration <= 0) throw new Error('The generated title has no usable video duration.')
  if (!Number.isFinite(options.at) || options.at < 0 || options.at + asset.duration > 86400) throw new Error('The title placement is outside the supported timeline.')
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
      keying: { ...DEFAULT_EDITOR_KEYING, enabled: true, similarity: .23, softness: .12, spill: .65 },
    }],
  })
  return { itemId: options.itemId, trackId: options.trackId, start }
}
