import type { GenerateParams, ViggleCharacterOptions } from '../types'

export const VIGGLE_SWAP_PROMPT = 'Replace the main character in source frame with character in second image. Preserve the exact pose, body orientation, hands, props, background, camera framing, lighting and image dimensions.'

export const newViggleCharacter = (): ViggleCharacterOptions => ({
  reference_path: '', image_model: 'flux2_klein_9b', frame_seconds: 0,
  swap_prompt: VIGGLE_SWAP_PROMPT, appearance_prompt: '',
})

export function vigglePreparationKey(source: unknown, character: unknown, seed: unknown): string {
  return JSON.stringify([source, character, seed])
}

/** Source timestamps stay absolute, even when generation uses a trimmed copy. */
export function viggleTimeline(params: Partial<GenerateParams>) {
  const finite = (value: unknown, fallback = 0) => Number.isFinite(Number(value)) ? Number(value) : fallback
  const duration = Math.max(0, finite(params._viggle_source_seconds))
  const end = Math.min(duration, Math.max(Math.min(0.1, duration), finite(params._viggle_trim_end, duration)))
  const start = Math.min(Math.max(0, end - 0.1), Math.max(0, finite(params._viggle_trim_start)))
  const lastFrame = Math.max(start, end - 1 / 24)
  const frame = Math.max(start, Math.min(lastFrame,
    finite(params.viggle_character?.frame_seconds ?? params._viggle_frame_seconds, start)))
  return {duration, start, end, frame, lastFrame, length: Math.max(0, end - start)}
}
