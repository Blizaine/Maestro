import type { ViggleCharacterOptions } from '../types'

export const VIGGLE_SWAP_PROMPT = 'Replace the main character in source frame with character in second image. Preserve the exact pose, body orientation, hands, props, background, camera framing, lighting and image dimensions.'

export const newViggleCharacter = (): ViggleCharacterOptions => ({
  reference_path: '', image_model: 'flux2_klein_9b', frame_seconds: 0,
  swap_prompt: VIGGLE_SWAP_PROMPT, appearance_prompt: '',
})

export function vigglePreparationKey(source: unknown, character: unknown, seed: unknown): string {
  return JSON.stringify([source, character, seed])
}
