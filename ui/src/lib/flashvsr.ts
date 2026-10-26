import type { EditorUpscaleMethod } from '../types'

// Versioned IDs preserve the model used by saved jobs and Editor projects.
// A global preference must never change the meaning of a saved v1.1 method.
export const flashvsrOptions: Array<{ value: Exclude<EditorUpscaleMethod, ''>; label: string; scale: number }> = [
  { value: 'flashvsr2', label: 'FlashVSR v1.1 2×', scale: 2 },
  { value: 'flashvsr3', label: 'FlashVSR v1.1 3×', scale: 3 },
  { value: 'flashvsr4', label: 'FlashVSR v1.1 4×', scale: 4 },
  { value: 'flashvsr2pass2', label: 'FlashVSR v1.1 two-pass 2×', scale: 2 },
  { value: 'flashvsr2pass4', label: 'FlashVSR v1.1 two-pass 4×', scale: 4 },
  { value: 'flashvsrplus2', label: 'FlashVSR+ Turbo 2×', scale: 2 },
  { value: 'flashvsrplus3', label: 'FlashVSR+ Turbo 3×', scale: 3 },
  { value: 'flashvsrplus4', label: 'FlashVSR+ Turbo 4×', scale: 4 },
  { value: 'flashvsrplus2pass2', label: 'FlashVSR+ Turbo two-pass 2×', scale: 2 },
  { value: 'flashvsrplus2pass4', label: 'FlashVSR+ Turbo two-pass 4×', scale: 4 },
]

export function flashvsrScale(method: EditorUpscaleMethod): number {
  return flashvsrOptions.find(option => option.value === method)?.scale ?? 1
}
