import type { EditorKeyingSettings } from '../types'

export const DEFAULT_EDITOR_KEYING: EditorKeyingSettings = {
  enabled: false, color: '#00ff00', similarity: 0.15, softness: 0.10, spill: 0.5,
}

function bounded(value: unknown, fallback: number, low = 0): number {
  return typeof value === 'number' && Number.isFinite(value) ? Math.max(low, Math.min(1, value)) : fallback
}

export function normalizeEditorKeying(value?: Partial<EditorKeyingSettings>): EditorKeyingSettings {
  return { enabled: value?.enabled === true,
    color: /^#[0-9a-f]{6}$/i.test(value?.color || '') ? value!.color!.toLowerCase() : DEFAULT_EDITOR_KEYING.color,
    similarity: bounded(value?.similarity, DEFAULT_EDITOR_KEYING.similarity, .00001),
    softness: bounded(value?.softness, DEFAULT_EDITOR_KEYING.softness),
    spill: bounded(value?.spill, DEFAULT_EDITOR_KEYING.spill) }
}

export function editorKeyColor(color: string): [number, number, number] {
  return [1, 3, 5].map(offset => Number.parseInt(color.slice(offset, offset + 2), 16) / 255) as [number, number, number]
}

/** 0 = no spill channel, 1 = green, 2 = blue. Red/custom keys keep their RGB. */
export function editorSpillChannel(color: string): number {
  const [red, green, blue] = editorKeyColor(color)
  if (green > red && green > blue) return 1
  if (blue > red && blue > green) return 2
  return 0
}

/** CPU fallback: RGB-distance key followed by spill reduction; preserve source alpha. */
export function applyEditorKeyPixels(pixels: Uint8ClampedArray, keying: EditorKeyingSettings): void {
  if (!keying.enabled) return
  const key = editorKeyColor(keying.color)
  const spillChannel = editorSpillChannel(keying.color)
  for (let offset = 0; offset < pixels.length; offset += 4) {
    const red = pixels[offset] / 255, green = pixels[offset + 1] / 255, blue = pixels[offset + 2] / 255
    const distance = Math.hypot(red - key[0], green - key[1], blue - key[2]) / Math.sqrt(3)
    const matte = keying.softness > .0001
      ? Math.max(0, Math.min(1, (distance - keying.similarity) / keying.softness))
      : distance > keying.similarity ? 1 : 0
    pixels[offset + 3] = Math.floor(pixels[offset + 3] * matte)
    if (spillChannel === 1) pixels[offset + 1] = Math.floor(255 * Math.max(0, green - keying.spill * Math.max(0, green - (red + blue) / 2)))
    if (spillChannel === 2) pixels[offset + 2] = Math.floor(255 * Math.max(0, blue - keying.spill * Math.max(0, blue - (red + green) / 2)))
  }
}
