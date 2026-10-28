import type { EditorColorSettings } from '../types'

export const DEFAULT_EDITOR_COLOR_SETTINGS: EditorColorSettings = {
  exposure: 0,
  contrast: 1,
  saturation: 1,
  temperature: 0,
}

export function normalizeEditorColorSettings(color?: EditorColorSettings): EditorColorSettings {
  const clamp = (value: number, min: number, max: number, fallback: number) => (
    Math.min(max, Math.max(min, Number.isFinite(value) ? value : fallback))
  )
  return {
    exposure: clamp(color?.exposure ?? 0, -2, 2, 0),
    contrast: clamp(color?.contrast ?? 1, 0, 2, 1),
    saturation: clamp(color?.saturation ?? 1, 0, 2, 1),
    temperature: clamp(color?.temperature ?? 0, -1, 1, 0),
  }
}

/** Matrix values for the SVG temperature filter: red and blue channel gain only. */
export function editorColorMatrix(color?: EditorColorSettings): string {
  const temperature = normalizeEditorColorSettings(color).temperature
  const redGain = (1 + 0.2 * temperature).toFixed(4)
  const blueGain = (1 - 0.2 * temperature).toFixed(4)
  return `${redGain} 0 0 0 0  0 1 0 0 0  0 0 ${blueGain} 0 0  0 0 0 1 0`
}

/** CSS filter order is exposure, contrast, saturation, then SVG temperature matrix. */
export function editorColorFilter(
  color?: EditorColorSettings,
  temperatureFilterId = 'editor-temperature',
): string {
  const settings = normalizeEditorColorSettings(color)
  const filterId = temperatureFilterId.replace(/[^a-zA-Z0-9_-]/g, '') || 'editor-temperature'
  return `brightness(${(2 ** settings.exposure).toFixed(4)}) contrast(${settings.contrast.toFixed(4)}) saturate(${settings.saturation.toFixed(4)}) url(#${filterId})`
}
