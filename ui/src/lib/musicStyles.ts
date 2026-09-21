import { useStore } from '../stores/useStore'
import type { MusicStyle } from '../api/musicTraining'

export interface SelectedMusicStyle { id: string; strength: number }

export function selectedMusicStyles(settings?: Record<string, unknown>): SelectedMusicStyle[] {
  const raw = Array.isArray(settings?.artist_loras) ? settings.artist_loras
    : settings?.artist_id ? [{id: settings.artist_id, strength: settings.artist_strength ?? 1}] : []
  const seen = new Set<string>()
  return raw.flatMap(value => {
    if (!value || typeof value !== 'object' || typeof value.id !== 'string' || !value.id || seen.has(value.id)) return []
    seen.add(value.id)
    const strength = Number(value.strength ?? 1)
    return [{id: value.id, strength: Number.isFinite(strength) ? Math.min(1.5, Math.max(0, strength)) : 1}]
  })
}

function setMusicStyles(styles: SelectedMusicStyle[]) {
  const {params, setParam} = useStore.getState()
  setParam('custom_settings', {...params.custom_settings, artist_loras: styles,
    artist_id: styles.length === 1 ? styles[0].id : '', artist_strength: styles.length === 1 ? styles[0].strength : 1, abc: ''})
  if (styles.length) {
    setParam('model_mode', 2)
    setParam('audio_prompt_type', '')
    setParam('audio_guide', undefined)
  }
}

// Training auditions explicitly replace the current selection with their bundle.
export function selectMusicStyle(style?: MusicStyle) {
  if (style) useStore.getState().setMusicInstrumental(false)
  setMusicStyles(style ? [{id: style.id, strength: 1}] : [])
}

export function toggleMusicStyle(style: MusicStyle, checked: boolean) {
  const current = selectedMusicStyles(useStore.getState().params.custom_settings)
  setMusicStyles(checked ? current.some(item => item.id === style.id) ? current : [...current, {id: style.id, strength: 1}]
    : current.filter(item => item.id !== style.id))
}

export function setMusicStyleStrength(id: string, strength: number) {
  setMusicStyles(selectedMusicStyles(useStore.getState().params.custom_settings).map(item => item.id === id ? {...item, strength} : item))
}

export function deselectMusicStyle(id: string) {
  const current = selectedMusicStyles(useStore.getState().params.custom_settings)
  if (current.some(item => item.id === id)) setMusicStyles(current.filter(item => item.id !== id))
}
