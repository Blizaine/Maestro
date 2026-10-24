import type { GenerateParams } from '../types'

const KEY = 'maestro_speech_writer'
export function speechWriterPreferences(value: Partial<GenerateParams>): Partial<GenerateParams> {
  const minutes = Number(value._speech_writer_target_minutes ?? 0)
  return {
    _speech_writer_instructions: String(value._speech_writer_instructions ?? '').slice(0, 8000),
    _speech_writer_target_minutes: Number.isFinite(minutes) ? Math.max(0, Math.min(60, minutes)) : 0,
  }
}

export function loadSpeechWriterPreferences(): Partial<GenerateParams> {
  try { return speechWriterPreferences(JSON.parse(localStorage.getItem(KEY) || '{}')) }
  catch { return speechWriterPreferences({}) }
}

export function saveSpeechWriterPreferences(params: GenerateParams) {
  try { localStorage.setItem(KEY, JSON.stringify(speechWriterPreferences(params))) }
  catch { /* Browser storage can be unavailable. Keep the current form usable. */ }
}
