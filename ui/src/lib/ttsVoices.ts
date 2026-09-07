import type { GenerateParams, ModelOptions, TtsVoice } from '../types'

export function ttsVoiceLimit(options: ModelOptions | null): number {
  if (!options?.audio_only || !options.any_audio_prompt) return 0
  const declared = options.max_voice_count
  if (typeof declared === 'number') return Math.max(0, Math.min(6, declared))
  if (options.architecture === 'kugelaudio_0_open') return 6
  return options.audio_prompt_type_sources?.selection?.some(mode => mode.includes('B')) ? 2 : 1
}

export function ttsAudioModeForCount(count: number, options: ModelOptions | null, current = ''): string {
  const selection = options?.audio_prompt_type_sources?.selection ?? ['A']
  const mode = current.replace(/[NV]/g, '')
  if (count === 0) return selection.includes('') ? '' : selection[0] ?? ''
  if (count === 1) return selection.find(value => value.includes('A') && !value.includes('B')) ?? 'A'
  // IndexTTS uses AB for voice + emotion and AB2 for two distinct speakers.
  // Keep an explicitly selected emotion mode; adding a second voice defaults to dialogue.
  if (mode.includes('B') && selection.includes(mode)) return mode
  return selection.find(value => value.includes('B') && value.includes('2'))
    ?? selection.find(value => value.includes('B')) ?? 'A'
}

export function ttsVoicePaths(voices: TtsVoice[], count: number): Partial<GenerateParams> {
  const paths: Record<string, string | undefined> = {}
  for (let i = 0; i < 6; i++) {
    paths[i === 0 ? 'audio_guide' : `audio_guide${i + 1}`] = i < count ? voices[i]?.path || undefined : undefined
  }
  return paths
}

export function ttsSpeakingVoiceCount(count: number, options: ModelOptions | null, mode: string): number {
  const limit = options?.architecture === 'index_tts2' && mode.replace(/[NV]/g, '') === 'AB'
    ? 1 : ttsVoiceLimit(options)
  return Math.min(count, limit)
}

export function ttsCharacterEnhancePrompt(prompt: string, voices: TtsVoice[], count: number): string {
  const active = voices.slice(0, count)
  if (!active.some(voice => voice.characterId)) return prompt
  const bindings = active.map((voice, i) => `Speaker ${i + 1}: ${JSON.stringify(voice.name || `Speaker ${i + 1}`)}`).join('\n')
  return `Saved character voice bindings:\n${bindings}\nUse these exact speaker names for dialogue labels. Keep each speaker's lines assigned to that speaker. Their voice references are already attached. Do not invent replacement speakers or speak these instructions.\n\nRequested speech:\n${prompt}`
}

export function applyTtsVoices(params: Record<string, unknown>, voices: TtsVoice[], count: number, options: ModelOptions | null): void {
  const active = voices.slice(0, Math.min(count, ttsVoiceLimit(options)))
  params._tts_original_prompt = params.prompt
  params._tts_voice_count = active.length
  Object.assign(params, ttsVoicePaths(active, active.length))
  const flags = String(params.audio_prompt_type || '').replace(/[^NV]/g, '')
  params.audio_prompt_type = active.length
    ? ttsAudioModeForCount(active.length, options, String(params.audio_prompt_type || '')) + flags
    : (ttsVoiceLimit(options) ? ttsAudioModeForCount(0, options) : '')
  for (let i = 0; i < 6; i++) {
    params[`_tts_speaker_name${i + 1}`] = active[i]?.name || ''
    params[`_tts_character_id${i + 1}`] = active[i]?.characterId || ''
    params[`_tts_character_name${i + 1}`] = active[i]?.characterName || ''
    params[`_tts_voice_filename${i + 1}`] = active[i]?.filename || ''
  }
  // Match whole speaker labels once. Sequential substitutions can remap a
  // generated "Speaker 1:" label or names that are suffixes of other names.
  const speakers = active.slice(0, ttsSpeakingVoiceCount(active.length, options, String(params.audio_prompt_type)))
  const names = new Map(speakers.map((voice, i) => [voice.name.trim().toLocaleLowerCase(), i]))
  params.prompt = String(params.prompt || '').replace(/^([ \t]*)([^\n:]+?)\s*:/gm, (label, space, name: string) => {
    const index = options?.architecture === 'chatterbox' && /^Speaker 1$/i.test(name.trim())
      ? 0 : names.get(name.trim().toLocaleLowerCase())
    if (index === undefined) return label
    return options?.architecture === 'chatterbox' ? space : `${space}Speaker ${index + 1}:`
  })
}
