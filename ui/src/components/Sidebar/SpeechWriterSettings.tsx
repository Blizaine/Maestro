import { useStore } from '../../stores/useStore'

export function SpeechWriterSettings() {
  const mode = useStore(s => s.generationMode)
  const audioMode = useStore(s => s.audioSubMode)
  const supported = useStore(s => s.modelOptions?.speech_writer_controls === true)
  const instructions = useStore(s => s.params._speech_writer_instructions ?? '')
  const target = useStore(s => s.params._speech_writer_target_minutes ?? 0)
  const setParam = useStore(s => s.setParam)
  if (mode !== 'audio' || audioMode !== 'speech' || !supported) return null
  return <fieldset className="space-y-3 rounded-lg border border-border p-3">
    <legend className="px-1 text-[11px] uppercase tracking-wider text-text-muted">Speech writer</legend>
    <label className="block space-y-1.5">
      <span className="text-xs text-text-secondary">Writing instructions (system prompt)</span>
      <textarea aria-label="Speech writing instructions" rows={5} maxLength={8000} value={instructions}
        onChange={event => setParam('_speech_writer_instructions', event.target.value)}
        placeholder="Tone, personality, storytelling style and dialogue preferences…"
        className="w-full resize-y rounded-lg border border-border bg-bg-tertiary px-3 py-2 text-xs text-text-primary focus:border-accent-blue focus:outline-none" />
    </label>
    <p className="text-[10px] leading-relaxed text-text-muted">Used by Write for speech and dialogue. These instructions guide the writer and are not spoken. The selected voice and speech model still control delivery.</p>
    <label className="flex items-center justify-between gap-3 text-xs text-text-secondary">
      Draft length (minutes)
      <input aria-label="Speech draft length in minutes" type="number" min={0} max={60} step={1} value={target}
        onChange={event => {
          const value = Number(event.target.value)
          if (Number.isFinite(value)) setParam('_speech_writer_target_minutes', Math.max(0, Math.min(60, value)))
        }} className="w-20 rounded-lg border border-border bg-bg-tertiary px-2 py-1.5 text-text-primary" />
    </label>
    <p className="text-[10px] leading-relaxed text-text-muted">0 follows your brief. For example, request a ten-minute story or dialogue. Long drafts are written in sections and returned as one editable script. Length is approximate; this does not set the audio generation limit.</p>
  </fieldset>
}
