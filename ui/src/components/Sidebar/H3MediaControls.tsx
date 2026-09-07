import { useStore } from '../../stores/useStore'

export function H3MediaControls() {
  const params = useStore(state => state.params)
  const options = useStore(state => state.modelOptions)
  const setParam = useStore(state => state.setParam)
  if (!String(options?.architecture || '').startsWith('minimax_h3') || options?.audio_only) return null
  const custom = params.custom_settings || {}
  const source = params.audio_prompt_type || ''
  const sourceAudio = (!options?.omni_reference && /[AK]/.test(source)) || source.includes('D')
  const fixed = String(params.model_type).includes('fused_turbo') || (params.activated_loras || []).some(path => /pdd|acc.?lora/i.test(path))
  const blocked = sourceAudio || fixed
  const enabled = custom.audio_refinement === 'enabled'
  const set = (key: string, value: string) => setParam('custom_settings', { ...custom, [key]: value })
  return <div className="space-y-3 text-xs border-t border-border pt-3">
    <label className="flex items-start gap-2">
      <input type="checkbox" checked={enabled} disabled={blocked && !enabled}
        onChange={event => set('audio_refinement', event.target.checked ? 'enabled' : 'none')} />
      <span>Audio Refinement Extra Phase</span>
    </label>
    <p className="text-[10px] text-text-muted">{blocked
      ? 'Available with generated audio on ordinary H3 checkpoints, without a fixed PDD adapter.'
      : 'Adds six steps at 0.5 denoising strength. Keeps the generated video fixed and refines the audio without LoRAs or re-injecting references.'}</p>
    {(params.video_prompt_type || '').includes('G') && <label className="block space-y-1">
      <span>Mask denoising</span>
      <select className="w-full bg-bg-tertiary border border-border rounded p-2"
        value={String(custom.h3_mask_mode || 'grouped_rows')} onChange={event => set('h3_mask_mode', event.target.value)}>
        <option value="grouped_rows">Grouped Rows (clean protected region)</option>
        <option value="shared_timestep">Shared Timestep (legacy)</option>
      </select>
    </label>}
  </div>
}
