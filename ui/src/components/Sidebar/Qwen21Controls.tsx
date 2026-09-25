import { useStore } from '../../stores/useStore'

/** The profile owns both the sampling recipe and its automatically loaded adapter. */
export function Qwen21Controls() {
  const options = useStore(s => s.modelOptions)
  const params = useStore(s => s.params)
  const setParam = useStore(s => s.setParam)
  const workflow = useStore(s => s.studioImageWorkflow)
  if (options?.architecture !== 'qwen_image_21_7B') return null

  const profiles = options.qwen21_acceleration_profiles || {}
  const solver = String(params.sample_solver || 'default')
  const accelerated = !!profiles[solver]
  const steps = profiles[solver]?.steps ?? 40
  const guidance = profiles[solver]?.guidance ?? 4
  return (
    <div className="space-y-3 rounded-lg border border-border bg-bg-tertiary/40 p-3">
      <label className="block space-y-1.5">
        <span className="text-[11px] uppercase tracking-wider text-text-muted">Image acceleration</span>
        <select
          aria-label="Image acceleration"
          value={solver}
          onChange={event => {
            const value = event.target.value
            const profile = profiles[value]
            setParam('sample_solver', value)
            setParam('num_inference_steps', profile?.steps ?? 40)
            setParam('guidance_scale', profile?.guidance ?? 4)
          }}
          className="w-full rounded-md border border-border bg-bg-tertiary px-2 py-2 text-xs text-text-primary"
        >
          <option value="default">Base model · 40 steps</option>
          {Object.entries(profiles).map(([value, profile]) => (
            <option key={value} value={value}>{profile.label}</option>
          ))}
        </select>
      </label>
      <p className="text-[10px] leading-relaxed text-text-muted">
        {accelerated
          ? 'Applies the matching Turbo LoRA at strength 1, its step schedule and CFG 1. The adapter downloads on first generation. Complex edits may work better with the base model; Turbo has limited validation with masks, 2K and many references.'
          : 'Uses the base model without a Turbo adapter. Recommended starting point: 40 steps and CFG 4.'}
      </p>
      {(params.num_inference_steps !== steps || params.guidance_scale !== guidance) && (
        <button type="button" className="text-[11px] text-accent-blue hover:underline"
          onClick={() => {
            setParam('num_inference_steps', steps)
            setParam('guidance_scale', guidance)
          }}>
          Apply recommended steps and guidance
        </button>
      )}
      <label className="flex items-start gap-2 text-[11px] text-text-secondary">
        <input type="checkbox" className="mt-0.5 accent-accent-blue"
          checked={params.custom_settings?.qwen21_kv_cache === 'Enabled'}
          onChange={event => setParam('custom_settings', {
            ...(params.custom_settings || {}),
            qwen21_kv_cache: event.target.checked ? 'Enabled' : 'Disabled',
          })}
        />
        <span>Cache reference attention
          <span className="mt-0.5 block text-[10px] text-text-muted">Faster with references when memory allows; automatically recomputes if the cache would exceed its VRAM budget.</span>
        </span>
      </label>
      {workflow === 'inpaint' && options.model_modes && (
        <label className="block space-y-1.5">
          <span className="text-[11px] uppercase tracking-wider text-text-muted">Inpainting method</span>
          <select aria-label="Inpainting method"
            value={Number(params.model_mode ?? options.model_modes.default)}
            onChange={event => setParam('model_mode', Number(event.target.value))}
            className="w-full rounded-md border border-border bg-bg-tertiary px-2 py-2 text-xs text-text-primary">
            {options.model_modes.choices.map(([label, value]) => <option key={value} value={value}>{label}</option>)}
          </select>
          <span className="block text-[10px] text-text-muted">LanPaint spends extra time refining the masked region at each generation step.</span>
        </label>
      )}
      {workflow === 'inpaint' && Number(params.model_mode ?? 0) === 0 && (
        <label className="block space-y-1.5 text-[11px] text-text-secondary">
          Edit strength <span className="float-right">{Number(params.denoising_strength ?? 1).toFixed(2)}</span>
          <input aria-label="Edit strength" type="range" min={0.05} max={1} step={0.05}
            value={Number(params.denoising_strength ?? 1)}
            onChange={event => setParam('denoising_strength', Number(event.target.value))}
            className="w-full" />
        </label>
      )}
    </div>
  )
}
