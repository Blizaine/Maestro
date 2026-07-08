import { useStore } from '../../stores/useStore'
import { ChoiceControl } from '../shared/ChoiceControl'

export function ModelOptionsPanel() {
  const modelOptions = useStore(s => s.modelOptions)
  const params = useStore(s => s.params)
  const setParam = useStore(s => s.setParam)

  if (!modelOptions) return null

  const {
    sample_solvers,
    flow_shift,
    guidance_max_phases,
    lock_guidance_phases,
    self_refiner,
  } = modelOptions

  const hasAnyOption = sample_solvers ||
    flow_shift || (guidance_max_phases > 1 && !lock_guidance_phases) || self_refiner

  if (!hasAnyOption) return null

  return (
    <div className="space-y-4">
      {/* Sampler / Solver */}
      {sample_solvers && sample_solvers.length > 0 && (
        <ChoiceControl
          config={{ choices: sample_solvers, label: 'Sampler' }}
          value={params.video_prompt_type || sample_solvers[0]?.[1] || ''}
          onChange={val => setParam('video_prompt_type', val)}
          label="Sampler"
        />
      )}

      {/* Guidance Phases */}
      {guidance_max_phases > 1 && !lock_guidance_phases && (
        <div>
          <label className="text-[11px] text-text-muted uppercase tracking-wider mb-1.5 block">
            Guidance Phases
          </label>
          <select
            value={params.guidance_phases ?? 1}
            onChange={e => setParam('guidance_phases', Number(e.target.value))}
            className="w-full bg-bg-tertiary border border-border rounded-lg px-3 py-2 text-sm text-text-primary focus:outline-none focus:border-accent-blue"
          >
            {Array.from({ length: guidance_max_phases }, (_, i) => i + 1).map(n => (
              <option key={n} value={n}>
                {n === 1 ? 'One Phase' : n === 2 ? 'Two Phases' : 'Three Phases'}
              </option>
            ))}
          </select>
        </div>
      )}

      {/* Flow Shift */}
      {flow_shift && (
        <div>
          <div className="flex items-center justify-between mb-1.5">
            <label className="text-[11px] text-text-muted uppercase tracking-wider">Flow Shift</label>
            <input
              type="number"
              value={params.flow_shift ?? 3.0}
              onChange={e => setParam('flow_shift', Number(e.target.value))}
              step={0.5}
              className="w-14 bg-bg-tertiary border border-border rounded px-2 py-0.5 text-xs text-text-primary text-center focus:outline-none"
            />
          </div>
          <input
            type="range"
            min={0}
            max={20}
            step={0.5}
            value={params.flow_shift ?? 3.0}
            onChange={e => setParam('flow_shift', Number(e.target.value))}
          />
        </div>
      )}

      {/* Self Refiner */}
      {self_refiner && (
        <div>
          <label className="text-[11px] text-text-muted uppercase tracking-wider mb-1.5 block">
            Self Refiner
          </label>
          <select
            value={params.self_refiner_setting ?? 0}
            onChange={e => setParam('self_refiner_setting', Number(e.target.value))}
            className="w-full bg-bg-tertiary border border-border rounded-lg px-3 py-2 text-sm text-text-primary focus:outline-none focus:border-accent-blue"
          >
            <option value={0}>Disabled</option>
            <option value={1}>Enabled with P1-Norm</option>
            <option value={2}>Enabled with P2-Norm</option>
          </select>
        </div>
      )}
    </div>
  )
}
