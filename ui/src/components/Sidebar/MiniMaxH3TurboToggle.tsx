import { Zap } from 'lucide-react'
import { useStore } from '../../stores/useStore'
import { InfoTooltip } from './InfoTooltip'

/** Compact, reproducible preset for Maestro's managed H3 Turbo adapter. */
export function MiniMaxH3TurboToggle() {
  const option = useStore(s => s.modelOptions?.minimax_h3_turbo)
  const enabled = useStore(s => s.params.minimax_h3_turbo_mode === true)
  const currentSteps = useStore(s => s.params.num_inference_steps)
  const defaultSteps = useStore(s => s.modelOptions?.default_num_inference_steps)
  const activatedLoras = useStore(s => s.params.activated_loras)
  const setParam = useStore(s => s.setParam)
  const toggleLora = useStore(s => s.toggleLora)
  const setLoraWeight = useStore(s => s.setLoraWeight)

  // The backend only advertises this option for Full FL2VA / Ref2VA models.
  // Pruned models therefore have no stale or disabled control to explain.
  if (!option) return null

  const handleChange = (checked: boolean) => {
    setParam('minimax_h3_turbo_mode', checked)
    if (checked) {
      if (!activatedLoras.includes(option.filename)) {
        toggleLora(option.filename)
      }
      // toggleLora updates the Zustand store synchronously, so the managed
      // adapter is available to setLoraWeight immediately. It remains a
      // normal selected LoRA in Advanced for user tuning after this default.
      setLoraWeight(option.filename, 0, option.weight)
      setParam('num_inference_steps', option.steps)
    } else {
      if (activatedLoras.includes(option.filename)) {
        toggleLora(option.filename)
      }
      if (currentSteps === option.steps && defaultSteps != null) {
        setParam('num_inference_steps', defaultSteps)
      }
    }
  }

  return (
    <div className={`rounded-lg border px-3 py-2 transition-colors ${
      enabled
        ? 'border-accent-blue/50 bg-accent-blue/10'
        : 'border-border bg-bg-tertiary/50'
    }`}>
      <div className="flex items-center gap-2">
        <label className="flex min-w-0 flex-1 cursor-pointer items-center gap-2 select-none">
          <input
            type="checkbox"
            checked={enabled}
            onChange={event => handleChange(event.target.checked)}
            className="accent-accent-blue"
          />
          <Zap size={13} className={enabled ? 'text-accent-blue' : 'text-text-muted'} />
          <span className="text-[11px] font-medium text-text-primary">
            {option.label}
          </span>
          {option.experimental && (
            <span className="rounded bg-amber-500/15 px-1.5 py-0.5 text-[8px] font-medium uppercase tracking-wider text-indicator-warning">
              Experimental
            </span>
          )}
        </label>
        <InfoTooltip label="About H3 Turbo mode" text={option.guide} />
      </div>
    </div>
  )
}
