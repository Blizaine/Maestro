import { AlertTriangle, Zap } from 'lucide-react'
import { useStore } from '../../stores/useStore'
import { InfoTooltip } from './InfoTooltip'

/** Compact, reproducible preset for Maestro's managed H3 Turbo adapter. */
export function MiniMaxH3TurboToggle() {
  const option = useStore(s => s.modelOptions?.minimax_h3_turbo)
  const advisory = useStore(s => s.modelOptions?.minimax_h3_runtime_advisory)
  const enabled = useStore(s => s.params.minimax_h3_turbo_mode === true)
  const currentSteps = useStore(s => s.params.num_inference_steps)
  const defaultSteps = useStore(s => s.modelOptions?.default_num_inference_steps)
  const activatedLoras = useStore(s => s.params.activated_loras)
  const setParam = useStore(s => s.setParam)
  const toggleLora = useStore(s => s.toggleLora)
  const setLoraWeight = useStore(s => s.setLoraWeight)
  const selectModel = useStore(s => s.selectModel)

  // The backend advertises the same managed adapter for Full and Pruned H3;
  // its loader converts the small AdaLN projection for the selected base.
  if (!option && !advisory) return null

  const handleChange = (checked: boolean) => {
    if (!option) return
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

  const useRecommendedPrunedTurbo = () => {
    const recommendedModel = advisory?.recommended_model_type
    if (!recommendedModel || !option) return

    // Model switching is synchronous in Zustand even though its option/default
    // fetches continue in the background. Rebuild the managed Turbo selection
    // from the new state so a Turbo LoRA active on Full is not mistaken for an
    // adapter that survived selectModel's intentional LoRA reset.
    selectModel(recommendedModel)
    const next = useStore.getState()
    next.setParam('minimax_h3_turbo_mode', true)
    if (!next.params.activated_loras.includes(option.filename)) {
      next.toggleLora(option.filename)
    }
    next.setLoraWeight(option.filename, 0, option.weight)
    next.setParam('num_inference_steps', option.steps)
  }

  return (
    <div className="space-y-2">
      {advisory && (
        <div className="rounded-lg border border-amber-500/40 bg-amber-500/10 px-3 py-2">
          <div className="flex items-start gap-2">
            <AlertTriangle size={13} className="mt-0.5 shrink-0 text-indicator-warning" />
            <div className="min-w-0 flex-1">
              <div className="text-[11px] font-medium text-text-primary">
                {advisory.title}
              </div>
              <p className="mt-1 text-[10px] leading-relaxed text-text-secondary">
                {advisory.message}
              </p>
              {advisory.recommended_model_type && option && (
                <button
                  type="button"
                  onClick={useRecommendedPrunedTurbo}
                  className="mt-2 rounded-md bg-amber-500/20 px-2 py-1 text-[10px] font-medium text-indicator-warning transition-colors hover:bg-amber-500/30"
                >
                  Use Pruned Turbo
                </button>
              )}
            </div>
          </div>
        </div>
      )}

      {option && (
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
      )}
    </div>
  )
}
