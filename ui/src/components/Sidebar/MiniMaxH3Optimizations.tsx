import { useState } from 'react'
import { AlertTriangle, ChevronDown, Gauge, Layers, Zap } from 'lucide-react'
import { useStore } from '../../stores/useStore'
import { InfoTooltip } from './InfoTooltip'

/** Main-sidecar controls for H3's independent speed optimizations. */
export function MiniMaxH3Optimizations() {
  const [expanded, setExpanded] = useState(true)
  const modelOptions = useStore(s => s.modelOptions)
  const option = useStore(s => s.modelOptions?.minimax_h3_turbo)
  const advisory = useStore(s => s.modelOptions?.minimax_h3_runtime_advisory)
  const params = useStore(s => s.params)
  const currentSteps = useStore(s => s.params.num_inference_steps)
  const defaultSteps = useStore(s => s.modelOptions?.default_num_inference_steps)
  const activatedLoras = useStore(s => s.params.activated_loras)
  const setParam = useStore(s => s.setParam)
  const toggleLora = useStore(s => s.toggleLora)
  const setLoraWeight = useStore(s => s.setLoraWeight)
  const selectModel = useStore(s => s.selectModel)

  const turboPresets = option?.presets?.length
    ? option.presets
    : option
      ? [{
          id: option.preset_id,
          label: option.version_label,
          status: 'validated',
          filename: option.filename,
          steps: option.steps,
          weight: option.weight,
          weight_min: 0.5,
          weight_max: 1.0,
          description: option.guide,
          revision: '',
        }]
      : []
  const selectedTurboPreset = (
    turboPresets.find(preset => preset.id === params.minimax_h3_turbo_preset)
    || turboPresets.find(preset => preset.id === option?.preset_id)
    || turboPresets[0]
  )
  const architecture = String(modelOptions?.architecture || '')
  const isH3 = architecture.startsWith('minimax_h3')
  const turboEnabled = params.minimax_h3_turbo_mode === true
  const solEnabled = params.override_attention === 'sol'
  const cacheEnabled = params.skip_steps_cache_type === 'first_block'
  const solSupported = modelOptions?.sol_attention_status?.supported === true
  const activeCount = [turboEnabled, solEnabled, cacheEnabled].filter(Boolean).length

  if (!isH3 && !advisory) return null

  const handleTurboChange = (checked: boolean) => {
    if (!option || !selectedTurboPreset) return
    setParam('minimax_h3_turbo_preset', selectedTurboPreset.id)
    setParam('minimax_h3_turbo_mode', checked)
    if (checked) {
      for (const preset of turboPresets) {
        if (
          preset.filename !== selectedTurboPreset.filename
          && activatedLoras.includes(preset.filename)
        ) {
          toggleLora(preset.filename)
        }
      }
      if (!activatedLoras.includes(selectedTurboPreset.filename)) {
        toggleLora(selectedTurboPreset.filename)
      }
      // Keep the managed adapter visible in Advanced so users can tune the
      // selected preset's starting weight after enabling the recipe.
      setLoraWeight(selectedTurboPreset.filename, 0, selectedTurboPreset.weight)
      setParam('num_inference_steps', selectedTurboPreset.steps)
      setParam('minimax_h3_turbo_mode', true)
    } else {
      for (const preset of turboPresets) {
        if (activatedLoras.includes(preset.filename)) {
          toggleLora(preset.filename)
        }
      }
      if (currentSteps === selectedTurboPreset.steps && defaultSteps != null) {
        setParam('num_inference_steps', defaultSteps)
      }
    }
  }

  const handleTurboPresetChange = (presetId: string) => {
    const nextPreset = turboPresets.find(preset => preset.id === presetId)
    if (!nextPreset) return
    for (const preset of turboPresets) {
      if (
        preset.filename !== nextPreset.filename
        && activatedLoras.includes(preset.filename)
      ) {
        toggleLora(preset.filename)
      }
    }
    setParam('minimax_h3_turbo_preset', nextPreset.id)
    if (turboEnabled) {
      if (!activatedLoras.includes(nextPreset.filename)) {
        toggleLora(nextPreset.filename)
      }
      setLoraWeight(nextPreset.filename, 0, nextPreset.weight)
      setParam('num_inference_steps', nextPreset.steps)
      setParam('minimax_h3_turbo_mode', true)
    }
  }

  const handleFirstBlockCacheChange = (checked: boolean) => {
    setParam('skip_steps_cache_type', checked ? 'first_block' : '')
    if (checked && params.skip_steps_multiplier == null) {
      setParam(
        'skip_steps_multiplier',
        modelOptions?.default_skip_steps_multiplier ?? 0.08,
      )
    }
  }

  const useRecommendedPrunedTurbo = () => {
    const recommendedModel = advisory?.recommended_model_type
    if (!recommendedModel || !option || !selectedTurboPreset) return

    // Model switching is synchronous in Zustand even though its option/default
    // fetches continue in the background. Rebuild the managed Turbo selection
    // from the new state after selectModel intentionally resets LoRAs.
    selectModel(recommendedModel)
    const next = useStore.getState()
    next.setParam('minimax_h3_turbo_preset', selectedTurboPreset.id)
    next.setParam('minimax_h3_turbo_mode', true)
    if (!next.params.activated_loras.includes(selectedTurboPreset.filename)) {
      next.toggleLora(selectedTurboPreset.filename)
    }
    next.setLoraWeight(selectedTurboPreset.filename, 0, selectedTurboPreset.weight)
    next.setParam('num_inference_steps', selectedTurboPreset.steps)
  }

  const solHelp = solSupported
    ? 'Uses H3-aware sparse attention with exact reference and audio conditioning. Its compiled kernels are cached across restarts, and unsupported calls fall back automatically.'
    : `${modelOptions?.sol_attention_status?.reason || 'Sol Engine is unavailable in this runtime.'} On supported hardware, run Maestro's normal Pinokio Update to install or repair the H3 performance runtime.`

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

      {(option || modelOptions?.sol_attention || modelOptions?.first_block_cache) && (
        <section className="overflow-hidden rounded-lg border border-border bg-bg-tertiary/35">
          <button
            type="button"
            onClick={() => setExpanded(value => !value)}
            aria-expanded={expanded}
            className={`flex w-full items-center justify-between px-3 py-2 text-left transition-colors hover:bg-bg-tertiary/70 ${
              expanded ? 'border-b border-border/70' : ''
            }`}
          >
            <div className="flex items-center gap-2">
              <span className="text-[10px] font-semibold uppercase tracking-wider text-text-secondary">
                H3 Optimizations
              </span>
              <span className="rounded bg-amber-500/15 px-1.5 py-0.5 text-[8px] font-medium uppercase tracking-wider text-indicator-warning">
                Experimental
              </span>
            </div>
            <span className="flex items-center gap-1.5">
              <span className={`text-[9px] ${activeCount ? 'text-accent-blue' : 'text-text-muted'}`}>
                {activeCount ? `${activeCount} active` : 'Optional'}
              </span>
              <ChevronDown
                size={13}
                className={`text-text-muted transition-transform ${expanded ? 'rotate-180' : ''}`}
              />
            </span>
          </button>

          {expanded && (
            <div className="divide-y divide-border/60">
            {option && (
              <div className={`flex items-center gap-2 px-3 py-2 transition-colors ${
                turboEnabled ? 'bg-accent-blue/10' : ''
              }`}>
                <label className="flex min-w-0 flex-1 cursor-pointer items-center gap-2.5 select-none">
                  <input
                    type="checkbox"
                    checked={turboEnabled}
                    onChange={event => handleTurboChange(event.target.checked)}
                    className="accent-accent-blue"
                  />
                  <Zap size={13} className={turboEnabled ? 'text-accent-blue' : 'text-text-muted'} />
                  <span className="min-w-0">
                    <span className="block text-[11px] font-medium text-text-primary">Turbo</span>
                    <span className="block text-[9px] text-text-muted">
                      {selectedTurboPreset?.steps ?? option.steps}-step / {selectedTurboPreset?.label ?? option.version_label}
                    </span>
                  </span>
                </label>
                <InfoTooltip
                  label="About H3 Turbo mode"
                  text={selectedTurboPreset?.description || option.guide}
                />
              </div>
            )}

            {option && turboEnabled && selectedTurboPreset && turboPresets.length > 1 && (
              <div className="space-y-1.5 bg-bg-secondary/35 px-3 py-2">
                <div className="flex items-center justify-between gap-2">
                  <label
                    htmlFor="h3-turbo-checkpoint"
                    className="text-[9px] font-medium uppercase tracking-wider text-text-muted"
                  >
                    Turbo checkpoint
                  </label>
                  <span className={`rounded px-1.5 py-0.5 text-[8px] font-medium uppercase tracking-wider ${
                    selectedTurboPreset.status === 'candidate'
                      ? 'bg-amber-500/15 text-indicator-warning'
                      : 'bg-accent-blue/10 text-accent-blue'
                  }`}>
                    {selectedTurboPreset.status}
                  </span>
                </div>
                <select
                  id="h3-turbo-checkpoint"
                  value={selectedTurboPreset.id}
                  onChange={event => handleTurboPresetChange(event.target.value)}
                  className="w-full rounded-md border border-border bg-bg-secondary px-2 py-1.5 text-[10px] text-text-primary outline-none focus:border-accent-blue"
                >
                  {turboPresets.map(preset => (
                    <option key={preset.id} value={preset.id}>
                      {preset.label}
                    </option>
                  ))}
                </select>
                <p className="text-[9px] leading-relaxed text-text-muted">
                  Starts at {selectedTurboPreset.weight.toFixed(2)} strength; tune it in Advanced.
                </p>
              </div>
            )}

            {modelOptions?.sol_attention && (
              <div className={`flex items-center gap-2 px-3 py-2 transition-colors ${
                solEnabled ? 'bg-accent-blue/10' : ''
              } ${solSupported ? '' : 'opacity-65'}`}>
                <label className={`flex min-w-0 flex-1 items-center gap-2.5 select-none ${
                  solSupported ? 'cursor-pointer' : 'cursor-not-allowed'
                }`}>
                  <input
                    type="checkbox"
                    checked={solEnabled}
                    disabled={!solSupported}
                    onChange={event => setParam(
                      'override_attention',
                      event.target.checked ? 'sol' : '',
                    )}
                    className="accent-accent-blue"
                  />
                  <Gauge size={13} className={solEnabled ? 'text-accent-blue' : 'text-text-muted'} />
                  <span className="min-w-0">
                    <span className="block text-[11px] font-medium text-text-primary">Sol Engine</span>
                    <span className="block text-[9px] text-text-muted">
                      {solSupported ? 'H3 sparse attention' : 'Unavailable in this runtime'}
                    </span>
                  </span>
                </label>
                <InfoTooltip label="About H3 Sol Engine" text={solHelp} />
              </div>
            )}

            {modelOptions?.first_block_cache && (
              <div className={`flex items-center gap-2 px-3 py-2 transition-colors ${
                cacheEnabled ? 'bg-accent-blue/10' : ''
              }`}>
                <label className="flex min-w-0 flex-1 cursor-pointer items-center gap-2.5 select-none">
                  <input
                    type="checkbox"
                    checked={cacheEnabled}
                    onChange={event => handleFirstBlockCacheChange(event.target.checked)}
                    className="accent-accent-blue"
                  />
                  <Layers size={13} className={cacheEnabled ? 'text-accent-blue' : 'text-text-muted'} />
                  <span className="min-w-0">
                    <span className="block text-[11px] font-medium text-text-primary">First Block Cache</span>
                    <span className="block text-[9px] text-text-muted">Reuse stable denoising work</span>
                  </span>
                </label>
                <InfoTooltip
                  label="About H3 First Block Cache"
                  text="Reuses stable transformer work after warmup. It is most effective on longer, higher-step H3 runs and can be combined with Turbo and Sol Engine. Threshold and warmup tuning remain in Advanced."
                />
              </div>
            )}
            </div>
          )}
        </section>
      )}
    </div>
  )
}
