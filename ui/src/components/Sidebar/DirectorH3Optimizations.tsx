import { useEffect, useState } from 'react'
import { Gauge, Layers, Zap } from 'lucide-react'
import { fetchModelOptions } from '../../api/client'
import { useStore } from '../../stores/useStore'
import type { ModelOptions } from '../../types'
import { InfoTooltip } from './InfoTooltip'

/**
 * Director-owned MiniMax H3 optimization controls.
 *
 * Director intentionally keeps these values separate from Studio: changing a
 * speed recipe here must affect newly generated Director shots, Dashboard
 * repair, and regeneration without silently changing Studio's active model.
 */
export function DirectorH3Optimizations() {
  const videoModel = useStore(s => s.selectedModelPerMode.video || 'ltx2_22B_distilled_1_1')
  const videoStepsByModel = useStore(s => s.directorVideoInferenceStepsByModel)
  const setVideoSteps = useStore(s => s.setDirectorVideoInferenceSteps)
  const turboModeByModel = useStore(s => s.directorH3TurboModeByModel)
  const setTurboMode = useStore(s => s.setDirectorH3TurboMode)
  const turboPresetByModel = useStore(s => s.directorH3TurboPresetByModel)
  const setTurboPreset = useStore(s => s.setDirectorH3TurboPreset)
  const solModeByModel = useStore(s => s.directorH3SolModeByModel)
  const setSolMode = useStore(s => s.setDirectorH3SolMode)
  const firstBlockCacheByModel = useStore(s => s.directorH3FirstBlockCacheByModel)
  const setFirstBlockCache = useStore(s => s.setDirectorH3FirstBlockCache)
  const cacheMultiplierByModel = useStore(s => s.directorH3FirstBlockCacheMultiplierByModel)
  const setCacheMultiplier = useStore(s => s.setDirectorH3FirstBlockCacheMultiplier)
  const cacheWarmupByModel = useStore(s => s.directorH3FirstBlockCacheWarmupByModel)
  const setCacheWarmup = useStore(s => s.setDirectorH3FirstBlockCacheWarmup)
  const savedVideoLoras = useStore(s => s.savedLoraPerMode.video)
  const directorSetLora = useStore(s => s.directorSetLora)
  const [modelOptions, setModelOptions] = useState<ModelOptions | null>(null)

  useEffect(() => {
    let cancelled = false
    setModelOptions(null)
    fetchModelOptions(videoModel)
      .then(options => {
        if (cancelled) return
        setModelOptions(options)
        const defaultSteps = options.default_num_inference_steps
        if (defaultSteps != null && Number.isFinite(defaultSteps)) {
          const current = useStore.getState().directorVideoInferenceStepsByModel[videoModel]
          if (current == null) setVideoSteps(videoModel, defaultSteps)
        }
      })
      .catch(() => {
        if (!cancelled) setModelOptions(null)
      })
    return () => { cancelled = true }
  }, [setVideoSteps, videoModel])

  const isH3 = String(modelOptions?.architecture || '').startsWith('minimax_h3')
  if (!isH3) return null

  const turboOption = modelOptions?.minimax_h3_turbo
  const turboPresets = turboOption?.presets?.length
    ? turboOption.presets
    : turboOption
      ? [{
          id: turboOption.preset_id,
          label: turboOption.version_label,
          status: 'validated',
          filename: turboOption.filename,
          steps: turboOption.steps,
          weight: turboOption.weight,
          weight_min: 0.5,
          weight_max: 1.0,
          description: turboOption.guide,
          revision: '',
        }]
      : []
  const selectedTurboPreset = (
    turboPresets.find(preset => preset.id === turboPresetByModel[videoModel])
    || turboPresets.find(preset => preset.id === turboOption?.preset_id)
    || turboPresets[0]
  )
  const turboSelected = Boolean(
    turboOption && selectedTurboPreset
    && turboModeByModel[videoModel] === true
    && savedVideoLoras?.activated_loras?.includes(selectedTurboPreset.filename)
  )
  const solStatus = modelOptions?.sol_attention_status
  const solSelected = Boolean(
    modelOptions?.sol_attention
    && solStatus?.supported
    && solModeByModel[videoModel] === true
  )
  const cacheSelected = Boolean(
    modelOptions?.first_block_cache
    && firstBlockCacheByModel[videoModel] === true
  )
  const cacheMultiplierChoices = modelOptions?.skip_steps_multiplier_choices || []
  const cacheMultiplier = (
    cacheMultiplierByModel[videoModel]
    ?? modelOptions?.default_skip_steps_multiplier
    ?? 0.08
  )
  const cacheWarmup = (
    cacheWarmupByModel[videoModel]
    ?? modelOptions?.default_skip_steps_start_step_perc
    ?? 25
  )
  const activeCount = [turboSelected, solSelected, cacheSelected].filter(Boolean).length
  const defaultSteps = modelOptions?.default_num_inference_steps
  const currentSteps = videoStepsByModel[videoModel] ?? defaultSteps

  const setDirectorTurbo = (checked: boolean) => {
    if (!turboOption || !selectedTurboPreset) return
    const current = savedVideoLoras || {
      activated_loras: [],
      loras_multipliers: '',
      loraWeights: {},
      availableLoras: [],
    }
    const managedTurboFiles = new Set(turboPresets.map(preset => preset.filename))
    const nextLoras = current.activated_loras.filter(filename => !managedTurboFiles.has(filename))
    const nextWeights = { ...current.loraWeights }
    for (const filename of managedTurboFiles) delete nextWeights[filename]
    if (checked) {
      nextLoras.push(selectedTurboPreset.filename)
      nextWeights[selectedTurboPreset.filename] = [selectedTurboPreset.weight]
    }
    const nextAvailable = current.availableLoras.includes(selectedTurboPreset.filename)
      ? current.availableLoras
      : [...current.availableLoras, selectedTurboPreset.filename]
    const multipliers = nextLoras.map(filename => (
      (nextWeights[filename] || [1]).map(value => value.toFixed(2)).join(';')
    )).join(' ')
    directorSetLora('video', nextLoras, multipliers, nextWeights, nextAvailable)
    setTurboMode(videoModel, checked)
    setTurboPreset(videoModel, selectedTurboPreset.id)
    if (checked) {
      setVideoSteps(videoModel, selectedTurboPreset.steps)
    } else if (currentSteps === selectedTurboPreset.steps && defaultSteps != null) {
      setVideoSteps(videoModel, defaultSteps)
    }
  }

  const changeDirectorTurboPreset = (presetId: string) => {
    const nextPreset = turboPresets.find(preset => preset.id === presetId)
    if (!nextPreset) return
    setTurboPreset(videoModel, nextPreset.id)
    if (!turboSelected) return

    const current = savedVideoLoras || {
      activated_loras: [],
      loras_multipliers: '',
      loraWeights: {},
      availableLoras: [],
    }
    const managedTurboFiles = new Set(turboPresets.map(preset => preset.filename))
    const nextLoras = current.activated_loras.filter(filename => !managedTurboFiles.has(filename))
    nextLoras.push(nextPreset.filename)
    const nextWeights = { ...current.loraWeights }
    for (const filename of managedTurboFiles) delete nextWeights[filename]
    nextWeights[nextPreset.filename] = [nextPreset.weight]
    const nextAvailable = current.availableLoras.includes(nextPreset.filename)
      ? current.availableLoras
      : [...current.availableLoras, nextPreset.filename]
    const multipliers = nextLoras.map(filename => (
      (nextWeights[filename] || [1]).map(value => value.toFixed(2)).join(';')
    )).join(' ')
    directorSetLora('video', nextLoras, multipliers, nextWeights, nextAvailable)
    setVideoSteps(videoModel, nextPreset.steps)
  }

  return (
    <section className="space-y-2">
      <div className="flex items-center justify-between gap-2">
        <div>
          <div className="text-[11px] font-semibold uppercase tracking-wider text-text-secondary">
            H3 Optimizations
          </div>
          <p className="mt-0.5 text-[9px] text-text-muted">
            Applied to every new Director shot, repair, and regeneration.
          </p>
        </div>
        {activeCount > 0 && (
          <span className="rounded-full bg-accent-blue/15 px-2 py-0.5 text-[9px] font-medium text-accent-blue">
            {activeCount} active
          </span>
        )}
      </div>

      {turboOption && (
        <div className={`rounded-lg border px-3 py-2.5 ${
          turboSelected
            ? 'border-accent-blue/50 bg-accent-blue/10'
            : 'border-border bg-bg-tertiary/50'
        }`}>
          <label className="flex cursor-pointer items-center gap-2.5">
            <input
              type="checkbox"
              checked={turboSelected}
              onChange={event => setDirectorTurbo(event.target.checked)}
              className="accent-accent-blue"
            />
            <Zap size={13} className={turboSelected ? 'text-accent-blue' : 'text-text-muted'} />
            <span className="min-w-0 flex-1">
              <span className="block text-[11px] font-medium text-text-primary">Turbo</span>
              <span className="block text-[9px] text-text-muted">
                {selectedTurboPreset?.steps ?? turboOption.steps}-step managed LoRA
              </span>
            </span>
            <span className="rounded bg-amber-500/15 px-1.5 py-0.5 text-[8px] font-medium uppercase tracking-wider text-indicator-warning">
              Experimental
            </span>
            <InfoTooltip
              label="About Director H3 Turbo"
              text={selectedTurboPreset?.description || turboOption.guide}
            />
          </label>
          {turboPresets.length > 1 && (
            <select
              value={selectedTurboPreset?.id || ''}
              onChange={event => changeDirectorTurboPreset(event.target.value)}
              className="mt-2 w-full rounded border border-border bg-bg-secondary px-2 py-1.5 text-[10px] text-text-primary focus:border-accent-blue focus:outline-none"
              aria-label="Director H3 Turbo preset"
            >
              {turboPresets.map(preset => (
                <option key={preset.id} value={preset.id}>{preset.label}</option>
              ))}
            </select>
          )}
          {turboSelected && selectedTurboPreset && (
            <p className="mt-1.5 text-[9px] text-text-muted">
              Starts at {selectedTurboPreset.weight.toFixed(2)} strength. Adjust it under Director Video LoRAs.
            </p>
          )}
        </div>
      )}

      {modelOptions?.sol_attention && (
        <div className={`rounded-lg border px-3 py-2.5 ${
          solSelected
            ? 'border-accent-blue/50 bg-accent-blue/10'
            : 'border-border bg-bg-tertiary/50'
        }`}>
          <label className={`flex items-center gap-2.5 ${
            solStatus?.supported ? 'cursor-pointer' : 'cursor-not-allowed opacity-70'
          }`}>
            <input
              type="checkbox"
              checked={solSelected}
              disabled={!solStatus?.supported}
              onChange={event => setSolMode(videoModel, event.target.checked)}
              className="accent-accent-blue"
            />
            <Gauge size={13} className={solSelected ? 'text-accent-blue' : 'text-text-muted'} />
            <span className="min-w-0 flex-1">
              <span className="block text-[11px] font-medium text-text-primary">Sol Engine</span>
              <span className="block text-[9px] text-text-muted">
                {solStatus?.supported ? 'H3 sparse attention' : 'Unavailable in this runtime'}
              </span>
            </span>
            <span className="rounded bg-amber-500/15 px-1.5 py-0.5 text-[8px] font-medium uppercase tracking-wider text-indicator-warning">
              Experimental
            </span>
            <InfoTooltip
              label="About Director H3 Sol Engine"
              text={solStatus?.supported
                ? 'Uses H3-aware sparse attention for every newly generated Director shot. Unsupported calls fall back automatically.'
                : solStatus?.reason || 'Sol Engine is unavailable in this runtime.'}
            />
          </label>
        </div>
      )}

      {modelOptions?.first_block_cache && (
        <div className={`rounded-lg border px-3 py-2.5 ${
          cacheSelected
            ? 'border-accent-blue/50 bg-accent-blue/10'
            : 'border-border bg-bg-tertiary/50'
        }`}>
          <label className="flex cursor-pointer items-center gap-2.5">
            <input
              type="checkbox"
              checked={cacheSelected}
              onChange={event => setFirstBlockCache(videoModel, event.target.checked)}
              className="accent-accent-blue"
            />
            <Layers size={13} className={cacheSelected ? 'text-accent-blue' : 'text-text-muted'} />
            <span className="min-w-0 flex-1">
              <span className="block text-[11px] font-medium text-text-primary">First Block Cache</span>
              <span className="block text-[9px] text-text-muted">Reuse stable denoising work</span>
            </span>
            <InfoTooltip
              label="About Director First Block Cache"
              text="Reuses stable transformer work after warmup. It is most effective on longer, higher-step H3 runs and can be combined with Turbo and Sol Engine."
            />
          </label>
          {cacheSelected && (
            <div className="mt-2.5 space-y-2.5 border-t border-border/70 pt-2.5">
              {cacheMultiplierChoices.length > 0 && (
                <div className="flex items-center justify-between gap-2">
                  <label className="text-[10px] text-text-muted">
                    {modelOptions.skip_steps_multiplier_label || 'Change threshold'}
                  </label>
                  <select
                    value={cacheMultiplier}
                    onChange={event => setCacheMultiplier(videoModel, Number(event.target.value))}
                    className="rounded border border-border bg-bg-secondary px-2 py-1 text-[10px] text-text-primary focus:border-accent-blue focus:outline-none"
                  >
                    {cacheMultiplierChoices.map(([label, value]) => (
                      <option key={value} value={value}>{label}</option>
                    ))}
                  </select>
                </div>
              )}
              <div>
                <div className="mb-1 flex items-center justify-between">
                  <label className="text-[10px] text-text-muted">Warmup</label>
                  <span className="text-[10px] tabular-nums text-text-muted">{cacheWarmup}%</span>
                </div>
                <input
                  type="range"
                  min={0}
                  max={75}
                  step={5}
                  value={cacheWarmup}
                  onChange={event => setCacheWarmup(videoModel, Number(event.target.value))}
                  className="w-full"
                />
              </div>
              <p className="text-[9px] text-text-muted">
                Higher thresholds reuse more work but can change motion or fine detail.
              </p>
            </div>
          )}
        </div>
      )}
    </section>
  )
}
