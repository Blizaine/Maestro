import { useState, useEffect, useRef } from 'react'
import { X, Save, Trash2, FolderOpen, SlidersHorizontal } from 'lucide-react'
import { useStore } from '../../stores/useStore'
import { PostProcessing } from './PostProcessing'
import { ControlVideoSection } from './ControlVideoSection'
import { LoraSelector } from '../SettingsDrawer/LoraSelector'
import { ResolutionPresets } from './ResolutionPresets'
import { AspectRatioGrid } from './AspectRatioGrid'
import { WindowSettings } from './DurationSlider'

function PresetManager() {
  const presets = useStore(s => s.presets)
  const loadPresets = useStore(s => s.loadPresets)
  const savePreset = useStore(s => s.savePreset)
  const loadPresetFn = useStore(s => s.loadPreset)
  const deletePreset = useStore(s => s.deletePreset)
  const generationMode = useStore(s => s.generationMode)
  const currentModel = useStore(s => s.params.model_type)
  const [saveName, setSaveName] = useState('')
  const [showSave, setShowSave] = useState(false)
  const [confirmDelete, setConfirmDelete] = useState<string | null>(null)

  useEffect(() => { loadPresets() }, [loadPresets])

  const modePresets = presets.filter(p => p.mode === generationMode && p.model_type === currentModel)

  const handleSave = () => {
    if (!saveName.trim()) return
    savePreset(saveName.trim())
    setSaveName('')
    setShowSave(false)
  }

  const handleDelete = (id: string) => {
    if (confirmDelete === id) {
      deletePreset(id)
      setConfirmDelete(null)
    } else {
      setConfirmDelete(id)
      setTimeout(() => setConfirmDelete(null), 3000)
    }
  }

  return (
    <div>
      <div className="flex items-center justify-between mb-1.5">
        <label className="text-[11px] text-text-muted uppercase tracking-wider">Presets</label>
        <button
          onClick={() => setShowSave(!showSave)}
          className="text-[10px] text-accent-blue hover:text-accent-blue-hover flex items-center gap-0.5"
        >
          <Save size={10} /> Save Current
        </button>
      </div>

      {showSave && (
        <div className="flex gap-1.5 mb-2">
          <input
            type="text"
            value={saveName}
            onChange={e => setSaveName(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && handleSave()}
            placeholder="Preset name..."
            className="flex-1 bg-bg-tertiary border border-border rounded px-2 py-1 text-xs text-text-primary focus:outline-none focus:border-accent-blue"
            autoFocus
          />
          <button
            onClick={handleSave}
            disabled={!saveName.trim()}
            className="px-2 py-1 text-xs bg-accent-blue text-white rounded hover:bg-accent-blue-hover disabled:opacity-50"
          >
            Save
          </button>
        </div>
      )}

      {modePresets.length > 0 ? (
        <div className="space-y-1 max-h-[120px] overflow-y-auto">
          {modePresets.map(p => (
            <div key={p.id} className="flex items-center gap-1.5 group">
              <button
                onClick={() => loadPresetFn(p)}
                className="flex-1 text-left px-2 py-1.5 rounded text-xs text-text-secondary hover:bg-bg-hover hover:text-text-primary transition-colors truncate flex items-center gap-1.5"
                title={`${p.name}\n${p.activated_loras.length} LoRA(s) - ${p.model_type}`}
              >
                <FolderOpen size={10} className="shrink-0 text-text-muted" />
                <span className="truncate">{p.name}</span>
              </button>
              <button
                onClick={() => handleDelete(p.id)}
                className={`p-1 rounded transition-colors shrink-0 ${
                  confirmDelete === p.id
                    ? 'text-red-400 bg-red-500/20'
                    : 'text-text-muted opacity-0 group-hover:opacity-100 hover:text-red-400'
                }`}
              >
                <Trash2 size={10} />
              </button>
            </div>
          ))}
        </div>
      ) : (
        <p className="text-[10px] text-text-muted">No {generationMode} presets for this model</p>
      )}
    </div>
  )
}

/** Active advanced features as human-readable labels. Drives the badge
 *  count AND its hover tooltip, so a surprising number names its source
 *  instead of sending the user hunting through every section. */
export function useAdvancedActiveItems(): string[] {
  const params = useStore(s => s.params)
  const modelOptions = useStore(s => s.modelOptions)
  const spatialUpsampling = useStore(s => s.spatialUpsampling)
  const filmGrainIntensity = useStore(s => s.filmGrainIntensity)

  const items: string[] = []
  if (params.seed !== -1) items.push(`Seed ${params.seed}`)
  if ((params.negative_prompt?.length ?? 0) > 0) items.push('Negative prompt')
  for (const l of params.activated_loras) items.push(`LoRA: ${l.replace(/\.(safetensors|sft)$/i, '')}`)
  if (spatialUpsampling) items.push(`Upscaling (${spatialUpsampling})`)
  if (filmGrainIntensity > 0) items.push('Film grain')
  if ((params.self_refiner_setting ?? 0) > 0) items.push('Self refiner')
  // injection_strength only matters when injected frames actually exist.
  // The persisted snapshot strips image_refs (file paths are ephemeral)
  // but kept the strength value — counting it alone produced a ghost
  // badge with nothing visibly active in the panel.
  const refCount = Array.isArray(params.image_refs) ? params.image_refs.length : (params.image_refs ? 1 : 0)
  if (params.injection_strength != null && params.injection_strength !== 1.0 && refCount > 0) items.push('Injection strength')
  // Process letter codes persist by design (the dropdown remembers the
  // user's choice across sessions), but their REQUIRED inputs are
  // ephemeral and stripped from persistence: frames injection ("F")
  // needs image refs, control-video letters ("V") need a guide file.
  // A remembered choice with no input does nothing at generation time,
  // so it must not count — this was the refresh-surviving ghost. Strip only
  // a TRAILING "T" (the extend-alignment flag); an internal "T" is a real
  // process letter (depth_temporal: TVG/PTVG/TEVG) and must survive.
  const vptVisible = (params.video_prompt_type || '').replace(/T$/, '')
  if (modelOptions?.guide_custom_choices && vptVisible) {
    const effective = vptVisible.includes('F')
      ? refCount > 0
      : vptVisible.includes('V')
        ? !!params.video_guide
        : true
    if (effective) items.push(`Process: ${vptVisible}`)
  }
  return items
}

/** Count active advanced features for the badge */
export function useAdvancedCount(): number {
  return useAdvancedActiveItems().length
}

export function AdvancedSettings() {
  const [open, setOpen] = useState(false)
  const params = useStore(s => s.params)
  const setParam = useStore(s => s.setParam)
  const modelOptions = useStore(s => s.modelOptions)
  const generationMode = useStore(s => s.generationMode)
  const editSubMode = useStore(s => s.editSubMode)
  const audioSubMode = useStore(s => s.audioSubMode)
  const isAudio = generationMode === 'audio'
  const isSfx = isAudio && audioSubMode === 'sfx'
  const isAudioOnly = modelOptions?.audio_only || isSfx
  const isVideo = generationMode === 'video'
  const isAvatar = generationMode === 'avatar'
  const isRecast = isAvatar && editSubMode === 'recast'
  const hasStartImage = useStore(s => !!(s.startImage || s.params.image_start))
  const hasEndImage = useStore(s => !!(s.endImage || s.params.image_end))
  const hasImageRefs = useStore(s => {
    const refs = s.params.image_refs
    return refs && refs.length > 0
  })
  const durationSeconds = useStore(s => s.durationSeconds)
  const setDurationSeconds = useStore(s => s.setDurationSeconds)
  const panelRef = useRef<HTMLDivElement>(null)
  const advancedItems = useAdvancedActiveItems()
  const advancedCount = advancedItems.length

  // Close on escape
  useEffect(() => {
    if (!open) return
    const handler = (e: KeyboardEvent) => { if (e.key === 'Escape') setOpen(false) }
    document.addEventListener('keydown', handler)
    return () => document.removeEventListener('keydown', handler)
  }, [open])

  return (
    <>
      {/* Trigger button */}
      <button
        onClick={() => setOpen(!open)}
        className={`flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs transition-colors border ${
          open ? 'border-accent-blue text-accent-blue' : 'border-border text-text-secondary hover:text-text-primary hover:border-border-light'
        }`}
      >
        <SlidersHorizontal size={13} />
        <span className="hidden md:inline">Advanced</span>
        {advancedCount > 0 && (
          <span
            title={advancedItems.join('\n')}
            className="min-w-[16px] h-4 rounded-full bg-accent-blue text-white text-[9px] font-bold flex items-center justify-center px-1"
          >
            {advancedCount}
          </span>
        )}
      </button>

      {/* Popup overlay — always mounted to preserve state (frames injection, etc.) */}
      {open && <div className="fixed inset-0 bg-black/30 z-50" onClick={() => setOpen(false)} />}
      <div
        ref={panelRef}
        className={`fixed top-0 h-full bg-bg-secondary border-r border-border z-50 flex flex-col shadow-2xl overflow-hidden transition-transform duration-200
          left-0 w-full md:left-[420px] md:w-[380px] md:max-w-[90vw] ${
          open ? 'translate-x-0' : '-translate-x-full md:-translate-x-[800px] pointer-events-none'
        }`}
        style={{ maxHeight: '100vh' }}
      >
            {/* Header */}
            <div className="px-4 py-3 border-b border-border flex items-center justify-between shrink-0">
              <span className="text-sm font-semibold text-text-primary">Advanced Settings</span>
              <button onClick={() => setOpen(false)} className="p-1 rounded-lg hover:bg-bg-hover text-text-secondary">
                <X size={16} />
              </button>
            </div>

            {/* Scrollable content */}
            <div className="flex-1 overflow-y-auto px-4 py-4 space-y-5">
              {/* Resolution + Aspect. Recast hides both resolution and
                  window settings: the endpoint owns them (SCAIL-2's native
                  832x480, 81-frame windows), so the controls shown here
                  would display values the generation ignores. */}
              {!isAudio && (
                <>
                  {!modelOptions?.hide_resolution_presets && !isRecast && <ResolutionPresets />}
                  {!isAvatar && <AspectRatioGrid />}
                </>
              )}

              {/* Window Settings */}
              {(isVideo || (isAvatar && !isRecast)) && <WindowSettings />}

              {/* TTS Settings */}
              {isAudioOnly && (
                <>
                  {/* Max Duration */}
                  {modelOptions?.duration_slider && (
                    <div>
                      <div className="flex items-center justify-between mb-1.5">
                        <label className="text-[11px] text-text-muted uppercase tracking-wider">
                          {modelOptions.duration_slider.label || 'Max Duration'}
                        </label>
                        <span className="text-xs text-text-secondary">{Math.round(durationSeconds)}s</span>
                      </div>
                      <input
                        type="range"
                        min={modelOptions.duration_slider.min} max={modelOptions.duration_slider.max} step={modelOptions.duration_slider.increment}
                        value={durationSeconds}
                        onChange={e => setDurationSeconds(parseFloat(e.target.value))}
                        className="w-full"
                      />
                    </div>
                  )}

                  {/* Speaker Pause */}
                  {modelOptions?.pause_between_sentences && (
                    <div>
                      <div className="flex items-center justify-between mb-1.5">
                        <label className="text-[11px] text-text-muted uppercase tracking-wider">Speaker Pause</label>
                        <span className="text-xs text-text-secondary">{(params.pause_seconds ?? 0.5).toFixed(2)}s</span>
                      </div>
                      <input
                        type="range" min={0} max={2} step={0.05}
                        value={params.pause_seconds ?? 0.5}
                        onChange={e => setParam('pause_seconds', parseFloat(e.target.value))}
                        className="w-full"
                      />
                    </div>
                  )}

                  {/* Temperature */}
                  {modelOptions?.temperature_enabled && (
                    <div>
                      <div className="flex items-center justify-between mb-1.5">
                        <label className="text-[11px] text-text-muted uppercase tracking-wider">Temperature</label>
                        <span className="text-xs text-text-secondary">{(params.temperature ?? 1.0).toFixed(2)}</span>
                      </div>
                      <input
                        type="range" min={0.1} max={1.5} step={0.01}
                        value={params.temperature ?? 1.0}
                        onChange={e => setParam('temperature', parseFloat(e.target.value))}
                        className="w-full"
                      />
                    </div>
                  )}

                  {/* Guidance Scale */}
                  <div>
                    <div className="flex items-center justify-between mb-1.5">
                      <label className="text-[11px] text-text-muted uppercase tracking-wider">Guidance (CFG)</label>
                      <span className="text-xs text-text-secondary">{(params.guidance_scale ?? 3.0).toFixed(1)}</span>
                    </div>
                    <input
                      type="range" min={1} max={20} step={0.1}
                      value={params.guidance_scale ?? 3.0}
                      onChange={e => setParam('guidance_scale', parseFloat(e.target.value))}
                      className="w-full"
                    />
                  </div>

                  {/* Auto-Split */}
                  {modelOptions?.custom_settings_def?.map(setting => (
                    <div key={setting.id}>
                      <label className="text-[11px] text-text-muted uppercase tracking-wider mb-1.5 block">{setting.name}</label>
                      <input
                        type="number"
                        placeholder="Empty = disabled"
                        value={String((params.custom_settings as Record<string, unknown> | undefined)?.[setting.id] ?? '')}
                        onChange={e => {
                          const val = e.target.value.trim()
                          const cs = { ...(params.custom_settings || {}) } as Record<string, unknown>
                          if (val === '') {
                            delete cs[setting.id]
                          } else {
                            cs[setting.id] = parseFloat(val)
                          }
                          setParam('custom_settings', Object.keys(cs).length > 0 ? cs : undefined)
                        }}
                        className="w-full bg-bg-tertiary border border-border rounded-lg px-3 py-2 text-sm text-text-primary focus:outline-none focus:border-accent-blue"
                      />
                      <p className="text-[10px] text-text-muted mt-1">{setting.label}</p>
                    </div>
                  ))}

                  {/* Compressor Settings — shown when Smooth Speaker Volumes is enabled */}
                  {params.tts_dynaudnorm && (
                    <div className="space-y-3 p-2.5 bg-bg-tertiary/50 rounded-lg border border-border/50">
                      <label className="text-[10px] text-text-muted uppercase tracking-wider block">Speaker Transition Compressor</label>
                      <div>
                        <div className="flex items-center justify-between mb-1">
                          <label className="text-[10px] text-text-muted">Threshold</label>
                          <span className="text-[10px] text-text-secondary">{params.tts_comp_threshold || -25}dB</span>
                        </div>
                        <input type="range" min={-50} max={-10} step={1}
                          value={params.tts_comp_threshold || -25}
                          onChange={e => setParam('tts_comp_threshold', parseInt(e.target.value))}
                          className="w-full" />
                        <p className="text-[9px] text-text-muted">Volume level where boosting kicks in. Lower = catches quieter parts.</p>
                      </div>
                      <div>
                        <div className="flex items-center justify-between mb-1">
                          <label className="text-[10px] text-text-muted">Attack</label>
                          <span className="text-[10px] text-text-secondary">{params.tts_comp_attack || 5}ms</span>
                        </div>
                        <input type="range" min={1} max={50} step={1}
                          value={params.tts_comp_attack || 5}
                          onChange={e => setParam('tts_comp_attack', parseInt(e.target.value))}
                          className="w-full" />
                        <p className="text-[9px] text-text-muted">How fast the compressor reacts. Low = catches brief dips at speaker transitions.</p>
                      </div>
                      <div>
                        <div className="flex items-center justify-between mb-1">
                          <label className="text-[10px] text-text-muted">Release</label>
                          <span className="text-[10px] text-text-secondary">{params.tts_comp_release || 100}ms</span>
                        </div>
                        <input type="range" min={20} max={500} step={10}
                          value={params.tts_comp_release || 100}
                          onChange={e => setParam('tts_comp_release', parseInt(e.target.value))}
                          className="w-full" />
                        <p className="text-[9px] text-text-muted">How fast it returns to normal after boosting. Higher = smoother.</p>
                      </div>
                      <div>
                        <div className="flex items-center justify-between mb-1">
                          <label className="text-[10px] text-text-muted">Makeup Gain</label>
                          <span className="text-[10px] text-text-secondary">{params.tts_comp_makeup || 4}dB</span>
                        </div>
                        <input type="range" min={0} max={12} step={1}
                          value={params.tts_comp_makeup || 4}
                          onChange={e => setParam('tts_comp_makeup', parseInt(e.target.value))}
                          className="w-full" />
                        <p className="text-[9px] text-text-muted">How much to boost the quiet parts. Higher = louder transitions.</p>
                      </div>
                    </div>
                  )}
                </>
              )}

              {/* Post Processing */}
              {!isAudio && <PostProcessing />}

              {/* Seed */}
              {((
                <div>
                  <div className="flex items-center justify-between mb-1.5">
                    <label className="text-[11px] text-text-muted uppercase tracking-wider">Seed</label>
                    <button onClick={() => setParam('seed', -1)} className="text-[10px] text-accent-blue hover:text-accent-blue-hover">
                      Random
                    </button>
                  </div>
                  <input
                    type="number"
                    value={params.seed}
                    onChange={e => setParam('seed', Number(e.target.value))}
                    className="w-full bg-bg-tertiary border border-border rounded-lg px-3 py-2 text-sm text-text-primary focus:outline-none focus:border-accent-blue"
                    placeholder="-1 for random"
                  />
                </div>
              ) as any)}

              {/* Self Refiner */}
              {modelOptions?.self_refiner && (
                <div>
                  <label className="text-[11px] text-text-muted uppercase tracking-wider mb-1.5 block">Self Refiner</label>
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

              {/* Stage 2 Steps */}
              {/* Pipeline Mode Toggle — distilled LTX models only */}
              {modelOptions?.lock_inference_steps && (
                <div className="space-y-3">
                  {/* Single / 2-Stage / 3-Stage segmented control — mutually exclusive */}
                  <div>
                    <label className="text-[11px] text-text-muted uppercase tracking-wider mb-1.5 block">Pipeline Mode</label>
                    <div className="flex bg-bg-tertiary rounded-lg p-0.5 border border-border">
                      <button
                        onClick={() => { setParam('progressive_pipeline', false); setParam('single_stage_pipeline', true) }}
                        className={`flex-1 text-[10px] py-1.5 rounded-md transition-all ${
                          !!params.single_stage_pipeline && !params.progressive_pipeline
                            ? 'bg-bg-active text-text-primary'
                            : 'text-text-secondary hover:text-text-primary'
                        }`}
                        title="Run at full target resolution in one pass. No upscale, no refine. Higher VRAM."
                      >
                        Single
                      </button>
                      <button
                        onClick={() => { setParam('progressive_pipeline', false); setParam('single_stage_pipeline', false) }}
                        className={`flex-1 text-[10px] py-1.5 rounded-md transition-all ${
                          !params.progressive_pipeline && !params.single_stage_pipeline
                            ? 'bg-bg-active text-text-primary'
                            : 'text-text-secondary hover:text-text-primary'
                        }`}
                        title="Half-res denoise, then 2x spatial upscale + refine. Balanced speed/quality."
                      >
                        Standard (2-Stage)
                      </button>
                      <button
                        onClick={() => { setParam('progressive_pipeline', true); setParam('single_stage_pipeline', false) }}
                        className={`flex-1 text-[10px] py-1.5 rounded-md transition-all ${
                          params.progressive_pipeline
                            ? 'bg-bg-active text-text-primary'
                            : 'text-text-secondary hover:text-text-primary'
                        }`}
                        title="Progressive 1/4 → 1/2 → full. Smoother motion, slower."
                      >
                        Progressive (3-Stage)
                      </button>
                    </div>
                  </div>

                  {/* Single-Stage: no extra controls — stage 1 runs at full res */}
                  {!!params.single_stage_pipeline && !params.progressive_pipeline && (
                    <div className="text-[10px] text-text-muted px-1">
                      Runs the distilled denoise at full target resolution in one pass. No stage-2 upscale or refine.
                      Uses ~4× the stage-1 VRAM of 2-Stage mode; drop to a smaller resolution preset if you OOM.
                    </div>
                  )}

                  {/* Standard 2-Stage: Stage 2 steps only */}
                  {!params.progressive_pipeline && !params.single_stage_pipeline && (
                    <div>
                      <div className="flex items-center justify-between mb-1.5">
                        <label className="text-[11px] text-text-muted uppercase tracking-wider">Stage 2 Steps</label>
                        <span className="text-xs text-text-secondary">{params.stage2_steps || 3}</span>
                      </div>
                      <input
                        type="range" min={2} max={7} step={1}
                        value={params.stage2_steps || 3}
                        onChange={e => setParam('stage2_steps', Number(e.target.value))}
                        className="w-full accent-accent-blue"
                      />
                      <div className="flex justify-between text-[10px] text-text-muted mt-0.5">
                        <span>2 (faster)</span><span>7 (more detail)</span>
                      </div>
                    </div>
                  )}

                  {/* Progressive 3-Stage controls */}
                  {!!params.progressive_pipeline && (
                    <div className="space-y-3 pt-1 border-t border-border/30">
                      <div>
                        <div className="flex items-center justify-between mb-1">
                          <label className="text-[10px] text-text-muted">Stage 1 Image Weight</label>
                          <span className="text-[10px] text-text-secondary">{(params.progressive_stage1_image_weight ?? 0.7).toFixed(2)}</span>
                        </div>
                        <input type="range" min={0.3} max={1.0} step={0.05}
                          value={params.progressive_stage1_image_weight ?? 0.7}
                          onChange={e => setParam('progressive_stage1_image_weight', parseFloat(e.target.value))}
                          className="w-full accent-accent-blue" />
                        <div className="flex justify-between text-[9px] text-text-muted mt-0.5">
                          <span>0.30 (more motion)</span><span>1.00 (match start image)</span>
                        </div>
                      </div>
                      <div>
                        <div className="flex items-center justify-between mb-1">
                          <label className="text-[10px] text-text-muted">Stage 2 Steps (half res)</label>
                          <span className="text-[10px] text-text-secondary">{params.progressive_stage2_steps ?? 5}</span>
                        </div>
                        <input type="range" min={1} max={8} step={1}
                          value={params.progressive_stage2_steps ?? 5}
                          onChange={e => setParam('progressive_stage2_steps', Number(e.target.value))}
                          className="w-full accent-accent-blue" />
                      </div>
                      <div>
                        <div className="flex items-center justify-between mb-1">
                          <label className="text-[10px] text-text-muted">Stage 3 Steps (full res)</label>
                          <span className="text-[10px] text-text-secondary">{params.progressive_stage3_steps ?? 3}</span>
                        </div>
                        <input type="range" min={1} max={8} step={1}
                          value={params.progressive_stage3_steps ?? 3}
                          onChange={e => setParam('progressive_stage3_steps', Number(e.target.value))}
                          className="w-full accent-accent-blue" />
                      </div>
                      <div>
                        <div className="flex items-center justify-between mb-1">
                          <label className="text-[10px] text-text-muted">Stage 2 Sigma</label>
                          <span className="text-[10px] text-text-secondary">{(params.progressive_stage2_sigma ?? 0.85).toFixed(2)}</span>
                        </div>
                        <input type="range" min={0.5} max={1.0} step={0.05}
                          value={params.progressive_stage2_sigma ?? 0.85}
                          onChange={e => setParam('progressive_stage2_sigma', parseFloat(e.target.value))}
                          className="w-full accent-accent-blue" />
                        <div className="flex justify-between text-[9px] text-text-muted mt-0.5">
                          <span>0.50 (preserve)</span><span>1.00 (regenerate)</span>
                        </div>
                      </div>
                      <div>
                        <div className="flex items-center justify-between mb-1">
                          <label className="text-[10px] text-text-muted">Stage 3 Sigma</label>
                          <span className="text-[10px] text-text-secondary">{(params.progressive_stage3_sigma ?? 0.85).toFixed(2)}</span>
                        </div>
                        <input type="range" min={0.5} max={1.0} step={0.05}
                          value={params.progressive_stage3_sigma ?? 0.85}
                          onChange={e => setParam('progressive_stage3_sigma', parseFloat(e.target.value))}
                          className="w-full accent-accent-blue" />
                        <div className="flex justify-between text-[9px] text-text-muted mt-0.5">
                          <span>0.50 (preserve)</span><span>1.00 (regenerate)</span>
                        </div>
                      </div>
                      <div>
                        <div className="flex items-center justify-between mb-1">
                          <label className="text-[10px] text-text-muted">Stage 3 Image Weight (full res)</label>
                          <span className="text-[10px] text-text-secondary">{(params.progressive_stage3_image_weight ?? 0.7).toFixed(2)}</span>
                        </div>
                        <input type="range" min={0.3} max={1.0} step={0.05}
                          value={params.progressive_stage3_image_weight ?? 0.7}
                          onChange={e => setParam('progressive_stage3_image_weight', parseFloat(e.target.value))}
                          className="w-full accent-accent-blue" />
                        <div className="flex justify-between text-[9px] text-text-muted mt-0.5">
                          <span>0.30 (more detail freedom)</span><span>1.00 (match start image)</span>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              )}

              {/* Reference Pipeline (10Eros — runs the author's published
                  ComfyUI workflow config: 9+3 eased steps, per-step CFG
                  2.0/1.5 then off, STG on blocks 14+19 for the first 4
                  steps, RF euler_ancestral). Shown only for models whose
                  def declares reference_pipeline support. */}
              {(modelOptions as Record<string, unknown> | null)?.reference_pipeline && (
                <div className="space-y-1">
                  <label className="flex items-center gap-2 cursor-pointer group">
                    <input type="checkbox"
                      checked={!!params.reference_pipeline}
                      onChange={e => setParam('reference_pipeline', e.target.checked ? true : undefined)}
                      className="accent-accent-blue" />
                    <span className="text-[11px] text-text-muted uppercase tracking-wider group-hover:text-text-secondary transition-colors">
                      Reference Pipeline (10Eros)
                    </span>
                  </label>
                  <p className="text-[9px] text-text-muted">
                    Runs the model author&apos;s ComfyUI workflow config: 9+3 steps on hand-tuned sigmas,
                    CFG only on the first 2 steps, STG on the first 4, ancestral sampling.
                    Steps / CFG / STG sliders below are ignored while this is on.
                  </p>
                </div>
              )}

              {/* Inference Steps (hidden for distilled/locked models and TTS) */}
              {!modelOptions?.lock_inference_steps && !isAudioOnly && (
                <div>
                  <div className="flex items-center justify-between mb-1.5">
                    <label className="text-[11px] text-text-muted uppercase tracking-wider">Inference Steps</label>
                    <input
                      type="number"
                      value={params.num_inference_steps}
                      onChange={e => setParam('num_inference_steps', Number(e.target.value))}
                      className="w-16 bg-bg-tertiary border border-border rounded px-2 py-0.5 text-xs text-text-primary text-center focus:outline-none focus:border-accent-blue"
                    />
                  </div>
                  <input
                    type="range" min={1} max={50} step={1}
                    value={params.num_inference_steps}
                    onChange={e => setParam('num_inference_steps', Number(e.target.value))}
                    className="w-full"
                  />
                </div>
              )}

              {/* Guidance Scale (hidden for TTS — shown in TTS section above) */}
              {!modelOptions?.lock_guidance_scale && !isAudioOnly && (
                <div>
                  <div className="flex items-center justify-between mb-1.5">
                    <label className="text-[11px] text-text-muted uppercase tracking-wider">Guidance Scale</label>
                    <input
                      type="number"
                      value={params.guidance_scale}
                      onChange={e => setParam('guidance_scale', Number(e.target.value))}
                      step={0.1}
                      className="w-16 bg-bg-tertiary border border-border rounded px-2 py-0.5 text-xs text-text-primary text-center focus:outline-none focus:border-accent-blue"
                    />
                  </div>
                  <input
                    type="range" min={0} max={20} step={0.1}
                    value={params.guidance_scale}
                    onChange={e => setParam('guidance_scale', Number(e.target.value))}
                    className="w-full"
                  />
                </div>
              )}

              {/* LTX-2 Dev Pipeline Controls — only for models with perturbation/CFG-Star support */}
              {(modelOptions as Record<string, unknown> | null)?.perturbation && (
                <>
                  {/* STG Scale */}
                  <div>
                    <div className="flex items-center justify-between mb-1.5">
                      <label className="text-[11px] text-text-muted uppercase tracking-wider">STG Scale</label>
                      <span className="text-xs text-text-secondary">{(params.stg_scale ?? 0) > 0 ? (params.stg_scale as number).toFixed(1) : 'Off'}</span>
                    </div>
                    <input type="range" min={0} max={3} step={0.1}
                      value={params.stg_scale ?? 0}
                      onChange={e => setParam('stg_scale', parseFloat(e.target.value))}
                      className="w-full" />
                    <p className="text-[9px] text-text-muted mt-0.5">Spatio-temporal guidance. 0 = off. Sharpens structure &amp; motion via a third denoising pass (~50% slower). Try 1.0.</p>
                  </div>

                  {/* CFG Rescale */}
                  <div>
                    <div className="flex items-center justify-between mb-1.5">
                      <label className="text-[11px] text-text-muted uppercase tracking-wider">CFG Rescale</label>
                      <span className="text-xs text-text-secondary">{(params.cfg_rescale ?? 0).toFixed(2)}</span>
                    </div>
                    <input type="range" min={0} max={1} step={0.05}
                      value={params.cfg_rescale ?? 0}
                      onChange={e => setParam('cfg_rescale', parseFloat(e.target.value))}
                      className="w-full" />
                    <p className="text-[9px] text-text-muted mt-0.5">Reduces over-saturation. 0.7 recommended.</p>
                  </div>

                  {/* Gradient Estimation */}
                  <div className="space-y-1.5">
                    <label className="flex items-center gap-2 cursor-pointer group">
                      <input type="checkbox"
                        checked={!!params.use_gradient_estimation}
                        onChange={e => setParam('use_gradient_estimation', e.target.checked ? true : undefined)}
                        className="accent-accent-blue" />
                      <span className="text-[11px] text-text-muted uppercase tracking-wider group-hover:text-text-secondary transition-colors">
                        Gradient Estimation
                      </span>
                    </label>
                    {params.use_gradient_estimation && (
                      <div className="pl-1 border-l border-border ml-1 space-y-1.5">
                        <p className="text-[9px] text-accent-blue/80">Use 20-25 steps instead of 30-40 for comparable quality.</p>
                        <div>
                          <div className="flex items-center justify-between mb-1">
                            <span className="text-[10px] text-text-muted">Gamma</span>
                            <span className="text-[9px] text-text-muted">{(params.ge_gamma ?? 2.0).toFixed(1)}</span>
                          </div>
                          <input type="range" min={1} max={4} step={0.1}
                            value={params.ge_gamma ?? 2.0}
                            onChange={e => setParam('ge_gamma', parseFloat(e.target.value))}
                            className="w-full" />
                        </div>
                      </div>
                    )}
                  </div>
                </>
              )}

              {/* Keyframe Conditioning Mode — Start/End frames */}
              {(isVideo || isAvatar) && (hasStartImage || hasEndImage) && (
                <div>
                  <label className="text-[11px] text-text-muted uppercase tracking-wider mb-1.5 block">Start/End Frame Mode</label>
                  <select
                    value={params.keyframe_conditioning_mode || 'replace'}
                    onChange={e => setParam('keyframe_conditioning_mode', e.target.value)}
                    className="w-full bg-bg-tertiary border border-border rounded px-2.5 py-1.5 text-xs text-text-primary focus:outline-none focus:border-accent-blue"
                  >
                    <option value="replace">Replace (Default)</option>
                    <option value="additive">Additive (Smooth)</option>
                  </select>
                  <p className="text-[9px] text-text-muted mt-0.5">Replace: exact adherence to source image. Additive: smoother blending.</p>
                </div>
              )}

              {/* Keyframe Conditioning Mode — Injected keyframes */}
              {(isVideo || isAvatar) && hasImageRefs && (
                <div>
                  <label className="text-[11px] text-text-muted uppercase tracking-wider mb-1.5 block">Injected Keyframe Mode</label>
                  <select
                    value={params.keyframe_inject_mode || 'additive'}
                    onChange={e => setParam('keyframe_inject_mode', e.target.value)}
                    className="w-full bg-bg-tertiary border border-border rounded px-2.5 py-1.5 text-xs text-text-primary focus:outline-none focus:border-accent-blue"
                  >
                    <option value="additive">Additive (Default)</option>
                    <option value="replace">Replace (Strict)</option>
                  </select>
                  <p className="text-[9px] text-text-muted mt-0.5">Additive: smooth transitions at injected frames. Replace: strict adherence.</p>
                </div>
              )}

              {/* Negative Prompt */}
              {!modelOptions?.no_negative_prompt && (
                <div>
                  <label className="text-[11px] text-text-muted uppercase tracking-wider mb-1.5 block">Negative Prompt</label>
                  <textarea
                    value={params.negative_prompt || ''}
                    onChange={e => setParam('negative_prompt', e.target.value)}
                    placeholder="What to avoid..."
                    rows={2}
                    className="w-full bg-bg-tertiary border border-border rounded-lg px-3 py-2 text-sm text-text-primary placeholder:text-text-muted focus:outline-none focus:border-accent-blue"
                    style={{ resize: 'vertical', minHeight: 48 }}
                  />
                </div>
              )}

              {/* MMAudio — video models only */}
              {isVideo && (
                <div className="space-y-2">
                  <label className="flex items-center gap-2 cursor-pointer group">
                    <input
                      type="checkbox"
                      checked={params.MMAudio_setting === 1}
                      onChange={e => setParam('MMAudio_setting', e.target.checked ? 1 : 0)}
                      className="accent-accent-blue"
                    />
                    <span className="text-[11px] text-text-muted uppercase tracking-wider group-hover:text-text-secondary transition-colors">
                      MMAudio (Soundtrack)
                    </span>
                  </label>
                  {params.MMAudio_setting === 1 && (
                    <div className="space-y-2 pl-1 border-l border-border ml-1">
                      <div>
                        <label className="text-[10px] text-text-muted block mb-1">Prompt (1-2 keywords)</label>
                        <input
                          type="text"
                          value={(params.MMAudio_prompt) || ''}
                          onChange={e => setParam('MMAudio_prompt', e.target.value)}
                          placeholder="e.g. rain, thunder"
                          className="w-full bg-bg-tertiary border border-border rounded px-2 py-1.5 text-xs text-text-primary placeholder:text-text-muted focus:outline-none focus:border-accent-blue"
                        />
                      </div>
                      <div>
                        <label className="text-[10px] text-text-muted block mb-1">Negative Prompt (1-2 keywords)</label>
                        <input
                          type="text"
                          value={(params.MMAudio_neg_prompt) || ''}
                          onChange={e => setParam('MMAudio_neg_prompt', e.target.value)}
                          placeholder="e.g. talking, speech"
                          className="w-full bg-bg-tertiary border border-border rounded px-2 py-1.5 text-xs text-text-primary placeholder:text-text-muted focus:outline-none focus:border-accent-blue"
                        />
                      </div>
                    </div>
                  )}
                </div>
              )}

              {/* Presets */}
              <PresetManager />

              {/* LoRAs */}
              <LoraSelector />

              {/* Control Video / Frames Injection. Hidden in Recast: the
                  endpoint pins "Replace One Person" + builds the mask from
                  the "who to replace" keyword, so SCAIL-2's Type of
                  Process dropdown would be a no-op there and mislead. */}
              {(modelOptions?.guide_preprocessing || modelOptions?.guide_custom_choices) &&
                !(generationMode === 'avatar' && editSubMode === 'recast') && (
                <ControlVideoSection />
              )}

              {/* Output Count */}
              <div>
                <div className="flex items-center justify-between mb-1.5">
                  <label className="text-[11px] text-text-muted uppercase tracking-wider">Output Count</label>
                  <span className="text-xs text-text-secondary">{params.repeat_generation || 1}</span>
                </div>
                <input
                  type="range" min={1} max={10} step={1}
                  value={params.repeat_generation || 1}
                  onChange={e => setParam('repeat_generation', Number(e.target.value))}
                  className="w-full"
                />
              </div>
            </div>
          </div>
    </>
  )
}
