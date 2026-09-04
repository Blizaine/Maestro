/* eslint-disable react-refresh/only-export-components -- the advanced badge hooks share this settings contract */
import { useState, useEffect, useRef } from 'react'
import { X, Save, Trash2, FolderOpen, SlidersHorizontal } from 'lucide-react'
import { useStore } from '../../stores/useStore'
import { PostProcessing } from './PostProcessing'
import { ControlVideoSection } from './ControlVideoSection'
import { LoraSelector } from '../SettingsDrawer/LoraSelector'
import { ResolutionPresets } from './ResolutionPresets'
import { AspectRatioGrid } from './AspectRatioGrid'
import { WindowSettings } from './DurationSlider'
import { DirectorH3Optimizations } from './DirectorH3Optimizations'
import type { GenerateParams } from '../../types'

const H3_LONG_SEQUENCE_EXPERIMENTS = [
  {
    id: 'h3_long_sequence_clean_tail',
    label: 'Clean-tail handoff',
    activeLabel: 'H3 clean-tail handoff',
    description: 'Drops the final 17 generated frames before selecting the next continuation tail.',
  },
  {
    id: 'h3_long_sequence_single_frame_after_three',
    label: 'Single-frame handoff after window 3',
    activeLabel: 'H3 one-frame fallback',
    description: 'From window 4 onward, keeps only the last boundary frame instead of recursive motion history.',
  },
  {
    id: 'h3_long_sequence_vary_seed',
    label: 'Vary seed per window',
    activeLabel: 'H3 per-window seeds',
    description: 'Keeps window 1 unchanged, then derives a repeatable seed for every continuation window.',
  },
  {
    id: 'h3_long_sequence_periodic_reset',
    label: 'Reset motion history every 3 windows',
    activeLabel: 'H3 periodic handoff reset',
    description: 'Windows 4, 7, 10, and so on use only the last boundary frame, then full motion history resumes.',
  },
  {
    id: 'h3_long_sequence_diagnostics',
    label: 'Log continuation diagnostics',
    activeLabel: 'H3 continuation logging',
    description: 'Prints each window seed, handoff mode, and sampled video/audio fingerprints to the console.',
  },
] as const

// Keep the diagnostic controls and runtime wiring available for future A/B
// work, but do not expose unfinished long-sequence experiments in releases.
// Flip this local development flag only while actively running those tests.
const H3_LONG_SEQUENCE_TESTS_VISIBLE = false

function PresetManager() {
  const presets = useStore(s => s.presets)
  const loadPresets = useStore(s => s.loadPresets)
  const savePreset = useStore(s => s.savePreset)
  const loadPresetFn = useStore(s => s.loadPreset)
  const deletePreset = useStore(s => s.deletePreset)
  const generationMode = useStore(s => s.generationMode)
  const currentModel = useStore(s => s.params.model_type)
  const modeLabel = generationMode === 'avatar' ? 'video transform' : generationMode
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
        <p className="text-[10px] text-text-muted">No {modeLabel} presets for this model</p>
      )}
    </div>
  )
}

function LtxExperimentalToggle({
  checked,
  onChange,
  label,
  badge,
  description,
}: {
  checked: boolean
  onChange: () => void
  label: string
  badge?: string
  description: string
}) {
  return (
    <div className="flex items-start justify-between gap-3">
      <div className="min-w-0">
        <div className="text-[11px] text-text-secondary">
          {label}
          {badge ? (
            <span className="ml-1.5 rounded border border-accent-blue/30 px-1 py-0.5 text-[8px] text-accent-blue">
              {badge}
            </span>
          ) : null}
        </div>
        <p className="mt-0.5 text-[9px] leading-relaxed text-text-muted">
          {description}
        </p>
      </div>
      <button
        type="button"
        role="switch"
        aria-checked={checked}
        aria-label={label}
        onClick={onChange}
        className={`relative mt-0.5 h-5 w-9 shrink-0 rounded-full transition-colors ${
          checked ? 'bg-accent-blue' : 'border border-border bg-bg-tertiary'
        }`}
      >
        <span className={`absolute left-0.5 top-0.5 h-4 w-4 rounded-full border border-border bg-white shadow transition-transform ${
          checked ? 'translate-x-4' : 'translate-x-0'
        }`} />
      </button>
    </div>
  )
}

function LtxFramesExperimentalControls() {
  const generationMode = useStore(s => s.generationMode)
  const workflow = useStore(s => s.studioVideoWorkflow)
  const modelType = useStore(s => s.params.model_type)
  const model = useStore(s => s.models.find(candidate => candidate.model_type === modelType))
  const servicesConfig = useStore(s => s.servicesConfig)
  const updateServicesConfig = useStore(s => s.updateServicesConfig)
  const family = String(model?.family || '').toLowerCase()
  const architecture = String(model?.architecture || '').toLowerCase()
  const isLtx = family === 'ltx2' || family === 'ltx25' || architecture.startsWith('ltx2')

  if (generationMode !== 'video' || workflow !== 'frames' || !isLtx || !servicesConfig) {
    return null
  }

  return (
    <div className="space-y-3 rounded-lg border border-border bg-bg-tertiary/25 p-3">
      <div>
        <label className="text-[11px] uppercase tracking-wider text-text-muted">
          LTX optional conditioning
        </label>
        <p className="mt-0.5 text-[9px] text-text-muted">Off by default. Applies only to Video Frames with an LTX model.</p>
      </div>
      <LtxExperimentalToggle
        checked={servicesConfig.voice_reference_enabled === true}
        onChange={() => updateServicesConfig({
          voice_reference_enabled: !servicesConfig.voice_reference_enabled,
        })}
        label="Voice Reference (ID-LoRA)"
        description="Adds a voice sample input for speaker identity conditioning with a compatible LTX ID-LoRA."
      />
      <LtxExperimentalToggle
        checked={servicesConfig.director_multishot_lora_mode === true}
        onChange={() => updateServicesConfig({
          director_multishot_lora_mode: !servicesConfig.director_multishot_lora_mode,
        })}
        label="Multi-Shot LoRA Prompting"
        badge="Beta"
        description="Uses storyboard-style shot prompts with compatible LTX multi-shot IC-LoRAs. Enable the matching LoRA separately."
      />
    </div>
  )
}

/** Active advanced features as human-readable labels. Drives the badge
 *  count AND its hover tooltip, so a surprising number names its source
 *  instead of sending the user hunting through every section. */
export function useAdvancedActiveItems(): string[] {
  const params = useStore(s => s.params)
  const modelOptions = useStore(s => s.modelOptions)
  const sidebarMode = useStore(s => s.sidebarMode)
  const directorVideoModel = useStore(s => s.selectedModelPerMode.video || '')
  const directorTurboMode = useStore(s => s.directorH3TurboModeByModel)
  const directorSolMode = useStore(s => s.directorH3SolModeByModel)
  const directorFirstBlockCache = useStore(s => s.directorH3FirstBlockCacheByModel)
  const spatialUpsampling = useStore(s => s.spatialUpsampling)
  const filmGrainIntensity = useStore(s => s.filmGrainIntensity)
  const generationMode = useStore(s => s.generationMode)
  const editSubMode = useStore(s => s.editSubMode)
  const slidingWindowLocked = useStore(s => s.slidingWindowLocked)
  const servicesConfig = useStore(s => s.servicesConfig)
  const studioVideoWorkflow = useStore(s => s.studioVideoWorkflow)
  const selectedModel = useStore(s => s.models.find(model => model.model_type === s.params.model_type))
  const isScailEdit = (
    generationMode === 'avatar'
    && (editSubMode === 'recast' || editSubMode === 'restyle')
  )
  const isScailHq = isScailEdit && params.model_type === 'scail2_14B'

  const items: string[] = []
  if (sidebarMode === 'director') {
    if (directorTurboMode[directorVideoModel] === true) items.push('H3 Turbo')
    if (directorSolMode[directorVideoModel] === true) {
      items.push(
        directorVideoModel.includes('fused_turbo')
          ? 'H3 SLA'
          : 'H3 Sol Engine',
      )
    }
    if (directorFirstBlockCache[directorVideoModel] === true) items.push('First Block Cache')
    return items
  }
  if (params.seed !== -1) items.push(`Seed ${params.seed}`)
  const selectedFamily = String(selectedModel?.family || '').toLowerCase()
  const selectedArchitecture = String(selectedModel?.architecture || '').toLowerCase()
  const isLtxFrames = generationMode === 'video'
    && studioVideoWorkflow === 'frames'
    && (
      selectedFamily === 'ltx2'
      || selectedFamily === 'ltx25'
      || selectedArchitecture.startsWith('ltx2')
    )
  if (isLtxFrames && servicesConfig?.voice_reference_enabled) items.push('LTX voice reference')
  if (isLtxFrames && servicesConfig?.director_multishot_lora_mode) items.push('LTX multi-shot prompting')
  if (
    String(modelOptions?.architecture || '').startsWith('minimax_h3')
    && slidingWindowLocked
  ) items.push('H3 window override')
  if (
    H3_LONG_SEQUENCE_TESTS_VISIBLE
    && String(modelOptions?.architecture || '').startsWith('minimax_h3')
    && modelOptions?.omni_reference !== true
    && params.minimax_h3_multi_window === true
  ) {
    const customSettings = params.custom_settings || {}
    for (const experiment of H3_LONG_SEQUENCE_EXPERIMENTS) {
      if (customSettings[experiment.id] === true) {
        items.push(experiment.activeLabel)
      }
    }
  }
  if (
    (
      modelOptions?.sliding_window_auto_prompt_pacing === true
      || (
        modelOptions?.omni_reference === true
        && params.minimax_h3_reference_sequence === true
      )
    )
    && params.minimax_h3_camera_coverage
    && params.minimax_h3_camera_coverage !== 'auto'
  ) {
    items.push(
      params.minimax_h3_camera_coverage === 'continuous'
        ? 'H3 continuous take'
        : 'H3 multi-shot coverage',
    )
  }
  if (
    (params.negative_prompt?.length ?? 0) > 0
    && (!isScailEdit || isScailHq)
  ) items.push('Negative prompt')
  for (const l of params.activated_loras) items.push(`LoRA: ${l.replace(/\.(safetensors|sft)$/i, '')}`)
  if (!isScailEdit && spatialUpsampling) items.push(`Upscaling (${spatialUpsampling})`)
  if (!isScailEdit && filmGrainIntensity > 0) items.push('Film grain')
  if (!isScailEdit && (params.self_refiner_setting ?? 0) > 0) items.push('Self refiner')
  if (
    modelOptions?.minimax_h3_text_encoder_choices?.length
    && params.minimax_h3_text_encoder
    && params.minimax_h3_text_encoder !== modelOptions.minimax_h3_text_encoder_default
  ) {
    const selected = modelOptions.minimax_h3_text_encoder_choices.find(
      choice => choice.value === params.minimax_h3_text_encoder
    )
    items.push(`H3 encoder: ${selected?.label || params.minimax_h3_text_encoder}`)
  }
  if (
    modelOptions?.ltx25_video_vae_choices?.length
    && params.ltx25_video_vae === 'nad'
  ) {
    items.push('LTX-2.5 NAD VAE')
  }
  // injection_strength only matters when injected frames actually exist.
  // The persisted snapshot strips image_refs (file paths are ephemeral)
  // but kept the strength value — counting it alone produced a ghost
  // badge with nothing visibly active in the panel.
  const refCount = Array.isArray(params.image_refs) ? params.image_refs.length : (params.image_refs ? 1 : 0)
  if (
    !isScailEdit
    && params.injection_strength != null
    && params.injection_strength !== 1.0
    && refCount > 0
  ) items.push('Injection strength')
  // Process letter codes persist by design (the dropdown remembers the
  // user's choice across sessions), but their REQUIRED inputs are
  // ephemeral and stripped from persistence: frames injection ("F")
  // needs image refs, control-video letters ("V") need a guide file.
  // A remembered choice with no input does nothing at generation time,
  // so it must not count — this was the refresh-surviving ghost. Strip only
  // a TRAILING "T" (the extend-alignment flag); an internal "T" is a real
  // process letter (depth_temporal: TVG/PTVG/TEVG) and must survive.
  const vptVisible = (params.video_prompt_type || '').replace(/T$/, '')
  if (!isScailEdit && modelOptions?.guide_custom_choices && vptVisible) {
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
  const isDirector = useStore(s => s.sidebarMode === 'director')
  const generationMode = useStore(s => s.generationMode)
  const editSubMode = useStore(s => s.editSubMode)
  const audioSubMode = useStore(s => s.audioSubMode)
  const isAudio = generationMode === 'audio'
  const isSfx = isAudio && audioSubMode === 'sfx'
  const isAudioOnly = modelOptions?.audio_only || isSfx
  const isVideo = generationMode === 'video'
  const isAvatar = generationMode === 'avatar'
  const isOutpaint = isAvatar && editSubMode === 'outpaint'
  const isRecast = isAvatar && editSubMode === 'recast'
  const isRepaint = isAvatar && editSubMode === 'restyle'
  const isScailEdit = isRecast || isRepaint
  const scailModelType = String(params.model_type || '')
  const isScailFast = (
    isScailEdit
    && (
      scailModelType === 'scail2_14B_fast'
      || scailModelType === 'scail2_14B_recast_fast'
    )
  )
  const isScailHq = isScailEdit && scailModelType === 'scail2_14B'
  const h3TurboMode = (
    params.minimax_h3_turbo_mode === true
    && modelOptions?.minimax_h3_turbo != null
  )
  const isH3 = String(modelOptions?.architecture || '').startsWith('minimax_h3')
  const showH3LongSequenceExperiments = (
    H3_LONG_SEQUENCE_TESTS_VISIBLE
    && isVideo
    && isH3
    && modelOptions?.omni_reference !== true
    && params.minimax_h3_multi_window === true
  )
  const showInferenceSteps = (
    !isAudioOnly
    && (isScailEdit || !modelOptions?.lock_inference_steps)
  )
  const inferenceStepsMin = Math.max(
    1,
    Math.round(Number(modelOptions?.inference_steps_min ?? 1)),
  )
  const inferenceStepsMax = Math.max(
    inferenceStepsMin,
    Math.round(Number(modelOptions?.inference_steps_max ?? 50)),
  )
  const setInferenceSteps = (value: number) => {
    if (!Number.isFinite(value)) return
    setParam(
      'num_inference_steps',
      Math.max(inferenceStepsMin, Math.min(inferenceStepsMax, Math.round(value))),
    )
  }
  const showGuidanceScale = (
    !isAudioOnly
    && (
      isScailEdit
        ? isScailHq
        : !modelOptions?.lock_guidance_scale
    )
  )
  const showNegativePrompt = (
    !modelOptions?.no_negative_prompt
    && (!isScailEdit || isScailHq)
  )
  const hasStartImage = useStore(s => !!(s.startImage || s.params.image_start))
  const hasEndImage = useStore(s => !!(s.endImage || s.params.image_end))
  const hasImageRefs = useStore(s => {
    const refs = s.params.image_refs
    return refs && refs.length > 0
  })
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
        type="button"
        onClick={() => setOpen(!open)}
        title={advancedCount > 0
          ? `Advanced settings (${advancedCount} active)\n${advancedItems.join('\n')}`
          : 'Advanced settings'}
        aria-label={`Advanced settings${advancedCount > 0 ? `, ${advancedCount} active` : ''}`}
        aria-expanded={open}
        className={`relative flex shrink-0 items-center justify-center rounded-lg border p-2 transition-colors ${
          open ? 'border-accent-blue text-accent-blue' : 'border-border text-text-secondary hover:text-text-primary hover:border-border-light'
        }`}
      >
        <SlidersHorizontal size={14} />
        {advancedCount > 0 && (
          <span
            title={advancedItems.join('\n')}
            className="absolute -right-1 -top-1 flex h-3.5 min-w-3.5 items-center justify-center rounded-full bg-accent-blue px-0.5 text-[8px] font-bold leading-none text-white shadow-sm"
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
              {isDirector ? (
                <>
                  <DirectorH3Optimizations />
                  <p className="rounded-lg border border-border/60 bg-bg-tertiary/35 px-3 py-2 text-[9px] leading-relaxed text-text-muted">
                    Director inference steps, maximum shot length, image guidance, and post-processing remain in the workflow&apos;s Advanced section. Settings here are saved with the Director project and reused by repair and regeneration.
                  </p>
                </>
              ) : (
                <>
              {/* Recast/Repaint own their output-quality profiles in the main
                  workflow. Their dedicated endpoints also choose adaptive
                  windows, so generic controls would be misleading here. */}
              {!isAudio && !isScailEdit && (
                <>
                  {!isOutpaint && !modelOptions?.hide_resolution_presets && <ResolutionPresets />}
                  {!isAvatar && <AspectRatioGrid />}
                </>
              )}

              {/* Presets belong with the creative adapter controls so users can
                  save or restore a setup before adjusting its LoRAs. */}
              <PresetManager />

              {/* Keep creative adapters near the top so users can choose them
                  before working through the lower-level tuning controls.
                  Official Outpaint owns its stage-one-only IC-LoRA schedule. */}
              {!isOutpaint && !modelOptions?.loras_disabled && <LoraSelector />}
              {!isOutpaint && modelOptions?.loras_disabled && (
                <p className="rounded-lg border border-amber-500/25 bg-amber-500/8 px-3 py-2 text-[9px] leading-relaxed text-text-muted">
                  This fused four-step checkpoint already contains its acceleration and style adapters, so additional LoRAs are disabled.
                </p>
              )}

              <LtxFramesExperimentalControls />

              {/* The Qwen conditioner is shared by every H3 transformer.
                  Expose it once here instead of multiplying model entries. */}
              {modelOptions?.minimax_h3_text_encoder_choices?.length ? (
                <div>
                  <label className="text-[11px] text-text-muted uppercase tracking-wider mb-1.5 block">
                    H3 Text Encoder
                  </label>
                  <select
                    value={params.minimax_h3_text_encoder || modelOptions.minimax_h3_text_encoder_default || modelOptions.minimax_h3_text_encoder_choices[0]?.value}
                    onChange={e => setParam(
                      'minimax_h3_text_encoder',
                      e.target.value as NonNullable<GenerateParams['minimax_h3_text_encoder']>,
                    )}
                    className="w-full bg-bg-tertiary border border-border rounded px-2.5 py-1.5 text-xs text-text-primary focus:outline-none focus:border-accent-blue"
                  >
                    {modelOptions.minimax_h3_text_encoder_choices.map(choice => (
                      <option key={choice.value} value={choice.value}>
                        {choice.label}
                      </option>
                    ))}
                  </select>
                  <p className="text-[9px] text-text-muted mt-1">
                    {modelOptions.minimax_h3_text_encoder_choices.find(
                      choice => choice.value === (params.minimax_h3_text_encoder || modelOptions.minimax_h3_text_encoder_default)
                    )?.size_hint || 'Changing this reloads the H3 model.'}
                  </p>
                </div>
              ) : null}

              {modelOptions?.ltx25_video_vae_choices?.length ? (
                <div>
                  <label className="text-[11px] text-text-muted uppercase tracking-wider mb-1.5 block">
                    LTX-2.5 Video Decoder
                  </label>
                  <select
                    value={params.ltx25_video_vae || modelOptions.ltx25_video_vae_default || modelOptions.ltx25_video_vae_choices[0]?.value}
                    onChange={e => setParam('ltx25_video_vae', e.target.value as 'fast' | 'nad')}
                    className="w-full bg-bg-tertiary border border-border rounded px-2.5 py-1.5 text-xs text-text-primary focus:outline-none focus:border-accent-blue"
                  >
                    {modelOptions.ltx25_video_vae_choices.map(choice => (
                      <option key={choice.value} value={choice.value}>
                        {choice.label}{choice.experimental ? ' (Experimental)' : ''}
                      </option>
                    ))}
                  </select>
                  <p className="text-[9px] text-text-muted mt-1">
                    {modelOptions.ltx25_video_vae_choices.find(
                      choice => choice.value === (params.ltx25_video_vae || modelOptions.ltx25_video_vae_default)
                    )?.description || 'Changing this reloads the LTX-2.5 model.'}
                  </p>
                </div>
              ) : null}

              {modelOptions?.first_block_cache && params.skip_steps_cache_type === 'first_block' && (
                <div className="space-y-2 p-2.5 bg-bg-tertiary/40 rounded-lg border border-border/60">
                  <div className="flex items-center justify-between gap-2">
                    <span className="text-[11px] text-text-muted uppercase tracking-wider">
                      First Block Cache Tuning
                    </span>
                    <span className="text-[9px] text-accent-blue">Enabled in Studio</span>
                  </div>
                  <div className="space-y-2 pl-1 border-l border-border ml-1">
                    <div>
                      <label className="text-[10px] text-text-muted block mb-1">
                        {modelOptions.skip_steps_multiplier_label || 'Cache Threshold'}
                      </label>
                      <select
                        value={params.skip_steps_multiplier ?? modelOptions.default_skip_steps_multiplier ?? 0.08}
                        onChange={e => setParam('skip_steps_multiplier', Number(e.target.value))}
                        className="w-full bg-bg-tertiary border border-border rounded px-2 py-1.5 text-xs text-text-primary focus:outline-none focus:border-accent-blue"
                      >
                        {(modelOptions.skip_steps_multiplier_choices || []).map(([label, value]) => (
                          <option key={value} value={value}>{label}</option>
                        ))}
                      </select>
                    </div>
                    <div>
                      <div className="flex items-center justify-between mb-1">
                        <label className="text-[10px] text-text-muted">Warmup</label>
                        <span className="text-[10px] text-text-secondary">
                          {params.skip_steps_start_step_perc ?? modelOptions.default_skip_steps_start_step_perc ?? 25}%
                        </span>
                      </div>
                      <input
                        type="range"
                        min={0}
                        max={75}
                        step={5}
                        value={params.skip_steps_start_step_perc ?? modelOptions.default_skip_steps_start_step_perc ?? 25}
                        onChange={e => setParam('skip_steps_start_step_perc', Number(e.target.value))}
                        className="w-full"
                      />
                    </div>
                  </div>
                  <p className="text-[9px] text-text-muted">
                    Higher thresholds reuse more work but can change motion or fine detail.
                  </p>
                </div>
              )}

              {/* Window Settings */}
              {(isVideo || (isAvatar && !isScailEdit))
                && (
                  modelOptions?.sliding_window
                  || isH3
                  || (
                    isVideo
                    && modelOptions?.omni_reference === true
                    && params.minimax_h3_reference_sequence === true
                  )
                )
                && <WindowSettings />}

              {isVideo
                && modelOptions?.sliding_window_auto_prompt_pacing === true
                && (
                  modelOptions?.omni_reference === true
                    ? params.minimax_h3_reference_sequence === true
                    : params.minimax_h3_multi_window === true
                )
                && (
                <div className="space-y-2">
                  <div>
                    <label className="text-[10px] text-text-muted block mb-1">
                      Camera Coverage
                    </label>
                    <select
                      value={params.minimax_h3_camera_coverage || 'auto'}
                      onChange={e => setParam(
                        'minimax_h3_camera_coverage',
                        e.target.value as 'auto' | 'continuous' | 'multi_shot',
                      )}
                      title="Auto chooses an editing grammar from the prompt. Continuous preserves a single take. Multi-shot allows timed H3 camera cuts inside every native window."
                      className="w-full bg-bg-tertiary border border-border rounded px-2 py-1.5 text-xs text-text-primary focus:outline-none focus:border-accent-blue"
                    >
                      <option value="auto">Auto</option>
                      <option value="continuous">Continuous take</option>
                      <option value="multi_shot">Cinematic multi-shot</option>
                    </select>
                  </div>
                </div>
              )}

              {isVideo
                && modelOptions?.omni_reference === true
                && params.minimax_h3_reference_sequence === true
                && (
                  <div>
                    <label className="text-[10px] text-text-muted block mb-1">
                      Sequence Camera Coverage
                    </label>
                    <select
                      value={params.minimax_h3_camera_coverage || 'auto'}
                      onChange={e => setParam(
                        'minimax_h3_camera_coverage',
                        e.target.value as 'auto' | 'continuous' | 'multi_shot',
                      )}
                      title="Auto chooses action, conversation, or atmospheric coverage from the prompt."
                      className="w-full bg-bg-tertiary border border-border rounded px-2 py-1.5 text-xs text-text-primary focus:outline-none focus:border-accent-blue"
                    >
                      <option value="auto">Auto</option>
                      <option value="continuous">Continuous take per clip</option>
                      <option value="multi_shot">Cinematic multi-shot</option>
                    </select>
                  </div>
                )}

              {/* TTS Settings */}
              {isAudioOnly && (
                <>
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
              {!isAudio && !isScailEdit && <PostProcessing />}

              {/* Seed */}
              {
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
              }

              {/* Self Refiner */}
              {!isScailEdit && modelOptions?.self_refiner && (
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
              {!isScailEdit
                && modelOptions?.lock_inference_steps
                && String(modelOptions.architecture || '').toLowerCase().startsWith('ltx2')
                && (
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
              {!isScailEdit && (modelOptions as Record<string, unknown> | null)?.reference_pipeline && (
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

              {/* Dedicated SCAIL edit endpoints honor this value for both
                  Fast and HQ; other distilled models retain their lock. */}
              {showInferenceSteps && (
                <div>
                  <div className="flex items-center justify-between mb-1.5">
                    <label className="text-[11px] text-text-muted uppercase tracking-wider">
                      {modelOptions?.inference_steps_label || 'Inference Steps'}
                    </label>
                    <input
                      type="number"
                      min={inferenceStepsMin}
                      max={inferenceStepsMax}
                      step={1}
                      value={params.num_inference_steps}
                      disabled={h3TurboMode}
                      onChange={e => setInferenceSteps(Number(e.target.value))}
                      className="w-16 bg-bg-tertiary border border-border rounded px-2 py-0.5 text-xs text-text-primary text-center focus:outline-none focus:border-accent-blue disabled:cursor-not-allowed disabled:opacity-50"
                    />
                  </div>
                  <input
                    type="range" min={inferenceStepsMin} max={inferenceStepsMax} step={1}
                    value={params.num_inference_steps}
                    disabled={h3TurboMode}
                    onChange={e => setInferenceSteps(Number(e.target.value))}
                    className="w-full disabled:cursor-not-allowed disabled:opacity-50"
                  />
                  {h3TurboMode && (
                    <p className="text-[9px] text-text-muted mt-0.5">
                      Turbo mode locks this preset to {modelOptions?.minimax_h3_turbo?.steps} steps.
                    </p>
                  )}
                  {!h3TurboMode && modelOptions?.inference_steps_help && (
                    <p className="text-[9px] text-text-muted mt-0.5">
                      {modelOptions.inference_steps_help}
                    </p>
                  )}
                  {isScailFast && (
                    <p className="text-[9px] text-text-muted mt-0.5">
                      Fast keeps its distilled CFG 1 recipe; guidance and
                      negative-prompt controls do not apply.
                    </p>
                  )}
                </div>
              )}

              {/* Guidance Scale (hidden for TTS — shown in TTS section above) */}
              {showGuidanceScale && (
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
              {!isScailEdit && (modelOptions as Record<string, unknown> | null)?.perturbation && (
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
              {!isScailEdit && (isVideo || isAvatar) && (hasStartImage || hasEndImage) && (
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
              {!isScailEdit && (isVideo || isAvatar) && hasImageRefs && (
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
              {showNegativePrompt && (
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

              {/* Dedicated SCAIL edit endpoints own their source video,
                  edited/reference frames, masks, and process selection. */}
              {(modelOptions?.guide_preprocessing || modelOptions?.guide_custom_choices) &&
                !isScailEdit && !modelOptions?.minimax_h3_media_sources && (
                <ControlVideoSection />
              )}

              {/* Dedicated Recast/Repaint submissions create one edit job. */}
              {!isScailEdit && <div>
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
              </div>}

              {showH3LongSequenceExperiments && (
                <div className="space-y-2.5 rounded-lg border border-amber-400/30 bg-amber-400/5 p-3">
                  <div className="flex items-center justify-between gap-2">
                    <label className="text-[11px] uppercase tracking-wider text-amber-300">
                      Long-sequence tests
                    </label>
                    <span className="rounded border border-amber-400/30 px-1 py-0.5 text-[8px] text-amber-300/90">
                      Experimental
                    </span>
                  </div>
                  <p className="text-[10px] leading-relaxed text-text-muted">
                    A/B controls for diagnosing repetition and cumulative over-processing in long H3 First / Last sequences. Defaults remain off.
                  </p>
                  <div className="space-y-2.5">
                    {H3_LONG_SEQUENCE_EXPERIMENTS.map(experiment => {
                      const customSettings = params.custom_settings || {}
                      const checked = customSettings[experiment.id] === true
                      return (
                        <label
                          key={experiment.id}
                          className="flex cursor-pointer items-start gap-2 group"
                          title={experiment.description}
                        >
                          <input
                            type="checkbox"
                            checked={checked}
                            onChange={event => {
                              const nextSettings = {
                                ...(params.custom_settings || {}),
                              }
                              if (event.target.checked) {
                                nextSettings[experiment.id] = true
                              } else {
                                delete nextSettings[experiment.id]
                              }
                              setParam(
                                'custom_settings',
                                Object.keys(nextSettings).length > 0
                                  ? nextSettings
                                  : undefined,
                              )
                            }}
                            className="mt-0.5 accent-accent-blue"
                          />
                          <span className="min-w-0">
                            <span className="block text-[11px] text-text-secondary transition-colors group-hover:text-text-primary">
                              {experiment.label}
                            </span>
                            <span className="mt-0.5 block text-[9px] leading-relaxed text-text-muted">
                              {experiment.description}
                            </span>
                          </span>
                        </label>
                      )
                    })}
                  </div>
                </div>
              )}
                </>
              )}
            </div>
          </div>
    </>
  )
}
