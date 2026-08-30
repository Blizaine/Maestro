import { ChevronDown, Check, Plus } from 'lucide-react'
import { useState, useRef, useEffect } from 'react'
import {
  useStore,
  getFamiliesForMode,
  getModelsForFamily,
  modelSupportsImageWorkflow,
  modelSupportsStudioVideoMediaIntent,
} from '../../stores/useStore'
import { InfoTooltip } from './InfoTooltip'

export function ModelSelector() {
  const models = useStore(s => s.models)
  const families = useStore(s => s.families)
  const enabledModels = useStore(s => s.enabledModels)
  const generationMode = useStore(s => s.generationMode)
  const editSubMode = useStore(s => s.editSubMode)
  const imageWorkflow = useStore(s => s.studioImageWorkflow)
  const studioVideoWorkflow = useStore(s => s.studioVideoWorkflow)
  const videoImageMode = useStore(s => Number(s.params.image_mode || 0))
  const hasFrameGuidance = useStore(s => Boolean(
    s.startImage
    || s.endImage
    || s.params.image_start
    || s.params.image_end
    || s.imageRefs.length
    || (
      Array.isArray(s.params.image_refs)
      && s.params.image_refs.length
      && s.params.frames_positions
    ),
  ))
  const hasOmniReferences = useStore(s => (
    s.params.minimax_h3_references?.some(reference => !(
      reference.type === 'audio' && reference.audio_intent === 'drive'
    )) === true
  ))
  const hasAudioDrive = useStore(s => Boolean(
    s.params.audio_guide
    || s.params.minimax_h3_references?.some(reference => (
      reference.type === 'audio' && reference.audio_intent === 'drive'
    )),
  ))
  const currentModelType = useStore(s => s.params.model_type)
  const selectModel = useStore(s => s.selectModel)
  const selectStudioVideoModel = useStore(s => s.selectStudioVideoModel)
  const openModelVisibility = useStore(s => s.openModelVisibility)
  // Mature Mode gate: models with nsfw_only flag are hidden from the
  // selector unless servicesConfig.nsfw_mode is enabled. Backend always
  // ships the entry (so the toggle can show/hide without a model reload)
  // but the UI clamps visibility here.
  const nsfwMode = useStore(s => s.servicesConfig?.nsfw_mode ?? false)

  const [open, setOpen] = useState(false)
  const containerRef = useRef<HTMLDivElement>(null)

  // Close on click outside
  useEffect(() => {
    if (!open) return
    function handleClick(e: MouseEvent) {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) {
        setOpen(false)
      }
    }
    document.addEventListener('mousedown', handleClick)
    return () => document.removeEventListener('mousedown', handleClick)
  }, [open])

  const audioSubMode = useStore(s => s.audioSubMode)

  const currentModel = models.find(m => m.model_type === currentModelType)
  const isPrimaryStudioCreate = generationMode === 'video'
    && studioVideoWorkflow === 'frames'
    && videoImageMode === 0
  const studioMediaIntent = { hasFrameGuidance, hasOmniReferences, hasAudioDrive }
  const currentModelCompatible = !isPrimaryStudioCreate
    || modelSupportsStudioVideoMediaIntent(currentModel, studioMediaIntent)
  const mediaConflict = hasFrameGuidance && hasOmniReferences
  const compatibilityDescription = mediaConflict
    ? 'Fixed start/end/keyframes cannot be combined with Omni references. Remove one of those input roles.'
    : hasOmniReferences
      ? 'Add or enable an H3 Omni model for these references or characters.'
      : hasFrameGuidance && hasAudioDrive
        ? 'LTX is required when fixed frames and an exact audio timeline are both active.'
        : hasFrameGuidance
          ? 'Add or enable an H3 First / Last or LTX model for these frame inputs.'
          : hasAudioDrive
            ? 'Add or enable an H3 Omni or LTX model for this audio timeline.'
            : 'Add or enable an H3 First / Last or LTX model for text generation.'
  const effectiveSubMode = generationMode === 'avatar' ? editSubMode : undefined
  const effectiveAudioSubMode = generationMode === 'audio' ? audioSubMode : undefined
  const modeFamilies = getFamiliesForMode(generationMode, families, effectiveSubMode, effectiveAudioSubMode)

  // Build grouped model list, filtered by:
  //   1. enabledModels (Settings → System → Model Visibility),
  //   2. nsfw_only gate (Mature Mode must be on for those to appear).
  const workflowFilter = (model: typeof models[number]) => (
    (generationMode !== 'image' || modelSupportsImageWorkflow(model, imageWorkflow))
    && (
      !isPrimaryStudioCreate
      || modelSupportsStudioVideoMediaIntent(model, studioMediaIntent)
    )
  )
  const groups = modeFamilies.map(family => ({
    family,
    models: getModelsForFamily(family.id, models, generationMode, effectiveSubMode)
      .filter(workflowFilter)
      .filter(m => enabledModels.has(m.model_type))
      .filter(m => !m.nsfw_only || nsfwMode),
  })).filter(g => g.models.length > 0)

  // How many models are available for this mode but NOT enabled — powers the
  // "+N" hint that nudges users toward Settings → Enabled Models.
  const disabledCount = modeFamilies.reduce((n, family) => {
    const avail = getModelsForFamily(family.id, models, generationMode, effectiveSubMode)
      .filter(workflowFilter)
      .filter(m => !m.nsfw_only || nsfwMode)
    return n + avail.filter(m => !enabledModels.has(m.model_type)).length
  }, 0)

  return (
    <div className="relative flex-1 min-w-0" ref={containerRef}>
      {/* Trigger button */}
      <button
        onClick={() => setOpen(!open)}
        title={currentModelCompatible
          ? currentModel?.selector_help || currentModel?.description
          : compatibilityDescription}
        className="w-full flex items-center gap-1.5 bg-bg-tertiary border border-border rounded-lg px-2.5 py-2 text-left hover:border-border-light transition-colors"
      >
        <span className="flex-1 min-w-0 truncate text-xs text-text-primary">
          {currentModelCompatible ? (currentModel?.name ?? 'Select model') : 'No compatible model'}
        </span>
        <ChevronDown size={14} className={`shrink-0 text-text-muted transition-transform ${open ? 'rotate-180' : ''}`} />
      </button>

      {/* Dropdown (opens upward) */}
      {open && (
        <div className="absolute bottom-full left-0 mb-1 w-[360px] max-w-[90vw] bg-bg-secondary border border-border rounded-lg shadow-xl overflow-hidden z-50">
          {/* Enable-more entry — sits above the enabled model list; opens
              Settings → Enabled Models expanded to this mode. */}
          {disabledCount > 0 && (
            <button
              onClick={() => { openModelVisibility(generationMode); setOpen(false) }}
              className="w-full flex items-center gap-2 px-3 py-2 text-left border-b border-border text-text-secondary hover:bg-bg-hover hover:text-accent-blue transition-colors"
            >
              <Plus size={13} className="shrink-0" />
              <span className="flex-1 text-xs">Enable more models</span>
              <span className="text-[10px] text-text-muted shrink-0">{disabledCount} available</span>
            </button>
          )}
          <div className="max-h-[360px] overflow-y-auto py-1">
            {groups.length === 0 && (
              <p className="px-3 py-3 text-[10px] leading-relaxed text-text-muted">
                {compatibilityDescription}
              </p>
            )}
            {groups.map(({ family, models: famModels }) => (
              <div key={family.id}>
                {/* Family header */}
                <div className="px-3 pt-2 pb-1 text-[10px] text-text-muted uppercase tracking-wider font-medium">
                  {family.label}
                </div>
                {/* Models in family */}
                {famModels.map(model => {
                  const isSelected = model.model_type === currentModelType
                  const help = model.selector_help
                  return (
                    <div
                      key={model.model_type}
                      className={`group w-full flex items-center transition-colors ${
                        isSelected
                          ? 'bg-accent-blue/10 text-text-primary'
                          : 'hover:bg-bg-hover text-text-secondary hover:text-text-primary'
                      }`}
                    >
                      <button
                        onClick={() => {
                          if (
                            generationMode === 'video'
                            && studioVideoWorkflow === 'frames'
                            && videoImageMode === 0
                          ) selectStudioVideoModel(model.model_type)
                          else selectModel(model.model_type)
                          setOpen(false)
                        }}
                        className="min-w-0 flex-1 px-3 py-1.5 flex items-center gap-2 text-left"
                      >
                        <span className="flex-1 min-w-0 text-xs truncate">{model.name}</span>
                        <ModelBadges model={model} />
                        {isSelected && <Check size={12} className="shrink-0 text-accent-blue" />}
                      </button>
                      {help && (
                        <span className="pr-2">
                          <InfoTooltip
                            text={help}
                            label={`About ${model.name}`}
                          />
                        </span>
                      )}
                    </div>
                  )
                })}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

function ModelBadges({ model }: {
  model: {
    model_type: string
    is_i2v: boolean
    is_t2v: boolean
    supports_end_frame?: boolean
    supports_audio?: boolean
    supports_audio_input?: boolean
    generates_audio?: boolean
    supports_ref_images?: boolean
  }
}) {
  const badges: Array<{ label: string; title: string }> = []
  const workflowIsAlreadyInName = model.model_type.startsWith('minimax_h3')
  if (!workflowIsAlreadyInName && model.is_i2v && model.supports_end_frame) {
    badges.push({ label: 'First / Last', title: 'Accepts a first frame, a last frame, or both' })
  } else if (!workflowIsAlreadyInName && model.is_i2v) {
    badges.push({ label: 'I2V', title: 'Accepts an input image' })
  }
  if (model.generates_audio) {
    badges.push({ label: 'Audio Out', title: 'Generates synchronized audio with the video' })
  }
  if (model.supports_audio_input) {
    badges.push({ label: 'Audio In', title: 'Accepts audio input; H3 Omni uses it as an ordered audio reference' })
  }
  if (model.supports_ref_images) {
    badges.push({ label: 'Refs', title: 'Accepts one or more reference images' })
  }
  if (badges.length === 0) return null
  return (
    <span className="flex gap-0.5 shrink-0">
      {badges.map(b => (
        <span
          key={b.label}
          title={b.title}
          className="text-[9px] px-1 py-0.5 rounded bg-bg-tertiary text-text-muted leading-none"
        >
          {b.label}
        </span>
      ))}
    </span>
  )
}
