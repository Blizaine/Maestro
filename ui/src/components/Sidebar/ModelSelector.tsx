import { ChevronDown, Check, Plus, Globe, RotateCcw } from 'lucide-react'
import { useState, useRef, useEffect, useCallback, type ReactNode } from 'react'
import {
  useStore,
  getFamiliesForMode,
  getModelsForFamily,
  modelSupportsImageWorkflow,
  modelSupportsStudioVideoMediaIntent,
} from '../../stores/useStore'
import { InfoTooltip } from './InfoTooltip'
import { SidebarDialog } from './SidebarPanels'
import { isLongCatAvatarModel } from '../../lib/avatarWorkflow'
import { modelNameForContext } from '../../lib/modelDisplay'
import { getModelVisibilityScope, isModelEnabledInScope } from '../../lib/modelVisibility'

type Placement = 'above' | 'below' | 'footer'

function ModelChoicesPanel({open, placement, onClose, children}: {open: boolean; placement: Placement; onClose: () => void; children: ReactNode}) {
  if (placement === 'footer') return <SidebarDialog open={open} title="Choose a model" variant="settings" onClose={onClose}>{open && children}</SidebarDialog>
  return open ? <div className={`${placement === 'below' ? 'relative mt-2 w-full' : 'absolute bottom-full left-0 mb-1 w-[360px] max-w-[90vw]'} bg-bg-secondary border border-border rounded-xl shadow-xl overflow-hidden z-50`}>{children}</div> : null
}

export function ModelSelector(props: { placement?: Placement }) {
  // A workflow change starts a new picker session, dismissing any open reset.
  const contextKey = useStore(s => [
    s.generationMode, s.studioVideoWorkflow, s.studioImageWorkflow,
    s.audioSubMode, s.editSubMode,
  ].join(':'))
  return <WorkflowModelSelector key={contextKey} {...props} />
}

function WorkflowModelSelector({ placement = 'above' }: { placement?: Placement }) {
  const models = useStore(s => s.models)
  const families = useStore(s => s.families)
  const enabledModels = useStore(s => s.enabledModels)
  const generationMode = useStore(s => s.generationMode)
  const editSubMode = useStore(s => s.editSubMode)
  const imageWorkflow = useStore(s => s.studioImageWorkflow)
  const hasImageReferences = useStore(s => s.imageRefs.length > 0)
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
  const hasFrameAudioDrive = useStore(s => Boolean(s.params.audio_guide))
  const hasReferenceAudioDrive = useStore(s => Boolean(
    s.params.minimax_h3_references?.some(reference => (
      reference.type === 'audio' && reference.audio_intent === 'drive'
    )),
  ))
  const currentModelType = useStore(s => s.params.model_type)
  const selectModel = useStore(s => s.selectModel)
  const controlsPending = useStore(s => s.modelsLoading || (!s.modelsLoaded && !!s.modelsLoadError))
  const selectStudioVideoModel = useStore(s => s.selectStudioVideoModel)
  const openModelVisibility = useStore(s => s.openModelVisibility)
  const resetEnabledModels = useStore(s => s.resetEnabledModels)
  // Mature Mode gate: models with nsfw_only flag are hidden from the
  // selector unless servicesConfig.nsfw_mode is enabled. Backend always
  // ships the entry (so the toggle can show/hide without a model reload)
  // but the UI clamps visibility here.
  const nsfwMode = useStore(s => s.servicesConfig?.nsfw_mode ?? false)

  const [{ open, confirmReset }, setPickerState] = useState({ open: false, confirmReset: false })
  const setOpen = useCallback((open: boolean) => {
    setPickerState({ open, confirmReset: false })
  }, [])
  const setConfirmReset = useCallback((confirmReset: boolean) => {
    setPickerState(state => ({ ...state, confirmReset }))
  }, [])
  const containerRef = useRef<HTMLDivElement>(null)
  const resetButtonRef = useRef<HTMLButtonElement>(null)
  const confirmResetRef = useRef<HTMLButtonElement>(null)

  // Close on click outside
  useEffect(() => {
    if (!open || placement === 'footer') return
    function handleClick(e: MouseEvent) {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) {
        setOpen(false)
      }
    }
    document.addEventListener('mousedown', handleClick)
    return () => document.removeEventListener('mousedown', handleClick)
  }, [open, placement, setOpen])

  const currentModel = models.find(m => m.model_type === currentModelType)
  const audioSubMode = useStore(s => s.audioSubMode)
  const modelVisibilityOverrides = useStore(s => s.modelVisibilityOverrides)
  const visibilityScope = getModelVisibilityScope({
    generationMode,
    studioVideoWorkflow,
    studioImageWorkflow: imageWorkflow,
    audioSubMode,
    editSubMode,
  })
  useEffect(() => {
    if (confirmReset) confirmResetRef.current?.focus()
  }, [confirmReset])
  const createWorkflow = studioVideoWorkflow === 'references' ? 'references'
    : studioVideoWorkflow === 'avatar' ? 'avatar' : 'frames'
  const isPrimaryStudioCreate = generationMode === 'video'
    && ['frames', 'references', 'avatar'].includes(studioVideoWorkflow)
    && videoImageMode === 0
  const studioMediaIntent = {
    workflow: createWorkflow,
    hasFrameGuidance: createWorkflow === 'frames' && hasFrameGuidance,
    hasOmniReferences: createWorkflow === 'references' && hasOmniReferences,
    hasAudioDrive: createWorkflow === 'references' ? hasReferenceAudioDrive : hasFrameAudioDrive,
  } as const
  const currentModelCompatible = Boolean(currentModel)
    && !currentModel?.world_model
    && (!currentModel?.nsfw_only || nsfwMode)
    && isModelEnabledInScope(currentModelType, enabledModels, modelVisibilityOverrides, visibilityScope)
    && (
    generationMode !== 'video' || studioVideoWorkflow === 'avatar' || !isLongCatAvatarModel(currentModel)
  ) && (
    generationMode !== 'image'
    || modelSupportsImageWorkflow(currentModel, imageWorkflow, hasImageReferences)
  ) && (
    !isPrimaryStudioCreate
    || modelSupportsStudioVideoMediaIntent(currentModel, studioMediaIntent)
  )
  const compatibilityDescription = generationMode === 'image'
    ? imageWorkflow === 'generate' && hasImageReferences
      ? 'Add or enable an image edit model for the supplied source or reference images.'
      : `Add or enable an image model compatible with ${imageWorkflow}.`
    : createWorkflow === 'avatar'
      ? 'Add or enable a LongCat Avatar Single or Multi model for image-and-voice animation.'
      : createWorkflow === 'references'
      ? 'Add or enable an H3 Omni model for these references or characters.'
      : hasFrameGuidance && hasFrameAudioDrive
        ? 'Add or enable a frame-capable model that accepts an exact audio timeline.'
        : hasFrameGuidance
          ? 'Add or enable an image-to-video model for these frame inputs.'
          : hasFrameAudioDrive
            ? 'Add or enable a video model that accepts an audio timeline.'
            : 'Add or enable a text-to-video or image-to-video model.'
  const effectiveSubMode = generationMode === 'avatar' ? editSubMode : undefined
  const effectiveAudioSubMode = generationMode === 'audio' ? audioSubMode : undefined
  const modeFamilies = getFamiliesForMode(generationMode, families, effectiveSubMode, effectiveAudioSubMode)

  // Build grouped model list, filtered by:
  //   1. the effective model choice for this workflow,
  //   2. nsfw_only gate (Mature Mode must be on for those to appear).
  const workflowFilter = (model: typeof models[number]) => (
    !model.world_model
    && (generationMode !== 'video' || (studioVideoWorkflow === 'animate'
      ? model.model_type === 'viggle_animate' : model.model_type !== 'viggle_animate'))
    && (generationMode !== 'video' || studioVideoWorkflow === 'avatar' || !isLongCatAvatarModel(model))
    &&
    (generationMode !== 'image' || modelSupportsImageWorkflow(model, imageWorkflow, hasImageReferences))
    && (
      !isPrimaryStudioCreate
      || modelSupportsStudioVideoMediaIntent(model, studioMediaIntent)
    )
  )
  const groups = modeFamilies.map(family => ({
    family,
    models: getModelsForFamily(family.id, models, generationMode, effectiveSubMode)
      .filter(workflowFilter)
      .filter(m => isModelEnabledInScope(m.model_type, enabledModels, modelVisibilityOverrides, visibilityScope))
      .filter(m => !m.nsfw_only || nsfwMode),
  })).filter(g => g.models.length > 0)

  // Count compatible choices hidden in this workflow. Keep settings reachable
  // even when every compatible default is already enabled.
  const disabledCount = modeFamilies.reduce((n, family) => {
    const avail = getModelsForFamily(family.id, models, generationMode, effectiveSubMode)
      .filter(workflowFilter)
      .filter(m => !m.nsfw_only || nsfwMode)
    return n + avail.filter(m => !isModelEnabledInScope(
      m.model_type, enabledModels, modelVisibilityOverrides, visibilityScope,
    )).length
  }, 0)

  return (
    <div className={`relative min-w-0 ${placement !== 'below' ? 'flex-1' : ''}`} ref={containerRef}>
      {/* Trigger button */}
      <button
        aria-label="Choose model" aria-expanded={open}
        disabled={controlsPending}
        onClick={() => setOpen(!open)}
        title={currentModelCompatible
          ? `${currentModel?.name || 'Select model'}\n${currentModel?.selector_help || currentModel?.description || ''}`
          : compatibilityDescription}
        className={`w-full flex items-center gap-1.5 bg-bg-tertiary border border-border rounded-lg px-2 py-2 text-left hover:border-border-light transition-colors ${placement === 'footer' ? 'min-h-11' : ''}`}
      >
        <span className={`flex-1 min-w-0 text-text-primary ${placement === 'footer' ? 'line-clamp-2 text-[11px] leading-tight' : 'truncate text-xs'}`}>
          {controlsPending ? 'Loading model controls…' : currentModelCompatible && currentModel
            ? modelNameForContext(currentModel, visibilityScope)
            : currentModel ? 'No compatible model' : 'Select model'}
        </span>
        <ChevronDown size={14} className={`shrink-0 text-text-muted transition-transform ${open ? 'rotate-180' : ''}`} />
      </button>

      <ModelChoicesPanel open={open} placement={placement} onClose={() => setOpen(false)}>
          {placement === 'footer' && <button type="button" onClick={() => { setOpen(false); useStore.getState().setLoraBrowserOpen(true, currentModelType) }}
            className="mb-2 flex min-h-10 w-full items-center gap-2 rounded-lg border border-border px-3 text-xs text-text-secondary hover:bg-bg-hover">
            <Globe size={14}/> Browse models, LoRAs & characters
          </button>}
          {/* Always keep visibility settings reachable, including modes whose
              defaults already have every compatible model enabled. */}
          <div className="flex border-b border-border">
            <button
              type="button"
              onClick={() => { openModelVisibility(visibilityScope); setOpen(false) }}
              className="min-w-0 flex-1 flex items-center gap-2 px-3 py-2 text-left text-text-secondary hover:bg-bg-hover hover:text-accent-blue transition-colors"
            >
              <Plus size={13} className="shrink-0" />
              <span className="flex-1 text-xs">{disabledCount > 0 ? 'Enable more models' : 'Manage models'}</span>
              {disabledCount > 0 && <span className="text-[10px] text-text-muted shrink-0">{disabledCount} available</span>}
            </button>
            <button
              type="button"
              ref={resetButtonRef}
              aria-label="Reset model visibility for current workflow"
              title="Reset this workflow to its default models"
              onClick={() => setConfirmReset(true)}
              className="shrink-0 flex items-center gap-1.5 border-l border-border px-3 py-2 text-text-secondary hover:bg-bg-hover hover:text-accent-blue transition-colors"
            >
              <RotateCcw size={13} className="shrink-0" />
              <span className="text-xs">Reset</span>
            </button>
          </div>
          {confirmReset && (
            <div
              role="alertdialog"
              aria-labelledby="model-visibility-reset-title"
              aria-describedby="model-visibility-reset-description"
              className="border-b border-border bg-bg-secondary px-3 py-3"
            >
              <p id="model-visibility-reset-title" className="text-xs font-medium text-text-primary">
                Reset this workflow to its default models?
              </p>
              <p id="model-visibility-reset-description" className="mt-1 text-[11px] leading-relaxed text-text-muted">
                This changes only the current workflow. No model files will be deleted. Hidden models can be re-enabled in the model picker with Enable more models.
              </p>
              <div className="mt-3 flex justify-end gap-2">
                <button
                  type="button"
                  onClick={() => { setConfirmReset(false); resetButtonRef.current?.focus() }}
                  className="rounded-md px-2.5 py-1.5 text-xs text-text-secondary hover:bg-bg-hover"
                >
                  Cancel
                </button>
                <button
                  type="button"
                  ref={confirmResetRef}
                  aria-label="Confirm reset"
                  onClick={() => {
                    resetEnabledModels(visibilityScope)
                    setConfirmReset(false)
                    resetButtonRef.current?.focus()
                  }}
                  className="rounded-md bg-accent-blue px-2.5 py-1.5 text-xs text-white hover:opacity-90"
                >
                  Reset
                </button>
              </div>
            </div>
          )}
          <div className={placement === 'footer' ? 'py-1' : 'max-h-[360px] overflow-y-auto py-1'}>
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
                        title={model.name}
                        onClick={() => {
                          if (
                            generationMode === 'video'
                            && ['frames', 'references', 'avatar'].includes(studioVideoWorkflow)
                            && videoImageMode === 0
                          ) selectStudioVideoModel(model.model_type)
                          else selectModel(model.model_type)
                          setOpen(false)
                        }}
                        className="min-w-0 flex-1 px-3 py-1.5 flex items-center gap-2 text-left"
                      >
                        <span className="flex-1 min-w-0 text-xs truncate">{modelNameForContext(model, visibilityScope)}</span>
                        <ModelBadges model={model} scope={visibilityScope} />
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
      </ModelChoicesPanel>
    </div>
  )
}

function ModelBadges({ model, scope }: {
  scope: ReturnType<typeof getModelVisibilityScope>
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
  if (model.is_i2v && model.supports_end_frame) {
    badges.push({ label: 'First / Last', title: 'Accepts a first frame, a last frame, or both' })
  } else if (model.is_i2v) {
    badges.push({ label: 'I2V', title: 'Accepts an input image' })
  }
  if (model.generates_audio) {
    badges.push({ label: 'Audio Out', title: 'Generates synchronized audio with the video' })
  }
  if (model.supports_audio_input) {
    badges.push({ label: 'Audio In', title: 'Accepts audio input; H3 Omni uses it as an ordered audio reference' })
  }
  if (model.supports_ref_images && scope !== 'video:references') {
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
