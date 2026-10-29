import { useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import { Globe, Sparkles, X } from 'lucide-react'
import { createPortal } from 'react-dom'
import { useStore } from '../stores/useStore'
import { AnimatedTitleControls } from './AnimatedTitleControls'
import { alignTitleSequence, DEFAULT_TITLE_SETTINGS, TITLE_EFFECTS, titleBrief, titleComposerValidation, titleEnhancementEnabled, titleFps, titleFrames, titleSeconds, titleModelType, titleModelProfile, normalizeTitleModelProfile, titleAdvancedParams, titleDimensions } from './editorAnimatedTitles'
import type { AnimatedTitleSettings } from './editorAnimatedTitles'
import { useEditorTitleStore } from './useEditorTitleStore'
import { useTitleComposer } from './useTitleComposer'
import { StudioTitleFormatControls } from './StudioTitleFormatControls'
import { ModelSelector } from '../components/Sidebar/ModelSelector'
import { AdvancedSettings } from '../components/Sidebar/AdvancedSettings'
import { GenerateButton } from '../components/Sidebar/GenerateButton'
import { sanitizeH3VaeUpsampling } from '../lib/h3VaeUpsampling'

const DRAFT = 'maestro-studio-title-draft-v1'
const CANVAS = { width: 1920, height: 1080 }
function normalizeDraft(raw: unknown): AnimatedTitleSettings {
  const defaults: AnimatedTitleSettings = { ...DEFAULT_TITLE_SETTINGS, aspect: '16:9' }
    if (!raw || typeof raw !== 'object' || Array.isArray(raw)) return defaults
    const value = raw as Partial<AnimatedTitleSettings>
    const normalized: AnimatedTitleSettings = {
      ...defaults,
      ...(typeof value.brief === 'string' ? { brief: value.brief.slice(0, 4000) } : {}),
      stylePreset: typeof value.stylePreset === 'string' && (value.stylePreset === 'custom' || Object.hasOwn(TITLE_EFFECTS, value.stylePreset)) ? value.stylePreset : 'custom',
      ...(typeof value.plannedKey === 'string' ? { plannedKey: value.plannedKey.slice(0, 9000) } : {}),
      ...(Array.isArray(value.plannerWarnings) ? { plannerWarnings: value.plannerWarnings.filter((item): item is string => typeof item === 'string').slice(0, 12).map(item => item.slice(0, 1000)) } : {}),
      ...(typeof value.rawTitleDraft === 'string' ? { rawTitleDraft: value.rawTitleDraft.slice(0, 64000) } : {}),
      ...(typeof value.titleDraftHistoryId === 'string' && /^[a-f0-9-]+$/.test(value.titleDraftHistoryId) ? { titleDraftHistoryId: value.titleDraftHistoryId } : {}),
      titleReviewRequired: value.titleReviewRequired === true,
      manualPlanEdited: value.manualPlanEdited === true,
      directDescription: value.directDescription === true,
      text: typeof value.text === 'string' ? value.text.slice(0, 160) : defaults.text,
      look: typeof value.look === 'string' ? value.look.slice(0, 1200) : defaults.look,
      direction: typeof value.direction === 'string' ? value.direction.slice(0, 1200) : defaults.direction,
      effect: typeof value.effect === 'string' && (value.effect === 'custom' || Object.hasOwn(TITLE_EFFECTS, value.effect)) ? value.effect : defaults.effect,
      seconds: typeof value.seconds === 'number' && Number.isFinite(value.seconds) ? Math.max(5, Math.min(14.5, value.seconds)) : defaults.seconds,
      aspect: value.aspect === '21:9' || value.aspect === '16:9' || value.aspect === '9:16' || value.aspect === '1:1' || value.aspect === '4:3' || value.aspect === '3:4' ? value.aspect : defaults.aspect,
      resolution: value.resolution === '480p' || value.resolution === '540p' || value.resolution === '720p' || value.resolution === '768p' || value.resolution === '1080p' ? value.resolution : defaults.resolution,
      background: ['black', 'white', 'blue', 'custom'].includes(String(value.background)) ? value.background : 'chroma',
      backgroundColor: typeof value.backgroundColor === 'string' && /^#[0-9a-f]{6}$/i.test(value.backgroundColor) ? value.backgroundColor : '#202020',
      model: normalizeTitleModelProfile(value.model),
      generationParams: titleAdvancedParams(value.generationParams),
      generation: value.generation && typeof value.generation === 'object' ? {
        ...(typeof value.generation.steps === 'number' && Number.isFinite(value.generation.steps) ? {steps: value.generation.steps} : {}),
        ...(typeof value.generation.guidance === 'number' && Number.isFinite(value.generation.guidance) ? {guidance: value.generation.guidance} : {}),
        ...(typeof value.generation.seed === 'number' && Number.isFinite(value.generation.seed) ? {seed: value.generation.seed} : {}),
      } : undefined,
      mode: value.mode === 'sequence' ? 'sequence' : 'single',
      sequenceDirection: typeof value.sequenceDirection === 'string' ? value.sequenceDirection.slice(0, 1200) : '',
      ...(value.sequenceFrames !== undefined ? { sequenceFrames: typeof value.sequenceFrames === 'number' ? value.sequenceFrames : NaN } : {}),
      // Keep malformed sequence drafts blocked for review rather than silently
      // turning them into a different single-title generation.
      ...(value.mode === 'sequence' || Array.isArray(value.phrases) ? { phrases: Array.isArray(value.phrases) ? value.phrases.slice(0, 7).map(phrase => ({
        id: typeof phrase?.id === 'string' ? phrase.id.slice(0, 100) : '',
        text: typeof phrase?.text === 'string' ? phrase.text.slice(0, 81) : '',
        seconds: typeof phrase?.seconds === 'number' && Number.isFinite(phrase.seconds) ? phrase.seconds : 0,
        animation: typeof phrase?.animation === 'string' ? phrase.animation.slice(0, 801) : '',
      })) : [] } : {}),
    }
    if (normalized.mode === 'sequence' && normalized.phrases?.every(phrase => Number.isFinite(phrase.seconds) && phrase.seconds >= 1 && phrase.seconds <= 10)) {
      try { return alignTitleSequence(normalized) } catch { /* Keep an invalid saved draft visible for correction. */ }
    }
    return normalized
}
function loadDraft(): AnimatedTitleSettings {
  try { return normalizeDraft(JSON.parse(localStorage.getItem(DRAFT) || 'null')) }
  catch { return normalizeDraft(null) }
}

/** Titles share standard generation controls; the composer owns only creative inputs and native clip geometry. */
export function StudioTitlesPanel({ footerHost }: { footerHost?: HTMLElement | null } = {}) {
  const [draft, setDraft] = useState(loadDraft)
  const workspace = useStore(s => s.activeWorkspace)
  const restored = useStore(s => s.params._studio_title_settings)
  const params = useStore(s => s.params)
  const options = useStore(s => s.modelOptions)
  const optionsLoading = useStore(s => s.modelOptionsLoading)
  const modelsLoaded = useStore(s => s.modelsLoaded)
  const models = useStore(s => s.models)
  const contextActive = useStore(s => Boolean(s.titleGenerationContext))
  const enterContext = useStore(s => s.enterTitleGenerationContext)
  const spatial = useStore(s => s.spatialUpsampling)
  const grain = useStore(s => s.filmGrainIntensity)
  const saturation = useStore(s => s.filmGrainSaturation)
  const cloneEnabled = useStore(s => s.voiceCloneEnabled)
  const cloneMode = useStore(s => s.voiceCloneMode)
  const cloneRefs = useStore(s => s.voiceCloneRefs)
  const enhancementEnabled = useStore(s => titleEnhancementEnabled(s))
  const defaultEnhancement = useStore(s => s.enhanceOnGenerationDefault)
  const setEnhancement = useStore(s => s.setEnhanceOnGeneration)
  const setDefaultEnhancement = useStore(s => s.setEnhanceOnGenerationDefault)
  const busy = useEditorTitleStore(s => s.busy)
  const intent = useEditorTitleStore(s => s.intent)
  const queueIntent = useEditorTitleStore(s => s.queueIntent)
  const error = useEditorTitleStore(s => s.error)
  const reconcile = useEditorTitleStore(s => s.reconcile)
  const submit = useEditorTitleStore(s => s.submitStudio)
  const loaded = useMemo(() => restored ? normalizeDraft(restored) : null, [restored])
  const base = loaded ?? draft
  const initialization = useRef<{done: boolean; restored: unknown}>({done: false, restored: undefined})
  useLayoutEffect(() => {
    if (!modelsLoaded) return
    // Consume a restored snapshot before the controls become interactive.
    // A delayed passive effect must not undo a model the user just selected.
    if (useStore.getState().params._studio_title_settings !== restored) return
    if (!initialization.current.done || (restored && initialization.current.restored !== restored)) {
      initialization.current = {done: true, restored}
      const initial = loaded ?? draft
      const legacy = initial.generation ?? {}
      enterContext(titleModelType(initial), {
        activated_loras: [], loras_multipliers: '',
        ...(legacy.steps !== undefined ? {num_inference_steps: legacy.steps} : {}),
        ...(legacy.guidance !== undefined ? {guidance_scale: legacy.guidance} : {}),
        ...(legacy.seed !== undefined ? {seed: legacy.seed} : {}),
        ...titleAdvancedParams(initial.generationParams),
      })
    }
    if (!restored) initialization.current.restored = undefined
  }, [modelsLoaded, restored, loaded, draft, enterContext])

  const model = models.find(item => item.model_type === params.model_type)
  const profile = useMemo(() => {
    if (!contextActive || !model || !options || optionsLoading || (options.model_type && options.model_type !== model.model_type)) return null
    try { return titleModelProfile(model, options) } catch { return null }
  }, [contextActive, model, options, optionsLoading])
  const advancedSnapshot = useMemo(() => {
    const value: Record<string, unknown> = {
      ...params,
      spatial_upsampling: sanitizeH3VaeUpsampling(spatial, options, 0),
      film_grain_intensity: grain, film_grain_saturation: saturation,
      voice_clone_enabled: cloneEnabled,
      voice_clone_mode: cloneMode,
      voice_clone_refs: cloneEnabled ? cloneRefs.filter(ref => ref.path).map(ref => ref.path) : [],
    }
    if (options?.perturbation && typeof value.stg_scale === 'number') value.perturbation_switch = value.stg_scale > 0 ? 2 : 0
    if (!options?.perturbation) for (const key of ['stg_scale', 'perturbation_switch', 'perturbation_layers', 'perturbation_start_perc', 'perturbation_end_perc']) delete value[key]
    if (!options?.first_block_cache) for (const key of ['skip_steps_cache_type', 'skip_steps_multiplier', 'skip_steps_start_step_perc']) delete value[key]
    if (!options?.reference_pipeline) delete value.reference_pipeline
    return JSON.stringify(titleAdvancedParams(value))
  }, [params, spatial, options, grain, saturation, cloneEnabled, cloneMode, cloneRefs])
  const generationParams = useMemo(() => JSON.parse(advancedSnapshot) as Record<string, unknown>, [advancedSnapshot])
  const settings = useMemo(() => {
    if (!profile) return base
    let next: AnimatedTitleSettings = {...base, model: profile, generation: undefined, generationParams}
    if (JSON.stringify(base.model) !== JSON.stringify(profile)) {
      try {
        const frames = titleFrames(titleSeconds(base), next)
        next = {...next, seconds: frames / profile.fps, sequenceFrames: frames}
        if (next.mode === 'sequence') next = alignTitleSequence(next, frames)
      } catch { /* Invalid legacy plans remain visible and blocked for review. */ }
    }
    return next
  }, [base, profile, generationParams])
  useEffect(() => {
    try { localStorage.setItem(DRAFT, JSON.stringify(settings)) } catch { /* Submissions separately require durable storage. */ }
  }, [settings])
  useEffect(() => {
    if (!profile || !contextActive) return
    try {
      const geometry = titleDimensions(settings, CANVAS)
      const frames = titleFrames(titleSeconds(settings), settings)
      const aspect = settings.aspect === 'project' ? '16:9' : settings.aspect
      useStore.setState(state => state.params.resolution === `${geometry.width}x${geometry.height}` && state.params.video_length === frames
        && state.resolutionPreset === settings.resolution && state.aspectRatio === aspect && state.durationSeconds === frames / profile.fps ? {} : {
        params: {...state.params, resolution: `${geometry.width}x${geometry.height}`, video_length: frames},
        resolutionPreset: settings.resolution, aspectRatio: aspect,
        durationSeconds: frames / profile.fps,
      })
    } catch { /* A malformed saved draft must not change generation geometry. */ }
  }, [settings, profile, contextActive])
  useEffect(() => {
    void reconcile()
    const timer = window.setInterval(() => void reconcile(), 5000)
    return () => window.clearInterval(timer)
  }, [reconcile])
  const patch = (value: Partial<AnimatedTitleSettings>) => {
    setDraft({...settings, ...value})
    if (restored) useStore.setState(state => state.params._studio_title_settings === restored
      ? {params: {...state.params, _studio_title_settings: undefined}} : {})
  }
  const composer = useTitleComposer(settings, patch, workspace)
  const controlBusy = busy || composer.busy || optionsLoading || !profile
  const activeTitle = Boolean(intent && !['inserted', 'completed', 'failed'].includes(intent.status))
  const validation = settings.manualPlanEdited ? 'Enhance this older title draft to review its wording before generating.' : titleComposerValidation(settings, CANVAS)
  const changeDuration = (frames: number) => {
    composer.clearFeedback()
    let next: AnimatedTitleSettings = {...settings, brief: titleBrief(settings), manualPlanEdited: false, seconds: frames / titleFps(settings), sequenceFrames: frames}
    if (next.mode === 'sequence') {
      try { next = alignTitleSequence(next, frames) } catch { /* Keep invalid saved wording visible for correction. */ }
    }
    patch(next)
  }
  const generate = async (action: 'now' | 'queue') => {
    const revision = useStore.getState().enhanceOnGenerationRevision
    await composer.generate(async planned => {
      const accepted = await submit(planned, action)
      const after = useEditorTitleStore.getState()
      if (accepted && !after.error) useStore.setState(state => state.enhanceOnGenerationRevision === revision
        ? {enhanceOnGeneration: null, enhanceOnGenerationRevision: revision + 1} : {})
    }, {enhance: enhancementEnabled})
  }
  const disabled = Boolean(validation || queueIntent) || controlBusy
  const footer = <div data-testid="titles-generate-footer" className="studio-footer shrink-0 bg-bg-secondary px-3 pb-2 pt-2">
    {enhancementEnabled && <div className="mb-2 flex justify-end">
      <button type="button" onClick={() => setEnhancement(false)} aria-label="Remove Enhance on generation" title="Skip enhancement for the next submission"
        className="flex min-h-8 max-w-full items-center gap-1.5 rounded-lg border border-accent-blue/40 bg-accent-blue/10 px-2 text-xs text-accent-blue">
        <Sparkles size={13}/><span>Enhance on generation</span><X size={13}/>
      </button>
    </div>}
    <StudioTitleFormatControls settings={settings} patch={patch} onDurationChange={changeDuration} busy={controlBusy}>
      <AdvancedSettings compact />
    </StudioTitleFormatControls>
    <div data-testid="studio-generate-bar" className="flex min-w-0 items-center gap-1.5">
      <button type="button" aria-label="Open model browser" aria-haspopup="dialog" title="Model Browser"
        onClick={() => useStore.getState().setLoraBrowserOpen(true, params.model_type)}
        className="flex min-h-11 w-8 shrink-0 items-center justify-center rounded-lg border border-border text-text-secondary hover:bg-bg-hover"><Globe size={17}/></button>
      <ModelSelector placement="footer" selection={{modelType: params.model_type, onSelect: type => useStore.getState().selectModel(type), disabled: controlBusy}} />
      <div className="min-w-[132px] flex-1"><GenerateButton stretch submission={{
        onGenerate: () => generate('now'), onQueue: () => generate('queue'), busy: busy || composer.busy,
        disabled: disabled || activeTitle, queueDisabled: disabled,
        disabledReason: validation || (queueIntent ? 'Checking the existing queue submission receipt.' : activeTitle ? 'A title is already running. Add the next title to the queue.' : controlBusy ? 'Loading model controls.' : undefined),
        queueDisabledReason: validation || (queueIntent ? 'Checking the existing queue submission receipt.' : controlBusy ? 'Loading model controls.' : undefined),
      }}/></div>
    </div>
  </div>
  return <>
    <section aria-label="Studio Titles" className="mx-3 mb-3 min-w-0 space-y-4 rounded-2xl border border-border bg-bg-primary/30 p-3">
      <AnimatedTitleControls settings={settings} patch={patch} busy={controlBusy} canvas={CANVAS} destination="studio" composer={composer} showAdvanced={false}
        enhancement={{enabled: enhancementEnabled, defaultEnabled: defaultEnhancement, onToggle: setEnhancement, onDefaultChange: setDefaultEnhancement}} />
      {error && <p role="alert" className="text-xs text-red-400">{error}</p>}
      {intent?.error && ['failed', 'completed', 'ready'].includes(intent.status) && <p role="alert" className="text-xs text-red-400">{intent.error}</p>}
      {queueIntent && <p role="status" className="text-xs text-text-secondary">{queueIntent.error || 'Checking the queue submission receipt.'} <button type="button" className="underline" onClick={() => void reconcile()}>Check receipt</button></p>}
      {activeTitle && <div role="status" aria-live="polite" className="text-xs text-text-secondary">
        {intent?.message || 'Preparing title'}{typeof intent?.progress === 'number' ? ` · ${Math.round(intent.progress)}%` : ''}
        {intent?.status === 'uncertain' && <button type="button" onClick={() => void reconcile()} className="ml-2 underline">Check receipt</button>}
      </div>}
    </section>
    {footerHost ? createPortal(footer, footerHost) : footer}
  </>
}
