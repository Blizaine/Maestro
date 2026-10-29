import { useEffect, useId, useMemo, useRef, useState } from 'react'
import { Loader2, SlidersHorizontal, Sparkles, X } from 'lucide-react'
import { createPortal } from 'react-dom'
import { useStore } from '../stores/useStore'
import { AnimatedTitleControls } from './AnimatedTitleControls'
import { alignTitleSequence, DEFAULT_TITLE_SETTINGS, TITLE_EFFECTS, titleBrief, titleComposerValidation, titleEnhancementEnabled, titlePlanIsCurrent, titleFps, titleFrames, titleSeconds, titleModelType, titleModelProfile, normalizeTitleModelProfile } from './editorAnimatedTitles'
import type { AnimatedTitleSettings } from './editorAnimatedTitles'
import { useEditorTitleStore } from './useEditorTitleStore'
import { useTitleComposer } from './useTitleComposer'
import { StudioTitleFormatControls } from './StudioTitleFormatControls'
import { ModelSelector } from '../components/Sidebar/ModelSelector'
import { fetchDefaults, fetchModelOptions } from '../api/client'

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

/** Studio titles use the same fields and isolated generation payload as Editor titles. */
export function StudioTitlesPanel({ footerHost }: { footerHost?: HTMLElement | null } = {}) {
  const [draft, setDraft] = useState(loadDraft)
  const workspace = useStore(state => state.activeWorkspace)
  const restored = useStore(state => state.params._studio_title_settings)
  const enhanceOnGeneration = useStore(state => state.enhanceOnGeneration)
  const enhanceOnGenerationDefault = useStore(state => state.enhanceOnGenerationDefault)
  const setEnhanceOnGeneration = useStore(state => state.setEnhanceOnGeneration)
  const setEnhanceOnGenerationDefault = useStore(state => state.setEnhanceOnGenerationDefault)
  const busy = useEditorTitleStore(state => state.busy)
  const intent = useEditorTitleStore(state => state.intent)
  const error = useEditorTitleStore(state => state.error)
  const reconcile = useEditorTitleStore(state => state.reconcile)
  const submit = useEditorTitleStore(state => state.submitStudio)
  const loadedSettings = useMemo(() => restored ? normalizeDraft(restored) : null, [restored])
  const settings = loadedSettings ?? draft
  const latestSettings = useRef(settings)
  useEffect(() => { latestSettings.current = settings }, [settings])
  const modelRequest = useRef(0)
  const [modelLoading, setModelLoading] = useState(false)
  const [modelError, setModelError] = useState('')
  const [advancedOpen, setAdvancedOpen] = useState(false)
  const [advancedAnchor, setAdvancedAnchor] = useState<HTMLButtonElement | null>(null)
  const advancedId = useId()
  useEffect(() => () => { modelRequest.current++ }, [])
  useEffect(() => {
    void reconcile()
    const timer = window.setInterval(() => void reconcile(), 5000)
    return () => window.clearInterval(timer)
  }, [reconcile])
  const enhancementEnabled = titleEnhancementEnabled({ enhanceOnGeneration, enhanceOnGenerationDefault })
  useEffect(() => {
    try { localStorage.setItem(DRAFT, JSON.stringify(settings)) } catch { /* Submission separately requires durable storage. */ }
  }, [settings])
  const blocked = busy || Boolean(intent && !['inserted', 'completed', 'failed'].includes(intent.status))
  const patch = (value: Partial<AnimatedTitleSettings>) => {
    setDraft({ ...settings, ...value })
    if (restored) useStore.setState(state => state.params._studio_title_settings === restored
      ? { params: { ...state.params, _studio_title_settings: undefined } } : state)
  }
  const composer = useTitleComposer(settings, patch, workspace)
  const controlBusy = busy || composer.busy || modelLoading
  const chooseModel = async (modelType: string) => {
    const request = ++modelRequest.current
    const sourceWorkspace = workspace, sourceRestored = restored
    setModelLoading(true); setModelError('')
    try {
      const model = useStore.getState().models.find(item => item.model_type === modelType)
      if (!model) throw new Error('This model is no longer available. Reopen the model picker.')
      const [options, defaults] = await Promise.all([fetchModelOptions(modelType), fetchDefaults(modelType)])
      if (request !== modelRequest.current || useStore.getState().activeWorkspace !== sourceWorkspace
        || useStore.getState().params._studio_title_settings !== sourceRestored) return
      const current = latestSettings.current
      const modelProfile = titleModelProfile(model, options)
      const base = { ...current, model: modelProfile, generation: {
        steps: Number(defaults.num_inference_steps ?? options.default_num_inference_steps ?? 30),
        guidance: Number(defaults.guidance_scale ?? options.default_guidance_scale ?? 1), seed: -1,
      } }
      const frames = titleFrames(titleSeconds(current), base)
      let next: AnimatedTitleSettings = { ...base, seconds: frames / titleFps(base), sequenceFrames: frames }
      if (next.mode === 'sequence') {
        try { next = alignTitleSequence(next, frames) } catch { /* Keep the plan visible for correction if the model has a shorter clip limit. */ }
      }
      composer.clearFeedback(); patch(next)
    } catch (reason) {
      if (request === modelRequest.current) setModelError(reason instanceof Error ? reason.message : 'Could not load this model.')
    } finally { if (request === modelRequest.current) setModelLoading(false) }
  }
  const changeDuration = (frames: number) => {
    composer.clearFeedback()
    const base = { ...settings, brief: titleBrief(settings), manualPlanEdited: false, seconds: frames / titleFps(settings), sequenceFrames: frames }
    if (base.mode === 'sequence' && (base.phrases?.length ?? 0) >= 2 && (base.phrases?.length ?? 0) <= 6) {
      try { patch(alignTitleSequence(base, frames)); return } catch { /* Keep the chosen whole-clip duration visible while an invalid sequence is repaired. */ }
    }
    patch(base)
  }
  const validation = titleComposerValidation(settings, CANVAS)
  let reviewReady = false
  try { reviewReady = settings.titleReviewRequired === true && titlePlanIsCurrent(settings) }
  catch { /* Invalid settings stay editable; they cannot be approved as a current plan. */ }
  const generate = async () => {
    const capturedRevision = useStore.getState().enhanceOnGenerationRevision
    const submitOwned = async (planned: AnimatedTitleSettings) => {
      const priorClientId = useEditorTitleStore.getState().intent?.clientId
      await submit(planned)
      const created = useEditorTitleStore.getState().intent
      if (created?.destination === 'studio' && created.clientId !== priorClientId) {
        useStore.setState(state => state.enhanceOnGenerationRevision === capturedRevision
          ? { enhanceOnGeneration: null, enhanceOnGenerationRevision: capturedRevision + 1 } : {})
      }
    }
    await composer.generate(submitOwned, { enhance: enhancementEnabled })
  }
  const generateLabel = reviewReady ? 'Generate reviewed plan' : 'Generate title'
  const footer = <div data-testid="titles-generate-footer" className="shrink-0 bg-bg-secondary px-3 pb-2 pt-2">
    {enhancementEnabled && <div className="mb-2 flex justify-end">
      <button type="button" onClick={() => setEnhanceOnGeneration(false)} aria-label="Remove Enhance on generation" title="Skip enhancement for the next submission"
        className="flex min-h-8 max-w-full items-center gap-1.5 rounded-lg border border-accent-blue/40 bg-accent-blue/10 px-2 text-xs text-accent-blue">
        <Sparkles size={13}/><span>Enhance on generation</span><X size={13}/>
      </button>
    </div>}
    <StudioTitleFormatControls settings={settings} patch={patch} onDurationChange={changeDuration} busy={controlBusy}>
      <button ref={setAdvancedAnchor} type="button" aria-label="Advanced title settings" title="Advanced settings"
        aria-expanded={advancedOpen} aria-controls={advancedId} aria-haspopup="dialog" disabled={controlBusy}
        onClick={() => setAdvancedOpen(value => !value)}
        className={`studio-setting-chip ${advancedOpen ? 'border-accent-blue text-accent-blue' : 'border-border text-text-secondary hover:border-border-light'}`}>
        <SlidersHorizontal size={14}/><span className="studio-advanced-label">Advanced</span>
      </button>
    </StudioTitleFormatControls>
    <div data-testid="studio-generate-bar" className="flex min-w-0 items-center gap-1.5">
      <ModelSelector placement="footer" selection={{modelType: titleModelType(settings), onSelect: modelType => void chooseModel(modelType), disabled: controlBusy}} />
      <button type="button" aria-label={generateLabel} disabled={blocked || controlBusy || Boolean(validation)} onClick={() => void generate()} title={validation || undefined}
        className="flex min-h-11 min-w-[132px] flex-1 items-center justify-center gap-2 rounded-lg bg-accent-blue px-3 py-2 text-sm font-semibold text-white disabled:opacity-40">
        {controlBusy ? <Loader2 size={15} className="animate-spin" /> : <Sparkles size={15} />}{generateLabel}
      </button>
    </div>
  </div>
  const footerNode = footerHost ? createPortal(footer, footerHost) : footer
  return <>
  <section aria-label="Studio Titles" className="mx-3 mb-3 min-w-0 space-y-4 rounded-2xl border border-border bg-bg-primary/30 p-3">
    <AnimatedTitleControls settings={settings} patch={patch} busy={controlBusy} canvas={CANVAS} destination="studio" composer={composer}
      advanced={{open: advancedOpen, anchor: advancedAnchor, id: advancedId, onClose: () => setAdvancedOpen(false)}}
      enhancement={{enabled: enhancementEnabled, defaultEnabled: enhanceOnGenerationDefault,
        onToggle: setEnhanceOnGeneration, onDefaultChange: setEnhanceOnGenerationDefault}} />
    {error && <p role="alert" className="text-xs text-red-400">{error}</p>}
    {modelError && <p role="alert" className="text-xs text-red-400">{modelError}</p>}
    {intent?.error && ['failed', 'completed', 'ready'].includes(intent.status) && <p role="alert" className="text-xs text-red-400">{intent.error}</p>}
    {intent && !['inserted', 'completed', 'failed'].includes(intent.status) && <div role="status" aria-live="polite" className="text-xs text-text-secondary">
      {intent.message || 'Preparing title'}{typeof intent.progress === 'number' ? ` · ${Math.round(intent.progress)}%` : ''}
      {intent.status === 'uncertain' && <button type="button" onClick={() => void reconcile()} className="ml-2 underline">Check receipt</button>}
    </div>}
  </section>
  {footerNode}
  </>
}
