import { create } from 'zustand'
import * as api from '../api/client'
import type { EditorAsset, EditorUpscaleMethod, GenerationJob, ModelDef } from '../types'
import { flashvsrScale } from '../lib/flashvsr'
import { modelSupportsStudioVideoMediaIntent, useStore } from '../stores/useStore'
import { animatedTitleParams, animatedTitlePrompt, titleBackground, titleDimensions, titleFps, titleFrames, titleModelProfile, titleModelType, titlePlanIsCurrent, titlePlanningKey, titleSeconds, titleText, titleSubmissionId } from './editorAnimatedTitles'
import type { AnimatedTitleSettings } from './editorAnimatedTitles'
import { useEditorStore } from './useEditorStore'

const STORAGE = 'maestro-editor-animated-title-v1'
const STUDIO_QUEUE_STORAGE = 'maestro-studio-title-queue-v1'
interface TitleIntent {
  id: string; clientId: string; projectId?: string; workspace: string; at: number
  /** Absent on legacy Editor-only intents. Studio never inserts into a project. */
  destination?: 'editor' | 'studio'
  settings: AnimatedTitleSettings; width: number; height: number
  status: 'submitting' | 'uncertain' | 'queued' | 'running' | 'ready' | 'inserted' | 'completed' | 'failed'
  jobId?: string; asset?: EditorAsset; progress?: number; message?: string; error?: string
}
interface StudioQueueIntent {
  id: string; clientId: string; workspace: string; at: number
  settings?: AnimatedTitleSettings; width: number; height: number
  status: 'submitting' | 'uncertain'
  jobId?: string; error?: string
}
function restore(): TitleIntent | null {
  try {
    const value = JSON.parse(localStorage.getItem(STORAGE) || 'null') as TitleIntent | null
    if (!value || typeof value.id !== 'string' || typeof value.clientId !== 'string'
      || (value.destination !== undefined && !['editor', 'studio'].includes(value.destination))
      || (value.destination !== 'studio' && typeof value.projectId !== 'string') || typeof value.workspace !== 'string'
      || !Number.isFinite(value.at) || value.at < 0 || !value.settings) return null
    animatedTitlePrompt(value.settings); titleFrames(titleSeconds(value.settings), value.settings)
    if (!['submitting', 'uncertain', 'queued', 'running', 'ready', 'inserted', 'completed', 'failed'].includes(value.status)) return null
    return value.status === 'submitting' ? { ...value, status: 'uncertain' } : value
  } catch { return null }
}
function remember(intent: TitleIntent | null) {
  // If durable intent cannot be saved, do not submit an unrecoverable GPU job.
  if (intent) localStorage.setItem(STORAGE, JSON.stringify(intent))
  else localStorage.removeItem(STORAGE)
}
function restoreStudioQueueIntent(): StudioQueueIntent | null {
  try {
    const value = JSON.parse(localStorage.getItem(STUDIO_QUEUE_STORAGE) || 'null') as StudioQueueIntent | null
    if (!value || typeof value.id !== 'string' || typeof value.clientId !== 'string'
      || typeof value.workspace !== 'string' || !value.workspace) return null
    // Keep ownership if a saved payload was partially damaged; exact receipt
    // recovery needs only this client ID and must never be replaced by a new POST.
    return { ...value, status: 'uncertain' }
  } catch { return null }
}
function rememberStudioQueueIntent(intent: StudioQueueIntent | null) {
  if (intent) localStorage.setItem(STUDIO_QUEUE_STORAGE, JSON.stringify(intent))
  else localStorage.removeItem(STUDIO_QUEUE_STORAGE)
}
async function backgroundFrame(width: number, height: number, settings: AnimatedTitleSettings): Promise<File> {
  const background = titleBackground(settings)
  const canvas = document.createElement('canvas')
  canvas.width = width; canvas.height = height
  const context = canvas.getContext('2d')
  if (!context) throw new Error('Could not prepare the title background.')
  context.fillStyle = background.color; context.fillRect(0, 0, width, height)
  const blob = await new Promise<Blob>((resolve, reject) => canvas.toBlob(value => value
    ? resolve(value) : reject(new Error('Could not encode the title background.')), 'image/png'))
  return new File([blob], `title-${background.fileLabel}-${width}x${height}.png`, { type: 'image/png' })
}
async function prepareStudioTitle(settings: AnimatedTitleSettings, workspace: string, clientId: string) {
  const canvas = { width: 1920, height: 1080 }
  // Snapshot the advanced controls before any model or upload request yields.
  const generationParams = settings.generationParams
    ? JSON.parse(JSON.stringify(settings.generationParams)) as Record<string, unknown>
    : undefined
  const authoredSettings = {
    ...settings,
    ...(generationParams ? { generationParams } : {}),
    text: settings.mode === 'sequence'
      ? (settings.phrases ?? []).map(phrase => phrase.text.trim()).join('\n') : settings.text.trim(),
    ...(settings.phrases ? { phrases: settings.phrases.map(phrase => ({ ...phrase })) } : {}),
  }
  const selectedModelType = titleModelType(authoredSettings)
  const { models } = await api.fetchModels()
  const model = models.find(item => item.model_type === selectedModelType)
  if (!model) throw new Error('The selected video model is no longer available. Choose another model and try again.')
  if (model.is_downloaded !== true) throw new Error(`Install ${model.name} in Settings before generating an animated title.`)
  const modelDefinition = model as ModelDef
  if (modelDefinition.world_model || !modelSupportsStudioVideoMediaIntent(modelDefinition, { workflow: 'frames', hasFrameGuidance: false,
    hasOmniReferences: false, hasAudioDrive: false })) {
    throw new Error('Choose a video model compatible with the standard Frames workflow.')
  }
  const [options, defaults] = await Promise.all([
    api.fetchModelOptions(selectedModelType), api.fetchDefaults(selectedModelType),
  ])
  if (options.model_type !== undefined && options.model_type !== selectedModelType) throw new Error('The selected model options changed. Refresh Titles and try again.')
  const profile = titleModelProfile(modelDefinition, options)
  const settingsCopy = { ...authoredSettings, model: profile }
  // Legacy Editor plans have no model profile in their saved key. The
  // Editor remains bound to H3, so migrate only that fixed-model key.
  if (!authoredSettings.model && authoredSettings.brief !== undefined && titlePlanIsCurrent(authoredSettings)) {
    settingsCopy.plannedKey = titlePlanningKey(settingsCopy)
  }
  animatedTitlePrompt(settingsCopy); titleFrames(titleSeconds(settingsCopy), settingsCopy)
  const { width, height } = titleDimensions(settingsCopy, canvas)
  const background = profile.startFrame
    ? await api.uploadImage(await backgroundFrame(width, height, settingsCopy), { reuseIdentical: true })
    : null
  if (useStore.getState().activeWorkspace !== workspace) {
    throw new Error('The workspace changed before submission. No title was generated.')
  }
  const params = {
    ...animatedTitleParams(settingsCopy, canvas, background?.path ?? '', workspace, clientId, { options, defaults }),
    _studio_video_workflow: 'titles',
    _studio_title_settings: settingsCopy,
  }
  return { settings: settingsCopy, width, height, params }
}
let submitting = false, reconciling = false
let submittingStudioQueue = false, reconcilingStudioQueue = false

interface TitleState {
  intent: TitleIntent | null; queueIntent: StudioQueueIntent | null; busy: boolean; error: string | null
  submit: (settings: AnimatedTitleSettings) => Promise<void>
  submitStudio: (settings: AnimatedTitleSettings, action?: 'now' | 'queue') => Promise<boolean>
  reconcile: () => Promise<void>
  reconcileQueue: () => Promise<void>
  insertReady: () => Promise<void>
  dismiss: () => void
}
export const useEditorTitleStore = create<TitleState>((set, get) => {
  const update = (intent: TitleIntent) => {
    try { remember(intent); set({ intent }) }
    catch { set({ intent, error: 'Browser storage is full. Keep this page open to track the existing title job.' }) }
  }
  const updateQueue = (intent: StudioQueueIntent) => {
    try { rememberStudioQueueIntent(intent); set({ queueIntent: intent }) }
    catch { set({ queueIntent: intent, error: 'Browser storage is full. Keep this page open to track the held title receipt.' }) }
  }
  const clearQueueIntent = () => {
    try { rememberStudioQueueIntent(null); set({ queueIntent: null }) }
    catch { set({ error: 'Could not clear the recovered held-title receipt. Check the queue again before submitting another title.' }) }
  }
  const submitFor = async (settings: AnimatedTitleSettings, destination: 'editor' | 'studio') => {
      if (submitting || get().busy) return
      if (destination === 'studio' && get().queueIntent) {
        set({ error: 'Check the existing held-title receipt before generating another title.' }); return
      }
      const prior = get().intent
      if (prior && !['inserted', 'completed', 'failed'].includes(prior.status)) {
        set({ error: 'Finish or stop tracking the current title before generating another.' }); return
      }
      submitting = true; set({ busy: true, error: null })
      let intent: TitleIntent | null = null, posted = false
      try {
        const current = destination === 'editor' ? useEditorStore.getState().project : null
        if (destination === 'editor' && !current) throw new Error('Open an Editor project first.')
        const workspace = current?.workspace ?? useStore.getState().activeWorkspace
        if (!workspace || ['__all__', '__uploads__'].includes(workspace)) throw new Error('Choose a workspace for the title.')
        const canvas = current ? { ...current.canvas } : { width: 1920, height: 1080 }
        const generationParams = settings.generationParams
          ? JSON.parse(JSON.stringify(settings.generationParams)) as Record<string, unknown>
          : undefined
        const authoredSettings = { ...settings, ...(generationParams ? { generationParams } : {}), text: settings.mode === 'sequence'
          ? (settings.phrases ?? []).map(phrase => phrase.text.trim()).join('\n') : settings.text.trim(),
          ...(settings.phrases ? { phrases: settings.phrases.map(phrase => ({ ...phrase })) } : {}) }
        const selectedModelType = titleModelType(authoredSettings)
        const { models } = await api.fetchModels()
        const model = models.find(item => item.model_type === selectedModelType)
        if (!model) throw new Error('The selected video model is no longer available. Choose another model and try again.')
        if (model.is_downloaded !== true) throw new Error(`Install ${model.name} in Settings before generating an animated title.`)
        const modelDefinition = model as ModelDef
        if (modelDefinition.world_model || !modelSupportsStudioVideoMediaIntent(modelDefinition, { workflow: 'frames', hasFrameGuidance: false,
          hasOmniReferences: false, hasAudioDrive: false })) {
          throw new Error('Choose a video model compatible with the standard Frames workflow.')
        }
        const [options, defaults] = await Promise.all([
          api.fetchModelOptions(selectedModelType), api.fetchDefaults(selectedModelType),
        ])
        if (options.model_type !== undefined && options.model_type !== selectedModelType) throw new Error('The selected model options changed. Refresh Titles and try again.')
        const profile = titleModelProfile(model as ModelDef, options)
        const settingsCopy = { ...authoredSettings, model: profile }
        // Legacy Editor plans have no model profile in their saved key. The
        // Editor remains bound to H3, so migrate only that fixed-model key.
        if (!authoredSettings.model && authoredSettings.brief !== undefined && titlePlanIsCurrent(authoredSettings)) {
          settingsCopy.plannedKey = titlePlanningKey(settingsCopy)
        }
        animatedTitlePrompt(settingsCopy); titleFrames(titleSeconds(settingsCopy), settingsCopy)
        const { width, height } = titleDimensions(settingsCopy, canvas)
        const at = current ? useEditorStore.getState().playhead : 0
        if (at + titleFrames(titleSeconds(settingsCopy), settingsCopy) / titleFps(settingsCopy) > 86400) {
          throw new Error('The title would exceed the 24-hour timeline limit.')
        }
        if (current) {
          const saved = await useEditorStore.getState().saveProject()
          if (!saved || saved.id !== current.id || saved.workspace !== current.workspace
            || useEditorStore.getState().project?.id !== current.id) throw new Error('The Editor project changed while preparing the title.')
        }
        const background = profile.startFrame
          ? await api.uploadImage(await backgroundFrame(width, height, settingsCopy), { reuseIdentical: true })
          : null
        if (current && (useEditorStore.getState().project?.id !== current.id || useEditorStore.getState().project?.workspace !== current.workspace)) {
          throw new Error('The Editor project changed before submission. No title was generated.')
        }
        if (destination === 'studio' && useStore.getState().activeWorkspace !== workspace) {
          throw new Error('The workspace changed before submission. No title was generated.')
        }
        intent = { id: titleSubmissionId(), clientId: titleSubmissionId(), projectId: current?.id,
          destination, workspace, at, settings: settingsCopy, width, height, status: 'submitting' }
        const params = { ...animatedTitleParams(settingsCopy, canvas, background?.path ?? '', workspace, intent.clientId,
          { options, defaults }),
          ...(destination === 'studio' ? { _studio_video_workflow: 'titles', _studio_title_settings: settingsCopy } : {}) }
        remember(intent); set({ intent }) // Validated and durable before the one and only generation POST.
        posted = true
        const receipt = await api.submitGeneration(params)
        if (!receipt.job_id) throw new Error('The title submission returned no job identity.')
        update({ ...intent, jobId: receipt.job_id, status: 'queued', message: 'Queued for title generation' })
      } catch (error) {
        const message = error instanceof Error ? error.message : 'Could not prepare this title.'
        if (intent && posted) {
          update({ ...intent, status: 'uncertain', error: `${message} Checking the exact submission receipt; this title will not be submitted again automatically.` })
        } else set({ error: message })
      } finally {
        submitting = false; set({ busy: false })
      }
      if (posted) void get().reconcile()
  }
  const submitStudioQueue = async (settings: AnimatedTitleSettings) => {
    if (submittingStudioQueue || submitting || get().busy) return false
    if (get().queueIntent) {
      set({ error: 'Check the existing held-title receipt before queuing another title.' })
      return false
    }
    submittingStudioQueue = true; set({ busy: true, error: null })
    let intent: StudioQueueIntent | null = null, posted = false
    try {
      const workspace = useStore.getState().activeWorkspace
      if (!workspace || ['__all__', '__uploads__'].includes(workspace)) throw new Error('Choose a workspace for the title.')
      const clientId = titleSubmissionId()
      const prepared = await prepareStudioTitle(settings, workspace, clientId)
      intent = { id: titleSubmissionId(), clientId, workspace, at: 0, settings: prepared.settings,
        width: prepared.width, height: prepared.height, status: 'submitting' }
      rememberStudioQueueIntent(intent)
      set({ queueIntent: intent }) // Durable before the one and only held-generation POST.
      posted = true
      const receipt = await api.submitGeneration(prepared.params, true)
      if (!receipt.job_id) throw new Error('The held title submission returned no job identity.')
      updateQueue({ ...intent, jobId: receipt.job_id, status: 'uncertain', error: 'Checking the exact held-title receipt.' })
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Could not prepare this title.'
      if (intent && posted) {
        updateQueue({ ...intent, status: 'uncertain', error: `${message} Checking the exact submission receipt; this title will not be submitted again automatically.` })
      } else set({ error: message })
    } finally {
      submittingStudioQueue = false; set({ busy: false })
    }
    if (posted) await get().reconcileQueue()
    return posted
  }
  return {
    intent: restore(), queueIntent: restoreStudioQueueIntent(), busy: false, error: null,
    submit: settings => submitFor(settings, 'editor'),
    submitStudio: async (settings, action = 'now') => {
      if (action === 'queue') return submitStudioQueue(settings)
      const prior = get().intent?.clientId
      await submitFor(settings, 'studio')
      return Boolean(get().intent && get().intent?.clientId !== prior)
    },
    reconcile: async () => {
      void get().reconcileQueue()
      const intent = get().intent
      if (!intent || submitting || reconciling || ['inserted', 'completed', 'failed'].includes(intent.status)) return
      if (intent.status === 'ready' && intent.destination !== 'studio') {
        const project = useEditorStore.getState().project
        if (!intent.error && project && project.id === intent.projectId && project.workspace === intent.workspace) await get().insertReady()
        return
      }
      reconciling = true
      try {
        let jobId = intent.jobId
        if (!jobId) {
          const receipt = await api.fetchJobBySubmission(intent.clientId)
          if (!receipt) {
            update({ ...intent, status: 'uncertain', error: 'The title receipt is not available yet. Check the queue or check again; nothing will be resubmitted.' })
            return
          }
          jobId = receipt.job_id
        }
        const job = await api.fetchJobStatus(jobId)
        if (job.client_submission_id !== intent.clientId || job.job_id !== jobId) throw new Error('The job receipt does not match this title submission.')
        if (get().intent?.id !== intent.id) return
        if (intent.destination === 'studio' && ['held', 'queued', 'running'].includes(job.status)) {
          // Reuse the ordinary Studio queue/progress polling after an exact
          // receipt check, including when this panel is subsequently closed.
          await useStore.getState().reconnectJobs({ id: jobId, showInGallery: true, kind: 'generation',
            status: job.status, progress: job.progress / 100, step: job.step, totalSteps: job.total_steps,
            phase: job.phase, message: job.message, outputFiles: job.output_files, error: job.error })
          if (get().intent?.id !== intent.id) return
        }
        if (job.status === 'failed' || job.status === 'cancelled') {
          update({ ...intent, jobId, status: 'failed', error: job.error || job.message || 'Title generation did not complete.' }); return
        }
        if (job.status !== 'completed') {
          update({ ...intent, jobId, status: job.status === 'running' ? 'running' : 'queued',
            progress: job.progress, message: job.phase || job.message || 'Title queued', error: undefined }); return
        }
        const videoFiles = [...new Set(job.output_files.filter(file => /\.(mp4|mov|webm|mkv)$/i.test(file)))]
        const filename = videoFiles.at(-1)
        if (!filename) throw new Error('The completed title has no video output. Check its generation history.')
        const name = filename.replace(/\\/g, '/').split('/').at(-1) || filename
        const library: EditorAsset = { id: `title-${intent.id}`, name, type: 'video', origin: 'output',
          workspace: intent.workspace, path: filename, url: api.getFileUrl(name, intent.workspace),
          duration: 0, width: 0, height: 0, fps: 0, has_audio: false }
        const probe = await api.probeEditorMedia(library, intent.workspace)
        if (probe.type !== 'video' || !(probe.duration > 0) || !(probe.width > 0) || !(probe.height > 0)) throw new Error('The title output could not be verified as usable video.')
        if (get().intent?.id !== intent.id) return
        const asset = { ...library, path: probe.path, duration: probe.duration, width: probe.width, height: probe.height,
          fps: probe.fps, has_audio: probe.has_audio, size: probe.size }
        const spatialUpscaling = intent.destination === 'studio'
          ? intent.settings.generationParams?.spatial_upsampling : undefined
        const namedScale = typeof spatialUpscaling === 'string'
          ? /^(?:lanczos|vae)([\d.]+)$|^(?:dlss5|h3_vae)\*([\d.]+)$/.exec(spatialUpscaling) : null
        const expectedScale = namedScale ? Number(namedScale[1] || namedScale[2])
          : typeof spatialUpscaling === 'string' ? flashvsrScale(spatialUpscaling as EditorUpscaleMethod) : 1
        const expectedWidth = Math.round(intent.width * expectedScale / 2) * 2
        const expectedHeight = Math.round(intent.height * expectedScale / 2) * 2
        // Finishing may retain the native clip alongside its processed output.
        const sizeMatches = (probe.width === intent.width && probe.height === intent.height)
          || (probe.width === expectedWidth && probe.height === expectedHeight)
        if (intent.destination === 'studio') {
          // Standard Output Count and finishing can produce several videos.
          // Check each returned video; the normal gallery retains all of them.
          const sizeWarnings = sizeMatches ? [] : [`${name}: ${probe.width}×${probe.height}`]
          for (const output of videoFiles.slice(0, -1)) {
            const outputName = output.replace(/\\/g, '/').split('/').at(-1) || output
            const other = await api.probeEditorMedia({origin: 'output', name: outputName,
              path: output, workspace: intent.workspace}, intent.workspace)
            if (other.type !== 'video' || !(other.duration > 0) || !(other.width > 0) || !(other.height > 0)) {
              throw new Error(`The title output ${outputName} could not be verified as usable video.`)
            }
            if (get().intent?.id !== intent.id) return
            if (!(other.width === intent.width && other.height === intent.height)
              && !(other.width === expectedWidth && other.height === expectedHeight)) {
              sizeWarnings.push(`${outputName}: ${other.width}×${other.height}`)
            }
          }
          const titleBackgroundInfo = titleBackground(intent.settings)
          update({ ...intent, jobId, asset, status: 'completed', progress: 100,
            message: `Title saved to ${intent.workspace}. ${titleBackgroundInfo.chromaKey
              ? `The ${titleBackgroundInfo.label.toLowerCase()} background is ready for keying.`
              : `The ${titleBackgroundInfo.label.toLowerCase()} background is part of the video.`}`,
            error: sizeWarnings.length ? `Unexpected output dimensions (expected ${intent.width}×${intent.height} or ${expectedWidth}×${expectedHeight}): ${sizeWarnings.join('; ')}.` : undefined })
          await useStore.getState().refreshOutputs()
          return
        }
        update({ ...intent, jobId, asset, status: 'ready', progress: 100,
          error: sizeMatches ? undefined : `The video is ${probe.width}×${probe.height}; ${expectedWidth}×${expectedHeight} was requested. Review it before adding.`,
          message: 'Title ready' })
        if (sizeMatches) await get().insertReady()
      } catch (error) {
        if (get().intent?.id === intent.id) update({ ...get().intent!, status: 'uncertain',
          error: error instanceof Error ? error.message : 'Could not check this title. Check again to recover its receipt.' })
      } finally { reconciling = false }
    },
    reconcileQueue: async () => {
      const intent = get().queueIntent
      if (!intent || submittingStudioQueue || reconcilingStudioQueue) return
      reconcilingStudioQueue = true
      let jobId = intent.jobId
      try {
        if (!jobId) {
          const receipt = await api.fetchJobBySubmission(intent.clientId)
          if (!receipt) {
            if (get().queueIntent?.id === intent.id) updateQueue({ ...intent, status: 'uncertain',
              error: 'The held-title receipt is not available yet. Check again; nothing will be resubmitted.' })
            return
          }
          if (receipt.client_submission_id !== intent.clientId || !receipt.job_id) {
            throw new Error('The held-title submission receipt does not match this title.')
          }
          jobId = receipt.job_id
          updateQueue({ ...intent, jobId, status: 'uncertain', error: 'Verifying the held title before adding it to the Studio queue.' })
        }
        const job = await api.fetchJobStatus(jobId)
        if (job.client_submission_id !== intent.clientId || job.job_id !== jobId) {
          throw new Error('The job receipt does not match this held title submission.')
        }
        if (get().queueIntent?.id !== intent.id) return
        // A separate Start Queue action can release the job while this exact
        // receipt is being checked. Adopt its verified current state either way.
        const confirmedJob: GenerationJob = {
          id: jobId,
          restoredFromHistory: true,
          showInGallery: job.show_in_gallery === true,
          directorPipelineId: job.director_pipeline_id ?? null,
          directorDetachedOperation: job.director_detached_operation === true,
          kind: job.kind || 'generation',
          status: job.status,
          progress: job.progress / 100,
          step: job.step,
          totalSteps: job.total_steps,
          phase: job.phase,
          message: job.message,
          outputFiles: job.output_files,
          finishingWarnings: job.finishing_warnings,
          error: job.error,
          oomInfo: job.oom_info ?? null,
          h3WindowPlan: job.h3_window_plan ?? null,
          enhancement: job.enhancement,
          preview: job.preview ?? null,
          previewNotice: job.preview_notice ?? null,
          currentClip: job.current_clip,
          totalClips: job.total_clips,
          currentWindow: job.current_window,
          totalWindows: job.total_windows,
          windowEtaSeconds: job.window_eta_seconds,
          clipEtaSeconds: job.clip_eta_seconds,
          generationEtaSeconds: job.generation_eta_seconds,
          projectEtaSeconds: job.project_eta_seconds,
          windowCompletionAt: job.window_completion_at,
          clipCompletionAt: job.clip_completion_at,
          generationCompletionAt: job.generation_completion_at,
          projectCompletionAt: job.project_completion_at,
          etaConfidence: job.eta_confidence,
          etaBasis: job.eta_basis,
          etaHistorySamples: job.eta_history_samples,
          etaHistoryMatch: job.eta_history_match,
        }
        await useStore.getState().reconnectJobs(confirmedJob)
        if (get().queueIntent?.id === intent.id) clearQueueIntent()
      } catch (error) {
        if (get().queueIntent?.id === intent.id) updateQueue({ ...intent, ...(jobId ? { jobId } : {}), status: 'uncertain',
          error: error instanceof Error
            ? `${error.message} Check the exact receipt again; this title will not be resubmitted.`
            : 'Could not check the held-title receipt. Check again; this title will not be resubmitted.' })
      } finally { reconcilingStudioQueue = false }
    },
    insertReady: async () => {
      const intent = get().intent
      if (!intent?.asset || intent.destination === 'studio' || !intent.projectId || !['ready', 'inserted'].includes(intent.status)) return
      try {
        const addAnimatedTitle = useEditorStore.getState().addAnimatedTitle as unknown as (asset: EditorAsset, options: {
          id: string; projectId: string; workspace: string; at: number; text: string; keepAudio: boolean
          background?: AnimatedTitleSettings['background']; backgroundColor?: string
        }) => boolean
        const added = addAnimatedTitle(intent.asset, {
          id: intent.id, projectId: intent.projectId, workspace: intent.workspace,
          at: intent.at, text: titleText(intent.settings), keepAudio: intent.settings.keepAudio,
          background: intent.settings.background, backgroundColor: intent.settings.backgroundColor,
        })
        if (added) {
          const titleBackgroundInfo = titleBackground(intent.settings)
          update({ ...intent, status: 'inserted', error: undefined, message: titleBackgroundInfo.chromaKey
            ? `Title added above your footage with a ${titleBackgroundInfo.label.toLowerCase()} key. Review spelling and key edges.`
            : `Title added above your footage with its ${titleBackgroundInfo.label.toLowerCase()} background intact. Review spelling and timing.` })
        }
      } catch (error) { set({ error: error instanceof Error ? error.message : 'Could not add the title to this project.' }) }
    },
    dismiss: () => { remember(null); set({ intent: null, error: null }) },
  }
})
