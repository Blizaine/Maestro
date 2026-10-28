import { create } from 'zustand'
import * as api from '../api/client'
import type { EditorAsset } from '../types'
import { animatedTitleParams, animatedTitlePrompt, titleDimensions, titleFrames, titleSubmissionId, TITLE_MODEL } from './editorAnimatedTitles'
import type { AnimatedTitleSettings } from './editorAnimatedTitles'
import { useEditorStore } from './useEditorStore'

const STORAGE = 'maestro-editor-animated-title-v1'
interface TitleIntent {
  id: string; clientId: string; projectId: string; workspace: string; at: number
  settings: AnimatedTitleSettings; width: number; height: number
  status: 'submitting' | 'uncertain' | 'queued' | 'running' | 'ready' | 'inserted' | 'failed'
  jobId?: string; asset?: EditorAsset; progress?: number; message?: string; error?: string
}
function restore(): TitleIntent | null {
  try {
    const value = JSON.parse(localStorage.getItem(STORAGE) || 'null') as TitleIntent | null
    if (!value || typeof value.id !== 'string' || typeof value.clientId !== 'string'
      || typeof value.projectId !== 'string' || typeof value.workspace !== 'string'
      || !Number.isFinite(value.at) || value.at < 0 || !value.settings) return null
    animatedTitlePrompt(value.settings); titleFrames(value.settings.seconds)
    if (!['submitting', 'uncertain', 'queued', 'running', 'ready', 'inserted', 'failed'].includes(value.status)) return null
    return value.status === 'submitting' ? { ...value, status: 'uncertain' } : value
  } catch { return null }
}
function remember(intent: TitleIntent | null) {
  // If durable intent cannot be saved, do not submit an unrecoverable GPU job.
  if (intent) localStorage.setItem(STORAGE, JSON.stringify(intent))
  else localStorage.removeItem(STORAGE)
}
async function greenFrame(width: number, height: number): Promise<File> {
  const canvas = document.createElement('canvas')
  canvas.width = width; canvas.height = height
  const context = canvas.getContext('2d')
  if (!context) throw new Error('Could not prepare the title background.')
  context.fillStyle = '#00ff00'; context.fillRect(0, 0, width, height)
  const blob = await new Promise<Blob>((resolve, reject) => canvas.toBlob(value => value
    ? resolve(value) : reject(new Error('Could not encode the title background.')), 'image/png'))
  return new File([blob], `title-green-${width}x${height}.png`, { type: 'image/png' })
}
let submitting = false, reconciling = false

interface TitleState {
  intent: TitleIntent | null; busy: boolean; error: string | null
  submit: (settings: AnimatedTitleSettings) => Promise<void>
  reconcile: () => Promise<void>
  insertReady: () => Promise<void>
  dismiss: () => void
}
export const useEditorTitleStore = create<TitleState>((set, get) => {
  const update = (intent: TitleIntent) => {
    try { remember(intent); set({ intent }) }
    catch { set({ intent, error: 'Browser storage is full. Keep this page open to track the existing title job.' }) }
  }
  return {
    intent: restore(), busy: false, error: null,
    submit: async settings => {
      if (submitting || get().busy) return
      const prior = get().intent
      if (prior && !['inserted', 'failed'].includes(prior.status)) {
        set({ error: 'Finish or stop tracking the current title before generating another.' }); return
      }
      submitting = true; set({ busy: true, error: null })
      let intent: TitleIntent | null = null, posted = false
      try {
        const current = useEditorStore.getState().project
        if (!current) throw new Error('Open an Editor project first.')
        const canvas = { ...current.canvas }
        const settingsCopy = { ...settings, text: settings.text.trim() }
        animatedTitlePrompt(settingsCopy); titleFrames(settingsCopy.seconds)
        const { width, height } = titleDimensions(settingsCopy, current.canvas)
        const at = useEditorStore.getState().playhead
        if (at + titleFrames(settingsCopy.seconds) / 24 > 86400) throw new Error('The title would exceed the 24-hour timeline limit.')
        const { models } = await api.fetchModels()
        if (!models.some(model => model.model_type === TITLE_MODEL && model.is_downloaded === true)) {
          throw new Error('Install H3 Fused 4-Step — Frames in Settings before generating an animated title.')
        }
        const saved = await useEditorStore.getState().saveProject()
        if (!saved || saved.id !== current.id || saved.workspace !== current.workspace
          || useEditorStore.getState().project?.id !== current.id) throw new Error('The Editor project changed while preparing the title.')
        const background = await api.uploadImage(await greenFrame(width, height), { reuseIdentical: true })
        if (useEditorStore.getState().project?.id !== current.id || useEditorStore.getState().project?.workspace !== current.workspace) {
          throw new Error('The Editor project changed before submission. No title was generated.')
        }
        intent = { id: titleSubmissionId(), clientId: titleSubmissionId(), projectId: saved.id,
          workspace: saved.workspace, at, settings: settingsCopy, width, height, status: 'submitting' }
        remember(intent); set({ intent }) // Durable before the one and only generation POST.
        const params = animatedTitleParams(settingsCopy, canvas, background.path, saved.workspace, intent.clientId)
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
    },
    reconcile: async () => {
      const intent = get().intent
      if (!intent || submitting || reconciling || ['inserted', 'failed'].includes(intent.status)) return
      if (intent.status === 'ready') {
        const project = useEditorStore.getState().project
        if (!intent.error && project?.id === intent.projectId && project.workspace === intent.workspace) await get().insertReady()
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
        if (job.status === 'failed' || job.status === 'cancelled') {
          update({ ...intent, jobId, status: 'failed', error: job.error || job.message || 'Title generation did not complete.' }); return
        }
        if (job.status !== 'completed') {
          update({ ...intent, jobId, status: job.status === 'running' ? 'running' : 'queued',
            progress: job.progress, message: job.phase || job.message || 'Title queued', error: undefined }); return
        }
        const filename = job.output_files.filter(file => /\.(mp4|mov|webm|mkv)$/i.test(file)).at(-1)
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
        const sizeMatches = probe.width === intent.width && probe.height === intent.height
        update({ ...intent, jobId, asset, status: 'ready', progress: 100,
          error: sizeMatches ? undefined : `The video is ${probe.width}×${probe.height}; ${intent.width}×${intent.height} was requested. Review it before adding.`,
          message: 'Title ready' })
        if (sizeMatches) await get().insertReady()
      } catch (error) {
        if (get().intent?.id === intent.id) update({ ...get().intent!, status: 'uncertain',
          error: error instanceof Error ? error.message : 'Could not check this title. Check again to recover its receipt.' })
      } finally { reconciling = false }
    },
    insertReady: async () => {
      const intent = get().intent
      if (!intent?.asset || !['ready', 'inserted'].includes(intent.status)) return
      try {
        const added = useEditorStore.getState().addAnimatedTitle(intent.asset, {
          id: intent.id, projectId: intent.projectId, workspace: intent.workspace,
          at: intent.at, text: intent.settings.text, keepAudio: intent.settings.keepAudio,
        })
        if (added) update({ ...intent, status: 'inserted', error: undefined, message: 'Title added above your footage. Review spelling and key edges.' })
      } catch (error) { set({ error: error instanceof Error ? error.message : 'Could not add the title to this project.' }) }
    },
    dismiss: () => { remember(null); set({ intent: null, error: null }) },
  }
})
