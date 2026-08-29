import { create } from 'zustand'
import * as api from '../api/client'
import { useStore } from '../stores/useStore'
import type {
  EditorAIReturnMode,
  EditorAIRoundTripTool,
  EditorAsset,
  EditorCanvas,
  EditorExportRecord,
  EditorExportSettings,
  EditorMarker,
  EditorMediaProbe,
  EditorProject,
  EditorProjectSummary,
  EditorTimelineItem,
  EditorTrack,
  EditorTrackType,
  GenerationJob,
  PipelineClipState,
  PipelineListItem,
  SavedPipelineState,
} from '../types'
import {
  canPlaceEditorItem,
  closestAvailableEditorStart,
  editorExportDimensions,
  editorExportFps,
  nextEditorItemStart,
  previousEditorItemEnd,
} from './editorUtils'
import { DEFAULT_EDITOR_FONT } from './editorFonts'
import { invalidateEditorMediaPreview } from './editorMediaPreview'

const HISTORY_LIMIT = 60
const MIN_ITEM_DURATION = 1 / 30
const ROUND_TRIP_STORAGE_KEY = 'maestro-editor-ai-round-trip-v1'
const DIRECTOR_VIDEO_EXTENSIONS = ['.mp4', '.webm', '.mkv', '.mov']
let initializeSequence = 0

export type EditorMobilePanel = 'media' | 'timeline' | 'inspector' | 'projects'

interface EditorClipboard {
  entries: Array<{
    item: EditorTimelineItem
    trackType: EditorTrackType
    trackId: string
    asset?: EditorAsset
  }>
  anchorStart: number
}

export type EditorAIRoundTripStatus = 'armed' | 'queued' | 'running' | 'completed' | 'failed'

export interface EditorAIRoundTrip {
  id: string
  projectId: string
  workspace: string
  itemId: string
  itemName: string
  tool: EditorAIRoundTripTool
  returnMode: EditorAIReturnMode
  status: EditorAIRoundTripStatus
  baselineJobIds: string[]
  claimedJobId?: string
  outputFile?: string
  error?: string
  createdAt: number
}

interface EditorState {
  workspace: string
  project: EditorProject | null
  projects: EditorProjectSummary[]
  library: EditorAsset[]
  libraryWorkspaces: string[]
  directorRuns: PipelineListItem[]
  directorImportingId: string | null
  directorRerunItemId: string | null
  loading: boolean
  saving: boolean
  dirty: boolean
  error: string | null
  selectedItemId: string | null
  selectedItemIds: string[]
  selectedTrackId: string | null
  playhead: number
  playing: boolean
  pixelsPerSecond: number
  snapping: boolean
  ripple: boolean
  history: EditorProject[]
  future: EditorProject[]
  clipboard: EditorClipboard | null
  roundTrip: EditorAIRoundTrip | null
  mobilePanel: EditorMobilePanel
  exportJobId: string | null
  exportProgress: number

  initialize: (workspace: string) => Promise<void>
  refreshProjects: () => Promise<void>
  refreshLibrary: () => Promise<void>
  createProject: (name?: string, canvas?: Partial<EditorCanvas>) => Promise<void>
  duplicateProject: () => Promise<void>
  loadProject: (projectId: string) => Promise<void>
  auditProjectMedia: () => Promise<void>
  saveProject: () => Promise<EditorProject | null>
  deleteProject: (projectId: string) => Promise<void>
  renameProject: (name: string) => void
  setCanvas: (patch: Partial<EditorCanvas>) => void
  setExportSettings: (patch: Partial<EditorExportSettings>) => void

  uploadMedia: (file: File, trackId?: string) => Promise<void>
  relinkMedia: (assetId: string, file: File) => Promise<void>
  addMedia: (asset: EditorAsset, at?: number, trackId?: string) => Promise<void>
  importDirectorRun: (pipelineId: string, at?: number) => Promise<void>
  rerunDirectorClip: (itemId: string, prompt?: string) => Promise<void>
  addTitle: (at?: number, trackId?: string) => void
  addTrack: (type: EditorTrackType) => void
  renameTrack: (trackId: string, name: string) => void
  setTrackVolume: (trackId: string, volume: number) => void
  setTrackZIndex: (trackId: string, zIndex: number) => void
  toggleTrackMute: (trackId: string) => void
  toggleTrackLock: (trackId: string) => void
  removeTrack: (trackId: string) => void

  selectItem: (itemId: string | null, trackId?: string | null, additive?: boolean) => void
  selectItems: (itemIds: string[], primaryItemId?: string | null, trackId?: string | null) => void
  setPlayhead: (seconds: number) => void
  setPlaying: (playing: boolean) => void
  setPixelsPerSecond: (value: number) => void
  setSnapping: (enabled: boolean) => void
  setRipple: (enabled: boolean) => void
  setMobilePanel: (panel: EditorMobilePanel) => void
  updateItem: (itemId: string, patch: Partial<EditorTimelineItem>) => void
  moveItem: (itemId: string, start: number, trackId?: string) => void
  trimItem: (itemId: string, edge: 'start' | 'end', time: number) => void
  splitSelected: () => void
  duplicateSelected: () => void
  copySelected: () => void
  cutSelected: () => void
  pasteClipboard: () => void
  detachSelectedAudio: () => void
  linkSelected: () => void
  unlinkSelected: () => void
  setActiveTake: (itemId: string, assetId: string) => void
  addMarker: (time?: number) => void
  updateMarker: (markerId: string, patch: Partial<EditorMarker>) => void
  removeMarker: (markerId: string) => void
  beginAIRoundTrip: (tool: EditorAIRoundTripTool, returnMode: EditorAIReturnMode) => Promise<void>
  cancelAIRoundTrip: () => void
  completeAIRoundTrip: (outputFile: string) => Promise<void>
  deleteSelected: () => void
  jumpToEdit: (direction: -1 | 1) => void
  undo: () => void
  redo: () => void
  exportProject: (mode?: 'now' | 'queue') => Promise<void>
}

function cloneProject(project: EditorProject): EditorProject {
  return typeof structuredClone === 'function'
    ? structuredClone(project)
    : JSON.parse(JSON.stringify(project)) as EditorProject
}

function cloneTimelineItem(
  item: EditorTimelineItem,
  patch: Partial<EditorTimelineItem> = {},
): EditorTimelineItem {
  const hasStylePatch = Object.prototype.hasOwnProperty.call(patch, 'style')
  const takeIds = patch.take_asset_ids ?? item.take_asset_ids
  const takeStates = patch.take_states ?? item.take_states
  const aiHistory = patch.ai_history ?? item.ai_history
  const director = patch.director ?? item.director
  return {
    ...item,
    ...patch,
    transform: { ...item.transform, ...(patch.transform || {}) },
    style: hasStylePatch
      ? patch.style ? { ...patch.style } : undefined
      : item.style
        ? { ...item.style }
        : undefined,
    take_asset_ids: takeIds ? [...takeIds] : undefined,
    take_states: takeStates
      ? Object.fromEntries(Object.entries(takeStates).map(([assetId, state]) => [assetId, { ...state }]))
      : undefined,
    ai_history: aiHistory ? aiHistory.map(entry => ({ ...entry })) : undefined,
    director: director
      ? {
          ...director,
          window_prompts: director.window_prompts ? [...director.window_prompts] : undefined,
        }
      : undefined,
  }
}

function newId(prefix: string): string {
  const random = typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function'
    ? crypto.randomUUID().replace(/-/g, '').slice(0, 16)
    : `${Date.now().toString(36)}${Math.random().toString(36).slice(2, 10)}`
  return `${prefix}-${random}`
}

function nextProjectVersionName(name: string, projects: EditorProjectSummary[]): string {
  const parsed = name.match(/^(.*?)(?:\s+v(\d+))$/i)
  const baseName = (parsed?.[1] || name).trim() || 'Untitled edit'
  let highest = parsed ? Number(parsed[2]) : 1
  projects.forEach(project => {
    const candidate = project.name.match(/^(.*?)(?:\s+v(\d+))$/i)
    if (!candidate || candidate[1].trim().toLocaleLowerCase() !== baseName.toLocaleLowerCase()) return
    highest = Math.max(highest, Number(candidate[2]))
  })
  return `${baseName} v${highest + 1}`.slice(0, 120)
}

function projectDuration(project: EditorProject | null): number {
  if (!project) return 0
  return project.tracks.reduce((maximum, track) => (
    track.items.reduce((trackMaximum, item) => (
      item.disabled ? trackMaximum : Math.max(trackMaximum, item.start + item.duration)
    ), maximum)
  ), 0)
}

function itemLocation(project: EditorProject, itemId: string): {
  track: EditorTrack
  item: EditorTimelineItem
  trackIndex: number
  itemIndex: number
} | null {
  for (let trackIndex = 0; trackIndex < project.tracks.length; trackIndex += 1) {
    const track = project.tracks[trackIndex]
    const itemIndex = track.items.findIndex(item => item.id === itemId)
    if (itemIndex >= 0) return { track, item: track.items[itemIndex], trackIndex, itemIndex }
  }
  return null
}

function uniqueItemIds(values: Iterable<string>): string[] {
  return Array.from(new Set(Array.from(values).filter(Boolean)))
}

function linkedEditorItemIds(project: EditorProject, seedIds: Iterable<string>): string[] {
  const ids = new Set(seedIds)
  const groups = new Set<string>()
  project.tracks.forEach(track => track.items.forEach(item => {
    if (ids.has(item.id) && item.link_group_id) groups.add(item.link_group_id)
  }))
  if (groups.size > 0) {
    project.tracks.forEach(track => track.items.forEach(item => {
      if (item.link_group_id && groups.has(item.link_group_id)) ids.add(item.id)
    }))
  }
  return uniqueItemIds(ids)
}

function selectedEditorItemIds(state: Pick<EditorState, 'project' | 'selectedItemId' | 'selectedItemIds'>): string[] {
  if (!state.project) return []
  const seeds = state.selectedItemIds.length > 0
    ? state.selectedItemIds
    : state.selectedItemId
      ? [state.selectedItemId]
      : []
  return linkedEditorItemIds(state.project, seeds)
}

function persistRoundTrip(value: EditorAIRoundTrip | null): void {
  if (typeof window === 'undefined') return
  try {
    if (value) window.sessionStorage.setItem(ROUND_TRIP_STORAGE_KEY, JSON.stringify(value))
    else window.sessionStorage.removeItem(ROUND_TRIP_STORAGE_KEY)
  } catch { /* Session persistence is helpful, never required. */ }
}

function restoreRoundTrip(): EditorAIRoundTrip | null {
  if (typeof window === 'undefined') return null
  try {
    const raw = window.sessionStorage.getItem(ROUND_TRIP_STORAGE_KEY)
    if (!raw) return null
    const value = JSON.parse(raw) as EditorAIRoundTrip
    if (!value?.id || !value.projectId || !value.itemId || !value.tool) return null
    return value
  } catch {
    return null
  }
}

function clampItemToSource(project: EditorProject, item: EditorTimelineItem): void {
  const asset = item.asset_id ? project.assets[item.asset_id] : undefined
  if (!asset || asset.type === 'image' || asset.duration <= 0) return
  const speed = Math.max(0.1, item.speed || 1)
  const latestSourceIn = Math.max(0, asset.duration - MIN_ITEM_DURATION * speed)
  item.source_in = Math.max(0, Math.min(latestSourceIn, item.source_in || 0))
  const maximumDuration = Math.max(MIN_ITEM_DURATION, (asset.duration - item.source_in) / speed)
  item.duration = Math.max(MIN_ITEM_DURATION, Math.min(item.duration, maximumDuration))
}

function commitProject(
  set: (patch: Partial<EditorState> | ((state: EditorState) => Partial<EditorState>)) => void,
  get: () => EditorState,
  updater: (project: EditorProject) => void,
): void {
  const current = get().project
  if (!current) return
  const before = cloneProject(current)
  const next = cloneProject(current)
  updater(next)
  next.updated_at = Date.now() / 1000
  set(state => ({
    project: next,
    dirty: true,
    history: [...state.history, before].slice(-HISTORY_LIMIT),
    future: [],
    error: null,
  }))
}

function libraryAssetFromOutput(
  output: api.ApiOutput,
  origin: EditorAsset['origin'],
  workspace?: string,
): EditorAsset {
  const assetWorkspace = origin === 'upload'
    ? '__uploads__'
    : output.workspace || workspace
  return {
    id: newId('library'),
    name: output.name,
    type: output.type,
    origin,
    workspace: assetWorkspace,
    favorite: Boolean(output.favorite),
    url: origin === 'upload'
      ? api.getUploadUrl(output.name)
      : api.getFileUrl(output.name, assetWorkspace),
    duration: 0,
    width: 0,
    height: 0,
    fps: 0,
    has_audio: output.type === 'audio' || output.type === 'video',
    size: output.size,
    created_at: output.created_at,
  }
}

function pathLeaf(path: string): string {
  return path.replace(/\\/g, '/').split('/').filter(Boolean).at(-1) || path
}

function isDirectorVideoFilename(filename: unknown): filename is string {
  if (typeof filename !== 'string') return false
  const lower = filename.toLowerCase()
  return DIRECTOR_VIDEO_EXTENSIONS.some(extension => lower.endsWith(extension))
}

function plannedDirectorClipDuration(clip: PipelineClipState): number {
  const planned = clip.planned_clip
  const duration = Number(planned?.end || 0) - Number(planned?.start || 0)
  if (Number.isFinite(duration) && duration > 0) return duration
  const frameDuration = Number(planned?.duration_frames || 0)
  return Number.isFinite(frameDuration) && frameDuration > 0 ? frameDuration / 24 : 0
}

function directorRunLabel(state: SavedPipelineState): string {
  const description = (state.scene_description || '').replace(/\s+/g, ' ').trim()
  return (description || `Director ${state.pipeline_id}`).slice(0, 54)
}

function directorMasterAudio(
  state: SavedPipelineState,
  clipWorkspace: string,
): EditorAsset | null {
  if (state.pipeline_type !== 'music_video') return null
  const manifest = state.asset_manifest?.audio_path
  const entry = manifest && typeof manifest === 'object'
    ? manifest as Record<string, unknown>
    : null
  const snapshotPath = typeof state._params_snapshot?.audio_path === 'string'
    ? state._params_snapshot.audio_path
    : ''
  const path = typeof entry?.path === 'string' ? entry.path : snapshotPath
  if (!path) return null
  const servePath = typeof entry?.serve_path === 'string'
    ? entry.serve_path
    : pathLeaf(path)
  const normalizedPath = path.replace(/\\/g, '/').toLowerCase()
  const isUpload = !entry && normalizedPath.includes('/uploads/')
  const workspace = state.workspace || clipWorkspace
  return {
    id: newId('library'),
    name: pathLeaf(path),
    type: 'audio',
    origin: isUpload ? 'upload' : 'project',
    workspace: isUpload ? '__uploads__' : workspace,
    path,
    url: isUpload
      ? api.getUploadUrl(pathLeaf(path))
      : api.getFileUrl(servePath, workspace),
    duration: 0,
    width: 0,
    height: 0,
    fps: 0,
    has_audio: true,
  }
}

function bestTrack(
  project: EditorProject,
  type: EditorTrackType,
  preferredTrackId?: string,
): EditorTrack {
  const preferred = preferredTrackId
    ? project.tracks.find(track => (
        track.id === preferredTrackId && track.type === type && !track.locked
      ))
    : undefined
  if (preferred) return preferred
  const existing = project.tracks.find(track => track.type === type && !track.locked)
  if (existing) return existing
  const sameType = project.tracks.filter(candidate => candidate.type === type)
  const topLayer = sameType.reduce(
    (maximum, candidate) => Math.max(maximum, candidate.z_index),
    type === 'text' ? 9 : -1,
  )
  const track: EditorTrack = {
    id: newId(type),
    name: type === 'video' ? 'Video' : type === 'audio' ? 'Audio' : 'Titles',
    type,
    z_index: topLayer + 1,
    muted: false,
    locked: false,
    volume: 1,
    items: [],
  }
  project.tracks.push(track)
  return track
}

function mediaTypeFromProbe(probe: EditorMediaProbe): EditorAsset['type'] {
  return probe.type === 'image' ? 'image' : probe.type === 'audio' ? 'audio' : 'video'
}

function trackEditorJob(
  jobId: string,
  initialStatus: api.ApiJobStatus['status'],
  onProgress: (progress: number) => void,
  onFinish: (status: api.ApiJobStatus) => void,
): void {
  const editorJob: GenerationJob = {
    id: jobId,
    kind: 'editor_export',
    status: initialStatus,
    progress: 0,
    step: 0,
    totalSteps: 0,
    phase: initialStatus === 'held' ? 'Held' : 'Queued',
    message: initialStatus === 'held' ? 'Editor export held' : 'Editor export queued',
    outputFiles: [],
    error: null,
    oomInfo: null,
  }
  useStore.setState(state => ({
    jobs: [editorJob, ...state.jobs.filter(job => job.id !== jobId)],
    isGenerating: state.isGenerating || initialStatus !== 'held',
  }))
  const timer = window.setInterval(async () => {
    try {
      const status = await api.fetchJobStatus(jobId)
      const progress = Math.max(0, Math.min(1, status.progress / 100))
      onProgress(progress)
      useStore.setState(state => ({
        jobs: state.jobs.map(job => job.id === jobId ? {
          ...job,
          kind: status.kind || 'editor_export',
          status: status.status,
          progress,
          step: status.step,
          totalSteps: status.total_steps,
          phase: status.phase,
          message: status.message,
          outputFiles: status.output_files,
          error: status.error,
        } : job),
      }))
      if (status.status === 'completed' || status.status === 'failed' || status.status === 'cancelled') {
        window.clearInterval(timer)
        onFinish(status)
        window.setTimeout(() => {
          useStore.setState(state => {
            const jobs = state.jobs.filter(job => job.id !== jobId)
            return {
              jobs,
              isGenerating: jobs.some(job => job.status === 'queued' || job.status === 'running'),
            }
          })
        }, status.status === 'completed' ? 1200 : 5000)
        if (status.status === 'completed') void useStore.getState().loadOutputs()
      }
    } catch (error) {
      window.clearInterval(timer)
      const message = error instanceof Error ? error.message : 'Editor export disconnected'
      onFinish({
        job_id: jobId,
        kind: 'editor_export',
        status: 'failed',
        progress: 0,
        step: 0,
        total_steps: 0,
        phase: 'Disconnected',
        message,
        output_files: [],
        error: message,
      })
      useStore.setState(state => ({
        jobs: state.jobs.map(job => job.id === jobId ? {
          ...job,
          status: 'failed',
          phase: 'Disconnected',
          message,
          error: message,
        } : job),
      }))
      window.setTimeout(() => {
        useStore.setState(state => {
          const jobs = state.jobs.filter(job => job.id !== jobId)
          return {
            jobs,
            isGenerating: jobs.some(job => job.status === 'queued' || job.status === 'running'),
          }
        })
      }, 5000)
    }
  }, 1500)
}

export const useEditorStore = create<EditorState>((set, get) => ({
  workspace: 'default',
  project: null,
  projects: [],
  library: [],
  libraryWorkspaces: [],
  directorRuns: [],
  directorImportingId: null,
  directorRerunItemId: null,
  loading: false,
  saving: false,
  dirty: false,
  error: null,
  selectedItemId: null,
  selectedItemIds: [],
  selectedTrackId: null,
  playhead: 0,
  playing: false,
  pixelsPerSecond: 54,
  snapping: true,
  ripple: false,
  history: [],
  future: [],
  clipboard: null,
  roundTrip: restoreRoundTrip(),
  mobilePanel: 'timeline',
  exportJobId: null,
  exportProgress: 0,

  initialize: async (workspace) => {
    const requestSequence = ++initializeSequence
    const existing = get().project
    if (existing && existing.workspace !== workspace && get().dirty) {
      await get().saveProject()
    }
    if (requestSequence !== initializeSequence) return
    set({ loading: true, workspace, error: null, playing: false })
    try {
      const { projects } = await api.fetchEditorProjects(workspace)
      if (requestSequence !== initializeSequence) return
      let project: EditorProject
      const current = get().project
      if (current?.workspace === workspace) {
        project = current
      } else if (projects[0]) {
        project = await api.fetchEditorProject(projects[0].id, workspace)
      } else {
        project = await api.createEditorProject({ workspace, name: 'Untitled edit' })
      }
      if (requestSequence !== initializeSequence) return
      set({
        project,
        projects: projects.some(summary => summary.id === project.id)
          ? projects
          : [{
              id: project.id,
              name: project.name,
              workspace: project.workspace,
              created_at: project.created_at,
              updated_at: project.updated_at,
              duration: projectDuration(project),
              asset_count: Object.keys(project.assets).length,
            }, ...projects],
        loading: false,
        dirty: false,
        history: [],
        future: [],
        clipboard: null,
        selectedItemId: null,
        selectedItemIds: [],
        selectedTrackId: null,
        playhead: 0,
      })
      await get().auditProjectMedia()
      await get().refreshLibrary()
    } catch (error) {
      if (requestSequence === initializeSequence) {
        set({ loading: false, error: error instanceof Error ? error.message : 'Editor failed to open' })
      }
    }
  },

  refreshProjects: async () => {
    try {
      const { projects } = await api.fetchEditorProjects(get().workspace)
      set({ projects })
    } catch (error) {
      set({ error: error instanceof Error ? error.message : 'Unable to load Editor projects' })
    }
  },

  refreshLibrary: async () => {
    try {
      const workspaceResponse = await api.fetchWorkspaces()
      const workspaceNames = Array.from(new Set([
        get().workspace,
        ...workspaceResponse.workspaces.map(workspace => workspace.name),
      ]))
      const [workspaceResults, uploads, directorResponse] = await Promise.all([
        Promise.all(workspaceNames.map(async workspace => ({
          workspace,
          result: await api.fetchOutputs(0, 0, { workspace }),
        }))),
        api.fetchOutputs(0, 0, { workspace: '__uploads__' }),
        api.fetchPipelineList().catch(() => ({ pipelines: [] })),
      ])
      const seen = new Set<string>()
      const library = [
        ...workspaceResults.flatMap(({ workspace, result }) => (
          result.outputs.map(output => libraryAssetFromOutput(output, 'output', workspace))
        )),
        ...uploads.outputs.map(output => libraryAssetFromOutput(output, 'upload', '__uploads__')),
      ].filter(asset => {
        const key = `${asset.workspace || asset.origin}:${asset.name}`
        if (seen.has(key)) return false
        seen.add(key)
        return true
      }).sort((a, b) => (b.created_at || 0) - (a.created_at || 0))
      set({
        library,
        libraryWorkspaces: workspaceNames,
        directorRuns: directorResponse.pipelines,
      })
    } catch (error) {
      set({ error: error instanceof Error ? error.message : 'Unable to load media' })
    }
  },

  createProject: async (name = 'Untitled edit', canvas) => {
    set({ loading: true, error: null, playing: false })
    try {
      const project = await api.createEditorProject({ workspace: get().workspace, name, canvas })
      set({
        project,
        loading: false,
        dirty: false,
        history: [],
        future: [],
        clipboard: null,
        selectedItemId: null,
        selectedItemIds: [],
        selectedTrackId: null,
        playhead: 0,
      })
      await get().auditProjectMedia()
      await get().refreshProjects()
    } catch (error) {
      set({ loading: false, error: error instanceof Error ? error.message : 'Unable to create project' })
    }
  },

  duplicateProject: async () => {
    const source = await get().saveProject()
    if (!source) return
    set({ loading: true, error: null, playing: false })
    try {
      const now = Date.now() / 1000
      const copy = cloneProject(source)
      copy.id = newId('project')
      copy.name = nextProjectVersionName(source.name, get().projects)
      copy.created_at = now
      copy.updated_at = now
      copy.exports = []
      const project = await api.createEditorProject({
        workspace: source.workspace,
        project: copy,
      })
      set({
        project,
        loading: false,
        dirty: false,
        history: [],
        future: [],
        clipboard: null,
        selectedItemId: null,
        selectedItemIds: [],
        selectedTrackId: null,
        playhead: 0,
      })
      await get().auditProjectMedia()
      await get().refreshProjects()
    } catch (error) {
      set({ loading: false, error: error instanceof Error ? error.message : 'Unable to create project version' })
    }
  },

  loadProject: async (projectId) => {
    if (get().dirty) await get().saveProject()
    set({ loading: true, error: null, playing: false })
    try {
      const project = await api.fetchEditorProject(projectId, get().workspace)
      set({
        project,
        loading: false,
        dirty: false,
        history: [],
        future: [],
        clipboard: null,
        selectedItemId: null,
        selectedItemIds: [],
        selectedTrackId: null,
        playhead: 0,
      })
      await get().auditProjectMedia()
    } catch (error) {
      set({ loading: false, error: error instanceof Error ? error.message : 'Unable to open project' })
    }
  },

  auditProjectMedia: async () => {
    const project = get().project
    if (!project || Object.keys(project.assets).length === 0) return
    try {
      const result = await api.fetchEditorMediaStatus(project)
      if (get().project?.id !== project.id) return
      const availability = new Map(result.assets.map(asset => [asset.asset_id, asset]))
      set(state => {
        if (!state.project || state.project.id !== project.id) return {}
        const assets = Object.fromEntries(Object.entries(state.project.assets).map(([assetId, asset]) => {
          const status = availability.get(assetId)
          return [assetId, {
            ...asset,
            missing: status ? !status.available : true,
            ...(status?.available && status.path ? { path: status.path } : {}),
          }]
        }))
        return { project: { ...state.project, assets } }
      })
    } catch {
      // Availability is a recovery aid. A temporary audit failure must not
      // prevent the project itself from opening.
    }
  },

  saveProject: async () => {
    const project = get().project
    if (!project || get().saving) return project
    set({ saving: true, error: null })
    try {
      const saved = await api.saveEditorProject(project)
      let needsFollowUpSave = false
      set(state => {
        if (state.project?.id !== saved.id) return { saving: false }
        if (state.project === project) {
          const assets = Object.fromEntries(Object.entries(saved.assets).map(([assetId, asset]) => [
            assetId,
            {
              ...asset,
              ...(project.assets[assetId]?.missing !== undefined
                ? { missing: project.assets[assetId].missing }
                : {}),
            },
          ]))
          return { project: { ...saved, assets }, saving: false, dirty: false }
        }
        // A newer immutable project snapshot was created while this request
        // was in flight. Keep it and immediately persist that newer state.
        needsFollowUpSave = true
        return { saving: false, dirty: true }
      })
      await get().refreshProjects()
      if (needsFollowUpSave) return await get().saveProject()
      return saved
    } catch (error) {
      set({ saving: false, error: error instanceof Error ? error.message : 'Unable to save project' })
      return null
    }
  },

  deleteProject: async (projectId) => {
    try {
      await api.deleteEditorProject(projectId, get().workspace)
      const remaining = get().projects.filter(project => project.id !== projectId)
      if (get().project?.id === projectId) {
        // Clear the deleted document before opening its replacement. loadProject
        // normally protects unsaved work, but that would recreate this project.
        set({
          project: null,
          dirty: false,
          history: [],
          future: [],
          clipboard: null,
          selectedItemId: null,
          selectedItemIds: [],
          selectedTrackId: null,
          playhead: 0,
          playing: false,
        })
        if (remaining[0]) await get().loadProject(remaining[0].id)
        else await get().createProject()
      }
      await get().refreshProjects()
    } catch (error) {
      set({ error: error instanceof Error ? error.message : 'Unable to delete project' })
    }
  },

  renameProject: name => commitProject(set, get, project => {
    project.name = name.trimStart().slice(0, 120) || 'Untitled edit'
  }),
  setCanvas: patch => commitProject(set, get, project => {
    project.canvas = { ...project.canvas, ...patch }
  }),
  setExportSettings: patch => commitProject(set, get, project => {
    project.export = { ...project.export, ...patch }
  }),

  uploadMedia: async (file, preferredTrackId) => {
    set({ error: null })
    try {
      const uploaded = await api.uploadImage(file)
      const type: EditorAsset['type'] = file.type.startsWith('audio/')
        ? 'audio'
        : file.type.startsWith('image/')
          ? 'image'
          : 'video'
      const asset: EditorAsset = {
        id: newId('library'),
        name: uploaded.filename,
        type,
        origin: 'upload',
        workspace: '__uploads__',
        favorite: false,
        path: uploaded.path,
        url: uploaded.url || api.getUploadUrl(uploaded.filename),
        duration: uploaded.duration_seconds || 0,
        width: 0,
        height: 0,
        fps: uploaded.fps || 0,
        has_audio: uploaded.has_audio ?? type !== 'image',
      }
      set(state => ({ library: [asset, ...state.library.filter(item => item.name !== asset.name || item.origin !== 'upload')] }))
      await get().addMedia(asset, undefined, preferredTrackId)
    } catch (error) {
      set({ error: error instanceof Error ? error.message : 'Media upload failed' })
    }
  },

  relinkMedia: async (assetId, file) => {
    const project = get().project
    const original = project?.assets[assetId]
    if (!project || !original) return
    set({ error: null })
    try {
      const uploaded = await api.uploadImage(file)
      const candidate: EditorAsset = {
        ...original,
        id: assetId,
        name: uploaded.filename,
        origin: 'upload',
        workspace: '__uploads__',
        path: uploaded.path,
        url: uploaded.url || api.getUploadUrl(uploaded.filename),
        duration: uploaded.duration_seconds || original.duration || 0,
        fps: uploaded.fps || original.fps || 0,
        has_audio: uploaded.has_audio ?? original.has_audio,
        missing: false,
      }
      const probe = await api.probeEditorMedia(candidate, project.workspace)
      if (probe.type !== original.type) {
        throw new Error(`Choose another ${original.type} file to relink “${original.name}”.`)
      }
      const replacement: EditorAsset = {
        ...candidate,
        type: probe.type,
        path: probe.path,
        duration: probe.duration,
        width: probe.width,
        height: probe.height,
        fps: probe.fps,
        has_audio: probe.has_audio,
        size: probe.size,
        missing: false,
      }
      commitProject(set, get, next => {
        next.assets[assetId] = replacement
        next.tracks.forEach(track => track.items.forEach(item => {
          if (item.asset_id !== assetId || replacement.type === 'image') return
          const minimum = 1 / Math.max(1, next.canvas.fps)
          item.source_in = Math.min(
            item.source_in,
            Math.max(0, replacement.duration - minimum * item.speed),
          )
          item.duration = Math.max(
            minimum,
            Math.min(item.duration, (replacement.duration - item.source_in) / item.speed),
          )
        }))
      })
      invalidateEditorMediaPreview(assetId)
      set(state => ({
        library: [replacement, ...state.library.filter(asset => (
          asset.name !== replacement.name || asset.origin !== replacement.origin
        ))],
      }))
    } catch (error) {
      set({ error: error instanceof Error ? error.message : 'Media relink failed' })
    }
  },

  addMedia: async (libraryAsset, at, preferredTrackId) => {
    const project = get().project
    if (!project) return
    const destinationProjectId = project.id
    set({ error: null })
    try {
      const probe = await api.probeEditorMedia(libraryAsset, project.workspace)
      if (get().project?.id !== destinationProjectId) return
      let asset = Object.values(project.assets).find(candidate => (
        candidate.origin === libraryAsset.origin && candidate.name === libraryAsset.name
      ))
      const assetId = asset?.id || newId('asset')
      asset = {
        ...libraryAsset,
        id: assetId,
        type: mediaTypeFromProbe(probe),
        path: probe.path,
        duration: probe.duration,
        width: probe.width,
        height: probe.height,
        fps: probe.fps,
        has_audio: probe.has_audio,
        size: probe.size,
      }
      const requestedStart = Math.max(0, at ?? get().playhead)
      let placedStart = requestedStart
      const itemId = newId('clip')
      const finalAsset = asset
      commitProject(set, get, next => {
        next.assets[assetId] = finalAsset
        const trackType: EditorTrackType = finalAsset.type === 'audio' ? 'audio' : 'video'
        const track = bestTrack(next, trackType, preferredTrackId)
        const duration = finalAsset.type === 'image'
          ? 5
          : Math.max(MIN_ITEM_DURATION, finalAsset.duration || 5)
        placedStart = closestAvailableEditorStart(
          track,
          requestedStart,
          duration,
          undefined,
          true,
        )
        track.items.push({
          id: itemId,
          asset_id: assetId,
          name: finalAsset.name,
          start: placedStart,
          duration,
          source_in: 0,
          speed: 1,
          volume: 1,
          opacity: 1,
          fit: 'contain',
          transform: { x: 0, y: 0, scale: 1, rotation: 0 },
          fade_in: 0,
          fade_out: 0,
        })
        track.items.sort((a, b) => a.start - b.start)
      })
      const selectedTrack = get().project?.tracks.find(track => track.items.some(item => item.id === itemId))
      set({ selectedItemId: itemId, selectedItemIds: [itemId], selectedTrackId: selectedTrack?.id || null, playhead: placedStart })
    } catch (error) {
      set({ error: error instanceof Error ? error.message : 'Unable to add media' })
    }
  },

  importDirectorRun: async (pipelineId, at) => {
    const currentProject = get().project
    if (!currentProject || get().directorImportingId) return
    const destinationProjectId = currentProject.id
    const summary = get().directorRuns.find(run => run.id === pipelineId)
    set({ directorImportingId: pipelineId, error: null, playing: false })
    try {
      const state = await api.fetchSavedPipeline(pipelineId)
      if (get().project?.id !== destinationProjectId) {
        throw new Error('The active Editor project changed before the Director import finished.')
      }
      const clips = (state.clips || []).filter(clip => clip && Number.isFinite(clip.index))
      if (clips.length === 0) throw new Error('This Director run has no planned shots to import.')
      const clipWorkspace = summary?.workspace || state.workspace || currentProject.workspace
      const runLabel = directorRunLabel(state)
      const outputVideos = (state.output_files || []).filter(isDirectorVideoFilename)
      const combinedFilename = outputVideos.find(filename => (
        filename.toLowerCase().includes('_multiclip')
      )) || outputVideos.at(-1) || ''
      const hasEveryIndividualClip = !state.seamless && clips.every(clip => (
        isDirectorVideoFilename(clip.video_filename)
      ))

      type PreparedDirectorClip = {
        asset: EditorAsset
        clip: PipelineClipState
        sourceIn: number
        duration: number
      }
      const prepared: PreparedDirectorClip[] = []
      const importedAssets: EditorAsset[] = []

      const probeVideo = async (filename: string): Promise<EditorAsset> => {
        const libraryAsset: EditorAsset = {
          id: newId('library'),
          name: pathLeaf(filename),
          type: 'video',
          origin: 'output',
          workspace: clipWorkspace,
          url: api.getFileUrl(filename, clipWorkspace),
          duration: 0,
          width: 0,
          height: 0,
          fps: 0,
          has_audio: true,
        }
        const probe = await api.probeEditorMedia(libraryAsset, currentProject.workspace)
        return {
          ...libraryAsset,
          id: newId('asset'),
          path: probe.path,
          duration: probe.duration,
          width: probe.width,
          height: probe.height,
          fps: probe.fps,
          has_audio: probe.has_audio,
          size: probe.size,
        }
      }

      if (hasEveryIndividualClip) {
        try {
          for (const clip of clips) {
            const asset = await probeVideo(clip.video_filename as string)
            importedAssets.push(asset)
            prepared.push({
              asset,
              clip,
              sourceIn: 0,
              duration: Math.max(MIN_ITEM_DURATION, asset.duration || plannedDirectorClipDuration(clip) || 5),
            })
          }
        } catch {
          // Legacy projects may retain per-shot names after their workspace
          // was renamed. Prefer the still-valid joined master below.
          prepared.length = 0
          importedAssets.length = 0
        }
      }
      if (prepared.length === 0 && combinedFilename) {
        try {
          // Seamless Director runs own one cumulative render. Reusing that
          // source with separate source-in ranges keeps every shot editable
          // without transcoding or duplicating the master file.
          const asset = await probeVideo(combinedFilename)
          importedAssets.push(asset)
          const fallbackDuration = Math.max(MIN_ITEM_DURATION, asset.duration / clips.length)
          const plannedDurations = clips.map(clip => (
            plannedDirectorClipDuration(clip) || fallbackDuration
          ))
          const plannedTotal = plannedDurations.reduce((total, duration) => total + duration, 0)
          const durationScale = asset.duration > 0 && plannedTotal > 0
            ? asset.duration / plannedTotal
            : 1
          let sourceCursor = 0
          clips.forEach((clip, index) => {
            const remaining = Math.max(0, asset.duration - sourceCursor)
            const requested = plannedDurations[index] * durationScale
            const duration = index === clips.length - 1
              ? Math.max(MIN_ITEM_DURATION, remaining || requested)
              : Math.max(MIN_ITEM_DURATION, Math.min(requested, remaining || requested))
            prepared.push({ asset, clip, sourceIn: sourceCursor, duration })
            sourceCursor += duration
          })
        } catch { /* Fall through to any surviving per-shot files. */ }
      }
      if (prepared.length === 0) {
        // A cancelled run may still have a useful completed prefix.
        for (const clip of clips) {
          if (!isDirectorVideoFilename(clip.video_filename)) continue
          try {
            const asset = await probeVideo(clip.video_filename)
            importedAssets.push(asset)
            prepared.push({
              asset,
              clip,
              sourceIn: 0,
              duration: Math.max(MIN_ITEM_DURATION, asset.duration || plannedDirectorClipDuration(clip) || 5),
            })
          } catch { /* Keep every completed partial shot that still exists. */ }
        }
      }
      if (prepared.length === 0) {
        throw new Error('This Director run has no completed video media to import yet.')
      }

      let masterAudio: EditorAsset | null = null
      let audioWarning = ''
      const audioCandidate = directorMasterAudio(state, clipWorkspace)
      if (audioCandidate) {
        try {
          const probe = await api.probeEditorMedia(audioCandidate, currentProject.workspace)
          masterAudio = {
            ...audioCandidate,
            id: newId('asset'),
            type: 'audio',
            path: probe.path,
            duration: probe.duration,
            width: 0,
            height: 0,
            fps: 0,
            has_audio: true,
            size: probe.size,
          }
          importedAssets.push(masterAudio)
        } catch {
          audioWarning = 'The Director shots were imported, but their original master soundtrack could not be found.'
        }
      } else if (state.pipeline_type === 'music_video') {
        audioWarning = 'The Director shots were imported, but this older music-video run has no saved master soundtrack.'
      }
      if (!masterAudio && state.pipeline_type === 'music_video' && combinedFilename) {
        // Older projects can lose the separately copied source song after a
        // workspace move while their final joined video remains intact. Its
        // embedded soundtrack is the exact full mix and is safe to expose as
        // an audio-only Editor asset.
        try {
          const joinedVideo = importedAssets.find(asset => (
            asset.type === 'video' && asset.name === pathLeaf(combinedFilename)
          )) || await probeVideo(combinedFilename)
          if (joinedVideo.has_audio) {
            masterAudio = {
              ...joinedVideo,
              id: newId('asset'),
              type: 'audio',
              has_audio: true,
            }
            importedAssets.push(masterAudio)
            audioWarning = ''
          }
        } catch { /* Keep the explicit missing-master warning above. */ }
      }

      const baseStart = Math.max(0, at ?? get().playhead)
      const videoTrackId = newId('video')
      const audioTrackId = masterAudio ? newId('audio') : null
      const firstItemId = newId('clip')
      let timelineCursor = baseStart
      commitProject(set, get, project => {
        importedAssets.forEach(asset => { project.assets[asset.id] = asset })
        const topVideoLayer = project.tracks
          .filter(track => track.type === 'video')
          .reduce((maximum, track) => Math.max(maximum, track.z_index), -1)
        const videoTrack: EditorTrack = {
          id: videoTrackId,
          name: `Director · ${runLabel}`.slice(0, 80),
          type: 'video',
          z_index: topVideoLayer + 1,
          muted: false,
          locked: false,
          volume: 1,
          items: [],
        }
        timelineCursor = baseStart
        prepared.forEach((entry, position) => {
          const prompt = entry.clip.window_prompts?.length
            ? entry.clip.window_prompts.join('\n\n')
            : entry.clip.video_prompt || ''
          videoTrack.items.push({
            id: position === 0 ? firstItemId : newId('clip'),
            asset_id: entry.asset.id,
            name: `Shot ${entry.clip.index + 1} · ${runLabel}`.slice(0, 140),
            start: timelineCursor,
            duration: entry.duration,
            source_in: entry.sourceIn,
            speed: 1,
            volume: 1,
            opacity: 1,
            fit: 'contain',
            transform: { x: 0, y: 0, scale: 1, rotation: 0 },
            muted: Boolean(masterAudio),
            fade_in: 0,
            fade_out: 0,
            take_asset_ids: [entry.asset.id],
            take_states: {
              [entry.asset.id]: { source_in: entry.sourceIn, speed: 1 },
            },
            director: {
              pipeline_id: state.pipeline_id,
              clip_index: entry.clip.index,
              pipeline_type: state.pipeline_type,
              workspace: clipWorkspace,
              video_prompt: prompt,
              window_prompts: entry.clip.window_prompts?.length
                ? [...entry.clip.window_prompts]
                : undefined,
            },
          })
          timelineCursor += entry.duration
        })
        project.tracks.push(videoTrack)

        if (masterAudio && audioTrackId) {
          project.tracks.push({
            id: audioTrackId,
            name: `Director music · ${runLabel}`.slice(0, 80),
            type: 'audio',
            z_index: 0,
            muted: false,
            locked: false,
            volume: 1,
            items: [{
              id: newId('audio-clip'),
              asset_id: masterAudio.id,
              name: masterAudio.name,
              start: baseStart,
              duration: Math.max(MIN_ITEM_DURATION, masterAudio.duration),
              source_in: 0,
              speed: 1,
              volume: 1,
              opacity: 1,
              fit: 'contain',
              transform: { x: 0, y: 0, scale: 1, rotation: 0 },
              fade_in: 0,
              fade_out: 0,
            }],
          })
        }
      })
      set(state => ({
        selectedItemId: firstItemId,
        selectedItemIds: [firstItemId],
        selectedTrackId: videoTrackId,
        playhead: baseStart,
        mobilePanel: 'timeline',
        library: [
          ...importedAssets,
          ...state.library.filter(asset => !importedAssets.some(imported => (
            imported.name === asset.name && imported.workspace === asset.workspace
          ))),
        ],
        error: audioWarning || (
          prepared.length < clips.length
            ? `Imported ${prepared.length} of ${clips.length} completed Director shots.`
            : null
        ),
      }))
    } catch (error) {
      set({ error: error instanceof Error ? error.message : 'Director run could not be imported.' })
    } finally {
      set({ directorImportingId: null })
    }
  },

  rerunDirectorClip: async (itemId, prompt) => {
    if (get().directorRerunItemId) return
    const project = get().project
    const located = project ? itemLocation(project, itemId) : null
    const provenance = located?.item.director
    if (!project || !located || located.track.type !== 'video' || !provenance) {
      set({ error: 'Select a Director shot in the timeline before rerunning it.' })
      return
    }
    const saved = await get().saveProject()
    if (!saved) return
    const requestedPrompt = prompt?.trim() || undefined
    set({ directorRerunItemId: itemId, error: null, playing: false })
    try {
      const result = await api.rerunClipVideo(
        provenance.pipeline_id,
        provenance.clip_index,
        requestedPrompt,
      )
      const outputName = pathLeaf(result.filename)
      const outputResponse = await api.fetchOutputs(0, 0, { workspace: provenance.workspace })
      const output = outputResponse.outputs.find(candidate => (
        candidate.name === outputName || candidate.name === result.filename
      ))
      const libraryAsset: EditorAsset = output
        ? libraryAssetFromOutput(output, 'output', provenance.workspace)
        : {
            id: newId('library'),
            name: outputName,
            type: 'video',
            origin: 'output',
            workspace: provenance.workspace,
            path: result.filename,
            url: api.getFileUrl(outputName, provenance.workspace),
            duration: 0,
            width: 0,
            height: 0,
            fps: 0,
            has_audio: true,
          }
      const probe = await api.probeEditorMedia(libraryAsset, saved.workspace)
      const generatedAsset: EditorAsset = {
        ...libraryAsset,
        id: newId('asset'),
        type: 'video',
        path: probe.path,
        duration: probe.duration,
        width: probe.width,
        height: probe.height,
        fps: probe.fps,
        has_audio: probe.has_audio,
        size: probe.size,
      }
      const active = get().project
      const sourceProject = active?.id === saved.id
        ? active
        : await api.fetchEditorProject(saved.id, saved.workspace)
      const before = cloneProject(sourceProject)
      const next = cloneProject(sourceProject)
      const target = itemLocation(next, itemId)
      if (!target?.item.director) throw new Error('The originating Director shot no longer exists.')
      next.assets[generatedAsset.id] = generatedAsset
      const originalAssetId = target.item.asset_id || ''
      target.item.take_asset_ids = uniqueItemIds([
        ...(target.item.take_asset_ids || []),
        originalAssetId,
        generatedAsset.id,
      ]).filter(id => Boolean(next.assets[id]))
      target.item.take_states = {
        ...(target.item.take_states || {}),
        ...(originalAssetId ? {
          [originalAssetId]: {
            source_in: target.item.source_in,
            speed: target.item.speed,
          },
        } : {}),
        [generatedAsset.id]: { source_in: 0, speed: 1 },
      }
      target.item.ai_history = [
        ...(target.item.ai_history || []),
        {
          id: newId('ai-result'),
          tool: 'director_rerun' as const,
          asset_id: generatedAsset.id,
          created_at: Date.now() / 1000,
        },
      ].slice(-100)
      target.item.asset_id = generatedAsset.id
      target.item.name = `Shot ${provenance.clip_index + 1} · rerun`
      target.item.source_in = 0
      target.item.speed = 1
      target.item.director = {
        ...target.item.director,
        video_prompt: requestedPrompt || target.item.director.video_prompt,
        window_prompts: requestedPrompt ? undefined : target.item.director.window_prompts,
      }
      clampItemToSource(next, target.item)
      next.updated_at = Date.now() / 1000
      const persisted = await api.saveEditorProject(next)
      set(state => ({
        ...(state.project?.id === persisted.id ? {
          project: persisted,
          dirty: false,
          history: [...state.history, before].slice(-HISTORY_LIMIT),
          future: [],
        } : {}),
        library: [generatedAsset, ...state.library.filter(asset => (
          asset.name !== generatedAsset.name || asset.workspace !== generatedAsset.workspace
        ))],
        error: null,
      }))
      await get().refreshProjects()
      void useStore.getState().loadOutputs()
    } catch (error) {
      set({ error: error instanceof Error ? error.message : 'Director shot rerun failed.' })
    } finally {
      set({ directorRerunItemId: null })
    }
  },

  addTitle: (at, preferredTrackId) => {
    const itemId = newId('title')
    const requestedStart = Math.max(0, at ?? get().playhead)
    let placedStart = requestedStart
    commitProject(set, get, project => {
      const track = bestTrack(project, 'text', preferredTrackId)
      placedStart = closestAvailableEditorStart(track, requestedStart, 3, undefined, true)
      track.items.push({
        id: itemId,
        name: 'Title',
        start: placedStart,
        duration: 3,
        source_in: 0,
        speed: 1,
        volume: 1,
        opacity: 1,
        fit: 'contain',
        transform: { x: 0, y: 0, scale: 1, rotation: 0 },
        fade_in: 0,
        fade_out: 0,
        text: 'Your title',
        style: {
          x: 0,
          y: 0,
          font_family: DEFAULT_EDITOR_FONT,
          font_size: 64,
          color: '#ffffff',
          background_color: '#000000',
          background_opacity: 0.32,
          text_align: 'center',
        },
      })
      track.items.sort((left, right) => left.start - right.start)
    })
    const track = get().project?.tracks.find(candidate => candidate.items.some(item => item.id === itemId))
    set({
      selectedItemId: itemId,
      selectedItemIds: [itemId],
      selectedTrackId: track?.id || null,
      playhead: placedStart,
      mobilePanel: 'inspector',
    })
  },

  addTrack: type => {
    const trackId = newId(type)
    commitProject(set, get, project => {
      const count = project.tracks.filter(track => track.type === type).length + 1
      const topLayer = project.tracks
        .filter(track => track.type === type)
        .reduce(
          (maximum, track) => Math.max(maximum, track.z_index),
          type === 'text' ? 9 : -1,
        )
      project.tracks.push({
        id: trackId,
        name: `${type === 'video' ? 'Video' : type === 'audio' ? 'Audio' : 'Titles'} ${count}`,
        type,
        z_index: topLayer + 1,
        muted: false,
        locked: false,
        volume: 1,
        items: [],
      })
    })
    set({ selectedItemId: null, selectedItemIds: [], selectedTrackId: trackId, mobilePanel: 'inspector' })
  },
  renameTrack: (trackId, name) => commitProject(set, get, project => {
    const track = project.tracks.find(candidate => candidate.id === trackId)
    if (track) track.name = name.trimStart().slice(0, 80) || track.type
  }),
  setTrackVolume: (trackId, volume) => commitProject(set, get, project => {
    const track = project.tracks.find(candidate => candidate.id === trackId)
    if (track) track.volume = Math.max(0, Math.min(4, volume))
  }),
  setTrackZIndex: (trackId, zIndex) => commitProject(set, get, project => {
    const track = project.tracks.find(candidate => candidate.id === trackId)
    if (track) track.z_index = Math.max(-100, Math.min(100, Math.round(zIndex)))
  }),
  toggleTrackMute: trackId => commitProject(set, get, project => {
    const track = project.tracks.find(candidate => candidate.id === trackId)
    if (track) track.muted = !track.muted
  }),
  toggleTrackLock: trackId => commitProject(set, get, project => {
    const track = project.tracks.find(candidate => candidate.id === trackId)
    if (track) track.locked = !track.locked
  }),
  removeTrack: trackId => {
    commitProject(set, get, project => {
      const index = project.tracks.findIndex(track => track.id === trackId)
      if (index < 0) return
      project.tracks.splice(index, 1)
    })
    if (!get().project?.tracks.some(track => track.id === trackId)) {
      set({ selectedItemId: null, selectedItemIds: [], selectedTrackId: null })
    }
  },

  selectItem: (itemId, trackId = null, additive = false) => set(state => {
    if (!itemId) {
      return {
        selectedItemId: null,
        selectedItemIds: [],
        selectedTrackId: trackId,
        ...(trackId ? { mobilePanel: 'inspector' as const } : {}),
      }
    }
    const linkedIds = state.project ? linkedEditorItemIds(state.project, [itemId]) : [itemId]
    const selectedItemIds = additive
      ? linkedIds.every(id => state.selectedItemIds.includes(id))
        ? state.selectedItemIds.filter(id => !linkedIds.includes(id))
        : uniqueItemIds([...state.selectedItemIds, ...linkedIds])
      : linkedIds
    const primary = selectedItemIds.includes(itemId)
      ? itemId
      : selectedItemIds.at(-1) || null
    return {
      selectedItemId: primary,
      selectedItemIds,
      selectedTrackId: trackId,
      mobilePanel: 'inspector' as const,
    }
  }),
  selectItems: (itemIds, primaryItemId = null, trackId = null) => set(state => {
    const selectedItemIds = state.project
      ? linkedEditorItemIds(state.project, itemIds)
      : uniqueItemIds(itemIds)
    const selectedItemId = primaryItemId && selectedItemIds.includes(primaryItemId)
      ? primaryItemId
      : selectedItemIds.at(-1) || null
    const primaryTrack = selectedItemId && state.project
      ? itemLocation(state.project, selectedItemId)?.track.id || trackId
      : trackId
    return {
      selectedItemId,
      selectedItemIds,
      selectedTrackId: primaryTrack,
      ...(selectedItemId || primaryTrack ? { mobilePanel: 'inspector' as const } : {}),
    }
  }),
  setPlayhead: seconds => set({ playhead: Math.max(0, seconds) }),
  setPlaying: playing => set({ playing }),
  setPixelsPerSecond: value => set({ pixelsPerSecond: Math.max(20, Math.min(180, value)) }),
  setSnapping: snapping => set({ snapping }),
  setRipple: ripple => set({ ripple }),
  setMobilePanel: mobilePanel => set({ mobilePanel }),
  addMarker: time => commitProject(set, get, project => {
    const marker: EditorMarker = {
      id: newId('marker'),
      time: Math.max(0, time ?? get().playhead),
      label: `Marker ${(project.markers?.length || 0) + 1}`,
      color: '#f59e0b',
    }
    project.markers = [...(project.markers || []), marker]
      .sort((left, right) => left.time - right.time)
  }),
  updateMarker: (markerId, patch) => commitProject(set, get, project => {
    const marker = (project.markers || []).find(candidate => candidate.id === markerId)
    if (!marker) return
    Object.assign(marker, patch)
    marker.time = Math.max(0, Number.isFinite(marker.time) ? marker.time : 0)
    marker.label = marker.label.trim().slice(0, 120) || 'Marker'
    if (!/^#[0-9a-f]{6}$/i.test(marker.color)) marker.color = '#f59e0b'
    project.markers.sort((left, right) => left.time - right.time)
  }),
  removeMarker: markerId => commitProject(set, get, project => {
    project.markers = (project.markers || []).filter(marker => marker.id !== markerId)
  }),
  updateItem: (itemId, patch) => commitProject(set, get, project => {
    const located = itemLocation(project, itemId)
    if (!located || located.track.locked) return
    const originalEnd = located.item.start + located.item.duration
    const changesStart = Object.prototype.hasOwnProperty.call(patch, 'start')
    const changesDuration = Object.prototype.hasOwnProperty.call(patch, 'duration')
    Object.assign(located.item, patch)
    located.item.start = Math.max(0, located.item.start)
    located.item.duration = Math.max(MIN_ITEM_DURATION, located.item.duration)
    clampItemToSource(project, located.item)
    if (changesStart) {
      located.item.start = closestAvailableEditorStart(
        located.track,
        located.item.start,
        located.item.duration,
        located.item.id,
      )
    } else if (changesDuration) {
      const nextStart = nextEditorItemStart(located.track, located.item.id, originalEnd)
      if (nextStart !== null) {
        located.item.duration = Math.max(
          MIN_ITEM_DURATION,
          Math.min(located.item.duration, nextStart - located.item.start),
        )
      }
    }
    located.item.fade_in = Math.max(0, Math.min(located.item.duration, located.item.fade_in || 0))
    located.item.fade_out = Math.max(0, Math.min(located.item.duration, located.item.fade_out || 0))
    if (changesStart) located.track.items.sort((left, right) => left.start - right.start)
  }),
  moveItem: (itemId, start, trackId) => {
    const current = get().project
    if (!current) return
    const selectedSeeds = get().selectedItemIds.includes(itemId)
      ? get().selectedItemIds
      : [itemId]
    const affectedIds = linkedEditorItemIds(current, selectedSeeds)
    commitProject(set, get, project => {
      const located = itemLocation(project, itemId)
      if (!located || located.track.locked) return
      const multiMove = affectedIds.length > 1
      const destination = !multiMove && trackId
        ? project.tracks.find(track => (
            track.id === trackId
            && track.type === located.track.type
            && !track.locked
          ))
        : located.track
      if (!destination) return

      const nextPrimaryStart = closestAvailableEditorStart(
        destination,
        start,
        located.item.duration,
        located.item.id,
      )
      if (multiMove) {
        let delta = nextPrimaryStart - located.item.start
        const selectedLocations = affectedIds
          .map(id => itemLocation(project, id))
          .filter((entry): entry is NonNullable<ReturnType<typeof itemLocation>> => Boolean(entry))
        const earliest = Math.min(...selectedLocations.map(entry => entry.item.start))
        if (earliest + delta < 0) delta = -earliest
        const moving = new Set(selectedLocations.map(entry => entry.item.id))
        const canMove = selectedLocations.every(entry => {
          const proposedStart = entry.item.start + delta
          const proposedEnd = proposedStart + entry.item.duration
          return !entry.track.items.some(other => (
            !moving.has(other.id)
            && proposedStart < other.start + other.duration - 1e-6
            && proposedEnd > other.start + 1e-6
          ))
        })
        if (!canMove) return
        selectedLocations.forEach(entry => { entry.item.start = Math.max(0, entry.item.start + delta) })
        project.tracks.forEach(track => track.items.sort((left, right) => left.start - right.start))
        return
      }
      located.item.start = nextPrimaryStart
      if (destination.id !== located.track.id) {
        located.track.items.splice(located.itemIndex, 1)
        destination.items.push(located.item)
      }
      located.track.items.sort((a, b) => a.start - b.start)
      if (destination.id !== located.track.id) {
        destination.items.sort((a, b) => a.start - b.start)
      }
    })
    const destination = get().project?.tracks.find(track => track.items.some(item => item.id === itemId))
    if (destination) set({ selectedTrackId: destination.id })
  },
  trimItem: (itemId, edge, time) => commitProject(set, get, project => {
    const located = itemLocation(project, itemId)
    if (!located || located.track.locked) return
    const { item } = located
    const originalStart = item.start
    const originalEnd = item.start + item.duration
    const affectedIds = linkedEditorItemIds(project, [itemId])
    if (affectedIds.length > 1) {
      const linked = affectedIds
        .map(id => itemLocation(project, id))
        .filter((entry): entry is NonNullable<ReturnType<typeof itemLocation>> => Boolean(entry))
      if (linked.some(entry => entry.track.locked)) return
      if (edge === 'start') {
        const previousEnd = previousEditorItemEnd(located.track, item.id, originalStart)
        const requestedStart = Math.max(
          previousEnd,
          Math.min(originalEnd - MIN_ITEM_DURATION, time),
        )
        const delta = requestedStart - originalStart
        const valid = linked.every(entry => {
          const proposedStart = entry.item.start + delta
          const proposedDuration = entry.item.duration - delta
          return proposedStart >= previousEditorItemEnd(entry.track, entry.item.id, entry.item.start) - 1e-6
            && proposedDuration >= MIN_ITEM_DURATION
        })
        if (!valid) return
        linked.forEach(entry => {
          entry.item.start += delta
          entry.item.duration -= delta
          entry.item.source_in = Math.max(0, entry.item.source_in + delta * entry.item.speed)
          entry.item.fade_in = Math.min(entry.item.duration, entry.item.fade_in || 0)
          entry.item.fade_out = Math.min(entry.item.duration, entry.item.fade_out || 0)
          clampItemToSource(project, entry.item)
        })
      } else {
        const requestedEnd = Math.max(originalStart + MIN_ITEM_DURATION, time)
        const delta = requestedEnd - originalEnd
        const valid = linked.every(entry => {
          const proposedEnd = entry.item.start + entry.item.duration + delta
          const nextStart = nextEditorItemStart(
            entry.track,
            entry.item.id,
            entry.item.start + entry.item.duration,
          )
          return entry.item.duration + delta >= MIN_ITEM_DURATION
            && (nextStart === null || proposedEnd <= nextStart + 1e-6)
        })
        if (!valid) return
        linked.forEach(entry => {
          entry.item.duration = Math.max(MIN_ITEM_DURATION, entry.item.duration + delta)
          entry.item.fade_in = Math.min(entry.item.duration, entry.item.fade_in || 0)
          entry.item.fade_out = Math.min(entry.item.duration, entry.item.fade_out || 0)
          clampItemToSource(project, entry.item)
        })
      }
      return
    }
    if (edge === 'start') {
      const previousEnd = previousEditorItemEnd(located.track, item.id, originalStart)
      const nextStart = Math.max(
        previousEnd,
        Math.min(originalEnd - MIN_ITEM_DURATION, time),
      )
      const delta = nextStart - originalStart
      item.start = nextStart
      item.duration = originalEnd - nextStart
      item.source_in = Math.max(0, item.source_in + delta * item.speed)
    } else {
      const nextStart = nextEditorItemStart(located.track, item.id, originalEnd)
      const requestedEnd = Math.max(item.start + MIN_ITEM_DURATION, time)
      const end = nextStart === null ? requestedEnd : Math.min(requestedEnd, nextStart)
      item.duration = Math.max(MIN_ITEM_DURATION, end - item.start)
    }
    item.fade_in = Math.min(item.duration, item.fade_in || 0)
    item.fade_out = Math.min(item.duration, item.fade_out || 0)
    clampItemToSource(project, item)
  }),
  splitSelected: () => {
    const selected = get().selectedItemId
    const playhead = get().playhead
    if (!selected) return
    const current = get().project
    if (!current) return
    const targets = selectedEditorItemIds(get())
      .filter(id => {
        const located = itemLocation(current, id)
        return Boolean(located && playhead > located.item.start + MIN_ITEM_DURATION
          && playhead < located.item.start + located.item.duration - MIN_ITEM_DURATION)
      })
    if (targets.length === 0) return
    const newIds = new Map(targets.map(id => [id, newId('clip')]))
    const newLinkGroups = new Map<string, string>()
    commitProject(set, get, project => {
      targets.forEach(targetId => {
        const located = itemLocation(project, targetId)
        if (!located || located.track.locked) return
        const { item, track, itemIndex } = located
        const splitOffset = playhead - item.start
        if (splitOffset <= MIN_ITEM_DURATION || splitOffset >= item.duration - MIN_ITEM_DURATION) return
        const rightLinkGroup = item.link_group_id
          ? newLinkGroups.get(item.link_group_id) || newId('link')
          : undefined
        if (item.link_group_id && rightLinkGroup) newLinkGroups.set(item.link_group_id, rightLinkGroup)
        const right = cloneTimelineItem(item, {
          id: newIds.get(targetId) || newId('clip'),
          start: playhead,
          duration: item.duration - splitOffset,
          source_in: item.source_in + splitOffset * item.speed,
          link_group_id: rightLinkGroup,
          fade_in: 0,
          transition_in: 'none',
        })
        item.duration = splitOffset
        item.fade_out = 0
        item.transition_out = 'none'
        item.fade_in = Math.min(item.duration, item.fade_in || 0)
        right.fade_out = Math.min(right.duration, right.fade_out || 0)
        track.items.splice(itemIndex + 1, 0, right)
      })
    })
    const primaryRightId = newIds.get(selected) || newIds.values().next().value
    const rightIds = Array.from(newIds.values())
    const track = get().project?.tracks.find(candidate => candidate.items.some(item => item.id === primaryRightId))
    if (track && primaryRightId) set({ selectedItemId: primaryRightId, selectedItemIds: rightIds, selectedTrackId: track.id })
  },
  duplicateSelected: () => {
    const selected = get().selectedItemId
    const current = get().project
    if (!selected || !current) return
    const targets = selectedEditorItemIds(get())
    if (targets.length === 0) return
    const duplicateIds = new Map(targets.map(id => [id, newId('clip')]))
    commitProject(set, get, project => {
      const locations = targets
        .map(id => itemLocation(project, id))
        .filter((entry): entry is NonNullable<ReturnType<typeof itemLocation>> => Boolean(entry && !entry.track.locked))
      if (locations.length === 0) return
      const moving = new Set(locations.map(entry => entry.item.id))
      const minimumStart = Math.min(...locations.map(entry => entry.item.start))
      const maximumEnd = Math.max(...locations.map(entry => entry.item.start + entry.item.duration))
      let delta = maximumEnd - minimumStart
      for (let attempt = 0; attempt < 100; attempt += 1) {
        let requiredShift = 0
        locations.forEach(entry => {
          const proposedStart = entry.item.start + delta
          const proposedEnd = proposedStart + entry.item.duration
          entry.track.items.forEach(other => {
            if (moving.has(other.id)) return
            if (proposedStart < other.start + other.duration - 1e-6 && proposedEnd > other.start + 1e-6) {
              requiredShift = Math.max(requiredShift, other.start + other.duration - proposedStart)
            }
          })
        })
        if (requiredShift <= 1e-6) break
        delta += requiredShift
      }
      const groupIds = new Map<string, string>()
      locations.forEach(entry => {
        const nextGroup = entry.item.link_group_id
          ? groupIds.get(entry.item.link_group_id) || newId('link')
          : undefined
        if (entry.item.link_group_id && nextGroup) groupIds.set(entry.item.link_group_id, nextGroup)
        entry.track.items.push(cloneTimelineItem(entry.item, {
          id: duplicateIds.get(entry.item.id) || newId('clip'),
          start: entry.item.start + delta,
          link_group_id: nextGroup,
        }))
      })
      project.tracks.forEach(track => track.items.sort((left, right) => left.start - right.start))
    })
    const primaryId = duplicateIds.get(selected) || duplicateIds.values().next().value
    const ids = Array.from(duplicateIds.values())
    const track = get().project?.tracks.find(candidate => candidate.items.some(item => item.id === primaryId))
    if (track && primaryId) set({ selectedItemId: primaryId, selectedItemIds: ids, selectedTrackId: track.id })
  },
  copySelected: () => {
    const project = get().project
    if (!project) return
    const targets = selectedEditorItemIds(get())
    const entries = targets.flatMap(itemId => {
      const located = itemLocation(project, itemId)
      if (!located) return []
      const asset = located.item.asset_id ? project.assets[located.item.asset_id] : undefined
      return [{
        item: cloneTimelineItem(located.item),
        trackType: located.track.type,
        trackId: located.track.id,
        asset: asset ? { ...asset } : undefined,
      }]
    })
    if (entries.length === 0) return
    set({
      clipboard: {
        entries,
        anchorStart: Math.min(...entries.map(entry => entry.item.start)),
      },
    })
  },
  cutSelected: () => {
    get().copySelected()
    if (get().clipboard) get().deleteSelected()
  },
  pasteClipboard: () => {
    const clipboard = get().clipboard
    const project = get().project
    if (!clipboard || !project) return
    const start = Math.max(0, get().playhead)
    const newIds = clipboard.entries.map(entry => ({ entry, id: newId(entry.trackType === 'text' ? 'title' : 'clip') }))
    const preferredTrackId = get().selectedTrackId || undefined
    commitProject(set, get, next => {
      const trackMap = new Map<string, EditorTrack>()
      const groupMap = new Map<string, string>()
      let offset = start - clipboard.anchorStart
      const minimumProposed = Math.min(...clipboard.entries.map(entry => entry.item.start + offset))
      if (minimumProposed < 0) offset -= minimumProposed
      newIds.forEach(({ entry }) => {
        if (entry.asset) next.assets[entry.asset.id] = { ...entry.asset }
        if (!trackMap.has(entry.trackId)) {
          const exact = next.tracks.find(track => track.id === entry.trackId && track.type === entry.trackType && !track.locked)
          trackMap.set(entry.trackId, exact || bestTrack(next, entry.trackType, preferredTrackId))
        }
      })
      // Keep a pasted group together. If any lane is occupied, advance the
      // whole group until every clip can be placed without overlap.
      for (let attempt = 0; attempt < 100; attempt += 1) {
        let requiredShift = 0
        newIds.forEach(({ entry }) => {
          const track = trackMap.get(entry.trackId)
          if (!track) return
          const proposedStart = entry.item.start + offset
          const proposedEnd = proposedStart + entry.item.duration
          track.items.forEach(other => {
            if (proposedStart < other.start + other.duration - 1e-6 && proposedEnd > other.start + 1e-6) {
              requiredShift = Math.max(requiredShift, other.start + other.duration - proposedStart)
            }
          })
        })
        if (requiredShift <= 1e-6) break
        offset += requiredShift
      }
      newIds.forEach(({ entry, id }) => {
        const track = trackMap.get(entry.trackId)
        if (!track) return
        const nextGroup = entry.item.link_group_id
          ? groupMap.get(entry.item.link_group_id) || newId('link')
          : undefined
        if (entry.item.link_group_id && nextGroup) groupMap.set(entry.item.link_group_id, nextGroup)
        track.items.push(cloneTimelineItem(entry.item, {
          id,
          start: entry.item.start + offset,
          link_group_id: nextGroup,
        }))
      })
      next.tracks.forEach(track => track.items.sort((left, right) => left.start - right.start))
    })
    const itemIds = newIds.map(entry => entry.id)
    const primaryId = itemIds[0]
    const track = get().project?.tracks.find(candidate => candidate.items.some(item => item.id === primaryId))
    const primary = track?.items.find(item => item.id === primaryId)
    if (track && primary) set({ selectedItemId: primaryId, selectedItemIds: itemIds, selectedTrackId: track.id, playhead: primary.start })
  },
  detachSelectedAudio: () => {
    const selected = get().selectedItemId
    const current = get().project
    if (!selected || !current) return
    const located = itemLocation(current, selected)
    const asset = located?.item.asset_id ? current.assets[located.item.asset_id] : undefined
    if (!located || located.track.type !== 'video' || !asset?.has_audio || located.item.muted) return
    const audioItemId = newId('audio')
    const linkGroupId = located.item.link_group_id || newId('link')
    commitProject(set, get, project => {
      const source = itemLocation(project, selected)
      if (!source || source.track.locked) return
      let audioTrack = project.tracks.find(track => (
        track.type === 'audio'
        && !track.locked
        && canPlaceEditorItem(track, source.item.start, source.item.duration)
      ))
      if (!audioTrack) {
        const audioTracks = project.tracks.filter(track => track.type === 'audio')
        const topLayer = audioTracks.reduce(
          (maximum, track) => Math.max(maximum, track.z_index),
          -1,
        )
        audioTrack = {
          id: newId('audio'),
          name: `Audio ${audioTracks.length + 1}`,
          type: 'audio',
          z_index: topLayer + 1,
          muted: false,
          locked: false,
          volume: 1,
          items: [],
        }
        project.tracks.push(audioTrack)
      }
      source.item.muted = true
      source.item.link_group_id = linkGroupId
      audioTrack.items.push(cloneTimelineItem(source.item, {
        id: audioItemId,
        name: `${source.item.name} · audio`,
        muted: false,
        link_group_id: linkGroupId,
        opacity: 1,
        style: undefined,
      }))
      audioTrack.items.sort((a, b) => a.start - b.start)
    })
    const track = get().project?.tracks.find(candidate => candidate.items.some(item => item.id === audioItemId))
    if (track) set({ selectedItemId: audioItemId, selectedItemIds: [selected, audioItemId], selectedTrackId: track.id })
  },
  linkSelected: () => {
    const current = get().project
    const selected = get().selectedItemIds
    if (!current || selected.length < 2) return
    const groupId = newId('link')
    commitProject(set, get, project => {
      selected.forEach(itemId => {
        const located = itemLocation(project, itemId)
        if (located && !located.track.locked) located.item.link_group_id = groupId
      })
    })
  },
  unlinkSelected: () => {
    const current = get().project
    if (!current) return
    const selected = selectedEditorItemIds(get())
    if (selected.length === 0) return
    commitProject(set, get, project => {
      selected.forEach(itemId => {
        const located = itemLocation(project, itemId)
        if (located && !located.track.locked) delete located.item.link_group_id
      })
    })
  },
  setActiveTake: (itemId, assetId) => commitProject(set, get, project => {
    const located = itemLocation(project, itemId)
    const asset = project.assets[assetId]
    if (!located || !asset || located.track.locked || located.track.type === 'text') return
    const currentAssetId = located.item.asset_id || ''
    const takeIds = uniqueItemIds([...(located.item.take_asset_ids || []), located.item.asset_id || '', assetId])
      .filter(id => Boolean(project.assets[id]))
    located.item.take_asset_ids = takeIds
    located.item.take_states = {
      ...(located.item.take_states || {}),
      ...(currentAssetId ? {
        [currentAssetId]: {
          source_in: located.item.source_in,
          speed: located.item.speed,
        },
      } : {}),
    }
    const nextState = located.item.take_states[assetId]
    located.item.asset_id = assetId
    located.item.name = asset.name
    located.item.source_in = nextState?.source_in ?? 0
    located.item.speed = nextState?.speed ?? 1
    clampItemToSource(project, located.item)
  }),
  beginAIRoundTrip: async (tool, returnMode) => {
    const existingRoundTrip = get().roundTrip
    if (existingRoundTrip && !['completed', 'failed'].includes(existingRoundTrip.status)) {
      set({ error: 'Finish or cancel the current Maestro AI round trip before starting another.' })
      return
    }
    const project = get().project
    const itemId = get().selectedItemId
    const located = project && itemId ? itemLocation(project, itemId) : null
    const asset = located?.item.asset_id ? project?.assets[located.item.asset_id] : null
    if (!project || !itemId || !located || located.track.type !== 'video' || asset?.type !== 'video') {
      set({ error: 'Select a video clip before sending it to Maestro AI.' })
      return
    }
    const saved = await get().saveProject()
    if (!saved) return
    const roundTrip: EditorAIRoundTrip = {
      id: newId('ai'),
      projectId: saved.id,
      workspace: saved.workspace,
      itemId,
      itemName: located.item.name,
      tool,
      returnMode,
      status: 'armed',
      baselineJobIds: useStore.getState().jobs.map(job => job.id),
      createdAt: Date.now(),
    }
    persistRoundTrip(roundTrip)
    set({ roundTrip, error: null, playing: false })

    const studio = useStore.getState()
    const sourcePath = asset.path || asset.name
    const clipEnd = Math.min(
      asset.duration || Number.POSITIVE_INFINITY,
      located.item.source_in + located.item.duration * located.item.speed,
    )
    if (tool === 'upscale' || tool === 'film_grain' || tool === 'revoice') {
      studio.sendClipToTools(sourcePath, asset.url, tool)
    } else {
      studio.setEditVideo(
        null,
        sourcePath,
        asset.url,
        asset.duration,
        `${asset.width || saved.canvas.width}x${asset.height || saved.canvas.height}`,
      )
      useStore.setState({
        editStartTime: Math.max(0, located.item.source_in),
        editEndTime: clipEnd,
      })
      studio.setStudioVideoWorkflow(
        tool === 'edit_anything'
          ? 'prompt_edit'
          : tool,
      )
    }
    useStore.getState().setSidebarMode('studio')
  },
  cancelAIRoundTrip: () => {
    persistRoundTrip(null)
    set({ roundTrip: null })
  },
  completeAIRoundTrip: async outputFile => {
    const pending = get().roundTrip
    if (!pending || pending.status === 'completed') return
    try {
      const outputName = outputFile.replace(/\\/g, '/').split('/').filter(Boolean).at(-1) || outputFile
      const outputResponse = await api.fetchOutputs(0, 0, { workspace: pending.workspace })
      const output = outputResponse.outputs.find(candidate => (
        candidate.name === outputName || candidate.name === outputFile
      ))
      const libraryAsset: EditorAsset = output
        ? libraryAssetFromOutput(output, 'output', pending.workspace)
        : {
            id: newId('library'),
            name: outputName,
            type: 'video',
            origin: 'output',
            workspace: pending.workspace,
            path: outputFile,
            url: api.getFileUrl(outputName, pending.workspace),
            duration: 0,
            width: 0,
            height: 0,
            fps: 0,
            has_audio: true,
          }
      const probe = await api.probeEditorMedia(libraryAsset, pending.workspace)
      const generatedAsset: EditorAsset = {
        ...libraryAsset,
        id: newId('asset'),
        type: mediaTypeFromProbe(probe),
        path: probe.path,
        duration: probe.duration,
        width: probe.width,
        height: probe.height,
        fps: probe.fps,
        has_audio: probe.has_audio,
        size: probe.size,
      }
      const current = get().project
      const sourceProject = current?.id === pending.projectId
        ? current
        : await api.fetchEditorProject(pending.projectId, pending.workspace)
      const before = cloneProject(sourceProject)
      const next = cloneProject(sourceProject)
      const located = itemLocation(next, pending.itemId)
      if (!located) throw new Error('The originating Editor clip no longer exists.')
      next.assets[generatedAsset.id] = generatedAsset
      const originalAssetId = located.item.asset_id || ''
      located.item.take_asset_ids = uniqueItemIds([
        ...(located.item.take_asset_ids || []),
        originalAssetId,
        generatedAsset.id,
      ]).filter(id => Boolean(next.assets[id]))
      const generatedUsesWholeSource = pending.tool === 'upscale'
        || pending.tool === 'film_grain'
        || pending.tool === 'revoice'
      located.item.take_states = {
        ...(located.item.take_states || {}),
        ...(originalAssetId ? {
          [originalAssetId]: {
            source_in: located.item.source_in,
            speed: located.item.speed,
          },
        } : {}),
        [generatedAsset.id]: {
          source_in: generatedUsesWholeSource ? located.item.source_in : 0,
          speed: generatedUsesWholeSource ? located.item.speed : 1,
        },
      }
      located.item.ai_history = [
        ...(located.item.ai_history || []),
        {
          id: newId('ai-result'),
          tool: pending.tool,
          asset_id: generatedAsset.id,
          created_at: Date.now() / 1000,
        },
      ].slice(-100)
      if (pending.returnMode === 'replace') {
        located.item.asset_id = generatedAsset.id
        located.item.name = generatedAsset.name
        located.item.source_in = located.item.take_states[generatedAsset.id].source_in
        located.item.speed = located.item.take_states[generatedAsset.id].speed
        clampItemToSource(next, located.item)
      }
      next.updated_at = Date.now() / 1000
      const saved = await api.saveEditorProject(next)
      const completed: EditorAIRoundTrip = {
        ...pending,
        status: 'completed',
        outputFile,
        error: undefined,
      }
      persistRoundTrip(completed)
      set(state => ({
        ...(state.project?.id === saved.id ? {
          project: saved,
          dirty: false,
          history: [...state.history, before].slice(-HISTORY_LIMIT),
          future: [],
        } : {}),
        library: [generatedAsset, ...state.library.filter(asset => asset.name !== generatedAsset.name)],
        roundTrip: completed,
        error: null,
      }))
      await get().refreshProjects()
      void useStore.getState().loadOutputs()
    } catch (error) {
      const failed: EditorAIRoundTrip = {
        ...pending,
        status: 'failed',
        error: error instanceof Error ? error.message : 'Maestro AI result could not return to Editor.',
      }
      persistRoundTrip(failed)
      set({ roundTrip: failed, error: failed.error || null })
    }
  },
  deleteSelected: () => {
    const selected = selectedEditorItemIds(get())
    if (selected.length === 0) return
    const ripple = get().ripple
    commitProject(set, get, project => {
      const selectedSet = new Set(selected)
      project.tracks.forEach(track => {
        if (track.locked) return
        const removed = track.items
          .filter(item => selectedSet.has(item.id))
          .sort((left, right) => left.start - right.start)
        if (removed.length === 0) return
        track.items = track.items.filter(item => !selectedSet.has(item.id))
        if (!ripple) return
        track.items.forEach(item => {
          const shift = removed.reduce((total, removedItem) => (
            item.start >= removedItem.start + removedItem.duration - 1e-6
              ? total + removedItem.duration
              : total
          ), 0)
          item.start = Math.max(0, item.start - shift)
        })
      })
    })
    set({ selectedItemId: null, selectedItemIds: [], selectedTrackId: null })
  },
  jumpToEdit: direction => {
    const project = get().project
    if (!project) return
    const frame = 1 / Math.max(1, project.canvas.fps)
    const points = Array.from(new Set([
      ...project.tracks.flatMap(track => (
        track.items.flatMap(item => [item.start, item.start + item.duration])
      )),
      ...(project.markers || []).map(marker => marker.time),
    ].map(point => Number(point.toFixed(6))))).sort((a, b) => a - b)
    const current = get().playhead
    const next = direction > 0
      ? points.find(point => point > current + frame / 2) ?? projectDuration(project)
      : [...points].reverse().find(point => point < current - frame / 2) ?? 0
    set({ playhead: Math.max(0, next), playing: false })
  },
  undo: () => {
    const current = get().project
    const history = get().history
    const previous = history.at(-1)
    if (!current || !previous) return
    set({
      project: cloneProject(previous),
      history: history.slice(0, -1),
      future: [cloneProject(current), ...get().future].slice(0, HISTORY_LIMIT),
      dirty: true,
      selectedItemId: null,
      selectedItemIds: [],
      selectedTrackId: null,
      playing: false,
    })
  },
  redo: () => {
    const current = get().project
    const future = get().future
    const next = future[0]
    if (!current || !next) return
    set({
      project: cloneProject(next),
      history: [...get().history, cloneProject(current)].slice(-HISTORY_LIMIT),
      future: future.slice(1),
      dirty: true,
      selectedItemId: null,
      selectedItemIds: [],
      selectedTrackId: null,
      playing: false,
    })
  },
  exportProject: async (mode = 'now') => {
    if (get().exportJobId) return
    const currentProject = get().project
    if (currentProject) {
      const usedAssetIds = new Set(currentProject.tracks.flatMap(track => (
        track.items.flatMap(item => item.asset_id ? [item.asset_id] : [])
      )))
      const missing = Object.values(currentProject.assets).filter(asset => (
        asset.missing && usedAssetIds.has(asset.id)
      ))
      if (missing.length > 0) {
        set({
          error: `Relink ${missing.length === 1 ? 'the offline source' : `${missing.length} offline sources`} before exporting: ${missing.slice(0, 3).map(asset => asset.name).join(', ')}`,
        })
        return
      }
    }
    const project = await get().saveProject()
    if (!project || projectDuration(project) <= 0) {
      set({ error: 'Add media or a title to the timeline before exporting.' })
      return
    }
    try {
      const result = await api.exportEditorProject(project, mode)
      set({ exportJobId: result.job_id, exportProgress: 0, error: null })
      trackEditorJob(
        result.job_id,
        result.status,
        progress => set({ exportProgress: progress }),
        status => {
          const failed = status.status !== 'completed'
          set({
            exportJobId: null,
            exportProgress: failed ? get().exportProgress : 1,
            error: failed ? status.error || status.message || 'Editor export failed' : null,
          })
          if (failed) return
          void (async () => {
            try {
              const serverProject = await api.fetchEditorProject(project.id, project.workspace)
              const outputFile = status.output_files.at(-1) || status.output_files[0]
              const serverRecord = serverProject.exports.find(record => record.filename === outputFile)
                || serverProject.exports[0]
              const fallbackDimensions = editorExportDimensions(project)
              const fallbackRecord: EditorExportRecord | null = outputFile ? {
                id: `export-${result.job_id}`,
                filename: outputFile,
                workspace: project.workspace,
                created_at: Date.now() / 1000,
                duration: projectDuration(project),
                width: fallbackDimensions.width,
                height: fallbackDimensions.height,
                fps: editorExportFps(project),
                codec: project.export.codec,
                quality: project.export.quality,
                encoder: project.export.encoder,
              } : null
              const record = serverRecord || fallbackRecord
              if (record) {
                set(state => {
                  if (!state.project || state.project.id !== project.id) return {}
                  const exports = [
                    record,
                    ...(state.project.exports || []).filter(entry => entry.filename !== record.filename),
                  ].slice(0, 50)
                  return {
                    project: {
                      ...state.project,
                      exports,
                      updated_at: Math.max(state.project.updated_at, serverProject.updated_at),
                    },
                  }
                })
              }
              await Promise.all([get().refreshProjects(), get().refreshLibrary()])
            } catch {
              // The universal queue already contains the successful output;
              // history synchronization is helpful but not render-critical.
              void get().refreshLibrary()
            }
          })()
        },
      )
    } catch (error) {
      set({ error: error instanceof Error ? error.message : 'Editor export failed to start' })
    }
  },
}))

// The Editor and Studio are separate app surfaces, but they share the same
// universal generation queue. Arm a round trip before opening Studio, claim
// the first subsequently-created generation job, then attach that job's
// output to the original Editor clip when it completes. This works for held
// queue entries as well as jobs started immediately.
let completingRoundTripJobId: string | null = null
useStore.subscribe(state => {
  const editor = useEditorStore.getState()
  const pending = editor.roundTrip
  if (!pending || pending.status === 'completed' || pending.status === 'failed') return
  const candidate = pending.claimedJobId
    ? state.jobs.find(job => job.id === pending.claimedJobId)
    : state.jobs.find(job => (
        job.kind !== 'editor_export'
        && !pending.baselineJobIds.includes(job.id)
        && ['held', 'queued', 'running', 'completed', 'failed', 'cancelled'].includes(job.status)
      ))
  if (!candidate) return
  const nextStatus: EditorAIRoundTripStatus = candidate.status === 'held' || candidate.status === 'queued'
    ? 'queued'
    : candidate.status === 'running'
      ? 'running'
      : candidate.status === 'completed'
        ? 'completed'
        : 'failed'
  if (pending.claimedJobId !== candidate.id || pending.status !== nextStatus) {
    const updated: EditorAIRoundTrip = {
      ...pending,
      claimedJobId: candidate.id,
      status: nextStatus === 'completed' ? 'running' : nextStatus,
      ...(nextStatus === 'failed' ? { error: candidate.error || candidate.message || 'Maestro AI generation failed.' } : {}),
    }
    persistRoundTrip(updated)
    useEditorStore.setState({ roundTrip: updated })
  }
  if (candidate.status === 'completed' && candidate.outputFiles.length > 0 && completingRoundTripJobId !== candidate.id) {
    completingRoundTripJobId = candidate.id
    void useEditorStore.getState().completeAIRoundTrip(candidate.outputFiles.at(-1) || candidate.outputFiles[0])
      .finally(() => { completingRoundTripJobId = null })
  } else if (candidate.status === 'completed' && candidate.outputFiles.length === 0) {
    const failed: EditorAIRoundTrip = {
      ...pending,
      claimedJobId: candidate.id,
      status: 'failed',
      error: 'The linked generation completed without returning a media file.',
    }
    persistRoundTrip(failed)
    useEditorStore.setState({ roundTrip: failed, error: failed.error || null })
  }
})

export { projectDuration }
