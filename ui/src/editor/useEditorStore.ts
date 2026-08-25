import { create } from 'zustand'
import * as api from '../api/client'
import { useStore } from '../stores/useStore'
import type {
  EditorAsset,
  EditorCanvas,
  EditorExportSettings,
  EditorMediaProbe,
  EditorProject,
  EditorProjectSummary,
  EditorTimelineItem,
  EditorTrack,
  EditorTrackType,
  GenerationJob,
} from '../types'

const HISTORY_LIMIT = 60
const MIN_ITEM_DURATION = 1 / 30
let initializeSequence = 0

export type EditorMobilePanel = 'media' | 'timeline' | 'inspector' | 'projects'

interface EditorState {
  workspace: string
  project: EditorProject | null
  projects: EditorProjectSummary[]
  library: EditorAsset[]
  loading: boolean
  saving: boolean
  dirty: boolean
  error: string | null
  selectedItemId: string | null
  selectedTrackId: string | null
  playhead: number
  playing: boolean
  pixelsPerSecond: number
  snapping: boolean
  ripple: boolean
  history: EditorProject[]
  future: EditorProject[]
  mobilePanel: EditorMobilePanel
  exportJobId: string | null
  exportProgress: number

  initialize: (workspace: string) => Promise<void>
  refreshProjects: () => Promise<void>
  refreshLibrary: () => Promise<void>
  createProject: (name?: string, canvas?: Partial<EditorCanvas>) => Promise<void>
  loadProject: (projectId: string) => Promise<void>
  saveProject: () => Promise<EditorProject | null>
  deleteProject: (projectId: string) => Promise<void>
  renameProject: (name: string) => void
  setCanvas: (patch: Partial<EditorCanvas>) => void
  setExportSettings: (patch: Partial<EditorExportSettings>) => void

  uploadMedia: (file: File) => Promise<void>
  addMedia: (asset: EditorAsset, at?: number, trackId?: string) => Promise<void>
  addTitle: (at?: number) => void
  addTrack: (type: EditorTrackType) => void
  renameTrack: (trackId: string, name: string) => void
  toggleTrackMute: (trackId: string) => void
  toggleTrackLock: (trackId: string) => void

  selectItem: (itemId: string | null, trackId?: string | null) => void
  setPlayhead: (seconds: number) => void
  setPlaying: (playing: boolean) => void
  setPixelsPerSecond: (value: number) => void
  setSnapping: (enabled: boolean) => void
  setRipple: (enabled: boolean) => void
  setMobilePanel: (panel: EditorMobilePanel) => void
  updateItem: (itemId: string, patch: Partial<EditorTimelineItem>) => void
  moveItem: (itemId: string, start: number) => void
  trimItem: (itemId: string, edge: 'start' | 'end', time: number) => void
  splitSelected: () => void
  duplicateSelected: () => void
  deleteSelected: () => void
  undo: () => void
  redo: () => void
  exportProject: () => Promise<void>
}

function cloneProject(project: EditorProject): EditorProject {
  return typeof structuredClone === 'function'
    ? structuredClone(project)
    : JSON.parse(JSON.stringify(project)) as EditorProject
}

function newId(prefix: string): string {
  const random = typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function'
    ? crypto.randomUUID().replace(/-/g, '').slice(0, 16)
    : `${Date.now().toString(36)}${Math.random().toString(36).slice(2, 10)}`
  return `${prefix}-${random}`
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
): EditorAsset {
  return {
    id: newId('library'),
    name: output.name,
    type: output.type,
    origin,
    url: origin === 'upload' ? api.getUploadUrl(output.name) : api.getFileUrl(output.name),
    duration: 0,
    width: 0,
    height: 0,
    fps: 0,
    has_audio: output.type === 'audio' || output.type === 'video',
    size: output.size,
    created_at: output.created_at,
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
  const track: EditorTrack = {
    id: newId(type),
    name: type === 'video' ? 'Video' : type === 'audio' ? 'Audio' : 'Titles',
    type,
    z_index: type === 'text' ? 10 : project.tracks.filter(candidate => candidate.type === type).length,
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

function trackEditorJob(jobId: string, onProgress: (progress: number) => void, onFinish: (error?: string) => void): void {
  const editorJob: GenerationJob = {
    id: jobId,
    kind: 'editor_export',
    status: 'queued',
    progress: 0,
    step: 0,
    totalSteps: 0,
    phase: 'Queued',
    message: 'Editor export queued',
    outputFiles: [],
    error: null,
    oomInfo: null,
  }
  useStore.setState(state => ({
    jobs: [editorJob, ...state.jobs.filter(job => job.id !== jobId)],
    isGenerating: true,
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
        onFinish(status.status === 'completed' ? undefined : status.error || status.message)
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
      onFinish(message)
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
  loading: false,
  saving: false,
  dirty: false,
  error: null,
  selectedItemId: null,
  selectedTrackId: null,
  playhead: 0,
  playing: false,
  pixelsPerSecond: 54,
  snapping: true,
  ripple: false,
  history: [],
  future: [],
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
        selectedItemId: null,
        selectedTrackId: null,
        playhead: 0,
      })
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
      const workspace = get().workspace
      const [outputs, uploads] = await Promise.all([
        api.fetchOutputs(0, 0, { workspace }),
        api.fetchOutputs(0, 0, { workspace: '__uploads__' }),
      ])
      const seen = new Set<string>()
      const library = [
        ...outputs.outputs.map(output => libraryAssetFromOutput(output, 'output')),
        ...uploads.outputs.map(output => libraryAssetFromOutput(output, 'upload')),
      ].filter(asset => {
        const key = `${asset.origin}:${asset.name}`
        if (seen.has(key)) return false
        seen.add(key)
        return true
      }).sort((a, b) => (b.created_at || 0) - (a.created_at || 0))
      set({ library })
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
        selectedItemId: null,
        selectedTrackId: null,
        playhead: 0,
      })
      await get().refreshProjects()
    } catch (error) {
      set({ loading: false, error: error instanceof Error ? error.message : 'Unable to create project' })
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
        selectedItemId: null,
        selectedTrackId: null,
        playhead: 0,
      })
    } catch (error) {
      set({ loading: false, error: error instanceof Error ? error.message : 'Unable to open project' })
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
          return { project: saved, saving: false, dirty: false }
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
          selectedItemId: null,
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

  uploadMedia: async file => {
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
        path: uploaded.path,
        url: uploaded.url || api.getUploadUrl(uploaded.filename),
        duration: uploaded.duration_seconds || 0,
        width: 0,
        height: 0,
        fps: uploaded.fps || 0,
        has_audio: uploaded.has_audio ?? type !== 'image',
      }
      set(state => ({ library: [asset, ...state.library.filter(item => item.name !== asset.name || item.origin !== 'upload')] }))
      await get().addMedia(asset)
    } catch (error) {
      set({ error: error instanceof Error ? error.message : 'Media upload failed' })
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
      const start = Math.max(0, at ?? get().playhead)
      const itemId = newId('clip')
      const finalAsset = asset
      commitProject(set, get, next => {
        next.assets[assetId] = finalAsset
        const trackType: EditorTrackType = finalAsset.type === 'audio' ? 'audio' : 'video'
        const track = bestTrack(next, trackType, preferredTrackId)
        track.items.push({
          id: itemId,
          asset_id: assetId,
          name: finalAsset.name,
          start,
          duration: finalAsset.type === 'image' ? 5 : Math.max(MIN_ITEM_DURATION, finalAsset.duration || 5),
          source_in: 0,
          speed: 1,
          volume: 1,
          opacity: 1,
          fit: 'contain',
          transform: { x: 0, y: 0, scale: 1, rotation: 0 },
        })
        track.items.sort((a, b) => a.start - b.start)
      })
      const selectedTrack = get().project?.tracks.find(track => track.items.some(item => item.id === itemId))
      set({ selectedItemId: itemId, selectedTrackId: selectedTrack?.id || null, playhead: start })
    } catch (error) {
      set({ error: error instanceof Error ? error.message : 'Unable to add media' })
    }
  },

  addTitle: at => {
    const itemId = newId('title')
    const start = Math.max(0, at ?? get().playhead)
    commitProject(set, get, project => {
      const track = bestTrack(project, 'text')
      track.items.push({
        id: itemId,
        name: 'Title',
        start,
        duration: 3,
        source_in: 0,
        speed: 1,
        volume: 1,
        opacity: 1,
        fit: 'contain',
        transform: { x: 0, y: 0, scale: 1, rotation: 0 },
        text: 'Your title',
        style: { x: 0, y: 0, font_size: 64, color: '#ffffff' },
      })
    })
    const track = get().project?.tracks.find(candidate => candidate.items.some(item => item.id === itemId))
    set({ selectedItemId: itemId, selectedTrackId: track?.id || null, playhead: start })
  },

  addTrack: type => commitProject(set, get, project => {
    const count = project.tracks.filter(track => track.type === type).length + 1
    project.tracks.push({
      id: newId(type),
      name: `${type === 'video' ? 'Video' : type === 'audio' ? 'Audio' : 'Titles'} ${count}`,
      type,
      z_index: type === 'text' ? 10 + count : count,
      muted: false,
      locked: false,
      volume: 1,
      items: [],
    })
  }),
  renameTrack: (trackId, name) => commitProject(set, get, project => {
    const track = project.tracks.find(candidate => candidate.id === trackId)
    if (track) track.name = name.trimStart().slice(0, 80) || track.type
  }),
  toggleTrackMute: trackId => commitProject(set, get, project => {
    const track = project.tracks.find(candidate => candidate.id === trackId)
    if (track) track.muted = !track.muted
  }),
  toggleTrackLock: trackId => commitProject(set, get, project => {
    const track = project.tracks.find(candidate => candidate.id === trackId)
    if (track) track.locked = !track.locked
  }),

  selectItem: (itemId, trackId = null) => set({ selectedItemId: itemId, selectedTrackId: trackId }),
  setPlayhead: seconds => set({ playhead: Math.max(0, seconds) }),
  setPlaying: playing => set({ playing }),
  setPixelsPerSecond: value => set({ pixelsPerSecond: Math.max(20, Math.min(180, value)) }),
  setSnapping: snapping => set({ snapping }),
  setRipple: ripple => set({ ripple }),
  setMobilePanel: mobilePanel => set({ mobilePanel }),
  updateItem: (itemId, patch) => commitProject(set, get, project => {
    const located = itemLocation(project, itemId)
    if (!located || located.track.locked) return
    Object.assign(located.item, patch)
    located.item.start = Math.max(0, located.item.start)
    located.item.duration = Math.max(MIN_ITEM_DURATION, located.item.duration)
  }),
  moveItem: (itemId, start) => commitProject(set, get, project => {
    const located = itemLocation(project, itemId)
    if (!located || located.track.locked) return
    located.item.start = Math.max(0, start)
    located.track.items.sort((a, b) => a.start - b.start)
  }),
  trimItem: (itemId, edge, time) => commitProject(set, get, project => {
    const located = itemLocation(project, itemId)
    if (!located || located.track.locked) return
    const { item } = located
    const originalStart = item.start
    const originalEnd = item.start + item.duration
    if (edge === 'start') {
      const nextStart = Math.max(0, Math.min(originalEnd - MIN_ITEM_DURATION, time))
      const delta = nextStart - originalStart
      item.start = nextStart
      item.duration = originalEnd - nextStart
      item.source_in = Math.max(0, item.source_in + delta * item.speed)
    } else {
      item.duration = Math.max(MIN_ITEM_DURATION, time - item.start)
    }
  }),
  splitSelected: () => {
    const selected = get().selectedItemId
    const playhead = get().playhead
    if (!selected) return
    const newItemId = newId('clip')
    commitProject(set, get, project => {
      const located = itemLocation(project, selected)
      if (!located || located.track.locked) return
      const { item, track, itemIndex } = located
      const splitOffset = playhead - item.start
      if (splitOffset <= MIN_ITEM_DURATION || splitOffset >= item.duration - MIN_ITEM_DURATION) return
      const right: EditorTimelineItem = {
        ...item,
        id: newItemId,
        start: playhead,
        duration: item.duration - splitOffset,
        source_in: item.source_in + splitOffset * item.speed,
        transform: { ...item.transform },
        style: item.style ? { ...item.style } : undefined,
      }
      item.duration = splitOffset
      track.items.splice(itemIndex + 1, 0, right)
    })
    const track = get().project?.tracks.find(candidate => candidate.items.some(item => item.id === newItemId))
    if (track) set({ selectedItemId: newItemId, selectedTrackId: track.id })
  },
  duplicateSelected: () => {
    const selected = get().selectedItemId
    if (!selected) return
    const duplicateId = newId('clip')
    commitProject(set, get, project => {
      const located = itemLocation(project, selected)
      if (!located || located.track.locked) return
      located.track.items.push({
        ...located.item,
        id: duplicateId,
        start: located.item.start + located.item.duration,
        transform: { ...located.item.transform },
        style: located.item.style ? { ...located.item.style } : undefined,
      })
      located.track.items.sort((a, b) => a.start - b.start)
    })
    const track = get().project?.tracks.find(candidate => candidate.items.some(item => item.id === duplicateId))
    if (track) set({ selectedItemId: duplicateId, selectedTrackId: track.id })
  },
  deleteSelected: () => {
    const selected = get().selectedItemId
    if (!selected) return
    const ripple = get().ripple
    commitProject(set, get, project => {
      const located = itemLocation(project, selected)
      if (!located || located.track.locked) return
      const removedStart = located.item.start
      const removedDuration = located.item.duration
      located.track.items.splice(located.itemIndex, 1)
      if (ripple) {
        located.track.items.forEach(item => {
          if (item.start >= removedStart + removedDuration - 1e-6) {
            item.start = Math.max(removedStart, item.start - removedDuration)
          }
        })
      }
    })
    set({ selectedItemId: null, selectedTrackId: null })
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
      selectedTrackId: null,
      playing: false,
    })
  },
  exportProject: async () => {
    if (get().exportJobId) return
    const project = await get().saveProject()
    if (!project || projectDuration(project) <= 0) {
      set({ error: 'Add media or a title to the timeline before exporting.' })
      return
    }
    try {
      const result = await api.exportEditorProject(project)
      set({ exportJobId: result.job_id, exportProgress: 0, error: null })
      trackEditorJob(
        result.job_id,
        progress => set({ exportProgress: progress }),
        error => set({ exportJobId: null, exportProgress: error ? get().exportProgress : 1, error: error || null }),
      )
    } catch (error) {
      set({ error: error instanceof Error ? error.message : 'Editor export failed to start' })
    }
  },
}))

export { projectDuration }
