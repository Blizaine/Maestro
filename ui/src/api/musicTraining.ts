export interface MusicStyle {
  id: string
  name: string
  trigger: string
  license: string
}

export interface MusicTrack {
  id?: string
  audio_path: string
  name?: string
  lyrics: string
  style: string
  holdout: boolean
  source_song?: string
  reviewed?: boolean
  preview_url?: string
}

export interface MusicAuditionSettings {
  enabled: boolean
  style?: string
  lyrics?: string
  seed?: number
  seconds?: number
  strength?: number
}

export interface MusicAudition {
  id: string
  branch: 'style' | 'audio'
  step: number
  checkpoint: string
  request_id: string
  status: 'running' | 'completed' | 'cancelled' | 'failed'
  settings: MusicAuditionSettings
  seconds?: number
  engine?: string
  error?: string
  metadata?: {truncated?: {semantic?: boolean}}
}

export interface MusicProject {
  id: string
  name: string
  trigger: string
  tracks: MusicTrack[]
  status: string
  progress: number
  message: string
  job_id?: string
  prepared?: Record<string, unknown>
  completed_steps?: number
  resume_available?: boolean
  training_options?: { rank: number; seed: number; learning_rate: number; lyric_alignment?: boolean }
  alignment?: {ready: boolean; tracks: Array<{track_id: string; coverage: number; status: string; words: number}>}
  audio_prepared?: Record<string, unknown>
  audio_resume_available?: boolean
  audio_completed_steps?: number
  audio_training_options?: {seed: number; learning_rate: number; conditioning_checkpoint: string}
  audio_baseline?: Record<string, number>
  audio_checkpoints?: Array<{step: number; file: string; scores: Record<string, number>; conditioning_checkpoint: string}>
  baseline?: Record<string, number>
  checkpoints: Array<{ step: number; file: string; scores: Record<string, number> }>
  audition_settings?: MusicAuditionSettings
  auditions?: MusicAudition[]
  reviewed_track_ids?: string[]
  review_drafts?: Record<string, {text: string; note: string; seconds: number; language: string; segments: Array<{start: number; end: number; text: string; uncertain: boolean}>}>
  sequence_coverage?: Array<{track_id: string; source_seconds: number; training_seconds: number; truncated: boolean}>
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`/api/v1/${path}`, init)
  const value = await response.json()
  if (!response.ok) throw new Error(typeof value.detail === 'string' ? value.detail : 'Music request failed')
  return value
}

const post = (body: unknown): RequestInit => ({method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)})
export const fetchMusicStyles = () => request<{styles: MusicStyle[]}>('music-styles')
export const fetchMusicProjects = () => request<{projects: MusicProject[]}>('music-training/projects')
export const createMusicProject = (name: string, trigger: string, tracks: MusicTrack[]) => request<MusicProject>('music-training/projects', post({name, trigger, tracks}))
export const startMusicPreparation = (id: string) => request<{job_id: string}>(`music-training/projects/${encodeURIComponent(id)}/prepare`, post({}))
export const startMusicTraining = (id: string, options: {steps: number; rank: number; seed: number; learning_rate: number; resume: boolean; lyric_alignment?: boolean; audition?: MusicAuditionSettings}) => request<{job_id: string}>(`music-training/projects/${encodeURIComponent(id)}/train`, post(options))
export const auditionMusicCheckpoint = (id: string, checkpoint: string) => request<MusicStyle>(`music-training/projects/${encodeURIComponent(id)}/audition-style`, post({checkpoint}))
export const auditionAudioCheckpoint = (id: string, checkpoint: string) => request<MusicStyle>(`music-training/projects/${encodeURIComponent(id)}/audition-style`, post({audio_checkpoint: checkpoint}))
export const forkMusicProject = (id: string) => request<MusicProject>(`music-training/projects/${encodeURIComponent(id)}/fork`, post({}))
export const prepareMusicSound = (id: string) => request<{job_id: string}>(`music-training/projects/${encodeURIComponent(id)}/prepare-audio`, post({}))
export const alignMusicLyrics = (id: string) => request<{job_id: string}>(`music-training/projects/${encodeURIComponent(id)}/align-lyrics`, post({}))
export const adaptMusicSound = (id: string, options: {steps: number; seed: number; learning_rate: number; resume: boolean; conditioning_checkpoint: string; audition?: MusicAuditionSettings}) => request<{job_id: string}>(`music-training/projects/${encodeURIComponent(id)}/adapt-audio`, post(options))
export const draftMusicLyrics = (id: string) => request<{job_id: string}>(`music-training/projects/${encodeURIComponent(id)}/review-data`, post({}))
export const reviewMusicTrack = (id: string, trackId: string, reviewed: boolean) => request<MusicProject>(`music-training/projects/${encodeURIComponent(id)}/review-track`, post({track_id: trackId, reviewed}))
export const renderMusicAudition = (id: string, branch: 'style' | 'audio', checkpoint: string, audition: MusicAuditionSettings) => request<{job_id: string}>(`music-training/projects/${encodeURIComponent(id)}/render-audition`, post({branch, checkpoint, audition}))
export const musicAuditionUrl = (id: string, auditionId: string) => `/api/v1/music-training/projects/${encodeURIComponent(id)}/auditions/${encodeURIComponent(auditionId)}`
export const musicRecordingUrl = (id: string, trackId: string) => `/api/v1/music-training/projects/${encodeURIComponent(id)}/recordings/${encodeURIComponent(trackId)}`
export const importMusicStyle = (form: FormData) => request<MusicStyle>('music-styles/import', {method: 'POST', body: form})
export const musicStyleExportUrl = (id: string) => `/api/v1/music-styles/${encodeURIComponent(id)}/export`
