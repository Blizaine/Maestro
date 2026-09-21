export interface MusicStyle {
  id: string
  name: string
  trigger: string
  license: string
  tokenizer_pair?: 'v4' | 'v9'
  tokenizer_revision?: string | null
  archived?: boolean
  in_selector?: boolean
  created_at?: number
  adapted_pair?: {step: number}
  training?: {project_id?: string; checkpoint?: string; joint_checkpoint?: string; audio_checkpoint?: string}
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
  branch: 'style' | 'audio' | 'joint'
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
  tokenizer_pair?: 'v4' | 'v9'
  status: string
  progress: number
  message: string
  job_id?: string
  preparation_draft?: boolean
  auto_training_root?: string
  auto_training?: {
    version: number; voice_steps: number; style_steps: number
    status: 'queued' | 'running' | 'completed' | 'cancelled' | 'failed' | 'interrupted'
    stage: string; done?: string[]; sound_project_id?: string; style_project_id?: string; style_id?: string
    selection_summary?: {training_clips: number; check_clips: number; unreviewed_clips: number; warnings: string[]}
  }
  preparation?: {version: number; revision: number; songs: Record<string, MusicPreparedSong>; dataset_project_id?: string}
  prepared?: Record<string, unknown>
  completed_steps?: number
  resume_available?: boolean
  training_options?: { steps?: number; rank: number; seed: number; learning_rate: number; lyric_alignment?: boolean }
  training_workflow?: 'author' | 'style' | 'advanced'
  alignment?: {ready: boolean; tracks: Array<{track_id: string; coverage: number; status: string; words: number}>}
  audio_prepared?: Record<string, unknown>
  audio_resume_available?: boolean
  audio_completed_steps?: number
  audio_training_options?: {seed: number; learning_rate: number; conditioning_checkpoint: string}
  audio_baseline?: Record<string, number>
  joint_resume_available?: boolean
  joint_completed_steps?: number
  joint_training_options?: {rank: number; seed: number; learning_rate: number; window_frames: number; initialization?: {style_id: string; style_name: string; ar_rank: number; nar_rank: number}}
  joint_checkpoints?: Array<{step: number; file: string; nar_file: string; scores: Record<string, number>}>
  audio_checkpoints?: Array<{step: number; file: string; scores: Record<string, number>; conditioning_checkpoint: string}>
  baseline?: Record<string, number>
  checkpoints: Array<{ step: number; file: string; scores: Record<string, number> }>
  audition_settings?: MusicAuditionSettings
  auditions?: MusicAudition[]
  reviewed_track_ids?: string[]
  review_drafts?: Record<string, {text: string; note: string; seconds: number; language: string; segments: Array<{start: number; end: number; text: string; uncertain: boolean}>}>
  sequence_coverage?: Array<{track_id: string; source_seconds: number; training_seconds: number; truncated: boolean}>
  adapted_pair?: {head_sha256: string; nar_sha256: string; step: number; source_project?: string}
  pair_prepared?: Record<string, unknown>
  pair_completed_steps?: number
  pair_resume_available?: boolean
  pair_skipped_tracks?: string[]
  pair_training_options?: {seed: number; steps?: number}
  pair_checkpoints?: Array<{step: number; file: string; scores: Record<string, number>; audio_updates: number}>
}

export interface MusicPreparedClip {
  id: string
  start: number
  end: number
  lyrics: string
  style: string
  delivery: 'unspecified' | 'rap' | 'sung' | 'mixed'
  included: boolean
  reviewed: boolean
  warnings: string[]
}

export interface MusicPreparedSong {
  duration: number
  language: string
  holdout: boolean
  speakers: Array<{id: string; label: string; word_count: number; confident_words: number; seconds: number; start: number; end: number}>
  selected_speakers: string[]
  clips: MusicPreparedClip[]
  warnings: string[]
  diarization_failed?: boolean
  language_override?: string
  recognized_words?: number
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`/api/v1/${path}`, init)
  const value = await response.json()
  if (!response.ok) throw new Error(typeof value.detail === 'string' ? value.detail : 'Music request failed')
  return value
}

const post = (body: unknown): RequestInit => ({method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)})
export const fetchMusicStyles = (includeArchived = false) => request<{styles: MusicStyle[]}>(`music-styles${includeArchived ? '?include_archived=true' : ''}`)
const updateMusicStyleLibrary = async (id: string, changes: {archived?: boolean; in_selector?: boolean}) => {
  const result = await request<MusicStyle>(`music-styles/${encodeURIComponent(id)}`, {method: 'PATCH', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(changes)})
  window.dispatchEvent(new Event('maestro:music-styles-changed'))
  return result
}
export const setMusicStyleArchived = (id: string, archived: boolean) => updateMusicStyleLibrary(id, {archived})
export const setMusicStyleListed = (id: string, listed: boolean) => updateMusicStyleLibrary(id, {in_selector: listed})
export const fetchMusicProjects = () => request<{projects: MusicProject[]}>('music-training/projects')
export const createMusicProject = (name: string, trigger: string, tracks: MusicTrack[], pair = 'v9') => request<MusicProject>('music-training/projects', post({name, trigger, tracks, tokenizer_pair: pair}))
export const createMusicDraft = (name: string, trigger: string, tracks: MusicTrack[], pair = 'v9') => request<MusicProject>('music-training/drafts', post({name, trigger, tracks, tokenizer_pair: pair}))
export const analyzeMusicSongs = (id: string, retryVoices = false) => request<{job_id: string}>(`music-training/projects/${encodeURIComponent(id)}/analyze-songs`, post({retry_voices: retryVoices}))
export const startAutoMusicTraining = (id: string, voiceSteps: number, styleSteps: number) => request<{job_id: string}>(`music-training/projects/${encodeURIComponent(id)}/auto-train`, post({voice_steps: voiceSteps, style_steps: styleSteps}))
export const rescanMusicSong = (id: string, trackId: string, revision: number, language: string) => request<{job_id: string}>(`music-training/projects/${encodeURIComponent(id)}/analyze-songs`, post({track_id: trackId, revision, language}))
export const prepareMusicPair = (id: string) => request<{job_id: string}>(`music-training/projects/${encodeURIComponent(id)}/prepare-pair`, post({}))
export const adaptMusicPair = (id: string, steps: number, resume: boolean, seed = 22005) => request<{job_id: string}>(`music-training/projects/${encodeURIComponent(id)}/adapt-pair`, post({steps, resume, seed}))
export const selectMusicPair = (id: string, checkpoint: string) => request<MusicProject>(`music-training/projects/${encodeURIComponent(id)}/select-pair`, post({checkpoint}))
export const compareMusicPair = (id: string, checkpoint: string) => request<{job_id: string}>(`music-training/projects/${encodeURIComponent(id)}/reconstruct-pair`, post({checkpoint}))
export const buildMusicDataset = (id: string) => request<{job_id: string}>(`music-training/projects/${encodeURIComponent(id)}/build-dataset`, post({}))
export const selectMusicVoices = (id: string, trackId: string, revision: number, selected: string[]) => request<MusicProject>(`music-training/projects/${encodeURIComponent(id)}/select-voices`, post({track_id: trackId, revision, selected_speakers: selected}))
export const saveMusicPreparation = (id: string, trackId: string, revision: number, clips: MusicPreparedClip[], holdout: boolean) => request<MusicProject>(`music-training/projects/${encodeURIComponent(id)}/save-preparation`, post({track_id: trackId, revision, clips, holdout}))
export const musicPreparationAudioUrl = (id: string, trackId: string) => `/api/v1/music-training/projects/${encodeURIComponent(id)}/preparation/${encodeURIComponent(trackId)}/audio`
export const startMusicPreparation = (id: string) => request<{job_id: string}>(`music-training/projects/${encodeURIComponent(id)}/prepare`, post({}))
export const startMusicTraining = (id: string, options: {steps: number; rank: number; seed: number; learning_rate: number; resume: boolean; lyric_alignment?: boolean; audition?: MusicAuditionSettings}) => request<{job_id: string}>(`music-training/projects/${encodeURIComponent(id)}/train`, post(options))
export const auditionMusicCheckpoint = (id: string, checkpoint: string) => request<MusicStyle>(`music-training/projects/${encodeURIComponent(id)}/audition-style`, post({checkpoint}))
export const auditionAudioCheckpoint = (id: string, checkpoint: string) => request<MusicStyle>(`music-training/projects/${encodeURIComponent(id)}/audition-style`, post({audio_checkpoint: checkpoint}))
export const forkMusicProject = (id: string, pair?: 'v4' | 'v9') => request<MusicProject>(`music-training/projects/${encodeURIComponent(id)}/fork`, post({tokenizer_pair: pair}))
export const trainJointMusic = (id: string, options: {steps: number; rank: number; seed: number; learning_rate: number; resume: boolean; window_frames: number; initial_style_id?: string; audition?: MusicAuditionSettings}) => request<{job_id: string}>(`music-training/projects/${encodeURIComponent(id)}/train-joint`, post(options))
export const auditionJointCheckpoint = (id: string, checkpoint: string) => request<MusicStyle>(`music-training/projects/${encodeURIComponent(id)}/audition-style`, post({joint_checkpoint: checkpoint}))
export const prepareMusicSound = (id: string) => request<{job_id: string}>(`music-training/projects/${encodeURIComponent(id)}/prepare-audio`, post({}))
export const alignMusicLyrics = (id: string) => request<{job_id: string}>(`music-training/projects/${encodeURIComponent(id)}/align-lyrics`, post({}))
export const adaptMusicSound = (id: string, options: {steps: number; seed: number; learning_rate: number; resume: boolean; conditioning_checkpoint: string; audition?: MusicAuditionSettings}) => request<{job_id: string}>(`music-training/projects/${encodeURIComponent(id)}/adapt-audio`, post(options))
export const draftMusicLyrics = (id: string) => request<{job_id: string}>(`music-training/projects/${encodeURIComponent(id)}/review-data`, post({}))
export const reviewMusicTrack = (id: string, trackId: string, reviewed: boolean) => request<MusicProject>(`music-training/projects/${encodeURIComponent(id)}/review-track`, post({track_id: trackId, reviewed}))
export const renderMusicAudition = (id: string, branch: 'style' | 'audio' | 'joint', checkpoint: string, audition: MusicAuditionSettings) => request<{job_id: string}>(`music-training/projects/${encodeURIComponent(id)}/render-audition`, post({branch, checkpoint, audition}))
export const musicAuditionUrl = (id: string, auditionId: string) => `/api/v1/music-training/projects/${encodeURIComponent(id)}/auditions/${encodeURIComponent(auditionId)}`
export const musicRecordingUrl = (id: string, trackId: string) => `/api/v1/music-training/projects/${encodeURIComponent(id)}/recordings/${encodeURIComponent(trackId)}`
export const importMusicStyle = (form: FormData) => request<MusicStyle>('music-styles/import', {method: 'POST', body: form})
export const musicStyleExportUrl = (id: string) => `/api/v1/music-styles/${encodeURIComponent(id)}/export`
