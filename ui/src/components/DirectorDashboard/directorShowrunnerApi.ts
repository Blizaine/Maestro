import type { PipelineClipState } from '../../types'

export interface DirectorShowrunnerFeatures {
  boundary_anchors: boolean
  motion_audio_continuation: boolean
  song_lyric_subtitles: boolean
  song_subtitles: boolean
  joined_cut_upscale: boolean
}

export interface DirectorJoinedCut {
  filename: string
}

export interface DirectorShowrunnerInfo {
  pipeline_id: string
  features: DirectorShowrunnerFeatures
  joined_cuts: DirectorJoinedCut[]
}

async function requestJson<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, init)
  const payload = await response.json().catch(() => ({})) as {
    detail?: string
    error?: string
  }
  if (!response.ok) {
    throw new Error(payload.detail || payload.error || `Director request failed (${response.status})`)
  }
  return payload as T
}

export function getDirectorShowrunnerInfo(pipelineId: string): Promise<DirectorShowrunnerInfo> {
  return requestJson(`/api/v1/director/pipelines/${encodeURIComponent(pipelineId)}/showrunner`)
}

export function updateDirectorBoundaryAnchors(
  pipelineId: string,
  clipIndex: number,
  anchors: { start_from_previous: boolean; end_on_next: boolean },
): Promise<{ pipeline_id: string; clip_index: number; video_stale: boolean }> {
  return requestJson(
    `/api/v1/director/pipelines/${encodeURIComponent(pipelineId)}/clips/${clipIndex}/anchors`,
    { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(anchors) },
  )
}

export function setDirectorSongSubtitles(
  pipelineId: string,
  enabled: boolean,
): Promise<{ pipeline_id: string; song_subtitles: boolean }> {
  return requestJson(
    `/api/v1/director/pipelines/${encodeURIComponent(pipelineId)}/delivery`,
    { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ song_subtitles: enabled }) },
  )
}

export function getDirectorLyricsSrt(
  pipelineId: string,
): Promise<{ filename: string; content: string; cue_count: number }> {
  return requestJson(`/api/v1/director/pipelines/${encodeURIComponent(pipelineId)}/lyrics.srt`)
}

export function upscaleDirectorShot(
  pipelineId: string,
  clip: PipelineClipState,
  spatial: string,
  options: Record<string, unknown>,
): Promise<{ job_ids: string[]; job_id: string; status: string }> {
  return requestJson(
    `/api/v1/director/pipelines/${encodeURIComponent(pipelineId)}/clips/${clip.index}/upscale`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ spatial_upsampling: spatial, ...options }),
    },
  )
}

export function upscaleDirectorJoinedCut(
  pipelineId: string,
  filename: string,
  spatial: string,
  options: Record<string, unknown>,
): Promise<{ job_ids: string[]; job_id: string; status: string }> {
  return requestJson(
    `/api/v1/director/pipelines/${encodeURIComponent(pipelineId)}/joined-cut/upscale`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ filename, spatial_upsampling: spatial, ...options }),
    },
  )
}

export type DirectorAnchoredClip = PipelineClipState & {
  start_from_previous?: boolean
  end_on_next?: boolean
  start_anchor_filename?: string | null
  end_anchor_filename?: string | null
}
