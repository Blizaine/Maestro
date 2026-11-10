export interface DirectorTake {
  id: string
  label: 'Draft' | 'Final' | string
  stage: 'draft' | 'final' | string
  pipeline_id: string
  filename: string
  resolution?: string | null
  seed?: number | null
  fingerprint?: string | null
  approved?: boolean
}

export interface DirectorReferenceEvidence {
  location_references: Array<{
    label: string
    status: 'available' | 'missing' | string
    filename: string
  }>
  voice_reference: {
    status: 'available' | 'missing' | 'not_attached' | string
    filename?: string | null
  }
  voice_verification: {
    status: 'not_verified' | string
    message: string
  }
}

export interface DirectorShotInspection {
  pipeline_id: string
  clip_index: number
  filename: string
  fingerprint: string
  workspace: string
  review: {
    creative_status: string
    technical: {
      probe: { status: string; error?: string; reason?: string }
      duration_seconds?: number | null
      expected_duration_seconds?: number | null
      duration_delta_seconds?: number | null
      geometry?: {
        width?: number | null
        height?: number | null
        display_aspect_ratio?: string | null
      }
      video_streams?: unknown[]
      audio?: { present: boolean | null; streams: unknown[] }
      container_format?: string | null
      software_decode: { status: string; error?: string; reason?: string }
    }
  }
  artifact_filenames: {
    contact_sheet?: string
    audio_preview?: string
    packet?: string
    stills?: string[]
  }
  reference_evidence: DirectorReferenceEvidence
}

export interface DirectorFinalizationResult {
  pipeline_id: string
  source_pipeline_id: string
  status: 'running' | string
  render_stage: 'final' | string
  resolution: string
}

async function mutateJson<T>(url: string, method: 'POST' | 'PUT', body: unknown): Promise<T> {
  const response = await fetch(url, {
    method,
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!response.ok) {
    const payload = await response.json().catch(() => ({})) as {
      detail?: string
      error?: string
    }
    throw new Error(payload.detail || payload.error || `Director request failed (${response.status})`)
  }
  return response.json() as Promise<T>
}

export function createDirectorFinal(
  pipelineId: string,
  resolutionPreset: string,
): Promise<DirectorFinalizationResult> {
  return mutateJson(
    `/api/v1/director/pipelines/${encodeURIComponent(pipelineId)}/finalize`,
    'POST',
    { resolution_preset: resolutionPreset },
  )
}

export async function inspectDirectorShot(
  pipelineId: string,
  clipIndex: number,
): Promise<DirectorShotInspection> {
  const response = await fetch(
    `/api/v1/director/pipelines/${encodeURIComponent(pipelineId)}/clips/${clipIndex}/inspect`,
    { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' },
  )
  if (!response.ok) {
    const payload = await response.json().catch(() => ({})) as {
      detail?: string
      error?: string
    }
    throw new Error(payload.detail || payload.error || `Shot review failed (${response.status})`)
  }
  return response.json() as Promise<DirectorShotInspection>
}

export function selectDirectorShotTake(
  pipelineId: string,
  clipIndex: number,
  takeId: string,
): Promise<{ pipeline_id: string; clip_index: number; selected_take_id: string; filename: string }> {
  return mutateJson(
    `/api/v1/director/pipelines/${encodeURIComponent(pipelineId)}/clips/${clipIndex}/take`,
    'PUT',
    { take_id: takeId },
  )
}
