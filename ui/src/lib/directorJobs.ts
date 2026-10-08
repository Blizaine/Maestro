import type { GenerationJob } from '../types'

/** Only an active render child of the displayed Director project shares its tile. */
export function isDirectorRenderChild(
  job: GenerationJob,
  pipelineId: string | null,
  pipeline: { status: string } | null,
): boolean {
  return Boolean(
    pipelineId
    && pipeline?.status === 'running'
    && job.directorPipelineId === pipelineId
    && job.directorDetachedOperation !== true
    && (job.status === 'queued' || job.status === 'running'),
  )
}
