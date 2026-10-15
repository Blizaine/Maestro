import type { OutputFile } from '../types'
import { outputIdentity } from './galleryIdentity'

export interface DirectorGalleryGroupState {
  pipelineId: string
  workspace: string
  expanded: boolean
  loading: boolean
  loaded: boolean
  clips: OutputFile[]
  error: string | null
}

export type DirectorGalleryGroups = Record<string, DirectorGalleryGroupState>

export function directorGalleryGroupKey(workspace: string | undefined, pipelineId: string): string {
  return JSON.stringify([workspace || 'default', pipelineId])
}

export function flattenDirectorGalleryOutputs(
  outputs: OutputFile[],
  groups: DirectorGalleryGroups,
): OutputFile[] {
  const flattened: OutputFile[] = []
  const seen = new Set<string>()
  const append = (file: OutputFile) => {
    const identity = file.id || `${file.workspace || 'default'}/${file.name}`
    if (seen.has(identity)) return
    seen.add(identity)
    flattened.push(file)
  }

  for (const output of outputs) {
    append(output)
    const group = output.director_group
    if (!group) continue
    const expanded = groups[directorGalleryGroupKey(output.workspace, group.pipeline_id)]
    if (!expanded?.expanded || !expanded.loaded) continue
    for (const clip of expanded.clips) append(clip)
  }
  return flattened
}

/** Remove one cached take. Group clip_count tracks output files, not shot indexes. */
export function removeDirectorGalleryTake(
  groups: DirectorGalleryGroups,
  output: OutputFile,
): { groups: DirectorGalleryGroups; wasCached: boolean } {
  const pipelineId = output.director_pipeline_id
  if (!pipelineId) return { groups, wasCached: false }
  const key = directorGalleryGroupKey(output.workspace, pipelineId)
  const group = groups[key]
  if (!group?.clips.some(clip => outputIdentity(clip) === outputIdentity(output))) return { groups, wasCached: false }

  const clips = group.clips.filter(clip => outputIdentity(clip) !== outputIdentity(output))
  return {
    groups: { ...groups, [key]: { ...group, clips } },
    wasCached: true,
  }
}
