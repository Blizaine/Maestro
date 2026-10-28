import type { EditorAnimatedProperty, EditorProject, EditorTimelineItem, EditorTrack } from '../types'
import { evaluateEditorProperty } from './editorAnimation'
import { applyEditorEditOperation } from './editorEditOperations'

export type EditorAutoActionType = 'insert' | 'move' | 'trim' | 'remove' | 'set-volume' | 'crossfade' | 'close-gap'

/** A reviewed, deterministic edit instruction. Times and durations are seconds. */
export interface EditorAutoAction {
  id: string
  type: EditorAutoActionType
  item_id?: string
  asset_id?: string
  track_id?: string
  start?: number
  duration?: number
  source_in?: number
  volume?: number
  seconds?: number
  reason?: string
}

export interface EditorAutoProposal {
  proposal_id: string
  project_id: string
  workspace: string
  base_revision: number
  client_signature: string
  mode: string
  status: string
  summary: string
  actions: EditorAutoAction[]
  warnings: string[]
  missing_media: Array<string | { type: string; description: string }>
  drafts?: unknown[]
  errors?: string[]
  writer?: { provider?: string; model_id?: string }
}

export interface EditorAutoEditPreview {
  project: EditorProject
  changes: string[]
}

const MAX_ACTIONS = 64
const MAX_PROJECT_DURATION = 24 * 60 * 60
const MAX_ITEMS_PER_TRACK = 5000
const OVERLAP_EPSILON = 1e-6

type PinnedTimelineItem = EditorTimelineItem & { pinned?: boolean }
type PinnedEditorTrack = EditorTrack & { pinned?: boolean }

function canonicalize(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(canonicalize)
  if (value && typeof value === 'object') {
    return Object.fromEntries(Object.entries(value as Record<string, unknown>)
      .filter(([key]) => !['missing', 'created_at', 'updated_at'].includes(key))
      .sort(([left], [right]) => left.localeCompare(right))
      .map(([key, entry]) => [key, canonicalize(entry)]))
  }
  return value
}

/** Stable stale-draft token. Runtime availability, revisions, timestamps, and export history do not affect it. */
export function editorAutoEditSignature(project: EditorProject): string {
  const savedContent = Object.fromEntries(Object.entries(project)
    .filter(([key]) => !['revision', 'created_at', 'updated_at', 'exports'].includes(key)))
  const content = JSON.stringify(canonicalize(savedContent))
  let first = 2166136261
  let second = 5381
  for (let index = 0; index < content.length; index += 1) {
    const code = content.charCodeAt(index)
    first = Math.imul(first ^ code, 16777619)
    second = Math.imul(second, 33) ^ code
  }
  return `${content.length}-${(first >>> 0).toString(36)}-${(second >>> 0).toString(36)}`
}

function cloneProject(project: EditorProject): EditorProject {
  if (typeof globalThis.structuredClone === 'function') return globalThis.structuredClone(project)
  return JSON.parse(JSON.stringify(project)) as EditorProject
}

function locate(project: EditorProject, itemId: string): { item: EditorTimelineItem; track: EditorTrack } | undefined {
  for (const track of project.tracks) {
    const item = track.items.find(candidate => candidate.id === itemId)
    if (item) return { item, track }
  }
  return undefined
}

function isPinned(item: EditorTimelineItem): boolean {
  return Boolean((item as PinnedTimelineItem).pinned)
}

function assertNotPinned(item: EditorTimelineItem, description: string): void {
  if (isPinned(item)) throw new Error(`${description} would change pinned clip “${item.name}”. Unpin it first.`)
}

function assertTrackNotPinned(track: EditorTrack, description: string): void {
  if ((track as PinnedEditorTrack).pinned) {
    throw new Error(`${description} would change pinned track “${track.name}”. Unpin it first.`)
  }
}

function requireKnownSourceDuration(asset: EditorProject['assets'][string], description: string): void {
  if (asset.type !== 'image' && (!finite(asset.duration) || asset.duration <= 0)) {
    throw new Error(`${description} needs a known source duration. Probe or relink “${asset.name}” first.`)
  }
}

function finite(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value)
}

function frameDuration(project: EditorProject): number {
  const fps = finite(project.canvas?.fps) && project.canvas.fps > 0 ? project.canvas.fps : 30
  return 1 / fps
}

function validateItemRange(project: EditorProject, item: EditorTimelineItem): void {
  if (!finite(item.start) || item.start < 0) throw new Error(`Clip “${item.name}” needs a finite start at or after zero.`)
  if (!finite(item.duration) || item.duration + OVERLAP_EPSILON < frameDuration(project)) {
    throw new Error(`Clip “${item.name}” must be at least one project frame long.`)
  }
  if (item.start + item.duration > MAX_PROJECT_DURATION + OVERLAP_EPSILON) {
    throw new Error(`Clip “${item.name}” extends past the 24-hour project limit.`)
  }
  if (!finite(item.speed) || item.speed < 0.1 || item.speed > 8) {
    throw new Error(`Clip “${item.name}” needs a playback speed between 0.1× and 8×.`)
  }
  if (!finite(item.source_in) || item.source_in < 0) {
    throw new Error(`Clip “${item.name}” needs a finite source in-point at or after zero.`)
  }
  const asset = item.asset_id ? project.assets[item.asset_id] : undefined
  if (item.asset_id && !asset) throw new Error(`Clip “${item.name}” references missing project media.`)
  if (asset && asset.type !== 'image' && finite(asset.duration) && asset.duration > 0
    && item.source_in + item.duration * item.speed > asset.duration + OVERLAP_EPSILON) {
    throw new Error(`Clip “${item.name}” exceeds the source media duration.`)
  }
}

function validateProjectRanges(project: EditorProject): void {
  const itemIds = new Set<string>()
  const trackIds = new Set<string>()
  for (const track of project.tracks) {
    if (!track.id || trackIds.has(track.id)) throw new Error('Project track IDs must be present and unique.')
    trackIds.add(track.id)
    if (track.items.length > MAX_ITEMS_PER_TRACK) throw new Error(`Track “${track.name}” exceeds the clip limit.`)
    for (const item of track.items) {
      if (!item.id || itemIds.has(item.id)) throw new Error('Project clip IDs must be present and unique.')
      itemIds.add(item.id)
      validateItemRange(project, item)
    }
  }
}

function validateTrackOverlaps(project: EditorProject): void {
  const tolerance = Math.max(OVERLAP_EPSILON, frameDuration(project))
  for (const track of project.tracks) {
    const ordered = [...track.items].sort((left, right) => left.start - right.start || left.id.localeCompare(right.id))
    let previous: EditorTimelineItem | undefined
    let furthestEnd = -Infinity
    for (const incoming of ordered) {
      const previousEnd = previous ? previous.start + previous.duration : -Infinity
      if (previous && furthestEnd > previousEnd + OVERLAP_EPSILON
        && incoming.start < furthestEnd - OVERLAP_EPSILON) {
        throw new Error(`Track “${track.name}” would contain a three-clip overlap, which the timeline cannot represent safely.`)
      }
      const hasOverlapMarker = incoming.transition_overlap !== undefined
      if (hasOverlapMarker && !previous) {
        throw new Error(`Clip “${incoming.name}” declares a dissolve without a preceding video clip.`)
      }
      if (hasOverlapMarker) {
        if (!previous) throw new Error(`Clip “${incoming.name}” declares a dissolve without a preceding video clip.`)
        if (track.type !== 'video') {
          throw new Error(`Clip “${incoming.name}” declares a dissolve on a non-video track.`)
        }
        const overlap = previousEnd - incoming.start
        const declared = incoming.transition_overlap
        if (overlap <= 1e-9) {
          throw new Error(`Clip “${incoming.name}” declares a dissolve but no longer overlaps its predecessor.`)
        }
        if (incoming.transition_in !== 'dissolve') {
          throw new Error(`Clip “${incoming.name}” declares an overlap without transition_in='dissolve'.`)
        }
        if (!finite(declared) || Math.abs(declared - overlap) > tolerance) {
          throw new Error(`Clip “${incoming.name}” has a dissolve marker that does not match its overlap.`)
        }
        if (overlap > previous.duration + 1e-9 || overlap > incoming.duration + 1e-9) {
          throw new Error(`Clip “${incoming.name}” has a dissolve longer than one of its clips.`)
        }
        if ((incoming.fade_in || 0) + 1e-9 < overlap) {
          throw new Error(`Clip “${incoming.name}” needs a fade-in covering its dissolve overlap.`)
        }
        if ((previous.fade_out || 0) > 1e-9) {
          throw new Error(`Clip “${incoming.name}” needs an opaque predecessor with no fade-out.`)
        }
        if (previous.disabled) {
          throw new Error(`Clip “${incoming.name}” needs an enabled predecessor for its dissolve.`)
        }
      } else if (previous && previousEnd - incoming.start > OVERLAP_EPSILON) {
        throw new Error(`Clips “${previous.name}” and “${incoming.name}” overlap without a valid explicit dissolve.`)
      }
      furthestEnd = Math.max(furthestEnd, incoming.start + incoming.duration)
      previous = incoming
    }
  }
}

function requireSelectedItem(
  project: EditorProject,
  action: EditorAutoAction,
): { item: EditorTimelineItem; track: EditorTrack } {
  if (!action.item_id) throw new Error(`${action.type} needs a clip ID.`)
  const current = locate(project, action.item_id)
  if (!current) throw new Error(`Clip “${action.item_id}” no longer exists.`)
  if (current.track.locked) throw new Error(`Unlock track “${current.track.name}” before applying this auto-edit.`)
  assertTrackNotPinned(current.track, 'This action')
  assertNotPinned(current.item, 'This action')
  return current
}

function assertNoLinkedGeometry(item: EditorTimelineItem, actionName: string): void {
  if (item.link_group_id) {
    throw new Error(`${actionName} cannot safely change linked clip geometry. Unlink “${item.name}” first.`)
  }
}

function validateInsertTarget(project: EditorProject, action: EditorAutoAction): {
  track: EditorTrack
  asset: NonNullable<EditorProject['assets'][string]>
} {
  if (!action.track_id || !action.asset_id) throw new Error('Insert needs a target track and a project media ID.')
  const track = project.tracks.find(candidate => candidate.id === action.track_id)
  if (!track) throw new Error(`Target track “${action.track_id}” is no longer in this project.`)
  if (track.locked) throw new Error(`Unlock track “${track.name}” before inserting media.`)
  if (track.items.length >= MAX_ITEMS_PER_TRACK) throw new Error(`Track “${track.name}” is at its clip limit.`)
  const asset = project.assets[action.asset_id]
  if (!asset) throw new Error(`Media “${action.asset_id}” is not in this project.`)
  if (asset.missing) throw new Error(`Media “${asset.name}” is offline. Relink it before inserting.`)
  requireKnownSourceDuration(asset, 'Insert')
  assertTrackNotPinned(track, 'Insert')
  if (track.type === 'video' && asset.type === 'audio') throw new Error('Audio media cannot be inserted on a video track.')
  if (track.type === 'audio' && asset.type !== 'audio' && !(asset.type === 'video' && asset.has_audio)) {
    throw new Error('Choose audio media for an audio track.')
  }
  if (track.type === 'text') throw new Error('Insert actions need a visual or audio media track.')
  return { track, asset }
}

function newInsertId(project: EditorProject, proposalId: string, actionIndex: number): string {
  const ids = new Set(project.tracks.flatMap(track => track.items.map(item => item.id)))
  const safeProposal = proposalId.replace(/[^a-zA-Z0-9_-]/g, '-').slice(0, 48) || 'proposal'
  const base = `auto-${safeProposal}-${actionIndex + 1}`
  let candidate = base
  let suffix = 2
  while (ids.has(candidate)) candidate = `${base}-${suffix++}`
  return candidate
}

function keyframeBoundaryValues(item: EditorTimelineItem, newDuration: number): Partial<Record<EditorAnimatedProperty, number>> {
  const values: Partial<Record<EditorAnimatedProperty, number>> = {}
  if (!item.keyframes) return values
  for (const [property, frames] of Object.entries(item.keyframes)) {
    if (!frames?.some(frame => finite(frame.time) && frame.time > newDuration + OVERLAP_EPSILON)) continue
    const animatedProperty = property as EditorAnimatedProperty
    values[animatedProperty] = evaluateEditorProperty(item, animatedProperty, item.start + newDuration)
  }
  return values
}

function trimKeyframes(
  item: EditorTimelineItem,
  boundaryValues: Partial<Record<EditorAnimatedProperty, number>>,
): void {
  if (!item.keyframes) return
  const next: NonNullable<EditorTimelineItem['keyframes']> = {}
  for (const [property, frames] of Object.entries(item.keyframes)) {
    if (!frames) continue
    const animatedProperty = property as EditorAnimatedProperty
    const needsBoundary = finite(boundaryValues[animatedProperty])
    const byTime = new Map<number, typeof frames[number]>()
    frames.forEach(frame => {
      if (finite(frame.time) && finite(frame.value) && frame.time >= 0 && frame.time <= item.duration + OVERLAP_EPSILON) {
        const time = Math.min(frame.time, item.duration)
        byTime.set(time, { ...frame, time })
      }
    })
    if (needsBoundary) {
      byTime.set(item.duration, {
        time: item.duration,
        value: boundaryValues[animatedProperty]!,
        easing: 'linear',
      })
    }
    const kept = [...byTime.values()].sort((left, right) => left.time - right.time)
    if (kept.length) next[property as keyof typeof next] = kept
  }
  item.keyframes = Object.keys(next).length ? next : undefined
}

function actionDescription(action: EditorAutoAction, itemName?: string, trackName?: string): string {
  const target = itemName ? `“${itemName}”` : 'a new clip'
  let description: string
  switch (action.type) {
    case 'insert': description = `Insert “${action.asset_id}” on “${trackName}” at ${action.start?.toFixed(2)}s for ${action.duration?.toFixed(2)}s.`; break
    case 'move': description = `Move ${target} to ${action.start?.toFixed(2)}s${trackName ? ` on “${trackName}”` : ''}.`; break
    case 'trim': description = `Trim ${target} to ${action.duration?.toFixed(2)}s${finite(action.source_in) ? ` from source ${action.source_in.toFixed(2)}s` : ''}.`; break
    case 'remove': description = `Remove ${target}; its media remains in the project bin.`; break
    case 'set-volume': description = `Set ${target} volume to ${action.volume?.toFixed(2)}×.`; break
    case 'crossfade': description = `Crossfade ${target} over the preceding clip for ${action.seconds?.toFixed(2)}s.`; break
    case 'close-gap': description = `Close the gap before ${target} and move following clips together.`; break
  }
  return action.reason?.trim() ? `${description} Reason: ${action.reason.trim()}` : description
}

function applyAction(
  project: EditorProject,
  proposal: EditorAutoProposal,
  action: EditorAutoAction,
  actionIndex: number,
): string {
  if (action.type === 'insert') {
    const { track, asset } = validateInsertTarget(project, action)
    if (!finite(action.start) || !finite(action.duration)) throw new Error('Insert needs a finite start and duration.')
    const item: EditorTimelineItem = {
      id: newInsertId(project, proposal.proposal_id, actionIndex),
      asset_id: asset.id,
      name: asset.name,
      start: action.start,
      duration: action.duration,
      source_in: action.source_in ?? 0,
      speed: 1,
      volume: action.volume ?? 1,
      opacity: 1,
      fit: 'contain',
      transform: { x: 0, y: 0, scale: 1, rotation: 0 },
      fade_in: 0,
      fade_out: 0,
    }
    if (!finite(item.source_in) || item.source_in < 0) throw new Error('Insert needs a finite source in-point at or after zero.')
    if (!finite(item.volume) || item.volume < 0 || item.volume > 4) throw new Error('Inserted clip volume must be between 0 and 4.')
    validateItemRange(project, item)
    track.items.push(item)
    track.items.sort((left, right) => left.start - right.start || left.id.localeCompare(right.id))
    return actionDescription(action, undefined, track.name)
  }

  const { item, track } = requireSelectedItem(project, action)
  if (action.type === 'move') {
    assertNoLinkedGeometry(item, 'Move')
    if (!finite(action.start)) throw new Error('Move needs a finite target start.')
    const targetTrack = action.track_id
      ? project.tracks.find(candidate => candidate.id === action.track_id)
      : track
    if (!targetTrack) throw new Error(`Target track “${action.track_id}” is no longer in this project.`)
    if (targetTrack.locked) throw new Error(`Unlock target track “${targetTrack.name}” before moving this clip.`)
    assertTrackNotPinned(targetTrack, 'Move')
    if (targetTrack.type !== track.type) throw new Error('Moving a clip across track types is not supported.')
    item.start = action.start
    if (targetTrack !== track) {
      track.items = track.items.filter(candidate => candidate.id !== item.id)
      targetTrack.items.push(item)
      targetTrack.items.sort((left, right) => left.start - right.start || left.id.localeCompare(right.id))
    }
    validateItemRange(project, item)
    return actionDescription(action, item.name, targetTrack.name)
  }
  if (action.type === 'trim') {
    assertNoLinkedGeometry(item, 'Trim')
    if (!finite(action.duration)) throw new Error('Trim needs a finite duration.')
    const sourceAsset = item.asset_id ? project.assets[item.asset_id] : undefined
    if (sourceAsset) requireKnownSourceDuration(sourceAsset, 'Trim')
    const boundaryValues = keyframeBoundaryValues(item, action.duration)
    if (action.start !== undefined) {
      if (!finite(action.start)) throw new Error('Trim start must be finite.')
      item.start = action.start
    }
    item.duration = action.duration
    if (action.source_in !== undefined) {
      if (!finite(action.source_in)) throw new Error('Trim source in-point must be finite.')
      item.source_in = action.source_in
    }
    item.fade_in = Math.min(item.duration, Math.max(0, item.fade_in || 0))
    item.fade_out = Math.min(item.duration, Math.max(0, item.fade_out || 0))
    trimKeyframes(item, boundaryValues)
    if ((item.transition_overlap || 0) > item.duration + OVERLAP_EPSILON) {
      throw new Error('Trim would shorten this clip below its explicit dissolve overlap.')
    }
    validateItemRange(project, item)
    return actionDescription(action, item.name)
  }
  if (action.type === 'remove') {
    assertNoLinkedGeometry(item, 'Remove')
    track.items = track.items.filter(candidate => candidate.id !== item.id)
    return actionDescription(action, item.name)
  }
  if (action.type === 'set-volume') {
    if (!finite(action.volume)) throw new Error('Set volume needs a finite value.')
    applyEditorEditOperation(project, { type: 'set-volume', itemId: item.id, volume: action.volume })
    return actionDescription(action, item.name)
  }
  if (action.type === 'crossfade') {
    if (!finite(action.seconds)) throw new Error('Crossfade needs a finite duration.')
    const previous = [...track.items].filter(candidate => candidate.id !== item.id && candidate.start < item.start)
      .sort((left, right) => right.start - left.start)[0]
    if (!previous) throw new Error('Crossfade needs a preceding clip on the same track.')
    assertNotPinned(previous, 'Crossfade')
    const affected = track.items.filter(candidate => candidate.start >= item.start - OVERLAP_EPSILON)
    affected.forEach(candidate => assertNotPinned(candidate, 'Crossfade'))
    applyEditorEditOperation(project, { type: 'crossfade', itemId: item.id, seconds: action.seconds })
    return actionDescription(action, item.name)
  }
  if (action.type === 'close-gap') {
    const movingIds = new Set(track.items.filter(candidate => candidate.start >= item.start - OVERLAP_EPSILON).map(candidate => candidate.id))
    const linkGroups = new Set(track.items.filter(candidate => movingIds.has(candidate.id) && candidate.link_group_id)
      .map(candidate => candidate.link_group_id))
    project.tracks.forEach(candidateTrack => candidateTrack.items.forEach(candidate => {
      if (candidate.link_group_id && linkGroups.has(candidate.link_group_id)) movingIds.add(candidate.id)
    }))
    for (const candidateId of movingIds) {
      const affected = locate(project, candidateId)
      if (!affected) throw new Error('A linked clip needed for gap closure no longer exists.')
      assertTrackNotPinned(affected.track, 'Gap closure')
      assertNotPinned(affected.item, 'Gap closure')
    }
    applyEditorEditOperation(project, { type: 'close-gap', itemId: item.id })
    return actionDescription(action, item.name)
  }
  throw new Error(`Unsupported auto-edit action “${String(action.type)}”.`)
}

/** Preview a proposal against a clone. Any unsafe action rejects the entire preview without mutating the source. */
export function previewEditorAutoEdit(
  currentProject: EditorProject,
  proposal: EditorAutoProposal,
  selectedIDs: string[],
): EditorAutoEditPreview {
  if (proposal.project_id !== currentProject.id || proposal.workspace !== currentProject.workspace) {
    throw new Error('This auto-edit proposal belongs to a different project or workspace.')
  }
  if (!finite(proposal.base_revision) || proposal.base_revision !== (currentProject.revision || 0)) {
    throw new Error('This auto-edit proposal is stale because the project revision changed. Generate a new proposal.')
  }
  if (proposal.client_signature !== editorAutoEditSignature(currentProject)) {
    throw new Error('This auto-edit proposal is stale because the project content changed. Generate a new proposal.')
  }
  if (!proposal.proposal_id.trim()) throw new Error('The auto-edit proposal needs an ID.')
  if (!Array.isArray(proposal.actions) || proposal.actions.length === 0) throw new Error('The proposal has no actions to preview.')
  if (proposal.actions.length > MAX_ACTIONS) throw new Error(`Auto-edit proposals are limited to ${MAX_ACTIONS} actions.`)
  const actionIds = proposal.actions.map(action => action.id)
  if (actionIds.some(id => typeof id !== 'string' || !id.trim()) || new Set(actionIds).size !== actionIds.length) {
    throw new Error('Proposal action IDs must be present and unique.')
  }
  if (!Array.isArray(selectedIDs) || selectedIDs.length === 0) throw new Error('Select at least one proposed action to preview.')
  if (new Set(selectedIDs).size !== selectedIDs.length) throw new Error('The selected action list contains duplicate IDs.')
  const knownActionIds = new Set(actionIds)
  for (const id of selectedIDs) {
    if (!knownActionIds.has(id)) throw new Error(`Selected proposal action “${id}” is no longer available.`)
  }
  // Validate all references before allowing partial acceptance. A malformed
  // or stale unchecked action must not make a corrupted proposal appear valid.
  for (const action of proposal.actions) {
    if (action.type === 'insert') {
      if (action.track_id && !currentProject.tracks.some(track => track.id === action.track_id)) {
        throw new Error(`Action “${action.id}” references an unknown track.`)
      }
      if (action.asset_id && !currentProject.assets[action.asset_id]) {
        throw new Error(`Action “${action.id}” references unknown project media.`)
      }
    } else if (action.item_id && !locate(currentProject, action.item_id)) {
      throw new Error(`Action “${action.id}” references an unknown clip.`)
    }
  }
  validateProjectRanges(currentProject)

  const draft = cloneProject(currentProject)
  const changes: string[] = []
  const acceptedActions = new Set(selectedIDs)
  proposal.actions.forEach((action, actionIndex) => {
    if (!acceptedActions.has(action.id)) return
    try {
      if (action.type === 'insert') {
        if (!action.track_id || !currentProject.tracks.some(track => track.id === action.track_id)) {
          throw new Error(`Target track “${action.track_id || ''}” is no longer in this project.`)
        }
      }
      // Strict JSON writers include nullable, inapplicable fields. Treat those
      // as absent rather than mistaking (for example) trim.start=null for a move.
      const applicableAction = Object.fromEntries(Object.entries(action).filter(([, value]) => value !== null)) as unknown as EditorAutoAction
      changes.push(applyAction(draft, proposal, applicableAction, actionIndex))
      validateProjectRanges(draft)
    } catch (error) {
      const message = error instanceof Error ? error.message : 'The action could not be applied safely.'
      throw new Error(`Action ${actionIndex + 1}: ${message}`)
    }
  })
  if (changes.length === 0) throw new Error('None of the selected proposal actions can be applied.')
  draft.tracks.forEach(track => track.items.sort((left, right) => left.start - right.start || left.id.localeCompare(right.id)))
  validateProjectRanges(draft)
  validateTrackOverlaps(draft)
  return { project: draft, changes }
}
