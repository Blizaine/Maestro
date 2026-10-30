import type { AudioSubMode, EditSubMode, GenerationMode, StudioImageWorkflow, StudioVideoWorkflow } from '../types'

export const MODEL_VISIBILITY_GROUPS = [
  { scope: 'video:frames', label: 'Video · Frames' },
  { scope: 'video:references', label: 'Video · References' },
  { scope: 'video:avatar', label: 'Video · Avatar' },
  { scope: 'video:extend', label: 'Video · Extend' },
  { scope: 'video:blend', label: 'Video · Blend' },
  { scope: 'video:animate', label: 'Video · Animate' },
  { scope: 'video:retake', label: 'Video · Retake' },
  { scope: 'video:prompt_edit', label: 'Video · Prompt Edit' },
  { scope: 'video:outpaint', label: 'Video · Outpaint' },
  { scope: 'video:repaint', label: 'Video · Repaint' },
  { scope: 'video:recast', label: 'Video · Recast' },
  { scope: 'video:inpaint', label: 'Video · Inpaint' },
  { scope: 'image:generate', label: 'Image · Generate' },
  { scope: 'image:inpaint', label: 'Image · Inpaint' },
  { scope: 'image:outpaint', label: 'Image · Outpaint' },
  { scope: 'audio:speech', label: 'Audio · Speech' },
  { scope: 'audio:music', label: 'Audio · Music' },
  { scope: 'audio:sfx', label: 'Audio · Sound Effects' },
] as const

export type ModelVisibilityScope = typeof MODEL_VISIBILITY_GROUPS[number]['scope']
export type ModelVisibilityOverrides = Partial<Record<ModelVisibilityScope, Record<string, boolean>>>

// Workflow defaults are deliberately narrower than the global baseline, which
// also supplies the unchanged image, audio and video editing workflows.
const H3_AND_LTX_DEFAULTS = ['minimax_h3', 'minimax_h3_fused_turbo', 'ltx2_25'] as const
const H3_REFERENCE_DEFAULTS = ['minimax_h3_ref2va', 'minimax_h3_ref2va_fused_turbo'] as const
const H3_FRAMES_DEFAULTS = [...H3_AND_LTX_DEFAULTS, 'minimax_h3_pika_flex'] as const
const H3_REFERENCES_DEFAULTS = [...H3_REFERENCE_DEFAULTS, 'minimax_h3_ref2va_pika_flex'] as const
export const DEFAULT_MODELS_BY_SCOPE: Partial<Record<ModelVisibilityScope, readonly string[]>> = {
  'video:frames': H3_FRAMES_DEFAULTS,
  'video:references': H3_REFERENCES_DEFAULTS,
  'video:avatar': ['longcat_avatar', 'longcat_avatar_multi'],
  'video:extend': [...H3_AND_LTX_DEFAULTS, ...H3_REFERENCE_DEFAULTS],
  'video:blend': [...H3_AND_LTX_DEFAULTS, ...H3_REFERENCE_DEFAULTS],
}

/** Explicit choices let reset work even after All/None changed the baseline. */
export function defaultVisibilityForScope(
  scope: ModelVisibilityScope, modelIds: Iterable<string>, baselineDefaults: ReadonlySet<string>,
): Record<string, boolean> {
  const enabled = new Set(DEFAULT_MODELS_BY_SCOPE[scope] ?? baselineDefaults)
  return Object.fromEntries([...new Set([...modelIds, ...baselineDefaults, ...enabled])]
    .map(id => [id, enabled.has(id)]))
}

export function defaultVisibilityOverrides(
  modelIds: Iterable<string>, baselineDefaults: ReadonlySet<string>,
): ModelVisibilityOverrides {
  const ids = [...modelIds]
  return Object.fromEntries(Object.keys(DEFAULT_MODELS_BY_SCOPE).map(scope => [scope,
    defaultVisibilityForScope(scope as ModelVisibilityScope, ids, baselineDefaults),
  ]))
}

export function getModelVisibilityScope(state: {
  generationMode: GenerationMode
  studioVideoWorkflow?: StudioVideoWorkflow
  studioImageWorkflow?: StudioImageWorkflow
  audioSubMode?: AudioSubMode
  editSubMode?: EditSubMode
}): ModelVisibilityScope {
  if (state.generationMode === 'image') {
    return state.studioImageWorkflow === 'inpaint' ? 'image:inpaint'
      : state.studioImageWorkflow === 'outpaint' ? 'image:outpaint' : 'image:generate'
  }
  if (state.generationMode === 'audio') {
    return state.audioSubMode === 'music' ? 'audio:music'
      : state.audioSubMode === 'sfx' ? 'audio:sfx' : 'audio:speech'
  }
  if (state.generationMode === 'avatar') {
    const edit = state.editSubMode ?? 'retake'
    return edit === 'restyle' ? 'video:repaint'
      : edit === 'edit_anything' ? 'video:prompt_edit' : `video:${edit}`
  }
  const scope = `video:${state.studioVideoWorkflow ?? 'frames'}`
  return MODEL_VISIBILITY_GROUPS.some(group => group.scope === scope)
    ? scope as ModelVisibilityScope : 'video:frames'
}

/** Old global choices remain the baseline; only edited workflow choices override it. */
export function isModelEnabledInScope(
  modelType: string,
  enabledModels: ReadonlySet<string>,
  overrides: ModelVisibilityOverrides,
  scope: ModelVisibilityScope,
): boolean {
  const choice = overrides[scope]?.[modelType]
  return typeof choice === 'boolean' ? choice : enabledModels.has(modelType)
}

export function enabledModelsInScope(
  enabledModels: ReadonlySet<string>, overrides: ModelVisibilityOverrides, scope: ModelVisibilityScope,
): Set<string> {
  const result = new Set(enabledModels)
  for (const [modelType, enabled] of Object.entries(overrides[scope] ?? {})) {
    if (enabled) result.add(modelType)
    else result.delete(modelType)
  }
  return result
}

export function normalizeModelVisibilityOverrides(raw: unknown): ModelVisibilityOverrides {
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) return {}
  const result: ModelVisibilityOverrides = {}
  for (const { scope } of MODEL_VISIBILITY_GROUPS) {
    const choices = (raw as Record<string, unknown>)[scope]
    if (!choices || typeof choices !== 'object' || Array.isArray(choices)) continue
    const entries = Object.entries(choices).filter(([id, enabled]) => (
      id.length > 0 && id.length <= 200 && typeof enabled === 'boolean'
      && id !== '__proto__' && id !== 'constructor' && id !== 'prototype'
    ))
    if (entries.length) result[scope] = Object.fromEntries(entries)
  }
  return result
}
