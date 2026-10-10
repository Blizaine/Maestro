import type { ModelDef } from '../types'
import type { ModelVisibilityScope } from './modelVisibility'

/**
 * Resolve a model_type slug (e.g. "ltx2_22B_distilled_1_1") to its
 * human-readable display name (e.g. "LTX-2.3 Distilled 1.1 22B").
 *
 * The slug is what the backend writes into generation metadata
 * (`params.model_type`), but it's a backend-internal identifier and
 * doesn't tell the user which model family or variant produced the
 * output. Surfaces that show "what model made this" should always
 * route through this helper.
 *
 * Falls back to the slug itself when:
 *   - The model isn't in the loaded registry (older outputs from a
 *     model that's since been removed/renamed), so the user at least
 *     sees *something* identifying instead of an empty span.
 *   - The slug is empty/missing, returns ''.
 */
export function modelDisplayName(slug: string | undefined | null, models: ModelDef[]): string {
  if (!slug) return ''
  const found = models.find(m => m.model_type === slug)
  return found?.name || slug
}

function stripTrailingDescriptor(name: string, descriptor: string): string {
  const tagsStart = name.search(/(?:\s+\([^()]*\))+\s*$/)
  const tags = tagsStart >= 0 ? name.slice(tagsStart).trim() : ''
  const core = (tagsStart >= 0 ? name.slice(0, tagsStart) : name).trimEnd()
  const match = new RegExp(`^(.+?)(?:\\s*[—–·|]\\s*|\\s+)${descriptor}$`, 'i').exec(core)
  if (!match) return name
  return [match[1].trimEnd(), tags].filter(Boolean).join(' ')
}

function stripContextDescriptors(name: string, descriptors: string[]): string {
  let result = name
  // A catalog label can include both the workflow and the model-specific
  // capability. Strip only terminal workflow descriptors, leaving any
  // parenthesized source or experimental note in place.
  for (let pass = 0; pass < 3; pass += 1) {
    const previous = result
    for (const descriptor of descriptors) {
      result = stripTrailingDescriptor(result, descriptor)
    }
    if (result === previous) break
  }
  return result
}

/**
 * Return a compact label for model selectors and visibility settings.
 * Registry names and generation metadata continue to use the full name.
 */
export function modelNameForContext(model: ModelDef, scope: ModelVisibilityScope): string {
  const original = model.name || model.model_type
  const modelType = model.model_type.toLowerCase()
  const architecture = model.architecture?.toLowerCase() ?? ''
  const isH3 = model.model_type.toLowerCase().startsWith('minimax_h3')
    || architecture.startsWith('minimax_h3')
    || model.family === 'minimax_h3'

  if (scope === 'video:references') {
    let name = stripContextDescriptors(original, ['references?', 'ref2va'])
    if (isH3 && (modelType.startsWith('minimax_h3_ref2va') || architecture.startsWith('minimax_h3_ref2va'))) {
      // Omni/Ref2VA identifies the active H3 workflow. Keep checkpoint,
      // size, version, quantization, and imported-creator details intact.
      name = name.replace(/\bOmni\b/gi, '')
    }
    if (isH3) name = name.replace(/^MiniMax\s+(?=H3\b)/i, '')
    name = name.replace(/(\bH3)\s*[—–·|]\s*(?=\S)/gi, '$1 ')
    return name.replace(/\s{2,}/g, ' ').trim() || original
  }

  if (scope === 'video:frames') {
    let name = stripContextDescriptors(original, [
      'frames?',
      'first\\s*(?:[/–-]\\s*|\\s+and\\s+)last',
    ])
    if (isH3) {
      // H3's First / Last suffix describes the Frames workflow itself, while
      // Pruned/Full and fused recipes distinguish actual checkpoints.
      name = name.replace(/^((?:MiniMax\s+)?H3)\s+First\s*(?:[/–-]\s*|\s+and\s+)Last\s*[—–·|]\s*/i, '$1 ')
      name = name.replace(/^MiniMax\s+(?=H3\b)/i, '')
      name = name.replace(/(\bH3)\s*[—–·|]\s*(?=\S)/gi, '$1 ')
    }
    return name.replace(/\s{2,}/g, ' ').trim() || original
  }

  return original
}
