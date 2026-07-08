import type { ModelDef } from '../types'

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
