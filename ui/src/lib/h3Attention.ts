import type { GenerateParams, ModelOptions } from '../types'

/** Preserve an explicit H3 backend; SLA capability alone never enables it. */
export function normalizeH3Attention(
  options: ModelOptions,
  requested: unknown,
): GenerateParams['override_attention'] {
  if (!options.sla_attention && !options.sol_attention) return undefined
  if (requested === 'sdpa') return 'sdpa'
  // Keep SLA intent on unsupported runtimes so the engine reports its dense fallback.
  if (requested === 'sla' && options.sla_attention) return 'sla'
  if (requested === 'sol' && options.sol_attention && options.sol_attention_status?.supported) {
    return 'sol'
  }
  return options.sla_attention
    && (options.sla_attention_default ?? options.minimax_h3_fused_turbo) === true ? 'sla' : ''
}

/** Older Director projects stored Fused SLA in the Sol preference. */
export function directorH3SlaEnabled(
  options: Pick<ModelOptions, 'sla_attention' | 'sla_attention_default' | 'minimax_h3_fused_turbo'> | null | undefined,
  model: string,
  choices: Record<string, boolean>,
  legacySolChoices: Record<string, boolean>,
): boolean {
  if (!options?.sla_attention) return false
  const defaultOn = (options.sla_attention_default ?? options.minimax_h3_fused_turbo) === true
  return (choices[model] ?? (defaultOn ? legacySolChoices[model] : undefined) ?? defaultOn) === true
}
