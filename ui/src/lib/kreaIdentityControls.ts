export interface KreaIdentitySettings {
  krea2_ref_boost: number
  krea2_ref_boost_a: number
  krea2_grounding_px: number
}

export const KREA_IDENTITY_DEFAULTS: Readonly<KreaIdentitySettings> = Object.freeze({
  krea2_ref_boost: 1,
  krea2_ref_boost_a: 1,
  krea2_grounding_px: 768,
})

const KREA_IDENTITY_MODELS = new Set(['krea2_raw_edit', 'krea2_turbo_edit'])
export const KREA_IDENTITY_SETTING_KEYS = [
  'krea2_ref_boost',
  'krea2_ref_boost_a',
  'krea2_grounding_px',
] as const

/** Accept a model id or model/options record, while matching only supported Krea edit variants. */
export function isKreaIdentityEdit(value: unknown): boolean {
  if (typeof value === 'string') return KREA_IDENTITY_MODELS.has(value.trim().toLowerCase())
  if (!value || typeof value !== 'object' || Array.isArray(value)) return false
  const record = value as Record<string, unknown>
  return [record.model_type, record.modelType, record.architecture]
    .some(candidate => typeof candidate === 'string' && KREA_IDENTITY_MODELS.has(candidate.trim().toLowerCase()))
}

function finiteNumber(value: unknown, fallback: number): number {
  const parsed = typeof value === 'number'
    ? value
    : typeof value === 'string' && value.trim() !== '' ? Number(value) : Number.NaN
  return Number.isFinite(parsed) ? parsed : fallback
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value))
}

/** Normalize the supported custom settings without carrying unrelated custom keys. */
export function normalizeKreaIdentitySettings(value: unknown): KreaIdentitySettings {
  const source = value && typeof value === 'object' && !Array.isArray(value)
    ? value as Record<string, unknown>
    : {}
  return {
    krea2_ref_boost: clamp(finiteNumber(source.krea2_ref_boost, KREA_IDENTITY_DEFAULTS.krea2_ref_boost), 0, 10),
    krea2_ref_boost_a: clamp(finiteNumber(source.krea2_ref_boost_a, KREA_IDENTITY_DEFAULTS.krea2_ref_boost_a), 0, 10),
    // Keep all backend-valid integer sizes. The range slider uses 64px steps,
    // while its paired number input accepts any whole pixel value.
    krea2_grounding_px: clamp(Math.round(finiteNumber(source.krea2_grounding_px, KREA_IDENTITY_DEFAULTS.krea2_grounding_px)), 384, 1536),
  }
}

/** Return exactly the persisted Krea keys. Useful when storing model preferences. */
export function pickKreaIdentitySettings(value: unknown): KreaIdentitySettings {
  return normalizeKreaIdentitySettings(value)
}

export function countKreaIdentityReferences({
  generationMode,
  workflow,
  imageMode,
  sourceImagePresent,
  referenceCount,
  editReferencesSupported,
}: {
  generationMode: unknown
  workflow: unknown
  imageMode: unknown
  sourceImagePresent: boolean
  referenceCount: number
  editReferencesSupported: boolean
}): number {
  if (generationMode !== 'image' || workflow === 'upscale') return 0
  const references = Number.isFinite(referenceCount) ? Math.max(0, Math.floor(referenceCount)) : 0
  if (Number(imageMode) === 2) {
    return (sourceImagePresent ? 1 : 0) + (editReferencesSupported ? references : 0)
  }
  return workflow === 'generate' ? references : 0
}

/** Labels only the settings that currently have participating reference images. */
export function activeKreaIdentityLabels(value: unknown, referenceCount: number): string[] {
  if (referenceCount < 1) return []
  const settings = normalizeKreaIdentitySettings(value)
  const labels: string[] = []
  if (settings.krea2_ref_boost !== KREA_IDENTITY_DEFAULTS.krea2_ref_boost) labels.push('Krea subject likeness')
  if (referenceCount >= 2 && settings.krea2_ref_boost_a !== KREA_IDENTITY_DEFAULTS.krea2_ref_boost_a) {
    labels.push('Krea scene/reference likeness')
  }
  if (settings.krea2_grounding_px !== KREA_IDENTITY_DEFAULTS.krea2_grounding_px) {
    labels.push(`Krea grounding ${settings.krea2_grounding_px}px`)
  }
  return labels
}
