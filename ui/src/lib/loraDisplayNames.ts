export interface LoraDisplayNameFields {
  filename?: string
  directory?: string
  display_name?: string
  display_name_override?: string | null
  suggested_name?: string
  version_label?: string | null
  managed?: boolean
}

export type LoraDisplayNameMap = Record<string, LoraDisplayNameFields>

export const LORA_DISPLAY_NAME_CHANGED_EVENT = 'maestro-lora-display-name-changed'

export function defaultLoraDisplayName(filename: string): string {
  return filename.replace(/\.(safetensors|sft)$/i, '')
}

export function getLoraDisplayName(filename: string, names: LoraDisplayNameMap): string {
  return names[filename]?.display_name?.trim() || defaultLoraDisplayName(names[filename]?.filename || filename)
}

/** Show release/variant labels, falling back to the source filename on collisions. */
export function getLoraSecondaryLabel(filename: string, names: LoraDisplayNameMap): string | null {
  const displayName = getLoraDisplayName(filename, names)
  const peers = Object.entries(names).filter(([candidate]) => getLoraDisplayName(candidate, names) === displayName)
  const versionLabel = names[filename]?.version_label?.trim() || ''
  if (peers.length < 2) return versionLabel || null
  const sourceFilename = names[filename]?.filename || filename
  const filenameCollision = peers.some(([candidate, fields]) =>
    candidate !== filename && (fields.filename || candidate) === sourceFilename,
  )
  const fallbackFilename = filenameCollision && names[filename]?.directory
    ? `${sourceFilename} · ${names[filename].directory}`
    : sourceFilename
  if (!versionLabel) return fallbackFilename
  const duplicateVersion = peers.some(([candidate, fields]) =>
    candidate !== filename && (fields.version_label?.trim() || '') === versionLabel,
  )
  return duplicateVersion ? fallbackFilename : versionLabel
}

export function indexLoraDisplayNames<T extends { filename: string } & LoraDisplayNameFields>(loras: T[]): LoraDisplayNameMap {
  return Object.fromEntries(loras.map(lora => [lora.filename, {
    filename: lora.filename,
    directory: 'directory' in lora ? String(lora.directory) : undefined,
    display_name: lora.display_name,
    display_name_override: lora.display_name_override,
    suggested_name: lora.suggested_name,
    version_label: lora.version_label,
    managed: lora.managed,
  }]))
}

export function notifyLoraDisplayNameChanged(scope: { model_type?: string; directory?: string }) {
  window.dispatchEvent(new CustomEvent(LORA_DISPLAY_NAME_CHANGED_EVENT, { detail: scope }))
}
