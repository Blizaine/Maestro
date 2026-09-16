/** Friendly labels only: keep original names and filenames for imports and bindings. */
export function characterDisplayName(value: string): string {
  const filename = value.trim().split(/[\\/]/).pop() || value.trim()
  const name = filename
    .replace(/^minimaxh3_/i, '')
    .replace(/(?:\.maestro)?\.safetensors$/i, '')
    .replace(/(?:^|[_.\s-]+)refmod$/i, '')
  return (name !== filename ? name.replace(/_/g, ' ').trim() : name) || filename
}
