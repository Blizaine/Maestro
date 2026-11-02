export type LoraAccelerationFilter = 'all' | 'hide' | 'only'

export function indexLoraAccelerationFlags(loras: Array<{filename: string; is_acceleration?: boolean}>): Record<string, boolean> {
  return Object.fromEntries(loras.filter(info => typeof info.is_acceleration === 'boolean')
    .map(info => [info.filename, info.is_acceleration === true]))
}

// Match the server's lightweight technical-name fallback. This works with an
// older backend too. A display-name edit or descriptive guide cannot relabel
// a file; unknown adapters stay in All / Hide Turbo.
const accelerationToken = /(?:^|[^a-z0-9])(?:turbo|lightning|lcm|tcd|pdd|dmd|dmad|hyper[-_ ]?sd|lightx2v)(?:$|[^a-z0-9])/i
const stepToken = /(?:^|[^a-z0-9])(?:[1-9]|1[0-6])[-_ ]?steps?(?:$|[^a-z0-9])/i

export function isAccelerationLora(filename: string, explicit?: boolean): boolean {
  if (typeof explicit === 'boolean') return explicit
  const name = filename.replace(/\\/g, '/').split('/').pop() || filename
  return accelerationToken.test(name) || stepToken.test(name)
}

export function matchesLoraAccelerationFilter(filename: string, filter: LoraAccelerationFilter, explicit?: boolean): boolean {
  return filter === 'all' || isAccelerationLora(filename, explicit) === (filter === 'only')
}
