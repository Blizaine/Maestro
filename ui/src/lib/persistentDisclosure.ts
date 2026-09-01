export function readPersistentDisclosure(key: string, fallback: boolean): boolean {
  try {
    const value = localStorage.getItem(key)
    if (value === '1') return true
    if (value === '0') return false
  } catch {
    // Storage can be unavailable in private/restricted browser contexts.
  }
  return fallback
}

export function writePersistentDisclosure(key: string, expanded: boolean): void {
  try {
    localStorage.setItem(key, expanded ? '1' : '0')
  } catch {
    // Disclosure state is a convenience; never block the controls on it.
  }
}
