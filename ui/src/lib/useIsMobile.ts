import { useCallback, useSyncExternalStore } from 'react'

function useMediaQuery(query: string): boolean {
  const subscribe = useCallback((notify: () => void) => {
    const mql = window.matchMedia(query)
    const handler = () => notify()
    mql.addEventListener('change', handler)
    return () => mql.removeEventListener('change', handler)
  }, [query])
  const getSnapshot = useCallback(() => window.matchMedia(query).matches, [query])
  return useSyncExternalStore(subscribe, getSnapshot, () => false)
}

export function useIsMobile(breakpoint = 768): boolean {
  return useMediaQuery(`(max-width: ${breakpoint - 1}px)`)
}

// The app sidecar should stay collapsible on touch-first phones in landscape,
// where the viewport can be wider than the normal mobile breakpoint. Keeping
// this separate leaves the editor's existing viewport-based behavior intact.
export function useIsMobileSidecar(breakpoint = 768): boolean {
  return useMediaQuery(
    `(max-width: ${breakpoint - 1}px), (pointer: coarse) and (hover: none) and (orientation: landscape)`,
  )
}
