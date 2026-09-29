import { useSyncExternalStore } from 'react'

function subscribe(notify: () => void): () => void {
  document.addEventListener('visibilitychange', notify)
  return () => document.removeEventListener('visibilitychange', notify)
}

function getSnapshot(): boolean {
  return document.visibilityState === 'visible'
}

/** False while the tab is in the background or the window is minimized. */
export function usePageVisible(): boolean {
  return useSyncExternalStore(subscribe, getSnapshot, () => true)
}
