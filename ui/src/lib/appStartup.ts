/** Startup reads can fail while Pinokio is bringing the server online. */
export const RETRY_STARTUP_EVENT = 'maestro:retry-startup'

export function createAppStartup({loadControls, controlsReady, loadBackground, onError}: {
  loadControls: (cancelled: () => boolean) => Promise<void>
  controlsReady: () => boolean
  loadBackground: () => void
  onError: (message: string) => void
}) {
  let disposed = false
  let running: Promise<void> | null = null
  let attempts = 0
  let backgroundStarted = false
  let retryTimer: ReturnType<typeof setTimeout> | undefined
  let backgroundTimer: ReturnType<typeof setTimeout> | undefined
  const run = (): Promise<void> => {
    if (disposed) return Promise.resolve()
    if (running) return running
    clearTimeout(retryTimer)
    running = (async () => {
      attempts += 1
      try {
        await loadControls(() => disposed)
        if (disposed) return
        if (!controlsReady()) throw new Error('Model controls are not available yet.')
        if (!backgroundStarted) {
          backgroundStarted = true
          // Let the hydrated controls paint before gallery/indexing requests.
          backgroundTimer = setTimeout(() => { if (!disposed) loadBackground() }, 150)
        }
      } catch (error) {
        if (disposed) return
        onError(error instanceof Error ? error.message : 'Could not connect to Maestro.')
        // Bounded automatic recovery; later online/pageshow/manual events can
        // start a fresh bounded attempt without reloading the document.
        if (attempts < 5) retryTimer = setTimeout(() => void run(), Math.min(8000, 1000 * 2 ** (attempts - 1)))
      } finally { running = null }
    })()
    return running
  }
  return {
    run,
    retry: () => { attempts = 0; return run() },
    dispose: () => { disposed = true; clearTimeout(retryTimer); clearTimeout(backgroundTimer) },
  }
}
