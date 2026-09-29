import { useEffect, useRef } from 'react'
import { usePageVisible } from './usePageVisible'

const POLL_INTERVAL_MS = 2000
// An unchanged answer this fast means the server did not hold the request.
const FAST_ANSWER_MS = 1000

/**
 * Feed `onData` from an endpoint that accepts `?since=<version>` and holds the
 * request until its data changes.
 *
 * A visible window keeps one request open; a hidden tab sends nothing and
 * asks again when it becomes visible. A server that answers
 * without a `version` is polled every 2 s instead.
 */
export function useLongPoll<T extends { version?: string }>(
  fetchSince: (since: string | undefined, signal: AbortSignal) => Promise<T>,
  onData: (data: T) => void,
  onError?: (error: unknown) => void,
): void {
  const visible = usePageVisible()
  const callbacks = useRef({ fetchSince, onData, onError })
  // Kept across tab switches: coming back with it, the server answers at once
  // only if something changed while the tab was hidden.
  const lastVersion = useRef<string | undefined>(undefined)

  useEffect(() => {
    callbacks.current = { fetchSince, onData, onError }
  })

  useEffect(() => {
    if (!visible) return
    const controller = new AbortController()
    let timer: number | undefined

    const schedule = (since: string | undefined, delay: number) => {
      timer = window.setTimeout(() => void request(since), Math.max(0, delay))
    }

    const request = async (since: string | undefined) => {
      const startedAt = Date.now()
      try {
        const data = await callbacks.current.fetchSince(since, controller.signal)
        if (controller.signal.aborted) return
        lastVersion.current = data.version
        callbacks.current.onData(data)
        const elapsed = Date.now() - startedAt
        if (data.version === undefined) schedule(undefined, POLL_INTERVAL_MS - elapsed)
        else if (data.version === since && elapsed < FAST_ANSWER_MS) schedule(since, POLL_INTERVAL_MS)
        else schedule(data.version, 0)
      } catch (error) {
        if (controller.signal.aborted) return
        lastVersion.current = undefined
        callbacks.current.onError?.(error)
        schedule(undefined, POLL_INTERVAL_MS)
      }
    }

    schedule(lastVersion.current, 0)
    return () => {
      controller.abort()
      window.clearTimeout(timer)
    }
  }, [visible])
}
