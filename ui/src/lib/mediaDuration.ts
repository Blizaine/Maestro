import { useEffect, useState, type RefObject } from 'react'
import type { OutputFile } from '../types'

const MAX_CONCURRENT_REQUESTS = 4
const MAX_QUEUED_REQUESTS = 64
const MAX_CACHE_ENTRIES = 256
const REQUEST_TIMEOUT_MS = 8000

type DurationCacheValue = number | null
type DurationRequest = {
  key: string
  url: string
  resolve: (duration: DurationCacheValue) => void
}

const durationCache = new Map<string, DurationCacheValue>()
const pendingDurations = new Map<string, Promise<DurationCacheValue>>()
const durationQueue: DurationRequest[] = []
let activeDurationRequests = 0

function positiveDuration(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) && value > 0 ? value : null
}

function readCachedDuration(key: string): { found: boolean; duration: DurationCacheValue } {
  if (!durationCache.has(key)) return { found: false, duration: null }
  const duration = durationCache.get(key) ?? null
  durationCache.delete(key)
  durationCache.set(key, duration)
  return { found: true, duration }
}

function cacheDuration(key: string, duration: DurationCacheValue) {
  durationCache.delete(key)
  durationCache.set(key, duration)
  while (durationCache.size > MAX_CACHE_ENTRIES) {
    const oldest = durationCache.keys().next().value
    if (oldest === undefined) break
    durationCache.delete(oldest)
  }
}

/** Convert a managed file URL to the same-origin endpoint that reads its facts. */
export function getMediaInfoUrl(fileUrl: string): string | null {
  try {
    const url = new URL(fileUrl, window.location.href)
    const filePrefix = '/api/v1/file/'
    if (url.origin !== window.location.origin || !url.pathname.startsWith(filePrefix)) return null
    url.pathname = url.pathname.replace(filePrefix, '/api/v1/media-info/')
    url.hash = ''
    return `${url.pathname}${url.search}`
  } catch {
    return null
  }
}

function startQueuedDurationRequests() {
  while (activeDurationRequests < MAX_CONCURRENT_REQUESTS && durationQueue.length > 0) {
    // A fast scroll can enqueue more files than the endpoint can serve at once.
    // Prefer the newest visible files and let old queued cards retry if needed.
    const request = durationQueue.pop()!
    activeDurationRequests += 1
    const controller = new AbortController()
    const timeout = window.setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS)

    fetch(request.url, {
      method: 'GET',
      credentials: 'same-origin',
      cache: 'no-store',
      headers: { Accept: 'application/json' },
      signal: controller.signal,
    })
      .then(response => response.ok ? response.json() : null)
      .then(body => positiveDuration(body?.duration_seconds))
      .catch(() => null)
      .then(duration => {
        cacheDuration(request.key, duration)
        pendingDurations.delete(request.key)
        request.resolve(duration)
      })
      .finally(() => {
        window.clearTimeout(timeout)
        activeDurationRequests -= 1
        startQueuedDurationRequests()
      })
  }
}

function requestDuration(key: string, url: string): Promise<DurationCacheValue> {
  const cached = readCachedDuration(key)
  if (cached.found) return Promise.resolve(cached.duration)

  const pending = pendingDurations.get(key)
  if (pending) return pending

  let resolve!: (duration: DurationCacheValue) => void
  const result = new Promise<DurationCacheValue>(done => { resolve = done })
  pendingDurations.set(key, result)
  durationQueue.push({ key, url, resolve })
  while (durationQueue.length > MAX_QUEUED_REQUESTS) {
    const evicted = durationQueue.shift()!
    pendingDurations.delete(evicted.key)
    evicted.resolve(null)
  }
  startQueuedDurationRequests()
  return result
}

function durationKey(file: Pick<OutputFile, 'url' | 'size' | 'created_at'>): string {
  return JSON.stringify([file.url, file.size, file.created_at])
}

/**
 * Read a video's measured duration only after its card or thumbnail is visible.
 * Identical file URLs and stat signatures share one bounded request.
 */
export function useMediaDuration(
  targetRef: RefObject<HTMLElement | null>,
  file: Pick<OutputFile, 'type' | 'url' | 'size' | 'created_at'>,
  seedSeconds?: unknown,
): number | null {
  const key = file.type === 'video' ? durationKey(file) : ''
  const seed = positiveDuration(seedSeconds)
  const [value, setValue] = useState<{ key: string; duration: number | null }>({ key: '', duration: null })

  useEffect(() => {
    if (file.type !== 'video') return

    const target = targetRef.current
    if (!target) return
    let current = true
    const setCurrentDuration = (duration: number | null) => {
      if (current) setValue({ key, duration })
    }
    const observer = new IntersectionObserver(entries => {
      if (!entries.some(entry => entry.isIntersecting)) return
      observer.disconnect()

      const cached = readCachedDuration(key)
      if (cached.found) {
        setCurrentDuration(cached.duration ?? seed)
        return
      }

      if (seed !== null) setCurrentDuration(seed)
      const url = getMediaInfoUrl(file.url)
      if (!url) {
        setCurrentDuration(seed)
        return
      }
      void requestDuration(key, url).then(duration => setCurrentDuration(duration ?? seed))
    }, { threshold: 0.01 })

    observer.observe(target)
    return () => {
      current = false
      observer.disconnect()
    }
  }, [file.type, file.url, key, seed, targetRef])

  return value.key === key ? value.duration : null
}

/** Format only a real positive duration as a compact gallery label. */
export function formatMediaDuration(seconds: unknown): string | null {
  const duration = positiveDuration(seconds)
  if (duration === null) return null

  const totalSeconds = Math.max(1, Math.round(duration))
  const hours = Math.floor(totalSeconds / 3600)
  const minutes = Math.floor((totalSeconds % 3600) / 60)
  const remainder = String(totalSeconds % 60).padStart(2, '0')
  if (hours > 0) return `${hours}:${String(minutes).padStart(2, '0')}:${remainder}`
  return `${Math.floor(totalSeconds / 60)}:${remainder}`
}
