const DB_NAME = 'maestro-thumbnails'
const STORE_NAME = 'thumbnails'
const DB_VERSION = 1

let dbInstance: IDBDatabase | null = null

function openDB(): Promise<IDBDatabase> {
  if (dbInstance) return Promise.resolve(dbInstance)
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION)
    req.onupgradeneeded = () => {
      req.result.createObjectStore(STORE_NAME)
    }
    req.onsuccess = () => {
      dbInstance = req.result
      resolve(req.result)
    }
    req.onerror = () => reject(req.error)
  })
}

export async function getCachedThumbnail(key: string): Promise<string | null> {
  try {
    const db = await openDB()
    return new Promise((resolve) => {
      const tx = db.transaction(STORE_NAME, 'readonly')
      const req = tx.objectStore(STORE_NAME).get(key)
      req.onsuccess = () => resolve(req.result ?? null)
      req.onerror = () => resolve(null)
    })
  } catch {
    return null
  }
}

export async function setCachedThumbnail(key: string, dataUrl: string): Promise<void> {
  try {
    const db = await openDB()
    return new Promise((resolve, reject) => {
      const tx = db.transaction(STORE_NAME, 'readwrite')
      tx.objectStore(STORE_NAME).put(dataUrl, key)
      tx.oncomplete = () => resolve()
      tx.onerror = () => reject(tx.error)
    })
  } catch {
    // silently fail
  }
}

/** Capture a video frame at the given time and return a data URL */
function captureVideoFrame(videoUrl: string, timeSeconds = 0.1): Promise<string> {
  return new Promise((resolve, reject) => {
    const video = document.createElement('video')
    video.muted = true
    video.preload = 'auto'
    video.src = videoUrl

    let settled = false
    const cleanup = () => {
      video.removeAttribute('src')
      video.load()
    }
    const fail = (e: unknown) => {
      if (settled) return
      settled = true
      cleanup()
      reject(e)
    }

    video.onloadeddata = () => {
      video.currentTime = timeSeconds
    }

    video.onseeked = () => {
      if (settled) return
      settled = true
      try {
        const canvas = document.createElement('canvas')
        canvas.width = video.videoWidth
        canvas.height = video.videoHeight
        const ctx = canvas.getContext('2d')
        if (!ctx) { cleanup(); reject(new Error('no canvas ctx')); return }
        ctx.drawImage(video, 0, 0)
        const dataUrl = canvas.toDataURL('image/webp', 0.7)
        cleanup()
        resolve(dataUrl)
      } catch (e) {
        cleanup()
        reject(e)
      }
    }

    video.onerror = () => fail(new Error('video load failed'))

    setTimeout(() => fail(new Error('timeout')), 10000)
  })
}

// --- Priority queue: most recently requested items are processed first ---
// This ensures visible thumbnails get captured before off-screen ones.
type QueueItem = {
  videoUrl: string
  name: string
  resolve: (dataUrl: string | null) => void
  timestamp: number
}

const queue: QueueItem[] = []
const pending = new Map<string, QueueItem[]>() // name -> list of resolvers waiting
let processing = false

async function processQueue() {
  if (processing) return
  processing = true

  while (queue.length > 0) {
    // Process newest request first (priority = most recently visible)
    queue.sort((a, b) => b.timestamp - a.timestamp)
    const item = queue.shift()!
    const name = item.name

    // Gather all resolvers waiting for this same thumbnail
    const waiters = pending.get(name) || []
    pending.delete(name)
    // Remove any remaining duplicates for this name from the queue
    for (let i = queue.length - 1; i >= 0; i--) {
      if (queue[i].name === name) {
        waiters.push(queue[i])
        queue.splice(i, 1)
      }
    }

    const allResolvers = [item, ...waiters]

    try {
      // Check cache first
      const cached = await getCachedThumbnail(name)
      if (cached) {
        for (const r of allResolvers) r.resolve(cached)
        continue
      }
      // Capture and cache
      const dataUrl = await captureVideoFrame(item.videoUrl)
      await setCachedThumbnail(name, dataUrl)
      for (const r of allResolvers) r.resolve(dataUrl)
    } catch {
      for (const r of allResolvers) r.resolve(null)
    }
  }

  processing = false
}

/**
 * Request a thumbnail for a video. Returns cached version instantly,
 * or queues a sequential capture with priority (newest requests first).
 * Deduplicates requests for the same file.
 */
export function requestThumbnail(videoUrl: string, name: string): Promise<string | null> {
  // Fast path: check if already in cache synchronously via the queue check
  return new Promise((resolve) => {
    queue.push({ videoUrl, name, resolve, timestamp: Date.now() })
    processQueue()
  })
}
