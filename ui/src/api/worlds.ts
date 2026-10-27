/** Typed client for the experimental Hy-Nomas Worlds workflow. */

const BASE = ''
const ACK_FIELD = 'acknowledge_pixelumm_noncommercial' as const

export const PIXELUMM_LICENSE_URL =
  'https://developer.download.nvidia.com/licenses/NVIDIA-OneWay-Noncommercial-License-22Mar2022.pdf'

export interface WorldRecord {
  world_id: string
  title?: string
  description?: string
  created_at?: number | string
  updated_at?: number | string
  video_url?: string
  poster_url?: string
  thumbnail_url?: string
  recording_available?: boolean
  segment_count?: number
  last_action?: string
}

export type WorldRequestFields =
  | { operation: 'create'; description: string; [ACK_FIELD]: true }
  | { operation: 'continue'; world_id: string; text: string; [ACK_FIELD]: true }
  | { operation: 'move'; world_id: string; action: WorldMoveAction; duration_ms: number; [ACK_FIELD]: true }

export type WorldMoveAction =
  | 'Forward' | 'Backward' | 'Left' | 'Right'
  | 'Look_Left' | 'Look_Right' | 'Look_Up' | 'Look_Down' | 'Wait'

export type WorldRequest = WorldRequestFields & {
  client_request_id: string
  request_digest: string
}

export interface WorldRequestReceipt {
  client_request_id: string
  request_digest: string
  state: string
  job_id?: string | null
  job_status?: string | null
  cancel_state?: 'none' | 'requested' | 'sent' | string
  detail?: string | null
  world?: WorldRecord
}

function stableValue(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(stableValue)
  if (value && typeof value === 'object') {
    return Object.fromEntries(
      Object.entries(value as Record<string, unknown>)
        .sort(([left], [right]) => left < right ? -1 : left > right ? 1 : 0)
        .map(([key, item]) => [key, stableValue(item)]),
    )
  }
  return value
}

function portableSha256(bytes: Uint8Array): string {
  const constants = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
  ]
  const state = [0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19]
  const bitLength = bytes.length * 8
  const paddedLength = Math.ceil((bytes.length + 9) / 64) * 64
  const padded = new Uint8Array(paddedLength)
  padded.set(bytes)
  padded[bytes.length] = 0x80
  const view = new DataView(padded.buffer)
  view.setUint32(paddedLength - 8, Math.floor(bitLength / 0x100000000))
  view.setUint32(paddedLength - 4, bitLength >>> 0)
  const rotateRight = (word: number, count: number) => (word >>> count) | (word << (32 - count))

  for (let offset = 0; offset < paddedLength; offset += 64) {
    const words = new Uint32Array(64)
    for (let index = 0; index < 16; index += 1) words[index] = view.getUint32(offset + index * 4)
    for (let index = 16; index < 64; index += 1) {
      const x = words[index - 15]
      const y = words[index - 2]
      const s0 = rotateRight(x, 7) ^ rotateRight(x, 18) ^ (x >>> 3)
      const s1 = rotateRight(y, 17) ^ rotateRight(y, 19) ^ (y >>> 10)
      words[index] = (words[index - 16] + s0 + words[index - 7] + s1) >>> 0
    }
    let [a, b, c, d, e, f, g, h] = state
    for (let index = 0; index < 64; index += 1) {
      const sum1 = rotateRight(e, 6) ^ rotateRight(e, 11) ^ rotateRight(e, 25)
      const choose = (e & f) ^ (~e & g)
      const temp1 = (h + sum1 + choose + constants[index] + words[index]) >>> 0
      const sum0 = rotateRight(a, 2) ^ rotateRight(a, 13) ^ rotateRight(a, 22)
      const majority = (a & b) ^ (a & c) ^ (b & c)
      const temp2 = (sum0 + majority) >>> 0
      h = g; g = f; f = e; e = (d + temp1) >>> 0
      d = c; c = b; b = a; a = (temp1 + temp2) >>> 0
    }
    state[0] = (state[0] + a) >>> 0; state[1] = (state[1] + b) >>> 0
    state[2] = (state[2] + c) >>> 0; state[3] = (state[3] + d) >>> 0
    state[4] = (state[4] + e) >>> 0; state[5] = (state[5] + f) >>> 0
    state[6] = (state[6] + g) >>> 0; state[7] = (state[7] + h) >>> 0
  }
  return state.map(word => word.toString(16).padStart(8, '0')).join('')
}

export async function digestWorldRequest(fields: WorldRequestFields): Promise<string> {
  const bytes = new TextEncoder().encode(JSON.stringify(stableValue(fields)))
  const webCrypto = typeof globalThis.crypto === 'undefined' ? undefined : globalThis.crypto
  if (webCrypto?.subtle) {
    const hash = await webCrypto.subtle.digest('SHA-256', bytes)
    return Array.from(new Uint8Array(hash), byte => byte.toString(16).padStart(2, '0')).join('')
  }
  // Local and LAN HTTP origins are not necessarily secure contexts, so keep
  // canonical digest generation available without SubtleCrypto.
  return portableSha256(bytes)
}

function newRequestUuid(): string {
  const webCrypto = typeof globalThis.crypto === 'undefined' ? undefined : globalThis.crypto
  if (typeof webCrypto?.randomUUID === 'function') return webCrypto.randomUUID()
  const random = new Uint8Array(16)
  if (webCrypto?.getRandomValues) webCrypto.getRandomValues(random)
  else for (let i = 0; i < random.length; i += 1) random[i] = Math.floor(Math.random() * 256)
  random[6] = (random[6] & 0x0f) | 0x40
  random[8] = (random[8] & 0x3f) | 0x80
  const hex = Array.from(random, byte => byte.toString(16).padStart(2, '0')).join('')
  return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`
}

export async function prepareWorldRequest(fields: WorldRequestFields): Promise<WorldRequest> {
  const normalized = { ...fields } as WorldRequestFields
  if (normalized.operation === 'create') normalized.description = normalized.description.trim()
  if (normalized.operation === 'continue') normalized.text = normalized.text.trim()
  const request_digest = await digestWorldRequest(normalized)
  return {
    ...normalized,
    client_request_id: newRequestUuid(),
    request_digest,
  } as WorldRequest
}

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${BASE}${path}`, {
    ...init,
    headers: {
      ...(init?.body ? { 'Content-Type': 'application/json' } : {}),
      ...init?.headers,
    },
  })
  if (!response.ok) {
    const error = await response.json().catch(() => ({})) as { detail?: string; error?: string }
    throw new Error(error.detail || error.error || `World request failed (${response.status}).`)
  }
  return response.json() as Promise<T>
}

async function requestBlob(path: string, init?: RequestInit): Promise<Blob> {
  const response = await fetch(`${BASE}${path}`, init)
  if (!response.ok) {
    const error = await response.json().catch(() => ({})) as { detail?: string; error?: string }
    throw new Error(error.detail || error.error || `World file request failed (${response.status}).`)
  }
  return response.blob()
}

export async function listWorlds(): Promise<WorldRecord[]> {
  const result = await requestJson<{ worlds: WorldRecord[] }>('/api/v1/worlds')
  return result.worlds
}

export async function getWorld(worldId: string): Promise<WorldRecord> {
  const result = await requestJson<{ world: WorldRecord }>(`/api/v1/worlds/${encodeURIComponent(worldId)}`)
  return result.world
}

export function submitWorldRequest(request: WorldRequest): Promise<WorldRequestReceipt> {
  return requestJson('/api/v1/worlds/requests', {
    method: 'POST',
    body: JSON.stringify(request),
  })
}

export function getWorldRequest(requestId: string): Promise<WorldRequestReceipt> {
  return requestJson(`/api/v1/worlds/requests/${encodeURIComponent(requestId)}`)
}

export function cancelWorldRequest(requestId: string): Promise<WorldRequestReceipt> {
  return requestJson(`/api/v1/worlds/requests/${encodeURIComponent(requestId)}/cancel`, {
    method: 'POST',
  })
}

export function downloadWorld(worldId: string): Promise<Blob> {
  return requestBlob(`/api/v1/worlds/${encodeURIComponent(worldId)}/save`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ [ACK_FIELD]: true }),
  })
}

export function recordWorld(worldId: string): Promise<Blob> {
  return requestBlob(`/api/v1/worlds/${encodeURIComponent(worldId)}/record`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ [ACK_FIELD]: true }),
  })
}

export function captureWorld(worldId: string): Promise<Blob> {
  return requestBlob(`/api/v1/worlds/${encodeURIComponent(worldId)}/capture`)
}

export function triggerBrowserDownload(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob)
  const anchor = document.createElement('a')
  anchor.href = url
  anchor.download = filename
  anchor.click()
  window.setTimeout(() => URL.revokeObjectURL(url), 60_000)
}
