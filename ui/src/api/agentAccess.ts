export const AGENT_ACCESS_AUTH_REQUIRED_EVENT = 'maestro-agent-access-auth-required'
export const AGENT_ACCESS_AUTH_REQUIRED_CODE = 'maestro_auth_required'
export const AGENT_ACCESS_SESSION_RESTORED_EVENT = 'maestro-agent-access-session-restored'

export interface AgentAccessStatus {
  available: boolean
  enabled: boolean
  owner_configured: boolean
  authenticated: boolean
  owner?: boolean
  installation_id?: string | null
  workstation_id?: string | null
  label?: string | null
  controller_url?: string | null
  local_setup_url?: string | null
  mcp_url?: string | null
  lan_mcp_urls?: string[]
  certificate_sha256?: string | null
  controller_status?: string | null
  discovery_status?: string | null
  backend_online?: boolean
}

export interface AgentAccessGrant {
  grant_id: string
  client_id: string
  client_name?: string
  scopes: string[]
  created_at?: number | string | null
  revoked?: boolean
  revoked_at?: number | string | null
}

export interface AgentAccessAuditEvent {
  id?: string
  time?: number | string | null
  timestamp?: number | string | null
  created_at?: number | string | null
  action?: string
  event?: string
  type?: string
  status?: string
  client_name?: string
  actor?: string
  outcome?: string
  summary?: string
  message?: string
}

export class AgentAccessApiError extends Error {
  readonly status: number
  readonly code: string | null

  constructor(status: number, message: string, code: string | null = null) {
    super(message)
    this.name = 'AgentAccessApiError'
    this.status = status
    this.code = code
  }
}

let nativeFetch: typeof window.fetch | null = null
let installedFetch: typeof window.fetch | null = null

/**
 * Install before React mounts. Only same-origin JSON API 401s carrying the
 * explicit owner-session code open the session gate; the original response is
 * returned untouched so existing request handlers keep working.
 */
export function installAgentAccessAuthInterceptor(): void {
  if (typeof window === 'undefined' || installedFetch) return

  const holder = window as Window & {
    __maestroAgentAccessNativeFetch?: typeof window.fetch
    __maestroAgentAccessInterceptedFetch?: typeof window.fetch
  }
  if (holder.__maestroAgentAccessInterceptedFetch && holder.__maestroAgentAccessNativeFetch) {
    nativeFetch = holder.__maestroAgentAccessNativeFetch
    installedFetch = holder.__maestroAgentAccessInterceptedFetch
    return
  }

  const original = window.fetch.bind(window)
  nativeFetch = original
  const intercepted: typeof window.fetch = async (input, init) => {
    let url: URL | null = null
    try {
      const target = input instanceof Request ? input.url : input instanceof URL ? input.href : String(input)
      url = new URL(target, window.location.href)
    } catch {
      // Preserve fetch's own behavior for malformed inputs.
    }

    const response = await original(input, init)
    if (!url || url.origin !== window.location.origin || !url.pathname.startsWith('/api/')) return response
    if (response.status !== 401 || !isJsonResponse(response)) return response

    try {
      const body = await response.clone().json() as {
        code?: unknown
        login_url?: unknown
        detail?: unknown
      }
      const nested = body?.detail && typeof body.detail === 'object'
        ? body.detail as { code?: unknown; login_url?: unknown }
        : null
      const code = nested?.code ?? body?.code
      const loginUrl = nested?.login_url ?? body?.login_url
      if (code !== AGENT_ACCESS_AUTH_REQUIRED_CODE) return response
      window.dispatchEvent(new CustomEvent(AGENT_ACCESS_AUTH_REQUIRED_EVENT, {
        detail: {
          code: AGENT_ACCESS_AUTH_REQUIRED_CODE,
          login_url: typeof loginUrl === 'string' ? loginUrl : null,
        },
      }))
    } catch {
      // A malformed JSON error is not an owner-session expiry signal.
    }
    return response
  }

  installedFetch = intercepted
  holder.__maestroAgentAccessNativeFetch = original
  holder.__maestroAgentAccessInterceptedFetch = intercepted
  window.fetch = intercepted
}

function isJsonResponse(response: Response): boolean {
  const contentType = response.headers.get('content-type')?.toLowerCase() || ''
  return contentType.includes('application/json') || contentType.includes('+json')
}

async function requestJson<T>(
  path: string,
  init: RequestInit = {},
  useNativeFetch = false,
): Promise<T> {
  const fetcher = useNativeFetch
    ? (nativeFetch || (typeof window !== 'undefined' ? window.fetch.bind(window) : fetch))
    : fetch
  const response = await fetcher(path, {
    credentials: 'same-origin',
    ...init,
  })
  const payload = await response.clone().json().catch(() => null) as Record<string, unknown> | null
  if (!response.ok) {
    const code = typeof payload?.code === 'string' ? payload.code : null
    const message = typeof payload?.message === 'string'
      ? payload.message
      : typeof payload?.detail === 'string'
        ? payload.detail
        : `Agent Access request failed (${response.status})`
    throw new AgentAccessApiError(response.status, message, code)
  }
  return (payload ?? {}) as T
}

export function fetchAgentAccessStatus(): Promise<AgentAccessStatus> {
  return requestJson('/api/v1/agent-access/status')
}

export function requestAgentAccessConsoleCode(): Promise<{ message?: string; expires_in: number }> {
  return requestJson('/api/v1/agent-access/request-console-code', { method: 'POST' })
}

export async function loginAgentAccessOwner(code: string): Promise<{ ok: boolean }> {
  const result = await requestJson<{ ok: boolean }>('/api/v1/agent-access/owner-login', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ code }),
  }, true)
  const session = await requestJson<AgentAccessStatus>('/api/v1/agent-access/status', {}, true)
  if (!result.ok || !session.authenticated || !session.owner) {
    throw new AgentAccessApiError(401, 'The code was accepted, but this browser did not keep the sign-in session. Request a fresh code on this page and sign in again; each code can be used only once.', 'maestro_owner_session_missing')
  }
  return result
}

export function updateAgentAccessSettings(settings: { enabled?: boolean; label?: string }): Promise<Record<string, unknown>> {
  return requestJson('/api/v1/agent-access/settings', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(settings),
  })
}

export async function fetchAgentAccessGrants(): Promise<AgentAccessGrant[]> {
  const result = await requestJson<{ grants?: AgentAccessGrant[] }>('/api/v1/agent-access/grants')
  return Array.isArray(result.grants) ? result.grants : []
}

export function revokeAgentAccessGrant(grantId: string): Promise<Record<string, unknown>> {
  return requestJson(`/api/v1/agent-access/grants/${encodeURIComponent(grantId)}`, { method: 'DELETE' })
}

export function createAgentAccessBrowserCode(): Promise<{ code: string; expires_in: number }> {
  return requestJson('/api/v1/agent-access/browser-code', { method: 'POST' })
}

export function logoutAgentAccessOwner(): Promise<Record<string, unknown>> {
  return requestJson('/api/v1/agent-access/logout', { method: 'POST' })
}

export async function fetchAgentAccessAudit(): Promise<AgentAccessAuditEvent[]> {
  const result = await requestJson<{ events?: AgentAccessAuditEvent[] }>('/api/v1/agent-access/audit')
  return Array.isArray(result.events) ? result.events : []
}
