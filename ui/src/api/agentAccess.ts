export const AGENT_ACCESS_AUTH_REQUIRED_EVENT = 'maestro-agent-access-auth-required'
export const AGENT_ACCESS_AUTH_REQUIRED_CODE = 'maestro_auth_required'
export const AGENT_ACCESS_SESSION_RESTORED_EVENT = 'maestro-agent-access-session-restored'

export interface AgentAccessStatus {
  available: boolean
  /** Legacy alias for remote_enabled. */
  enabled: boolean
  local_enabled?: boolean
  remote_enabled?: boolean
  owner_configured: boolean
  authenticated: boolean
  owner?: boolean
  local_approval_available?: boolean
  auth_method?: string | null
  installation_id?: string | null
  workstation_id?: string | null
  label?: string | null
  controller_url?: string | null
  local_setup_url?: string | null
  mcp_url?: string | null
  lan_mcp_urls?: string[]
  preferred_mcp_url?: string | null
  tailscale?: AgentAccessTailscaleStatus | null
  pinokio_remote?: AgentAccessPinokioRemoteStatus | null
  certificate_sha256?: string | null
  controller_status?: string | null
  discovery_status?: string | null
  backend_online?: boolean
}

export interface AgentAccessTailscaleStatus {
  enabled: boolean
  installed: boolean
  connected: boolean
  hostname: string | null
  certificate_ready: boolean
  ready: boolean
  mcp_url: string | null
  error: string | null
  expires_at: string | null
}

/** Read-only details about the browser address published by Pinokio Remote. */
export interface PinokioRemoteAccessStatus {
  available: boolean
  phase: string
  browser_url: string | null
  node_hostname: string | null
  dns_name: string | null
  app_id: string | null
  reason: string | null
}

/** Owner-managed MCP access layered on the public Pinokio Remote detection. */
export interface AgentAccessPinokioRemoteStatus extends PinokioRemoteAccessStatus {
  enabled: boolean
  ready: boolean
  mcp_url: string | null
}

export interface AgentAccessGrant {
  grant_id: string
  client_id: string
  client_name?: string
  scopes: string[]
  created_at?: number | string | null
  revoked?: boolean
  revoked_at?: number | string | null
  transport?: 'local' | 'remote' | string
}

export type AgentAccessScope = 'observe' | 'generate' | 'manage' | 'admin'

export interface AgentAccessLocalClient {
  id: string
  name: string
  scopes: AgentAccessScope[]
  grant_id?: string | null
  revoked?: boolean
  created_at?: number | string | null
  /** Safe, non-secret launch information. Never contains credentials. */
  command?: string | null
  args?: string[] | null
}

export interface AgentAccessCreativeWorkflow {
  id: string
  label: string
  model_type?: string | null
  [key: string]: unknown
}

export interface AgentAccessCreativeModel {
  model_type: string
  name: string
  workflows: string[]
  is_downloaded: boolean
}

export interface AgentAccessCreativeDefault {
  model_type?: string
  duration_seconds?: number | null
  resolution?: string | null
}

export interface AgentAccessCreativeDefaults {
  defaults: Record<string, AgentAccessCreativeDefault>
  workflows: AgentAccessCreativeWorkflow[]
  models: AgentAccessCreativeModel[]
  version: 1
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

export async function fetchAgentAccessStatus(): Promise<AgentAccessStatus> {
  const status = await requestJson<AgentAccessStatus>('/api/v1/agent-access/status')
  if (status.pinokio_remote !== undefined && status.pinokio_remote !== null) {
    status.pinokio_remote = safeAgentAccessPinokioRemoteStatus(status.pinokio_remote)
  }
  return status
}

function nullableText(record: Record<string, unknown>, key: string): string | null {
  const value = record[key]
  if (value === null) return null
  if (value === undefined) {
    throw new AgentAccessApiError(502, `Maestro omitted the Pinokio Remote ${key} value.`)
  }
  if (typeof value !== 'string') {
    throw new AgentAccessApiError(502, `Maestro returned an invalid Pinokio Remote ${key} value.`)
  }
  return value
}

function safePinokioRemoteAccessStatus(value: unknown): PinokioRemoteAccessStatus {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw new AgentAccessApiError(502, 'Maestro returned an invalid Pinokio Remote status.')
  }
  const record = value as Record<string, unknown>
  if (typeof record.available !== 'boolean' || typeof record.phase !== 'string') {
    throw new AgentAccessApiError(502, 'Maestro returned an incomplete Pinokio Remote status.')
  }
  return {
    available: record.available,
    phase: record.phase,
    browser_url: nullableText(record, 'browser_url'),
    node_hostname: nullableText(record, 'node_hostname'),
    dns_name: nullableText(record, 'dns_name'),
    app_id: nullableText(record, 'app_id'),
    reason: nullableText(record, 'reason'),
  }
}

function safeAgentAccessPinokioRemoteStatus(value: unknown): AgentAccessPinokioRemoteStatus {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw new AgentAccessApiError(502, 'Maestro returned an invalid Pinokio Remote MCP status.')
  }
  const record = value as Record<string, unknown>
  const detected = safePinokioRemoteAccessStatus(record)
  if (typeof record.enabled !== 'boolean' || typeof record.ready !== 'boolean') {
    throw new AgentAccessApiError(502, 'Maestro returned an incomplete Pinokio Remote MCP status.')
  }
  return { ...detected, enabled: record.enabled, ready: record.ready, mcp_url: nullableText(record, 'mcp_url') }
}

/** Older Maestro versions do not have this endpoint; treat only 404 as unsupported. */
export async function fetchPinokioRemoteAccessStatus(): Promise<PinokioRemoteAccessStatus | null> {
  try {
    const result = await requestJson<unknown>('/api/v1/remote-access/pinokio/status', { credentials: 'omit' })
    return safePinokioRemoteAccessStatus(result)
  } catch (error) {
    if (error instanceof AgentAccessApiError && error.status === 404) return null
    throw error
  }
}

export function updateAgentAccessPinokioRemoteSettings(settings: { enabled: boolean }): Promise<AgentAccessPinokioRemoteStatus> {
  return requestJson<unknown>('/api/v1/agent-access/pinokio', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(settings),
  }).then(safeAgentAccessPinokioRemoteStatus)
}

export function fetchAgentAccessTailscaleStatus(): Promise<AgentAccessTailscaleStatus> {
  return requestJson('/api/v1/agent-access/tailscale')
}

export function updateAgentAccessTailscaleSettings(settings: { enabled: boolean }): Promise<AgentAccessTailscaleStatus> {
  return requestJson('/api/v1/agent-access/tailscale', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(settings),
  })
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

export async function approveAgentAccessLocal(): Promise<{ ok: boolean }> {
  if (!globalThis.crypto?.getRandomValues) {
    throw new AgentAccessApiError(503, 'This browser cannot create a local approval proof. Open Maestro in a current browser on this PC.')
  }
  const randomness = globalThis.crypto.getRandomValues(new Uint8Array(32))
  const proof = Array.from(randomness, value => value.toString(16).padStart(2, '0')).join('')
  const challenge = await requestJson<{ nonce?: string }>('/api/v1/agent-access/local-challenge', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ proof }),
  }, true)
  if (typeof challenge.nonce !== 'string' || !challenge.nonce || challenge.nonce.length > 256) {
    throw new AgentAccessApiError(502, 'Local approval is not ready. Try Approve on this PC again.')
  }
  const result = await requestJson<{ ok: boolean }>('/api/v1/agent-access/local-approve', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ nonce: challenge.nonce, proof, approve: true }),
  }, true)
  const session = await requestJson<AgentAccessStatus>('/api/v1/agent-access/status', {}, true)
  if (!result.ok || !session.authenticated || !session.owner) {
    throw new AgentAccessApiError(401, 'Local approval could not be verified. Try Approve on this PC again, or refresh after Maestro has finished starting.', 'maestro_owner_session_missing')
  }
  return result
}

export function updateAgentAccessSettings(settings: { enabled?: boolean; local_enabled?: boolean; remote_enabled?: boolean; label?: string }): Promise<Record<string, unknown>> {
  return requestJson('/api/v1/agent-access/settings', {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(settings),
  })
}

const AGENT_ACCESS_SCOPES = new Set<AgentAccessScope>(['observe', 'generate', 'manage', 'admin'])

function safeLocalClient(value: unknown): AgentAccessLocalClient {
  if (!value || typeof value !== 'object') {
    throw new AgentAccessApiError(502, 'Maestro returned an invalid local MCP client record.')
  }
  const record = value as Record<string, unknown>
  if (typeof record.id !== 'string' || typeof record.name !== 'string' || !Array.isArray(record.scopes)) {
    throw new AgentAccessApiError(502, 'Maestro returned an incomplete local MCP client record.')
  }
  const scopes = record.scopes.filter((scope): scope is AgentAccessScope => typeof scope === 'string' && AGENT_ACCESS_SCOPES.has(scope as AgentAccessScope))
  const args = Array.isArray(record.args) && record.args.every(arg => typeof arg === 'string')
    ? record.args as string[]
    : null
  return {
    id: record.id,
    name: record.name,
    scopes,
    grant_id: typeof record.grant_id === 'string' ? record.grant_id : null,
    revoked: record.revoked === true,
    created_at: typeof record.created_at === 'string' || typeof record.created_at === 'number' ? record.created_at : null,
    command: typeof record.command === 'string' ? record.command : null,
    args,
  }
}

export async function fetchAgentAccessLocalClients(): Promise<AgentAccessLocalClient[]> {
  const result = await requestJson<{ clients?: unknown }>('/api/v1/agent-access/local-clients')
  if (!Array.isArray(result.clients)) return []
  return result.clients.map(safeLocalClient)
}

export async function createAgentAccessLocalClient(input: { name: string; scopes: AgentAccessScope[] }): Promise<AgentAccessLocalClient> {
  const name = input.name.trim()
  const scopes = [...new Set(input.scopes.filter(scope => AGENT_ACCESS_SCOPES.has(scope))) ]
  if (!name || !scopes.length) {
    throw new AgentAccessApiError(400, 'Enter a client name and choose at least one access scope.')
  }
  const result = await requestJson<unknown>('/api/v1/agent-access/local-clients', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name, scopes }),
  })
  return safeLocalClient(result)
}

export async function fetchAgentAccessLocalClientExtension(clientId: string): Promise<Blob> {
  if (!clientId || clientId.length > 128 || /[\\/]/.test(clientId) || hasControlCharacter(clientId)) {
    throw new AgentAccessApiError(400, 'The local client ID is invalid.')
  }
  const response = await fetch(`/api/v1/agent-access/local-clients/${encodeURIComponent(clientId)}/extension`, {
    credentials: 'same-origin',
  })
  if (!response.ok) {
    const payload = await response.clone().json().catch(() => null) as Record<string, unknown> | null
    const code = typeof payload?.code === 'string' ? payload.code : null
    const message = typeof payload?.message === 'string'
      ? payload.message
      : typeof payload?.detail === 'string'
        ? payload.detail
        : `Agent Access request failed (${response.status})`
    throw new AgentAccessApiError(response.status, message, code)
  }
  const contentType = response.headers.get('content-type')?.split(';', 1)[0].trim().toLowerCase()
  if (contentType !== 'application/octet-stream') {
    throw new AgentAccessApiError(502, 'Maestro returned an invalid Claude Desktop extension package.')
  }
  return response.blob()
}

function hasControlCharacter(value: string): boolean {
  return Array.from(value).some(character => character.charCodeAt(0) < 0x20 || character.charCodeAt(0) === 0x7f)
}

function localClientServerName(id: string): string {
  return `maestro-${id.replace(/[^a-zA-Z0-9]/g, '').toLowerCase() || 'local'}`
}

export function formatLocalAgentClientJson(client: AgentAccessLocalClient): string {
  const command = typeof client.command === 'string' ? client.command : ''
  const args = Array.isArray(client.args) ? client.args.filter((value): value is string => typeof value === 'string') : []
  return JSON.stringify({
    mcpServers: {
      [localClientServerName(client.id)]: { command, args },
    },
  }, null, 2)
}

export function formatLocalAgentClientCodexConfig(client: AgentAccessLocalClient): string {
  const command = typeof client.command === 'string' ? client.command : ''
  const args = Array.isArray(client.args) ? client.args.filter((value): value is string => typeof value === 'string') : []
  const name = localClientServerName(client.id)
  return [
    `[mcp_servers.${name}]`,
    `command = ${JSON.stringify(command)}`,
    `args = [${args.map(value => JSON.stringify(value)).join(', ')}]`,
  ].join('\n')
}

export function formatLocalAgentClientClaudeCommand(client: AgentAccessLocalClient): string {
  const command = typeof client.command === 'string' ? client.command : ''
  const args = Array.isArray(client.args) ? client.args.filter((value): value is string => typeof value === 'string') : []
  const quote = isWindowsCommand(command) ? quotePowerShellArgument : quotePosixArgument
  const invocation = ['claude', 'mcp', 'add', '--transport', 'stdio', localClientServerName(client.id), '--', command, ...args]
    .map(quote)
    .join(' ')
  return isWindowsCommand(command) ? `& ${invocation}` : invocation
}

function isWindowsCommand(command: string): boolean {
  return /^[a-zA-Z]:[\\/]/.test(command) || command.startsWith('\\\\')
}

function quotePowerShellArgument(value: string): string {
  return `'${value.replace(/'/g, "''")}'`
}

function quotePosixArgument(value: string): string {
  return "'" + value.replace(/'/g, "'\\''") + "'"
}

export function fetchAgentAccessCreativeDefaults(): Promise<AgentAccessCreativeDefaults> {
  return requestJson('/api/v1/agent-access/creative-defaults')
}

export function updateAgentAccessCreativeDefaults(defaults: AgentAccessCreativeDefaults['defaults']): Promise<AgentAccessCreativeDefaults> {
  return requestJson('/api/v1/agent-access/creative-defaults', {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ defaults }),
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
