import { useCallback, useEffect, useMemo, useState, type FormEvent, type ReactNode } from 'react'
import {
  Activity,
  Check,
  CircleAlert,
  Copy,
  ExternalLink,
  Fingerprint,
  Link2,
  LoaderCircle,
  LogIn,
  LogOut,
  RefreshCw,
  Shield,
  ShieldCheck,
  Trash2,
} from 'lucide-react'
import {
  AGENT_ACCESS_AUTH_REQUIRED_EVENT,
  AGENT_ACCESS_SESSION_RESTORED_EVENT,
  AgentAccessApiError,
  createAgentAccessBrowserCode,
  fetchAgentAccessAudit,
  fetchAgentAccessGrants,
  fetchAgentAccessStatus,
  loginAgentAccessOwner,
  logoutAgentAccessOwner,
  requestAgentAccessConsoleCode,
  revokeAgentAccessGrant,
  updateAgentAccessSettings,
  type AgentAccessAuditEvent,
  type AgentAccessGrant,
  type AgentAccessStatus,
} from '../../api/agentAccess'

type CopyTarget = 'codex' | 'claude' | 'url' | null

export function AgentAccessSettingsPanel() {
  const [status, setStatus] = useState<AgentAccessStatus | null>(null)
  const [grants, setGrants] = useState<AgentAccessGrant[]>([])
  const [auditEvents, setAuditEvents] = useState<AgentAccessAuditEvent[]>([])
  const [loading, setLoading] = useState(true)
  const [backendMissing, setBackendMissing] = useState(false)
  const [busyAction, setBusyAction] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [message, setMessage] = useState<string | null>(null)
  const [consoleCodeRequested, setConsoleCodeRequested] = useState(false)
  const [ownerCode, setOwnerCode] = useState('')
  const [labelDraft, setLabelDraft] = useState('')
  const [copyTarget, setCopyTarget] = useState<CopyTarget>(null)
  const [pairingCode, setPairingCode] = useState<string | null>(null)
  const [pairingExpiresAt, setPairingExpiresAt] = useState<number | null>(null)
  const [pairingSecondsLeft, setPairingSecondsLeft] = useState(0)
  const [showAudit, setShowAudit] = useState(false)
  const [selectedMcpUrl, setSelectedMcpUrl] = useState('')

  const ownerSession = status?.authenticated === true && status.owner === true
  const secureOrigin = isSecureOrigin()
  const mcpUrls = useMemo(() => [...new Set([status?.mcp_url, ...(status?.lan_mcp_urls || [])]
    .filter((value): value is string => typeof value === 'string' && value.endsWith('/mcp') && safeControllerUrl(value) !== null))], [status])
  const mcpUrl = mcpUrls.includes(selectedMcpUrl) ? selectedMcpUrl : status?.mcp_url || ''
  const controllerUrl = safeControllerUrl(mcpUrl.endsWith('/mcp') ? mcpUrl.slice(0, -4) : status?.controller_url)

  const refresh = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const next = await fetchAgentAccessStatus()
      setStatus(next)
      setBackendMissing(false)
      setLabelDraft(next.label || '')
      if (next.authenticated && next.owner) {
        const [grantResult, auditResult] = await Promise.allSettled([
          fetchAgentAccessGrants(),
          fetchAgentAccessAudit(),
        ])
        if (grantResult.status === 'fulfilled') setGrants(grantResult.value)
        else setGrants([])
        if (auditResult.status === 'fulfilled') setAuditEvents(auditResult.value)
        else setAuditEvents([])
      } else {
        setGrants([])
        setAuditEvents([])
        setPairingCode(null)
        setPairingExpiresAt(null)
      }
    } catch (requestError) {
      setStatus(null)
      setGrants([])
      setAuditEvents([])
      if (requestError instanceof AgentAccessApiError && requestError.status === 404) {
        setBackendMissing(true)
      } else {
        setError(requestError instanceof Error ? requestError.message : 'Could not load Agent Access status.')
      }
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    void refresh()
    const onSessionRestored = () => { void refresh() }
    const onAuthRequired = () => { void refresh() }
    window.addEventListener(AGENT_ACCESS_SESSION_RESTORED_EVENT, onSessionRestored)
    window.addEventListener(AGENT_ACCESS_AUTH_REQUIRED_EVENT, onAuthRequired)
    return () => {
      window.removeEventListener(AGENT_ACCESS_SESSION_RESTORED_EVENT, onSessionRestored)
      window.removeEventListener(AGENT_ACCESS_AUTH_REQUIRED_EVENT, onAuthRequired)
    }
  }, [refresh])

  useEffect(() => {
    if (!pairingExpiresAt) return
    const updateCountdown = () => {
      const seconds = Math.max(0, Math.ceil((pairingExpiresAt - Date.now()) / 1000))
      setPairingSecondsLeft(seconds)
      if (seconds === 0) {
        setPairingCode(null)
        setPairingExpiresAt(null)
      }
    }
    updateCountdown()
    const timer = window.setInterval(updateCountdown, 1000)
    return () => window.clearInterval(timer)
  }, [pairingExpiresAt])

  const handleRequestConsoleCode = async () => {
    setBusyAction('console-code')
    setError(null)
    setMessage(null)
    try {
      const result = await requestAgentAccessConsoleCode()
      setConsoleCodeRequested(true)
      const ttl = Number.isFinite(result.expires_in) ? Math.max(1, result.expires_in) : 600
      const minutes = Math.max(1, Math.ceil(ttl / 60))
      setMessage(`A one-time owner code was printed to the Pinokio terminal. It expires in about ${minutes} minutes.`)
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : 'Could not request an owner code.')
    } finally {
      setBusyAction(null)
    }
  }

  const handleOwnerLogin = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    if (!secureOrigin || !ownerCode.trim()) return
    setBusyAction('owner-login')
    setError(null)
    setMessage(null)
    try {
      await loginAgentAccessOwner(ownerCode.trim())
      setOwnerCode('')
      setConsoleCodeRequested(false)
      setMessage('Owner session is active in this browser.')
      await refresh()
    } catch (loginError) {
      if (loginError instanceof AgentAccessApiError && loginError.status === 404) {
        setBackendMissing(true)
      } else {
        setError(loginError instanceof Error ? loginError.message : 'Owner sign-in failed.')
      }
    } finally {
      setBusyAction(null)
    }
  }

  const handleToggle = async (enabled: boolean) => {
    setBusyAction('enabled')
    setError(null)
    setMessage(null)
    try {
      await updateAgentAccessSettings({ enabled })
      setMessage(enabled ? 'Agent Access is enabled.' : 'Agent Access is disabled.')
      await refresh()
    } catch (updateError) {
      setError(updateError instanceof Error ? updateError.message : 'Could not update Agent Access.')
    } finally {
      setBusyAction(null)
    }
  }

  const handleSaveLabel = async () => {
    setBusyAction('label')
    setError(null)
    setMessage(null)
    try {
      await updateAgentAccessSettings({ label: labelDraft.trim() })
      setMessage('Installation name saved.')
      await refresh()
    } catch (updateError) {
      setError(updateError instanceof Error ? updateError.message : 'Could not save the installation name.')
    } finally {
      setBusyAction(null)
    }
  }

  const handleRevoke = async (grant: AgentAccessGrant) => {
    setBusyAction(`revoke:${grant.grant_id}`)
    setError(null)
    setMessage(null)
    try {
      await revokeAgentAccessGrant(grant.grant_id)
      setMessage(`${grant.client_name || 'Agent'} access was revoked.`)
      await refresh()
    } catch (revokeError) {
      setError(revokeError instanceof Error ? revokeError.message : 'Could not revoke this grant.')
    } finally {
      setBusyAction(null)
    }
  }

  const handlePairingCode = async () => {
    setBusyAction('browser-code')
    setError(null)
    setMessage(null)
    try {
      const result = await createAgentAccessBrowserCode()
      setPairingCode(result.code)
      const ttl = Number.isFinite(result.expires_in) ? Math.max(1, result.expires_in) : 300
      setPairingExpiresAt(Date.now() + ttl * 1000)
      setMessage('Use this short-lived code in the Maestro browser flow. It is kept only in this page until it expires.')
    } catch (pairError) {
      setError(pairError instanceof Error ? pairError.message : 'Could not create a browser pairing code.')
    } finally {
      setBusyAction(null)
    }
  }

  const handleLogout = async () => {
    setBusyAction('logout')
    setError(null)
    setMessage(null)
    try {
      await logoutAgentAccessOwner()
      setOwnerCode('')
      setPairingCode(null)
      setPairingExpiresAt(null)
      setMessage('This browser has been signed out.')
      await refresh()
    } catch (logoutError) {
      setError(logoutError instanceof Error ? logoutError.message : 'Could not sign out this browser.')
    } finally {
      setBusyAction(null)
    }
  }

  const handleCopy = async (target: Exclude<CopyTarget, null>) => {
    if (!mcpUrl) return
    setError(null)
    try {
      await navigator.clipboard.writeText(clientSetupText(target, mcpUrl, status?.certificate_sha256 || ''))
      setCopyTarget(target)
      window.setTimeout(() => setCopyTarget(current => current === target ? null : current), 1800)
    } catch {
      setError('Clipboard access is unavailable in this browser. Select the connection URL and copy it manually.')
    }
  }

  const sortedGrants = useMemo(
    () => [...grants].sort((a, b) => Number(Boolean(a.revoked)) - Number(Boolean(b.revoked))),
    [grants],
  )

  return (
    <div className="space-y-5 pb-5">
      <section className="rounded-xl border border-border bg-bg-secondary p-4">
        <div className="flex items-start gap-3">
          <div className="mt-0.5 rounded-lg bg-accent-blue/10 p-2 text-accent-blue"><Shield size={17} /></div>
          <div className="min-w-0 flex-1">
            <div className="flex flex-wrap items-center gap-2">
              <h3 className="text-sm font-semibold text-text-primary">Maestro Agent Access</h3>
              <StatusPill label={status ? (status.enabled ? 'Enabled' : 'Off') : backendMissing ? 'Restart needed' : 'Checking'} good={status?.enabled === true} />
            </div>
            <p className="mt-1.5 text-xs leading-relaxed text-text-secondary">
              Connect Codex, Claude, and other MCP clients with scoped grants you can review and revoke. MCP and discovery start off. Owner setup also protects Maestro's existing API; turning MCP off later keeps that protection.
            </p>
          </div>
          <button type="button" onClick={() => void refresh()} disabled={loading} className="rounded-lg p-2 text-text-muted hover:bg-bg-hover hover:text-text-primary disabled:opacity-40" aria-label="Refresh Agent Access status">
            <RefreshCw size={15} className={loading ? 'animate-spin' : ''} />
          </button>
        </div>

        {backendMissing ? (
          <div className="mt-4 flex gap-2 rounded-lg border border-amber-500/30 bg-amber-500/10 p-3 text-xs leading-relaxed text-text-secondary" role="status">
            <CircleAlert size={15} className="mt-0.5 shrink-0 text-amber-400" />
            <span>This Maestro backend does not have Agent Access yet. Restart Maestro after installing the update, then refresh this panel.</span>
          </div>
        ) : status && !status.available ? (
          <div className="mt-4 flex gap-2 rounded-lg border border-amber-500/30 bg-amber-500/10 p-3 text-xs leading-relaxed text-text-secondary" role="status">
            <CircleAlert size={15} className="mt-0.5 shrink-0 text-amber-400" />
            <span>Agent Access is unavailable on this installation.</span>
          </div>
        ) : null}

        <div className="mt-4 grid grid-cols-1 gap-2 sm:grid-cols-2">
          <InfoChip label="Controller" value={status?.controller_status || 'Checking'} good={isHealthyStatus(status?.controller_status)} />
          <InfoChip label="Backend" value={status?.backend_online === undefined ? 'Checking' : status.backend_online ? 'Online' : 'Offline'} good={status?.backend_online === true} />
          <InfoChip label="Discovery" value={status?.discovery_status || 'Not reported'} good={isHealthyStatus(status?.discovery_status)} />
        </div>
      </section>

      {error && <Notice kind="error">{error}</Notice>}
      {message && <Notice kind="success">{message}</Notice>}

      {!backendMissing && status && (
        <>
          <section className="space-y-3 rounded-xl border border-border bg-bg-secondary p-4">
            <div className="flex items-center justify-between gap-3">
              <div>
                <h4 className="text-xs font-semibold text-text-primary">Owner setup</h4>
                <p className="mt-1 text-[10px] leading-relaxed text-text-muted">
                  {ownerSession
                    ? 'This browser has an owner session.'
                    : status.owner_configured
                      ? 'Owner access is configured. Sign in to manage this installation.'
                      : 'Request a one-time code from the local Pinokio console to set up owner access.'}
                </p>
              </div>
              {ownerSession && (
                <button type="button" onClick={() => void handleLogout()} disabled={busyAction !== null} className="inline-flex min-h-9 items-center gap-1.5 rounded-lg border border-border px-2.5 text-[10px] text-text-secondary hover:bg-bg-hover disabled:opacity-50">
                  {busyAction === 'logout' ? <LoaderCircle size={13} className="animate-spin" /> : <LogOut size={13} />}
                  Sign out
                </button>
              )}
            </div>

            {!ownerSession && (
              <>
                <button
                  type="button"
                  onClick={() => void handleRequestConsoleCode()}
                  disabled={!secureOrigin || busyAction !== null || !status.available}
                  className="inline-flex min-h-10 w-full items-center justify-center gap-2 rounded-lg border border-border px-3 py-2 text-xs font-medium text-text-primary transition-colors hover:bg-bg-hover disabled:cursor-not-allowed disabled:opacity-40"
                >
                  {busyAction === 'console-code' ? <LoaderCircle size={14} className="animate-spin" /> : <LogIn size={14} />}
                  Request one-time console code
                </button>
                {!secureOrigin && <SecureOriginNote href={controllerUrl} />}
                {consoleCodeRequested && (
                  <p className="rounded-lg bg-bg-primary p-3 text-[10px] leading-relaxed text-text-secondary">
                    Check the Pinokio terminal for the code, then paste it below. The code is never shown in this page.
                  </p>
                )}
                {secureOrigin && (
                  <form className="space-y-2" onSubmit={event => void handleOwnerLogin(event)}>
                    <label htmlFor="agent-access-owner-code" className="block text-[10px] text-text-secondary">One-time owner code</label>
                    <div className="flex gap-2">
                      <input
                        id="agent-access-owner-code"
                        value={ownerCode}
                        onChange={event => setOwnerCode(event.target.value)}
                        autoComplete="one-time-code"
                        spellCheck={false}
                        placeholder="Paste code from Pinokio"
                        className="min-w-0 flex-1 rounded-lg border border-border bg-bg-primary px-3 py-2 font-mono text-xs text-text-primary outline-none focus:border-accent-blue"
                        disabled={busyAction !== null}
                      />
                      <button type="submit" disabled={!ownerCode.trim() || busyAction !== null} className="min-h-9 rounded-lg bg-accent-blue px-3 text-[10px] font-medium text-white hover:opacity-90 disabled:opacity-40">
                        {busyAction === 'owner-login' ? 'Signing in…' : 'Sign in'}
                      </button>
                    </div>
                  </form>
                )}
              </>
            )}

            {ownerSession && (
              <div className="space-y-3 border-t border-border pt-3">
                <ToggleRow
                  checked={status.enabled}
                  disabled={busyAction !== null || !status.available}
                  label="Allow agent connections"
                  description="Turning this off pauses agent requests without revoking grants or tokens. Owner sign-in stays available for settings and revocation."
                  onChange={value => void handleToggle(value)}
                />
                <div className="rounded-lg bg-bg-primary p-3 text-[10px] leading-relaxed text-text-muted">
                  Existing agent grants and tokens remain in place while paused. Revoke a grant below to remove that client’s access; browser owner sign-in remains protected.
                </div>
              </div>
            )}
          </section>

          <section className="space-y-3 rounded-xl border border-border bg-bg-secondary p-4">
            <div className="flex items-center gap-2">
              <Fingerprint size={15} className="text-text-muted" />
              <h4 className="text-xs font-semibold text-text-primary">Connection details</h4>
            </div>
            {status.label !== undefined && (
              <div className="space-y-1.5">
                <label htmlFor="agent-access-label" className="text-[10px] text-text-secondary">Installation name</label>
                <div className="flex gap-2">
                  <input
                    id="agent-access-label"
                    value={labelDraft}
                    maxLength={64}
                    onChange={event => setLabelDraft(event.target.value)}
                    disabled={!ownerSession || busyAction !== null}
                    className="min-w-0 flex-1 rounded-lg border border-border bg-bg-primary px-3 py-2 text-xs text-text-primary outline-none focus:border-accent-blue disabled:opacity-50"
                    placeholder="My Maestro"
                  />
                  <button type="button" onClick={() => void handleSaveLabel()} disabled={!ownerSession || busyAction !== null || labelDraft.trim() === (status.label || '')} className="rounded-lg border border-border px-3 text-[10px] text-text-secondary hover:bg-bg-hover disabled:opacity-40">Save</button>
                </div>
              </div>
            )}
            <InfoLine label="Installation ID" value={status.installation_id || 'Not reported'} mono />
            <InfoLine label="Secure controller" value={controllerUrl || 'Not available'} href={controllerUrl || undefined} />
            <InfoLine label="Setup URL (on this workstation)" value={status.local_setup_url || 'Not available'} href={safeLocalUrl(status.local_setup_url) || undefined} />
            {mcpUrls.length > 1 && (
              <label className="block space-y-1 text-[10px] text-text-secondary">
                <span>Connection address</span>
                <select aria-label="MCP connection address" value={mcpUrl} onChange={event => { setSelectedMcpUrl(event.target.value); setCopyTarget(null) }} className="w-full rounded-lg border border-border bg-bg-primary px-2 py-2 text-[10px]">
                  {mcpUrls.map(url => <option key={url} value={url}>{url === status.mcp_url ? 'Discoverable hostname' : 'Direct LAN address'} — {url}</option>)}
                </select>
                <span className="block text-text-muted">Use a direct LAN address if your network blocks discovery. Trust the same verified certificate.</span>
              </label>
            )}
            <InfoLine label="MCP connection URL" value={mcpUrl || 'Not available'} mono copyable={!!mcpUrl} onCopy={() => void handleCopy('url')} copied={copyTarget === 'url'} />
            <InfoLine label="TLS certificate SHA-256" value={status.certificate_sha256 || 'Not reported'} mono />
            {status.certificate_sha256 && (
              <a href="/api/v1/agent-access/certificate" download="maestro-agent-access.pem" className="inline-flex min-h-9 items-center gap-2 text-[10px] text-accent-blue hover:underline">Download public TLS certificate</a>
            )}
            <div className="grid grid-cols-1 gap-2 pt-1 sm:grid-cols-2">
              <button type="button" onClick={() => void handleCopy('codex')} disabled={!mcpUrl} className="inline-flex min-h-10 items-center justify-center gap-2 rounded-lg border border-border px-3 text-[10px] text-text-secondary hover:bg-bg-hover disabled:opacity-40">
                {copyTarget === 'codex' ? <Check size={13} /> : <Copy size={13} />}
                {copyTarget === 'codex' ? 'Codex setup copied' : 'Copy Codex MCP setup'}
              </button>
              <button type="button" onClick={() => void handleCopy('claude')} disabled={!mcpUrl} className="inline-flex min-h-10 items-center justify-center gap-2 rounded-lg border border-border px-3 text-[10px] text-text-secondary hover:bg-bg-hover disabled:opacity-40">
                {copyTarget === 'claude' ? <Check size={13} /> : <Copy size={13} />}
                {copyTarget === 'claude' ? 'Claude setup copied' : 'Copy Claude MCP setup'}
              </button>
            </div>
            <p className="text-[10px] leading-relaxed text-text-muted">
              Add the URL once in each client’s MCP settings. These instructions contain the endpoint and fingerprint only; they never contain an access token. Approve the client from Maestro when it requests access.
            </p>
          </section>

          {ownerSession && (
            <>
              <section className="space-y-3 rounded-xl border border-border bg-bg-secondary p-4">
                <div className="flex items-start justify-between gap-3">
                  <div>
                    <h4 className="text-xs font-semibold text-text-primary">Browser pairing</h4>
                    <p className="mt-1 text-[10px] leading-relaxed text-text-muted">Create a short-lived code for the Maestro browser pairing flow.</p>
                  </div>
                  <button type="button" onClick={() => void handlePairingCode()} disabled={busyAction !== null} className="inline-flex min-h-9 shrink-0 items-center gap-1.5 rounded-lg border border-border px-2.5 text-[10px] text-text-secondary hover:bg-bg-hover disabled:opacity-50">
                    {busyAction === 'browser-code' ? <LoaderCircle size={13} className="animate-spin" /> : <Link2 size={13} />}
                    New code
                  </button>
                </div>
                {pairingCode && (
                  <div className="flex items-center justify-between gap-3 rounded-lg border border-border bg-bg-primary p-3">
                    <code className="break-all font-mono text-sm tracking-wider text-text-primary">{pairingCode}</code>
                    <span className="shrink-0 text-[10px] text-text-muted">{pairingSecondsLeft}s</span>
                  </div>
                )}
              </section>

              <section className="space-y-3 rounded-xl border border-border bg-bg-secondary p-4">
                <div className="flex items-center justify-between gap-2">
                  <div className="flex items-center gap-2">
                    <ShieldCheck size={15} className="text-text-muted" />
                    <h4 className="text-xs font-semibold text-text-primary">Agent grants</h4>
                  </div>
                  <span className="rounded-full bg-bg-primary px-2 py-1 text-[10px] text-text-muted">{grants.filter(grant => !grant.revoked).length} active</span>
                </div>
                {grants.length === 0 ? (
                  <p className="rounded-lg bg-bg-primary p-3 text-[10px] text-text-muted">No agents have been approved yet.</p>
                ) : (
                  <div className="space-y-2">
                    {sortedGrants.map(grant => (
                      <GrantRow
                        key={grant.grant_id}
                        grant={grant}
                        busy={busyAction === `revoke:${grant.grant_id}`}
                        disabled={busyAction !== null}
                        onRevoke={() => void handleRevoke(grant)}
                      />
                    ))}
                  </div>
                )}
              </section>

              <section className="rounded-xl border border-border bg-bg-secondary">
                <button type="button" aria-expanded={showAudit} onClick={() => setShowAudit(value => !value)} className="flex min-h-12 w-full items-center justify-between gap-3 px-4 text-left">
                  <span className="flex items-center gap-2 text-xs font-semibold text-text-primary"><Activity size={15} className="text-text-muted" />Recent access activity</span>
                  <span className="text-[10px] text-text-muted">{showAudit ? 'Hide' : `${auditEvents.length} events`}</span>
                </button>
                {showAudit && (
                  <div className="space-y-2 border-t border-border p-3">
                    {auditEvents.length === 0 ? (
                      <p className="text-[10px] text-text-muted">No recent access events.</p>
                    ) : auditEvents.slice(0, 12).map((event, index) => <AuditRow key={event.id || `${event.timestamp || event.created_at || index}`} event={event} />)}
                  </div>
                )}
              </section>
            </>
          )}
        </>
      )}

      {loading && !status && !backendMissing && <p className="text-center text-[10px] text-text-muted">Loading Agent Access status…</p>}
    </div>
  )
}

function ToggleRow({
  checked,
  disabled,
  label,
  description,
  onChange,
}: {
  checked: boolean
  disabled: boolean
  label: string
  description: string
  onChange: (next: boolean) => void
}) {
  return (
    <div className={`flex items-start justify-between gap-3 ${disabled ? 'opacity-50' : ''}`}>
      <div className="min-w-0 flex-1">
        <div className="text-xs text-text-primary">{label}</div>
        <p className="mt-0.5 text-[10px] leading-relaxed text-text-muted">{description}</p>
      </div>
      <button type="button" role="switch" aria-checked={checked} aria-label={label} disabled={disabled} onClick={() => onChange(!checked)} className={`relative mt-0.5 inline-flex h-5 w-9 shrink-0 items-center rounded-full transition-colors ${checked ? 'bg-accent-blue' : 'border border-border bg-bg-primary'} disabled:cursor-not-allowed`}>
        <span className={`inline-block h-3.5 w-3.5 rounded-full border border-border bg-white transition-transform ${checked ? 'translate-x-4' : 'translate-x-0.5'}`} />
      </button>
    </div>
  )
}

function StatusPill({ label, good }: { label: string; good: boolean }) {
  return <span className={`rounded-full px-2 py-0.5 text-[9px] ${good ? 'bg-emerald-500/10 text-emerald-400' : 'bg-bg-primary text-text-muted'}`}>{label}</span>
}

function InfoChip({ label, value, good = false }: { label: string; value: string; good?: boolean }) {
  return (
    <div className="flex min-w-0 items-center justify-between gap-2 rounded-lg bg-bg-primary px-3 py-2">
      <span className="text-[10px] text-text-muted">{label}</span>
      <span className={`truncate text-[10px] ${good ? 'text-emerald-400' : 'text-text-secondary'}`}>{value}</span>
    </div>
  )
}

function InfoLine({
  label,
  value,
  mono = false,
  href,
  copyable = false,
  copied = false,
  onCopy,
}: {
  label: string
  value: string
  mono?: boolean
  href?: string
  copyable?: boolean
  copied?: boolean
  onCopy?: () => void
}) {
  return (
    <div className="space-y-1">
      <div className="text-[10px] text-text-muted">{label}</div>
      <div className="flex min-w-0 items-start gap-2 rounded-lg bg-bg-primary px-3 py-2">
        {href ? (
          <a href={href} target="_blank" rel="noreferrer" className={`min-w-0 flex-1 break-all text-[10px] text-accent-blue hover:underline ${mono ? 'font-mono' : ''}`}>{value}</a>
        ) : (
          <span className={`min-w-0 flex-1 break-all text-[10px] text-text-secondary ${mono ? 'font-mono' : ''}`}>{value}</span>
        )}
        {copyable && (
          <button type="button" onClick={onCopy} aria-label={`Copy ${label}`} className="shrink-0 rounded p-1 text-text-muted hover:bg-bg-hover hover:text-text-primary">
            {copied ? <Check size={13} /> : <Copy size={13} />}
          </button>
        )}
        {href && <ExternalLink size={12} className="mt-0.5 shrink-0 text-text-muted" />}
      </div>
    </div>
  )
}

function GrantRow({ grant, busy, disabled, onRevoke }: { grant: AgentAccessGrant; busy: boolean; disabled: boolean; onRevoke: () => void }) {
  const createdAt = formatDate(grant.created_at)
  return (
    <div className="rounded-lg bg-bg-primary p-3">
      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0">
          <div className="truncate text-xs font-medium text-text-primary">{grant.client_name || 'Unnamed agent'}</div>
          <div className="mt-0.5 break-all font-mono text-[9px] text-text-muted">{grant.client_id}</div>
        </div>
        {grant.revoked ? (
          <span className="rounded-full bg-red-500/10 px-2 py-1 text-[9px] text-red-300">Revoked</span>
        ) : (
          <button type="button" onClick={onRevoke} disabled={disabled} className="inline-flex min-h-8 shrink-0 items-center gap-1 rounded-md border border-border px-2 text-[9px] text-text-secondary hover:border-red-500/50 hover:text-red-300 disabled:opacity-40">
            {busy ? <LoaderCircle size={11} className="animate-spin" /> : <Trash2 size={11} />}
            Revoke
          </button>
        )}
      </div>
      <div className="mt-2 flex flex-wrap gap-1">{grant.scopes.map(scope => <span key={scope} className="rounded bg-bg-secondary px-1.5 py-0.5 text-[9px] text-text-secondary">{scope}</span>)}</div>
      {createdAt && <div className="mt-2 text-[9px] text-text-muted">Approved {createdAt}</div>}
    </div>
  )
}

function AuditRow({ event }: { event: AgentAccessAuditEvent }) {
  const title = firstText(event.action, event.event, event.type) || 'Access event'
  const detail = [
    firstText(event.actor, event.client_name),
    firstText(event.status, event.outcome, event.summary),
  ].filter(Boolean).join(' · ')
  const timestamp = formatDate(event.time ?? event.timestamp ?? event.created_at)
  return (
    <div className="flex items-start justify-between gap-3 rounded-lg bg-bg-primary px-3 py-2">
      <div className="min-w-0">
        <div className="text-[10px] font-medium text-text-primary">{title}</div>
        {detail && <div className="mt-0.5 break-words text-[9px] text-text-muted">{detail}</div>}
      </div>
      {timestamp && <time className="shrink-0 text-[9px] text-text-muted">{timestamp}</time>}
    </div>
  )
}

function Notice({ kind, children }: { kind: 'error' | 'success'; children: ReactNode }) {
  return <div className={`rounded-lg border p-3 text-xs leading-relaxed ${kind === 'error' ? 'border-red-500/30 bg-red-500/10 text-red-300' : 'border-emerald-500/30 bg-emerald-500/10 text-emerald-300'}`} role={kind === 'error' ? 'alert' : 'status'}>{children}</div>
}

function SecureOriginNote({ href }: { href: string | null }) {
  return (
    <div className="rounded-lg bg-bg-primary p-3 text-[10px] leading-relaxed text-text-muted">
      Owner sign-in is available only over HTTPS or a local browser address.
      {href && <a href={href} target="_blank" rel="noreferrer" className="ml-1 inline-flex items-center gap-1 text-accent-blue hover:underline">Open secure controller <ExternalLink size={10} /></a>}
    </div>
  )
}

function isSecureOrigin(): boolean {
  if (typeof window === 'undefined') return false
  if (window.location.protocol === 'https:') return true
  const hostname = window.location.hostname.toLowerCase().replace(/^\[|\]$/g, '')
  if (hostname === 'localhost' || hostname === '::1') return true
  const octets = hostname.split('.').map(Number)
  return octets.length === 4
    && octets.every(octet => Number.isInteger(octet) && octet >= 0 && octet <= 255)
    && octets[0] === 127
}

function safeControllerUrl(value: unknown): string | null {
  if (typeof value !== 'string' || !value) return null
  try {
    const url = new URL(value, window.location.href)
    if (url.username || url.password) return null
    if (url.protocol === 'https:') return url.href
    if (url.protocol === 'http:' && isLoopbackHost(url.hostname) && isLocalBrowserHost()) return url.href
    return null
  } catch {
    return null
  }
}

function safeLocalUrl(value: unknown): string | null {
  if (typeof value !== 'string' || !value) return null
  try {
    const url = new URL(value, window.location.href)
    if (url.username || url.password) return null
    if (url.protocol === 'https:' || (url.protocol === 'http:' && isLoopbackHost(url.hostname) && isLocalBrowserHost())) return url.href
    return null
  } catch {
    return null
  }
}

function isLoopbackHost(hostname: string): boolean {
  const host = hostname.toLowerCase().replace(/^\[|\]$/g, '')
  if (host === 'localhost' || host === '::1') return true
  const octets = host.split('.').map(Number)
  return octets.length === 4
    && octets.every(octet => Number.isInteger(octet) && octet >= 0 && octet <= 255)
    && octets[0] === 127
}

function isLocalBrowserHost(): boolean {
  return typeof window !== 'undefined' && isLoopbackHost(window.location.hostname)
}

function isHealthyStatus(value: string | null | undefined): boolean {
  return Boolean(value && ['ready', 'online', 'available', 'healthy', 'ok'].includes(value.toLowerCase()))
}

function firstText(...values: unknown[]): string | null {
  const value = values.find(candidate => typeof candidate === 'string' && candidate.trim())
  return typeof value === 'string' ? value : null
}

function formatDate(value: number | string | null | undefined): string | null {
  if (value == null || value === '') return null
  let date: Date
  if (typeof value === 'number') date = new Date(value > 10_000_000_000 ? value : value * 1000)
  else date = new Date(value)
  if (Number.isNaN(date.getTime())) return null
  return new Intl.DateTimeFormat(undefined, { dateStyle: 'short', timeStyle: 'short' }).format(date)
}

function clientSetupText(target: Exclude<CopyTarget, null>, url: string, fingerprint: string): string {
  const certificate = fingerprint ? `\nTLS certificate SHA-256: ${fingerprint}` : ''
  if (target === 'codex') {
    return [
      'Codex MCP setup for Maestro',
      'Add this server once to Codex, or merge it into ~/.codex/config.toml:',
      '',
      '[mcp_servers.maestro]',
      `url = ${JSON.stringify(url)}`,
      certificate,
      '',
      'No bearer token is included. Approve the requested scopes in Maestro when prompted.',
    ].filter(Boolean).join('\n')
  }
  if (target === 'claude') {
    return [
      'Claude MCP setup for Maestro',
      'Claude Desktop: Settings → Connectors → Add custom connector, then paste this server URL:',
      url,
      'Claude Desktop remote connectors need a URL reachable from Claude. For a local URL, use Claude Code on the same computer:',
      'claude mcp add --transport http maestro ' + JSON.stringify(url),
      certificate,
      '',
      'No bearer token is included. Approve the requested scopes in Maestro when prompted.',
    ].filter(Boolean).join('\n')
  }
  return url
}
