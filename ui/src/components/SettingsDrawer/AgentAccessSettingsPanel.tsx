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
  approveAgentAccessLocal,
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

type CopyTarget = 'agent' | 'codex' | 'claude' | 'url' | null

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
  const [consoleCodeExpiresInMinutes, setConsoleCodeExpiresInMinutes] = useState<number | null>(null)
  const [ownerCode, setOwnerCode] = useState('')
  const [labelDraft, setLabelDraft] = useState('')
  const [copyTarget, setCopyTarget] = useState<CopyTarget>(null)
  const [pairingCode, setPairingCode] = useState<string | null>(null)
  const [pairingExpiresAt, setPairingExpiresAt] = useState<number | null>(null)
  const [pairingSecondsLeft, setPairingSecondsLeft] = useState(0)
  const [selectedMcpUrl, setSelectedMcpUrl] = useState('')

  const ownerSession = status?.authenticated === true && status.owner === true
  const secureOrigin = isSecureOrigin()
  const localApproval = status?.local_approval_available === true
  const mcpUrls = useMemo(() => [...new Set([status?.mcp_url, ...(status?.lan_mcp_urls || [])]
    .filter((value): value is string => typeof value === 'string' && value.endsWith('/mcp') && safeControllerUrl(value) !== null))], [status])
  const mcpUrl = mcpUrls.includes(selectedMcpUrl) ? selectedMcpUrl : mcpUrls[0] || ''
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
    setOwnerCode('')
    setConsoleCodeRequested(false)
    setConsoleCodeExpiresInMinutes(null)
    setBusyAction('console-code')
    setError(null)
    setMessage(null)
    try {
      const result = await requestAgentAccessConsoleCode()
      setConsoleCodeRequested(true)
      const ttl = Number.isFinite(result.expires_in) ? Math.max(1, result.expires_in) : 600
      setConsoleCodeExpiresInMinutes(Math.max(1, Math.ceil(ttl / 60)))
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : 'Could not request a Browser sign-in code.')
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
      setConsoleCodeExpiresInMinutes(null)
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

  const handleLocalApproval = async () => {
    if (!localApproval || busyAction !== null) return
    setBusyAction('local-approval')
    setError(null)
    setMessage(null)
    try {
      await approveAgentAccessLocal()
      setOwnerCode('')
      setConsoleCodeRequested(false)
      setMessage('This browser is approved. You can now manage agent access.')
      await refresh()
      window.dispatchEvent(new Event(AGENT_ACCESS_SESSION_RESTORED_EVENT))
    } catch (approvalError) {
      setError(approvalError instanceof Error ? approvalError.message : 'Could not approve this browser. Try again or use a browser sign-in code.')
    } finally { setBusyAction(null) }
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
      setMessage('A separate Browser sign-in code is ready for the additional-browser pairing flow. It expires automatically.')
    } catch (pairError) {
      setError(pairError instanceof Error ? pairError.message : 'Could not create a Browser sign-in code.')
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
      setError('Clipboard access is unavailable. Expand Connection details to select the connection URL and copy it manually.')
    }
  }

  const sortedGrants = useMemo(
    () => [...grants].sort((a, b) => Number(Boolean(a.revoked)) - Number(Boolean(b.revoked))),
    [grants],
  )

  const step1State = ownerSession ? 'done' : 'current'
  const step2State = status?.enabled ? 'done' : ownerSession ? 'current' : 'locked'
  const canAddClient = ownerSession && status?.enabled === true && status.available

  return (
    <div className="space-y-4 pb-5">
      <section className="rounded-xl border border-border bg-bg-secondary p-4">
        <div className="flex items-start gap-3">
          <div className="mt-0.5 rounded-lg bg-accent-blue/10 p-2 text-accent-blue"><Shield size={17} /></div>
          <div className="min-w-0 flex-1">
            <div className="flex flex-wrap items-center gap-2">
              <h3 className="text-sm font-semibold text-text-primary">Maestro Agent Access</h3>
              <StatusPill label={status ? (status.enabled ? 'Enabled' : 'Off') : backendMissing ? 'Restart needed' : 'Checking'} good={status?.enabled === true} />
            </div>
            <p className="mt-1.5 text-xs leading-relaxed text-text-secondary">Follow these three steps to connect an MCP client. Each client completes a separate OAuth approval in Maestro; review or revoke its grant here.</p>
          </div>
          <button type="button" onClick={() => void refresh()} disabled={loading} className="rounded-lg p-2 text-text-muted hover:bg-bg-hover hover:text-text-primary disabled:opacity-40" aria-label="Refresh Agent Access status">
            <RefreshCw size={15} className={loading ? 'animate-spin' : ''} />
          </button>
        </div>
        {backendMissing ? (
          <div className="mt-4 flex gap-2 rounded-lg border border-amber-500/30 bg-amber-500/10 p-3 text-xs leading-relaxed text-text-secondary" role="status">
            <CircleAlert size={15} className="mt-0.5 shrink-0 text-amber-400" /><span>This Maestro backend does not have Agent Access yet. Restart Maestro after installing the update, then refresh this panel.</span>
          </div>
        ) : status && !status.available ? (
          <div className="mt-4 flex gap-2 rounded-lg border border-amber-500/30 bg-amber-500/10 p-3 text-xs leading-relaxed text-text-secondary" role="status">
            <CircleAlert size={15} className="mt-0.5 shrink-0 text-amber-400" /><span>Agent Access is unavailable on this installation.</span>
          </div>
        ) : null}
      </section>

      {error && <Notice kind="error">{error}</Notice>}
      {message && <Notice kind="success">{message}</Notice>}

      {!backendMissing && status && (
        <>
          <JourneyStep number={1} title="Sign in to this browser once" state={step1State}>
            {ownerSession ? (
              <div className="flex flex-wrap items-center justify-between gap-3">
                <p className="text-xs leading-relaxed text-text-secondary">This browser is signed in and can manage Agent Access.</p>
                <button type="button" onClick={() => void handleLogout()} disabled={busyAction !== null} className="inline-flex min-h-9 items-center gap-1.5 rounded-lg border border-border px-2.5 text-[10px] text-text-secondary hover:bg-bg-hover disabled:opacity-50">
                  {busyAction === 'logout' ? <LoaderCircle size={13} className="animate-spin" /> : <LogOut size={13} />}Sign out
                </button>
              </div>
            ) : (
              <div className="space-y-3">
                {!status.owner_configured && (
                  <div className="rounded-lg border border-accent-blue/20 bg-accent-blue/5 p-3 text-[11px] leading-relaxed text-text-secondary">
                    Before first setup: signing in here also protects Maestro in other browsers, which will need their own sign-in. Agent connections stay off until you allow them; turning agents off later keeps browser sign-in protection in place.
                  </div>
                )}
                {localApproval ? (
                  <div className="space-y-2">
                    <p className="text-xs leading-relaxed text-text-secondary">You are connected directly on this PC. Approve this browser with one click; no code or terminal is needed.</p>
                    <button type="button" onClick={() => void handleLocalApproval()} disabled={busyAction !== null || !status.available} className="inline-flex min-h-11 w-full items-center justify-center gap-2 rounded-lg bg-accent-blue px-4 py-2 text-xs font-semibold text-bg-primary disabled:opacity-50">
                      {busyAction === 'local-approval' ? <LoaderCircle size={14} className="animate-spin" /> : <ShieldCheck size={14} />}
                      {busyAction === 'local-approval' ? 'Approving…' : 'Approve on this PC'}
                    </button>
                  </div>
                ) : status.owner_configured ? <p className="text-xs leading-relaxed text-text-secondary">Open Maestro on its PC through Pinokio to approve locally, or use a Browser sign-in code from an existing owner.</p> : null}
                <details open={!localApproval || consoleCodeRequested ? true : undefined} className={localApproval ? 'rounded-lg border border-border p-3' : ''}>
                  <summary className="mb-3 cursor-pointer text-xs text-text-secondary">Use a browser sign-in code</summary>
                {!secureOrigin && <SecureOriginNote href={controllerUrl} />}
                {consoleCodeRequested && secureOrigin && (
                  <div className="space-y-1.5 rounded-lg bg-bg-primary p-3 text-[11px] leading-relaxed text-text-secondary" role="status">
                    <p>Open <strong className="font-semibold text-text-primary">Pinokio &gt; this Maestro app &gt; Agent Access Terminal</strong>.</p>
                    <p>Copy only the newest line labeled <code className="font-mono text-text-primary">Browser code:</code>. It expires in about {consoleCodeExpiresInMinutes || 10} minutes and is single-use. If you request another, use only the newest code.</p>
                    <p>Paste it below, then select <strong className="font-semibold text-text-primary">Sign in</strong>.</p>
                  </div>
                )}
                <button type="button" onClick={() => void handleRequestConsoleCode()} disabled={!secureOrigin || busyAction !== null || !status.available} className="inline-flex min-h-11 w-full items-center justify-center gap-2 rounded-lg border border-border px-3 py-2 text-xs font-medium text-text-primary transition-colors hover:bg-bg-hover disabled:cursor-not-allowed disabled:opacity-40">
                  {busyAction === 'console-code' ? <LoaderCircle size={14} className="animate-spin" /> : <LogIn size={14} />}
                  {consoleCodeRequested ? 'Request a new Browser sign-in code' : 'Request Browser sign-in code'}
                </button>
                {secureOrigin && (
                  <form className="space-y-2" onSubmit={event => void handleOwnerLogin(event)}>
                    {!consoleCodeRequested && <p className="text-[10px] leading-relaxed text-text-muted">Already have a Browser sign-in code? Paste it below.</p>}
                    <label htmlFor="agent-access-owner-code" className="block text-[11px] font-medium text-text-secondary">Browser sign-in code</label>
                    <div className="flex flex-col gap-2 sm:flex-row">
                      <input id="agent-access-owner-code" value={ownerCode} onChange={event => setOwnerCode(event.target.value)} autoComplete="one-time-code" autoCapitalize="none" spellCheck={false} placeholder="Paste your Browser sign-in code here" className="min-h-10 min-w-0 flex-1 rounded-lg border border-border bg-bg-primary px-3 py-2 font-mono text-xs text-text-primary outline-none focus:border-accent-blue" disabled={busyAction !== null} />
                      <button type="submit" disabled={!ownerCode.trim() || busyAction !== null} className="min-h-10 rounded-lg bg-accent-blue px-4 text-xs font-medium text-white hover:opacity-90 disabled:opacity-40">{busyAction === 'owner-login' ? 'Signing in…' : 'Sign in'}</button>
                    </div>
                  </form>
                )}
                </details>
              </div>
            )}
          </JourneyStep>

          <JourneyStep number={2} title="Allow agent connections" state={step2State}>
            {ownerSession ? (
              <div className="space-y-3">
                <ToggleRow checked={status.enabled} disabled={busyAction !== null || !status.available} label={status.enabled ? 'Agent connections are allowed' : 'Allow agent connections'} description={status.enabled ? 'Clients can request access. Each client completes a separate OAuth approval in Maestro; review or revoke its grant here.' : 'Agent connections are paused. Turning them on allows clients to request access; each client still needs separate OAuth approval in Maestro.'} onChange={value => void handleToggle(value)} />
                {!status.enabled && <p className="rounded-lg bg-bg-primary p-3 text-[10px] leading-relaxed text-text-muted">Existing grants stay in place while paused. Browser sign-in protection remains active.</p>}
              </div>
            ) : <p className="text-xs leading-relaxed text-text-muted">{status.enabled ? 'Agent connections are allowed. Sign in to manage this setting and review grants.' : 'Sign in to manage whether agent connections are allowed.'}</p>}
          </JourneyStep>

          <JourneyStep number={3} title="Add an MCP client" state={canAddClient ? 'current' : 'locked'}>
            {canAddClient ? (
              <div className="space-y-3">
                <p className="text-xs leading-relaxed text-text-secondary">Let your agent handle setup with a copied prompt, or choose manual client instructions. Copying text does not connect or approve the client.</p>
                <div className="rounded-lg border border-border bg-bg-primary p-3 text-[11px] leading-relaxed text-text-secondary">On first connection, a client may ask you to trust Maestro’s TLS certificate. Verify its SHA-256 fingerprint in Connection details before accepting.</div>
                {!mcpUrl && <p className="text-[10px] text-amber-300">The connection URL is not available yet. Refresh this panel after the service is ready.</p>}
                <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
                  <CopySetupButton target="agent" copied={copyTarget === 'agent'} disabled={!mcpUrl || !status.certificate_sha256} onClick={() => void handleCopy('agent')} />
                  <CopySetupButton target="codex" copied={copyTarget === 'codex'} disabled={!mcpUrl} onClick={() => void handleCopy('codex')} />
                  <CopySetupButton target="claude" copied={copyTarget === 'claude'} disabled={!mcpUrl} onClick={() => void handleCopy('claude')} />
                  <CopySetupButton target="url" copied={copyTarget === 'url'} disabled={!mcpUrl} onClick={() => void handleCopy('url')} />
                </div>
                {copyTarget === 'agent' && <p className="text-[10px] text-emerald-400" role="status">Setup prompt copied. Paste it into chat with your agent. It will guide setup; client OAuth approval still happens in Maestro.</p>}
                {copyTarget === 'url' && <p className="text-[10px] text-emerald-400" role="status">Server URL copied. Add it in the client’s MCP settings. This does not connect or approve the client; complete its OAuth approval in Maestro.</p>}
                {copyTarget && copyTarget !== 'agent' && copyTarget !== 'url' && <p className="text-[10px] text-emerald-400" role="status">Setup instructions copied. Follow them in the client. This does not connect or approve the client; complete its OAuth approval in Maestro.</p>}
              </div>
            ) : (
              <p className="text-xs leading-relaxed text-text-muted">{!ownerSession ? 'Sign in first to continue.' : !status.enabled ? 'Allow agent connections in step 2 to continue.' : 'Agent Access is unavailable. Refresh the panel when the service is ready.'}</p>
            )}
          </JourneyStep>

          {ownerSession && (
            <section className="space-y-3 rounded-xl border border-border bg-bg-secondary p-4" aria-labelledby="agent-access-grants-heading">
              <div className="flex items-center justify-between gap-2">
                <div className="flex items-center gap-2"><ShieldCheck size={15} className="text-text-muted" /><h4 id="agent-access-grants-heading" className="text-xs font-semibold text-text-primary">Agent grants</h4></div>
                <span className="rounded-full bg-bg-primary px-2 py-1 text-[10px] text-text-muted">{grants.filter(grant => !grant.revoked).length} active</span>
              </div>
              <p className="text-[10px] leading-relaxed text-text-muted">Review or revoke a client’s access here. Grants remain visible while agent connections are paused.</p>
              {grants.length === 0 ? <p className="rounded-lg bg-bg-primary p-3 text-[10px] text-text-muted">No agents have been approved yet.</p> : (
                <div className="space-y-2">
                  {sortedGrants.map(grant => <GrantRow key={grant.grant_id} grant={grant} busy={busyAction === 'revoke:' + grant.grant_id} disabled={busyAction !== null} onRevoke={() => void handleRevoke(grant)} />)}
                </div>
              )}
            </section>
          )}

          <details className="rounded-xl border border-border bg-bg-secondary">
            <summary className="flex min-h-12 cursor-pointer list-none items-center justify-between gap-3 px-4 text-left">
              <span className="flex min-w-0 items-center gap-2"><Fingerprint size={15} className="shrink-0 text-text-muted" /><span className="min-w-0"><span className="block text-xs font-semibold text-text-primary">Connection details</span><span className="mt-0.5 block text-[10px] text-text-muted">Advanced URLs, certificate fingerprint, health, browser pairing, and access activity</span></span></span>
              <span className="shrink-0 text-[10px] text-text-muted">Expand</span>
            </summary>
            <div className="space-y-4 border-t border-border p-3 sm:p-4">
              <section className="space-y-3">
                <h4 className="text-xs font-semibold text-text-primary">Connection details</h4>
                {status.label !== undefined && (
                  <div className="space-y-1.5">
                    <label htmlFor="agent-access-label" className="text-[10px] text-text-secondary">Installation name</label>
                    <div className="flex flex-col gap-2 sm:flex-row">
                      <input id="agent-access-label" value={labelDraft} maxLength={64} onChange={event => setLabelDraft(event.target.value)} disabled={!ownerSession || busyAction !== null} className="min-h-10 min-w-0 flex-1 rounded-lg border border-border bg-bg-primary px-3 py-2 text-xs text-text-primary outline-none focus:border-accent-blue disabled:opacity-50" placeholder="My Maestro" />
                      <button type="button" onClick={() => void handleSaveLabel()} disabled={!ownerSession || busyAction !== null || labelDraft.trim() === (status.label || '')} className="min-h-10 rounded-lg border border-border px-3 text-[10px] text-text-secondary hover:bg-bg-hover disabled:opacity-40">Save</button>
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
                {status.certificate_sha256 && <a href="/api/v1/agent-access/certificate" download="maestro-agent-access.pem" className="inline-flex min-h-9 items-center gap-2 text-[10px] text-accent-blue hover:underline">Download public TLS certificate</a>}
              </section>
              <section className="space-y-2 rounded-lg bg-bg-primary p-3">
                <h4 className="text-xs font-semibold text-text-primary">Service status</h4>
                <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
                  <InfoChip label="Controller" value={status.controller_status || 'Checking'} good={isHealthyStatus(status.controller_status)} />
                  <InfoChip label="Backend" value={status.backend_online === undefined ? 'Checking' : status.backend_online ? 'Online' : 'Offline'} good={status.backend_online === true} />
                  <InfoChip label="Discovery" value={status.discovery_status || 'Not reported'} good={isHealthyStatus(status.discovery_status)} />
                </div>
              </section>
              {ownerSession && (
                <>
                  <section className="space-y-3 rounded-lg border border-border p-3">
                    <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
                      <div><h4 className="text-xs font-semibold text-text-primary">Additional browser access</h4><p className="mt-1 text-[10px] leading-relaxed text-text-muted">Create a separate Browser sign-in code for the Maestro browser pairing flow. It expires automatically.</p></div>
                      <button type="button" onClick={() => void handlePairingCode()} disabled={busyAction !== null} className="inline-flex min-h-10 shrink-0 items-center justify-center gap-1.5 rounded-lg border border-border px-3 text-[10px] text-text-secondary hover:bg-bg-hover disabled:opacity-50">
                        {busyAction === 'browser-code' ? <LoaderCircle size={13} className="animate-spin" /> : <Link2 size={13} />}Create Browser sign-in code
                      </button>
                    </div>
                    {pairingCode && <div className="flex items-center justify-between gap-3 rounded-lg border border-border bg-bg-primary p-3"><code className="break-all font-mono text-sm tracking-wider text-text-primary">{pairingCode}</code><span className="shrink-0 text-[10px] text-text-muted">{pairingSecondsLeft}s</span></div>}
                  </section>
                  <section className="rounded-lg border border-border">
                    <details>
                      <summary className="flex min-h-11 cursor-pointer list-none items-center justify-between gap-3 px-3 text-left">
                        <span className="flex items-center gap-2 text-xs font-semibold text-text-primary"><Activity size={15} className="text-text-muted" />Recent access activity</span><span className="text-[10px] text-text-muted">{auditEvents.length} events</span>
                      </summary>
                      <div className="space-y-2 border-t border-border p-3">
                        {auditEvents.length === 0 ? <p className="text-[10px] text-text-muted">No recent access events.</p> : auditEvents.slice(0, 12).map((event, index) => <AuditRow key={event.id || String(event.timestamp || event.created_at || index)} event={event} />)}
                      </div>
                    </details>
                  </section>
                </>
              )}
            </div>
          </details>
        </>
      )}

      {loading && !status && !backendMissing && <p className="text-center text-[10px] text-text-muted">Loading Agent Access status…</p>}
    </div>
  )
}

type JourneyStepState = 'done' | 'current' | 'locked'

function JourneyStep({ number, title, state, children }: {
  number: number
  title: string
  state: JourneyStepState
  children: ReactNode
}) {
  const stateLabel = state === 'done' ? 'Complete' : state === 'current' ? 'Current step' : 'Next'
  const stateTone = state === 'done'
    ? 'border-emerald-500/30'
    : state === 'current'
      ? 'border-accent-blue/40'
      : 'border-border'
  const markerTone = state === 'done'
    ? 'bg-emerald-500/10 text-emerald-400'
    : state === 'current'
      ? 'bg-accent-blue/10 text-accent-blue'
      : 'bg-bg-primary text-text-muted'

  return (
    <section aria-current={state === 'current' ? 'step' : undefined} className={'rounded-xl border bg-bg-secondary p-4 ' + stateTone}>
      <div className="flex items-start gap-3">
        <span className={'flex h-7 w-7 shrink-0 items-center justify-center rounded-full text-xs font-semibold ' + markerTone}>
          {state === 'done' ? <Check size={14} /> : number}
        </span>
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2">
            <h4 className="text-sm font-semibold text-text-primary">{title}</h4>
            <span className="rounded-full bg-bg-primary px-2 py-0.5 text-[9px] text-text-muted">{stateLabel}</span>
          </div>
          <div className="mt-3 space-y-3">{children}</div>
        </div>
      </div>
    </section>
  )
}

function CopySetupButton({ target, copied, disabled, onClick }: {
  target: Exclude<CopyTarget, null>
  copied: boolean
  disabled: boolean
  onClick: () => void
}) {
  const label = target === 'codex'
    ? 'Codex'
    : target === 'claude'
      ? 'Claude'
      : target === 'url'
        ? 'server URL for another MCP client'
        : 'setup prompt for my agent'
  const buttonText = target === 'agent'
    ? copied ? 'Setup prompt copied' : 'Copy setup prompt for my agent'
    : target === 'url'
      ? copied ? 'Server URL copied' : 'Copy server URL for another MCP client'
      : copied ? label + ' setup copied' : 'Copy ' + label + ' setup'

  return (
    <button type="button" onClick={onClick} disabled={disabled} className={'inline-flex min-h-11 items-center justify-center gap-2 rounded-lg border px-3 text-[10px] font-medium hover:bg-bg-hover disabled:opacity-40 ' + (target === 'agent' ? 'border-accent-blue/30 bg-accent-blue/10 text-accent-blue sm:col-span-2' : 'border-border text-text-secondary')}>
      {copied ? <Check size={13} /> : <Copy size={13} />}
      {buttonText}
    </button>
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
          <details className="mt-1">
            <summary className="cursor-pointer text-[9px] text-text-muted">Client ID</summary>
            <code className="mt-1 block break-all font-mono text-[9px] text-text-muted">{grant.client_id}</code>
          </details>
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
  if (target === 'agent') {
    return [
      'Configure Maestro as an MCP server in this client.',
      `MCP server URL: ${url}`,
      `Expected public TLS certificate SHA-256 fingerprint: ${fingerprint || 'NOT REPORTED — stop and ask me to verify it before trusting any certificate.'}`,
      'Preserve the client’s existing configuration. Add only a Maestro server entry; do not remove or rewrite unrelated settings.',
      'If another Maestro installation is already configured, keep it and add this installation under a distinct server name.',
      'Use the client’s normal certificate trust flow. Verify that the presented certificate exactly matches this fingerprint before trusting it. Never disable TLS verification or use an insecure bypass.',
      'Use the normal OAuth approval flow. When Maestro requests approval, show me the requested scopes and wait for my approval there. Keep codes and tokens out of chat and setup instructions; let the client manage OAuth credentials in its normal protected credential store.',
      'After setup, verify with a read-only server status or help request only. Do not render or create images, video, audio, or other media during setup.',
      'If this client cannot configure HTTP MCP or verify the certificate safely, stop and explain the limitation. Do not try an unsupported install command.',
    ].join('\n')
  }
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
      'Claude Desktop remote connectors need a URL reachable from Claude. For a local URL, use Claude Code on a computer that can reach this Maestro:',
      'claude mcp add --transport http maestro ' + JSON.stringify(url),
      certificate,
      '',
      'No bearer token is included. Approve the requested scopes in Maestro when prompted.',
    ].filter(Boolean).join('\n')
  }
  return url
}
