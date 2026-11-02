import { useCallback, useEffect, useState } from 'react'
import { Check, Copy, ExternalLink, LoaderCircle, RefreshCw, Smartphone } from 'lucide-react'
import { QRCodeSVG } from 'qrcode.react'
import {
  AgentAccessApiError,
  fetchPinokioRemoteAccessStatus,
  updateAgentAccessPinokioRemoteSettings,
  type AgentAccessPinokioRemoteStatus,
  type AgentAccessStatus,
  type PinokioRemoteAccessStatus,
} from '../../api/agentAccess'

interface PinokioRemoteAccessProps {
  ownerSession: boolean
  remoteMcpEnabled: boolean
  managedStatus: AgentAccessPinokioRemoteStatus | null
  onManagedStatusChange: (status: AgentAccessPinokioRemoteStatus) => void
  refreshOwnerStatus: () => Promise<AgentAccessStatus | null>
}

export function PinokioRemoteAccess({
  ownerSession,
  remoteMcpEnabled,
  managedStatus,
  onManagedStatusChange,
  refreshOwnerStatus,
}: PinokioRemoteAccessProps) {
  const [detectedStatus, setDetectedStatus] = useState<PinokioRemoteAccessStatus | null>(null)
  const [suppressManagedFallback, setSuppressManagedFallback] = useState(true)
  const [loading, setLoading] = useState(true)
  const [checking, setChecking] = useState(false)
  const [detectionError, setDetectionError] = useState<string | null>(null)
  const [actionError, setActionError] = useState<string | null>(null)
  const [actionMessage, setActionMessage] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const [copied, setCopied] = useState(false)

  const refreshDetection = useCallback(async () => {
    setChecking(true)
    setDetectionError(null)
    setDetectedStatus(null)
    setSuppressManagedFallback(true)
    try {
      const next = await fetchPinokioRemoteAccessStatus()
      setDetectedStatus(next)
      setSuppressManagedFallback(false)
      return next
    } catch (error) {
      setDetectionError(error instanceof Error ? error.message : 'Could not read Pinokio Remote status.')
      return null
    } finally {
      setChecking(false)
      setLoading(false)
    }
  }, [])

  useEffect(() => { void refreshDetection() }, [refreshDetection])

  const handleRefresh = async () => {
    setActionError(null)
    setActionMessage(null)
    await refreshDetection()
    if (ownerSession) {
      try {
        await refreshOwnerStatus()
      } catch {
        // The Agent Access panel reports its own status refresh errors.
      }
    }
  }

  const status = suppressManagedFallback ? null : detectedStatus ?? managedStatus
  const browserUrl = status?.available && status.phase.toLowerCase() === 'ready'
    ? safePinokioBrowserUrl(status.browser_url)
    : null
  const browserReady = Boolean(browserUrl)
  const mcpUrl = !suppressManagedFallback && browserReady && isAgentAccessPinokioRemoteReady(managedStatus)
    ? safePinokioMcpUrl(managedStatus?.mcp_url)
    : null
  const enabled = managedStatus?.enabled === true
  const ready = Boolean(enabled && managedStatus?.ready && mcpUrl)

  const handleCopy = async () => {
    if (!browserUrl) return
    setActionError(null)
    try {
      await navigator.clipboard.writeText(browserUrl)
      setCopied(true)
      window.setTimeout(() => setCopied(false), 1600)
    } catch {
      setActionError('Clipboard access is unavailable. Select the browser address and copy it manually.')
    }
  }

  const handleToggle = async (nextEnabled: boolean) => {
    setBusy(true)
    setActionError(null)
    setActionMessage(null)
    try {
      const next = await updateAgentAccessPinokioRemoteSettings({ enabled: nextEnabled })
      onManagedStatusChange(next)
      if (!nextEnabled) {
        setActionMessage('Pinokio Remote MCP access was disabled. Browser access and other transports are unchanged.')
        await refreshOwnerStatus()
        return
      }

      setActionMessage(browserReady && isAgentAccessPinokioRemoteReady(next)
        ? 'Pinokio Remote MCP access is ready. Each client still needs its own authorization in Maestro.'
        : 'Pinokio Remote MCP access is enabled. Maestro is refreshing its Agent Access controller; refresh status in a moment to confirm readiness.')
      await new Promise<void>(resolve => window.setTimeout(resolve, 1100))
      const refreshedDetection = await refreshDetection()
      const refreshed = await refreshOwnerStatus()
      const confirmed = refreshed?.pinokio_remote
      const confirmedReady = Boolean(
        refreshedDetection?.available
        && refreshedDetection.phase.toLowerCase() === 'ready'
        && safePinokioBrowserUrl(refreshedDetection.browser_url)
        && isAgentAccessPinokioRemoteReady(confirmed),
      )
      setActionMessage(confirmedReady
        ? 'Pinokio Remote MCP access is ready. Each client still needs its own authorization in Maestro.'
        : 'Pinokio Remote MCP access was enabled. Maestro has not confirmed readiness yet; refresh after its Agent Access controller restarts before connecting.')
    } catch (error) {
      if (error instanceof AgentAccessApiError && error.status === 404) {
        setActionError('Pinokio Remote MCP settings are not available on this Maestro version. Refresh after updating Maestro.')
      } else {
        setActionError(error instanceof Error ? error.message : 'Could not update Pinokio Remote MCP access.')
      }
    } finally {
      setBusy(false)
    }
  }

  const showManagement = Boolean(managedStatus) || ownerSession
  const canTurnOn = ownerSession && remoteMcpEnabled && browserReady
  const toggleDisabled = busy || !ownerSession || (!enabled && !canTurnOn)

  return (
    <section className="space-y-3 rounded-xl border border-border bg-bg-secondary p-4" aria-labelledby="pinokio-remote-access-heading">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <h3 id="pinokio-remote-access-heading" className="text-xs font-semibold text-text-primary">Pinokio Remote</h3>
          <p className="mt-1 text-[10px] leading-relaxed text-text-muted">Private browser access and optional MCP access through this app&apos;s verified Pinokio Remote address.</p>
        </div>
        <button
          type="button"
          onClick={() => void handleRefresh()}
          disabled={checking || busy}
          aria-label="Refresh Pinokio Remote status"
          className="inline-flex min-h-8 shrink-0 items-center gap-1.5 rounded-lg border border-border px-2 text-[10px] text-text-secondary hover:bg-bg-hover disabled:opacity-50"
        >
          {checking ? <LoaderCircle size={12} className="animate-spin" /> : <RefreshCw size={12} />}
          Refresh
        </button>
      </div>

      <div className="space-y-2 rounded-lg border border-border bg-bg-primary p-3" aria-labelledby="pinokio-browser-access-heading">
        <div className="flex items-center gap-2">
          <Smartphone size={14} className="text-accent-blue" />
          <h4 id="pinokio-browser-access-heading" className="text-[11px] font-semibold text-text-primary">Browser access</h4>
        </div>
        {loading ? (
          <p className="text-[10px] text-text-muted" role="status">Checking Pinokio Remote…</p>
        ) : browserUrl ? (
          <div className="space-y-3 rounded-md border border-emerald-500/30 bg-emerald-500/5 p-3">
            <div className="flex flex-col items-center gap-2">
              <div className="rounded-lg bg-white p-2">
                <QRCodeSVG value={browserUrl} size={132} level="M" marginSize={1} />
              </div>
              <div className="text-center text-[9px] leading-relaxed text-emerald-100">
                Open this link on a device authorized for this Pinokio Remote app. This PC does not need a second Tailscale install.
              </div>
            </div>
            <div className="break-all rounded bg-bg-primary/70 px-2 py-1.5 text-[9px] text-text-secondary">{browserUrl}</div>
            <div className="grid grid-cols-2 gap-2">
              <button
                type="button"
                onClick={() => void handleCopy()}
                className="flex min-h-9 items-center justify-center gap-1 rounded-md border border-border px-2 py-1.5 text-[10px] text-text-secondary hover:bg-bg-hover hover:text-text-primary"
              >
                {copied ? <Check size={10} /> : <Copy size={10} />}
                {copied ? 'Copied' : 'Copy link'}
              </button>
              <a
                href={browserUrl}
                target="_blank"
                rel="noreferrer"
                className="flex min-h-9 items-center justify-center gap-1 rounded-md border border-border px-2 py-1.5 text-[10px] text-text-secondary hover:bg-bg-hover hover:text-text-primary"
              >
                Open <ExternalLink size={10} />
              </a>
            </div>
          </div>
        ) : (
          <div className="rounded-md bg-bg-tertiary p-2.5 text-[10px] leading-relaxed text-text-muted" role={detectionError ? 'alert' : 'status'}>
            {detectionError
              ? `Could not check Pinokio Remote status: ${detectionError}`
              : status
                ? pinokioStatusReason(status.reason, status.phase)
                : 'Pinokio Remote status is unavailable on this Maestro version.'}
          </div>
        )}
      </div>

      <div className="space-y-2 rounded-lg border border-border bg-bg-primary p-3" aria-labelledby="pinokio-agent-access-heading">
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0">
            <h4 id="pinokio-agent-access-heading" className="text-[11px] font-semibold text-text-primary">AI agent access</h4>
            <p className="mt-1 text-[10px] leading-relaxed text-text-muted">Remote MCP must also be on. This only changes Maestro&apos;s Agent Access controller; it does not change your Pinokio account, Pinokio network, or Tailscale. The MCP URL shares Maestro&apos;s published web route, so leave Maestro and Agent Access running while using it. Existing direct MCP endpoints and Local MCP remain independent.</p>
          </div>
          {managedStatus && <span className={`shrink-0 rounded-full px-2 py-1 text-[9px] ${ready ? 'bg-emerald-500/10 text-emerald-300' : enabled ? 'bg-amber-500/10 text-amber-200' : 'bg-bg-tertiary text-text-muted'}`}>{ready ? 'Ready' : enabled ? 'Starting' : 'Off'}</span>}
        </div>
        {showManagement ? (
          <>
            <button
              type="button"
              role="switch"
              aria-checked={enabled}
              aria-label="Use Pinokio Remote for MCP"
              disabled={toggleDisabled}
              onClick={() => void handleToggle(!enabled)}
              className={`inline-flex min-h-10 w-full items-center justify-between gap-3 rounded-lg border border-border px-3 text-left text-[10px] text-text-primary hover:bg-bg-hover disabled:cursor-not-allowed disabled:opacity-50 ${enabled ? 'border-accent-blue/40 bg-accent-blue/5' : 'bg-bg-primary'}`}
            >
              <span className="font-medium">{busy ? 'Updating Pinokio Remote MCP…' : 'Use Pinokio Remote for MCP'}</span>
              <span className={`relative h-5 w-9 shrink-0 rounded-full transition-colors ${enabled ? 'bg-accent-blue' : 'border border-border bg-bg-tertiary'}`}>
                <span className={`absolute top-0.5 inline-block h-3.5 w-3.5 rounded-full bg-white transition-transform ${enabled ? 'translate-x-4' : 'translate-x-0.5'}`} />
              </span>
            </button>
            {!ownerSession && <p className="text-[10px] leading-relaxed text-text-muted">Sign in to manage AI agent access. The browser link above remains available on authorized devices.</p>}
            {ownerSession && !remoteMcpEnabled && !enabled && <p className="text-[10px] leading-relaxed text-text-muted">Enable Remote MCP above first. This Pinokio Remote opt-in does not turn Remote MCP on by itself.</p>}
            {ownerSession && !browserReady && !enabled && remoteMcpEnabled && <p className="text-[10px] leading-relaxed text-text-muted">The Pinokio Remote browser link must be ready before MCP access can be enabled.</p>}
            {ready && mcpUrl && <p className="break-all rounded-md bg-bg-tertiary px-2 py-1.5 font-mono text-[9px] text-text-secondary">MCP URL: {mcpUrl}</p>}
          </>
        ) : (
          <p className="text-[10px] leading-relaxed text-text-muted">Sign in to manage Pinokio Remote MCP access. Browser access does not enable AI agent access.</p>
        )}
        {actionError && <p className="rounded-md border border-rose-500/30 bg-rose-500/5 p-2 text-[10px] leading-relaxed text-rose-200" role="alert">{actionError}</p>}
        {actionMessage && <p className="rounded-md border border-emerald-500/30 bg-emerald-500/5 p-2 text-[10px] leading-relaxed text-emerald-100" role="status">{actionMessage}</p>}
        {enabled && !ready && <p className="text-[10px] leading-relaxed text-text-muted" role="status">MCP is enabled, but the backend has not confirmed readiness yet. Refresh after Maestro refreshes its Agent Access controller; do not connect until it reports Ready.</p>}
      </div>
    </section>
  )
}

function isAgentAccessPinokioRemoteReady(value: AgentAccessPinokioRemoteStatus | null | undefined): boolean {
  return Boolean(value?.enabled && value.ready && safePinokioMcpUrl(value.mcp_url))
}

function safePinokioBrowserUrl(value: unknown): string | null {
  if (typeof value !== 'string' || !value) return null
  try {
    const url = new URL(value)
    if (url.protocol !== 'https:' || !isPinokioRemoteHostname(url.hostname) || url.username || url.password) return null
    return value
  } catch {
    return null
  }
}

function safePinokioMcpUrl(value: unknown): string | null {
  const safe = safePinokioBrowserUrl(value)
  if (!safe) return null
  try {
    const url = new URL(safe)
    return url.protocol === 'https:'
      && isPinokioRemoteHostname(url.hostname)
      && url.pathname === '/mcp'
      && !url.search
      && !url.hash
      ? safe
      : null
  } catch {
    return null
  }
}

function isPinokioRemoteHostname(hostname: string): boolean {
  return hostname.toLowerCase().endsWith('.ts.net')
}

function pinokioStatusReason(reason: string | null, phase: string): string {
  const code = (reason || phase).trim().toLowerCase().replace(/[\s-]+/g, '_')
  if (['app_not_installed', 'app_not_published', 'remote_disabled'].includes(code)) {
    return 'Start Maestro and enable Remote for this app in Pinokio, then refresh.'
  }
  if (['controller_starting', 'starting', 'restarting', 'agent_access_restarting'].includes(code)) {
    return 'Maestro is refreshing its Agent Access controller. Refresh this status in a moment.'
  }
  if (['node_unavailable', 'node_offline', 'remote_unavailable', 'unavailable', 'not_available'].includes(code)) {
    return 'Pinokio Remote is unavailable right now. Check that Maestro is running from Pinokio and refresh.'
  }
  if (['ready', 'published'].includes(code)) {
    return 'Pinokio Remote reported ready without a usable HTTPS browser link. Refresh or check this app’s Remote settings in Pinokio.'
  }
  return 'Pinokio Remote is not ready. Start Maestro and enable Remote for this app in Pinokio, then refresh.'
}
