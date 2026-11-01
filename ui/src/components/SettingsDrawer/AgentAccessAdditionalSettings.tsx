import { useCallback, useEffect, useMemo, useRef, useState, type FormEvent } from 'react'
import { Check, Copy, LoaderCircle, RefreshCw, ShieldCheck } from 'lucide-react'
import {
  createAgentAccessLocalClient,
  fetchAgentAccessCreativeDefaults,
  fetchAgentAccessLocalClientExtension,
  fetchAgentAccessLocalClients,
  formatLocalAgentClientClaudeCommand,
  formatLocalAgentClientCodexConfig,
  formatLocalAgentClientJson,
  updateAgentAccessCreativeDefaults,
  type AgentAccessCreativeDefault,
  type AgentAccessCreativeDefaults,
  type AgentAccessLocalClient,
  type AgentAccessScope,
} from '../../api/agentAccess'

const SCOPE_CHOICES: Array<{ id: AgentAccessScope; label: string; description: string }> = [
  { id: 'observe', label: 'Observe', description: 'Read status and inspect Maestro data.' },
  { id: 'generate', label: 'Generate', description: 'Start supported image, video, and audio generation.' },
  { id: 'manage', label: 'Manage', description: 'Change supported settings and manage resources.' },
  { id: 'admin', label: 'Admin', description: 'Use administrative Agent Access actions.' },
]

export function LocalMcpSettings({ ownerSession, enabled, available, refreshKey }: {
  ownerSession: boolean
  enabled: boolean
  available: boolean
  refreshKey: number
}) {
  const [clients, setClients] = useState<AgentAccessLocalClient[]>([])
  const [loading, setLoading] = useState(false)
  const [loadError, setLoadError] = useState(false)
  const [creating, setCreating] = useState(false)
  const [name, setName] = useState('')
  const [scopes, setScopes] = useState<AgentAccessScope[]>(['observe'])
  const [error, setError] = useState<string | null>(null)
  const [message, setMessage] = useState<string | null>(null)
  const [copied, setCopied] = useState<string | null>(null)
  const [downloadingClientId, setDownloadingClientId] = useState<string | null>(null)
  const setupGeneration = useRef(0)

  const refresh = useCallback(async () => {
    const generation = setupGeneration.current
    if (!ownerSession) {
      setClients([])
      setLoading(false)
      return
    }
    setLoading(true)
    setLoadError(false)
    setError(null)
    try {
      const next = await fetchAgentAccessLocalClients()
      if (generation === setupGeneration.current) {
        setClients(next)
        setLoadError(false)
      }
    } catch (requestError) {
      if (generation === setupGeneration.current) {
        setLoadError(true)
        setError(requestError instanceof Error ? requestError.message : 'Could not load local MCP clients.')
      }
    } finally {
      if (generation === setupGeneration.current) setLoading(false)
    }
  }, [ownerSession])

  const invalidateSetup = useCallback(() => {
    setupGeneration.current += 1
    setClients([])
    setCopied(null)
    setDownloadingClientId(null)
    setLoadError(false)
    setMessage(null)
    setError(null)
  }, [])

  useEffect(() => {
    invalidateSetup()
    void refresh()
  }, [invalidateSetup, refresh, refreshKey])

  const handleCreate = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    if (!ownerSession || !enabled || !available || !name.trim() || scopes.length === 0) return
    setCreating(true)
    setError(null)
    setMessage(null)
    try {
      const client = await createAgentAccessLocalClient({ name: name.trim(), scopes })
      setClients(current => [client, ...current.filter(item => item.id !== client.id)])
      setName('')
      setMessage(`Local client “${client.name}” was created with the selected scopes.`)
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : 'Could not create the local MCP client.')
    } finally {
      setCreating(false)
    }
  }

  const handleCopy = async (client: AgentAccessLocalClient, target: 'codex' | 'json' | 'claude') => {
    if (!enabled) return
    const generation = setupGeneration.current
    try {
      const value = target === 'json' ? formatLocalAgentClientJson(client) : target === 'codex' ? formatLocalAgentClientCodexConfig(client) : formatLocalAgentClientClaudeCommand(client)
      await navigator.clipboard.writeText(value)
      if (generation !== setupGeneration.current) return
      setCopied(`${client.id}:${target}`)
      window.setTimeout(() => setCopied(current => current === `${client.id}:${target}` ? null : current), 1800)
    } catch {
      if (generation === setupGeneration.current) setError('Clipboard access is unavailable. Select and copy the configuration shown for this client.')
    }
  }

  const handleDownload = async (client: AgentAccessLocalClient) => {
    if (!enabled || client.revoked || downloadingClientId) return
    const generation = setupGeneration.current
    setDownloadingClientId(client.id)
    setError(null)
    setMessage(null)
    try {
      const bundle = await fetchAgentAccessLocalClientExtension(client.id)
      if (generation !== setupGeneration.current) return
      const url = URL.createObjectURL(bundle)
      const anchor = document.createElement('a')
      anchor.href = url
      anchor.download = `maestro-local-${client.id}.mcpb`
      anchor.style.display = 'none'
      document.body.appendChild(anchor)
      anchor.click()
      anchor.remove()
      window.setTimeout(() => URL.revokeObjectURL(url), 1000)
      setMessage(`Claude Desktop extension downloaded for “${client.name}”. Install it in Claude Desktop on this same computer.`)
    } catch (requestError) {
      if (generation === setupGeneration.current) setError(requestError instanceof Error ? requestError.message : 'Could not download the Claude Desktop extension.')
    } finally {
      if (generation === setupGeneration.current) setDownloadingClientId(null)
    }
  }

  const activeClients = clients.filter(client => !client.revoked)
  const revokedClients = clients.filter(client => client.revoked)

  return (
    <section className="space-y-3 rounded-xl border border-accent-blue/20 bg-accent-blue/5 p-4" aria-labelledby="agent-access-local-heading">
      <div className="flex items-start gap-3">
        <div className="mt-0.5 rounded-lg bg-accent-blue/10 p-2 text-accent-blue"><ShieldCheck size={16} /></div>
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2">
            <h4 id="agent-access-local-heading" className="text-xs font-semibold text-text-primary">Local MCP client setup</h4>
            <span className="rounded-full bg-accent-blue/10 px-2 py-0.5 text-[9px] font-semibold text-accent-blue">Recommended on this computer</span>
          </div>
          <p className="mt-1 text-[10px] leading-relaxed text-text-secondary">Runs Maestro’s MCP server directly on this computer over STDIO. It does not need Tailscale, a network URL, or certificate setup. An owner must approve each local client and its chosen scopes.</p>
        </div>
        <button type="button" onClick={() => void refresh()} disabled={loading || !ownerSession} className="rounded-lg p-2 text-text-muted hover:bg-bg-hover hover:text-text-primary disabled:opacity-40" aria-label="Refresh local MCP clients">
          <RefreshCw size={14} className={loading ? 'animate-spin' : ''} />
        </button>
      </div>

      {!ownerSession ? (
        <p className="rounded-lg bg-bg-primary p-3 text-[10px] leading-relaxed text-text-muted">Sign in as the owner to create or review local client identities.</p>
      ) : !enabled ? (
        <p className="rounded-lg bg-bg-primary p-3 text-[10px] leading-relaxed text-text-muted">Enable Local MCP above to create a client. Remote MCP can stay off; local setup does not require Tailscale or a certificate.</p>
      ) : !available ? (
        <p className="rounded-lg bg-bg-primary p-3 text-[10px] leading-relaxed text-text-muted">Local MCP is unavailable on this installation.</p>
      ) : (
        <form className="space-y-3 rounded-lg border border-border bg-bg-primary p-3" onSubmit={event => void handleCreate(event)}>
          <div className="space-y-1.5">
            <label htmlFor="agent-access-local-client-name" className="text-[10px] font-medium text-text-secondary">Local client name</label>
            <input id="agent-access-local-client-name" value={name} maxLength={64} onChange={event => setName(event.target.value)} placeholder="e.g. Codex on this PC" disabled={creating} className="min-h-10 w-full min-w-0 rounded-lg border border-border bg-bg-secondary px-3 py-2 text-xs text-text-primary outline-none focus:border-accent-blue disabled:opacity-50" />
          </div>
          <fieldset className="space-y-2">
            <legend className="text-[10px] font-medium text-text-secondary">Choose this client’s access scopes</legend>
            <div className="grid grid-cols-1 gap-2">
              {SCOPE_CHOICES.map(scope => (
                <label key={scope.id} className="flex min-w-0 items-start gap-2 rounded-lg border border-border bg-bg-secondary px-2.5 py-2">
                  <input type="checkbox" checked={scopes.includes(scope.id)} disabled={creating} onChange={event => setScopes(current => event.target.checked ? [...current, scope.id] : current.filter(value => value !== scope.id))} className="mt-0.5 accent-blue-500" />
                  <span className="min-w-0"><span className="block text-[10px] font-semibold text-text-primary">{scope.label}</span><span className="block text-[9px] leading-relaxed text-text-muted">{scope.description}</span></span>
                </label>
              ))}
            </div>
          </fieldset>
          <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
            <p className="text-[9px] leading-relaxed text-text-muted">Only the scopes you check are granted. You can revoke later in Agent grants.</p>
            <button type="submit" disabled={creating || !name.trim() || scopes.length === 0} className="inline-flex min-h-10 shrink-0 items-center justify-center gap-2 rounded-lg bg-accent-blue px-3 text-[10px] font-semibold text-bg-primary disabled:opacity-40">
              {creating ? <LoaderCircle size={13} className="animate-spin" /> : null}{creating ? 'Creating…' : 'Create local client'}
            </button>
          </div>
        </form>
      )}

      {error && <p className="rounded-lg border border-amber-500/30 bg-amber-500/5 p-2.5 text-[10px] leading-relaxed text-amber-100" role="alert">{error}</p>}
      {message && <p className="rounded-lg border border-emerald-500/30 bg-emerald-500/5 p-2.5 text-[10px] leading-relaxed text-emerald-100" role="status">{message}</p>}

      {ownerSession && (
        <div className="space-y-2">
          <div className="flex items-center justify-between gap-2">
            <h5 className="text-[10px] font-semibold text-text-primary">Local clients</h5>
            <span className="rounded-full bg-bg-primary px-2 py-1 text-[9px] text-text-muted">{activeClients.length} active</span>
          </div>
          {loading && clients.length === 0 ? <p className="text-[10px] text-text-muted" role="status">Loading local clients…</p> : loadError && clients.length === 0 ? <p className="rounded-lg bg-bg-primary p-3 text-[10px] text-text-muted">Could not load local clients. Refresh to try again.</p> : activeClients.length === 0 ? <p className="rounded-lg bg-bg-primary p-3 text-[10px] text-text-muted">{revokedClients.length > 0 ? 'No active local clients.' : 'No local clients have been created.'}</p> : (
            <div className="space-y-2">
              {activeClients.map(client => <LocalClientSetup key={client.id} client={client} enabled={enabled} copied={copied} downloading={downloadingClientId === client.id} onCopy={target => void handleCopy(client, target)} onDownload={() => void handleDownload(client)} />)}
            </div>
          )}
          {revokedClients.length > 0 && <p className="text-[9px] text-text-muted">{revokedClients.length} revoked local {revokedClients.length === 1 ? 'client is' : 'clients are'} hidden from setup.</p>}
        </div>
      )}

      <p className="rounded-lg border border-border bg-bg-primary p-2.5 text-[9px] leading-relaxed text-text-muted">For Claude Desktop, download the .mcpb package for the local client above, then open Claude Desktop → Settings → Extensions → Install Extension and select the downloaded file. Install it on the same computer where this Maestro instance runs. Claude Desktop supplies Node.js for the extension; it starts a local STDIO relay and does not use a private HTTP connector URL.</p>
    </section>
  )
}

function LocalClientSetup({ client, enabled, copied, downloading, onCopy, onDownload }: {
  client: AgentAccessLocalClient
  enabled: boolean
  copied: string | null
  downloading: boolean
  onCopy: (target: 'codex' | 'json' | 'claude') => void
  onDownload: () => void
}) {
  const codexConfig = formatLocalAgentClientCodexConfig(client)
  const config = formatLocalAgentClientJson(client)
  const command = formatLocalAgentClientClaudeCommand(client)
  const validCommand = Boolean(client.command && client.args && !client.revoked)
  return (
    <details className="rounded-lg border border-border bg-bg-primary">
      <summary className="flex min-h-11 cursor-pointer list-none flex-wrap items-center justify-between gap-2 px-3 py-2">
        <span className="min-w-0"><span className="block truncate text-[10px] font-semibold text-text-primary">{client.name}</span><span className="mt-0.5 block text-[9px] text-text-muted">Scopes: {client.scopes.join(', ') || 'none'}</span></span>
        <span className="shrink-0 text-[9px] text-text-muted">Setup</span>
      </summary>
      <div className="space-y-2 border-t border-border p-3">
        {!validCommand ? <p className="text-[9px] leading-relaxed text-text-muted">Maestro has not returned a safe launch command for this client. Refresh the client list before copying setup.</p> : (
          <>
            <pre className="max-h-40 overflow-auto whitespace-pre-wrap break-all rounded-lg bg-bg-secondary p-2 font-mono text-[9px] leading-relaxed text-text-secondary" aria-label={`${client.name} Codex config`}>{codexConfig}</pre>
            <pre className="max-h-40 overflow-auto whitespace-pre-wrap break-all rounded-lg bg-bg-secondary p-2 font-mono text-[9px] leading-relaxed text-text-secondary" aria-label={`${client.name} ChatGPT STDIO JSON config`}>{config}</pre>
            <pre className="max-h-24 overflow-auto whitespace-pre-wrap break-all rounded-lg bg-bg-secondary p-2 font-mono text-[9px] leading-relaxed text-text-secondary" aria-label={`${client.name} Claude Code command`}>{command}</pre>
            <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
              <button type="button" onClick={onDownload} disabled={!enabled || !validCommand || downloading} className="inline-flex min-h-9 items-center justify-center gap-1.5 rounded-lg bg-accent-blue px-2.5 text-[9px] font-semibold text-bg-primary disabled:opacity-40">
                {downloading ? <LoaderCircle size={12} className="animate-spin" /> : null}{downloading ? 'Preparing extension…' : 'Download Claude Desktop extension'}
              </button>
              <button type="button" onClick={() => onCopy('codex')} disabled={!enabled || !validCommand} className="inline-flex min-h-9 items-center justify-center gap-1.5 rounded-lg border border-border px-2.5 text-[9px] font-medium text-text-secondary hover:bg-bg-hover disabled:opacity-40">
                {copied === `${client.id}:codex` ? <Check size={12} /> : <Copy size={12} />}{copied === `${client.id}:codex` ? 'Codex config copied' : 'Copy Codex config.toml'}
              </button>
              <button type="button" onClick={() => onCopy('json')} disabled={!enabled || !validCommand} className="inline-flex min-h-9 items-center justify-center gap-1.5 rounded-lg border border-border px-2.5 text-[9px] font-medium text-text-secondary hover:bg-bg-hover disabled:opacity-40">
                {copied === `${client.id}:json` ? <Check size={12} /> : <Copy size={12} />}{copied === `${client.id}:json` ? 'ChatGPT config copied' : 'Copy ChatGPT STDIO JSON'}
              </button>
              <button type="button" onClick={() => onCopy('claude')} disabled={!enabled || !validCommand} className="inline-flex min-h-9 items-center justify-center gap-1.5 rounded-lg border border-border px-2.5 text-[9px] font-medium text-text-secondary hover:bg-bg-hover disabled:opacity-40">
                {copied === `${client.id}:claude` ? <Check size={12} /> : <Copy size={12} />}{copied === `${client.id}:claude` ? 'Claude Code command copied' : 'Copy Claude Code command'}
              </button>
            </div>
            <p className="text-[9px] leading-relaxed text-text-muted">The copied configs and extension contain only the executable, arguments, and this client’s ID. Maestro keeps local credentials in its protected settings; no token or credential path is packaged.</p>
          </>
        )}
      </div>
    </details>
  )
}

export function AgentGenerationDefaults({ ownerSession }: { ownerSession: boolean }) {
  const [data, setData] = useState<AgentAccessCreativeDefaults | null>(null)
  const [draft, setDraft] = useState<Record<string, AgentAccessCreativeDefault>>({})
  const [loading, setLoading] = useState(false)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [message, setMessage] = useState<string | null>(null)

  const refresh = useCallback(async () => {
    if (!ownerSession) {
      setData(null)
      setDraft({})
      return
    }
    setLoading(true)
    setError(null)
    try {
      const next = await fetchAgentAccessCreativeDefaults()
      setData(next)
      setDraft(cloneDefaults(next.defaults))
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : 'Could not load agent generation defaults.')
    } finally {
      setLoading(false)
    }
  }, [ownerSession])

  useEffect(() => { void refresh() }, [refresh])

  const dirty = useMemo(() => Boolean(data && JSON.stringify(normalizeDefaults(draft)) !== JSON.stringify(normalizeDefaults(data.defaults))), [data, draft])

  const save = async () => {
    if (!ownerSession || !dirty || saving) return
    setSaving(true)
    setError(null)
    setMessage(null)
    try {
      const next = await updateAgentAccessCreativeDefaults(normalizeDefaults(draft))
      setData(next)
      setDraft(cloneDefaults(next.defaults))
      setMessage('Saved. The values shown here are the defaults Maestro stored.')
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : 'Could not save agent generation defaults.')
    } finally {
      setSaving(false)
    }
  }

  const update = (workflowId: string, field: keyof AgentAccessCreativeDefault, value: string) => {
    setDraft(current => {
      const next = cloneDefaults(current)
      const entry = { ...(next[workflowId] || {}) }
      if (field === 'model_type') {
        if (value) entry.model_type = value
        else delete entry.model_type
      } else if (field === 'duration_seconds') {
        if (value.trim() === '') delete entry.duration_seconds
        else {
          const parsed = Number(value)
          if (Number.isFinite(parsed) && parsed >= 0) entry.duration_seconds = parsed
        }
      } else if (field === 'resolution') {
        if (value.trim()) entry.resolution = value.trim()
        else delete entry.resolution
      }
      if (Object.keys(entry).length) next[workflowId] = entry
      else delete next[workflowId]
      return next
    })
    setMessage(null)
  }

  return (
    <details className="rounded-xl border border-border bg-bg-secondary">
      <summary className="flex min-h-12 cursor-pointer list-none items-center justify-between gap-3 px-4 text-left">
        <span className="min-w-0"><span className="block text-xs font-semibold text-text-primary">Agent generation defaults</span><span className="mt-0.5 block text-[10px] text-text-muted">Optional model, duration, and resolution preferences</span></span>
        <span className="shrink-0 text-[10px] text-text-muted">{loading ? 'Loading…' : 'Expand'}</span>
      </summary>
      <div className="space-y-3 border-t border-border p-3 sm:p-4">
        {!ownerSession ? <p className="text-[10px] leading-relaxed text-text-muted">Sign in as the owner to view or edit agent generation defaults.</p> : loading && !data ? <p className="text-[10px] text-text-muted" role="status">Loading saved defaults…</p> : !data ? (
          <div className="space-y-2"><p className="text-[10px] leading-relaxed text-text-muted">Agent generation defaults are unavailable. Refresh to try again.</p><button type="button" onClick={() => void refresh()} disabled={loading} className="inline-flex min-h-9 items-center gap-1.5 rounded-lg border border-border px-2.5 text-[9px] text-text-secondary disabled:opacity-50"><RefreshCw size={12} />Refresh</button></div>
        ) : (
          <>
            <p className="text-[9px] leading-relaxed text-text-muted">Blank model inherits Maestro’s compatible choice. Blank duration and resolution use Maestro’s defaults. Speech duration 0 means Auto. These preferences apply to Agent requests only; saving here does not change Studio or Director selections.</p>
            <div className="space-y-2">
              {data.workflows.map(workflow => <WorkflowDefaultRow key={workflow.id} id={workflow.id} label={workflow.label} models={data.models} value={draft[workflow.id] || {}} savedValue={data.defaults[workflow.id] || {}} onChange={(field, value) => update(workflow.id, field, value)} />)}
            </div>
            {error && <p className="rounded-lg border border-amber-500/30 bg-amber-500/5 p-2.5 text-[10px] leading-relaxed text-amber-100" role="alert">{error}</p>}
            {message && <p className="rounded-lg border border-emerald-500/30 bg-emerald-500/5 p-2.5 text-[10px] leading-relaxed text-emerald-100" role="status">{message}</p>}
            <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
              <button type="button" onClick={() => void refresh()} disabled={loading || saving} className="inline-flex min-h-9 items-center gap-1.5 self-start rounded-lg border border-border px-2.5 text-[9px] text-text-secondary disabled:opacity-50"><RefreshCw size={12} className={loading ? 'animate-spin' : ''} />Reload saved values</button>
              <button type="button" onClick={() => void save()} disabled={!dirty || saving} className="inline-flex min-h-9 items-center justify-center gap-1.5 rounded-lg bg-accent-blue px-3 text-[10px] font-semibold text-bg-primary disabled:opacity-40">{saving ? <LoaderCircle size={12} className="animate-spin" /> : null}{saving ? 'Saving…' : 'Save defaults'}</button>
            </div>
          </>
        )}
      </div>
    </details>
  )
}

function WorkflowDefaultRow({ id, label, models, value, savedValue, onChange }: {
  id: string
  label: string
  models: AgentAccessCreativeDefaults['models']
  value: AgentAccessCreativeDefault
  savedValue: AgentAccessCreativeDefault
  onChange: (field: keyof AgentAccessCreativeDefault, value: string) => void
}) {
  const compatible = models.filter(model => model.workflows.includes(id))
  const downloaded = compatible.filter(model => model.is_downloaded)
  const savedModel = models.find(model => model.model_type === value.model_type)
  const lastSavedModel = models.find(model => model.model_type === savedValue.model_type)
  const options = [...(savedModel && !downloaded.some(model => model.model_type === savedModel.model_type) ? [savedModel] : []), ...downloaded]
  const imageWorkflow = id.startsWith('image.')
  return (
    <div className="grid grid-cols-1 gap-2 rounded-lg border border-border bg-bg-primary p-3 sm:grid-cols-2">
      <div className="min-w-0 sm:col-span-2"><h5 className="text-[10px] font-semibold text-text-primary">{label}</h5><p className="text-[9px] text-text-muted">{id}</p></div>
      <label className="min-w-0 space-y-1 text-[9px] text-text-secondary">
        <span>Model</span>
        <select aria-label={`${label} model`} value={value.model_type || ''} onChange={event => onChange('model_type', event.target.value)} className="min-h-9 w-full min-w-0 rounded-lg border border-border bg-bg-secondary px-2 text-[10px] text-text-primary">
          <option value="">Inherit compatible Maestro choice</option>
          {options.map(model => <option key={model.model_type} value={model.model_type} disabled={!model.is_downloaded || !model.workflows.includes(id)}>{model.name}{!model.workflows.includes(id) ? ' (not compatible)' : model.is_downloaded ? '' : ' (not downloaded)'}</option>)}
        </select>
      </label>
      {!imageWorkflow && (
        <label className="min-w-0 space-y-1 text-[9px] text-text-secondary">
          <span>Duration (seconds){id === 'audio.speech' ? ' · 0 = Auto' : ''}</span>
          <input aria-label={`${label} duration seconds`} type="number" min="0" step="1" value={value.duration_seconds ?? ''} placeholder={id === 'audio.speech' ? 'Default; enter 0 for Auto' : 'Default / Auto'} onChange={event => onChange('duration_seconds', event.target.value)} className="min-h-9 w-full min-w-0 rounded-lg border border-border bg-bg-secondary px-2 text-[10px] text-text-primary" />
        </label>
      )}
      <label className="min-w-0 space-y-1 text-[9px] text-text-secondary">
        <span>Resolution</span>
        <input aria-label={`${label} resolution`} type="text" value={value.resolution || ''} placeholder="Default" onChange={event => onChange('resolution', event.target.value)} className="min-h-9 w-full min-w-0 rounded-lg border border-border bg-bg-secondary px-2 text-[10px] text-text-primary" />
      </label>
      <p className="min-w-0 break-all text-[9px] leading-relaxed text-text-muted sm:col-span-2">Form value: {savedModel?.name || value.model_type || 'Inherit Maestro choice'} · duration: {value.duration_seconds === undefined ? 'Default / Auto' : `${value.duration_seconds} seconds${id === 'audio.speech' && value.duration_seconds === 0 ? ' (Auto)' : ''}`} · resolution: {value.resolution || 'Default'}</p>
      <p className="min-w-0 break-all text-[9px] leading-relaxed text-text-muted sm:col-span-2">Last saved by Maestro: model_type {savedValue.model_type || 'unset (inherit)'}{lastSavedModel ? ` (${lastSavedModel.name})` : ''} · duration_seconds {savedValue.duration_seconds === undefined ? 'unset (default / Auto)' : `${savedValue.duration_seconds}${id === 'audio.speech' && savedValue.duration_seconds === 0 ? ' (Auto)' : ''}`} · resolution {savedValue.resolution || 'unset (default)'}</p>
    </div>
  )
}

function cloneDefaults(value: Record<string, AgentAccessCreativeDefault>): Record<string, AgentAccessCreativeDefault> {
  return Object.fromEntries(Object.entries(value || {}).map(([key, entry]) => [key, { ...entry }]))
}

function normalizeDefaults(value: Record<string, AgentAccessCreativeDefault>): Record<string, AgentAccessCreativeDefault> {
  return Object.fromEntries(
    Object.entries(value)
      .sort(([a], [b]) => a.localeCompare(b))
      .map(([key, entry]) => [key, Object.fromEntries(
        Object.entries(entry)
          .filter(([, item]) => item !== undefined)
          .sort(([a], [b]) => a.localeCompare(b)),
      )]),
  )
}
