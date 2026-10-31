import { useCallback, useEffect, useRef, useState, type FormEvent } from 'react'
import { ExternalLink, LockKeyhole, X } from 'lucide-react'
import {
  AGENT_ACCESS_AUTH_REQUIRED_CODE,
  AGENT_ACCESS_AUTH_REQUIRED_EVENT,
  AGENT_ACCESS_SESSION_RESTORED_EVENT,
  AgentAccessApiError,
  fetchAgentAccessStatus,
  loginAgentAccessOwner,
  requestAgentAccessConsoleCode,
  type AgentAccessStatus,
} from '../api/agentAccess'

type AuthRequiredEvent = CustomEvent<{ code?: string; login_url?: string | null }>

export function AgentAccessSessionGate() {
  const [visible, setVisible] = useState(false)
  const [status, setStatus] = useState<AgentAccessStatus | null>(null)
  const [controllerUrl, setControllerUrl] = useState<string | null>(null)
  const [loginCode, setLoginCode] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [backendMissing, setBackendMissing] = useState(false)
  const [submitting, setSubmitting] = useState(false)
  const [requestingCode, setRequestingCode] = useState(false)
  const [codeMessage, setCodeMessage] = useState<string | null>(null)
  const [dismissed, setDismissed] = useState(false)
  const dismissedRef = useRef(false)
  const visibleRef = useRef(false)
  const requestInFlight = useRef(false)
  const codeInput = useRef<HTMLInputElement>(null)

  const openSignIn = useCallback(() => {
    if (requestInFlight.current || visibleRef.current || dismissedRef.current) return
    requestInFlight.current = true
    void fetchAgentAccessStatus().then(next => {
      if (dismissedRef.current) return
      setStatus(next)
      const secureUrl = secureControllerUrl(next.controller_url)
      if (secureUrl) setControllerUrl(secureUrl)
      setBackendMissing(false)
      if (next.authenticated && next.owner) return
      visibleRef.current = true
      setVisible(true)
      setError(null)
    }).catch(requestError => {
      if (dismissedRef.current) return
      visibleRef.current = true
      setVisible(true)
      setStatus(null)
      setBackendMissing(requestError instanceof AgentAccessApiError && requestError.status === 404)
      setError(requestError instanceof Error ? requestError.message : 'Could not check sign-in status.')
    }).finally(() => { requestInFlight.current = false })
  }, [])

  useEffect(() => {
    const onAuthRequired = (rawEvent: Event) => {
      const event = rawEvent as AuthRequiredEvent
      if (event.detail?.code !== AGENT_ACCESS_AUTH_REQUIRED_CODE) return
      if (visibleRef.current || dismissedRef.current) return
      const eventUrl = secureControllerUrl(event.detail.login_url)
      setControllerUrl(eventUrl)
      openSignIn()
    }

    window.addEventListener(AGENT_ACCESS_AUTH_REQUIRED_EVENT, onAuthRequired)
    return () => window.removeEventListener(AGENT_ACCESS_AUTH_REQUIRED_EVENT, onAuthRequired)
  }, [openSignIn])

  const loginAllowed = isSecureOrigin()

  const handleLogin = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    if (!loginAllowed || !loginCode.trim() || submitting || requestingCode) return
    setSubmitting(true)
    setError(null)
    try {
      await loginAgentAccessOwner(loginCode.trim())
      setLoginCode('')
      visibleRef.current = false
      setVisible(false)
      dismissedRef.current = false
      setDismissed(false)
      setBackendMissing(false)
      window.dispatchEvent(new Event(AGENT_ACCESS_SESSION_RESTORED_EVENT))
    } catch (loginError) {
      if (loginError instanceof AgentAccessApiError && loginError.status === 404) {
        setBackendMissing(true)
        setError(null)
      } else {
        setError(loginError instanceof Error ? loginError.message : 'Owner sign-in failed.')
      }
    } finally {
      setSubmitting(false)
    }
  }

  const dismiss = () => {
    dismissedRef.current = true
    visibleRef.current = false
    setDismissed(true)
    setVisible(false)
  }

  const requestFreshCode = async () => {
    if (requestingCode || submitting) return
    setRequestingCode(true)
    setError(null)
    setCodeMessage(null)
    try {
      const result = await requestAgentAccessConsoleCode()
      setLoginCode('')
      const seconds = Number.isFinite(result.expires_in) ? Math.max(1, result.expires_in) : 600
      const expiry = seconds < 60 ? `${Math.ceil(seconds)} seconds` : `${Math.ceil(seconds / 60)} minute${seconds > 60 ? 's' : ''}`
      setCodeMessage(`Ready in Pinokio → this Maestro app → Agent Access Terminal. Copy the latest “Browser code:” and paste it here. It works once and expires in about ${expiry}.`)
      codeInput.current?.focus()
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : 'Could not request a fresh code.')
    } finally { setRequestingCode(false) }
  }

  if (!visible) return dismissed ? (
    <button type="button" onClick={() => { dismissedRef.current = false; setDismissed(false); openSignIn() }} className="fixed bottom-4 right-4 z-[149] inline-flex items-center gap-2 rounded-lg border border-border bg-bg-secondary px-4 py-3 text-xs text-text-primary shadow-lg">
      <LockKeyhole size={14} /> Sign in to Maestro
    </button>
  ) : null

  return (
    <div className="fixed inset-0 z-[150] flex items-center justify-center bg-black/65 p-3 sm:p-6" role="presentation">
      <section
        className="max-h-[calc(100dvh-1.5rem)] w-full max-w-md overflow-y-auto rounded-2xl border border-border bg-bg-secondary shadow-2xl"
        role="dialog"
        aria-modal="true"
        aria-labelledby="agent-access-session-title"
      >
        <div className="flex items-start gap-3 border-b border-border px-5 py-4">
          <div className="mt-0.5 rounded-lg bg-accent-blue/10 p-2 text-accent-blue">
            <LockKeyhole size={17} />
          </div>
          <div className="min-w-0 flex-1">
            <h2 id="agent-access-session-title" className="text-sm font-semibold text-text-primary">Sign in to Maestro</h2>
            <p className="mt-1 text-xs leading-relaxed text-text-secondary">
              Sign in once in this browser. Your current work stays open behind this panel.
            </p>
          </div>
          <button
            type="button"
            onClick={dismiss}
            className="rounded-md p-1.5 text-text-muted transition-colors hover:bg-bg-hover hover:text-text-primary"
            aria-label="Dismiss sign-in"
          >
            <X size={16} />
          </button>
        </div>

        <div className="space-y-4 px-5 py-4">
          {backendMissing ? (
            <div className="rounded-lg border border-amber-500/30 bg-amber-500/10 p-3 text-xs leading-relaxed text-text-secondary">
              Agent Access is not available from this backend yet. Restart Maestro after installing the update, then sign in again.
            </div>
          ) : !loginAllowed ? (
            <div className="rounded-lg border border-amber-500/30 bg-amber-500/10 p-3 text-xs leading-relaxed text-text-secondary">
              Owner sign-in needs HTTPS or a local browser address. Open Maestro through its secure controller URL to continue.
            </div>
          ) : (
            <>
              {status?.owner_configured === false && (
                <p className="text-xs leading-relaxed text-text-secondary">
                  Signing in also protects this Maestro installation. Other browsers will need to sign in once, even while agent connections are off.
                </p>
              )}
              <div className="space-y-2 rounded-lg border border-border bg-bg-primary p-3 text-xs text-text-secondary">
                <button type="button" onClick={() => void requestFreshCode()} disabled={requestingCode || submitting} className="min-h-10 w-full rounded-lg border border-border px-3 py-2 font-medium text-text-primary hover:bg-bg-hover disabled:opacity-50">
                  {requestingCode ? 'Requesting code…' : 'Request a fresh code'}
                </button>
                <ol className="list-decimal space-y-1 pl-4 leading-relaxed" id="agent-access-sign-in-steps">
                  <li>Open this Maestro app in Pinokio, then <strong>Agent Access Terminal</strong>.</li>
                  <li>Copy the newest line labeled <strong>Browser code:</strong> and paste it below.</li>
                </ol>
                {codeMessage && <p className="text-accent-blue" role="status">{codeMessage}</p>}
              </div>
              <form onSubmit={handleLogin} className="space-y-3">
                <label className="block text-xs text-text-secondary" htmlFor="agent-access-session-code">
                  Browser sign-in code
                </label>
                <input
                  id="agent-access-session-code"
                  ref={codeInput}
                  autoFocus
                  autoComplete="one-time-code"
                  autoCapitalize="none"
                  spellCheck={false}
                  aria-describedby="agent-access-sign-in-steps"
                  inputMode="text"
                  value={loginCode}
                  onChange={event => setLoginCode(event.target.value)}
                  className="w-full rounded-lg border border-border bg-bg-primary px-3 py-2.5 font-mono text-sm text-text-primary outline-none focus:border-accent-blue"
                  placeholder="Paste the code from Pinokio"
                  disabled={submitting || requestingCode}
                />
                <button
                  type="submit"
                  disabled={submitting || requestingCode || !loginCode.trim()}
                  className="w-full rounded-lg bg-accent-blue px-4 py-2.5 text-xs font-medium text-white transition-opacity hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-45"
                >
                  {submitting ? 'Signing in…' : 'Sign in to Maestro'}
                </button>
              </form>
            </>
          )}

          {error && <p className="rounded-lg border border-red-500/30 bg-red-500/10 p-3 text-xs text-red-300" role="alert">{error}</p>}

          {controllerUrl && !loginAllowed && (
            <a
              href={controllerUrl}
              target="_blank"
              rel="noreferrer"
              className="inline-flex min-h-10 items-center gap-2 rounded-lg border border-border px-3 py-2 text-xs text-text-primary transition-colors hover:bg-bg-hover"
            >
              Open secure Maestro controller <ExternalLink size={13} />
            </a>
          )}

          <p className="text-[10px] leading-relaxed text-text-muted">
            Maestro remembers this browser until you sign out or the session expires. You approve each agent separately.
          </p>
        </div>
      </section>
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

function secureControllerUrl(value: unknown): string | null {
  if (typeof value !== 'string' || !value) return null
  try {
    const url = new URL(value, window.location.href)
    if (url.protocol !== 'https:' || url.username || url.password) return null
    return url.href
  } catch {
    return null
  }
}
