import { useEffect, useRef, useState, type FormEvent } from 'react'
import { ExternalLink, LockKeyhole, X } from 'lucide-react'
import {
  AGENT_ACCESS_AUTH_REQUIRED_CODE,
  AGENT_ACCESS_AUTH_REQUIRED_EVENT,
  AGENT_ACCESS_SESSION_RESTORED_EVENT,
  AgentAccessApiError,
  fetchAgentAccessStatus,
  loginAgentAccessOwner,
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
  const requestInFlight = useRef(false)

  useEffect(() => {
    const onAuthRequired = (rawEvent: Event) => {
      const event = rawEvent as AuthRequiredEvent
      if (event.detail?.code !== AGENT_ACCESS_AUTH_REQUIRED_CODE) return
      setVisible(true)
      setError(null)
      const eventUrl = secureControllerUrl(event.detail.login_url)
      setControllerUrl(eventUrl)
      if (requestInFlight.current) return
      requestInFlight.current = true
      void fetchAgentAccessStatus().then(next => {
        setStatus(next)
        const secureUrl = secureControllerUrl(next.controller_url)
        if (secureUrl) setControllerUrl(secureUrl)
        setBackendMissing(false)
      }).catch(requestError => {
        setStatus(null)
        setBackendMissing(requestError instanceof AgentAccessApiError && requestError.status === 404)
        if (!(requestError instanceof AgentAccessApiError && requestError.status === 404)) {
          setError(requestError instanceof Error ? requestError.message : 'Could not check Agent Access status.')
        }
      }).finally(() => {
        requestInFlight.current = false
      })
    }

    window.addEventListener(AGENT_ACCESS_AUTH_REQUIRED_EVENT, onAuthRequired)
    return () => window.removeEventListener(AGENT_ACCESS_AUTH_REQUIRED_EVENT, onAuthRequired)
  }, [])

  const loginAllowed = isSecureOrigin()

  const handleLogin = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    if (!loginAllowed || !loginCode.trim() || submitting) return
    setSubmitting(true)
    setError(null)
    try {
      await loginAgentAccessOwner(loginCode.trim())
      setLoginCode('')
      setVisible(false)
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

  if (!visible) return null

  return (
    <div className="fixed inset-0 z-[150] flex items-center justify-center bg-black/65 p-3 sm:p-6" role="presentation">
      <section
        className="w-full max-w-md overflow-hidden rounded-2xl border border-border bg-bg-secondary shadow-2xl"
        role="dialog"
        aria-modal="true"
        aria-labelledby="agent-access-session-title"
      >
        <div className="flex items-start gap-3 border-b border-border px-5 py-4">
          <div className="mt-0.5 rounded-lg bg-accent-blue/10 p-2 text-accent-blue">
            <LockKeyhole size={17} />
          </div>
          <div className="min-w-0 flex-1">
            <h2 id="agent-access-session-title" className="text-sm font-semibold text-text-primary">Owner session expired</h2>
            <p className="mt-1 text-xs leading-relaxed text-text-secondary">
              Sign in again to manage Maestro Agent Access. Your current work stays open behind this panel.
            </p>
          </div>
          <button
            type="button"
            onClick={() => setVisible(false)}
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
                  First, request a one-time code from the Agent Access tab. Maestro prints it only in the Pinokio terminal.
                </p>
              )}
              <form onSubmit={handleLogin} className="space-y-3">
                <label className="block text-xs text-text-secondary" htmlFor="agent-access-session-code">
                  One-time owner code
                </label>
                <input
                  id="agent-access-session-code"
                  autoFocus
                  autoComplete="one-time-code"
                  inputMode="text"
                  value={loginCode}
                  onChange={event => setLoginCode(event.target.value)}
                  className="w-full rounded-lg border border-border bg-bg-primary px-3 py-2.5 font-mono text-sm text-text-primary outline-none focus:border-accent-blue"
                  placeholder="Paste the code from Pinokio"
                  disabled={submitting}
                />
                <button
                  type="submit"
                  disabled={submitting || !loginCode.trim()}
                  className="w-full rounded-lg bg-accent-blue px-4 py-2.5 text-xs font-medium text-white transition-opacity hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-45"
                >
                  {submitting ? 'Signing in…' : 'Sign in to Agent Access'}
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
            This sign-in only restores the local owner session. Agent connections keep their separate grants and scopes.
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
