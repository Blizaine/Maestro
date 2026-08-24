import { useEffect, useState, type ReactNode } from 'react'
import {
  Bell,
  BellRing,
  Check,
  Copy,
  ExternalLink,
  MonitorSpeaker,
  RefreshCw,
  ShieldCheck,
  Smartphone,
  Volume2,
} from 'lucide-react'
import { QRCodeSVG } from 'qrcode.react'
import { useStore } from '../../stores/useStore'
import * as api from '../../api/client'
import type { TailscaleRemoteAccessStatus } from '../../types'
import {
  disableBackgroundPush,
  enableBackgroundPush,
  getBackgroundPushState,
  getBrowserNotificationAvailability,
  getDeviceNotificationPreferences,
  playDeviceNotificationChime,
  requestBrowserNotificationPermission,
  subscribeDeviceNotificationPreferences,
  syncBackgroundPush,
  testBackgroundPush,
  testBrowserNotification,
  updateDeviceNotificationPreferences,
  type BackgroundPushState,
  type BrowserNotificationAvailability,
  type DeviceNotificationPreferences,
} from '../../lib/notifications'

interface ToggleProps {
  checked: boolean
  onChange: (checked: boolean) => void
  label: string
  description?: ReactNode
  disabled?: boolean
}

function Toggle({ checked, onChange, label, description, disabled = false }: ToggleProps) {
  return (
    <div className={`flex items-start justify-between gap-3 ${disabled ? 'opacity-50' : ''}`}>
      <div className="min-w-0 flex-1">
        <div className="text-xs text-text-primary">{label}</div>
        {description && (
          <div className="mt-0.5 text-[10px] leading-relaxed text-text-muted">{description}</div>
        )}
      </div>
      <button
        type="button"
        role="switch"
        aria-checked={checked}
        aria-label={label}
        disabled={disabled}
        onClick={() => onChange(!checked)}
        className={`relative mt-0.5 inline-flex h-5 w-9 shrink-0 items-center rounded-full transition-colors ${
          checked ? 'bg-accent-blue' : 'border border-border bg-bg-primary'
        } ${disabled ? 'cursor-not-allowed' : ''}`}
      >
        <span
          className={`inline-block h-3.5 w-3.5 rounded-full border border-border bg-white transition-transform ${
            checked ? 'translate-x-4' : 'translate-x-0.5'
          }`}
        />
      </button>
    </div>
  )
}

function permissionLabel(availability: BrowserNotificationAvailability): string {
  if (!availability.supported) return 'Not supported by this browser'
  if (!availability.secure) return 'Requires HTTPS or localhost'
  if (availability.permission === 'granted') return 'Browser permission granted'
  if (availability.permission === 'denied') return 'Blocked in browser settings'
  return 'Permission will be requested when enabled'
}

export function NotificationSettingsPanel() {
  const systemConfig = useStore(state => state.systemConfig)
  const updateSystemConfig = useStore(state => state.updateSystemConfig)
  const [preferences, setPreferences] = useState<DeviceNotificationPreferences>(
    getDeviceNotificationPreferences,
  )
  const [availability, setAvailability] = useState(getBrowserNotificationAvailability)
  const [message, setMessage] = useState<string | null>(null)
  const [testingHost, setTestingHost] = useState(false)
  const [testingPush, setTestingPush] = useState(false)
  const [pushState, setPushState] = useState<BackgroundPushState | null>(null)
  const [tailscale, setTailscale] = useState<TailscaleRemoteAccessStatus | null>(null)
  const [loadingRemote, setLoadingRemote] = useState(true)
  const [changingRemote, setChangingRemote] = useState(false)
  const [copiedRemoteUrl, setCopiedRemoteUrl] = useState(false)

  useEffect(() => subscribeDeviceNotificationPreferences(setPreferences), [])

  useEffect(() => {
    let cancelled = false
    void Promise.all([
      getBackgroundPushState(),
      api.fetchTailscaleRemoteAccessStatus(),
    ]).then(([nextPush, nextTailscale]) => {
      if (cancelled) return
      setPushState(nextPush)
      setTailscale(nextTailscale)
    }).catch(error => {
      if (!cancelled) {
        setMessage(error instanceof Error ? error.message : 'Remote access status is unavailable.')
      }
    }).finally(() => {
      if (!cancelled) setLoadingRemote(false)
    })
    return () => { cancelled = true }
  }, [])

  useEffect(() => {
    const refreshAvailability = () => setAvailability(getBrowserNotificationAvailability())
    window.addEventListener('focus', refreshAvailability)
    document.addEventListener('visibilitychange', refreshAvailability)
    return () => {
      window.removeEventListener('focus', refreshAvailability)
      document.removeEventListener('visibilitychange', refreshAvailability)
    }
  }, [])

  const updateDevice = (partial: Partial<DeviceNotificationPreferences>) => {
    setMessage(null)
    const next = updateDeviceNotificationPreferences(partial)
    setPreferences(next)
    if (
      next.browserNotifications
      && (
        'onlyWhenHidden' in partial
        || 'notifyCompleted' in partial
        || 'notifyFailed' in partial
        || 'notifyQueue' in partial
      )
    ) {
      void syncBackgroundPush(next).then(setPushState).catch(error => {
        setMessage(error instanceof Error ? error.message : 'Could not update background notification preferences.')
      })
    }
  }

  const handleBrowserToggle = async (enabled: boolean) => {
    setMessage(null)
    if (!enabled) {
      updateDevice({ browserNotifications: false })
      try {
        setPushState(await disableBackgroundPush())
      } catch (error) {
        setMessage(error instanceof Error ? error.message : 'Could not remove the background subscription.')
      }
      return
    }
    const permission = await requestBrowserNotificationPermission()
    const nextAvailability = getBrowserNotificationAvailability()
    setAvailability(nextAvailability)
    if (permission === 'granted') {
      const next = updateDeviceNotificationPreferences({ browserNotifications: true })
      setPreferences(next)
      try {
        setPushState(await enableBackgroundPush(next))
        setMessage('System and closed-app background notifications are enabled on this device.')
      } catch (error) {
        // Keep foreground/browser notifications enabled even when an older
        // Maestro host has not installed the optional Web Push runtime yet.
        setMessage(
          `System notifications are enabled. ${
            error instanceof Error ? error.message : 'Background delivery could not be enrolled.'
          }`,
        )
      }
    } else {
      updateDevice({ browserNotifications: false })
      setMessage(nextAvailability.reason || 'Notification permission was not granted.')
    }
  }

  const handleBrowserTest = async () => {
    setMessage(null)
    const shown = await testBrowserNotification()
    setAvailability(getBrowserNotificationAvailability())
    setPreferences(getDeviceNotificationPreferences())
    setMessage(shown
      ? 'Test alert sent.'
      : 'The browser could not display a system notification. In-app alerts still work.')
  }

  const handleDeviceSoundToggle = async (enabled: boolean) => {
    updateDevice({ deviceSound: enabled })
    if (enabled) {
      const played = await playDeviceNotificationChime('completion')
      if (!played) setMessage('This browser could not start audio. Try Test chime after interacting with the page.')
    }
  }

  const handleBackgroundTest = async () => {
    setTestingPush(true)
    setMessage(null)
    try {
      const delivered = await testBackgroundPush()
      setMessage(delivered
        ? 'Background push sent. It may take a moment to appear.'
        : 'This device is not enrolled for background notifications.')
    } catch (error) {
      setMessage(error instanceof Error ? error.message : 'Background push test failed.')
    } finally {
      setTestingPush(false)
      setPushState(await getBackgroundPushState())
    }
  }

  const refreshRemoteStatus = async () => {
    setLoadingRemote(true)
    setMessage(null)
    try {
      setTailscale(await api.fetchTailscaleRemoteAccessStatus())
    } catch (error) {
      setMessage(error instanceof Error ? error.message : 'Could not read Tailscale status.')
    } finally {
      setLoadingRemote(false)
    }
  }

  const handleRemoteToggle = async (enabled: boolean) => {
    setChangingRemote(true)
    setMessage(null)
    try {
      const status = enabled
        ? await api.enableTailscaleRemoteAccess()
        : await api.disableTailscaleRemoteAccess()
      setTailscale(status)
      setMessage(enabled
        ? 'Private HTTPS access is ready. Install Tailscale on your phone and use the address below.'
        : 'Private Tailscale access was disabled.')
    } catch (error) {
      setMessage(error instanceof Error ? error.message : 'Could not update private access.')
      await refreshRemoteStatus()
    } finally {
      setChangingRemote(false)
    }
  }

  const copyRemoteUrl = async () => {
    if (!tailscale?.https_url) return
    await navigator.clipboard.writeText(tailscale.https_url)
    setCopiedRemoteUrl(true)
    window.setTimeout(() => setCopiedRemoteUrl(false), 1500)
  }

  const hostEnabled = systemConfig?.host_notification_sound_enabled ?? false
  const hostVolume = systemConfig?.host_notification_sound_volume ?? 50

  const handleHostTest = async () => {
    setTestingHost(true)
    setMessage(null)
    try {
      await api.testHostNotificationSound(hostVolume)
      setMessage('Test sound sent to the Maestro host computer.')
    } catch (error) {
      setMessage(error instanceof Error ? error.message : 'Unable to play the host sound.')
    } finally {
      setTestingHost(false)
    }
  }

  return (
    <div className="space-y-5">
      <div>
        <h3 className="text-[11px] font-medium uppercase tracking-wider text-text-secondary">Notifications</h3>
        <p className="mt-1 text-[10px] leading-relaxed text-text-muted">
          Maestro always shows a small in-app alert. The options below add system notifications or sound.
        </p>
      </div>

      <section className="space-y-3 rounded-lg border border-border bg-bg-tertiary p-3">
        <div className="flex items-center gap-2">
          <Bell size={15} className="text-accent-blue" />
          <div>
            <div className="text-xs font-medium text-text-primary">This browser or device</div>
            <div className="text-[10px] text-text-muted">Saved separately in each browser.</div>
          </div>
        </div>

        <Toggle
          checked={preferences.browserNotifications}
          onChange={handleBrowserToggle}
          label="System notifications"
          disabled={!availability.supported || !availability.secure || availability.permission === 'denied'}
          description={availability.reason || permissionLabel(availability)}
        />

        {preferences.browserNotifications && (
          <div className={`rounded-md border px-2.5 py-2 text-[9px] leading-relaxed ${
            pushState?.subscribed
              ? 'border-emerald-500/30 bg-emerald-500/5 text-emerald-200'
              : 'border-amber-500/30 bg-amber-500/5 text-amber-100'
          }`}>
            <div className="flex items-center gap-1.5 font-medium">
              {pushState?.subscribed ? <Check size={11} /> : <Bell size={11} />}
              {pushState?.subscribed
                ? 'Background delivery active on this device'
                : 'Foreground notifications active'}
            </div>
            <div className="mt-1">
              {pushState?.subscribed
                ? 'Maestro can alert this device after the page or iPhone Home Screen app is closed.'
                : (pushState?.reason || 'Background enrollment is still being prepared.')}
            </div>
          </div>
        )}

        <Toggle
          checked={preferences.onlyWhenHidden}
          onChange={checked => updateDevice({ onlyWhenHidden: checked })}
          label="Only notify when Maestro is in the background"
          description="In-app alerts still appear while Maestro is visible."
        />

        <div className="grid grid-cols-3 gap-1.5 border-y border-border/50 py-2.5">
          {([
            ['notifyCompleted', 'Complete'],
            ['notifyFailed', 'Failed'],
            ['notifyQueue', 'Queue'],
          ] as const).map(([key, label]) => (
            <label key={key} className="flex cursor-pointer items-center gap-1.5 text-[10px] text-text-secondary">
              <input
                type="checkbox"
                checked={preferences[key]}
                onChange={event => updateDevice({ [key]: event.target.checked })}
                className="accent-blue-500"
              />
              {label}
            </label>
          ))}
        </div>

        <Toggle
          checked={preferences.deviceSound}
          onChange={handleDeviceSoundToggle}
          label="Chime on this device"
          description="Works while this Maestro page is open, including mobile browsers."
        />

        <div className="space-y-1">
          <div className="flex items-center justify-between text-[10px] text-text-muted">
            <span className="flex items-center gap-1"><Volume2 size={11} /> Device volume</span>
            <span>{preferences.deviceSoundVolume}%</span>
          </div>
          <input
            type="range"
            min={0}
            max={100}
            step={5}
            value={preferences.deviceSoundVolume}
            onChange={event => updateDevice({ deviceSoundVolume: Number(event.target.value) })}
            className="w-full"
          />
        </div>

        <div className="flex gap-2">
          <button
            type="button"
            onClick={handleBrowserTest}
            disabled={!availability.supported || !availability.secure || availability.permission === 'denied'}
            className="flex-1 rounded-md border border-border px-2 py-1.5 text-[10px] text-text-secondary transition-colors hover:bg-bg-hover hover:text-text-primary disabled:cursor-not-allowed disabled:opacity-40"
          >
            Test notification
          </button>
          <button
            type="button"
            onClick={() => void playDeviceNotificationChime('completion')}
            className="flex-1 rounded-md border border-border px-2 py-1.5 text-[10px] text-text-secondary transition-colors hover:bg-bg-hover hover:text-text-primary"
          >
            Test chime
          </button>
        </div>

        <button
          type="button"
          onClick={handleBackgroundTest}
          disabled={testingPush || !pushState?.subscribed}
          className="w-full rounded-md border border-border px-2 py-1.5 text-[10px] text-text-secondary transition-colors hover:bg-bg-hover hover:text-text-primary disabled:cursor-not-allowed disabled:opacity-40"
        >
          {testingPush ? 'Sending background push…' : 'Test closed-app notification'}
        </button>

        <p className="text-[9px] leading-relaxed text-text-muted">
          iPhone/iPad: use Maestro through HTTPS, remove any older Maestro Home Screen shortcut, then add it to the Home Screen again and open that installed app. Safari and Chrome tabs cannot request notification permission on iOS.
        </p>

        {availability.ios && (
          <div className={`rounded-md border px-2.5 py-2 text-[9px] leading-relaxed ${
            availability.secure && availability.standalone
              ? 'border-emerald-500/30 bg-emerald-500/5 text-emerald-200'
              : 'border-amber-500/30 bg-amber-500/5 text-amber-100'
          }`}>
            <div className="font-medium">iPhone notification check</div>
            <div className="mt-1">Secure HTTPS address: {availability.secure ? 'Yes' : 'No'}</div>
            <div>Opened from installed Home Screen app: {availability.standalone ? 'Yes' : 'No'}</div>
            {!availability.secure && (
              <div className="mt-1 break-all">
                Current address: {window.location.origin}. Apple blocks system notifications from local HTTP addresses.
              </div>
            )}
            {availability.secure && availability.standalone && (
              <div className="mt-1">This device is ready to request notification permission.</div>
            )}
          </div>
        )}

        <p className="text-[9px] leading-relaxed text-text-muted">
          Closed-app delivery uses the browser vendor&apos;s standard encrypted Web Push service. Maestro&apos;s signing key and your device subscription remain on your Maestro computer; there is no Maestro cloud account or relay.
        </p>
      </section>

      <section className="space-y-3 rounded-lg border border-border bg-bg-tertiary p-3">
        <div className="flex items-center justify-between gap-2">
          <div className="flex items-center gap-2">
            <ShieldCheck size={15} className="text-accent-blue" />
            <div>
              <div className="text-xs font-medium text-text-primary">Private phone access</div>
              <div className="text-[10px] text-text-muted">Optional Tailscale HTTPS · never public</div>
            </div>
          </div>
          <button
            type="button"
            onClick={refreshRemoteStatus}
            disabled={loadingRemote}
            aria-label="Refresh Tailscale status"
            className="rounded p-1 text-text-muted transition-colors hover:bg-bg-hover hover:text-text-primary disabled:opacity-40"
          >
            <RefreshCw size={12} className={loadingRemote ? 'animate-spin' : ''} />
          </button>
        </div>

        {loadingRemote && !tailscale ? (
          <div className="text-[10px] text-text-muted">Checking this computer…</div>
        ) : tailscale && !tailscale.installed ? (
          <div className="space-y-2">
            <p className="text-[10px] leading-relaxed text-text-muted">
              Install the free Tailscale Personal app on this computer and your phone, then sign both into your own account. Maestro never joins or manages your tailnet.
            </p>
            <a
              href={tailscale.install_url}
              target="_blank"
              rel="noreferrer"
              className="flex w-full items-center justify-center gap-1.5 rounded-md border border-border px-2 py-1.5 text-[10px] text-text-secondary transition-colors hover:bg-bg-hover hover:text-text-primary"
            >
              Install Tailscale on this computer <ExternalLink size={10} />
            </a>
          </div>
        ) : tailscale && !tailscale.connected ? (
          <div className="space-y-2 rounded-md border border-amber-500/30 bg-amber-500/5 px-2.5 py-2 text-[10px] leading-relaxed text-amber-100">
            <div className="font-medium">Tailscale is installed but not connected</div>
            <div>Open the Tailscale app on this computer, sign in, then refresh this status.</div>
          </div>
        ) : tailscale ? (
          <>
            <Toggle
              checked={tailscale.configured && tailscale.enabled}
              onChange={checked => void handleRemoteToggle(checked)}
              label="Private HTTPS access"
              description="Tailscale Serve securely proxies only devices in your personal tailnet to this local Maestro instance."
              disabled={changingRemote}
            />

            {tailscale.https_url && tailscale.configured && (
              <div className="space-y-3 rounded-md border border-emerald-500/30 bg-emerald-500/5 p-3">
                <div className="flex flex-col items-center gap-2">
                  <div className="rounded-lg bg-white p-2">
                    <QRCodeSVG
                      value={tailscale.https_url}
                      size={132}
                      level="M"
                      marginSize={1}
                    />
                  </div>
                  <div className="text-center text-[9px] leading-relaxed text-emerald-100">
                    Scan after installing Tailscale on your phone and signing into the same account.
                  </div>
                </div>
                <div className="break-all rounded bg-bg-primary/70 px-2 py-1.5 text-[9px] text-text-secondary">
                  {tailscale.https_url}
                </div>
                <div className="grid grid-cols-2 gap-2">
                  <button
                    type="button"
                    onClick={copyRemoteUrl}
                    className="flex items-center justify-center gap-1 rounded-md border border-border px-2 py-1.5 text-[10px] text-text-secondary hover:bg-bg-hover hover:text-text-primary"
                  >
                    {copiedRemoteUrl ? <Check size={10} /> : <Copy size={10} />}
                    {copiedRemoteUrl ? 'Copied' : 'Copy URL'}
                  </button>
                  <a
                    href={tailscale.https_url}
                    target="_blank"
                    rel="noreferrer"
                    className="flex items-center justify-center gap-1 rounded-md border border-border px-2 py-1.5 text-[10px] text-text-secondary hover:bg-bg-hover hover:text-text-primary"
                  >
                    Open <ExternalLink size={10} />
                  </a>
                </div>
              </div>
            )}

            <ol className="space-y-1 text-[9px] leading-relaxed text-text-muted">
              <li className="flex gap-1.5"><span>1.</span><span>Install Tailscale on the phone and use the same personal account.</span></li>
              <li className="flex gap-1.5"><span>2.</span><span>Open the secure URL in Safari, then Share → Add to Home Screen.</span></li>
              <li className="flex gap-1.5"><span>3.</span><span>Open the installed Maestro app and enable System notifications above.</span></li>
            </ol>
          </>
        ) : null}

        <div className="flex items-start gap-1.5 rounded-md border border-border/60 bg-bg-primary/40 px-2.5 py-2 text-[9px] leading-relaxed text-text-muted">
          <Smartphone size={11} className="mt-0.5 shrink-0" />
          <span>Tailscale&apos;s Personal plan can be used independently by each user. This feature does not use Funnel and does not expose Maestro to the public internet.</span>
        </div>
      </section>

      <section className="space-y-3 rounded-lg border border-border bg-bg-tertiary p-3">
        <div className="flex items-center gap-2">
          <MonitorSpeaker size={15} className="text-accent-blue" />
          <div>
            <div className="text-xs font-medium text-text-primary">Maestro host computer</div>
            <div className="text-[10px] text-text-muted">Useful when you leave the generation machine running.</div>
          </div>
        </div>

        <Toggle
          checked={hostEnabled}
          onChange={checked => void updateSystemConfig({ host_notification_sound_enabled: checked })}
          label="Completion sound on host"
          description="Rings once per Studio generation or complete Director project—not once per internal clip."
          disabled={!systemConfig}
        />

        <div className="space-y-1">
          <div className="flex items-center justify-between text-[10px] text-text-muted">
            <span className="flex items-center gap-1"><BellRing size={11} /> Host volume</span>
            <span>{hostVolume}%</span>
          </div>
          <input
            type="range"
            min={0}
            max={100}
            step={5}
            value={hostVolume}
            disabled={!systemConfig}
            onChange={event => void updateSystemConfig({
              host_notification_sound_volume: Number(event.target.value),
            })}
            className="w-full disabled:opacity-40"
          />
        </div>

        <button
          type="button"
          onClick={handleHostTest}
          disabled={testingHost || !systemConfig}
          className="w-full rounded-md border border-border px-2 py-1.5 text-[10px] text-text-secondary transition-colors hover:bg-bg-hover hover:text-text-primary disabled:cursor-not-allowed disabled:opacity-40"
        >
          {testingHost ? 'Playing…' : 'Test host sound'}
        </button>
      </section>

      {message && (
        <div className="rounded-md border border-border bg-bg-primary px-2.5 py-2 text-[10px] leading-relaxed text-text-secondary">
          {message}
        </div>
      )}
    </div>
  )
}
