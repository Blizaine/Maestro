import * as api from '../api/client'

export type MaestroNotificationCategory =
  | 'completion'
  | 'failure'
  | 'queue'
  | 'cancelled'
  | 'test'

export interface DeviceNotificationPreferences {
  browserNotifications: boolean
  deviceSound: boolean
  deviceSoundVolume: number
  onlyWhenHidden: boolean
  notifyCompleted: boolean
  notifyFailed: boolean
  notifyQueue: boolean
}

export interface MaestroAlert {
  id: string
  key: string
  category: MaestroNotificationCategory
  title: string
  body: string
  createdAt: number
}

export interface MaestroNotificationEvent {
  key: string
  category: MaestroNotificationCategory
  title: string
  body: string
  /** Keep the in-app toast but suppress the OS/browser notification. */
  system?: boolean
  /** Keep the in-app toast but suppress Maestro's device chime. */
  sound?: boolean
  /** Test controls bypass event-category and hidden-tab preferences. */
  force?: boolean
}

export interface BrowserNotificationAvailability {
  supported: boolean
  secure: boolean
  permission: NotificationPermission | 'unsupported'
  ios: boolean
  standalone: boolean
  serviceWorker: boolean
  reason: string | null
}

export interface BackgroundPushState {
  supported: boolean
  subscribed: boolean
  endpoint: string | null
  subscriptionCount: number
  reason: string | null
}

const PREFS_KEY = 'maestro-notification-preferences-v1'
const SEEN_KEY = 'maestro-notification-seen-v1'
const MAX_SEEN_KEYS = 240

const DEFAULT_PREFERENCES: DeviceNotificationPreferences = {
  browserNotifications: false,
  deviceSound: false,
  deviceSoundVolume: 55,
  onlyWhenHidden: true,
  notifyCompleted: true,
  notifyFailed: true,
  notifyQueue: true,
}

type PreferenceListener = (preferences: DeviceNotificationPreferences) => void
type AlertListener = (alert: MaestroAlert) => void

const preferenceListeners = new Set<PreferenceListener>()
const alertListeners = new Set<AlertListener>()

function clampVolume(value: unknown): number {
  const parsed = Number(value)
  if (!Number.isFinite(parsed)) return DEFAULT_PREFERENCES.deviceSoundVolume
  return Math.max(0, Math.min(100, Math.round(parsed)))
}

function loadPreferences(): DeviceNotificationPreferences {
  if (typeof window === 'undefined') return { ...DEFAULT_PREFERENCES }
  try {
    const parsed = JSON.parse(localStorage.getItem(PREFS_KEY) || '{}') as Partial<DeviceNotificationPreferences>
    return {
      browserNotifications: parsed.browserNotifications === true,
      deviceSound: parsed.deviceSound === true,
      deviceSoundVolume: clampVolume(parsed.deviceSoundVolume),
      onlyWhenHidden: parsed.onlyWhenHidden !== false,
      notifyCompleted: parsed.notifyCompleted !== false,
      notifyFailed: parsed.notifyFailed !== false,
      notifyQueue: parsed.notifyQueue !== false,
    }
  } catch {
    return { ...DEFAULT_PREFERENCES }
  }
}

let preferences = loadPreferences()

export function getDeviceNotificationPreferences(): DeviceNotificationPreferences {
  return preferences
}

export function updateDeviceNotificationPreferences(
  partial: Partial<DeviceNotificationPreferences>,
): DeviceNotificationPreferences {
  preferences = {
    ...preferences,
    ...partial,
    deviceSoundVolume: partial.deviceSoundVolume == null
      ? preferences.deviceSoundVolume
      : clampVolume(partial.deviceSoundVolume),
  }
  try {
    localStorage.setItem(PREFS_KEY, JSON.stringify(preferences))
  } catch {
    // Private browsing and locked-down browsers can deny localStorage.
    // The preference still applies for the current Maestro session.
  }
  preferenceListeners.forEach(listener => listener(preferences))
  return preferences
}

export function subscribeDeviceNotificationPreferences(
  listener: PreferenceListener,
): () => void {
  preferenceListeners.add(listener)
  return () => preferenceListeners.delete(listener)
}

export function subscribeMaestroAlerts(listener: AlertListener): () => void {
  alertListeners.add(listener)
  return () => alertListeners.delete(listener)
}

function isIosDevice(): boolean {
  if (typeof navigator === 'undefined') return false
  return /iPad|iPhone|iPod/i.test(navigator.userAgent)
    || (navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1)
}

function isStandaloneWebApp(): boolean {
  if (typeof window === 'undefined') return false
  const iosNavigator = navigator as Navigator & { standalone?: boolean }
  return window.matchMedia?.('(display-mode: standalone)').matches === true
    || iosNavigator.standalone === true
}

export function getBrowserNotificationAvailability(): BrowserNotificationAvailability {
  const ios = isIosDevice()
  const standalone = isStandaloneWebApp()
  const serviceWorker = typeof navigator !== 'undefined'
    && 'serviceWorker' in navigator
    && typeof ServiceWorkerRegistration !== 'undefined'
    && 'showNotification' in ServiceWorkerRegistration.prototype
  const notificationApi = typeof window !== 'undefined' && 'Notification' in window
  const permission = notificationApi ? Notification.permission : 'unsupported'

  // WebKit exposes iPhone/iPad notification permission only to installed
  // Home Screen web apps. Chrome and other iOS browsers use WebKit too.
  if (ios && !standalone) {
    return {
      supported: false,
      secure: typeof window !== 'undefined' && window.isSecureContext,
      permission,
      ios,
      standalone,
      serviceWorker,
      reason: 'On iPhone and iPad, open Maestro from an installed Home Screen app—not a Safari or Chrome tab.',
    }
  }

  if (typeof window === 'undefined' || !window.isSecureContext) {
    return {
      supported: notificationApi && serviceWorker,
      secure: false,
      permission,
      ios,
      standalone,
      serviceWorker,
      reason: 'System notifications require HTTPS. A phone opening Maestro through a local http:// address is not a secure browser context.',
    }
  }

  if (!notificationApi || !serviceWorker) {
    return {
      supported: false,
      secure: true,
      permission,
      ios,
      standalone,
      serviceWorker,
      reason: !serviceWorker
        ? 'This browser does not support the service worker required for mobile notifications.'
        : 'This browser does not expose notification permission to Maestro.',
    }
  }

  return {
    supported: true,
    secure: true,
    permission: Notification.permission,
    ios,
    standalone,
    serviceWorker,
    reason: Notification.permission === 'denied'
      ? 'Notifications are blocked in this browser. Allow them in site settings.'
      : null,
  }
}

export async function requestBrowserNotificationPermission(): Promise<NotificationPermission | 'unsupported'> {
  const availability = getBrowserNotificationAvailability()
  if (!availability.supported || !availability.secure) return availability.permission
  if (availability.permission !== 'default') return availability.permission
  try {
    return await Notification.requestPermission()
  } catch {
    return 'denied'
  }
}

let audioContext: AudioContext | null = null

function getAudioContext(): AudioContext | null {
  if (audioContext) return audioContext
  if (typeof window === 'undefined') return null
  const AudioContextConstructor = window.AudioContext
    || (window as typeof window & { webkitAudioContext?: typeof AudioContext }).webkitAudioContext
  if (!AudioContextConstructor) return null
  try {
    audioContext = new AudioContextConstructor()
    return audioContext
  } catch {
    return null
  }
}

/** Prime Web Audio during a real user gesture. This matters after a reload on
 * mobile/Safari where a persisted chime preference cannot unlock audio from a
 * later background completion event by itself. */
export async function prepareDeviceNotificationAudio(): Promise<boolean> {
  const context = getAudioContext()
  if (!context) return false
  try {
    if (context.state === 'suspended') await context.resume()
    return context.state === 'running'
  } catch {
    return false
  }
}

export async function playDeviceNotificationChime(
  category: MaestroNotificationCategory = 'completion',
  volume = preferences.deviceSoundVolume,
): Promise<boolean> {
  const context = getAudioContext()
  if (!context) return false
  try {
    if (context.state === 'suspended') await context.resume()
    const now = context.currentTime + 0.01
    const normalizedVolume = Math.max(0, Math.min(1, volume / 100))
    const notes = category === 'failure'
      ? [392.0, 311.13]
      : category === 'queue'
        ? [523.25, 659.25, 783.99]
        : [523.25, 659.25]

    notes.forEach((frequency, index) => {
      const startsAt = now + index * 0.11
      const endsAt = startsAt + (category === 'failure' ? 0.24 : 0.2)
      const oscillator = context.createOscillator()
      const gain = context.createGain()
      oscillator.type = 'sine'
      oscillator.frequency.setValueAtTime(frequency, startsAt)
      gain.gain.setValueAtTime(0.0001, startsAt)
      gain.gain.exponentialRampToValueAtTime(
        Math.max(0.0001, normalizedVolume * 0.085),
        startsAt + 0.025,
      )
      gain.gain.exponentialRampToValueAtTime(0.0001, endsAt)
      oscillator.connect(gain)
      gain.connect(context.destination)
      oscillator.start(startsAt)
      oscillator.stop(endsAt + 0.02)
    })
    return true
  } catch {
    return false
  }
}

function readSeenKeys(): string[] {
  try {
    const parsed = JSON.parse(sessionStorage.getItem(SEEN_KEY) || '[]')
    return Array.isArray(parsed) ? parsed.filter(item => typeof item === 'string') : []
  } catch {
    return []
  }
}

function markSeen(key: string): boolean {
  const seen = readSeenKeys()
  if (seen.includes(key)) return false
  seen.push(key)
  try {
    sessionStorage.setItem(SEEN_KEY, JSON.stringify(seen.slice(-MAX_SEEN_KEYS)))
  } catch {
    // Dedupe remains best-effort when sessionStorage is unavailable.
  }
  return true
}

let originalTitle = 'Maestro'
let unseenAlertCount = 0
let visibilityListenerInstalled = false

function installVisibilityListener() {
  if (visibilityListenerInstalled || typeof document === 'undefined') return
  visibilityListenerInstalled = true
  originalTitle = document.title || 'Maestro'
  document.addEventListener('visibilitychange', () => {
    if (!document.hidden) {
      unseenAlertCount = 0
      document.title = originalTitle
    }
  })
}

function updateHiddenTabTitle(category: MaestroNotificationCategory) {
  if (typeof document === 'undefined' || !document.hidden || category === 'test') return
  installVisibilityListener()
  unseenAlertCount += 1
  const marker = category === 'failure' ? '!' : '✓'
  document.title = `${marker} ${unseenAlertCount} · ${originalTitle}`
}

function eventEnabled(category: MaestroNotificationCategory): boolean {
  if (category === 'completion') return preferences.notifyCompleted
  if (category === 'failure') return preferences.notifyFailed
  if (category === 'queue') return preferences.notifyQueue
  return category === 'test'
}

function emitToast(event: MaestroNotificationEvent) {
  const alert: MaestroAlert = {
    id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
    key: event.key,
    category: event.category,
    title: event.title,
    body: event.body,
    createdAt: Date.now(),
  }
  alertListeners.forEach(listener => listener(alert))
}

async function readyServiceWorker(
  timeoutMs = 3000,
): Promise<ServiceWorkerRegistration | null> {
  if (!('serviceWorker' in navigator)) return null
  try {
    // Registration also happens during app startup. Repeating it here makes
    // the notification test resilient if that first registration was
    // interrupted while the page was loading or updating.
    await navigator.serviceWorker.register('/maestro-sw.js', { scope: '/' })
    return await Promise.race([
      navigator.serviceWorker.ready,
      new Promise<null>(resolve => window.setTimeout(() => resolve(null), timeoutMs)),
    ])
  } catch {
    return null
  }
}

function decodeApplicationServerKey(encoded: string): ArrayBuffer {
  const padding = '='.repeat((4 - (encoded.length % 4)) % 4)
  const base64 = (encoded + padding).replace(/-/g, '+').replace(/_/g, '/')
  const raw = window.atob(base64)
  const bytes = new Uint8Array(raw.length)
  for (let index = 0; index < raw.length; index += 1) {
    bytes[index] = raw.charCodeAt(index)
  }
  return bytes.buffer
}

function pushPreferences(value: DeviceNotificationPreferences): Record<string, boolean> {
  return {
    notifyCompleted: value.notifyCompleted,
    notifyFailed: value.notifyFailed,
    notifyQueue: value.notifyQueue,
    onlyWhenHidden: value.onlyWhenHidden,
  }
}

function deviceLabel(): string {
  if (isIosDevice()) return 'iPhone or iPad Home Screen app'
  const extendedNavigator = navigator as Navigator & {
    userAgentData?: { platform?: string }
  }
  const platform = extendedNavigator.userAgentData?.platform || navigator.platform || 'Browser'
  return `${platform} browser`
}

async function currentPushSubscription(): Promise<PushSubscription | null> {
  const registration = await readyServiceWorker()
  if (!registration || !('pushManager' in registration)) return null
  return registration.pushManager.getSubscription()
}

export async function getBackgroundPushState(): Promise<BackgroundPushState> {
  try {
    const [host, subscription] = await Promise.all([
      api.fetchWebPushStatus(),
      currentPushSubscription(),
    ])
    return {
      supported: host.supported,
      subscribed: Boolean(subscription),
      endpoint: subscription?.endpoint || null,
      subscriptionCount: host.subscription_count,
      reason: host.reason,
    }
  } catch (error) {
    return {
      supported: false,
      subscribed: false,
      endpoint: null,
      subscriptionCount: 0,
      reason: error instanceof Error ? error.message : 'Background notification status is unavailable.',
    }
  }
}

export async function enableBackgroundPush(
  value: DeviceNotificationPreferences = preferences,
): Promise<BackgroundPushState> {
  const availability = getBrowserNotificationAvailability()
  if (!availability.supported || !availability.secure || Notification.permission !== 'granted') {
    throw new Error(availability.reason || 'Notification permission is required first.')
  }
  const host = await api.fetchWebPushStatus()
  if (!host.supported) {
    throw new Error(host.reason || 'Background Web Push is unavailable on this Maestro host.')
  }
  const registration = await readyServiceWorker()
  if (!registration || !('pushManager' in registration)) {
    throw new Error('This browser does not provide the Push API.')
  }
  let subscription = await registration.pushManager.getSubscription()
  if (!subscription) {
    subscription = await registration.pushManager.subscribe({
      userVisibleOnly: true,
      applicationServerKey: decodeApplicationServerKey(host.public_key),
    })
  }
  const registered = await api.subscribeWebPush(
    subscription.toJSON(),
    pushPreferences(value),
    window.location.origin,
    deviceLabel(),
  )
  return {
    supported: true,
    subscribed: true,
    endpoint: subscription.endpoint,
    subscriptionCount: registered.subscription_count,
    reason: null,
  }
}

export async function syncBackgroundPush(
  value: DeviceNotificationPreferences = preferences,
): Promise<BackgroundPushState> {
  const subscription = await currentPushSubscription()
  if (!value.browserNotifications || Notification.permission !== 'granted') {
    return getBackgroundPushState()
  }
  if (!subscription) return enableBackgroundPush(value)
  const host = await api.fetchWebPushStatus()
  if (!host.supported) {
    return {
      supported: false,
      subscribed: true,
      endpoint: subscription.endpoint,
      subscriptionCount: host.subscription_count,
      reason: host.reason,
    }
  }
  await api.subscribeWebPush(
    subscription.toJSON(),
    pushPreferences(value),
    window.location.origin,
    deviceLabel(),
  )
  return {
    supported: true,
    subscribed: true,
    endpoint: subscription.endpoint,
    subscriptionCount: host.subscription_count,
    reason: null,
  }
}

export async function disableBackgroundPush(): Promise<BackgroundPushState> {
  const subscription = await currentPushSubscription()
  if (subscription) {
    try {
      await api.unsubscribeWebPush(subscription.endpoint)
    } finally {
      await subscription.unsubscribe()
    }
  }
  return getBackgroundPushState()
}

export async function testBackgroundPush(): Promise<boolean> {
  const subscription = await currentPushSubscription()
  if (!subscription) return false
  const result = await api.testWebPush(subscription.endpoint)
  return result.delivered > 0
}

async function showBrowserNotification(event: MaestroNotificationEvent): Promise<boolean> {
  const availability = getBrowserNotificationAvailability()
  if (
    !availability.supported
    || !availability.secure
    || availability.permission !== 'granted'
  ) return false

  // Persistent service-worker notifications are the standards-based mobile
  // path and also work on desktop. Unlike `new Notification()`, this is
  // supported by installed iOS Home Screen web apps.
  const registration = await readyServiceWorker()
  if (registration) {
    try {
      await registration.showNotification(event.title, {
        body: event.body,
        icon: '/maestro-icon.png',
        badge: '/maestro-icon.png',
        tag: event.key,
        silent: true,
        data: { url: window.location.href },
      })
      return true
    } catch {
      // Desktop browsers can still use the non-persistent fallback below.
    }
  }

  if (availability.ios) return false
  try {
    const notification = new Notification(event.title, {
      body: event.body,
      icon: '/maestro-icon.png',
      tag: event.key,
      silent: true,
    })
    notification.onclick = () => {
      window.focus()
      notification.close()
    }
    return true
  } catch {
    return false
  }
}

/**
 * Publish one user-facing terminal event. The key makes polling reconnects,
 * React Strict Mode, and Director/queue views idempotent.
 */
export function announceMaestroEvent(event: MaestroNotificationEvent): boolean {
  if (!event.force && !markSeen(event.key)) return false
  if (event.force) markSeen(event.key)

  installVisibilityListener()
  emitToast(event)
  updateHiddenTabTitle(event.category)

  const enabled = event.force || eventEnabled(event.category)
  if (!enabled || event.category === 'cancelled') return true

  const allowSystem = event.system !== false
    && (event.force || preferences.browserNotifications)
    && (event.force || !preferences.onlyWhenHidden || document.hidden)
  if (allowSystem) void showBrowserNotification(event)

  const allowSound = event.sound !== false
    && (event.force || preferences.deviceSound)
  if (allowSound) {
    void playDeviceNotificationChime(event.category)
  }
  return true
}

export async function testBrowserNotification(): Promise<boolean> {
  const permission = await requestBrowserNotificationPermission()
  if (permission !== 'granted') return false
  updateDeviceNotificationPreferences({ browserNotifications: true })
  const event: MaestroNotificationEvent = {
    key: `test-browser-${Date.now()}`,
    category: 'test',
    title: 'Maestro notifications are ready',
    body: 'Completion and failure alerts will appear on this device.',
    system: false,
    sound: false,
    force: true,
  }
  announceMaestroEvent(event)
  return await showBrowserNotification(event)
}
