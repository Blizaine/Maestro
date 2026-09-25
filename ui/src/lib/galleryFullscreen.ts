type WebkitElement = HTMLElement & { webkitRequestFullscreen?: () => void | Promise<void> }
type WebkitDocument = Document & {
  webkitFullscreenElement?: Element | null
  webkitExitFullscreen?: () => void | Promise<void>
}

const HOME_SCREEN_HELP_KEY = 'maestro_gallery_home_screen_help_dismissed_v1'
const HOME_SCREEN_HELP = 'To browse without the Safari toolbar, use Share → Add to Home Screen, then open Maestro from that icon.'
let homeScreenHelpDismissed = false

export function dismissGalleryFullscreenHelp(message: string): void {
  if (message !== HOME_SCREEN_HELP) return
  homeScreenHelpDismissed = true
  try { localStorage.setItem(HOME_SCREEN_HELP_KEY, '1') } catch { /* Still dismiss for this visit if storage is unavailable. */ }
}

export function galleryFullscreenElement(): Element | null {
  return document.fullscreenElement || (document as WebkitDocument).webkitFullscreenElement || null
}

export function isStandaloneGallery(): boolean {
  return window.matchMedia('(display-mode: standalone), (display-mode: fullscreen)').matches
    || (navigator as Navigator & { standalone?: boolean }).standalone === true
}

export function galleryFullscreenHelp(): string | null {
  if (/iPhone|iPod/.test(navigator.userAgent)) {
    if (homeScreenHelpDismissed) return null
    try { if (localStorage.getItem(HOME_SCREEN_HELP_KEY) === '1') return null } catch { /* Storage may be unavailable in a private browser. */ }
    return HOME_SCREEN_HELP
  }
  return 'Your browser could not enter fullscreen. The gallery is still available here; open Maestro in its own browser tab to try fullscreen again.'
}

/** Call directly from the opening click, while browser user activation is live. */
export async function enterGalleryFullscreen(host: HTMLElement): Promise<string | null> {
  const existing = galleryFullscreenElement()
  if (existing?.contains(host)) return null
  try {
    if (host.requestFullscreen && document.fullscreenEnabled !== false) {
      await host.requestFullscreen({ navigationUI: 'hide' })
    } else if ((host as WebkitElement).webkitRequestFullscreen) {
      await (host as WebkitElement).webkitRequestFullscreen!()
    } else {
      return isStandaloneGallery() ? null : galleryFullscreenHelp()
    }
    // The user may close the viewer before the asynchronous request completes.
    if (!host.isConnected) await exitGalleryFullscreen(host)
    return null
  } catch {
    return isStandaloneGallery() ? null : galleryFullscreenHelp()
  }
}

export async function exitGalleryFullscreen(host: HTMLElement): Promise<void> {
  // Do not exit a fullscreen page/player owned by something else.
  if (galleryFullscreenElement() !== host) return
  try {
    if (document.exitFullscreen) await document.exitFullscreen()
    else await (document as WebkitDocument).webkitExitFullscreen?.()
  } catch { /* Removing the host also lets the browser exit fullscreen. */ }
}

export function createGalleryViewerSurface(requestFullscreen: boolean) {
  const returnFocus = document.activeElement instanceof HTMLElement ? document.activeElement : null
  const host = document.createElement('div')
  host.dataset.galleryViewerPortal = 'true'
  host.style.background = 'black'
  document.body.appendChild(host)
  // Do not defer this into a React effect: Safari/Firefox may have already
  // consumed the click's transient activation by the time an effect runs.
  const fullscreenRequest = requestFullscreen ? enterGalleryFullscreen(host) : Promise.resolve(null)
  return {
    host, returnFocus, fullscreenRequest,
    release: () => {
      void exitGalleryFullscreen(host)
      host.remove()
    },
  }
}
