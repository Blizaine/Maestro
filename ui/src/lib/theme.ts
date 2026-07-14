/**
 * Theme management for Maestro.
 *
 * Themes are CSS-variable overrides applied via `[data-theme="..."]` on
 * the <html> element. The actual variable values live in src/index.css.
 *
 * Two-dimensional model (GitHub-style):
 *   - mode: 'dark' | 'light' | 'auto' — auto follows the OS scheme
 *     (prefers-color-scheme) and live-switches when the OS changes.
 *   - one theme choice per variant: a dark theme and a light theme.
 * The effective theme = the chosen theme of whichever variant the mode
 * resolves to.
 *
 * Persistence: localStorage under "maestro-theme-mode" /
 * "maestro-theme-dark" / "maestro-theme-light". The legacy single-theme
 * key ("maestro-theme") seeds the matching slot on first load after
 * upgrade, so nobody's chosen look changes. An inline script in
 * index.html applies the resolved theme to <html> before React mounts
 * so there's no flash of the default theme on page load.
 *
 * Adding a new theme: add a `[data-theme="<id>"]` block in index.css
 * with the variable overrides, add the id to THEMES below (with its
 * variant + counterpart), and it shows up in the Settings/System
 * dropdowns automatically.
 */

export type ThemeId = 'default' | 'golden-hour' | 'onyx' | 'ivory' | 'daylight' | 'pearl'
export type ThemeMode = 'dark' | 'light' | 'auto'

export interface ThemeDescriptor {
  id: ThemeId
  label: string
  /** Which mode slot this theme belongs to. */
  variant: 'dark' | 'light'
  description: string
  /** Three-color preview swatch shown in the settings dropdown. */
  swatch: { bg: string; surface: string; accent: string }
}

/* THEMES ordering doubles as the dropdown order within each variant.
 * The `id` of the cool palette stays 'default' (despite no longer
 * being the default) so existing localStorage values from users who
 * explicitly chose it don't break — only the LABEL changed to
 * 'Classic'. */
export const THEMES: ThemeDescriptor[] = [
  {
    id: 'golden-hour',
    label: 'Golden Hour',
    variant: 'dark',
    description:
      'Default. Warm cinematic palette — near-black surfaces, warm-tinted borders, amber highlights, sunset gradient on primary actions.',
    swatch: { bg: '#0a0a0a', surface: '#181818', accent: '#f97316' },
  },
  {
    id: 'default',
    label: 'Classic',
    variant: 'dark',
    description: 'The original cool charcoal palette with blue accents.',
    swatch: { bg: '#0a0a0f', surface: '#1a1a25', accent: '#3b82f6' },
  },
  {
    id: 'onyx',
    label: 'Onyx',
    variant: 'dark',
    description:
      'Minimalist monochrome — pure black backgrounds, neutral grey surfaces, white-toned accents. No color tint.',
    swatch: { bg: '#000000', surface: '#1a1a1a', accent: '#aaaaaa' },
  },
  {
    id: 'ivory',
    label: 'Ivory',
    variant: 'light',
    description:
      'Warm paper surfaces, coffee-toned text, burnt-orange accents. Golden Hour in daylight.',
    swatch: { bg: '#f2ede2', surface: '#f9f6ee', accent: '#c2410c' },
  },
  {
    id: 'daylight',
    label: 'Daylight',
    variant: 'light',
    description: 'Cool paper surfaces with blue accents. Classic in daylight.',
    swatch: { bg: '#f4f5f7', surface: '#fafbfc', accent: '#2563eb' },
  },
  {
    id: 'pearl',
    label: 'Pearl',
    variant: 'light',
    description:
      'Light monochrome — white and grey surfaces, charcoal accents. Onyx in daylight.',
    swatch: { bg: '#f2f2f2', surface: '#f9f9f9', accent: '#525252' },
  },
]

export const DARK_THEMES = THEMES.filter(t => t.variant === 'dark')
export const LIGHT_THEMES = THEMES.filter(t => t.variant === 'light')

/** Dark <-> light siblings. Used to seed the "other" slot when
 * migrating a legacy single-theme preference so mode-switching lands
 * on the chosen theme's stylistic twin instead of an arbitrary one. */
export const COUNTERPART: Record<ThemeId, ThemeId> = {
  'golden-hour': 'ivory',
  ivory: 'golden-hour',
  default: 'daylight',
  daylight: 'default',
  onyx: 'pearl',
  pearl: 'onyx',
}

export interface ThemePrefs {
  mode: ThemeMode
  dark: ThemeId
  light: ThemeId
}

const MODE_KEY = 'maestro-theme-mode'
const DARK_KEY = 'maestro-theme-dark'
const LIGHT_KEY = 'maestro-theme-light'
const LEGACY_KEY = 'maestro-theme'

const DEFAULT_PREFS: ThemePrefs = { mode: 'dark', dark: 'golden-hour', light: 'ivory' }

function isVariant(id: string | null, variant: 'dark' | 'light'): id is ThemeId {
  return !!id && THEMES.some(t => t.id === id && t.variant === variant)
}

export function getStoredPrefs(): ThemePrefs {
  try {
    const legacy = localStorage.getItem(LEGACY_KEY)
    const legacyIsLight = isVariant(legacy, 'light')
    const legacyIsDark = isVariant(legacy, 'dark')

    const modeRaw = localStorage.getItem(MODE_KEY)
    const mode: ThemeMode =
      modeRaw === 'dark' || modeRaw === 'light' || modeRaw === 'auto'
        ? modeRaw
        // Migration: a stored light theme means the user chose light.
        : legacyIsLight ? 'light' : 'dark'

    const darkRaw = localStorage.getItem(DARK_KEY)
    const dark: ThemeId = isVariant(darkRaw, 'dark')
      ? darkRaw
      : legacyIsDark ? (legacy as ThemeId)
      : legacyIsLight ? COUNTERPART[legacy as ThemeId]
      : DEFAULT_PREFS.dark

    const lightRaw = localStorage.getItem(LIGHT_KEY)
    const light: ThemeId = isVariant(lightRaw, 'light')
      ? lightRaw
      : legacyIsLight ? (legacy as ThemeId)
      : legacyIsDark ? COUNTERPART[legacy as ThemeId]
      : DEFAULT_PREFS.light

    return { mode, dark, light }
  } catch {
    /* localStorage may be blocked (private mode, etc.) */
    return { ...DEFAULT_PREFS }
  }
}

/** Is the OS currently asking for a light scheme? */
export function osPrefersLight(): boolean {
  try {
    return window.matchMedia('(prefers-color-scheme: light)').matches
  } catch {
    return false
  }
}

/** The theme that should actually render for the given prefs. */
export function resolveTheme(prefs: ThemePrefs): ThemeId {
  const light = prefs.mode === 'light' || (prefs.mode === 'auto' && osPrefersLight())
  return light ? prefs.light : prefs.dark
}

/* Last-applied prefs + a lazily-registered OS-scheme listener so that
 * mode 'auto' live-switches when the OS flips (e.g. scheduled dark
 * mode at sunset) without a reload. Module-level singleton — the
 * listener stays for the page lifetime. */
let _current: ThemePrefs | null = null
let _listenerInstalled = false
const _changeSubs = new Set<() => void>()

function installOsListener(): void {
  if (_listenerInstalled) return
  _listenerInstalled = true
  try {
    const mq = window.matchMedia('(prefers-color-scheme: light)')
    mq.addEventListener('change', () => {
      if (_current?.mode === 'auto') {
        applyResolvedTheme(_current)
        _changeSubs.forEach(fn => fn())
      }
    })
  } catch {
    /* matchMedia unavailable — auto degrades to dark */
  }
}

/** Subscribe to effective-theme changes caused by the OS (auto mode).
 * Returns an unsubscribe function. */
export function onOsThemeChange(fn: () => void): () => void {
  _changeSubs.add(fn)
  return () => { _changeSubs.delete(fn) }
}

function applyResolvedTheme(prefs: ThemePrefs): void {
  const id = resolveTheme(prefs)
  const html = document.documentElement
  if (id === 'default') {
    html.removeAttribute('data-theme')
  } else {
    html.setAttribute('data-theme', id)
  }
  // Keep mobile browser chrome (address bar tint) in sync with the
  // page background. The pre-mount script in index.html does the same
  // for cold loads.
  const meta = document.querySelector('meta[name="theme-color"]')
  const swatch = THEMES.find(t => t.id === id)?.swatch
  if (meta && swatch) meta.setAttribute('content', swatch.bg)
  // Briefly enable transitions so the swap is animated. Remove the
  // class after the transition finishes so theme tokens elsewhere
  // (e.g. progress-bar fills, range-slider thumbs) don't pay the
  // 200ms transition cost on every interaction.
  html.classList.add('theme-transition')
  window.setTimeout(() => html.classList.remove('theme-transition'), 250)
}

export function applyThemePrefs(prefs: ThemePrefs): void {
  _current = prefs
  installOsListener()
  applyResolvedTheme(prefs)
  try {
    localStorage.setItem(MODE_KEY, prefs.mode)
    localStorage.setItem(DARK_KEY, prefs.dark)
    localStorage.setItem(LIGHT_KEY, prefs.light)
    // Keep the legacy key pointing at the resolved theme so a
    // downgrade to an older build still shows something sensible.
    localStorage.setItem(LEGACY_KEY, resolveTheme(prefs))
  } catch {
    /* localStorage blocked — theme still applies for this session */
  }
}
