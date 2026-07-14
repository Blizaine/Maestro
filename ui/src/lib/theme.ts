/**
 * Theme management for Maestro.
 *
 * Themes are CSS-variable overrides applied via `[data-theme="..."]` on
 * the <html> element. The actual variable values live in src/index.css.
 *
 * Persistence: localStorage under "maestro-theme". An inline script in
 * index.html applies the persisted theme to <html> before React mounts
 * so there's no flash of the default theme on page load.
 *
 * Adding a new theme: add a `[data-theme="<id>"]` block in index.css
 * with the variable overrides, add the id to THEMES below, and it
 * shows up in the Settings/System dropdown automatically.
 */

export type ThemeId = 'default' | 'golden-hour' | 'onyx' | 'ivory'

export interface ThemeDescriptor {
  id: ThemeId
  label: string
  description: string
  /** Three-color preview swatch shown in the settings dropdown. */
  swatch: { bg: string; surface: string; accent: string }
}

/* THEMES ordering doubles as the dropdown order. Default theme is
 * listed first so it's the most prominent option. The `id` of the
 * cool palette stays 'default' (despite no longer being the default)
 * so existing localStorage values from users who explicitly chose
 * it don't break — only the LABEL changed to 'Classic'. */
export const THEMES: ThemeDescriptor[] = [
  {
    id: 'golden-hour',
    label: 'Golden Hour',
    description:
      'Default. Warm cinematic palette — near-black surfaces, warm-tinted borders, amber highlights, sunset gradient on primary actions.',
    swatch: { bg: '#0a0a0a', surface: '#181818', accent: '#f97316' },
  },
  {
    id: 'default',
    label: 'Classic',
    description: 'The original cool charcoal palette with blue accents.',
    swatch: { bg: '#0a0a0f', surface: '#1a1a25', accent: '#3b82f6' },
  },
  {
    id: 'onyx',
    label: 'Onyx',
    description:
      'Minimalist monochrome — pure black backgrounds, neutral grey surfaces, white-toned accents. No color tint.',
    swatch: { bg: '#000000', surface: '#1a1a1a', accent: '#aaaaaa' },
  },
  {
    id: 'ivory',
    label: 'Ivory',
    description:
      'Light theme — warm paper surfaces, coffee-toned text, burnt-orange accents. Golden Hour in daylight.',
    swatch: { bg: '#f2ede2', surface: '#f9f6ee', accent: '#c2410c' },
  },
]

const STORAGE_KEY = 'maestro-theme'

/* DEFAULT_THEME is the fallback used when no theme has been
 * explicitly chosen (empty localStorage). Users who previously
 * picked 'default' (Classic) will keep that preference because their
 * localStorage value is intact; only fresh installs and users who
 * never opened Settings get the new default. */
export const DEFAULT_THEME: ThemeId = 'golden-hour'

export function getStoredTheme(): ThemeId {
  try {
    const t = localStorage.getItem(STORAGE_KEY)
    if (t && THEMES.some(theme => theme.id === t)) {
      return t as ThemeId
    }
  } catch {
    /* localStorage may be blocked (private mode, etc.) — fall through */
  }
  return DEFAULT_THEME
}

export function applyTheme(id: ThemeId): void {
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
  try {
    localStorage.setItem(STORAGE_KEY, id)
  } catch {
    /* localStorage blocked — theme still applies for this session */
  }
}
