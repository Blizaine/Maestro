/**
 * Theme management for Maestro.
 *
 * Themes are CSS-variable overrides applied via `[data-theme="..."]` on
 * the <html> element. The actual variable values live in src/index.css.
 *
 * The user-facing model is four theme families (Graphite Studio, Golden
 * Hour, Classic, Onyx), each with dark and light variants and an appearance mode:
 *   - mode: 'dark' | 'light' | 'auto' — auto follows the OS scheme
 *     (prefers-color-scheme) and live-switches when the OS changes.
 * The effective CSS theme = the chosen family's variant for whichever
 * scheme the mode resolves to. Light variants use internal CSS ids rather
 * than separate user-facing theme names.
 *
 * Persistence: localStorage under "maestro-theme-mode" and
 * "maestro-theme-family". On the first v4 load, existing theme settings
 * are reset once to Graphite Studio dark. The fixed v4 migration marker
 * keeps that reset from running again on patch updates. Afterward, the
 * original single-theme key ("maestro-theme", may hold a light variant
 * id) and short-lived per-variant key ("maestro-theme-dark") remain
 * readable for compatibility. An inline script in index.html applies
 * the same migration and resolved theme before React mounts.
 *
 * Adding a new family: add `[data-theme]` blocks for both variants in
 * index.css, add the family to FAMILIES below, and extend the maps in
 * the index.html pre-mount script.
 */

export type ThemeId = 'graphite-studio' | 'graphite-daylight' | 'default' | 'golden-hour' | 'onyx' | 'ivory' | 'daylight' | 'pearl'
export type FamilyId = 'graphite-studio' | 'default' | 'golden-hour' | 'onyx'
export type ThemeMode = 'dark' | 'light' | 'auto'

export interface ThemeVariant {
  /** The [data-theme] id this variant renders as. */
  id: ThemeId
  /** Three-color preview swatch shown in the settings picker. */
  swatch: { bg: string; surface: string; accent: string }
}

export interface ThemeFamily {
  id: FamilyId
  label: string
  description: string
  dark: ThemeVariant
  light: ThemeVariant
}

/* FAMILIES ordering doubles as the dropdown order. The Classic
 * family's id stays 'default' (despite no longer being the default)
 * so localStorage values from users who explicitly chose it don't
 * break — only the LABEL says 'Classic'. */
export const FAMILIES: ThemeFamily[] = [
  {
    id: 'graphite-studio',
    label: 'Graphite Studio',
    description:
      'Default. Matte graphite surfaces and flat orange highlights; silver-grey surfaces and burnt orange in daylight.',
    dark: { id: 'graphite-studio', swatch: { bg: '#111316', surface: '#191c20', accent: '#ff985f' } },
    light: { id: 'graphite-daylight', swatch: { bg: '#d9dfe6', surface: '#e6ebf0', accent: '#b34312' } },
  },
  {
    id: 'golden-hour',
    label: 'Golden Hour',
    description:
      'Warm cinematic palette — near-black surfaces and amber highlights at night; warm paper and burnt orange in daylight.',
    dark: { id: 'golden-hour', swatch: { bg: '#0a0a0a', surface: '#181818', accent: '#f97316' } },
    light: { id: 'ivory', swatch: { bg: '#f2ede2', surface: '#f9f6ee', accent: '#c2410c' } },
  },
  {
    id: 'default',
    label: 'Classic',
    description:
      'The original cool palette with blue accents — charcoal at night, cool paper in daylight.',
    dark: { id: 'default', swatch: { bg: '#0a0a0f', surface: '#1a1a25', accent: '#3b82f6' } },
    light: { id: 'daylight', swatch: { bg: '#f4f5f7', surface: '#fafbfc', accent: '#2563eb' } },
  },
  {
    id: 'onyx',
    label: 'Onyx',
    description:
      'Minimalist monochrome — pure black at night, white and grey in daylight. No color tint.',
    dark: { id: 'onyx', swatch: { bg: '#000000', surface: '#1a1a1a', accent: '#aaaaaa' } },
    light: { id: 'pearl', swatch: { bg: '#f2f2f2', surface: '#f9f9f9', accent: '#525252' } },
  },
]

/** Any theme id (either variant) -> its family. */
const FAMILY_OF: Record<ThemeId, FamilyId> = {
  'graphite-studio': 'graphite-studio',
  'graphite-daylight': 'graphite-studio',
  'golden-hour': 'golden-hour',
  ivory: 'golden-hour',
  default: 'default',
  daylight: 'default',
  onyx: 'onyx',
  pearl: 'onyx',
}

const LIGHT_IDS: ReadonlySet<string> = new Set(['graphite-daylight', 'ivory', 'daylight', 'pearl'])

export interface ThemePrefs {
  mode: ThemeMode
  family: FamilyId
}

const MODE_KEY = 'maestro-theme-mode'
const FAMILY_KEY = 'maestro-theme-family'
/** Fixed major-version marker: v4 patch releases must preserve user choices. */
const V4_MIGRATION_KEY = 'maestro-theme-v4-migrated'
const V4_MIGRATION_VALUE = '4'
/** Short-lived key from the interim two-picker build; holds a dark id. */
const INTERIM_DARK_KEY = 'maestro-theme-dark'
/** Original single-theme key; may hold either variant. Still written
 * with the resolved theme so downgrades show something sensible. */
const LEGACY_KEY = 'maestro-theme'

const DEFAULT_PREFS: ThemePrefs = { mode: 'dark', family: 'graphite-studio' }

/** Persist the compatible theme keys, then mark the one-time v4 reset complete. */
function writeThemePrefs(storage: Storage, prefs: ThemePrefs): boolean {
  try {
    storage.setItem(MODE_KEY, prefs.mode)
    storage.setItem(FAMILY_KEY, prefs.family)
    // Keep the legacy key pointing at the resolved theme so a downgrade to
    // an older build still shows something sensible.
    storage.setItem(LEGACY_KEY, resolveTheme(prefs))
    // This must remain the final write. If any preference write fails, a
    // later load can safely retry the migration instead of trusting partial
    // theme state as a completed v4 migration.
    storage.setItem(V4_MIGRATION_KEY, V4_MIGRATION_VALUE)
    return true
  } catch {
    /* Storage can be blocked or quota-limited; keep the current session usable. */
    return false
  }
}

function isFamily(id: string | null): id is FamilyId {
  return !!id && FAMILIES.some(f => f.id === id)
}

export function getStoredPrefs(): ThemePrefs {
  try {
    const storage = localStorage
    if (storage.getItem(V4_MIGRATION_KEY) !== V4_MIGRATION_VALUE) {
      // v4 intentionally starts every existing install on the new default.
      // If storage is partially writable, leave the marker absent so this is
      // retried on a later load; the current page still uses these defaults.
      writeThemePrefs(storage, DEFAULT_PREFS)
      return { ...DEFAULT_PREFS }
    }

    const legacyRaw = storage.getItem(LEGACY_KEY)
    const legacy = legacyRaw && Object.prototype.hasOwnProperty.call(FAMILY_OF, legacyRaw)
      ? (legacyRaw as ThemeId)
      : null

    const modeRaw = storage.getItem(MODE_KEY)
    const mode: ThemeMode =
      modeRaw === 'dark' || modeRaw === 'light' || modeRaw === 'auto'
        ? modeRaw
        // Migration: a stored light variant means the user chose light.
        : legacy && LIGHT_IDS.has(legacy) ? 'light' : 'dark'

    const familyRaw = storage.getItem(FAMILY_KEY)
    const interimDark = storage.getItem(INTERIM_DARK_KEY)
    const family: FamilyId = isFamily(familyRaw)
      ? familyRaw
      : isFamily(interimDark) ? interimDark
      : legacy ? FAMILY_OF[legacy]
      : DEFAULT_PREFS.family

    return { mode, family }
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

/** The variant ('dark' | 'light') the given prefs resolve to right now. */
export function resolveVariant(prefs: ThemePrefs): 'dark' | 'light' {
  return prefs.mode === 'light' || (prefs.mode === 'auto' && osPrefersLight())
    ? 'light'
    : 'dark'
}

/** The CSS theme id that should actually render for the given prefs. */
export function resolveTheme(prefs: ThemePrefs): ThemeId {
  const fam = FAMILIES.find(f => f.id === prefs.family) ?? FAMILIES[0]
  return fam[resolveVariant(prefs)].id
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
  const fam = FAMILIES.find(f => f.id === prefs.family) ?? FAMILIES[0]
  const swatch = fam[resolveVariant(prefs)].swatch
  if (meta) meta.setAttribute('content', swatch.bg)
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
    writeThemePrefs(localStorage, prefs)
  } catch {
    /* localStorage blocked — theme still applies for this session */
  }
}
