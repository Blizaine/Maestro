import type { AnimatedTitleSettings } from './editorAnimatedTitles'

const FIELD = 'w-full min-w-0 rounded-lg border border-border bg-bg-tertiary px-3 py-2 text-xs text-text-primary outline-none focus:border-accent-blue/60'

export function TitleBackgroundControl({ settings, patch, busy, editor = false }: {
  settings: AnimatedTitleSettings
  patch: (value: Partial<AnimatedTitleSettings>) => void
  busy: boolean
  editor?: boolean
}) {
  return <div className="min-w-0 space-y-2">
    <label className="block min-w-0 space-y-1.5 text-xs text-text-secondary">Background
      <select aria-label="Title background" value={settings.background ?? 'chroma'} disabled={busy}
        onChange={event => patch({ background: event.target.value as AnimatedTitleSettings['background'],
          ...(event.target.value === 'custom' ? { backgroundColor: settings.backgroundColor ?? '#202020' } : {}) })} className={FIELD}>
        <option value="chroma">Green (key)</option><option value="blue">Blue (key)</option>
        <option value="black">Black</option><option value="white">White</option><option value="custom">Custom color</option>
      </select>
    </label>
    {settings.background === 'custom' && <label className="flex items-center gap-2 text-xs text-text-secondary">Color
      <input aria-label="Title background color" type="color" disabled={busy} value={settings.backgroundColor ?? '#202020'}
        onChange={event => patch({ backgroundColor: event.target.value })}
        className="h-9 w-12 cursor-pointer rounded border border-border bg-bg-tertiary p-1 disabled:opacity-40" />
      <span className="font-mono text-[11px]">{settings.backgroundColor ?? '#202020'}</span>
    </label>}
    {editor && <p className="text-[11px] leading-relaxed text-text-muted">{!settings.background || settings.background === 'chroma'
      ? 'The Editor removes green automatically so the title shows over footage below.'
      : settings.background === 'blue' ? 'The Editor removes blue automatically so the title shows over footage below.'
      : settings.background === 'black' ? 'Black stays opaque. The title clip covers footage below for its full frame.'
      : 'This background stays opaque and covers footage below.'}</p>}
  </div>
}
