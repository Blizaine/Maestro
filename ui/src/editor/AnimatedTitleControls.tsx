import { titleFormState, TITLE_EFFECTS, TITLE_EFFECT_LABELS } from './editorAnimatedTitles'
import type { AnimatedTitleSettings } from './editorAnimatedTitles'

const FIELD = 'w-full rounded-lg border border-border bg-bg-tertiary px-3 py-2 text-xs text-text-primary outline-none focus:border-accent-blue/60'

/** Shared title fields; placement and submission remain owned by each workspace. */
export function AnimatedTitleControls({ settings, patch, busy, canvas, destination = 'editor', at = 0 }: {
  settings: AnimatedTitleSettings
  patch: (value: Partial<AnimatedTitleSettings>) => void
  busy: boolean
  canvas: { width: number; height: number }
  destination?: 'editor' | 'studio'
  at?: number
}) {
  const background = settings.background ?? 'chroma'
  const { geometry, prompt, duration, validation } = titleFormState(settings, canvas)
  return <>
        <p className="text-xs leading-relaxed text-text-secondary">H3 Fused 4-Step · Frames. Matching {background === 'chroma' ? 'green' : 'black'} start/end frames are prepared automatically. {destination === 'studio' ? 'The finished clip is saved to your gallery.' : background === 'chroma' ? 'The result is inserted with chroma key enabled so footage below shows through.' : 'The result stays opaque, so its black background covers footage below.'}</p>
        <label className="block space-y-1.5 text-xs text-text-secondary">Title text
          <textarea aria-label="Animated title text" value={settings.text} maxLength={160} rows={2} disabled={busy}
            onChange={event => patch({ text: event.target.value })} placeholder="Create something extraordinary" className={FIELD} />
          <span className="block text-[11px] leading-relaxed text-text-muted">For a sequence, put each phrase on its own line, then describe their order and transitions below.</span>
        </label>
        <label className="block space-y-1.5 text-xs text-text-secondary">Title background
          <select aria-label="Title background" value={background} disabled={busy}
            onChange={event => patch({ background: event.target.value as NonNullable<AnimatedTitleSettings['background']> })} className={FIELD}>
            <option value="chroma">Chroma key (green)</option><option value="black">Black background</option>
          </select>
          <span className="block text-[11px] leading-relaxed text-text-muted">{background === 'chroma'
            ? destination === 'studio' ? 'The video keeps its green background for keying in the Editor or another editing app.' : 'Green is keyed automatically so the title and effects show over footage below.'
            : destination === 'studio' ? 'The video keeps its opaque black background.' : 'Black stays opaque. The title clip covers footage below for its full frame.'}</span>
        </label>
        <label className="block space-y-1.5 text-xs text-text-secondary">Lettering and visual style
          <input aria-label="Title visual style" value={settings.look} maxLength={1200} disabled={busy}
            onChange={event => patch({ look: event.target.value })} placeholder="Describe any style: anime, hand-drawn, metallic, minimal…" className={FIELD} />
        </label>
        <label className="block space-y-1.5 text-xs text-text-secondary">Describe the animation
          <textarea aria-label="Title animation description" value={settings.direction} maxLength={1200} rows={3} disabled={busy}
            onChange={event => patch({ direction: event.target.value, effect: 'custom' })}
            placeholder="Describe how the title appears, moves and disappears, including any timing you want." className={FIELD} />
        </label>
        <details className="rounded-lg border border-border bg-bg-tertiary p-3 text-xs text-text-secondary">
          <summary className="cursor-pointer">Use an animation preset (optional)</summary>
          <label className="mt-3 block space-y-1.5">Animation preset
            <select aria-label="Title animation preset" value={settings.effect} disabled={busy}
              onChange={event => {
                const effect = event.target.value as AnimatedTitleSettings['effect']
                patch({ effect, ...(effect !== 'custom' ? { direction: TITLE_EFFECTS[effect] } : {}) })
              }} className={FIELD}>
              <option value="custom">None — use my description</option>
              {Object.entries(TITLE_EFFECT_LABELS).map(([value, label]) => <option key={value} value={value}>{label}</option>)}
            </select>
          </label>
          <p className="mt-2 text-[11px] leading-relaxed text-text-muted">A preset fills the description as a starting point. Edit it freely; only the description is sent as the animation direction.</p>
        </details>
        <div className="grid grid-cols-2 gap-3">
          <label className="space-y-1.5 text-xs text-text-secondary">Duration (seconds)
            <input aria-label="Title duration" type="number" min={5} max={14.5} step={.1} value={settings.seconds} disabled={busy}
              onChange={event => patch({ seconds: Number(event.target.value) })} className={FIELD} />
          </label>
          <label className="space-y-1.5 text-xs text-text-secondary">Aspect ratio
            <select aria-label="Title aspect ratio" value={settings.aspect} disabled={busy} onChange={event => patch({ aspect: event.target.value as AnimatedTitleSettings['aspect'] })} className={FIELD}>
              {destination === 'editor' && <option value="project">Match project</option>}<option value="16:9">16:9 · landscape</option><option value="9:16">9:16 · portrait</option><option value="1:1">1:1 · square</option>
            </select>
          </label>
          <label className="space-y-1.5 text-xs text-text-secondary">Resolution
            <select aria-label="Title resolution" value={settings.resolution} disabled={busy} onChange={event => patch({ resolution: event.target.value as AnimatedTitleSettings['resolution'] })} className={FIELD}>
              <option value="480p">480p · faster</option><option value="720p">720p</option>
            </select>
          </label>
        </div>
        {destination === 'editor' && <label className="flex items-center gap-2 text-xs text-text-secondary"><input type="checkbox" checked={settings.keepAudio} disabled={busy}
          onChange={event => patch({ keepAudio: event.target.checked })} className="accent-[var(--color-accent-blue)]" />Keep generated clip audio</label>}
        {destination === 'studio' && <p className="text-[11px] leading-relaxed text-text-muted">Generated video audio is retained. You can mute it in the Editor or another editing app.</p>}
        <div className="rounded-lg border border-border bg-bg-tertiary p-3 text-[11px] leading-relaxed text-text-muted">
          {geometry.width}×{geometry.height} · {duration ? `${duration.toFixed(2)}s at 24 fps` : 'Check duration'}{destination === 'editor' ? ` · inserts at ${at.toFixed(2)}s.` : ' · saved to the gallery.'}<br />
          Background: {background === 'chroma' ? destination === 'studio' ? 'Green screen, ready for keying.' : 'Chroma key (green), removed automatically.' : 'Black, opaque.'}<br />
          No reference characters or Studio settings are reused. Review the wording, timing and edge quality; generated results can vary.
        </div>
        {prompt && <details className="text-xs text-text-secondary"><summary className="cursor-pointer">View prepared prompt</summary><pre className="mt-2 whitespace-pre-wrap rounded-lg bg-bg-tertiary p-3 text-[11px] leading-relaxed">{prompt}</pre></details>}
    {validation && (!geometry.width || !duration) && <p role="alert" className="text-xs text-red-400">{validation}</p>}
  </>
}
