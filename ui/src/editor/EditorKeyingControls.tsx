import { useId, useState } from 'react'
import { Pipette, RotateCcw } from 'lucide-react'
import type { EditorKeyingSettings } from '../types'
import { DEFAULT_EDITOR_KEYING, editorSpillChannel, normalizeEditorKeying } from './editorKeying'

export function EditorKeyingControls({ value, disabled, onChange }: {
  value?: EditorKeyingSettings; disabled: boolean; onChange: (value: EditorKeyingSettings) => void
}) {
  const current = normalizeEditorKeying(value)
  const id = useId()
  const [colorDraft, setColorDraft] = useState<string | null>(null)
  const patch = (update: Partial<EditorKeyingSettings>) => onChange(normalizeEditorKeying({ ...current, ...update }))
  const commitColor = () => {
    if (colorDraft !== null && /^#[0-9a-f]{6}$/i.test(colorDraft)) patch({ color: colorDraft })
    setColorDraft(null)
  }
  return <section aria-label="Chroma key" className="space-y-3 rounded-xl border border-border bg-bg-tertiary p-3">
    <div className="flex items-start justify-between gap-2">
      <label className="flex cursor-pointer items-center gap-2 text-xs font-semibold">
        <input type="checkbox" aria-label="Enable chroma key" checked={current.enabled} disabled={disabled}
          onChange={event => patch({ enabled: event.target.checked })} className="accent-[var(--color-accent-blue)]" />
        <Pipette size={14} /> Chroma key
      </label>
      <button type="button" disabled={disabled} aria-label="Reset chroma key" onClick={() => { setColorDraft(null); onChange({ ...DEFAULT_EDITOR_KEYING }) }}
        className="flex items-center gap-1 rounded-md px-1.5 py-1 text-[11px] text-text-secondary hover:bg-bg-hover disabled:opacity-40"><RotateCcw size={11} />Reset</button>
    </div>
    <p className="text-[11px] leading-relaxed text-text-muted">Remove a solid background to reveal the video or image on a lower track.</p>
    {current.enabled && <>
      <div className="flex gap-2">
        {([['Green', '#00ff00'], ['Blue', '#0000ff']] as const).map(([label, color]) => <button key={color} type="button" disabled={disabled}
          aria-pressed={current.color === color} onClick={() => { setColorDraft(null); patch({ color }) }}
          className={`flex flex-1 items-center justify-center gap-1.5 rounded-lg border px-2 py-2 text-xs disabled:opacity-40 ${current.color === color ? 'border-accent-blue/60 bg-accent-blue/10' : 'border-border hover:bg-bg-hover'}`}>
          <span className="h-3 w-3 rounded-full border border-white/20" style={{ background: color }} />{label}</button>)}
      </div>
      <label htmlFor={`${id}-color`} className="block text-[11px] text-text-secondary">Key color</label>
      <div className="flex gap-2">
        <input id={`${id}-color`} type="color" aria-label="Pick key color" value={current.color} disabled={disabled}
          onChange={event => { setColorDraft(null); patch({ color: event.target.value }) }} className="h-9 w-12 cursor-pointer rounded border border-border bg-bg-secondary p-1" />
        <input aria-label="Key color hex" value={colorDraft ?? current.color} maxLength={7} disabled={disabled}
          onChange={event => setColorDraft(event.target.value)} onBlur={commitColor}
          onKeyDown={event => { if (event.key === 'Enter') event.currentTarget.blur(); if (event.key === 'Escape') { event.preventDefault(); event.stopPropagation(); setColorDraft(null) } }}
          className="min-w-0 flex-1 rounded-lg border border-border bg-bg-secondary px-3 text-xs font-mono" />
      </div>
      {([
        ['similarity', 'Color tolerance', 'Increase to remove more shades of the background.'],
        ['softness', 'Edge softness', 'Blend fine dust and semi-transparent edges instead of cutting them off.'],
        ['spill', 'Spill reduction', 'Reduce green or blue tint remaining around the subject.'],
      ] as const).map(([property, label, detail]) => <label key={property} className="block space-y-1.5 text-[11px] text-text-secondary">
        <span className="flex justify-between gap-2"><span>{label}</span><span className="font-mono">{Math.round(current[property] * 100)}%</span></span>
        <input type="range" aria-label={label} min={property === 'similarity' ? .00001 : 0} max={1} step={.005}
          value={current[property]} disabled={disabled || (property === 'spill' && editorSpillChannel(current.color) === 0)}
          onChange={event => patch({ [property]: Number(event.target.value) })} className="w-full accent-[var(--color-accent-blue)] disabled:opacity-40" />
        <span className="block leading-relaxed text-text-muted">{detail}</span>
      </label>)}
      <p className="text-[11px] leading-relaxed text-text-muted">Preview uses the original source. Keying is applied before color adjustments; exports keep the same settings.</p>
    </>}
  </section>
}
