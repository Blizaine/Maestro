import type { AnimatedTitleSettings } from './editorAnimatedTitles'

export function TitleGenerationControls({ settings, patch, busy }: {
  settings: AnimatedTitleSettings; patch: (value: Partial<AnimatedTitleSettings>) => void; busy: boolean
}) {
  const values = settings.generation ?? {}
  const update = (key: 'steps' | 'guidance' | 'seed', value: number) => patch({ generation: { ...values, [key]: value } })
  const field = 'w-full min-w-0 rounded-lg border border-border bg-bg-tertiary px-3 py-2 text-xs text-text-primary outline-none focus:border-accent-blue/60'
  return <fieldset className="space-y-3 border-b border-border pb-4">
    <legend className="mb-3 text-xs font-semibold text-text-primary">Generation</legend>
    <div className="grid grid-cols-2 gap-3">
      <label className="space-y-1.5 text-xs text-text-secondary">Steps
        <input aria-label="Title inference steps" type="number" min={1} max={200} step={1} disabled={busy}
          value={values.steps ?? 4} onChange={event => update('steps', Number(event.target.value))} className={field}/>
      </label>
      <label className="space-y-1.5 text-xs text-text-secondary">Guidance
        <input aria-label="Title guidance" type="number" min={0} max={30} step={.1} disabled={busy}
          value={values.guidance ?? 1} onChange={event => update('guidance', Number(event.target.value))} className={field}/>
      </label>
    </div>
    <label className="block space-y-1.5 text-xs text-text-secondary">Seed <span className="text-text-muted">(−1 for random)</span>
      <input aria-label="Title seed" type="number" min={-1} max={4294967295} step={1} disabled={busy}
        value={values.seed ?? -1} onChange={event => update('seed', Number(event.target.value))} className={field}/>
    </label>
  </fieldset>
}
