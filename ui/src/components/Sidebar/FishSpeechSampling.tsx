import { useStore } from '../../stores/useStore'

export function FishSpeechSampling() {
  const architecture = useStore(s => s.modelOptions?.architecture)
  const topK = useStore(s => s.params.top_k ?? 0)
  const topP = useStore(s => s.params.top_p ?? 0.95)
  const setParam = useStore(s => s.setParam)
  if (architecture !== 'fish_drama3') return null

  return <div className="space-y-3">
    <label className="block">
      <span className="mb-1.5 flex items-center justify-between text-[11px] uppercase tracking-wider text-text-muted">
        Top-k <span className="text-xs normal-case text-text-secondary">{topK === 0 ? 'No cap' : topK}</span>
      </span>
      <input aria-label="Fish speech Top-k" type="number" min={0} max={4097} step={1} value={topK}
        onChange={event => {
          const value = Number(event.target.value)
          if (Number.isFinite(value)) setParam('top_k', Math.max(0, Math.min(4097, Math.round(value))))
        }}
        className="w-full rounded-lg border border-border bg-bg-tertiary px-3 py-2 text-sm text-text-primary focus:border-accent-blue focus:outline-none" />
    </label>
    <p className="text-[10px] leading-relaxed text-text-muted">
      Start with 0 (no cap). Low Top-k values can make Drama 3 fall silent before finishing the script. This reduces that risk; it does not guarantee complete speech or a voice match.
    </p>
    <label className="block">
      <span className="mb-1.5 flex items-center justify-between text-[11px] uppercase tracking-wider text-text-muted">
        Top-p <span className="text-xs text-text-secondary">{topP.toFixed(2)}</span>
      </span>
      <input aria-label="Fish speech Top-p" type="range" min={0.1} max={1} step={0.01} value={topP}
        onChange={event => setParam('top_p', Number(event.target.value))} className="w-full" />
    </label>
  </div>
}
