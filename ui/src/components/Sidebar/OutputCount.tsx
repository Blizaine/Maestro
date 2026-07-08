import { useStore } from '../../stores/useStore'

export function OutputCount() {
  const count = useStore(s => s.outputCount)
  const setCount = useStore(s => s.setOutputCount)

  return (
    <div>
      <div className="flex items-center justify-between mb-1.5">
        <label className="text-[11px] text-text-muted uppercase tracking-wider">Outputs</label>
        <span className="text-xs text-text-secondary">{count}</span>
      </div>
      <input
        type="range"
        min={1}
        max={6}
        step={1}
        value={count}
        onChange={e => setCount(Number(e.target.value))}
      />
    </div>
  )
}
