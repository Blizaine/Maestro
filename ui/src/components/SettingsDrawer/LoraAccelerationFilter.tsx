import { useStore } from '../../stores/useStore'
import type { LoraAccelerationFilter as Filter } from '../../lib/loraAcceleration'

export function LoraAccelerationFilter() {
  const filter = useStore(state => state.loraAccelerationFilter)
  const setFilter = useStore(state => state.setLoraAccelerationFilter)
  return <label className="mb-2 flex min-w-0 items-center gap-2 text-[10px] text-text-muted">
    <span className="shrink-0">Turbo LoRAs</span>
    <select aria-label="Turbo LoRA filter" value={filter} onChange={event => setFilter(event.target.value as Filter)}
      title="Filter Turbo and other acceleration adapters, including Lightning, PDD and low-step LoRAs. Selected adapters stay active and visible."
      className="min-w-0 flex-1 rounded-lg border border-border bg-bg-tertiary px-2 py-1 text-xs text-text-secondary focus:outline-none focus:border-accent-blue">
      <option value="all">All LoRAs</option>
      <option value="hide">Hide Turbo</option>
      <option value="only">Turbo only</option>
    </select>
  </label>
}
