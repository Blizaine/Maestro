import { useStore } from '../../stores/useStore'
import type { AspectRatio } from '../../types'

const standardRatios: { value: AspectRatio; icon: string }[] = [
  { value: '16:9', icon: '▬' },
  { value: '9:16', icon: '▮' },
  { value: '1:1', icon: '◼' },
  { value: '4:3', icon: '▭' },
  { value: '3:4', icon: '▯' },
]

export function AspectRatioGrid() {
  const aspectRatio = useStore(s => s.aspectRatio)
  const setAspectRatio = useStore(s => s.setAspectRatio)
  const generationMode = useStore(s => s.generationMode)
  const isImage = generationMode === 'image'

  const ratios = isImage
    ? [{ value: 'auto' as AspectRatio, icon: '⊞' }, ...standardRatios]
    : standardRatios

  return (
    <div>
      <label className="text-[11px] text-text-muted uppercase tracking-wider mb-1.5 block">Aspect Ratio</label>
      <div className="flex gap-1">
        {ratios.map(r => (
          <button
            key={r.value}
            onClick={() => setAspectRatio(r.value)}
            className={`flex-1 flex flex-col items-center gap-0.5 py-2 rounded-lg border text-[10px] transition-all ${
              aspectRatio === r.value
                ? 'border-accent-blue bg-bg-active text-text-primary'
                : 'border-border text-text-muted hover:border-border-light hover:text-text-secondary'
            }`}
          >
            <span className="text-sm leading-none">{r.icon}</span>
            <span>{r.value === 'auto' ? 'Auto' : r.value}</span>
          </button>
        ))}
      </div>
    </div>
  )
}
