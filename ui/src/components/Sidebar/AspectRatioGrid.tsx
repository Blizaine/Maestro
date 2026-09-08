import { useStore } from '../../stores/useStore'
import type { AspectRatio } from '../../types'

const standardRatios: { value: AspectRatio; icon: string }[] = [
  { value: '16:9', icon: '▬' },
  { value: '9:16', icon: '▮' },
  { value: '1:1', icon: '◼' },
  { value: '4:3', icon: '▭' },
  { value: '3:4', icon: '▯' },
]

export function AspectRatioGrid({ menu = false, onSelect }: { menu?: boolean; onSelect?: () => void }) {
  const aspectRatio = useStore(s => s.aspectRatio)
  const setAspectRatio = useStore(s => s.setAspectRatio)
  const generationMode = useStore(s => s.generationMode)
  const modelOptions = useStore(s => s.modelOptions)
  const isImage = generationMode === 'image'
  const supportsUltraWide = Object.values(modelOptions?.resolution_presets || {}).some(
    preset => preset?.values?.['21:9'] != null,
  )
  const modelRatios = supportsUltraWide
    ? [{ value: '21:9' as AspectRatio, icon: '━' }, ...standardRatios]
    : standardRatios

  const ratios = isImage || modelOptions?.supports_auto_aspect
    ? [{ value: 'auto' as AspectRatio, icon: '⊞' }, ...modelRatios]
    : modelRatios

  return (
    <div>
      {!menu && <label className="text-[11px] text-text-muted uppercase tracking-wider mb-1.5 block">Aspect Ratio</label>}
      <div className={menu ? 'flex flex-col gap-0.5' : 'grid grid-cols-3 gap-1'}>
        {ratios.map(r => (
          <button
            key={r.value}
            type="button"
            role={menu ? 'menuitemradio' : undefined}
            aria-label={r.value === 'auto' ? 'Auto' : r.value}
            aria-checked={menu ? aspectRatio === r.value : undefined}
            aria-pressed={menu ? undefined : aspectRatio === r.value}
            onClick={() => { setAspectRatio(r.value); onSelect?.() }}
            className={`flex items-center rounded-lg transition-colors ${menu ? 'min-h-9 gap-2.5 px-2.5 text-xs' : 'flex-1 flex-col gap-0.5 py-2 border text-[10px]'} ${
              aspectRatio === r.value
                ? 'border-accent-blue bg-bg-active text-text-primary'
                : 'border-border text-text-muted hover:border-border-light hover:text-text-secondary'
            }`}
          >
            <span aria-hidden="true" className="text-sm leading-none">{r.icon}</span>
            <span>{r.value === 'auto' ? 'Auto' : r.value}</span>
            {menu && <span aria-hidden="true" className={`ml-auto ${aspectRatio === r.value ? '' : 'invisible'}`}>✓</span>}
          </button>
        ))}
      </div>
    </div>
  )
}
