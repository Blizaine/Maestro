import { useStore } from '../../stores/useStore'
import type { ScailResolutionProfile } from '../../types'
import { InfoTooltip } from './InfoTooltip'

interface ScailResolutionSelectorProps {
  value: ScailResolutionProfile
  onChange: (value: ScailResolutionProfile) => void
  workflow: 'Recast' | 'Repaint'
}

const OPTIONS: Array<{
  value: ScailResolutionProfile
  label: string
  detail: string
}> = [
  {
    value: '480p',
    label: '480p',
    detail: 'Fast',
  },
  {
    value: '512p',
    label: '512p',
    detail: 'Balanced',
  },
  {
    value: '704p',
    label: '704p',
    detail: 'Experimental',
  },
]

/** Shared spatial-quality selector for the two dedicated SCAIL-2 workflows. */
export function ScailResolutionSelector({
  value,
  onChange,
  workflow,
}: ScailResolutionSelectorProps) {
  const totalVramGb = useStore(s => s.systemStats?.gpu.vram_total_gb || 0)

  return (
    <div className="space-y-1">
      <div className="flex items-center gap-1">
        <label className="text-[10px] text-text-muted uppercase tracking-wider">
          Output quality
        </label>
        <InfoTooltip
          label={`About ${workflow} output quality`}
          text="480p (832×480) is fastest. 512p (896×512) balances detail and VRAM. 704p (up to 1280×704) is experimental, requires at least 16 GB VRAM, and automatically uses shorter 33–49 frame windows. Resolution does not change the selected model, inference steps, or guidance."
        />
      </div>
      <div className="grid grid-cols-3 gap-1">
        {OPTIONS.map(option => {
          const disabled = (
            option.value === '704p'
            && totalVramGb > 0
            && totalVramGb < 16
          )
          return (
            <button
              key={option.value}
              type="button"
              disabled={disabled}
              aria-pressed={value === option.value}
              title={disabled ? `704p ${workflow} requires at least 16 GB VRAM` : undefined}
              onClick={() => onChange(option.value)}
              className={`rounded border px-2 py-1.5 text-left transition-colors ${
                value === option.value
                  ? 'border-accent-blue bg-accent-blue/10 text-text-primary'
                  : 'border-border bg-bg-secondary/40 text-text-secondary hover:border-accent-blue/50'
              } ${disabled ? 'cursor-not-allowed opacity-45' : ''}`}
            >
              <span className="block text-[10px] font-medium">{option.label}</span>
              <span className="block text-[8px] text-text-muted">{option.detail}</span>
            </button>
          )
        })}
      </div>
    </div>
  )
}
