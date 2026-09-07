import { useState } from 'react'
import { ChevronDown, Monitor, RectangleHorizontal } from 'lucide-react'
import { useStore } from '../../stores/useStore'
import { ResolutionPresets } from './ResolutionPresets'
import { AspectRatioGrid } from './AspectRatioGrid'

/** The everyday canvas choices stay one tap away, without a permanent grid. */
export function OutputFormatControls() {
  const [expanded, setExpanded] = useState<'resolution' | 'aspect' | null>(null)
  const mode = useStore(s => s.generationMode)
  const editMode = useStore(s => s.editSubMode)
  const workflow = useStore(s => s.studioVideoWorkflow)
  const options = useStore(s => s.modelOptions)
  const preset = useStore(s => s.resolutionPreset)
  const ratio = useStore(s => s.aspectRatio)
  const resolution = useStore(s => s.params.resolution)
  if (mode === 'audio' || mode === 'tools'
    || (mode === 'avatar' && ['outpaint', 'recast', 'restyle'].includes(editMode))) return null
  const sourceAspect = mode === 'avatar' || (mode === 'video' && workflow === 'animate')
  const canResolve = !options?.hide_resolution_presets
  const toggle = (panel: 'resolution' | 'aspect') => setExpanded(value => value === panel ? null : panel)
  const chip = 'flex min-h-10 min-w-0 items-center justify-center gap-2 rounded-xl border px-3 text-xs transition-colors'
  return (
    <div className="min-w-0 space-y-2" aria-label="Output format">
      <div className="flex flex-wrap items-center gap-2">
        {canResolve && <button type="button" aria-label={`Resolution: ${preset}`} aria-expanded={expanded === 'resolution'}
          onClick={() => toggle('resolution')}
          className={`${chip} ${expanded === 'resolution' ? 'border-accent-blue bg-accent-blue/10 text-text-primary' : 'border-border bg-bg-tertiary text-text-secondary hover:border-border-light'}`}>
          <Monitor size={14} /><span>{preset === 'auto' ? 'Auto size' : preset}</span><ChevronDown size={12} />
        </button>}
        <button type="button" aria-label={`Aspect ratio: ${sourceAspect ? 'source' : ratio}`}
          aria-expanded={expanded === 'aspect'} disabled={sourceAspect}
          onClick={() => toggle('aspect')}
          title={sourceAspect ? 'Follows the source video framing' : 'Choose output aspect ratio'}
          className={`${chip} ${expanded === 'aspect' && !sourceAspect ? 'border-accent-blue bg-accent-blue/10 text-text-primary' : 'border-border bg-bg-tertiary text-text-secondary hover:border-border-light'}`}>
          <RectangleHorizontal size={14} /><span>{sourceAspect ? 'Source ratio' : ratio === 'auto' ? 'Auto ratio' : ratio}</span>
          {!sourceAspect && <ChevronDown size={12} />}
        </button>
        {!sourceAspect && !resolution.startsWith('auto') && <span className="text-[9px] text-text-muted tabular-nums">{resolution.replace('x', ' × ')}</span>}
      </div>
      {((expanded === 'resolution' && canResolve) || (expanded === 'aspect' && !sourceAspect)) && (
        <div className="rounded-xl border border-border bg-bg-tertiary/40 p-2.5">
          {expanded === 'resolution' ? <ResolutionPresets /> : <AspectRatioGrid />}
        </div>
      )}
    </div>
  )
}
