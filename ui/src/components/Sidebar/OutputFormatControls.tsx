import { useState } from 'react'
import { ChevronDown, Clock, Monitor, RectangleHorizontal } from 'lucide-react'
import { useStore } from '../../stores/useStore'
import { ResolutionPresets } from './ResolutionPresets'
import { AspectRatioGrid } from './AspectRatioGrid'
import { DurationSlider, WindowSettings } from './DurationSlider'
import { AudioDurationControl } from './AudioDurationControl'
import { H3MultiWindowControls } from './H3MultiWindowControls'
import { formatDuration } from '../../lib/durationPlanning'

/** The everyday canvas choices stay one tap away, without a permanent grid. */
export function OutputFormatControls() {
  const [expanded, setExpanded] = useState<'resolution' | 'aspect' | 'duration' | null>(null)
  const mode = useStore(s => s.generationMode)
  const editMode = useStore(s => s.editSubMode)
  const workflow = useStore(s => s.studioVideoWorkflow)
  const options = useStore(s => s.modelOptions)
  const preset = useStore(s => s.resolutionPreset)
  const ratio = useStore(s => s.aspectRatio)
  const resolution = useStore(s => s.params.resolution)
  const duration = useStore(s => s.durationSeconds)
  const durationMode = useStore(s => s.params._duration_planning_mode ?? 'auto')
  const imageMode = useStore(s => s.params.image_mode)
  const audioMode = useStore(s => s.audioSubMode)
  const hasDuration = (mode === 'video' && imageMode !== 4)
    || (mode === 'audio' && !['sfx', 'mixer', 'revoice'].includes(audioMode) && !!options?.duration_slider)
  if ((mode === 'audio' && !hasDuration) || mode === 'tools'
    || (mode === 'avatar' && ['outpaint', 'recast', 'restyle'].includes(editMode))) return null
  const sourceAspect = mode === 'avatar' || (mode === 'video' && workflow === 'animate')
  const canResolve = mode !== 'audio' && !options?.hide_resolution_presets
  const toggle = (panel: 'resolution' | 'aspect' | 'duration') => setExpanded(value => value === panel ? null : panel)
  const chip = 'flex flex-1 min-h-11 min-w-0 items-center justify-center gap-1.5 rounded-xl border px-2 text-xs transition-colors'
  return (
    <div className="min-w-0 space-y-2" aria-label="Output format">
      <div className="flex flex-wrap items-center gap-2">
        {canResolve && <button type="button" aria-label={`Resolution: ${preset}`} aria-expanded={expanded === 'resolution'}
          onClick={() => toggle('resolution')}
          className={`${chip} ${expanded === 'resolution' ? 'border-accent-blue bg-accent-blue/10 text-text-primary' : 'border-border bg-bg-tertiary text-text-secondary hover:border-border-light'}`}>
          <Monitor size={14} /><span>{preset === 'auto' ? 'Auto size' : preset}</span><ChevronDown size={12} />
        </button>}
        {mode !== 'audio' && <button type="button" aria-label={`Aspect ratio: ${sourceAspect ? 'source' : ratio}`}
          aria-expanded={expanded === 'aspect'} disabled={sourceAspect}
          onClick={() => toggle('aspect')}
          title={sourceAspect ? 'Follows the source video framing' : 'Choose output aspect ratio'}
          className={`${chip} ${expanded === 'aspect' && !sourceAspect ? 'border-accent-blue bg-accent-blue/10 text-text-primary' : 'border-border bg-bg-tertiary text-text-secondary hover:border-border-light'}`}>
          <RectangleHorizontal size={14} /><span>{sourceAspect ? 'Source ratio' : ratio === 'auto' ? 'Auto ratio' : ratio}</span>
          {!sourceAspect && <ChevronDown size={12} />}
        </button>}
        {hasDuration && <button type="button" aria-label={`Duration: ${formatDuration(duration, true)}`} aria-expanded={expanded === 'duration'}
          onClick={() => toggle('duration')} title={`${durationMode === 'auto' ? 'Auto · ' : ''}${formatDuration(duration, true)} · duration and window settings`}
          className={`${chip} ${expanded === 'duration' ? 'border-accent-blue bg-accent-blue/10 text-text-primary' : 'border-border bg-bg-tertiary text-text-secondary hover:border-border-light'}`}>
          <Clock size={14} className="shrink-0"/><span className="truncate tabular-nums">{formatDuration(duration, true)}</span><ChevronDown size={12} className="shrink-0"/>
        </button>}
      </div>
      {((expanded === 'resolution' && canResolve) || (expanded === 'aspect' && mode !== 'audio' && !sourceAspect)) && (
        <div className="rounded-xl border border-border bg-bg-tertiary/40 p-2.5">
          {expanded === 'resolution' ? <ResolutionPresets /> : <AspectRatioGrid />}
        </div>
      )}
      {/* Duration owns automatic window sizing. Keep it mounted while hidden. */}
      {hasDuration && <div hidden={expanded !== 'duration'} className="space-y-3 rounded-xl border border-border bg-bg-tertiary/40 p-2.5">
        {mode === 'audio' ? <AudioDurationControl/> : <><DurationSlider/><WindowSettings/><H3MultiWindowControls section="continuity"/></>}
      </div>}
      {expanded === 'resolution' && canResolve && !sourceAspect && !resolution.startsWith('auto') && <p className="text-[10px] text-text-muted tabular-nums">{resolution.replace('x', ' × ')} pixels</p>}
    </div>
  )
}
