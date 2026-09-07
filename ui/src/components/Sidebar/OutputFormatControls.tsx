import { useId, useState } from 'react'
import { Clock, Monitor, RectangleHorizontal } from 'lucide-react'
import { useStore } from '../../stores/useStore'
import { ResolutionPresets } from './ResolutionPresets'
import { AspectRatioGrid } from './AspectRatioGrid'
import { DurationSlider, WindowSettings } from './DurationSlider'
import { AudioDurationControl } from './AudioDurationControl'
import { H3MultiWindowControls } from './H3MultiWindowControls'
import { formatDuration } from '../../lib/durationPlanning'
import { SidebarDialog } from './SidebarPanels'

/** The everyday canvas choices stay one tap away, without a permanent grid. */
export function OutputFormatControls() {
  const [expanded, setExpanded] = useState<'resolution' | 'aspect' | 'duration' | null>(null)
  const id = useId()
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
  const chip = 'studio-setting-chip'
  return (
    <>
        {canResolve && <button type="button" aria-label={`Resolution: ${preset}`} aria-expanded={expanded === 'resolution'}
          aria-controls={`${id}-resolution`} title="Resolution"
          onClick={() => toggle('resolution')}
          className={`${chip} ${expanded === 'resolution' ? 'border-accent-blue bg-accent-blue/10 text-text-primary' : 'border-border bg-bg-tertiary text-text-secondary hover:border-border-light'}`}>
          <Monitor size={13} className="studio-setting-icon" /><span>{preset === 'auto' ? 'Auto' : preset}</span>
        </button>}
        {mode !== 'audio' && <button type="button" aria-label={`Aspect ratio: ${sourceAspect ? 'source' : ratio}`}
          aria-expanded={expanded === 'aspect'} disabled={sourceAspect}
          aria-controls={`${id}-aspect`}
          onClick={() => toggle('aspect')}
          title={sourceAspect ? 'Follows the source video framing' : 'Choose output aspect ratio'}
          className={`${chip} ${expanded === 'aspect' && !sourceAspect ? 'border-accent-blue bg-accent-blue/10 text-text-primary' : 'border-border bg-bg-tertiary text-text-secondary hover:border-border-light'}`}>
          <RectangleHorizontal size={13} className="studio-setting-icon" /><span>{sourceAspect ? 'Source' : ratio === 'auto' ? 'Auto' : ratio}</span>
        </button>}
        {hasDuration && <button type="button" aria-label={`Duration: ${formatDuration(duration, true)}`} aria-expanded={expanded === 'duration'}
          aria-controls={`${id}-duration`}
          onClick={() => toggle('duration')} title={`${durationMode === 'auto' ? 'Auto · ' : ''}${formatDuration(duration, true)} · duration and window settings`}
          className={`${chip} ${expanded === 'duration' ? 'border-accent-blue bg-accent-blue/10 text-text-primary' : 'border-border bg-bg-tertiary text-text-secondary hover:border-border-light'}`}>
          <Clock size={13} className="studio-setting-icon"/><span className="truncate tabular-nums">{durationMode === 'auto' ? 'Auto' : formatDuration(duration, true)}</span>
        </button>}
      <SidebarDialog id={`${id}-resolution`} variant="settings" open={expanded === 'resolution' && canResolve} title="Resolution" onClose={() => setExpanded(null)}>
        {expanded === 'resolution' && canResolve && <ResolutionPresets />}
        {!sourceAspect && !resolution.startsWith('auto') && <p className="mt-3 text-[10px] text-text-muted tabular-nums">{resolution.replace('x', ' × ')} pixels</p>}
      </SidebarDialog>
      <SidebarDialog id={`${id}-aspect`} variant="settings" open={expanded === 'aspect' && mode !== 'audio' && !sourceAspect} title="Aspect ratio" onClose={() => setExpanded(null)}>
        {expanded === 'aspect' && mode !== 'audio' && !sourceAspect && <AspectRatioGrid />}
      </SidebarDialog>
      {/* Duration owns automatic window sizing. Keep it mounted while hidden. */}
      {hasDuration && <SidebarDialog id={`${id}-duration`} variant="settings" open={expanded === 'duration'} title="Duration & windows" onClose={() => setExpanded(null)}><div className="space-y-3">
        {mode === 'audio' ? <AudioDurationControl/> : <><DurationSlider/><WindowSettings/><H3MultiWindowControls section="continuity"/></>}
      </div></SidebarDialog>}
    </>
  )
}
