import { useId, useState } from 'react'
import { Clock, Monitor, RectangleHorizontal } from 'lucide-react'
import { useStore } from '../../stores/useStore'
import { ResolutionPresets } from './ResolutionPresets'
import { AspectRatioGrid } from './AspectRatioGrid'
import { DurationSlider } from './DurationSlider'
import { AudioDurationControl } from './AudioDurationControl'
import { H3MultiWindowControls } from './H3MultiWindowControls'
import { formatDuration } from '../../lib/durationPlanning'
import { SidebarDialog } from './SidebarPanels'
import { SidebarMenu } from './SidebarMenu'

/** The everyday canvas choices stay one tap away, without a permanent grid. */
export function OutputFormatControls() {
  const [expanded, setExpanded] = useState<'resolution' | 'aspect' | 'duration' | null>(null)
  const [resolutionAnchor, setResolutionAnchor] = useState<HTMLButtonElement | null>(null)
  const [aspectAnchor, setAspectAnchor] = useState<HTMLButtonElement | null>(null)
  const [durationAnchor, setDurationAnchor] = useState<HTMLButtonElement | null>(null)
  const id = useId()
  const mode = useStore(s => s.generationMode)
  const editMode = useStore(s => s.editSubMode)
  const workflow = useStore(s => s.studioVideoWorkflow)
  const options = useStore(s => s.modelOptions)
  const preset = useStore(s => s.resolutionPreset)
  const ratio = useStore(s => s.aspectRatio)
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
        {canResolve && <button ref={setResolutionAnchor} type="button" aria-label={`Resolution: ${preset}`} aria-expanded={expanded === 'resolution'} aria-haspopup="menu"
          aria-controls={`${id}-resolution`} title="Resolution"
          onClick={() => toggle('resolution')}
          className={`${chip} ${expanded === 'resolution' ? 'border-accent-blue bg-accent-blue/10 text-text-primary' : 'border-border bg-bg-tertiary text-text-secondary hover:border-border-light'}`}>
          <Monitor size={13} className="studio-setting-icon" /><span>{preset === 'auto' ? 'Auto' : preset}</span>
        </button>}
        {mode !== 'audio' && <button ref={setAspectAnchor} type="button" aria-label={`Aspect ratio: ${sourceAspect ? 'source' : ratio}`} aria-haspopup="menu"
          aria-expanded={expanded === 'aspect'} disabled={sourceAspect}
          aria-controls={`${id}-aspect`}
          onClick={() => toggle('aspect')}
          title={sourceAspect ? 'Follows the source video framing' : 'Choose output aspect ratio'}
          className={`${chip} ${expanded === 'aspect' && !sourceAspect ? 'border-accent-blue bg-accent-blue/10 text-text-primary' : 'border-border bg-bg-tertiary text-text-secondary hover:border-border-light'}`}>
          <RectangleHorizontal size={13} className="studio-setting-icon" /><span>{sourceAspect ? 'Source' : ratio === 'auto' ? 'Auto' : ratio}</span>
        </button>}
        {hasDuration && <button ref={setDurationAnchor} type="button" aria-label={`Duration: ${durationMode === 'auto' ? 'Auto · ' : ''}${formatDuration(duration, true)}`} aria-expanded={expanded === 'duration'} aria-haspopup="dialog"
          aria-controls={`${id}-duration`}
          onClick={() => toggle('duration')} title={`${durationMode === 'auto' ? 'Auto · ' : ''}${formatDuration(duration, true)} · duration and window settings`}
          className={`${chip} ${expanded === 'duration' ? 'border-accent-blue bg-accent-blue/10 text-text-primary' : 'border-border bg-bg-tertiary text-text-secondary hover:border-border-light'}`}>
          <Clock size={13} className="studio-setting-icon"/>
          <span className="flex flex-col items-center tabular-nums leading-3">
            {durationMode === 'auto' && <span className="text-[9px] leading-[10px] text-text-muted">Auto</span>}
            <span>{formatDuration(duration, true)}</span>
          </span>
        </button>}
      <SidebarMenu id={`${id}-resolution`} anchor={resolutionAnchor} open={expanded === 'resolution' && canResolve} label="Resolution" width="content" align="start" onClose={() => setExpanded(null)}>
        <ResolutionPresets menu onSelect={() => setExpanded(null)} />
      </SidebarMenu>
      <SidebarMenu id={`${id}-aspect`} anchor={aspectAnchor} open={expanded === 'aspect' && mode !== 'audio' && !sourceAspect} label="Aspect ratio" width="content" align="start" onClose={() => setExpanded(null)}>
        <AspectRatioGrid menu onSelect={() => setExpanded(null)} />
      </SidebarMenu>
      {/* Duration owns automatic window sizing. Keep it mounted while hidden. */}
      {hasDuration && <SidebarDialog id={`${id}-duration`} variant="settings" anchor={durationAnchor} hideHeader={mode === 'video'} fixedHeight={mode === 'video' ? 360 : undefined} open={expanded === 'duration'} title="Duration & windows" onClose={() => setExpanded(null)}><div className="space-y-3">
        {mode === 'audio' ? <AudioDurationControl/> : <><DurationSlider includeWindowSettings/><H3MultiWindowControls section="continuity"/></>}
      </div></SidebarDialog>}
    </>
  )
}
