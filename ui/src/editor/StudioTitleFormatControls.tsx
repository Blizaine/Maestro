import { useId, useState, type ReactNode } from 'react'
import { Clock, Monitor, RectangleHorizontal } from 'lucide-react'
import { SidebarDialog } from '../components/Sidebar/SidebarPanels'
import { SidebarMenu } from '../components/Sidebar/SidebarMenu'
import { formatDuration } from '../lib/durationPlanning'
import {
  TITLE_SEQUENCE_DURATIONS,
  titleFrames,
  titleSeconds,
  titleDurationChoices,
  titleFps,
} from './editorAnimatedTitles'
import type { AnimatedTitleSettings } from './editorAnimatedTitles'

const RESOLUTIONS = ['480p', '540p', '720p', '768p', '1080p'] as const
const ASPECT_RATIOS = ['21:9', '16:9', '9:16', '1:1', '4:3', '3:4'] as const

function selectedFrames(settings: AnimatedTitleSettings) {
  try { return titleFrames(titleSeconds(settings), settings) }
  catch { return TITLE_SEQUENCE_DURATIONS[0].frames }
}

function exactDurationText(frames: number, fps: number) {
  return `${(frames / fps).toFixed(3)}s`
}

/** Studio-only format controls. Values belong to the title draft, not useStore's ordinary generation settings. */
export function StudioTitleFormatControls({ settings, patch, onDurationChange, busy, children }: {
  settings: AnimatedTitleSettings
  patch: (value: Partial<AnimatedTitleSettings>) => void
  onDurationChange: (frames: number) => void
  busy: boolean
  children?: ReactNode
}) {
  const id = useId()
  const [expanded, setExpanded] = useState<'resolution' | 'aspect' | 'duration' | null>(null)
  const [resolutionAnchor, setResolutionAnchor] = useState<HTMLButtonElement | null>(null)
  const [aspectAnchor, setAspectAnchor] = useState<HTMLButtonElement | null>(null)
  const [durationAnchor, setDurationAnchor] = useState<HTMLButtonElement | null>(null)
  const frames = selectedFrames(settings)
  const choices = titleDurationChoices(settings)
  const fps = titleFps(settings)
  const durationIndex = Math.max(0, choices.findIndex(option => option.frames === frames))
  const durationLabel = formatDuration(frames / fps, true)
  const chip = 'studio-setting-chip'
  const selectedChip = 'border-accent-blue bg-accent-blue/10 text-text-primary'
  const idleChip = 'border-border bg-bg-tertiary text-text-secondary hover:border-border-light'

  const toggle = (panel: 'resolution' | 'aspect' | 'duration') => setExpanded(value => value === panel ? null : panel)
  const close = () => setExpanded(null)

  return <>
    <div data-testid="titles-output-format-controls" role="group" aria-label="Title output format"
      className="mb-2 flex min-w-0 flex-wrap items-center justify-end gap-1">
      <button ref={setResolutionAnchor} type="button" aria-label={`Resolution: ${settings.resolution}`}
        aria-expanded={expanded === 'resolution'} aria-haspopup="menu" aria-controls={`${id}-resolution`}
        title="Resolution" disabled={busy} onClick={() => toggle('resolution')}
        className={`${chip} ${expanded === 'resolution' ? selectedChip : idleChip} disabled:opacity-40`}>
        <Monitor size={13} className="studio-setting-icon" /><span>{settings.resolution}</span>
      </button>
      <button ref={setAspectAnchor} type="button" aria-label={`Aspect ratio: ${settings.aspect}`}
        aria-expanded={expanded === 'aspect'} aria-haspopup="menu" aria-controls={`${id}-aspect`}
        title="Choose title aspect ratio" disabled={busy} onClick={() => toggle('aspect')}
        className={`${chip} ${expanded === 'aspect' ? selectedChip : idleChip} disabled:opacity-40`}>
        <RectangleHorizontal size={13} className="studio-setting-icon" /><span>{settings.aspect}</span>
      </button>
      <button ref={setDurationAnchor} type="button" aria-label={`Duration: ${durationLabel}`}
        aria-expanded={expanded === 'duration'} aria-haspopup="dialog" aria-controls={`${id}-duration`}
        title="Title duration" disabled={busy} onClick={() => toggle('duration')}
        className={`${chip} ${expanded === 'duration' ? selectedChip : idleChip} disabled:opacity-40`}>
        <Clock size={13} className="studio-setting-icon" /><span className="tabular-nums">{durationLabel}</span>
      </button>
      {children}
    </div>

    <SidebarMenu id={`${id}-resolution`} anchor={resolutionAnchor} open={expanded === 'resolution'}
      label="Resolution" width="content" align="start" onClose={close}>
      <div className="flex flex-col gap-0.5">
        {RESOLUTIONS.map(resolution => <button key={resolution} type="button" role="menuitemradio" aria-label={resolution}
          aria-checked={settings.resolution === resolution} disabled={busy}
          onClick={() => { patch({ resolution }); close() }}
          className={`flex min-h-9 items-center justify-between gap-3 rounded-md px-2.5 text-left text-xs transition-colors ${settings.resolution === resolution ? 'bg-bg-active text-text-primary' : 'text-text-secondary hover:text-text-primary'} disabled:opacity-40`}>
          <span className="flex items-center gap-2">{resolution}
            {resolution === '768p' && (!settings.model || settings.model.nativeH3) && <span className="text-[9px] text-text-muted">Native H3 tier</span>}
            {resolution === '1080p' && <span className="text-[9px] text-indicator-warning">Experimental</span>}
          </span>
          <span aria-hidden="true" className={settings.resolution === resolution ? '' : 'invisible'}>✓</span>
        </button>)}
      </div>
    </SidebarMenu>

    <SidebarMenu id={`${id}-aspect`} anchor={aspectAnchor} open={expanded === 'aspect'}
      label="Aspect ratio" width="content" align="start" onClose={close}>
      <div className="flex flex-col gap-0.5">
        {ASPECT_RATIOS.map((aspect, index) => <button key={aspect} type="button" role="menuitemradio"
          aria-label={aspect} aria-checked={settings.aspect === aspect} disabled={busy}
          onClick={() => { patch({ aspect }); close() }}
          className={`flex min-h-9 items-center gap-2.5 rounded-md px-2.5 text-xs transition-colors ${settings.aspect === aspect ? 'bg-bg-active text-text-primary' : 'text-text-secondary hover:text-text-primary'} disabled:opacity-40`}>
          <span aria-hidden="true" className="w-4 text-center text-sm leading-none">{['━', '▬', '▮', '◼', '▭', '▯'][index]}</span>
          <span>{aspect}</span>
          <span aria-hidden="true" className={`ml-auto ${settings.aspect === aspect ? '' : 'invisible'}`}>✓</span>
        </button>)}
      </div>
    </SidebarMenu>

    <SidebarDialog id={`${id}-duration`} variant="settings" anchor={durationAnchor}
      open={expanded === 'duration'} title="Title duration" onClose={close} fixedHeight={360}>
      <div className="space-y-4">
        <label className="block space-y-3 text-xs text-text-secondary">
          <span className="flex items-start justify-between gap-3">
            <span>Title clip duration</span>
            <span className="shrink-0 text-right font-medium tabular-nums text-text-primary">
              {exactDurationText(frames, fps)} · {frames} frames
            </span>
          </span>
          <input aria-label="Title clip duration" type="range" min={0}
            max={choices.length - 1} step={1} value={durationIndex} disabled={busy}
            aria-valuetext={`${exactDurationText(frames, fps)} · ${frames} frames at ${fps} fps`}
            onChange={event => onDurationChange(choices[Number(event.target.value)].frames)}
            className="w-full accent-[var(--color-accent-blue)]" />
          <span className="flex justify-between text-[10px] tabular-nums text-text-muted">
            <span>{exactDurationText(choices[0].frames, fps)}</span>
            <span>{exactDurationText(choices.at(-1)!.frames, fps)}</span>
          </span>
        </label>
        <label className="block space-y-1.5 text-xs text-text-secondary">Native duration preset
          <select aria-label="Native title duration preset" value={frames} disabled={busy}
            onChange={event => onDurationChange(Number(event.target.value))}
            className="w-full rounded-lg border border-border bg-bg-tertiary px-3 py-2 text-xs text-text-primary outline-none focus:border-accent-blue/60">
            {choices.map(option => <option key={option.frames} value={option.frames}>
              {option.label}
            </option>)}
          </select>
        </label>
        <p className="text-[11px] leading-relaxed text-text-muted">Native clip duration at {fps} fps.</p>
      </div>
    </SidebarDialog>
  </>
}
