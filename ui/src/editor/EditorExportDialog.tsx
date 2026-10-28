import { useEffect, useMemo, useState } from 'react'
import {
  CheckCircle2,
  Clock3,
  Download,
  ExternalLink,
  Film,
  Loader2,
  Sparkles,
  Volume2,
  VolumeX,
  X,
} from 'lucide-react'
import * as api from '../api/client'
import { useStore } from '../stores/useStore'
import type { EditorExportCapabilities, EditorExportSettings, EditorUpscaleMethod } from '../types'
import { editorExportDimensions, editorExportFps, editorProjectDuration, formatEditorTime } from './editorUtils'
import { useEditorStore } from './useEditorStore'
import { useEditorDialog } from './useEditorDialog'
import { flashvsrOptions } from '../lib/flashvsr'

const RESOLUTIONS: Array<{ value: EditorExportSettings['resolution']; label: string; detail: string }> = [
  { value: 'canvas', label: 'Project', detail: 'Match the edit canvas' },
  { value: '2160p', label: '4K', detail: '2160p delivery edge' },
  { value: '1080p', label: '1080p', detail: 'High-quality delivery' },
  { value: '720p', label: '720p', detail: 'Smaller, faster render' },
  { value: '480p', label: '480p', detail: 'Fast review copy' },
]

const QUALITY: Array<{ value: EditorExportSettings['quality']; label: string; detail: string }> = [
  { value: 'draft', label: 'Draft', detail: 'Fastest' },
  { value: 'balanced', label: 'Balanced', detail: 'Smaller file' },
  { value: 'high', label: 'High', detail: 'Best quality' },
]

const UPSCALE_OPTIONS: Array<{ value: EditorUpscaleMethod; label: string; scale: number }> = [
  { value: '', label: 'Off', scale: 1 },
  ...flashvsrOptions,
]

const DELIVERY_PRESETS: Array<{
  id: string
  label: string
  detail: string
  settings: Pick<EditorExportSettings, 'resolution' | 'codec' | 'quality'>
}> = [
  {
    id: 'review',
    label: 'Review copy',
    detail: '720p · H.264 · Draft',
    settings: { resolution: '720p', codec: 'h264', quality: 'draft' },
  },
  {
    id: 'balanced',
    label: 'Balanced delivery',
    detail: '1080p · H.264 · Balanced',
    settings: { resolution: '1080p', codec: 'h264', quality: 'balanced' },
  },
  {
    id: 'master',
    label: 'High-quality master',
    detail: '4K · H.265 · High',
    settings: { resolution: '2160p', codec: 'h265', quality: 'high' },
  },
]

type CapabilityState = 'loading' | 'ready' | 'error'

function encoderLabel(encoder: EditorExportSettings['encoder']): string {
  return ({
    auto: 'Automatic selection',
    software: 'Software encoder',
    nvidia: 'NVIDIA NVENC',
    intel: 'Intel Quick Sync',
    apple: 'Apple VideoToolbox',
  })[encoder]
}

function jobStatusLabel(status?: string): string {
  if (status === 'held') return 'Waiting in the export queue'
  if (status === 'queued') return 'Queued for rendering'
  if (status === 'running') return 'Rendering project'
  if (status === 'completed') return 'Export completed'
  if (status === 'cancelled') return 'Export cancelled'
  if (status === 'failed') return 'Export failed'
  return 'Preparing export'
}

function safeFilename(value: string): string {
  return value.replace(/[\\/:*?"<>|]+/g, ' ').replace(/\s+/g, ' ').slice(0, 120)
}

export function EditorExportDialog({ open, onClose, overlayItemIds }: { open: boolean; onClose: () => void; overlayItemIds?: string[] }) {
  const dialogRef = useEditorDialog(open, onClose)
  const [capabilities, setCapabilities] = useState<EditorExportCapabilities | null>(null)
  const [capabilityState, setCapabilityState] = useState<CapabilityState>('loading')
  const [capabilityError, setCapabilityError] = useState('')
  const [capabilityRefresh, setCapabilityRefresh] = useState(0)
  const [observedExportJob, setObservedExportJob] = useState(() => {
    const state = useEditorStore.getState()
    return state.exportJobId && state.project
      ? {jobId: state.exportJobId, projectId: state.project.id, workspace: state.project.workspace}
      : null
  })
  const project = useEditorStore(state => state.project)
  const transparent = Boolean(overlayItemIds?.length) || project?.export.format === 'prores4444'
  const exportOptions = overlayItemIds?.length ? { itemIds: overlayItemIds, format: 'prores4444' as const } : undefined
  const exportJobId = useEditorStore(state => state.exportJobId)
  const exportProgress = useEditorStore(state => state.exportProgress)
  const editorError = useEditorStore(state => state.error)
  const setExportSettings = useEditorStore(state => state.setExportSettings)
  const exportProject = useEditorStore(state => state.exportProject)
  const stopGeneration = useStore(state => state.stopGeneration)
  const activeExportIdentity = exportJobId && project
    ? {jobId: exportJobId, projectId: project.id, workspace: project.workspace}
    : observedExportJob
  const trackedJobId = activeExportIdentity
    && activeExportIdentity.projectId === project?.id
    && activeExportIdentity.workspace === project?.workspace
    ? activeExportIdentity.jobId
    : null
  const exportQueueJob = useStore(state => (
    trackedJobId ? state.jobs.find(job => job.id === trackedJobId) : undefined
  ))
  const exportQueueStatus = exportQueueJob?.status
  const exportOutcome = exportQueueJob && ['completed', 'failed', 'cancelled'].includes(exportQueueJob.status)
    ? {
        jobId: trackedJobId || exportQueueJob.id,
        status: exportQueueJob.status as 'completed' | 'failed' | 'cancelled',
        message: exportQueueJob.error || exportQueueJob.message || jobStatusLabel(exportQueueJob.status),
      }
    : null

  useEffect(() => {
    if (!open || capabilityState !== 'loading') return
    let active = true
    void api.fetchEditorExportCapabilities().then(result => {
      if (!result?.encoders || typeof result.recommended !== 'string') {
        throw new Error('The export capability response was incomplete.')
      }
        if (
          !result?.encoders
          || ['software', 'nvidia', 'intel', 'apple'].some(encoder => typeof result.encoders[encoder as keyof EditorExportCapabilities['encoders']] !== 'boolean')
          || !['software', 'nvidia', 'intel', 'apple'].includes(result.recommended)
        ) {
          throw new Error('The export capability response was incomplete.')
        }
        if (active) {
        setCapabilities(result)
        setCapabilityState('ready')
      }
    }).catch(error => {
      if (active) {
        setCapabilities(null)
        setCapabilityState('error')
        setCapabilityError(error instanceof Error ? error.message : 'Could not inspect local FFmpeg encoders.')
      }
    })
    return () => { active = false }
  }, [capabilityRefresh, capabilityState, open])

  const projectDimensions = { width: project?.canvas.width || 0, height: project?.canvas.height || 0 }
  const dimensions = useMemo(() => (
    project ? editorExportDimensions(project) : { width: 0, height: 0 }
  ), [project])
  const upscaleOption = UPSCALE_OPTIONS.find(option => (
    option.value === (project?.export.spatial_upsampling || '')
  )) || UPSCALE_OPTIONS[0]
  const deliveryDimensions = {
    width: dimensions.width * (transparent ? 1 : upscaleOption.scale),
    height: dimensions.height * (transparent ? 1 : upscaleOption.scale),
  }
  const outputFps = project ? editorExportFps(project) : 0
  const selectedItems = project?.tracks.flatMap(track => track.items).filter(item => overlayItemIds?.includes(item.id)) || []
  const duration = overlayItemIds?.length
    ? selectedItems.length ? Math.max(...selectedItems.map(item => item.start + item.duration)) - Math.min(...selectedItems.map(item => item.start)) : 0
    : editorProjectDuration(project)
  const lastExport = project?.exports?.[0]
  const selectedEncoder = transparent ? 'software' : project?.export.encoder || 'auto'
  const effectiveEncoder = selectedEncoder === 'auto'
    ? capabilities?.recommended && capabilities.recommended !== 'auto' ? capabilities.recommended : null
    : selectedEncoder
  const encoderAvailable = capabilityState === 'ready' && effectiveEncoder
    ? capabilities?.encoders[effectiveEncoder]
    : undefined
  const unavailableExplicitEncoder = capabilityState === 'ready'
    && !transparent
    && selectedEncoder !== 'auto'
    && encoderAvailable === false
  const alphaUnavailable = transparent && (capabilityState !== 'ready' || capabilities?.prores4444 !== true)
  const exportDisabled = duration <= 0 || unavailableExplicitEncoder || alphaUnavailable
  const finishingEnabled = !transparent && Boolean(project?.export.spatial_upsampling || (project?.export.film_grain_intensity || 0) > 0)
  const activeJobPhase = exportQueueJob?.phase || ''
  const activeJobMessage = exportQueueJob?.message || 'Preparing export'

  useEffect(() => useEditorStore.subscribe(state => {
    if (!state.exportJobId || !state.project) return
    setObservedExportJob({
      jobId: state.exportJobId,
      projectId: state.project.id,
      workspace: state.project.workspace,
    })
  }), [])

  if (!open || !project) return null

  return (
    <div
      ref={dialogRef}
      data-testid="editor-export-dialog"
      className="fixed inset-0 z-[150] flex h-[100dvh] items-center justify-center overflow-y-auto bg-black/65 px-2 py-2 backdrop-blur-sm sm:px-4"
      style={{
        paddingTop: 'max(0.5rem, env(safe-area-inset-top))',
        paddingBottom: 'max(0.5rem, env(safe-area-inset-bottom))',
      }}
      onMouseDown={event => {
        if (event.target === event.currentTarget) onClose()
      }}
    >
      <section role="dialog" aria-modal="true" aria-labelledby="editor-export-title" className="flex max-h-[calc(100dvh-1rem)] w-full max-w-[840px] flex-col overflow-hidden rounded-2xl border border-border bg-bg-secondary shadow-2xl">
        <header className="flex shrink-0 items-center justify-between border-b border-border px-4 py-3.5">
          <div className="flex items-center gap-2.5">
            <span className="grid h-8 w-8 place-items-center rounded-lg bg-cta/15 text-cta"><Film size={16} /></span>
            <div>
              <h2 id="editor-export-title" className="text-sm font-semibold text-text-primary">{overlayItemIds ? 'Export transparent overlay' : 'Export video'}</h2>
              <p className="text-[11px] text-text-muted">{project.name} · {formatEditorTime(duration)} timeline</p>
            </div>
          </div>
          <button type="button" onClick={onClose} className="rounded-lg p-2 text-text-muted hover:bg-bg-hover hover:text-text-primary" aria-label="Close export settings">
            <X size={16} />
          </button>
        </header>

        <div className="min-h-0 flex-1 overflow-y-auto p-4">
          <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_280px]">
            <div className="space-y-4">
              <label className="block space-y-1.5 text-[11px] font-medium uppercase tracking-wider text-text-muted">Output format
                <select aria-label="Export format" disabled={Boolean(overlayItemIds)} value={transparent ? 'prores4444' : 'mp4'}
                  onChange={event => setExportSettings(event.target.value === 'prores4444'
                    ? { format: 'prores4444', encoder: 'software', spatial_upsampling: '', film_grain_intensity: 0 }
                    : { format: 'mp4' })} className="w-full rounded-lg border border-border bg-bg-tertiary px-3 py-2 text-xs normal-case text-text-primary">
                  <option value="mp4">MP4 · Finished video</option><option value="prores4444">MOV · ProRes 4444 with alpha</option>
                </select>
              </label>
              {transparent && <p className="rounded-lg border border-accent-blue/25 bg-accent-blue/5 p-3 text-[11px] leading-relaxed text-text-secondary">
                {overlayItemIds ? 'Only the selected overlay is exported, starting at zero. Background footage and unrelated audio tracks are excluded.' : 'All visible tracks are included over a transparent canvas. Opaque footage covers the pixels beneath it; use a clip’s “Export transparent MOV” action for an isolated overlay.'}
                {' '}ProRes files are large and intended for editing software. A browser player may show black behind transparent pixels. AI finishing is unavailable for alpha exports.
              </p>}
              <div className="space-y-1.5">
                <label htmlFor="editor-export-filename" className="text-[11px] font-medium uppercase tracking-wider text-text-muted">File name</label>
                <div className="flex items-center rounded-lg border border-border bg-bg-tertiary focus-within:border-accent-blue/60">
                  <input
                    id="editor-export-filename"
                    value={project.export.filename}
                    placeholder={`${project.name} edit`}
                    onChange={event => setExportSettings({ filename: safeFilename(event.target.value) })}
                    className="min-w-0 flex-1 bg-transparent px-3 py-2 text-xs text-text-primary outline-none"
                  />
                  <span className="border-l border-border px-2.5 text-[10px] text-text-muted">{transparent ? '.mov' : '.mp4'}</span>
                </div>
              </div>

              <div className="space-y-1.5">
                <label className="text-[11px] font-medium uppercase tracking-wider text-text-muted">Resolution</label>
                <div className="grid grid-cols-2 gap-1.5 sm:grid-cols-3 xl:grid-cols-5">
                  {RESOLUTIONS.map(option => (
                    <button
                      key={option.value}
                      type="button"
                      onClick={() => setExportSettings({ resolution: option.value })}
                      className={`rounded-lg border px-2 py-2 text-left transition-colors ${project.export.resolution === option.value ? 'border-accent-blue/60 bg-accent-blue/10 text-text-primary' : 'border-border bg-bg-tertiary text-text-secondary hover:border-border-light hover:bg-bg-hover'}`}
                      title={option.detail}
                    >
                      <span className="block text-[11px] font-medium">{option.label}</span>
                      <span className="mt-0.5 block truncate text-[10px] text-text-muted">{option.detail}</span>
                    </button>
                  ))}
                </div>
              </div>

              <div hidden={transparent} className="space-y-1.5" data-testid="editor-delivery-presets">
                <div className="text-[11px] font-medium uppercase tracking-wider text-text-muted">Delivery preset</div>
                <div className="grid gap-1.5 sm:grid-cols-3">
                  {DELIVERY_PRESETS.map(preset => {
                    const selected = project.export.resolution === preset.settings.resolution
                      && project.export.codec === preset.settings.codec
                      && project.export.quality === preset.settings.quality
                    return (
                      <button
                        key={preset.id}
                        type="button"
                        aria-pressed={selected}
                        data-testid={`editor-export-preset-${preset.id}`}
                        onClick={() => setExportSettings(preset.settings)}
                        className={`rounded-lg border px-3 py-2 text-left transition-colors ${selected ? 'border-accent-blue/60 bg-accent-blue/10 text-text-primary' : 'border-border bg-bg-tertiary text-text-secondary hover:border-border-light hover:bg-bg-hover'}`}
                      >
                        <span className="block text-[11px] font-medium">{preset.label}</span>
                        <span className="mt-0.5 block text-[10px] text-text-muted">{preset.detail}</span>
                      </button>
                    )
                  })}
                </div>
                <p className="text-[10px] leading-relaxed text-text-muted">Presets change output size, codec, and quality only. Your edit canvas and finishing choices stay as set.</p>
              </div>

              <div className="grid gap-3 sm:grid-cols-3">
                <div className="space-y-1.5">
                  <label htmlFor="editor-export-frame-rate" className="text-[11px] font-medium uppercase tracking-wider text-text-muted">Frame rate</label>
                  <select
                    id="editor-export-frame-rate"
                    value={String(project.export.frame_rate)}
                    onChange={event => setExportSettings({
                      frame_rate: event.target.value === 'project'
                        ? 'project'
                        : Number(event.target.value) as 24 | 30 | 60,
                    })}
                    className="w-full rounded-lg border border-border bg-bg-tertiary px-2.5 py-2 text-xs text-text-primary outline-none focus:border-accent-blue/60"
                  >
                    <option value="project">Project ({project.canvas.fps.toFixed(project.canvas.fps % 1 ? 2 : 0)} fps)</option>
                    <option value="24">24 fps</option>
                    <option value="30">30 fps</option>
                    <option value="60">60 fps</option>
                  </select>
                </div>
                <div className="space-y-1.5">
                  <label htmlFor="editor-export-codec" className="text-[11px] font-medium uppercase tracking-wider text-text-muted">Codec</label>
                  <select
                    id="editor-export-codec"
                    value={transparent ? 'prores4444' : project.export.codec}
                    disabled={transparent}
                    onChange={event => setExportSettings({ codec: event.target.value as EditorExportSettings['codec'] })}
                    className="w-full rounded-lg border border-border bg-bg-tertiary px-2.5 py-2 text-xs text-text-primary outline-none focus:border-accent-blue/60"
                  >
                    <option value="h264">H.264 · Most compatible</option>
                    {transparent && <option value="prores4444">ProRes 4444 · Alpha</option>}
                    <option value="h265">H.265 · Smaller master</option>
                  </select>
                </div>
                <div className="space-y-1.5">
                  <label htmlFor="editor-export-encoder" className="text-[11px] font-medium uppercase tracking-wider text-text-muted">Encoder</label>
                  <select
                    id="editor-export-encoder"
                    data-testid="editor-export-encoder"
                    value={transparent ? 'software' : project.export.encoder || 'auto'}
                    disabled={transparent}
                    onChange={event => setExportSettings({ encoder: event.target.value as EditorExportSettings['encoder'] })}
                    className="w-full rounded-lg border border-border bg-bg-tertiary px-2.5 py-2 text-xs text-text-primary outline-none focus:border-accent-blue/60"
                  >
                    <option value="auto">Auto{capabilities ? ` · ${encoderLabel(capabilities.recommended)}` : ''}</option>
                    <option value="software">Software · compatible</option>
                    <option value="nvidia" disabled={capabilities ? !capabilities.encoders.nvidia : false}>NVIDIA NVENC</option>
                    <option value="intel" disabled={capabilities ? !capabilities.encoders.intel : false}>Intel Quick Sync</option>
                    <option value="apple" disabled={capabilities ? !capabilities.encoders.apple : false}>Apple VideoToolbox</option>
                  </select>
                  <div
                    role={capabilityState === 'error' ? 'alert' : 'status'}
                    aria-live="polite"
                    data-testid="editor-export-capability-status"
                    className={`rounded-lg border px-2.5 py-2 text-[11px] leading-relaxed ${capabilityState === 'error' ? 'border-amber-500/30 bg-amber-500/5 text-amber-200' : capabilityState === 'ready' && unavailableExplicitEncoder ? 'border-red-500/30 bg-red-500/5 text-red-200' : 'border-border bg-bg-secondary text-text-secondary'}`}
                  >
                    {capabilityState === 'loading' && 'Checking local FFmpeg encoders…'}
                    {capabilityState === 'error' && (
                  <span className="flex flex-wrap items-center justify-between gap-2">
                        <span>Encoder readiness is unavailable: {capabilityError} Auto will be resolved by the export worker.</span>
                        <button type="button" onClick={() => {
                          setCapabilities(null)
                          setCapabilityError('')
                          setCapabilityState('loading')
                          setCapabilityRefresh(value => value + 1)
                        }} className="rounded-md border border-border px-2 py-1 font-medium text-text-primary hover:bg-bg-hover">Check again</button>
                      </span>
                    )}
                    {transparent && capabilityState === 'ready' && (capabilities?.prores4444 ? 'Ready · Software ProRes 4444 with alpha' : 'ProRes 4444 is unavailable in this FFmpeg build.')}
                    {!transparent && capabilityState === 'ready' && effectiveEncoder && encoderAvailable === true && (
                      <>Ready · {encoderLabel(effectiveEncoder)}{selectedEncoder === 'auto' ? ` recommended by Auto` : ''}</>
                    )}
                    {!transparent && capabilityState === 'ready' && effectiveEncoder && encoderAvailable === false && (
                      <>{encoderLabel(effectiveEncoder)} is unavailable here. Choose Auto or Software to export.</>
                    )}
                  </div>
                </div>
              </div>

              <div className="space-y-1.5">
                <label hidden={transparent} className="text-[11px] font-medium uppercase tracking-wider text-text-muted">Quality</label>
                <div className="grid grid-cols-3 gap-1.5">
                  {!transparent && QUALITY.map(option => (
                    <button
                      key={option.value}
                      type="button"
                      onClick={() => setExportSettings({ quality: option.value })}
                      className={`rounded-lg border px-2.5 py-2 text-left transition-colors ${project.export.quality === option.value ? 'border-accent-blue/60 bg-accent-blue/10 text-text-primary' : 'border-border bg-bg-tertiary text-text-secondary hover:border-border-light hover:bg-bg-hover'}`}
                    >
                      <span className="block text-[11px] font-medium">{option.label}</span>
                      <span className="text-[10px] text-text-muted">{option.detail}</span>
                    </button>
                  ))}
                </div>
              </div>

              <button
                type="button"
                onClick={() => setExportSettings({ include_audio: !project.export.include_audio })}
                aria-pressed={project.export.include_audio}
                className={`flex w-full items-center justify-between rounded-lg border px-3 py-2.5 text-left ${project.export.include_audio ? 'border-accent-blue/35 bg-accent-blue/5' : 'border-border bg-bg-tertiary'}`}
              >
                <span className="flex items-center gap-2 text-[11px] text-text-secondary">
                  {project.export.include_audio ? <Volume2 size={13} className="text-accent-blue" /> : <VolumeX size={13} className="text-text-muted" />}
                  {overlayItemIds ? 'Include selected clip audio' : 'Include timeline audio'}
                </span>
                <span className={`h-4 w-7 rounded-full p-0.5 transition-colors ${project.export.include_audio ? 'bg-accent-blue' : 'bg-bg-active'}`}>
                  <span className={`block h-3 w-3 rounded-full bg-white transition-transform ${project.export.include_audio ? 'translate-x-3' : ''}`} />
                </span>
              </button>

              <div hidden={transparent} className="space-y-3 rounded-xl border border-border bg-bg-tertiary p-3" data-testid="editor-export-finishing">
                <div className="flex items-start gap-2">
                  <span className="mt-0.5 grid h-6 w-6 shrink-0 place-items-center rounded-md bg-cta/10 text-cta">
                    <Sparkles size={12} />
                  </span>
                  <div>
                    <div className="text-xs font-medium text-text-primary">Source finishing</div>
                    <p className="mt-0.5 text-[11px] leading-relaxed text-text-muted">These optional passes run after the timeline render. Upscale runs before grain; your timeline clips stay unchanged.</p>
                  </div>
                </div>

                <div className="space-y-1.5">
                  <label htmlFor="editor-export-upscale" className="text-[11px] font-medium uppercase tracking-wider text-text-muted">AI upscale</label>
                  <select
                    id="editor-export-upscale"
                    value={project.export.spatial_upsampling || ''}
                    onChange={event => setExportSettings({ spatial_upsampling: event.target.value as EditorUpscaleMethod })}
                    className="w-full rounded-lg border border-border bg-bg-secondary px-2.5 py-2 text-xs text-text-primary outline-none focus:border-accent-blue/60"
                  >
                    {UPSCALE_OPTIONS.map(option => (
                      <option key={option.value || 'off'} value={option.value}>{option.label}</option>
                    ))}
                  </select>
                  {upscaleOption.scale > 1 && (
                    <p className="text-[11px] leading-relaxed text-amber-200">FlashVSR adds a {upscaleOption.scale}× finishing pass. Its model may need to download on first use, and this can add substantial render time.</p>
                  )}
                </div>

                <button
                  type="button"
                  onClick={() => setExportSettings({
                    film_grain_intensity: (project.export.film_grain_intensity || 0) > 0 ? 0 : 0.15,
                  })}
                  className={`flex w-full items-center justify-between rounded-lg border px-3 py-2 text-left ${(project.export.film_grain_intensity || 0) > 0 ? 'border-accent-blue/35 bg-accent-blue/5' : 'border-border bg-bg-secondary'}`}
                >
                  <span className="text-[11px] text-text-secondary">Film grain</span>
                  <span className={`h-4 w-7 rounded-full p-0.5 transition-colors ${(project.export.film_grain_intensity || 0) > 0 ? 'bg-accent-blue' : 'bg-bg-active'}`}>
                    <span className={`block h-3 w-3 rounded-full bg-white transition-transform ${(project.export.film_grain_intensity || 0) > 0 ? 'translate-x-3' : ''}`} />
                  </span>
                </button>

                {(project.export.film_grain_intensity || 0) > 0 && (
                  <div className="grid gap-3 sm:grid-cols-2">
                    <label className="space-y-1.5 text-[11px] text-text-muted">
                      <span className="flex justify-between"><span>Intensity</span><span>{(project.export.film_grain_intensity || 0).toFixed(2)}</span></span>
                      <input
                        type="range"
                        min="0.01"
                        max="1"
                        step="0.01"
                        value={project.export.film_grain_intensity || 0.15}
                        onChange={event => setExportSettings({ film_grain_intensity: Number(event.target.value) })}
                        className="w-full accent-accent-blue"
                      />
                    </label>
                    <label className="space-y-1.5 text-[11px] text-text-muted">
                      <span className="flex justify-between"><span>Color</span><span>{(project.export.film_grain_saturation ?? 0.5).toFixed(2)}</span></span>
                      <input
                        type="range"
                        min="0"
                        max="1"
                        step="0.01"
                        value={project.export.film_grain_saturation ?? 0.5}
                        onChange={event => setExportSettings({ film_grain_saturation: Number(event.target.value) })}
                        className="w-full accent-accent-blue"
                      />
                    </label>
                  </div>
                )}
              </div>
            </div>

            <aside className="space-y-3">
              <div className="rounded-xl border border-border bg-bg-tertiary p-3" data-testid="editor-export-summary">
                <div className="text-[11px] font-medium uppercase tracking-wider text-text-muted">Delivery summary</div>
                <dl className="mt-2.5 space-y-2 text-[11px]">
                  <div className="flex justify-between gap-3"><dt className="text-text-muted">Project canvas</dt><dd className="font-mono text-text-secondary">{projectDimensions.width}×{projectDimensions.height}</dd></div>
                  <div className="flex justify-between gap-3"><dt className="text-text-muted">Render frame</dt><dd className="font-mono text-text-secondary">{dimensions.width}×{dimensions.height}</dd></div>
                  <div className="flex justify-between gap-3"><dt className="text-text-muted">Delivered frame</dt><dd className="font-mono text-text-primary">{deliveryDimensions.width}×{deliveryDimensions.height}</dd></div>
                  <div className="flex justify-between gap-3"><dt className="text-text-muted">Rate</dt><dd className="font-mono text-text-secondary">{outputFps.toFixed(outputFps % 1 ? 2 : 0)} fps</dd></div>
                  <div className="flex justify-between gap-3"><dt className="text-text-muted">Duration</dt><dd className="font-mono text-text-secondary">{formatEditorTime(duration)}</dd></div>
                  <div className="flex justify-between gap-3"><dt className="text-text-muted">Format</dt><dd className="text-right text-text-secondary">{transparent ? 'MOV · ProRes 4444 · Alpha' : `MP4 · ${project.export.codec.toUpperCase()}`}</dd></div>
                </dl>
                <div className="mt-3 rounded-lg border border-border bg-bg-secondary px-2.5 py-2 text-[11px] leading-relaxed text-text-secondary" data-testid="editor-export-stage-summary">
                  {finishingEnabled ? (
                    <>
                      <div className="font-medium text-text-primary">Finishing stages</div>
                      <div className="mt-1">{[
                        project.export.spatial_upsampling ? `AI upscale ${upscaleOption.scale}×` : null,
                        (project.export.film_grain_intensity || 0) > 0 ? 'Film grain' : null,
                      ].filter(Boolean).join(' → ')}</div>
                    </>
                  ) : 'No finishing stages selected. The export uses the render frame above.'}
                </div>
              </div>

              {lastExport && (
                <div className="rounded-xl border border-green-500/25 bg-green-500/5 p-3">
                  <div className="flex items-center gap-1.5 text-[11px] font-medium text-indicator-success"><CheckCircle2 size={13} /> Latest export</div>
                  <div className="mt-1.5 truncate text-[11px] text-text-secondary" title={lastExport.filename}>{lastExport.filename}</div>
                  <div className="mt-2 grid grid-cols-2 gap-1.5">
                    <a href={api.getFileUrl(lastExport.filename, lastExport.workspace)} target="_blank" rel="noreferrer" className="flex items-center justify-center gap-1 rounded-md border border-border bg-bg-secondary px-2 py-2 text-[11px] text-text-secondary hover:border-accent-blue/40 hover:text-accent-blue">
                      <ExternalLink size={12} /> Open
                    </a>
                    <a href={api.getFileUrl(lastExport.filename, lastExport.workspace)} download={lastExport.filename} className="flex items-center justify-center gap-1 rounded-md border border-border bg-bg-secondary px-2 py-2 text-[11px] text-text-secondary hover:border-accent-blue/40 hover:text-accent-blue">
                      <Download size={12} /> Save file
                    </a>
                  </div>
                </div>
              )}

              {(project.exports || []).length > 1 && (
                <div className="rounded-xl border border-border bg-bg-tertiary p-3">
                  <div className="text-[11px] font-medium uppercase tracking-wider text-text-muted">Recent exports</div>
                  <div className="mt-2 space-y-1.5">
                    {(project.exports || []).slice(1, 5).map(record => (
                      <a key={record.id} href={api.getFileUrl(record.filename, record.workspace)} target="_blank" rel="noreferrer" className="flex items-center justify-between gap-2 rounded-md px-1.5 py-1.5 text-[11px] text-text-muted hover:bg-bg-hover hover:text-text-secondary">
                        <span className="truncate">{record.filename}</span>
                        <ExternalLink size={9} className="shrink-0" />
                      </a>
                    ))}
                  </div>
                </div>
              )}
            </aside>
          </div>
        </div>

        <footer className="shrink-0 border-t border-border bg-bg-tertiary/50 px-4 py-3">
          <div className="space-y-3">
            <div
              role={exportOutcome?.status === 'failed' || (!exportJobId && editorError) ? 'alert' : 'status'}
              aria-live="polite"
              data-testid="editor-export-job-status"
              className={`rounded-lg border px-3 py-2 text-[11px] leading-relaxed ${exportOutcome?.status === 'failed' || (!exportJobId && editorError) ? 'border-red-500/30 bg-red-500/5 text-red-200' : exportOutcome?.status === 'cancelled' ? 'border-amber-500/25 bg-amber-500/5 text-amber-100' : 'border-border bg-bg-secondary text-text-secondary'}`}
            >
              {exportJobId ? (
                <div className="flex flex-wrap items-center justify-between gap-x-4 gap-y-1">
                  <span className="flex items-center gap-2">
                    {exportQueueStatus === 'held' ? <Clock3 size={14} className="shrink-0 text-text-muted" /> : <Loader2 size={14} className="shrink-0 animate-spin text-accent-blue" />}
                    <span>
                      <strong className="font-semibold text-text-primary">{jobStatusLabel(exportQueueStatus)}</strong>
                      <span className="ml-2">{activeJobPhase || activeJobMessage}</span>
                      {activeJobPhase && activeJobMessage !== activeJobPhase && <span className="ml-2 text-text-muted">· {activeJobMessage}</span>}
                    </span>
                  </span>
                  <span className="font-mono tabular-nums text-text-primary">{Math.round(exportProgress * 100)}%</span>
                </div>
              ) : exportOutcome ? (
                <span><strong className="font-semibold text-text-primary">{jobStatusLabel(exportOutcome.status)}.</strong> {exportOutcome.status === 'failed' ? exportOutcome.message : exportOutcome.status === 'cancelled' ? 'The render stopped and temporary output was discarded.' : exportOutcome.message}</span>
              ) : editorError ? (
                <span>{editorError}</span>
              ) : (
                <span>The project stays editable while the export renders to a new file.</span>
              )}
            </div>

            {exportJobId && (
              <div className="h-1.5 overflow-hidden rounded-full bg-bg-active" aria-label="Export progress">
                <div className="h-full rounded-full bg-accent-blue transition-all" style={{ width: `${Math.max(1, exportProgress * 100)}%` }} />
              </div>
            )}

            <div className="flex flex-col-reverse gap-2 sm:flex-row sm:items-center sm:justify-between">
              {exportJobId ? (
                <button
                  type="button"
                  data-testid="editor-export-cancel"
                  onClick={() => stopGeneration(exportJobId)}
                  disabled={exportQueueStatus === 'cancelled'}
                  className="min-h-10 rounded-lg border border-red-500/30 px-4 py-2 text-sm font-medium text-red-200 hover:bg-red-500/10 disabled:cursor-not-allowed disabled:opacity-50"
                >
                  {exportQueueStatus === 'cancelled' ? 'Cancelling…' : 'Cancel export'}
                </button>
              ) : (
                <p className="text-[11px] leading-relaxed text-text-muted">Export creates a new {transparent ? 'MOV with alpha' : 'MP4'} and does not alter the timeline or source media.</p>
              )}
              <div className="flex w-full flex-col gap-2 sm:w-auto sm:flex-row sm:justify-end">
                {!exportJobId && (
                  <button
                    type="button"
                    data-testid="editor-export-queue"
                    onClick={() => void exportProject('queue', exportOptions)}
                    disabled={exportDisabled}
                    className="min-h-10 rounded-lg border border-border bg-bg-secondary px-4 py-2 text-sm font-medium text-text-secondary hover:border-accent-blue/40 hover:text-accent-blue disabled:cursor-not-allowed disabled:opacity-40"
                    title="Hold this export in Maestro's universal queue"
                  >
                    Queue export
                  </button>
                )}
                <button
                  type="button"
                  data-testid="editor-export-submit"
                  onClick={() => void exportProject('now', exportOptions)}
                  disabled={Boolean(exportJobId) || exportDisabled}
                  className="min-h-10 rounded-lg bg-cta px-5 py-2 text-sm font-semibold text-white shadow-accent-glow hover:brightness-105 disabled:cursor-not-allowed disabled:opacity-40"
                >
                  <Download size={15} className="mr-1.5 inline" />Export video
                </button>
              </div>
            </div>
            {unavailableExplicitEncoder && (
              <p role="alert" data-testid="editor-export-unsupported-encoder" className="text-[11px] text-red-200">The selected encoder is unavailable. Choose Auto or Software before starting this export.</p>
            )}
          </div>
        </footer>
      </section>
    </div>
  )
}
