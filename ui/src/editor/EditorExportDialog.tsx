import { useEffect, useMemo, useState } from 'react'
import {
  CheckCircle2,
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
  { value: 'flashvsr2', label: 'FlashVSR 2×', scale: 2 },
  { value: 'flashvsr3', label: 'FlashVSR 3×', scale: 3 },
  { value: 'flashvsr4', label: 'FlashVSR 4×', scale: 4 },
  { value: 'flashvsr2pass2', label: 'FlashVSR two-pass 2×', scale: 2 },
  { value: 'flashvsr2pass4', label: 'FlashVSR two-pass 4×', scale: 4 },
]

function safeFilename(value: string): string {
  return value.replace(/[\\/:*?"<>|]+/g, ' ').replace(/\s+/g, ' ').slice(0, 120)
}

export function EditorExportDialog({ open, onClose }: { open: boolean; onClose: () => void }) {
  const [capabilities, setCapabilities] = useState<EditorExportCapabilities | null>(null)
  const project = useEditorStore(state => state.project)
  const exportJobId = useEditorStore(state => state.exportJobId)
  const exportProgress = useEditorStore(state => state.exportProgress)
  const setExportSettings = useEditorStore(state => state.setExportSettings)
  const exportProject = useEditorStore(state => state.exportProject)
  const stopGeneration = useStore(state => state.stopGeneration)
  const exportQueueStatus = useStore(state => (
    exportJobId ? state.jobs.find(job => job.id === exportJobId)?.status : undefined
  ))

  useEffect(() => {
    if (!open) return
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [onClose, open])

  useEffect(() => {
    if (!open) return
    let active = true
    void api.fetchEditorExportCapabilities().then(result => {
      if (active) setCapabilities(result)
    }).catch(() => {
      if (active) setCapabilities(null)
    })
    return () => { active = false }
  }, [open])

  const dimensions = useMemo(() => (
    project ? editorExportDimensions(project) : { width: 0, height: 0 }
  ), [project])
  const upscaleOption = UPSCALE_OPTIONS.find(option => (
    option.value === (project?.export.spatial_upsampling || '')
  )) || UPSCALE_OPTIONS[0]
  const deliveryDimensions = {
    width: dimensions.width * upscaleOption.scale,
    height: dimensions.height * upscaleOption.scale,
  }
  const outputFps = project ? editorExportFps(project) : 0
  const duration = editorProjectDuration(project)
  const lastExport = project?.exports?.[0]

  if (!open || !project) return null

  return (
    <div
      className="fixed inset-0 z-[150] flex h-[100dvh] items-center justify-center overflow-hidden bg-black/65 px-3 backdrop-blur-sm"
      style={{
        paddingTop: 'max(0.75rem, env(safe-area-inset-top))',
        paddingBottom: 'max(0.75rem, env(safe-area-inset-bottom))',
      }}
      onMouseDown={event => {
      if (event.target === event.currentTarget) onClose()
    }}>
      <section role="dialog" aria-modal="true" aria-label="Export Editor project" className="flex max-h-full w-full max-w-[720px] flex-col overflow-hidden rounded-2xl border border-border bg-bg-secondary shadow-2xl">
        <header className="flex shrink-0 items-center justify-between border-b border-border px-4 py-3.5">
          <div className="flex items-center gap-2.5">
            <span className="grid h-8 w-8 place-items-center rounded-lg bg-cta/15 text-cta"><Film size={16} /></span>
            <div>
              <h2 className="text-sm font-semibold text-text-primary">Export video</h2>
              <p className="text-[9px] text-text-muted">{project.name} · {formatEditorTime(duration)} timeline</p>
            </div>
          </div>
          <button type="button" onClick={onClose} className="rounded-lg p-2 text-text-muted hover:bg-bg-hover hover:text-text-primary" aria-label="Close export settings">
            <X size={16} />
          </button>
        </header>

        <div className="min-h-0 flex-1 overflow-y-auto p-4">
          <div className="grid gap-4 md:grid-cols-[1fr_240px]">
            <div className="space-y-4">
              <div className="space-y-1.5">
                <label className="text-[9px] font-medium uppercase tracking-wider text-text-muted">File name</label>
                <div className="flex items-center rounded-lg border border-border bg-bg-tertiary focus-within:border-accent-blue/60">
                  <input
                    value={project.export.filename}
                    placeholder={`${project.name} edit`}
                    onChange={event => setExportSettings({ filename: safeFilename(event.target.value) })}
                    className="min-w-0 flex-1 bg-transparent px-3 py-2 text-xs text-text-primary outline-none"
                  />
                  <span className="border-l border-border px-2.5 text-[10px] text-text-muted">.mp4</span>
                </div>
              </div>

              <div className="space-y-1.5">
                <label className="text-[9px] font-medium uppercase tracking-wider text-text-muted">Resolution</label>
                <div className="grid grid-cols-2 gap-1.5 sm:grid-cols-5">
                  {RESOLUTIONS.map(option => (
                    <button
                      key={option.value}
                      type="button"
                      onClick={() => setExportSettings({ resolution: option.value })}
                      className={`rounded-lg border px-2 py-2 text-left transition-colors ${project.export.resolution === option.value ? 'border-accent-blue/60 bg-accent-blue/10 text-text-primary' : 'border-border bg-bg-tertiary text-text-secondary hover:border-border-light hover:bg-bg-hover'}`}
                      title={option.detail}
                    >
                      <span className="block text-[10px] font-medium">{option.label}</span>
                      <span className="mt-0.5 block truncate text-[8px] text-text-muted">{option.detail}</span>
                    </button>
                  ))}
                </div>
              </div>

              <div className="grid gap-3 sm:grid-cols-3">
                <div className="space-y-1.5">
                  <label className="text-[9px] font-medium uppercase tracking-wider text-text-muted">Frame rate</label>
                  <select
                    value={String(project.export.frame_rate)}
                    onChange={event => setExportSettings({
                      frame_rate: event.target.value === 'project'
                        ? 'project'
                        : Number(event.target.value) as 24 | 30 | 60,
                    })}
                    className="w-full rounded-lg border border-border bg-bg-tertiary px-2.5 py-2 text-[10px] text-text-primary outline-none focus:border-accent-blue/60"
                  >
                    <option value="project">Project ({project.canvas.fps.toFixed(project.canvas.fps % 1 ? 2 : 0)} fps)</option>
                    <option value="24">24 fps</option>
                    <option value="30">30 fps</option>
                    <option value="60">60 fps</option>
                  </select>
                </div>
                <div className="space-y-1.5">
                  <label className="text-[9px] font-medium uppercase tracking-wider text-text-muted">Codec</label>
                  <select
                    value={project.export.codec}
                    onChange={event => setExportSettings({ codec: event.target.value as EditorExportSettings['codec'] })}
                    className="w-full rounded-lg border border-border bg-bg-tertiary px-2.5 py-2 text-[10px] text-text-primary outline-none focus:border-accent-blue/60"
                  >
                    <option value="h264">H.264 · Most compatible</option>
                    <option value="h265">H.265 · Smaller master</option>
                  </select>
                </div>
                <div className="space-y-1.5">
                  <label className="text-[9px] font-medium uppercase tracking-wider text-text-muted">Encoder</label>
                  <select
                    value={project.export.encoder || 'auto'}
                    onChange={event => setExportSettings({ encoder: event.target.value as EditorExportSettings['encoder'] })}
                    className="w-full rounded-lg border border-border bg-bg-tertiary px-2.5 py-2 text-[10px] text-text-primary outline-none focus:border-accent-blue/60"
                  >
                    <option value="auto">Auto{capabilities ? ` · ${capabilities.recommended}` : ''}</option>
                    <option value="software">Software · compatible</option>
                    <option value="nvidia" disabled={capabilities ? !capabilities.encoders.nvidia : false}>NVIDIA NVENC</option>
                    <option value="intel" disabled={capabilities ? !capabilities.encoders.intel : false}>Intel Quick Sync</option>
                    <option value="apple" disabled={capabilities ? !capabilities.encoders.apple : false}>Apple VideoToolbox</option>
                  </select>
                </div>
              </div>

              <div className="space-y-1.5">
                <label className="text-[9px] font-medium uppercase tracking-wider text-text-muted">Quality</label>
                <div className="grid grid-cols-3 gap-1.5">
                  {QUALITY.map(option => (
                    <button
                      key={option.value}
                      type="button"
                      onClick={() => setExportSettings({ quality: option.value })}
                      className={`rounded-lg border px-2.5 py-2 text-left transition-colors ${project.export.quality === option.value ? 'border-accent-blue/60 bg-accent-blue/10 text-text-primary' : 'border-border bg-bg-tertiary text-text-secondary hover:border-border-light hover:bg-bg-hover'}`}
                    >
                      <span className="block text-[10px] font-medium">{option.label}</span>
                      <span className="text-[8px] text-text-muted">{option.detail}</span>
                    </button>
                  ))}
                </div>
              </div>

              <button
                type="button"
                onClick={() => setExportSettings({ include_audio: !project.export.include_audio })}
                className={`flex w-full items-center justify-between rounded-lg border px-3 py-2.5 text-left ${project.export.include_audio ? 'border-accent-blue/35 bg-accent-blue/5' : 'border-border bg-bg-tertiary'}`}
              >
                <span className="flex items-center gap-2 text-[10px] text-text-secondary">
                  {project.export.include_audio ? <Volume2 size={13} className="text-accent-blue" /> : <VolumeX size={13} className="text-text-muted" />}
                  Include timeline audio
                </span>
                <span className={`h-4 w-7 rounded-full p-0.5 transition-colors ${project.export.include_audio ? 'bg-accent-blue' : 'bg-bg-active'}`}>
                  <span className={`block h-3 w-3 rounded-full bg-white transition-transform ${project.export.include_audio ? 'translate-x-3' : ''}`} />
                </span>
              </button>

              <div className="space-y-3 rounded-xl border border-border bg-bg-tertiary p-3">
                <div className="flex items-start gap-2">
                  <span className="mt-0.5 grid h-6 w-6 shrink-0 place-items-center rounded-md bg-cta/10 text-cta">
                    <Sparkles size={12} />
                  </span>
                  <div>
                    <div className="text-[10px] font-medium text-text-primary">Finishing</div>
                    <p className="mt-0.5 text-[8px] leading-relaxed text-text-muted">Optional passes run after the timeline is rendered. Upscale runs before grain.</p>
                  </div>
                </div>

                <div className="space-y-1.5">
                  <label className="text-[9px] font-medium uppercase tracking-wider text-text-muted">AI upscale</label>
                  <select
                    value={project.export.spatial_upsampling || ''}
                    onChange={event => setExportSettings({ spatial_upsampling: event.target.value as EditorUpscaleMethod })}
                    className="w-full rounded-lg border border-border bg-bg-secondary px-2.5 py-2 text-[10px] text-text-primary outline-none focus:border-accent-blue/60"
                  >
                    {UPSCALE_OPTIONS.map(option => (
                      <option key={option.value || 'off'} value={option.value}>{option.label}</option>
                    ))}
                  </select>
                  {upscaleOption.scale > 1 && (
                    <p className="text-[8px] leading-relaxed text-text-muted">FlashVSR may download its model on first use and substantially increases render time.</p>
                  )}
                </div>

                <button
                  type="button"
                  onClick={() => setExportSettings({
                    film_grain_intensity: (project.export.film_grain_intensity || 0) > 0 ? 0 : 0.15,
                  })}
                  className={`flex w-full items-center justify-between rounded-lg border px-3 py-2 text-left ${(project.export.film_grain_intensity || 0) > 0 ? 'border-accent-blue/35 bg-accent-blue/5' : 'border-border bg-bg-secondary'}`}
                >
                  <span className="text-[10px] text-text-secondary">Film grain</span>
                  <span className={`h-4 w-7 rounded-full p-0.5 transition-colors ${(project.export.film_grain_intensity || 0) > 0 ? 'bg-accent-blue' : 'bg-bg-active'}`}>
                    <span className={`block h-3 w-3 rounded-full bg-white transition-transform ${(project.export.film_grain_intensity || 0) > 0 ? 'translate-x-3' : ''}`} />
                  </span>
                </button>

                {(project.export.film_grain_intensity || 0) > 0 && (
                  <div className="grid gap-3 sm:grid-cols-2">
                    <label className="space-y-1 text-[8px] text-text-muted">
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
                    <label className="space-y-1 text-[8px] text-text-muted">
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
              <div className="rounded-xl border border-border bg-bg-tertiary p-3">
                <div className="text-[9px] font-medium uppercase tracking-wider text-text-muted">Delivery summary</div>
                <dl className="mt-2.5 space-y-1.5 text-[10px]">
                  <div className="flex justify-between gap-3"><dt className="text-text-muted">Frame</dt><dd className="font-mono text-text-secondary">{deliveryDimensions.width}×{deliveryDimensions.height}</dd></div>
                  <div className="flex justify-between gap-3"><dt className="text-text-muted">Rate</dt><dd className="font-mono text-text-secondary">{outputFps.toFixed(outputFps % 1 ? 2 : 0)} fps</dd></div>
                  <div className="flex justify-between gap-3"><dt className="text-text-muted">Duration</dt><dd className="font-mono text-text-secondary">{formatEditorTime(duration)}</dd></div>
                  <div className="flex justify-between gap-3"><dt className="text-text-muted">Format</dt><dd className="text-right text-text-secondary">MP4 · {project.export.codec.toUpperCase()}</dd></div>
                </dl>
              </div>

              {lastExport && (
                <div className="rounded-xl border border-green-500/25 bg-green-500/5 p-3">
                  <div className="flex items-center gap-1.5 text-[10px] font-medium text-indicator-success"><CheckCircle2 size={12} /> Latest export</div>
                  <div className="mt-1.5 truncate text-[9px] text-text-secondary" title={lastExport.filename}>{lastExport.filename}</div>
                  <div className="mt-2 grid grid-cols-2 gap-1.5">
                    <a href={api.getFileUrl(lastExport.filename, lastExport.workspace)} target="_blank" rel="noreferrer" className="flex items-center justify-center gap-1 rounded-md border border-border bg-bg-secondary px-2 py-1.5 text-[9px] text-text-secondary hover:border-accent-blue/40 hover:text-accent-blue">
                      <ExternalLink size={10} /> Open
                    </a>
                    <a href={api.getFileUrl(lastExport.filename, lastExport.workspace)} download={lastExport.filename} className="flex items-center justify-center gap-1 rounded-md border border-border bg-bg-secondary px-2 py-1.5 text-[9px] text-text-secondary hover:border-accent-blue/40 hover:text-accent-blue">
                      <Download size={10} /> Save file
                    </a>
                  </div>
                </div>
              )}

              {(project.exports || []).length > 1 && (
                <div className="rounded-xl border border-border bg-bg-tertiary p-3">
                  <div className="text-[9px] font-medium uppercase tracking-wider text-text-muted">Recent exports</div>
                  <div className="mt-2 space-y-1.5">
                    {(project.exports || []).slice(1, 5).map(record => (
                      <a key={record.id} href={api.getFileUrl(record.filename, record.workspace)} target="_blank" rel="noreferrer" className="flex items-center justify-between gap-2 rounded-md px-1.5 py-1 text-[9px] text-text-muted hover:bg-bg-hover hover:text-text-secondary">
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
          {exportJobId ? (
            <div className="space-y-2">
              <div className="flex items-center justify-between text-[10px]">
                <span className="flex items-center gap-1.5 text-text-secondary">
                  <Loader2 size={11} className={`${exportQueueStatus === 'held' ? '' : 'animate-spin'} text-accent-blue`} />
                  {exportQueueStatus === 'held' ? 'Held in the universal queue' : 'Rendering through the universal queue'}
                </span>
                <span className="flex items-center gap-2">
                  <span className="font-mono text-text-primary">{Math.round(exportProgress * 100)}%</span>
                  <button type="button" onClick={() => stopGeneration(exportJobId)} className="rounded border border-red-500/25 px-1.5 py-0.5 text-[8px] text-red-300 hover:bg-red-500/10">Cancel</button>
                </span>
              </div>
              <div className="h-1.5 overflow-hidden rounded-full bg-bg-active"><div className="h-full rounded-full bg-accent-blue transition-all" style={{ width: `${Math.max(1, exportProgress * 100)}%` }} /></div>
            </div>
          ) : (
            <div className="flex items-center justify-between gap-3">
              <p className="hidden text-[9px] text-text-muted sm:block">The project stays editable while a new non-destructive file is rendered.</p>
              <button
                type="button"
                onClick={() => void exportProject('queue')}
                disabled={duration <= 0}
                className="ml-auto flex items-center gap-1.5 rounded-lg border border-border bg-bg-secondary px-3 py-2 text-[10px] font-medium text-text-secondary hover:border-accent-blue/40 hover:text-accent-blue disabled:opacity-40"
                title="Hold this export in Maestro's universal queue"
              >
                Add to queue
              </button>
              <button
                type="button"
                onClick={() => void exportProject('now')}
                disabled={duration <= 0}
                className="flex items-center gap-1.5 rounded-lg bg-cta px-4 py-2 text-xs font-semibold text-white shadow-accent-glow disabled:opacity-40"
              >
                <Download size={13} /> Export video
              </button>
            </div>
          )}
        </footer>
      </section>
    </div>
  )
}
