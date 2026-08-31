import { useEffect, useMemo, useRef, useState } from 'react'
import {
  ArrowDown,
  ArrowUp,
  Check,
  Clock,
  ListVideo,
  Loader2,
  Pause,
  Pencil,
  Play,
  Square,
  Trash2,
  X,
} from 'lucide-react'
import { useStore } from '../stores/useStore'

const ACTIVE_JOB_STATUSES = new Set(['held', 'queued', 'running'])
const ACTIVE_DIRECTOR_STATUSES = new Set(['held', 'queued', 'running'])
const ACTIVE_PIPELINE_STATUSES = new Set(['running', 'paused'])

function compactStatus(value: string): string {
  return value.replace(/_/g, ' ').replace(/^./, letter => letter.toUpperCase())
}

function progressPercent(step: number, totalSteps: number, progress: number): number {
  if (totalSteps > 0) return Math.max(0, Math.min(100, (step / totalSteps) * 100))
  return Math.max(0, Math.min(100, progress * 100))
}

export function GlobalQueuePopover({
  iconSize = 16,
  panelAlign = 'icon',
}: {
  iconSize?: number
  panelAlign?: 'icon' | 'header-edge'
}) {
  const [open, setOpen] = useState(false)
  const [startingAll, setStartingAll] = useState(false)
  const rootRef = useRef<HTMLDivElement>(null)

  const jobs = useStore(state => state.jobs)
  const stopGeneration = useStore(state => state.stopGeneration)
  const startStudioQueue = useStore(state => state.startStudioQueue)
  const directorQueue = useStore(state => state.directorQueue)
  const directorQueueLoading = useStore(state => state.directorQueueLoading)
  const loadDirectorQueue = useStore(state => state.loadDirectorQueue)
  const loadDirectorQueueEntry = useStore(state => state.loadDirectorQueueEntry)
  const startDirectorQueue = useStore(state => state.startDirectorQueue)
  const pauseDirectorQueue = useStore(state => state.pauseDirectorQueue)
  const removeDirectorQueueEntry = useStore(state => state.removeDirectorQueueEntry)
  const moveDirectorQueueEntry = useStore(state => state.moveDirectorQueueEntry)
  const pipelineId = useStore(state => state.pipelineId)
  const pipelineStatus = useStore(state => state.pipelineStatus)
  const stopPipeline = useStore(state => state.stopPipeline)
  const setSidebarOpen = useStore(state => state.setSidebarOpen)
  const setSettingsOpen = useStore(state => state.setSettingsOpen)

  const studioJobs = useMemo(
    () => jobs.filter(job => ACTIVE_JOB_STATUSES.has(job.status)),
    [jobs],
  )
  const studioHeldCount = studioJobs.filter(job => job.status === 'held').length
  const directorEntries = directorQueue?.entries || []
  const pendingDirectorCount = directorEntries.filter(entry => (
    ACTIVE_DIRECTOR_STATUSES.has(entry.status)
  )).length
  const startableDirectorCount = directorEntries.filter(entry => (
    entry.status === 'held' || entry.status === 'queued'
  )).length
  const activePipeline = Boolean(
    pipelineId && pipelineStatus && ACTIVE_PIPELINE_STATUSES.has(pipelineStatus.status),
  )
  const activePipelineIsQueued = Boolean(
    activePipeline && directorEntries.some(entry => (
      entry.status === 'running'
      && (!entry.pipeline_id || entry.pipeline_id === pipelineId)
    )),
  )
  const totalCount = studioJobs.length
    + pendingDirectorCount
    + (activePipeline && !activePipelineIsQueued ? 1 : 0)

  useEffect(() => {
    void loadDirectorQueue()
  }, [loadDirectorQueue])

  useEffect(() => {
    if (!directorQueue?.running) return
    const timer = window.setInterval(() => void loadDirectorQueue(), 2500)
    return () => window.clearInterval(timer)
  }, [directorQueue?.running, loadDirectorQueue])

  useEffect(() => {
    if (!open) return
    const closeOnOutsideClick = (event: MouseEvent) => {
      if (rootRef.current && !rootRef.current.contains(event.target as Node)) {
        setOpen(false)
      }
    }
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === 'Escape') setOpen(false)
    }
    document.addEventListener('mousedown', closeOnOutsideClick)
    document.addEventListener('keydown', closeOnEscape)
    return () => {
      document.removeEventListener('mousedown', closeOnOutsideClick)
      document.removeEventListener('keydown', closeOnEscape)
    }
  }, [open])

  const toggleOpen = () => {
    const next = !open
    setOpen(next)
    if (next) {
      setSettingsOpen(false)
      void loadDirectorQueue()
    }
  }

  const openDirectorEntry = async (entryId: string) => {
    await loadDirectorQueueEntry(entryId)
    setOpen(false)
    setSidebarOpen(true)
  }

  const startAllQueues = async () => {
    if (startingAll) return
    setStartingAll(true)
    try {
      // Release Studio first. Director's existing GPU gate sees those jobs
      // as queued/running and waits, so one click safely starts both systems.
      if (studioHeldCount > 0) await startStudioQueue()
      if (
        startableDirectorCount > 0
        && (!directorQueue?.running || directorQueue.paused)
      ) {
        await startDirectorQueue()
      }
    } finally {
      setStartingAll(false)
    }
  }

  return (
    <div ref={rootRef} className="relative">
      <button
        type="button"
        onClick={toggleOpen}
        className={`relative rounded-lg ${iconSize >= 20 ? 'p-2' : 'p-1.5'} transition-colors ${
          open
            ? 'bg-bg-active text-text-primary'
            : 'text-text-secondary hover:bg-bg-hover hover:text-text-primary'
        }`}
        title="Generation queue"
        aria-label={`Generation queue, ${totalCount} ${totalCount === 1 ? 'item' : 'items'}`}
        aria-expanded={open}
        aria-haspopup="dialog"
      >
        <ListVideo size={iconSize} />
        {totalCount > 0 && (
          <span className="absolute -right-1.5 -top-1.5 flex h-4 min-w-4 items-center justify-center rounded-full bg-accent-blue px-1 text-[9px] font-semibold leading-none text-white shadow-sm">
            {totalCount > 99 ? '99+' : totalCount}
          </span>
        )}
      </button>

      {open && (
        <div
          role="dialog"
          aria-label="Generation queue"
          className={`absolute ${panelAlign === 'header-edge' ? '-right-12' : 'right-0'} top-full z-[80] mt-2 flex max-h-[min(70vh,640px)] w-[min(390px,calc(100vw-1rem))] flex-col overflow-hidden rounded-xl border border-border bg-bg-secondary shadow-2xl`}
        >
          <div className="flex items-center justify-between border-b border-border px-3 py-2.5">
            <div className="min-w-0">
              <div className="flex items-center gap-2">
                <ListVideo size={14} className="text-accent-blue" />
                <h2 className="text-xs font-semibold text-text-primary">Generation Queue</h2>
                <span className="rounded-full bg-accent-blue/15 px-1.5 py-0.5 text-[9px] text-accent-blue">
                  {totalCount} {totalCount === 1 ? 'item' : 'items'}
                </span>
              </div>
              <p className="mt-0.5 text-[9px] text-text-muted">Studio and Director in one place</p>
            </div>
            <button
              type="button"
              onClick={() => setOpen(false)}
              className="rounded p-1 text-text-muted hover:bg-bg-hover hover:text-text-primary"
              aria-label="Close queue"
            >
              <X size={14} />
            </button>
          </div>

          <div className="min-h-0 flex-1 space-y-3 overflow-y-auto p-2.5">
            {activePipeline && !activePipelineIsQueued && pipelineStatus && (
              <section className="space-y-1.5">
                <div className="flex items-center justify-between px-1">
                  <span className="text-[10px] font-medium uppercase tracking-wider text-text-muted">Director now</span>
                  <span className="text-[9px] text-accent-blue">1 active</span>
                </div>
                <div className="rounded-lg border border-accent-blue/25 bg-bg-tertiary p-2">
                  <div className="flex items-center gap-2">
                    <Loader2 size={11} className="shrink-0 animate-spin text-accent-blue" />
                    <div className="min-w-0 flex-1">
                      <div className="truncate text-[10px] text-text-secondary">
                        {pipelineStatus.progress?.message || compactStatus(pipelineStatus.phase)}
                      </div>
                      <div className="text-[9px] text-text-muted">Director · {compactStatus(pipelineStatus.status)}</div>
                    </div>
                    <button
                      type="button"
                      onClick={() => void stopPipeline()}
                      className="rounded p-1 text-red-400 hover:bg-red-500/10 hover:text-red-300"
                      title="Stop Director generation"
                    >
                      <Square size={10} />
                    </button>
                  </div>
                  {pipelineStatus.progress && pipelineStatus.progress.total > 0 && (
                    <div className="mt-2 h-1 overflow-hidden rounded-full bg-bg-active">
                      <div
                        className="h-full rounded-full bg-accent-blue transition-all"
                        style={{
                          width: `${Math.max(0, Math.min(100, (pipelineStatus.progress.current / pipelineStatus.progress.total) * 100))}%`,
                        }}
                      />
                    </div>
                  )}
                </div>
              </section>
            )}

            {studioJobs.length > 0 && (
              <section className="space-y-1.5">
                <div className="flex items-center justify-between px-1">
                  <span className="text-[10px] font-medium uppercase tracking-wider text-text-muted">Studio</span>
                  {studioHeldCount > 0 ? (
                    <button
                      type="button"
                      onClick={() => void startAllQueues()}
                      disabled={startingAll}
                      className="flex items-center gap-1 rounded border border-green-500/30 bg-green-500/10 px-2 py-1 text-[9px] text-indicator-success disabled:opacity-40"
                      title="Start all held Studio jobs, then any held Director projects"
                    >
                      {startingAll
                        ? <Loader2 size={9} className="animate-spin" />
                        : <Play size={9} />}
                      Start queue
                    </button>
                  ) : (
                    <span className="text-[9px] text-text-muted">{studioJobs.length} active</span>
                  )}
                </div>
                <div className="space-y-1">
                  {studioJobs.map((job, index) => {
                    const percent = progressPercent(job.step, job.totalSteps, job.progress)
                    const label = job.phase || job.message || (
                      job.status === 'held'
                        ? 'Ready - waiting for Start Queue'
                        : job.status === 'queued'
                          ? 'Waiting to start'
                          : 'Generating'
                    )
                    return (
                      <div key={job.id || `pending-${index}`} className="rounded-lg border border-border bg-bg-tertiary p-2">
                        <div className="flex items-center gap-2">
                          {job.status === 'running'
                            ? <Loader2 size={11} className="shrink-0 animate-spin text-accent-blue" />
                            : <Clock size={11} className="shrink-0 text-text-muted" />}
                          <div className="min-w-0 flex-1">
                            <div className="truncate text-[10px] text-text-secondary">{label}</div>
                            <div className="text-[9px] text-text-muted">
                              Studio · {compactStatus(job.status)}
                              {job.totalSteps > 0 ? ` · Step ${job.step}/${job.totalSteps}` : ''}
                            </div>
                          </div>
                          <button
                            type="button"
                            onClick={() => {
                              if (job.id) stopGeneration(job.id)
                            }}
                            disabled={!job.id}
                            className="rounded p-1 text-red-400 hover:bg-red-500/10 hover:text-red-300 disabled:cursor-wait disabled:opacity-30"
                            title={!job.id
                              ? 'Waiting for the server to accept this job'
                              : job.status === 'held'
                                ? 'Remove held generation'
                                : job.status === 'queued'
                                  ? 'Cancel queued generation'
                                : 'Stop generation'}
                          >
                            {job.status === 'held' || job.status === 'queued'
                              ? <X size={11} />
                              : <Square size={10} />}
                          </button>
                        </div>
                        <div className="mt-2 h-1 overflow-hidden rounded-full bg-bg-active">
                          <div
                            className={`h-full rounded-full bg-accent-blue transition-all ${percent === 0 && job.status === 'running' ? 'w-full animate-pulse opacity-60' : ''}`}
                            style={percent > 0 ? { width: `${percent}%` } : undefined}
                          />
                        </div>
                      </div>
                    )
                  })}
                </div>
              </section>
            )}

            {directorEntries.length > 0 && (
              <section className="space-y-1.5">
                <div className="flex items-center justify-between gap-2 px-1">
                  <div className="flex items-center gap-1.5">
                    <span className="text-[10px] font-medium uppercase tracking-wider text-text-muted">Director</span>
                    <span className="text-[9px] text-text-muted">{pendingDirectorCount} waiting</span>
                  </div>
                  {directorQueue?.running && !directorQueue.paused ? (
                    <button
                      type="button"
                      onClick={() => void pauseDirectorQueue()}
                      disabled={directorQueueLoading}
                      className="flex items-center gap-1 rounded border border-orange-500/30 bg-orange-500/10 px-2 py-1 text-[9px] text-chip-orange disabled:opacity-40"
                      title="Finish the active Director project, then stop dispatching"
                    >
                      <Pause size={9} /> Pause after current
                    </button>
                  ) : (
                    <button
                      type="button"
                      onClick={() => void startAllQueues()}
                      disabled={(startableDirectorCount === 0 && studioHeldCount === 0) || directorQueueLoading || startingAll}
                      className="flex items-center gap-1 rounded border border-green-500/30 bg-green-500/10 px-2 py-1 text-[9px] text-indicator-success disabled:opacity-40"
                    >
                      {startingAll
                        ? <Loader2 size={9} className="animate-spin" />
                        : <Play size={9} />}
                      Start queue
                    </button>
                  )}
                </div>
                <div className="space-y-1">
                  {directorEntries.map((entry, index) => (
                    <div key={entry.id} className="flex items-center gap-1.5 rounded-lg border border-border bg-bg-tertiary px-2 py-1.5">
                      {entry.status === 'running'
                        ? <Loader2 size={10} className="shrink-0 animate-spin text-accent-blue" />
                        : entry.status === 'completed'
                          ? <Check size={10} className="shrink-0 text-indicator-success" />
                          : <Clock size={10} className="shrink-0 text-text-muted" />}
                      <button
                        type="button"
                        onClick={() => void openDirectorEntry(entry.id)}
                        disabled={directorQueueLoading}
                        className="min-w-0 flex-1 text-left"
                        title={entry.error || entry.message || entry.scene_description}
                      >
                        <div className="truncate text-[10px] text-text-secondary">
                          {entry.scene_description || `${entry.pipeline_type.replace(/_/g, ' ')} project`}
                        </div>
                        <div className={`truncate text-[8px] ${entry.status === 'failed' ? 'text-red-400' : 'text-text-muted'}`}>
                          {compactStatus(entry.status)} · {entry.error || entry.message || entry.video_model}
                        </div>
                      </button>
                      <button
                        type="button"
                        onClick={() => void openDirectorEntry(entry.id)}
                        disabled={directorQueueLoading}
                        title="Open Director project"
                        className="rounded p-1 text-text-muted hover:bg-bg-hover hover:text-accent-blue disabled:opacity-40"
                      >
                        <Pencil size={9} />
                      </button>
                      {entry.status !== 'running' && (
                        <>
                          <button
                            type="button"
                            onClick={() => void moveDirectorQueueEntry(entry.id, -1)}
                            disabled={index === 0 || directorQueueLoading}
                            className="rounded p-1 text-text-muted hover:bg-bg-hover disabled:opacity-20"
                            title="Move up"
                          >
                            <ArrowUp size={9} />
                          </button>
                          <button
                            type="button"
                            onClick={() => void moveDirectorQueueEntry(entry.id, 1)}
                            disabled={index === directorEntries.length - 1 || directorQueueLoading}
                            className="rounded p-1 text-text-muted hover:bg-bg-hover disabled:opacity-20"
                            title="Move down"
                          >
                            <ArrowDown size={9} />
                          </button>
                          <button
                            type="button"
                            onClick={() => void removeDirectorQueueEntry(entry.id)}
                            disabled={directorQueueLoading}
                            className="rounded p-1 text-text-muted hover:bg-red-500/10 hover:text-red-400 disabled:opacity-40"
                            title="Remove from queue"
                          >
                            <Trash2 size={9} />
                          </button>
                        </>
                      )}
                    </div>
                  ))}
                </div>
              </section>
            )}

            {!activePipeline && studioJobs.length === 0 && directorEntries.length === 0 && (
              <div className="flex min-h-36 flex-col items-center justify-center gap-2 px-6 text-center">
                <ListVideo size={28} className="text-text-muted/60" />
                <div className="text-xs font-medium text-text-secondary">Queue is empty</div>
                <p className="text-[10px] leading-relaxed text-text-muted">
                  Queued Studio generations and held Director projects will appear here.
                </p>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
