import { useEffect, useMemo, useRef, useState } from 'react'
import { Archive, Check, Clock3, Download, Loader2, Package, RotateCcw, Upload, X } from 'lucide-react'
import * as api from '../api/client'
import type { ApiJobStatus } from '../api/client'
import type { EditorProject } from '../types'
import { useEditorStore } from './useEditorStore'
import { useEditorDialog } from './useEditorDialog'

type ProjectToolTab = 'history' | 'portable'

interface EditorProjectRevision {
  revision: number
  updated_at: number | string
  name?: string | null
}

interface OwnedPackageJob {
  jobId: string
  projectId: string
  projectName: string
  workspace: string
  revision: number
  status: ApiJobStatus['status']
  progress: number
  message: string
  outputFiles: string[]
  error: string | null
  cancelRequested?: boolean
}

const ACTIVE_JOB_STATUSES = new Set<ApiJobStatus['status']>(['held', 'queued', 'running'])

function projectEndpoint(projectId: string): string {
  return `/api/v1/editor/projects/${encodeURIComponent(projectId)}`
}

async function readJson<T>(response: Response, fallback: string): Promise<T> {
  const body = await response.json().catch(() => ({})) as { detail?: string; error?: string }
  if (!response.ok) throw new Error(body.detail || body.error || `${fallback} (${response.status})`)
  return body as T
}

function displayDate(value: number | string): string {
  const numeric = typeof value === 'number' ? value : Number.NaN
  const date = Number.isFinite(numeric)
    ? new Date(numeric > 1e12 ? numeric : numeric * 1000)
    : new Date(value)
  return Number.isNaN(date.getTime()) ? 'Date unavailable' : date.toLocaleString()
}

function fileLeaf(filename: string): string {
  return filename.split(/[\\/]/).filter(Boolean).at(-1) || filename
}

function jobProgress(progress: number): number {
  if (!Number.isFinite(progress)) return 0
  const percent = progress <= 1 ? progress * 100 : progress
  return Math.min(100, Math.max(0, percent))
}

function uploadProjectArchive(
  file: File,
  workspace: string,
  onProgress: (percent: number, uploaded: boolean) => void,
): { promise: Promise<EditorProject>; abort: () => void } {
  const form = new FormData()
  form.append('file', file)
  form.append('workspace', workspace)

  const request = new XMLHttpRequest()
  const promise = new Promise<EditorProject>((resolve, reject) => {
    request.open('POST', '/api/v1/editor/projects/import')
    request.upload.onprogress = event => {
      if (!event.lengthComputable) return
      onProgress(Math.round((event.loaded / Math.max(1, event.total)) * 100), false)
    }
    request.onload = () => {
      let body: EditorProject | { detail?: string; error?: string }
      try {
        body = JSON.parse(request.responseText) as EditorProject | { detail?: string; error?: string }
      } catch {
        reject(new Error(request.status >= 200 && request.status < 300
          ? 'The imported project response could not be read.'
          : `Project import failed (${request.status}).`))
        return
      }
      if (request.status < 200 || request.status >= 300) {
        reject(new Error(('detail' in body && body.detail) || ('error' in body && body.error) || `Project import failed (${request.status}).`))
        return
      }
      onProgress(100, true)
      resolve(body as EditorProject)
    }
    request.onerror = () => reject(new Error('Project import could not reach the server.'))
    request.onabort = () => reject(new Error('Project import was cancelled.'))
    request.send(form)
  })

  return { promise, abort: () => request.abort() }
}

export function EditorProjectTools({ open, onClose }: { open: boolean; onClose: () => void }) {
  const dialogRef = useEditorDialog(open, onClose)
  const project = useEditorStore(state => state.project)
  const dirty = useEditorStore(state => state.dirty)
  const saving = useEditorStore(state => state.saving)
  const saveProject = useEditorStore(state => state.saveProject)
  const loadProject = useEditorStore(state => state.loadProject)
  const [tab, setTab] = useState<ProjectToolTab>('history')
  const [revisions, setRevisions] = useState<EditorProjectRevision[]>([])
  const [historyLoading, setHistoryLoading] = useState(false)
  const [historyError, setHistoryError] = useState('')
  const [operation, setOperation] = useState<'restore' | 'package' | 'import' | null>(null)
  const [actionError, setActionError] = useState('')
  const [actionMessage, setActionMessage] = useState('')
  const [packageJob, setPackageJob] = useState<OwnedPackageJob | null>(null)
  const [packagePollError, setPackagePollError] = useState('')
  const [importFile, setImportFile] = useState<File | null>(null)
  const [importProgress, setImportProgress] = useState(0)
  const [importUploaded, setImportUploaded] = useState(false)
  const importRequest = useRef<{ abort: () => void } | null>(null)

  const activePackage = Boolean(packageJob && ACTIVE_JOB_STATUSES.has(packageJob.status))
  const zipFiles = useMemo(() => packageJob?.outputFiles.filter(filename => /\.zip$/i.test(filename)) || [], [packageJob?.outputFiles])
  const currentRevision = project?.revision
  const packageJobId = packageJob?.jobId
  const packageStatus = packageJob?.status

  useEffect(() => {
    if (!open || !project) return
    const controller = new AbortController()
    const endpoint = new URL(projectEndpoint(project.id) + '/revisions', window.location.origin)
    endpoint.searchParams.set('workspace', project.workspace)
    setHistoryLoading(true)
    setHistoryError('')
    setRevisions([])
    void fetch(endpoint.toString(), { signal: controller.signal, cache: 'no-store' })
      .then(response => readJson<{ revisions: EditorProjectRevision[] }>(response, 'Could not load project history'))
      .then(result => {
        if (!controller.signal.aborted) setRevisions(Array.isArray(result.revisions) ? result.revisions : [])
      })
      .catch(error => {
        if (!controller.signal.aborted) setHistoryError(error instanceof Error ? error.message : 'Could not load project history.')
      })
      .finally(() => { if (!controller.signal.aborted) setHistoryLoading(false) })
    return () => controller.abort()
  }, [open, project?.id, project?.workspace])

  useEffect(() => {
    if (!open) return
    const onKeyDown = (event: KeyboardEvent) => { if (event.key === 'Escape') onClose() }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [onClose, open])

  useEffect(() => {
    if (!packageJobId || !packageStatus || !ACTIVE_JOB_STATUSES.has(packageStatus)) return
    const jobId = packageJobId
    let active = true
    let timer: ReturnType<typeof setTimeout> | undefined
    const poll = async () => {
      try {
        const result = await api.fetchJobStatus(jobId)
        if (!active) return
        setPackagePollError('')
        setPackageJob(current => current?.jobId === jobId ? {
          ...current,
          status: result.status,
          progress: jobProgress(result.progress),
          message: result.message || result.phase || 'Preparing portable project…',
          outputFiles: Array.isArray(result.output_files) ? result.output_files : [],
          error: result.error,
        } : current)
        if (result.status === 'completed') setActionMessage('Portable ZIP is ready to download.')
        else if (result.status === 'failed' || result.status === 'cancelled') setActionMessage('')
        if (ACTIVE_JOB_STATUSES.has(result.status)) timer = setTimeout(poll, 1200)
      } catch (error) {
        if (!active) return
        setPackagePollError(error instanceof Error ? error.message : 'Package status is temporarily unavailable.')
        timer = setTimeout(poll, 2000)
      }
    }
    void poll()
    return () => {
      active = false
      if (timer) clearTimeout(timer)
    }
  }, [packageJobId, packageStatus])

  const saveCurrentProject = async (projectId: string): Promise<EditorProject> => {
    let state = useEditorStore.getState()
    if (state.project?.id !== projectId) throw new Error('The active project changed. Reopen Project tools and try again.')
    if (state.saving) throw new Error('Wait for the current project save to finish, then try again.')

    if (state.dirty || state.project.revision == null) {
      const saved = await saveProject()
      state = useEditorStore.getState()
      if (!saved || state.dirty || state.project?.id !== projectId) {
        throw new Error(state.error || 'Save the current project before continuing.')
      }
    }

    const savedProject = useEditorStore.getState().project
    if (!savedProject || savedProject.id !== projectId) throw new Error('The active project changed. Try again.')
    if (!Number.isFinite(savedProject.revision)) throw new Error('The server did not return a project revision. Save the project and try again.')
    return savedProject
  }

  const loadReturnedProject = async (target: EditorProject) => {
    if (!target.id) throw new Error('The server did not return a project ID.')
    await loadProject(target.id)
    const state = useEditorStore.getState()
    if (state.project?.id !== target.id) throw new Error(state.error || 'The project was updated, but could not be opened.')
    if (Number.isFinite(target.revision) && state.project.revision !== target.revision) {
      throw new Error('The project opened at a different revision than the server returned. Refresh project history before continuing.')
    }
  }

  const restoreRevision = async (revision: EditorProjectRevision) => {
    if (!project || operation || saving) return
    setOperation('restore')
    setActionError('')
    setActionMessage('')
    try {
      const current = await saveCurrentProject(project.id)
      const response = await fetch(`${projectEndpoint(current.id)}/restore`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          workspace: current.workspace,
          revision: revision.revision,
          expected_revision: current.revision,
        }),
      })
      const restored = await readJson<EditorProject>(response, 'Could not restore this revision')
      if (restored.id !== current.id) throw new Error('The server restored a different project than requested.')
      await loadReturnedProject(restored)
      setActionMessage(`Restored revision ${revision.revision}. It is now saved as a new revision.`)
    } catch (error) {
      setActionError(error instanceof Error ? error.message : 'Could not restore this revision.')
    } finally {
      setOperation(null)
    }
  }

  const createPackage = async () => {
    if (!project || operation || saving || activePackage) return
    setOperation('package')
    setActionError('')
    setActionMessage('')
    setPackagePollError('')
    try {
      const current = await saveCurrentProject(project.id)
      const response = await fetch(`${projectEndpoint(current.id)}/package`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ workspace: current.workspace, revision: current.revision }),
      })
      const created = await readJson<{ job_id: string; status?: ApiJobStatus['status'] }>(response, 'Could not package this project')
      if (!created.job_id) throw new Error('The server did not return a package job ID.')
      setPackageJob({
        jobId: created.job_id,
        projectId: current.id,
        projectName: current.name,
        workspace: current.workspace,
        revision: current.revision as number,
        status: created.status || 'queued',
        progress: 0,
        message: 'Waiting for package worker…',
        outputFiles: [],
        error: null,
      })
      setActionMessage('A portable package is being prepared. You can keep editing while it runs.')
    } catch (error) {
      setActionError(error instanceof Error ? error.message : 'Could not package this project.')
    } finally {
      setOperation(null)
    }
  }

  const cancelPackage = async () => {
    if (!packageJob || !ACTIVE_JOB_STATUSES.has(packageJob.status) || packageJob.cancelRequested) return
    setActionError('')
    setPackagePollError('')
    setPackageJob(current => current?.jobId === packageJob.jobId ? { ...current, cancelRequested: true } : current)
    try {
      await api.cancelJob(packageJob.jobId)
      setActionMessage('Cancellation requested. Waiting for the package worker to stop…')
    } catch (error) {
      setPackageJob(current => current?.jobId === packageJob.jobId ? { ...current, cancelRequested: false } : current)
      setActionError(error instanceof Error ? error.message : 'Could not cancel the package job.')
    }
  }

  const importProject = async () => {
    if (!project || !importFile || operation || saving) return
    setOperation('import')
    setActionError('')
    setActionMessage('')
    setImportProgress(0)
    setImportUploaded(false)
    try {
      const current = await saveCurrentProject(project.id)
      const upload = uploadProjectArchive(importFile, current.workspace, (percent, uploaded) => {
        setImportProgress(percent)
        setImportUploaded(uploaded)
      })
      importRequest.current = upload
      const imported = await upload.promise
      importRequest.current = null
      if (!imported.id) throw new Error('The server did not return the imported project.')
      await loadReturnedProject(imported)
      setImportFile(null)
      setActionMessage(`Opened imported project “${imported.name}”. The source project and its assets were left in place.`)
    } catch (error) {
      importRequest.current = null
      const message = error instanceof Error ? error.message : 'Could not import this project.'
      if (message === 'Project import was cancelled.') setActionMessage(message)
      else setActionError(message)
    } finally {
      setOperation(null)
    }
  }

  const cancelImport = () => {
    importRequest.current?.abort()
    importRequest.current = null
  }

  if (!open || !project) return null

  const packageFiles = zipFiles

  return (
    <div
      ref={dialogRef}
      className="fixed inset-0 z-[160] flex h-[100dvh] items-center justify-center overflow-y-auto bg-black/65 px-3 py-4 backdrop-blur-sm"
      style={{ paddingTop: 'max(1rem, env(safe-area-inset-top))', paddingBottom: 'max(1rem, env(safe-area-inset-bottom))' }}
      onMouseDown={event => { if (event.target === event.currentTarget) onClose() }}
    >
      <section
        role="dialog"
        aria-modal="true"
        aria-labelledby="editor-project-tools-title"
        className="flex max-h-[min(760px,calc(100dvh-2rem))] w-full max-w-[720px] flex-col overflow-hidden rounded-2xl border border-border bg-bg-secondary shadow-2xl"
      >
        <header className="flex items-start justify-between gap-4 border-b border-border px-5 py-4">
          <div className="flex min-w-0 items-start gap-3">
            <span className="mt-0.5 rounded-lg bg-accent-blue/10 p-2 text-accent-blue"><Archive size={17} /></span>
            <div className="min-w-0">
              <h2 id="editor-project-tools-title" className="text-sm font-semibold text-text-primary">Project tools</h2>
              <p className="mt-1 truncate text-[11px] text-text-muted">{project.name} · revision {project.revision ?? 'unsaved'}</p>
            </div>
          </div>
          <button type="button" onClick={onClose} aria-label="Close Project tools" className="rounded-md p-1.5 text-text-muted hover:bg-bg-hover hover:text-text-primary">
            <X size={16} />
          </button>
        </header>

        <div role="tablist" aria-label="Project tools sections" className="flex gap-1 border-b border-border px-4 pt-2">
          <button type="button" role="tab" aria-selected={tab === 'history'} onClick={() => setTab('history')} className={`rounded-t-lg px-3 py-2 text-xs ${tab === 'history' ? 'border border-b-0 border-border bg-bg-primary text-text-primary' : 'text-text-muted hover:text-text-primary'}`}>
            History
          </button>
          <button type="button" role="tab" aria-selected={tab === 'portable'} onClick={() => setTab('portable')} className={`rounded-t-lg px-3 py-2 text-xs ${tab === 'portable' ? 'border border-b-0 border-border bg-bg-primary text-text-primary' : 'text-text-muted hover:text-text-primary'}`}>
            Portable project
          </button>
        </div>

        <div className="min-h-0 flex-1 overflow-y-auto bg-bg-primary p-4 sm:p-5">
          {tab === 'history' ? (
            <div role="tabpanel" aria-label="History" className="space-y-4">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div>
                  <h3 className="text-xs font-semibold text-text-primary">Recovery snapshots</h3>
                  <p className="mt-1 text-[11px] text-text-muted">Restore a saved version. Restoring adds a new revision and keeps the other snapshots.</p>
                </div>
                <button type="button" onClick={() => {
                  if (!project) return
                  const endpoint = new URL(projectEndpoint(project.id) + '/revisions', window.location.origin)
                  endpoint.searchParams.set('workspace', project.workspace)
                  setHistoryLoading(true); setHistoryError('')
                  void fetch(endpoint.toString(), { cache: 'no-store' })
                    .then(response => readJson<{ revisions: EditorProjectRevision[] }>(response, 'Could not load project history'))
                    .then(result => setRevisions(Array.isArray(result.revisions) ? result.revisions : []))
                    .catch(error => setHistoryError(error instanceof Error ? error.message : 'Could not load project history.'))
                    .finally(() => setHistoryLoading(false))
                }} disabled={historyLoading} className="rounded-md border border-border px-2.5 py-1.5 text-[11px] text-text-secondary hover:bg-bg-hover disabled:opacity-50">
                  {historyLoading ? <Loader2 size={12} className="mr-1 inline animate-spin" /> : <Clock3 size={12} className="mr-1 inline" />}
                  Refresh
                </button>
              </div>

              {historyError && <div role="alert" className="rounded-lg border border-red-500/30 bg-red-500/10 p-3 text-xs text-red-300">{historyError}</div>}
              {actionError && <div role="alert" className="rounded-lg border border-red-500/30 bg-red-500/10 p-3 text-xs text-red-300">{actionError}</div>}
              {actionMessage && <div role="status" className="rounded-lg border border-accent-blue/20 bg-accent-blue/5 p-3 text-xs text-text-secondary">{actionMessage}</div>}
              {historyLoading && revisions.length === 0 ? (
                <div className="flex items-center gap-2 rounded-lg border border-border bg-bg-secondary p-4 text-xs text-text-muted"><Loader2 size={14} className="animate-spin" /> Loading saved snapshots…</div>
              ) : revisions.length ? (
                <ol className="space-y-2">
                  {revisions.map(snapshot => {
                    const isCurrent = snapshot.revision === currentRevision
                    return (
                      <li key={snapshot.revision} className="flex flex-wrap items-center justify-between gap-3 rounded-xl border border-border bg-bg-secondary p-3">
                        <div className="min-w-0">
                          <div className="flex items-center gap-2 text-xs font-medium text-text-primary">
                            <span>Revision {snapshot.revision}</span>
                            {isCurrent && <span className="inline-flex items-center gap-1 rounded-full bg-accent-blue/10 px-2 py-0.5 text-[9px] font-medium text-accent-blue"><Check size={10} /> Current</span>}
                            {snapshot.name && snapshot.name !== project.name && <span className="truncate text-text-secondary">{snapshot.name}</span>}
                          </div>
                          <div className="mt-1 text-[10px] text-text-muted">{displayDate(snapshot.updated_at)}</div>
                        </div>
                        <button
                          type="button"
                          onClick={() => void restoreRevision(snapshot)}
                          disabled={isCurrent || operation !== null || saving}
                          aria-label={`Restore revision ${snapshot.revision}`}
                          className="inline-flex shrink-0 items-center gap-1.5 rounded-lg border border-border px-3 py-2 text-[11px] font-medium text-text-secondary hover:border-accent-blue/40 hover:bg-accent-blue/5 hover:text-text-primary disabled:cursor-not-allowed disabled:opacity-40"
                        >
                          {operation === 'restore' ? <Loader2 size={12} className="animate-spin" /> : <RotateCcw size={12} />}
                          Restore
                        </button>
                      </li>
                    )
                  })}
                </ol>
              ) : !historyLoading && !historyError ? (
                <div className="rounded-xl border border-dashed border-border bg-bg-secondary p-6 text-center">
                  <Clock3 size={18} className="mx-auto text-text-muted" />
                  <p className="mt-2 text-xs font-medium text-text-secondary">No saved snapshots yet</p>
                  <p className="mt-1 text-[10px] text-text-muted">Save this project to create its first recovery point.</p>
                </div>
              ) : null}
            </div>
          ) : (
            <div role="tabpanel" aria-label="Portable project" className="space-y-4">
              <section className="rounded-xl border border-border bg-bg-secondary p-4">
                <div className="flex items-start gap-3">
                  <Package size={17} className="mt-0.5 shrink-0 text-accent-blue" />
                  <div className="min-w-0 flex-1">
                    <h3 className="text-xs font-semibold text-text-primary">Create a portable project ZIP</h3>
                    <p className="mt-1 text-[11px] leading-relaxed text-text-muted">The package includes this saved project and its referenced assets. It runs as a background job, so you can continue editing.</p>
                    <button
                      type="button"
                      onClick={() => void createPackage()}
                      disabled={operation !== null || saving || activePackage || !project}
                      className="mt-3 inline-flex items-center gap-1.5 rounded-lg bg-accent-blue px-3 py-2 text-[11px] font-semibold text-white hover:bg-accent-blue/90 disabled:cursor-not-allowed disabled:opacity-45"
                    >
                      {operation === 'package' || activePackage ? <Loader2 size={12} className="animate-spin" /> : <Package size={12} />}
                      {activePackage ? 'Package running' : operation === 'package' ? 'Saving project…' : 'Create package'}
                    </button>
                  </div>
                </div>
              </section>

              {packageJob && (
                <section aria-label="Package job" className="rounded-xl border border-border bg-bg-secondary p-4">
                  <div className="flex items-start justify-between gap-3">
                    <div className="min-w-0">
                      <h3 className="truncate text-xs font-semibold text-text-primary">{packageJob.projectName}</h3>
                      <p className="mt-1 text-[10px] text-text-muted">Revision {packageJob.revision} · {packageJob.status}</p>
                    </div>
                    {activePackage && (
                      <button type="button" onClick={() => void cancelPackage()} disabled={packageJob.cancelRequested} className="rounded-md border border-border px-2 py-1 text-[10px] text-text-secondary hover:border-red-500/40 hover:text-red-300 disabled:opacity-45">
                        {packageJob.cancelRequested ? 'Cancelling…' : 'Cancel package'}
                      </button>
                    )}
                  </div>
                  {activePackage && (
                    <div className="mt-3">
                      <div className="mb-1 flex justify-between text-[10px] text-text-muted"><span>{packageJob.message || 'Preparing ZIP…'}</span><span>{Math.round(packageJob.progress)}%</span></div>
                      <div role="progressbar" aria-label="Project package progress" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(packageJob.progress)} className="h-1.5 overflow-hidden rounded-full bg-bg-hover">
                        <div className="h-full rounded-full bg-accent-blue transition-[width]" style={{ width: `${packageJob.progress}%` }} />
                      </div>
                    </div>
                  )}
                  {packageJob.status === 'completed' && packageFiles.length > 0 && (
                    <div className="mt-3 space-y-2">
                      {packageFiles.map(filename => (
                        <a key={filename} href={api.getFileUrl(filename, packageJob.workspace)} download={fileLeaf(filename)} className="flex items-center gap-2 rounded-lg border border-accent-blue/30 bg-accent-blue/5 px-3 py-2 text-[11px] font-medium text-accent-blue hover:bg-accent-blue/10">
                          <Download size={13} /> Download {fileLeaf(filename)}
                        </a>
                      ))}
                    </div>
                  )}
                  {packageJob.status === 'completed' && packageFiles.length === 0 && <p className="mt-3 text-[11px] text-text-muted">Package completed, but the job did not return a downloadable file.</p>}
                  {(packageJob.status === 'failed' || packageJob.error) && <div role="alert" className="mt-3 text-[11px] text-red-300">{packageJob.error || 'Project packaging failed.'}</div>}
                  {packageJob.status === 'cancelled' && <p role="status" className="mt-3 text-[11px] text-text-muted">Package cancelled.</p>}
                  {packagePollError && <p role="status" className="mt-3 text-[10px] text-amber-300">{packagePollError} Retrying status…</p>}
                </section>
              )}

              <section className="rounded-xl border border-border bg-bg-secondary p-4">
                <div className="flex items-start gap-3">
                  <Upload size={17} className="mt-0.5 shrink-0 text-text-secondary" />
                  <div className="min-w-0 flex-1">
                    <h3 className="text-xs font-semibold text-text-primary">Import a project ZIP</h3>
                    <p className="mt-1 text-[11px] leading-relaxed text-text-muted">Import creates a new project and keeps the source project and its assets in place.</p>
                    <label htmlFor="editor-project-archive" className="mt-3 block text-[10px] font-medium text-text-secondary">Project ZIP file</label>
                    <input
                      id="editor-project-archive"
                      type="file"
                      accept=".zip,application/zip"
                      aria-label="Choose project ZIP"
                      disabled={operation === 'import'}
                      onChange={event => { setImportFile(event.target.files?.[0] || null); setActionError(''); setActionMessage('') }}
                      className="mt-1 block w-full text-[11px] text-text-secondary file:mr-3 file:rounded-md file:border-0 file:bg-bg-hover file:px-3 file:py-2 file:text-[10px] file:font-medium file:text-text-primary hover:file:bg-bg-hover/80 disabled:opacity-50"
                    />
                    <div className="mt-3 flex flex-wrap items-center gap-2">
                      <button type="button" onClick={() => void importProject()} disabled={!importFile || operation !== null || saving} className="inline-flex items-center gap-1.5 rounded-lg border border-border px-3 py-2 text-[11px] font-medium text-text-secondary hover:bg-bg-hover hover:text-text-primary disabled:cursor-not-allowed disabled:opacity-40">
                        {operation === 'import' ? <Loader2 size={12} className="animate-spin" /> : <Upload size={12} />}
                        Import ZIP
                      </button>
                      {operation === 'import' && <button type="button" onClick={cancelImport} className="rounded-lg border border-border px-3 py-2 text-[11px] text-text-secondary hover:text-text-primary">Cancel import</button>}
                    </div>
                    {operation === 'import' && (
                      <div className="mt-3" role="status" aria-live="polite">
                        <div className="mb-1 flex justify-between text-[10px] text-text-muted">
                          <span>{importUploaded ? 'Uploaded; opening imported project…' : 'Uploading and importing ZIP…'}</span>
                          <span>{importUploaded ? 'Processing' : `${importProgress}%`}</span>
                        </div>
                        <div role="progressbar" aria-label="Project import progress" aria-valuemin={0} aria-valuemax={100} aria-valuenow={importProgress} className="h-1.5 overflow-hidden rounded-full bg-bg-hover">
                          <div className="h-full rounded-full bg-accent-blue transition-[width]" style={{ width: `${importProgress}%` }} />
                        </div>
                      </div>
                    )}
                    {actionError && <div role="alert" className="mt-3 rounded-lg border border-red-500/30 bg-red-500/10 p-2.5 text-[11px] text-red-300">{actionError}</div>}
                    {actionMessage && <div role="status" className="mt-3 rounded-lg border border-accent-blue/20 bg-accent-blue/5 p-2.5 text-[11px] text-text-secondary">{actionMessage}</div>}
                  </div>
                </div>
              </section>
            </div>
          )}
        </div>
        <footer className="flex items-center justify-between gap-3 border-t border-border bg-bg-secondary px-5 py-3">
          <span className="text-[10px] text-text-muted">{dirty ? 'Unsaved edits will be saved before project changes.' : 'Project files remain in their current workspace.'}</span>
          <button type="button" onClick={onClose} className="rounded-lg border border-border px-3 py-1.5 text-[11px] font-medium text-text-secondary hover:bg-bg-hover">Done</button>
        </footer>
      </section>
    </div>
  )
}
