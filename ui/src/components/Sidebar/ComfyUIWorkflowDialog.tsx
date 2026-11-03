import { useEffect, useMemo, useRef, useState, type ChangeEvent } from 'react'
import { AlertTriangle, Check, FileJson2, Loader2, Play, RefreshCw, Trash2, Upload, X } from 'lucide-react'
import { uploadAudio, uploadImage } from '../../api/client'
import * as api from '../../api/comfyui'
import { useStore } from '../../stores/useStore'

const OWNED_RUN_KEY = 'maestro.comfyui.owned-run.v1'

type WorkflowLaunch = api.ComfyUIWorkflowLaunchRequest & { requestId: number }

function isScalarInputRecord(value: unknown): value is Record<string, api.ComfyUIScalar> {
  return Boolean(value && typeof value === 'object' && !Array.isArray(value)
    && Object.values(value).every(item =>
      typeof item === 'string' || typeof item === 'boolean' || (typeof item === 'number' && Number.isFinite(item))))
}
const STARTER_BINDINGS = JSON.stringify([
  { name: 'prompt', label: 'Prompt', node_id: '6', input: 'text', type: 'string', required: true },
], null, 2)

type OwnedRun = {
  client_submission_id: string
  workflow_id: string
  job_id?: string | null
  status: string
  detail?: string | null
}

function newUuid(): string {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') return crypto.randomUUID()
  const bytes = new Uint8Array(16)
  if (typeof crypto !== 'undefined' && crypto.getRandomValues) crypto.getRandomValues(bytes)
  else for (let index = 0; index < bytes.length; index += 1) bytes[index] = Math.floor(Math.random() * 256)
  bytes[6] = (bytes[6] & 0x0f) | 0x40
  bytes[8] = (bytes[8] & 0x3f) | 0x80
  const hex = Array.from(bytes, item => item.toString(16).padStart(2, '0')).join('')
  return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`
}

function storeOwnedRun(run: OwnedRun | null): void {
  try {
    if (run) localStorage.setItem(OWNED_RUN_KEY, JSON.stringify(run))
    else localStorage.removeItem(OWNED_RUN_KEY)
  } catch { /* Storage can be unavailable in private browsing. */ }
}

function readOwnedRun(): OwnedRun | null {
  try {
    const parsed = JSON.parse(localStorage.getItem(OWNED_RUN_KEY) || 'null') as Partial<OwnedRun> | null
    if (!parsed || typeof parsed.client_submission_id !== 'string' || !/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(parsed.client_submission_id) || typeof parsed.workflow_id !== 'string') return null
    return {
      client_submission_id: parsed.client_submission_id,
      workflow_id: parsed.workflow_id,
      job_id: typeof parsed.job_id === 'string' ? parsed.job_id : null,
      status: typeof parsed.status === 'string' ? parsed.status : 'checking',
      detail: typeof parsed.detail === 'string' ? parsed.detail : null,
    }
  } catch { return null }
}

function responseStatus(receipt: api.ComfyUIJobReceipt): string {
  return receipt.status || receipt.state || 'accepted'
}

function stringifyIssue(value: unknown): string {
  if (typeof value === 'string') return value
  if (value && typeof value === 'object') {
    const row = value as Record<string, unknown>
    const label = [row.node_id, row.class_type, row.name, row.filename, row.path, row.message, row.reason]
      .filter(item => typeof item === 'string' || typeof item === 'number')
      .map(String)
    return label.length ? label.join(' · ') : JSON.stringify(value)
  }
  return String(value)
}

function initialValues(bindings: api.ComfyUIBinding[]): Record<string, api.ComfyUIScalar> {
  const values: Record<string, api.ComfyUIScalar> = {}
  for (const binding of bindings) {
    if (binding.default !== undefined) values[binding.name] = binding.default
    else if (binding.type === 'boolean') values[binding.name] = false
    else if (binding.type === 'choice' && binding.options?.length) values[binding.name] = binding.options[0]
  }
  return values
}

function collectInputs(
  bindings: api.ComfyUIBinding[],
  current: Record<string, api.ComfyUIScalar>,
): { inputs: Record<string, api.ComfyUIScalar>; missing: string[]; invalid: string[] } {
  const inputs: Record<string, api.ComfyUIScalar> = {}
  const missing: string[] = []
  const invalid: string[] = []
  for (const binding of bindings) {
    const value = current[binding.name] ?? binding.default
    if (value === undefined) {
      if (binding.required) missing.push(binding.label || binding.name)
      continue
    }
    if (binding.required && typeof value === 'string' && !value.trim()) {
      missing.push(binding.label || binding.name)
      continue
    }
    if (binding.type === 'integer' && typeof value === 'number' && !Number.isInteger(value)) {
      invalid.push(`${binding.label || binding.name} must be a whole number`)
      continue
    }
    if ((binding.type === 'integer' || binding.type === 'number') && typeof value === 'number'
      && ((binding.min !== undefined && value < binding.min) || (binding.max !== undefined && value > binding.max))) {
      invalid.push(`${binding.label || binding.name} must be between ${binding.min ?? '−∞'} and ${binding.max ?? '∞'}`)
      continue
    }
    inputs[binding.name] = value
  }
  return { inputs, missing, invalid }
}

function displayValue(value: api.ComfyUIScalar | undefined): string {
  return value === undefined || value === null ? '' : String(value)
}

export function ComfyUIWorkflowButton({
  label = 'ComfyUI workflows',
  className = 'inline-flex items-center gap-1.5 rounded-lg border border-border bg-bg-primary px-2.5 py-2 text-[10px] text-text-secondary hover:text-text-primary',
}: { label?: string; className?: string } = {}) {
  const [open, setOpen] = useState(false)
  const workspace = useStore(state => state.activeWorkspace)
  const reconnectJobs = useStore(state => state.reconnectJobs)
  return <>
    <button type="button" onClick={() => setOpen(true)} className={className}>{label}</button>
    <ComfyUIWorkflowDialog
      open={open}
      workspace={workspace}
      onClose={() => setOpen(false)}
      onJobAccepted={() => { void reconnectJobs().catch(() => undefined) }}
    />
  </>
}

export function ComfyUIWorkflowDialogHost() {
  const workspace = useStore(state => state.activeWorkspace)
  const reconnectJobs = useStore(state => state.reconnectJobs)
  const [launch, setLaunch] = useState<WorkflowLaunch | null>(null)
  const requestId = useRef(0)

  useEffect(() => {
    const onLaunch = (event: Event) => {
      const detail = (event as CustomEvent<api.ComfyUIWorkflowLaunchRequest>).detail
      if (!detail || typeof detail.workflowId !== 'string' || !isScalarInputRecord(detail.inputs)) return
      requestId.current += 1
      setLaunch({ ...detail, requestId: requestId.current })
    }
    window.addEventListener(api.COMFYUI_OPEN_WORKFLOW_EVENT, onLaunch)
    return () => window.removeEventListener(api.COMFYUI_OPEN_WORKFLOW_EVENT, onLaunch)
  }, [])

  return <ComfyUIWorkflowDialog
    open={Boolean(launch)}
    launchRequest={launch}
    workspace={workspace}
    onClose={() => setLaunch(null)}
    onJobAccepted={() => { void reconnectJobs().catch(() => undefined) }}
  />
}

export function ComfyUIWorkflowDialog({
  open,
  workspace,
  onClose,
  onJobAccepted,
  launchRequest = null,
}: {
  open: boolean
  workspace: string
  onClose: () => void
  onJobAccepted?: () => void
  launchRequest?: WorkflowLaunch | null
}) {
  const [workflows, setWorkflows] = useState<api.ComfyUIWorkflow[]>([])
  const [templates, setTemplates] = useState<api.ComfyUIWorkflow[]>([])
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [values, setValues] = useState<Record<string, api.ComfyUIScalar>>({})
  const [validation, setValidation] = useState<api.ComfyUIWorkflowValidation | null>(null)
  const [validationFor, setValidationFor] = useState<string | null>(null)
  const [workflowOptions, setWorkflowOptions] = useState<api.ComfyUIWorkflowOptions | null>(null)
  const [optionsLoading, setOptionsLoading] = useState(false)
  const [optionsError, setOptionsError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [validating, setValidating] = useState(false)
  const [saving, setSaving] = useState(false)
  const [deleting, setDeleting] = useState(false)
  const [running, setRunning] = useState(false)
  const [uploading, setUploading] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [importName, setImportName] = useState('')
  const [importKind, setImportKind] = useState<api.ComfyUIKind>('image')
  const [workflowJson, setWorkflowJson] = useState('')
  const [bindingsJson, setBindingsJson] = useState(STARTER_BINDINGS)
  const [importError, setImportError] = useState<string | null>(null)
  const [ownedRun, setOwnedRun] = useState<OwnedRun | null>(null)
  const [checkingReceipt, setCheckingReceipt] = useState(false)
  const dialogRef = useRef<HTMLElement>(null)
  const closeRef = useRef<HTMLButtonElement>(null)
  const appliedLaunchRequest = useRef<number | null>(null)

  const allWorkflows = useMemo(() => [
    ...templates.map(item => ({ ...item, template: true })),
    ...workflows,
  ], [templates, workflows])
  const selected = allWorkflows.find(item => item.id === selectedId) ?? null
  const selectedRef = useRef(selected)
  selectedRef.current = selected
  const launchRequestRef = useRef(launchRequest)
  launchRequestRef.current = launchRequest
  const collected = useMemo(() => selected ? collectInputs(selected.bindings, values) : { inputs: {}, missing: [], invalid: [] }, [selected, values])
  const currentValidationKey = selected ? `${selected.id}:${JSON.stringify(collected.inputs)}` : null
  const currentValidation = validationFor === currentValidationKey ? validation : null
  const validationReady = currentValidation?.valid === true

  const acceptReceipt = (run: OwnedRun, receipt: api.ComfyUIJobReceipt) => {
    const next: OwnedRun = {
      ...run,
      job_id: receipt.job_id ?? run.job_id ?? null,
      status: responseStatus(receipt),
      detail: receipt.detail ?? receipt.error ?? null,
    }
    setOwnedRun(next)
    storeOwnedRun(next)
    if (next.job_id) onJobAccepted?.()
  }

  const reconcileReceipt = async (run: OwnedRun) => {
    setCheckingReceipt(true)
    try {
      const receipt = await api.getComfyUIJobBySubmission(run.client_submission_id)
      acceptReceipt(run, receipt)
    } catch (reason) {
      const next = { ...run, status: 'outcome unknown', detail: reason instanceof Error ? reason.message : 'Receipt is not available yet.' }
      setOwnedRun(next)
      storeOwnedRun(next)
    } finally { setCheckingReceipt(false) }
  }

  const resumeMonitoring = async (run: OwnedRun) => {
    if (!/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(run.client_submission_id)) {
      setError('This saved submission ID is invalid. Create a new request explicitly.')
      return
    }
    setCheckingReceipt(true)
    setError(null)
    try {
      const receipt = await api.resumeComfyUIJobBySubmission(run.client_submission_id)
      acceptReceipt(run, receipt)
    } catch (reason) {
      const next = { ...run, status: 'outcome unknown', detail: reason instanceof Error ? reason.message : 'Could not resume monitoring for this owned job.' }
      setOwnedRun(next)
      storeOwnedRun(next)
    } finally { setCheckingReceipt(false) }
  }

  useEffect(() => {
    if (!open) return
    let cancelled = false
    const load = async () => {
      setLoading(true); setError(null)
      try {
        const result = await api.fetchComfyUIWorkflows()
        if (cancelled) return
        setWorkflows(result.workflows)
        setTemplates(result.templates)
        const all = [...result.templates, ...result.workflows]
        const initialWorkflow = (launchRequest && all.find(item => item.id === launchRequest.workflowId))
          || result.templates[0]
          || result.workflows[0]
          || null
        setSelectedId(initialWorkflow?.id ?? null)
        setValues(initialWorkflow
          ? { ...initialValues(initialWorkflow.bindings), ...(launchRequest?.workflowId === initialWorkflow.id ? launchRequest.inputs : {}) }
          : {})
      } catch (reason) {
        if (!cancelled) setError(reason instanceof Error ? reason.message : 'Could not load ComfyUI workflows.')
      } finally { if (!cancelled) setLoading(false) }
    }
    void load()
    const savedRun = readOwnedRun()
    if (savedRun) {
      setOwnedRun(savedRun)
      if (savedRun.status === 'submitting' || savedRun.status === 'outcome unknown' || !savedRun.job_id) void reconcileReceipt(savedRun)
    }
    const previousFocus = document.activeElement instanceof HTMLElement ? document.activeElement : null
    const focusableSelector = 'a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])'
    const raf = window.requestAnimationFrame(() => (closeRef.current || dialogRef.current)?.focus())
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') { event.stopPropagation(); onClose(); return }
      if (event.key !== 'Tab') return
      const dialog = dialogRef.current
      if (!dialog) return
      const items = Array.from(dialog.querySelectorAll<HTMLElement>(focusableSelector))
      if (items.length === 0) { event.preventDefault(); dialog.focus(); return }
      const first = items[0]
      const last = items[items.length - 1]
      if (event.shiftKey && (document.activeElement === first || !dialog.contains(document.activeElement))) {
        event.preventDefault(); last.focus()
      } else if (!event.shiftKey && (document.activeElement === last || !dialog.contains(document.activeElement))) {
        event.preventDefault(); first.focus()
      }
    }
    document.addEventListener('keydown', onKeyDown, true)
    return () => {
      cancelled = true
      window.cancelAnimationFrame(raf)
      document.removeEventListener('keydown', onKeyDown, true)
      if (previousFocus?.isConnected) previousFocus.focus()
    }
    // Receipt recovery is intentionally once per dialog opening.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open])

  useEffect(() => {
    if (!open) return
    const refresh = async () => {
      try {
        const result = await api.fetchComfyUIWorkflows()
        setWorkflows(result.workflows)
        setTemplates(result.templates)
      } catch (reason) {
        setError(reason instanceof Error ? reason.message : 'Could not refresh ComfyUI workflows.')
      }
    }
    window.addEventListener(api.COMFYUI_STATE_CHANGED_EVENT, refresh)
    return () => window.removeEventListener(api.COMFYUI_STATE_CHANGED_EVENT, refresh)
  }, [open])

  useEffect(() => {
    if (!open || !launchRequest) return
    const workflow = allWorkflows.find(item => item.id === launchRequest.workflowId)
    if (!workflow) {
      if (!loading && allWorkflows.length > 0) setError('The saved ComfyUI workflow is no longer available. Choose a workflow from the library.')
      return
    }
    appliedLaunchRequest.current = launchRequest.requestId
    setSelectedId(workflow.id)
    setValues({ ...initialValues(workflow.bindings), ...launchRequest.inputs })
    setValidation(null)
    setValidationFor(null)
    setError(null)
  }, [open, launchRequest, allWorkflows, loading])

  useEffect(() => {
    const currentSelected = selectedRef.current
    if (!open || !currentSelected) return
    setValidation(null)
    setValidationFor(null)
    const currentLaunchRequest = launchRequestRef.current
    if (currentLaunchRequest && currentLaunchRequest.requestId === appliedLaunchRequest.current) return
    setValues(initialValues(currentSelected.bindings))
  }, [open, selected?.id])

  useEffect(() => {
    if (!open || !selectedId) {
      setWorkflowOptions(null)
      setOptionsLoading(false)
      setOptionsError(null)
      return
    }
    const controller = new AbortController()
    setWorkflowOptions(null)
    setOptionsLoading(true)
    setOptionsError(null)
    api.fetchComfyUIWorkflowOptions(selectedId, controller.signal)
      .then(result => {
        if (!controller.signal.aborted) setWorkflowOptions(result)
      })
      .catch(reason => {
        if (!controller.signal.aborted) {
          setOptionsError(reason instanceof Error ? reason.message : 'Server choices are unavailable; you can enter values manually.')
        }
      })
      .finally(() => {
        if (!controller.signal.aborted) setOptionsLoading(false)
      })
    return () => controller.abort()
  }, [open, selectedId])

  useEffect(() => {
    if (!open || !selectedId || !selected) return
    let cancelled = false
    const requestKey = `${selectedId}:${JSON.stringify(collected.inputs)}`
    setValidation(null)
    setValidationFor(null)
    setValidating(true)
    const timer = window.setTimeout(() => {
      api.validateComfyUIWorkflow(selectedId, collected.inputs).then(result => {
        if (cancelled) return
        setValidation(result)
        setValidationFor(requestKey)
      }).catch(reason => {
        if (!cancelled) setError(reason instanceof Error ? reason.message : 'Could not validate this workflow.')
      }).finally(() => { if (!cancelled) setValidating(false) })
    }, 250)
    return () => { cancelled = true; window.clearTimeout(timer) }
  }, [open, selectedId, selected, collected.inputs])

  const selectWorkflow = (workflow: api.ComfyUIWorkflow) => {
    setSelectedId(workflow.id)
    setValues(initialValues(workflow.bindings))
    setError(null)
    setValidation(null)
    setValidationFor(null)
  }

  const refreshValidation = async () => {
    if (!selected) return
    const workflowId = selected.id
    const requestKey = `${workflowId}:${JSON.stringify(collected.inputs)}`
    setValidating(true); setError(null)
    try {
      const result = await api.validateComfyUIWorkflow(workflowId, collected.inputs)
      setValidation(result); setValidationFor(requestKey)
    } catch (reason) { setError(reason instanceof Error ? reason.message : 'Could not validate this workflow.') }
    finally { setValidating(false) }
  }

  const loadImportFile = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    event.target.value = ''
    if (!file) return
    try { setWorkflowJson(await file.text()); setImportError(null) }
    catch { setImportError('Could not read that workflow file.') }
  }

  const startFromTemplate = (workflow: api.ComfyUIWorkflow) => {
    setImportName(`${workflow.name} copy`)
    setImportKind(workflow.kind)
    setWorkflowJson(JSON.stringify(workflow.workflow, null, 2))
    setBindingsJson(JSON.stringify(workflow.bindings, null, 2))
    setImportError(null)
  }

  const importWorkflow = async () => {
    setImportError(null)
    let graph: Record<string, unknown>
    let bindings: api.ComfyUIBinding[]
    try {
      const parsed = JSON.parse(workflowJson) as unknown
      if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) throw new Error('Workflow JSON must be an API-format object.')
      graph = parsed as Record<string, unknown>
    } catch (reason) {
      setImportError(reason instanceof Error ? reason.message : 'Workflow JSON is not valid.')
      return
    }
    try {
      const parsed = JSON.parse(bindingsJson) as unknown
      if (!Array.isArray(parsed) || parsed.some(item => !item || typeof item !== 'object' || Array.isArray(item))) {
        throw new Error('Bindings must be a JSON array of binding objects.')
      }
      bindings = parsed as api.ComfyUIBinding[]
      for (const binding of bindings) {
        if (!binding.name || !binding.node_id || !binding.input || !['string', 'integer', 'number', 'boolean', 'choice', 'image', 'video', 'audio'].includes(binding.type)) {
          throw new Error('Each binding needs name, node_id, input, and a supported type.')
        }
      }
    } catch (reason) {
      setImportError(reason instanceof Error ? reason.message : 'Bindings JSON is not valid.')
      return
    }
    if (!importName.trim()) { setImportError('Enter a workflow name.'); return }
    setSaving(true)
    try {
      const created = await api.createComfyUIWorkflow({ name: importName.trim(), kind: importKind, workflow: graph, bindings })
      setWorkflows(current => [created, ...current.filter(item => item.id !== created.id)])
      setSelectedId(created.id)
      setValues(initialValues(created.bindings))
      setValidation(null); setValidationFor(null)
      setError(null)
      setImportName('')
      // Saving/importing a graph only adds it to the library. Generation is a separate user action.
    } catch (reason) {
      setImportError(reason instanceof Error ? reason.message : 'Could not import this workflow.')
    } finally { setSaving(false) }
  }

  const deleteSelected = async () => {
    if (!selected || selected.template || deleting) return
    setDeleting(true); setError(null)
    try {
      await api.deleteComfyUIWorkflow(selected.id)
      const remaining = templates[0] ?? workflows.find(item => item.id !== selected.id) ?? null
      setWorkflows(current => current.filter(item => item.id !== selected.id))
      setSelectedId(remaining?.id ?? null)
      setValues(remaining ? initialValues(remaining.bindings) : {})
    } catch (reason) { setError(reason instanceof Error ? reason.message : 'Could not delete this workflow.') }
    finally { setDeleting(false) }
  }

  const uploadBinding = async (binding: api.ComfyUIBinding, file: File) => {
    setUploading(binding.name); setError(null)
    try {
      const result = binding.type === 'audio' ? await uploadAudio(file) : await uploadImage(file)
      setValues(current => ({ ...current, [binding.name]: result.path }))
    } catch (reason) { setError(reason instanceof Error ? reason.message : `Could not upload ${binding.label || binding.name}.`) }
    finally { setUploading(null) }
  }

  const handleGenerate = async () => {
    if (!selected || !validationReady || collected.missing.length || collected.invalid.length || running || ownedRun?.status === 'submitting' || ownedRun?.status === 'outcome unknown') return
    const run: OwnedRun = {
      client_submission_id: newUuid(),
      workflow_id: selected.id,
      status: 'submitting',
    }
    setOwnedRun(run)
    storeOwnedRun(run)
    setRunning(true); setError(null)
    try {
      const receipt = await api.runComfyUIWorkflow(selected.id, {
        client_submission_id: run.client_submission_id,
        inputs: collected.inputs,
        workspace,
      })
      acceptReceipt(run, receipt)
    } catch (reason) {
      // The POST may have reached ComfyUI even when its response was lost. Resolve by GET only.
      const unknown = { ...run, status: 'checking receipt', detail: reason instanceof Error ? reason.message : 'The submission response was lost.' }
      setOwnedRun(unknown); storeOwnedRun(unknown)
      await reconcileReceipt(unknown)
    } finally { setRunning(false) }
  }

  if (!open) return null

  return <div className="fixed inset-0 z-[90] flex items-center justify-center bg-black/55 p-2 sm:p-4" onMouseDown={event => { if (event.target === event.currentTarget) onClose() }}>
    <section ref={dialogRef} role="dialog" aria-modal="true" aria-labelledby="comfyui-workflow-title" tabIndex={-1} className="flex max-h-[96vh] w-full max-w-6xl flex-col overflow-hidden rounded-2xl border border-border bg-bg-primary shadow-2xl sm:max-h-[92vh]">
      <header className="flex shrink-0 items-center gap-2 border-b border-border px-3 py-2.5 sm:px-4">
        <FileJson2 size={16} className="shrink-0 text-accent-blue" />
        <div className="min-w-0 flex-1">
          <h2 id="comfyui-workflow-title" className="text-sm font-semibold text-text-primary">ComfyUI workflow library</h2>
          <p className="mt-0.5 text-[9px] text-text-muted">Outputs save to <span className="font-medium text-text-secondary">{workspace || 'default'}</span></p>
        </div>
        <button ref={closeRef} type="button" onClick={onClose} aria-label="Close ComfyUI workflow library" className="rounded-lg border border-border p-1.5 text-text-secondary hover:text-text-primary"><X size={14} /></button>
      </header>

      {error && <div role="alert" className="shrink-0 border-b border-red-500/30 bg-red-500/10 px-3 py-2 text-[10px] text-chip-red">{error}</div>}
      {ownedRun && <div role="status" className={`flex shrink-0 flex-wrap items-center gap-2 border-b px-3 py-2 text-[10px] ${ownedRun.status === 'outcome unknown' ? 'border-amber-500/30 bg-amber-500/10' : 'border-border bg-bg-secondary'}`}>
        {ownedRun.status === 'outcome unknown' ? <AlertTriangle size={13} className="text-indicator-warning" /> : ownedRun.job_id ? <Check size={13} className="text-indicator-success" /> : <Loader2 size={13} className="animate-spin text-accent-blue" />}
        <span className="min-w-0 flex-1 text-text-primary">Owned ComfyUI job: {ownedRun.job_id || ownedRun.status}</span>
        {ownedRun.job_id && <code className="max-w-full break-all font-mono text-text-secondary">{ownedRun.job_id}</code>}
        {(ownedRun.status === 'outcome unknown' || ownedRun.status === 'uncertain') && <>
          <button type="button" onClick={() => void reconcileReceipt(ownedRun)} disabled={checkingReceipt} className="rounded border border-border px-2 py-1 text-text-secondary hover:text-text-primary disabled:opacity-50">{checkingReceipt ? 'Checking…' : 'Check receipt'}</button>
          <button type="button" onClick={() => void resumeMonitoring(ownedRun)} disabled={checkingReceipt} className="rounded border border-border px-2 py-1 text-text-secondary hover:text-text-primary disabled:opacity-50">{checkingReceipt ? 'Resuming…' : 'Resume monitoring'}</button>
        </>}
        {ownedRun.status === 'prepared' && <span className="basis-full text-text-secondary">This saved intent was never dispatched and cannot be replayed. Cancel it, then explicitly create a new request.</span>}
        {ownedRun.detail && (ownedRun.status === 'outcome unknown' || ownedRun.status === 'uncertain') && <span className="basis-full text-text-secondary">{ownedRun.detail}</span>}
      </div>}

      <div className="grid min-h-0 flex-1 grid-cols-1 md:grid-cols-[minmax(190px,0.7fr)_minmax(0,1.6fr)]">
        <aside className="min-h-0 overflow-y-auto border-b border-border p-3 md:border-b-0 md:border-r">
          <div className="mb-2 flex items-center justify-between gap-2">
            <h3 className="text-[11px] font-semibold text-text-primary">Workflows and templates</h3>
            {loading && <Loader2 size={12} className="animate-spin text-text-muted" />}
          </div>
          {templates.length > 0 && <div className="mb-3 space-y-1">
            <p className="px-1 text-[9px] uppercase tracking-wide text-text-muted">Built-in templates</p>
            {templates.map(workflow => <WorkflowRow key={workflow.id} workflow={workflow} selected={selectedId === workflow.id} onClick={() => selectWorkflow(workflow)} />)}
          </div>}
          <div className="space-y-1">
            <p className="px-1 text-[9px] uppercase tracking-wide text-text-muted">Your workflows</p>
            {workflows.length ? workflows.map(workflow => <WorkflowRow key={workflow.id} workflow={workflow} selected={selectedId === workflow.id} onClick={() => selectWorkflow(workflow)} />)
              : <p className="px-1 py-2 text-[9px] text-text-muted">Import an API-format graph to add a workflow.</p>}
          </div>

          <div className="mt-4 border-t border-border pt-3">
            <h3 className="mb-1 text-[10px] font-semibold text-text-primary">Import API workflow</h3>
            <p className="mb-2 text-[9px] leading-relaxed text-text-muted">Export a ComfyUI API-format JSON graph. Bindings connect named form fields to node inputs.</p>
            <label className="mb-1 block text-[9px] text-text-secondary">Name
              <input value={importName} onChange={event => setImportName(event.target.value)} className="mt-1 w-full rounded-md border border-border bg-bg-primary px-2 py-1.5 text-[10px] text-text-primary outline-none focus:border-accent-blue" />
            </label>
            <label className="mb-1.5 block text-[9px] text-text-secondary">Output kind
              <select value={importKind} onChange={event => setImportKind(event.target.value as api.ComfyUIKind)} className="mt-1 w-full rounded-md border border-border bg-bg-primary px-2 py-1.5 text-[10px] text-text-primary">
                <option value="image">Image</option><option value="video">Video</option><option value="audio">Audio</option>
              </select>
            </label>
            <label className="mb-1 block text-[9px] text-text-secondary">Workflow JSON
              <textarea aria-label="Workflow JSON" value={workflowJson} onChange={event => setWorkflowJson(event.target.value)} rows={5} spellCheck={false} placeholder={'{"6":{"class_type":"CLIPTextEncode","inputs":{"text":"..."}}}'} className="mt-1 w-full resize-y rounded-md border border-border bg-bg-primary px-2 py-1.5 font-mono text-[9px] text-text-primary outline-none placeholder:text-text-muted focus:border-accent-blue" />
            </label>
            <label className="mb-1 block text-[9px] text-text-secondary">Input bindings JSON
              <textarea aria-label="Input bindings JSON" value={bindingsJson} onChange={event => setBindingsJson(event.target.value)} rows={5} spellCheck={false} className="mt-1 w-full resize-y rounded-md border border-border bg-bg-primary px-2 py-1.5 font-mono text-[9px] text-text-primary outline-none focus:border-accent-blue" />
            </label>
            <p className="mb-2 break-words text-[8px] text-text-muted">Fields: name, node_id, input, type, required, default, options, min, max. No graph editor is used.</p>
            <div className="flex flex-wrap gap-1.5">
              <label className="inline-flex cursor-pointer items-center gap-1 rounded-md border border-border px-2 py-1.5 text-[9px] text-text-secondary hover:text-text-primary"><Upload size={10} /> Choose JSON<input type="file" accept="application/json,.json" className="sr-only" onChange={event => void loadImportFile(event)} /></label>
              <button type="button" onClick={() => void importWorkflow()} disabled={saving || !workflowJson.trim() || !importName.trim()} className="rounded-md bg-accent-blue px-2.5 py-1.5 text-[9px] font-medium text-white hover:brightness-110 disabled:opacity-45">{saving ? 'Saving…' : 'Import workflow'}</button>
            </div>
            {importError && <p role="alert" className="mt-1.5 break-words text-[9px] text-chip-red">{importError}</p>}
          </div>
        </aside>

        <main className="min-h-0 overflow-y-auto p-3 sm:p-4">
          {!selected ? <div className="flex min-h-48 flex-col items-center justify-center text-center text-text-muted"><FileJson2 size={24} /><p className="mt-2 max-w-xs text-[10px]">Choose a workflow or import an API-format ComfyUI JSON graph.</p></div> : <div className="mx-auto max-w-3xl space-y-3">
            <div className="flex flex-wrap items-start gap-2">
              <div className="min-w-0 flex-1">
                <div className="flex flex-wrap items-center gap-1.5"><h3 className="text-sm font-semibold text-text-primary">{selected.name}</h3>{selected.template && <span className="rounded-full border border-border px-1.5 py-0.5 text-[8px] text-text-muted">Template</span>}<span className="rounded-full bg-bg-tertiary px-1.5 py-0.5 text-[8px] capitalize text-text-secondary">{selected.kind}</span></div>
                {selected.description && <p className="mt-1 text-[10px] leading-relaxed text-text-secondary">{selected.description}</p>}
              </div>
              <button type="button" onClick={() => void refreshValidation()} disabled={validating} className="inline-flex items-center gap-1 rounded-md border border-border px-2 py-1.5 text-[9px] text-text-secondary hover:text-text-primary disabled:opacity-50">{validating ? <Loader2 size={10} className="animate-spin" /> : <RefreshCw size={10} />} Recheck</button>
              {selected.template && <button type="button" onClick={() => startFromTemplate(selected)} className="inline-flex items-center gap-1 rounded-md border border-border px-2 py-1.5 text-[9px] text-text-secondary hover:text-text-primary">Use as starter</button>}
              {!selected.template && <button type="button" onClick={() => void deleteSelected()} disabled={deleting} title="Delete workflow" aria-label={`Delete ${selected.name}`} className="rounded-md border border-border p-1.5 text-text-muted hover:text-red-400 disabled:opacity-50">{deleting ? <Loader2 size={11} className="animate-spin" /> : <Trash2 size={11} />}</button>}
            </div>

            <section className="rounded-lg border border-border bg-bg-secondary p-2.5">
              <h4 className="mb-1.5 text-[10px] font-semibold text-text-primary">Preflight</h4>
              {validating && <p role="status" className="flex items-center gap-1.5 text-[9px] text-text-muted"><Loader2 size={11} className="animate-spin" /> Checking nodes and models…</p>}
              {!validating && !currentValidation && <p className="text-[9px] text-text-muted">Preflight has not completed.</p>}
              {!validating && currentValidation && <>
                <p className={`flex items-center gap-1.5 text-[9px] ${currentValidation.valid ? 'text-indicator-success' : 'text-indicator-warning'}`}>{currentValidation.valid ? <Check size={11} /> : <AlertTriangle size={11} />}{currentValidation.valid ? 'Ready to run' : 'Needs attention'}</p>
                {currentValidation.missing_nodes.length > 0 && <IssueList title="Missing nodes" items={currentValidation.missing_nodes.map(stringifyIssue)} />}
                {currentValidation.missing_models.length > 0 && <IssueList title="Missing models" items={currentValidation.missing_models.map(stringifyIssue)} />}
                {currentValidation.errors.length > 0 && <IssueList title="Validation details" items={currentValidation.errors.map(stringifyIssue)} />}
              </>}
            </section>

            {selected.bindings.length > 0 ? <section className="space-y-2 rounded-lg border border-border bg-bg-secondary p-2.5">
              <div><h4 className="text-[10px] font-semibold text-text-primary">Inputs</h4><p className="mt-0.5 text-[9px] text-text-muted">Required fields are marked. Values are checked in preflight and sent only when you press Generate.</p></div>
              {optionsLoading && <p role="status" className="text-[9px] text-text-muted">Loading available model and input choices·</p>}
              {optionsError && <p role="status" className="text-[9px] text-indicator-warning">Could not load server choices. You can still enter values manually. {optionsError}</p>}
              {selected.bindings.map(binding => <BindingField
                key={`${selected.id}:${binding.name}`}
                binding={binding}
                choices={workflowOptions?.choices[binding.name] ?? []}
                value={values[binding.name]}
                uploading={uploading === binding.name}
                onChange={value => setValues(current => ({ ...current, [binding.name]: value }))}
                onClear={() => setValues(current => { const next = { ...current }; delete next[binding.name]; return next })}
                onFile={file => void uploadBinding(binding, file)}
              />)}
            </section> : <p className="rounded-lg border border-border bg-bg-secondary p-2.5 text-[9px] text-text-muted">This workflow has no input bindings.</p>}

            {collected.missing.length > 0 && <p role="alert" className="text-[9px] text-indicator-warning">Required inputs: {collected.missing.join(', ')}</p>}
            {collected.invalid.length > 0 && <p role="alert" className="text-[9px] text-indicator-warning">{collected.invalid.join('; ')}</p>}
            <div className="flex flex-wrap items-center gap-2 border-t border-border pt-3">
              <button type="button" onClick={() => void handleGenerate()} disabled={!validationReady || collected.missing.length > 0 || collected.invalid.length > 0 || running || ownedRun?.status === 'submitting' || ownedRun?.status === 'outcome unknown'} className="inline-flex items-center gap-1.5 rounded-lg bg-accent-blue px-3 py-2 text-[10px] font-medium text-white hover:brightness-110 disabled:cursor-not-allowed disabled:opacity-45">
                {running ? <Loader2 size={12} className="animate-spin" /> : <Play size={12} />} Generate
              </button>
              <span className="text-[9px] text-text-muted">Current workspace: <span className="text-text-secondary">{workspace || 'default'}</span></span>
            </div>
          </div>}
        </main>
      </div>
    </section>
  </div>
}

function WorkflowRow({ workflow, selected, onClick }: { workflow: api.ComfyUIWorkflow; selected: boolean; onClick: () => void }) {
  return <button type="button" onClick={onClick} className={`w-full rounded-lg border px-2.5 py-2 text-left transition-colors ${selected ? 'border-accent-blue/70 bg-accent-blue/10' : 'border-border bg-bg-secondary hover:border-border-light'}`}>
    <span className="block truncate text-[10px] font-medium text-text-primary">{workflow.name}</span>
    <span className="mt-0.5 block text-[8px] capitalize text-text-muted">{workflow.kind}{workflow.description ? ` · ${workflow.description}` : ''}</span>
  </button>
}

function IssueList({ title, items }: { title: string; items: string[] }) {
  return <div className="mt-1.5 text-[9px] text-text-secondary"><div className="font-medium text-text-primary">{title}</div><ul className="mt-0.5 list-inside list-disc space-y-0.5">{items.map((item, index) => <li key={`${index}:${item}`} className="break-words">{item}</li>)}</ul></div>
}

function BindingField({ binding, choices = [], value, uploading, onChange, onClear, onFile }: {
  binding: api.ComfyUIBinding
  choices?: api.ComfyUIScalar[]
  value: api.ComfyUIScalar | undefined
  uploading: boolean
  onChange: (value: api.ComfyUIScalar) => void
  onClear: () => void
  onFile: (file: File) => void
}) {
  const id = 'comfyui-input-' + binding.name.replace(/[^a-zA-Z0-9_-]/g, '-')
  const choicesId = id + '-choices'
  const label = binding.label || binding.name
  const common = 'w-full rounded-md border border-border bg-bg-primary px-2 py-1.5 text-[10px] text-text-primary outline-none focus:border-accent-blue'
  const isMedia = binding.type === 'image' || binding.type === 'video' || binding.type === 'audio'
  const liveChoices = (binding.options ?? []).concat(choices).filter((option, index, options) => options.indexOf(option) === index)
  const staleChoice = binding.type === 'choice' && value !== undefined && !liveChoices.includes(value)
  const allChoices = staleChoice ? [value, ...liveChoices] : liveChoices
  const updateText = (next: string) => {
    if (!next) { onClear(); return }
    const matchingChoice = choices.find(option => String(option) === next)
    onChange(matchingChoice ?? next)
  }
  return <div className="grid gap-1.5 sm:grid-cols-[minmax(110px,0.7fr)_minmax(0,1.3fr)] sm:items-start">
    <label htmlFor={id} className="pt-1 text-[9px] text-text-secondary">{label}{binding.required && <span className="ml-0.5 text-indicator-warning">*</span>}<span className="mt-0.5 block text-[8px] text-text-muted">{binding.name} · {binding.type}</span></label>
    <div className="min-w-0">
      {isMedia ? <div className="flex min-w-0 flex-wrap items-center gap-1.5">
        <label htmlFor={id} className="inline-flex cursor-pointer items-center gap-1 rounded-md border border-border bg-bg-primary px-2 py-1.5 text-[9px] text-text-secondary hover:text-text-primary">{uploading ? <Loader2 size={10} className="animate-spin" /> : <Upload size={10} />}{uploading ? 'Uploading…' : 'Upload ' + binding.type}<input id={id} type="file" accept={binding.type === 'image' ? 'image/*' : binding.type === 'video' ? 'video/*' : 'audio/*'} disabled={uploading} className="sr-only" onChange={event => { const file = event.target.files?.[0]; event.target.value = ''; if (file) onFile(file) }} /></label>
        {value !== undefined && <><span className="min-w-0 flex-1 break-all font-mono text-[8px] text-text-muted">{displayValue(value)}</span><button type="button" onClick={onClear} className="text-[8px] text-text-muted underline">Clear</button></>}
      </div> : binding.type === 'boolean' ? <label className="flex min-h-8 items-center gap-1.5 text-[10px] text-text-primary"><input id={id} type="checkbox" checked={Boolean(value)} onChange={event => onChange(event.target.checked)} /> Enabled</label>
        : binding.type === 'choice' ? <select id={id} value={value === undefined ? '' : JSON.stringify(value)} onChange={event => { try { onChange(JSON.parse(event.target.value) as api.ComfyUIScalar) } catch { onClear() } }} className={common}>
          <option value="">Choose…</option>{allChoices.map((option, index) => <option key={index + ':' + String(option)} value={JSON.stringify(option)}>{String(option)}{staleChoice && option === value ? ' … saved, no longer listed' : ''}</option>)}
        </select> : binding.type === 'string' && choices.length > 0
          ? <div className="space-y-1">
            <input id={id} list={choicesId} type="text" required={binding.required} value={displayValue(value)} onChange={event => updateText(event.target.value)} placeholder="Choose a server value or enter one" className={common} />
            <datalist id={choicesId}>{choices.map((option, index) => <option key={index + ':' + String(option)} value={String(option)} />)}</datalist>
            {choices.length > 0 && value !== undefined && !choices.some(option => String(option) === String(value)) && <p className="text-[8px] text-indicator-warning">Saved value is not in the server list; it is kept as a custom value.</p>}
            <p className="text-[8px] text-text-muted">ComfyUI server choices · custom values are allowed</p>
          </div> : <input id={id} type={binding.type === 'integer' || binding.type === 'number' ? 'number' : 'text'} step={binding.type === 'integer' ? 1 : binding.type === 'number' ? 'any' : undefined} min={binding.min} max={binding.max} required={binding.required} value={displayValue(value)} onChange={event => {
            const next = event.target.value
            if (!next) { onClear(); return }
            if (binding.type === 'integer' || binding.type === 'number') { if (Number.isFinite(Number(next))) onChange(Number(next)) }
            else onChange(next)
          }} className={common} />}
      {binding.min !== undefined || binding.max !== undefined ? <p className="mt-0.5 text-[8px] text-text-muted">{binding.min !== undefined ? 'Min ' + binding.min : ''}{binding.min !== undefined && binding.max !== undefined ? ' · ' : ''}{binding.max !== undefined ? 'Max ' + binding.max : ''}</p> : null}
    </div>
  </div>
}
