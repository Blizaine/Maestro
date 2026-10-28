import { useEffect, useMemo, useState } from 'react'
import { AlertTriangle, Check, Clock3, Loader2, RotateCw, WandSparkles, X } from 'lucide-react'
import type { EditorProject } from '../types'
import {
  editorAutoEditSignature,
  previewEditorAutoEdit,
  type EditorAutoAction,
  type EditorAutoProposal,
} from './editorAutoEdit'
import { useEditorStore } from './useEditorStore'
import { useEditorDialog } from './useEditorDialog'

type AutoEditMode = 'director' | 'assemble' | 'polish' | 'revise'
type DialogTab = 'plan' | 'history'

interface AutoEditCapabilities {
  director_available?: boolean
  writer_available: boolean
  writer_model?: string | null
  writer_provider?: string | null
  reason?: string | null
  content_basis?: string | null
}

interface ProposalSummary {
  proposal_id: string
  project_id: string
  workspace: string
  base_revision: number
  mode: string
  status: string
  summary: string
  created_at?: number | string
  updated_at?: number | string
}

interface ProposalResponse extends Omit<EditorAutoProposal, 'missing_media'> {
  missing_media: Array<string | { type: string; description: string }>
  created_at?: number | string
  updated_at?: number | string
  client_intent_id?: string
}

interface ReceiptRequest {
  proposalId: string
  projectId: string
  workspace: string
  revision: number
  signature: string
}

const inputClass = 'w-full rounded-md border border-border bg-bg-tertiary px-2.5 py-2 text-[12px] text-text-primary outline-none focus:border-accent-blue/60 disabled:opacity-45'
const modes: Array<{ id: AutoEditMode; label: string; hint: string }> = [
  { id: 'director', label: 'Director order', hint: 'Restore the shot order recorded by Director, keeping each source duration.' },
  { id: 'assemble', label: 'Assemble', hint: 'Build a first cut from the project’s available media.' },
  { id: 'polish', label: 'Polish', hint: 'Refine pacing and transitions in the current cut.' },
  { id: 'revise', label: 'Revise', hint: 'Make a focused change described in the brief.' },
]

function projectEndpoint(projectId: string): string {
  return `/api/v1/editor/auto-edit/proposals/${encodeURIComponent(projectId)}`
}

async function readJson<T>(response: Response, fallback: string): Promise<T> {
  const body = await response.json().catch(() => ({})) as { detail?: string; error?: string }
  if (!response.ok) throw new Error(body.detail || body.error || `${fallback} (${response.status})`)
  return body as T
}

function createIntentId(): string {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') return crypto.randomUUID()
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, character => {
    const random = Math.random() * 16 | 0
    return (character === 'x' ? random : (random & 0x3 | 0x8)).toString(16)
  })
}

function actionId(action: EditorAutoAction): string {
  return action.id
}

function formatTime(value: number | string | undefined): string {
  if (value === undefined) return 'Time unavailable'
  const numeric = typeof value === 'number' ? value : Number(value)
  const date = Number.isFinite(numeric)
    ? new Date(numeric > 1e12 ? numeric : numeric * 1000)
    : new Date(value)
  return Number.isNaN(date.getTime()) ? 'Time unavailable' : date.toLocaleString()
}

function describeAction(project: EditorProject, action: EditorAutoAction): string {
  const item = project.tracks.flatMap(track => track.items).find(candidate => candidate.id === action.item_id)
  const asset = action.asset_id ? project.assets[action.asset_id] : undefined
  const targetTrack = project.tracks.find(track => track.id === action.track_id)
  const target = item?.name || (asset ? `“${asset.name}”` : 'clip')
  switch (action.type) {
    case 'insert': return `Insert ${target} on ${targetTrack?.name || 'a track'} at ${(action.start ?? 0).toFixed(2)}s for ${(action.duration ?? 0).toFixed(2)}s.`
    case 'move': return `Move “${target}” to ${(action.start ?? 0).toFixed(2)}s${targetTrack ? ` on ${targetTrack.name}` : ''}.`
    case 'trim': return `Trim “${target}” to ${(action.duration ?? 0).toFixed(2)}s${action.source_in !== undefined ? ` from source ${(action.source_in).toFixed(2)}s` : ''}.`
    case 'remove': return `Remove “${target}” from the timeline; its media stays in the project.`
    case 'set-volume': return `Set “${target}” volume to ${(action.volume ?? 1).toFixed(2)}×.`
    case 'crossfade': return `Add a ${(action.seconds ?? 0).toFixed(2)}s crossfade before “${target}”.`
    case 'close-gap': return `Close the gap before “${target}”.`
  }
}

function asProposal(value: unknown): ProposalResponse {
  if (!value || typeof value !== 'object') throw new Error('The proposal response was not an object.')
  const raw = value as Record<string, unknown>
  const proposalId = typeof raw.proposal_id === 'string' ? raw.proposal_id : ''
  if (!proposalId || typeof raw.project_id !== 'string' || typeof raw.workspace !== 'string') {
    throw new Error('The proposal response is missing its project identity.')
  }
  if (typeof raw.base_revision !== 'number' || !Number.isFinite(raw.base_revision) || typeof raw.client_signature !== 'string') {
    throw new Error('The proposal response is missing its revision or content signature.')
  }
  if (typeof raw.mode !== 'string' || typeof raw.status !== 'string' || !Array.isArray(raw.actions)) {
    throw new Error('The proposal response is missing its mode, status, or action list.')
  }
  if (raw.actions.some(action => !action || typeof action !== 'object'
    || typeof (action as { id?: unknown }).id !== 'string' || !(action as { id: string }).id.trim())
    || new Set(raw.actions.map(action => (action as { id: string }).id)).size !== raw.actions.length) {
    throw new Error('The proposal response has missing or duplicate action IDs.')
  }
  return {
    ...(raw as unknown as ProposalResponse),
    proposal_id: proposalId,
    actions: raw.actions.map(action => Object.fromEntries(Object.entries(action as Record<string, unknown>).filter(([, entry]) => entry !== null))) as unknown as EditorAutoAction[],
    warnings: Array.isArray(raw.warnings) ? raw.warnings.filter((entry): entry is string => typeof entry === 'string') : [],
    missing_media: Array.isArray(raw.missing_media) ? raw.missing_media as ProposalResponse['missing_media'] : [],
    errors: Array.isArray(raw.errors) ? raw.errors.filter((entry): entry is string => typeof entry === 'string') : [],
  }
}

function sameReceipt(proposal: ProposalResponse, request: ReceiptRequest): boolean {
  return proposal.proposal_id === request.proposalId
    && proposal.project_id === request.projectId
    && proposal.workspace === request.workspace
    && proposal.base_revision === request.revision
    && proposal.client_signature === request.signature
}

function errorText(error: unknown, fallback: string): string {
  return error instanceof Error && error.message ? error.message : fallback
}

function proposalListUrl(projectId: string, workspace: string): string {
  const url = new URL('/api/v1/editor/auto-edit/proposals', window.location.origin)
  url.searchParams.set('workspace', workspace)
  url.searchParams.set('project_id', projectId)
  return url.toString()
}

function missingMediaText(value: ProposalResponse['missing_media'][number]): string {
  if (typeof value === 'string') return value
  const description = value.description || 'Suggested media is unavailable.'
  return value.type ? `${value.type}: ${description}` : description
}

function draftText(value: unknown): string {
  if (typeof value === 'string') return value
  if (value && typeof value === 'object' && typeof (value as { raw?: unknown }).raw === 'string') {
    const draft = value as { raw: string; truncated?: boolean }
    return `${draft.truncated ? '[Stored draft excerpt — exceeded the 64 KiB limit]\n' : ''}${draft.raw}`
  }
  try { return JSON.stringify(value, null, 2) } catch { return String(value) }
}

function changeDescription(value: unknown): string {
  if (typeof value === 'string') return value
  if (value && typeof value === 'object' && typeof (value as { description?: unknown }).description === 'string') {
    return (value as { description: string }).description
  }
  return 'Validated edit action.'
}

export function EditorAutoEditDialog({ open, onClose }: { open: boolean; onClose: () => void }) {
  const projectId = useEditorStore(state => state.project?.id)
  const workspace = useEditorStore(state => state.project?.workspace)
  const sessionKey = projectId && workspace ? `${projectId}\u0000${workspace}` : 'no-project'
  return <EditorAutoEditDialogSession key={sessionKey} open={open} onClose={onClose} />
}

function EditorAutoEditDialogSession({ open, onClose }: { open: boolean; onClose: () => void }) {
  const dialogRef = useEditorDialog(open, onClose)
  const project = useEditorStore(state => state.project)
  const saving = useEditorStore(state => state.saving)
  const saveProject = useEditorStore(state => state.saveProject)
  const applyAutoEditProposal = useEditorStore(state => state.applyAutoEditProposal)
  const [tab, setTab] = useState<DialogTab>('plan')
  const [mode, setMode] = useState<AutoEditMode>('director')
  const [brief, setBrief] = useState('')
  const [targetSeconds, setTargetSeconds] = useState('')
  const [capabilities, setCapabilities] = useState<AutoEditCapabilities | null>(null)
  const [capabilityLoading, setCapabilityLoading] = useState(false)
  const [capabilityError, setCapabilityError] = useState('')
  const [history, setHistory] = useState<ProposalSummary[]>([])
  const [historyLoading, setHistoryLoading] = useState(false)
  const [historyError, setHistoryError] = useState('')
  const [proposal, setProposal] = useState<ProposalResponse | null>(null)
  const [selectedActionIds, setSelectedActionIds] = useState<string[]>([])
  const [busy, setBusy] = useState<'plan' | 'load' | 'recover' | 'apply' | null>(null)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const [receiptRequest, setReceiptRequest] = useState<ReceiptRequest | null>(null)

  useEffect(() => {
    const projectId = project?.id
    const workspace = project?.workspace
    if (!open || !projectId || !workspace) return
    const controller = new AbortController()
    setCapabilityLoading(true)
    setCapabilityError('')
    setHistoryLoading(true)
    setHistoryError('')
    void fetch('/api/v1/editor/auto-edit/capabilities', { signal: controller.signal, cache: 'no-store' })
      .then(response => readJson<AutoEditCapabilities>(response, 'Could not check proposal writers'))
      .then(result => {
        if (!controller.signal.aborted) setCapabilities({
          director_available: result.director_available,
          writer_available: Boolean(result.writer_available),
          writer_model: result.writer_model,
          writer_provider: result.writer_provider,
          reason: result.reason,
          content_basis: result.content_basis,
        })
      })
      .catch(reason => {
        if (!controller.signal.aborted) {
          setCapabilities(null)
          setCapabilityError(errorText(reason, 'Writer availability could not be checked.'))
        }
      })
      .finally(() => { if (!controller.signal.aborted) setCapabilityLoading(false) })

    void fetch(proposalListUrl(projectId, workspace), { signal: controller.signal, cache: 'no-store' })
      .then(response => readJson<{ proposals?: ProposalSummary[] }>(response, 'Could not load proposal history'))
      .then(result => {
        if (!controller.signal.aborted) {
          const entries = Array.isArray(result.proposals) ? result.proposals : []
          setHistory(entries.filter(entry => entry.project_id === projectId && entry.workspace === workspace))
        }
      })
      .catch(reason => { if (!controller.signal.aborted) setHistoryError(errorText(reason, 'Could not load proposal history.')) })
      .finally(() => { if (!controller.signal.aborted) setHistoryLoading(false) })
    return () => controller.abort()
  }, [open, project?.id, project?.workspace])

  const activeProposal = proposal && project
    && proposal.project_id === project.id && proposal.workspace === project.workspace
    ? proposal
    : null
  const activeProposalActionIds = useMemo(() => activeProposal
    ? activeProposal.actions.map(actionId)
    : [], [activeProposal])
  const selectedActionIdsInProposalOrder = useMemo(
    () => activeProposalActionIds.filter(id => selectedActionIds.includes(id)),
    [activeProposalActionIds, selectedActionIds],
  )
  const applied = Boolean(activeProposal && project?.auto_edit_history?.some(entry => entry.proposal_id === activeProposal.proposal_id))
  const previewState = useMemo(() => {
    if (!project || !activeProposal || applied) return { changes: [] as unknown[], error: '' }
    if (selectedActionIdsInProposalOrder.length === 0) return { changes: [] as unknown[], error: 'Select at least one proposal action to preview.' }
    try {
      const result = previewEditorAutoEdit(project, activeProposal, selectedActionIdsInProposalOrder)
      return { changes: result.changes as unknown[], error: '' }
    } catch (reason) {
      return { changes: [] as unknown[], error: errorText(reason, 'This proposal cannot be previewed safely.') }
    }
  }, [activeProposal, applied, project, selectedActionIdsInProposalOrder])

  const canUseWriterModes = Boolean(capabilities?.writer_available)
  const modeAvailable = (candidate: AutoEditMode) => candidate === 'director'
    ? capabilities?.director_available !== false
    : canUseWriterModes
  const generateDisabled = !project || saving || busy !== null || !modeAvailable(mode)
    || (receiptRequest !== null) || (mode !== 'director' && !brief.trim())
    || (targetSeconds !== '' && (!Number.isFinite(Number(targetSeconds)) || Number(targetSeconds) < 0 || Number(targetSeconds) > 86400))

  const refreshHistory = async (target: EditorProject) => {
    setHistoryLoading(true)
    setHistoryError('')
    try {
      const result = await readJson<{ proposals?: ProposalSummary[] }>(await fetch(proposalListUrl(target.id, target.workspace), { cache: 'no-store' }), 'Could not load proposal history')
      const entries = Array.isArray(result.proposals) ? result.proposals : []
      if (useEditorStore.getState().project?.id === target.id && useEditorStore.getState().project?.workspace === target.workspace) {
        setHistory(entries.filter(entry => entry.project_id === target.id && entry.workspace === target.workspace))
      }
    } catch (reason) {
      setHistoryError(errorText(reason, 'Could not load proposal history.'))
    } finally { setHistoryLoading(false) }
  }

  const installProposal = (value: unknown, expectedId: string, target: EditorProject, expectedReceipt?: ReceiptRequest) => {
    const next = asProposal(value)
    if (next.proposal_id !== expectedId || next.project_id !== target.id || next.workspace !== target.workspace) {
      throw new Error('The server returned a proposal for a different request, project, or workspace.')
    }
    if (expectedReceipt && !sameReceipt(next, expectedReceipt)) {
      throw new Error('The proposal receipt does not match the revision and content that were submitted.')
    }
    if (useEditorStore.getState().project?.id !== target.id || useEditorStore.getState().project?.workspace !== target.workspace) {
      throw new Error('The active project changed while this proposal was loading. It was not selected.')
    }
    setProposal(next)
    setSelectedActionIds(next.actions.map(actionId))
    setReceiptRequest(null)
    setTab('plan')
    setError('')
    setNotice(next.status === 'ready' ? 'Proposal loaded for review. No timeline changes have been applied.' : `Proposal status: ${next.status}. Review its diagnostics before applying anything.`)
  }

  const fetchProposal = async (proposalId: string, target: EditorProject, expectedReceipt?: ReceiptRequest) => {
    const response = await fetch(projectEndpoint(proposalId), { cache: 'no-store' })
    const value = await readJson<unknown>(response, 'Could not retrieve this proposal')
    installProposal(value, proposalId, target, expectedReceipt)
  }

  const recoverReceipt = async (request: ReceiptRequest) => {
    const current = useEditorStore.getState().project
    if (!current || current.id !== request.projectId || current.workspace !== request.workspace) {
      setError('The active project changed. This receipt will not be applied to the new project.')
      return
    }
    setBusy('recover')
    setError('')
    try {
      await fetchProposal(request.proposalId, current, request)
      setNotice('Recovered the existing proposal by request ID. The plan was not submitted again.')
    } catch (reason) {
      setError(`${errorText(reason, 'Could not recover the proposal receipt.')} You may check this request ID again; the plan will not be resubmitted.`)
    } finally { setBusy(null) }
  }

  const generateProposal = async () => {
    if (!project || generateDisabled) return
    setBusy('plan')
    setError('')
    setNotice('')
    setProposal(null)
    setSelectedActionIds([])
    setReceiptRequest(null)
    const projectId = project.id
    const workspace = project.workspace
    let request: ReceiptRequest | null = null
    let requestMayHaveReceipt = false
    try {
      const saved = await saveProject()
      const savedState = useEditorStore.getState()
      if (!saved || savedState.project?.id !== projectId || savedState.project.workspace !== workspace || savedState.dirty) {
        throw new Error(savedState.error || 'Save the current project before preparing a proposal.')
      }
      if (savedState.saving) throw new Error('Wait for the project save to finish before preparing a proposal.')
      const current = savedState.project
      if (!Number.isFinite(current.revision)) throw new Error('The saved project has no server revision. Save the project and try again.')
      const signature = editorAutoEditSignature(current)
      const proposalId = createIntentId()
      request = {
        proposalId,
        projectId,
        workspace,
        revision: current.revision as number,
        signature,
      }
      if (!useEditorStore.getState().project || useEditorStore.getState().project?.id !== projectId) {
        throw new Error('The active project changed during the save. Start again from the new project.')
      }
      const body: Record<string, unknown> = {
        proposal_id: proposalId,
        workspace,
        project_id: projectId,
        revision: current.revision,
        client_signature: signature,
        mode,
        brief: mode === 'director' ? '' : brief.trim(),
        target_seconds: mode === 'director' || !targetSeconds.trim() ? null : Number(targetSeconds),
      }
      requestMayHaveReceipt = true
      const response = await fetch('/api/v1/editor/auto-edit/plan', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })
      if (!response.ok) {
        const failure = await response.json().catch(() => ({})) as { detail?: string; error?: string }
        const message = failure.detail || failure.error || `The proposal request failed (${response.status}).`
        if (response.status >= 500) throw Object.assign(new Error(message), { receiptMayExist: true })
        requestMayHaveReceipt = false
        throw Object.assign(new Error(message), { receiptMayExist: false })
      }
      const result = await response.json() as { proposal?: unknown } | unknown
      const record = result && typeof result === 'object' && 'proposal' in result
        ? (result as { proposal: unknown }).proposal
        : result
      const activeNow = useEditorStore.getState().project
      if (!activeNow || activeNow.id !== projectId || activeNow.workspace !== workspace) {
        setNotice('The proposal was prepared for the previous project. Switch back to review it; nothing was applied to the new project.')
        void refreshHistory(current)
        return
      }
      try {
        installProposal(record, proposalId, current, request)
        requestMayHaveReceipt = false
        setNotice('Proposal ready for review. No timeline changes have been applied.')
        void refreshHistory(current)
      } catch {
        // A successful response with a missing/incomplete record still has an
        // unambiguous request ID, so retrieve that exact receipt instead.
        setReceiptRequest(request)
        await recoverReceipt(request)
        void refreshHistory(current)
      }
    } catch (reason) {
      const mayHaveReceipt = Boolean((reason as { receiptMayExist?: boolean })?.receiptMayExist)
        || requestMayHaveReceipt
      if (mayHaveReceipt && request) {
        const activeNow = useEditorStore.getState().project
        if (!activeNow || activeNow.id !== request.projectId || activeNow.workspace !== request.workspace) {
          setNotice('The proposal request belonged to the previous project. It was not submitted again or applied to the active project.')
          return
        }
        setReceiptRequest(request)
        setNotice('The request receipt was not confirmed. Checking the same request ID; it will not be submitted again.')
        await recoverReceipt(request)
        void refreshHistory(project)
      } else {
        setError(errorText(reason, 'Could not prepare a proposal.'))
      }
    } finally { setBusy(null) }
  }

  const resumeProposal = async (summary: ProposalSummary) => {
    if (!project || busy) return
    setBusy('load')
    setError('')
    setNotice('')
    try {
      await fetchProposal(summary.proposal_id, project)
    } catch (reason) {
      setError(errorText(reason, 'Could not load this saved proposal.'))
    } finally { setBusy(null) }
  }

  const refreshActiveProposal = async () => {
    if (!project || !activeProposal || busy) return
    setBusy('load')
    setError('')
    try { await fetchProposal(activeProposal.proposal_id, project) }
    catch (reason) { setError(errorText(reason, 'Could not refresh this proposal.')) }
    finally { setBusy(null) }
  }

  const toggleAction = (id: string, checked: boolean) => {
    setSelectedActionIds(current => checked
      ? current.includes(id) ? current : [...current, id]
      : current.filter(candidate => candidate !== id))
    setError('')
  }

  const applySelected = () => {
    if (!project || !activeProposal || selectedActionIdsInProposalOrder.length === 0 || previewState.error || applied || activeProposal.status !== 'ready') return
    setBusy('apply')
    setError('')
    setNotice('')
    const succeeded = applyAutoEditProposal(activeProposal, selectedActionIdsInProposalOrder)
    setBusy(null)
    if (succeeded) setNotice(`Applied ${selectedActionIds.length} selected ${selectedActionIds.length === 1 ? 'action' : 'actions'} in one undo step.`)
    else setError(useEditorStore.getState().error || 'The selected actions could not be applied.')
  }

  if (!open || !project) return null

  const writerLabel = capabilities?.writer_available
    ? [capabilities.writer_provider, capabilities.writer_model].filter(Boolean).join(' · ') || 'Local proposal writer'
    : capabilityLoading ? 'Checking installed proposal writer…'
      : `Writer modes unavailable: ${capabilities?.reason || capabilityError || 'No local proposal writer is available.'}`
  const provenance = activeProposal?.mode === 'director'
    ? 'Director · saved shot indices; no language model'
    : activeProposal?.writer?.model_id
      ? `${activeProposal.writer.provider || 'Local writer'} · ${activeProposal.writer.model_id}`
      : 'Writer identity was not recorded for this saved proposal'
  const canApply = Boolean(activeProposal && activeProposal.status === 'ready' && !applied
    && selectedActionIdsInProposalOrder.length > 0 && !previewState.error && previewState.changes.length > 0 && !busy)

  return (
    <div
      ref={dialogRef}
      className="fixed inset-0 z-[160] flex h-[100dvh] items-center justify-center overflow-y-auto bg-black/65 px-3 py-4 backdrop-blur-sm"
      style={{ paddingTop: 'max(1rem, env(safe-area-inset-top))', paddingBottom: 'max(1rem, env(safe-area-inset-bottom))' }}
      onMouseDown={event => { if (event.target === event.currentTarget) onClose() }}
    >
      <section role="dialog" aria-modal="true" aria-labelledby="editor-auto-edit-title" className="flex max-h-[min(850px,calc(100dvh-2rem))] w-full max-w-[820px] flex-col overflow-hidden rounded-2xl border border-border bg-bg-secondary shadow-2xl">
        <header className="flex shrink-0 items-start justify-between gap-4 border-b border-border px-4 py-3 sm:px-5">
          <div className="flex min-w-0 items-start gap-3">
            <span className="mt-0.5 rounded-lg bg-accent-blue/10 p-2 text-accent-blue"><WandSparkles size={17} /></span>
            <div className="min-w-0">
              <h2 id="editor-auto-edit-title" className="text-sm font-semibold text-text-primary">Auto-edit proposal</h2>
              <p className="mt-1 truncate text-[12px] text-text-muted">{project.name} · revision {project.revision ?? 'unsaved'} · review before applying</p>
            </div>
          </div>
          <button type="button" onClick={onClose} aria-label="Close Auto-edit proposal" className="rounded-md p-1.5 text-text-muted hover:bg-bg-hover hover:text-text-primary"><X size={16} /></button>
        </header>

        <div role="tablist" aria-label="Auto-edit sections" className="flex shrink-0 gap-1 border-b border-border px-4 pt-2">
          <button type="button" role="tab" aria-selected={tab === 'plan'} onClick={() => setTab('plan')} className={`rounded-t-lg px-3 py-2 text-xs ${tab === 'plan' ? 'border border-b-0 border-border bg-bg-primary text-text-primary' : 'text-text-muted hover:text-text-primary'}`}>Plan</button>
          <button type="button" role="tab" aria-selected={tab === 'history'} onClick={() => setTab('history')} className={`rounded-t-lg px-3 py-2 text-xs ${tab === 'history' ? 'border border-b-0 border-border bg-bg-primary text-text-primary' : 'text-text-muted hover:text-text-primary'}`}>
            History {history.length > 0 && <span className="ml-1 text-[11px] text-text-muted">{history.length}</span>}
          </button>
        </div>

        <div className="min-h-0 flex-1 overflow-y-auto bg-bg-primary p-3 sm:p-5">
          {tab === 'plan' ? (
            <div role="tabpanel" aria-label="Plan" className="space-y-4">
              <section className="space-y-3 rounded-xl border border-border bg-bg-secondary p-3 sm:p-4">
                <div className="flex flex-wrap items-start justify-between gap-2">
                  <div><h3 className="text-xs font-semibold text-text-primary">Describe a plan</h3><p className="mt-1 text-[12px] leading-relaxed text-text-muted">Planning creates a reviewable proposal only. Your timeline changes after you select actions and apply them.</p></div>
                  <div role="status" className="max-w-full rounded-full border border-border bg-bg-tertiary px-2.5 py-1 text-[11px] text-text-muted">{writerLabel}</div>
                </div>
                <div className="grid gap-3 sm:grid-cols-[minmax(0,1fr)_180px]">
                  <label className="space-y-1 text-[12px] font-medium text-text-secondary">Planning mode
                    <select aria-label="Planning mode" value={mode} onChange={event => setMode(event.target.value as AutoEditMode)} className={inputClass}>
                      {modes.map(option => <option key={option.id} value={option.id} disabled={!modeAvailable(option.id)}>{option.label}{!modeAvailable(option.id) ? ' · unavailable' : ''}</option>)}
                    </select>
                  </label>
                  <label className="space-y-1 text-[12px] font-medium text-text-secondary">Target duration (seconds)
                    <input aria-label="Target duration seconds" type="number" min="0" max="86400" step="0.1" value={targetSeconds} onChange={event => setTargetSeconds(event.target.value)} disabled={mode === 'director'} className={inputClass} placeholder={mode === 'director' ? 'Keep source durations' : 'Optional'} />
                  </label>
                </div>
                <p className="-mt-2 text-[11px] text-text-muted">{modes.find(option => option.id === mode)?.hint}{mode === 'director' ? ' Audio and protected clips remain in place. Review soundtrack alignment after reordering.' : ''}</p>
                <label className="block space-y-1 text-[12px] font-medium text-text-secondary">Brief
                  <textarea aria-label="Auto-edit brief" rows={3} maxLength={4000} value={brief} onChange={event => setBrief(event.target.value)} disabled={mode === 'director'} placeholder={mode === 'director' ? 'Uses saved Director shot indices.' : 'Describe pacing, emphasis, or the change you want.'} className={`${inputClass} resize-y`} />
                </label>
                <div className="flex flex-wrap items-center justify-between gap-3">
                  <p className="max-w-[480px] text-[11px] leading-relaxed text-text-muted">The plan uses project clip names, timing, and provenance. Missing-media notes are suggestions only; this tool does not generate or fetch assets.</p>
                  <button type="button" onClick={() => void generateProposal()} disabled={generateDisabled} className="inline-flex shrink-0 items-center gap-2 rounded-lg bg-accent-blue px-3.5 py-2 text-[13px] font-semibold text-white hover:bg-accent-blue/90 disabled:cursor-not-allowed disabled:opacity-45">
                    {busy === 'plan' ? <Loader2 size={13} className="animate-spin" /> : <WandSparkles size={13} />}
                    {busy === 'plan' ? 'Saving and planning…' : 'Generate proposal'}
                  </button>
                </div>
                {receiptRequest && <div className="flex flex-wrap items-center justify-between gap-2 rounded-lg border border-amber-500/25 bg-amber-500/5 p-2.5 text-[12px] text-amber-200"><span>Request {receiptRequest.proposalId} has an unresolved receipt. Check this ID before planning again.</span><button type="button" onClick={() => void recoverReceipt(receiptRequest)} disabled={busy !== null} className="rounded-md border border-amber-500/30 px-2.5 py-1.5 font-medium hover:bg-amber-500/10 disabled:opacity-40">{busy === 'recover' ? 'Checking…' : 'Check proposal receipt'}</button></div>}
              </section>

              {activeProposal && (
                <section aria-label="Proposal review" className="space-y-3 rounded-xl border border-border bg-bg-secondary p-3 sm:p-4">
                  <div className="flex flex-wrap items-start justify-between gap-3">
                    <div className="min-w-0"><div className="flex flex-wrap items-center gap-2"><h3 className="text-xs font-semibold text-text-primary">Proposal review</h3><span className="rounded-full bg-bg-tertiary px-2 py-0.5 text-[11px] uppercase tracking-wide text-text-secondary">{activeProposal.status}</span></div><p className="mt-1 text-[12px] text-text-muted">{activeProposal.summary || 'No summary supplied.'}</p><p className="mt-1 text-[11px] text-text-muted">{provenance} · base revision {activeProposal.base_revision}</p></div>
                    <button type="button" onClick={() => void refreshActiveProposal()} disabled={busy !== null} className="inline-flex shrink-0 items-center gap-1.5 rounded-md border border-border px-2 py-1.5 text-[12px] text-text-secondary hover:bg-bg-hover disabled:opacity-45"><RotateCw size={11} /> Refresh</button>
                  </div>

                  {activeProposal.errors && activeProposal.errors.length > 0 && <div role="alert" className="rounded-lg border border-red-500/25 bg-red-500/5 p-2.5 text-[12px] text-red-200"><div className="mb-1 font-semibold">Proposal errors</div><ul className="list-disc space-y-1 pl-4">{activeProposal.errors.map((entry, index) => <li key={`${index}-${entry}`}>{entry}</li>)}</ul></div>}
                  {activeProposal.warnings.length > 0 && <div className="rounded-lg border border-amber-500/20 bg-amber-500/5 p-2.5 text-[12px] text-amber-100"><div className="mb-1 flex items-center gap-1 font-semibold"><AlertTriangle size={11} /> Review warnings</div><ul className="list-disc space-y-1 pl-4">{activeProposal.warnings.map((entry, index) => <li key={`${index}-${entry}`}>{entry}</li>)}</ul></div>}
                  {activeProposal.missing_media.length > 0 && <div className="rounded-lg border border-border bg-bg-tertiary p-2.5 text-[12px] text-text-secondary"><div className="mb-1 font-semibold">Suggested missing media</div><ul className="list-disc space-y-1 pl-4">{activeProposal.missing_media.map((entry, index) => <li key={`${index}-${missingMediaText(entry)}`}>{missingMediaText(entry)}</li>)}</ul><p className="mt-1 text-[11px] text-text-muted">These are notes for your review; no media was created or downloaded.</p></div>}

                  {activeProposal.drafts && activeProposal.drafts.length > 0 && <details className="rounded-lg border border-border bg-bg-tertiary px-3 py-2"><summary className="cursor-pointer text-[12px] font-medium text-text-secondary">Show writer draft and provenance ({activeProposal.drafts.length})</summary><div className="mt-2 space-y-2">{activeProposal.drafts.map((draft, index) => <pre key={index} className="max-h-48 overflow-auto whitespace-pre-wrap break-words rounded-md bg-bg-primary p-2 text-[11px] leading-relaxed text-text-muted">{draftText(draft)}</pre>)}</div></details>}

                  {activeProposal.actions.length > 0 ? <>
                    <div className="flex flex-wrap items-center justify-between gap-2"><div><h4 className="text-[13px] font-semibold text-text-primary">Actions · {selectedActionIds.length} of {activeProposal.actions.length} selected</h4><p className="mt-0.5 text-[11px] text-text-muted">Toggle actions to preview a partial edit. All actions start selected.</p></div><div className="flex gap-1"><button type="button" onClick={() => setSelectedActionIds(activeProposalActionIds)} className="rounded border border-border px-2 py-1 text-[11px] text-text-secondary hover:bg-bg-hover">Select all</button><button type="button" onClick={() => setSelectedActionIds([])} className="rounded border border-border px-2 py-1 text-[11px] text-text-secondary hover:bg-bg-hover">Clear</button></div></div>
                    <ol className="space-y-1.5">
                      {activeProposal.actions.map((action, index) => {
                        const id = actionId(action)
                        const checked = selectedActionIds.includes(id)
                        return <li key={id}><label className={`flex cursor-pointer items-start gap-2.5 rounded-lg border p-2.5 ${checked ? 'border-accent-blue/30 bg-accent-blue/5' : 'border-border bg-bg-tertiary opacity-70'}`}><input type="checkbox" aria-label={`Include action ${index + 1}`} checked={checked} onChange={event => toggleAction(id, event.target.checked)} disabled={applied || busy === 'apply'} className="mt-0.5 accent-[var(--color-accent-blue)]" /><span className="min-w-0 flex-1"><span className="block text-[12px] font-medium text-text-primary">{describeAction(project, action)}</span>{action.reason && <span className="mt-0.5 block text-[11px] leading-relaxed text-text-muted">{action.reason}</span>}</span><span className="shrink-0 font-mono text-[10px] text-text-muted">{action.type}</span></label></li>
                      })}
                    </ol>
                    {previewState.error ? <div role="alert" className="rounded-lg border border-amber-500/25 bg-amber-500/5 p-2.5 text-[12px] text-amber-100">Preview unavailable: {previewState.error}</div> : <div role="status" className="rounded-lg border border-border bg-bg-tertiary p-2.5 text-[12px] text-text-secondary"><div className="font-semibold">Validated preview · {previewState.changes.length} {previewState.changes.length === 1 ? 'change' : 'changes'}</div><p className="mt-1">The editor checked the selected actions against a project copy. Your timeline has not changed.</p><ol className="mt-2 list-decimal space-y-1 pl-4">{previewState.changes.map((change, index) => <li key={index}>{changeDescription(change)}</li>)}</ol></div>}
                    {applied ? <p role="status" className="rounded-lg border border-accent-blue/20 bg-accent-blue/5 p-2.5 text-[12px] text-accent-blue">This proposal has already been applied. Undo is available in the editor history.</p> : null}
                    <div className="flex flex-wrap items-center justify-between gap-2 border-t border-border pt-3"><p className="text-[11px] text-text-muted">Applying is one undoable editor action. Pinned or locked clips and stale proposals are rejected.</p><button type="button" onClick={applySelected} disabled={!canApply} className="inline-flex items-center gap-1.5 rounded-lg bg-cta px-3.5 py-2 text-[13px] font-semibold text-white disabled:cursor-not-allowed disabled:opacity-40"><Check size={13} /> Apply selected actions</button></div>
                  </> : <p className="rounded-lg border border-dashed border-border p-3 text-[12px] text-text-muted">This proposal has no actions to apply.</p>}
                </section>
              )}

              {error && <p role="alert" className="rounded-lg border border-red-500/25 bg-red-500/5 p-3 text-[12px] text-red-200">{error}</p>}
              {notice && <p role="status" className="rounded-lg border border-accent-blue/20 bg-accent-blue/5 p-3 text-[12px] text-accent-blue">{notice}</p>}
            </div>
          ) : (
            <div role="tabpanel" aria-label="History" className="space-y-3">
              <div className="flex flex-wrap items-center justify-between gap-2"><div><h3 className="text-xs font-semibold text-text-primary">Saved proposals</h3><p className="mt-1 text-[12px] text-text-muted">Resume a prior proposal for this project. Its revision and content signature are checked before preview or apply.</p></div><button type="button" onClick={() => void refreshHistory(project)} disabled={historyLoading} className="inline-flex items-center gap-1.5 rounded-md border border-border px-2.5 py-1.5 text-[12px] text-text-secondary hover:bg-bg-hover disabled:opacity-50">{historyLoading ? <Loader2 size={11} className="animate-spin" /> : <Clock3 size={11} />} Refresh</button></div>
              {historyError && <p role="alert" className="rounded-lg border border-red-500/25 bg-red-500/5 p-3 text-[12px] text-red-200">{historyError}</p>}
              {historyLoading && history.length === 0 ? <p className="rounded-lg border border-border bg-bg-secondary p-4 text-[12px] text-text-muted">Loading saved proposals…</p> : history.length > 0 ? <ol className="space-y-2">{history.map(entry => <li key={entry.proposal_id} className="flex flex-wrap items-center justify-between gap-3 rounded-xl border border-border bg-bg-secondary p-3"><div className="min-w-0"><div className="flex flex-wrap items-center gap-2"><span className="text-[13px] font-semibold text-text-primary">{entry.summary || 'Untitled proposal'}</span><span className="rounded-full bg-bg-tertiary px-2 py-0.5 text-[10px] uppercase tracking-wide text-text-muted">{entry.status}</span></div><p className="mt-1 text-[11px] text-text-muted">{entry.mode} · base revision {entry.base_revision} · {formatTime(entry.updated_at ?? entry.created_at)}</p><p className="mt-0.5 truncate font-mono text-[10px] text-text-muted">{entry.proposal_id}</p></div><button type="button" onClick={() => void resumeProposal(entry)} disabled={busy !== null} aria-label={`Resume proposal ${entry.proposal_id}`} className="inline-flex shrink-0 items-center gap-1.5 rounded-md border border-border px-2.5 py-2 text-[12px] text-text-secondary hover:border-accent-blue/40 hover:text-text-primary disabled:opacity-45">{busy === 'load' ? <Loader2 size={11} className="animate-spin" /> : <RotateCw size={11} />} Resume</button></li>)}</ol> : !historyError && <p className="rounded-xl border border-dashed border-border bg-bg-secondary p-6 text-center text-[12px] text-text-muted">{historyLoading ? 'Loading saved proposals…' : 'No saved proposals for this project yet.'}</p>}
              {error && <p role="alert" className="rounded-lg border border-red-500/25 bg-red-500/5 p-3 text-[12px] text-red-200">{error}</p>}
              {notice && <p role="status" className="rounded-lg border border-accent-blue/20 bg-accent-blue/5 p-3 text-[12px] text-accent-blue">{notice}</p>}
            </div>
          )}
        </div>
      </section>
    </div>
  )
}
