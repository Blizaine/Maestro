import { useEffect, useId, useMemo, useState, type ChangeEvent } from 'react'
import { Check, ChevronDown, ChevronRight, Loader2, Save } from 'lucide-react'
import * as api from '../../api/comfyui'
import { ComfyUIWorkflowButton } from './ComfyUIWorkflowDialog'

type NativeField = {
  name: string
  label: string
  valueType: 'text' | 'number' | 'image' | 'video' | 'audio'
  autoMap?: boolean
  aliases?: string[]
}

function isMediaBinding(binding: api.ComfyUIBinding): boolean {
  return binding.type === 'image' || binding.type === 'video' || binding.type === 'audio'
}

const FIELDS_BY_KIND: Record<api.ComfyUIKind, NativeField[]> = {
  image: [
    { name: 'prompt', label: 'Prompt', valueType: 'text', aliases: ['prompt', 'positive_prompt', 'prompt_text', 'positive'] },
    { name: 'negative_prompt', label: 'Negative prompt', valueType: 'text', aliases: ['negative_prompt', 'negative'] },
    { name: 'seed', label: 'Seed', valueType: 'number', aliases: ['seed'] },
    { name: 'width', label: 'Width', valueType: 'number', aliases: ['width'] },
    { name: 'height', label: 'Height', valueType: 'number', aliases: ['height'] },
    { name: 'num_inference_steps', label: 'Steps', valueType: 'number', aliases: ['steps', 'num_inference_steps', 'num_steps'] },
    { name: 'guidance_scale', label: 'Guidance scale', valueType: 'number', aliases: ['cfg', 'cfg_scale', 'guidance_scale'] },
    { name: 'image_start', label: 'Start image', valueType: 'image', autoMap: false },
    { name: 'image_end', label: 'End image', valueType: 'image', autoMap: false },
    { name: 'image_refs.0', label: 'First image reference', valueType: 'image', autoMap: false },
    { name: 'image_refs.1', label: 'Second image reference', valueType: 'image', autoMap: false },
  ],
  video: [
    { name: 'prompt', label: 'Prompt', valueType: 'text', aliases: ['prompt', 'positive_prompt', 'prompt_text', 'positive'] },
    { name: 'negative_prompt', label: 'Negative prompt', valueType: 'text', aliases: ['negative_prompt', 'negative'] },
    { name: 'seed', label: 'Seed', valueType: 'number', aliases: ['seed'] },
    { name: 'width', label: 'Width', valueType: 'number', aliases: ['width'] },
    { name: 'height', label: 'Height', valueType: 'number', aliases: ['height'] },
    { name: 'num_inference_steps', label: 'Steps', valueType: 'number', aliases: ['steps', 'num_inference_steps', 'num_steps'] },
    { name: 'guidance_scale', label: 'Guidance scale', valueType: 'number', aliases: ['cfg', 'cfg_scale', 'guidance_scale'] },
    { name: 'video_length', label: 'Frames', valueType: 'number', aliases: ['frames', 'video_length'] },
    { name: 'fps', label: 'Frames per second', valueType: 'number', aliases: ['fps'] },
    { name: 'image_start', label: 'Start image', valueType: 'image', autoMap: false },
    { name: 'image_end', label: 'End image', valueType: 'image', autoMap: false },
    { name: 'image_refs.0', label: 'First image reference', valueType: 'image', autoMap: false },
    { name: 'image_refs.1', label: 'Second image reference', valueType: 'image', autoMap: false },
    { name: 'video_guide', label: 'Video guide', valueType: 'video', autoMap: false },
    { name: 'audio_guide', label: 'Audio guide', valueType: 'audio', autoMap: false },
    { name: 'audio_frame_offset', label: 'Driving audio offset (frames)', valueType: 'number', autoMap: false },
    { name: 'audio_start_seconds', label: 'Driving audio offset (seconds)', valueType: 'number', autoMap: false },
    { name: 'duration_seconds', label: 'Duration (seconds)', valueType: 'number', autoMap: false },
  ],
  audio: [
    { name: 'prompt', label: 'Prompt', valueType: 'text', aliases: ['prompt', 'positive_prompt', 'prompt_text', 'positive'] },
    { name: 'negative_prompt', label: 'Negative prompt', valueType: 'text', aliases: ['negative_prompt', 'negative'] },
    { name: 'seed', label: 'Seed', valueType: 'number', aliases: ['seed'] },
    { name: 'duration_seconds', label: 'Duration (seconds)', valueType: 'number', aliases: ['duration_seconds'] },
    { name: 'audio_guide', label: 'Audio guide', valueType: 'audio', autoMap: false },
  ],
}

function initialDefaults(bindings: api.ComfyUIBinding[]): Record<string, api.ComfyUIScalar> {
  const values: Record<string, api.ComfyUIScalar> = {}
  for (const binding of bindings) {
    if (isMediaBinding(binding)) continue
    if (binding.default !== undefined) values[binding.name] = binding.default
    else if (binding.type === 'boolean') values[binding.name] = false
    else if (binding.type === 'choice' && binding.options?.length) values[binding.name] = binding.options[0]
  }
  return values
}

function autoMap(fields: NativeField[], workflow: api.ComfyUIWorkflow): Record<string, string> {
  const result: Record<string, string> = {}
  const used = new Set<string>()
  const normalize = (value: string) => value.toLowerCase().replace(/[^a-z0-9]/g, '')
  for (const field of fields) {
    if (field.autoMap === false || !field.aliases?.length) continue
    const aliases = field.aliases.map(normalize)
    const found = workflow.bindings.find(binding =>
      (aliases.includes(normalize(binding.name)) || aliases.includes(normalize(binding.label ?? '')))
      && !used.has(binding.name) && bindingMatchesField(field, binding))
    if (found) {
      result[field.name] = found.name
      used.add(found.name)
    }
  }

  // This exact built-in template exposes one required image binding. Keep other
  // templates and imported workflows explicit because media mappings are specific.
  if (workflow.template && workflow.id === 'flux3_klein_distilled_reference_edit') {
    const referenceField = fields.find(field => field.name === 'image_refs.0' && field.valueType === 'image')
    const referenceBinding = workflow.bindings.find(binding => binding.name === 'reference_image' && binding.type === 'image')
    if (referenceField && referenceBinding && !used.has(referenceBinding.name)) {
      result[referenceField.name] = referenceBinding.name
    }
  }
  return result
}

function bindingMatchesField(field: NativeField, binding: api.ComfyUIBinding): boolean {
  if (field.valueType === 'image' || field.valueType === 'video' || field.valueType === 'audio') {
    return binding.type === field.valueType
  }
  if (field.valueType === 'text') return binding.type === 'string' || binding.type === 'choice'
  return binding.type === 'integer' || binding.type === 'number' || binding.type === 'choice'
}

function validationValue(binding: api.ComfyUIBinding): api.ComfyUIScalar {
  if (binding.default !== undefined) return binding.default
  if (binding.options?.length) return binding.options[0]
  if (binding.type === 'boolean') return false
  if (binding.type === 'integer' || binding.type === 'number') return binding.min ?? 1
  if (binding.type === 'image' || binding.type === 'video' || binding.type === 'audio') return '__maestro_pending_media__'
  return 'Maestro route validation'
}

function validationMessage(value: unknown): string {
  if (typeof value === 'string') return value
  if (value && typeof value === 'object') {
    const item = value as Record<string, unknown>
    const details = [item.node_id, item.class_type, item.input, item.value, item.message, item.reason]
      .filter(part => typeof part === 'string' || typeof part === 'number')
      .map(String)
    return details.join(' · ') || JSON.stringify(value)
  }
  return String(value)
}

function defaultText(value: api.ComfyUIScalar | undefined): string {
  return value === undefined ? '' : String(value)
}

export function ComfyUIEngineControl({ modelType, kind }: { modelType: string; kind: api.ComfyUIKind }) {
  const instanceId = useId().replace(/[^a-zA-Z0-9_-]/g, '')
  const panelId = `comfyui-engine-panel-${instanceId}`
  const [status, setStatus] = useState<api.ComfyUIStatus | null>(null)
  const [workflows, setWorkflows] = useState<api.ComfyUIWorkflow[]>([])
  const [route, setRoute] = useState<api.ComfyUIRouteStatus | null>(null)
  const [selectedId, setSelectedId] = useState('')
  const [fieldBindings, setFieldBindings] = useState<Record<string, string>>({})
  const [inputDefaults, setInputDefaults] = useState<Record<string, api.ComfyUIScalar>>({})
  const [expanded, setExpanded] = useState(false)
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [validating, setValidating] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [validation, setValidation] = useState<api.ComfyUIWorkflowValidation | null>(null)
  const [workflowOptions, setWorkflowOptions] = useState<api.ComfyUIWorkflowOptions | null>(null)
  const [optionsLoading, setOptionsLoading] = useState(false)
  const [optionsError, setOptionsError] = useState<string | null>(null)
  const [refreshVersion, setRefreshVersion] = useState(0)

  const available = useMemo(() => workflows.filter(workflow => workflow.kind === kind), [workflows, kind])
  const selected = available.find(workflow => workflow.id === selectedId) ?? null
  const fields = FIELDS_BY_KIND[kind]
  const mappedNames = new Set(Object.values(fieldBindings).filter(Boolean))
  const unmappedBindings = selected?.bindings.filter(binding => !mappedNames.has(binding.name)) ?? []
  const duplicateTargets = Object.values(fieldBindings).filter(Boolean).length !== new Set(Object.values(fieldBindings).filter(Boolean)).size
  const promptMapped = Boolean(fieldBindings.prompt)
  const missingRequiredDefaults = unmappedBindings.filter(binding => {
    if (!binding.required) return false
    if (isMediaBinding(binding)) return true
    const value = inputDefaults[binding.name] ?? binding.default
    return value === undefined || (typeof value === 'string' && !value.trim())
  })
  const canConfigure = Boolean(selected && promptMapped && !duplicateTargets && !missingRequiredDefaults.length)
  const routeEnabled = route?.enabled === true
  const routeKindMismatch = routeEnabled && Boolean(route?.kind && route.kind !== kind)
  const routeWorkflowName = route?.name || workflows.find(workflow => workflow.id === route?.workflow_id)?.name || 'ComfyUI workflow'

  useEffect(() => {
    const refresh = () => setRefreshVersion(version => version + 1)
    window.addEventListener(api.COMFYUI_STATE_CHANGED_EVENT, refresh)
    return () => window.removeEventListener(api.COMFYUI_STATE_CHANGED_EVENT, refresh)
  }, [])

  useEffect(() => {
    let cancelled = false
    setLoading(true)
    setError(null)
    setStatus(null)
    setWorkflows([])
    setRoute(null)
    setSelectedId('')
    setFieldBindings({})
    setInputDefaults({})
    setValidation(null)

    const load = async () => {
      try {
        const nextStatus = await api.fetchComfyUIStatus()
        if (cancelled) return
        setStatus(nextStatus)
        const nextRoute = nextStatus.routes?.[modelType] ?? null
        setRoute(nextRoute)
        if (!nextStatus.enabled) return
        const result = await api.fetchComfyUIWorkflows()
        if (cancelled) return
        const all = [...(result.templates ?? []).map(workflow => ({ ...workflow, template: true })), ...(result.workflows ?? [])]
        setWorkflows(all)
        const sameKind = all.filter(workflow => workflow.kind === kind)
        const savedWorkflow = nextRoute?.kind === kind || !nextRoute?.kind
          ? sameKind.find(workflow => workflow.id === nextRoute?.workflow_id)
          : undefined
        const initialWorkflow = savedWorkflow ?? sameKind[0]
        if (initialWorkflow) {
          setSelectedId(initialWorkflow.id)
          setFieldBindings(nextRoute?.kind === kind || !nextRoute?.kind
            ? nextRoute?.field_bindings ?? autoMap(fields, initialWorkflow)
            : autoMap(fields, initialWorkflow))
          setInputDefaults({ ...initialDefaults(initialWorkflow.bindings), ...(nextRoute?.kind === kind ? nextRoute.input_defaults : {}) })
        }
      } catch (reason) {
        if (!cancelled) setError(reason instanceof Error ? reason.message : 'Could not load ComfyUI routing settings.')
      } finally {
        if (!cancelled) setLoading(false)
      }
    }
    void load()
    return () => { cancelled = true }
  }, [modelType, kind, fields, refreshVersion])

  useEffect(() => {
    if (!expanded || !selectedId) {
      setWorkflowOptions(null)
      setOptionsError(null)
      setOptionsLoading(false)
      return
    }
    const controller = new AbortController()
    setWorkflowOptions(null)
    setOptionsError(null)
    setOptionsLoading(true)
    api.fetchComfyUIWorkflowOptions(selectedId, controller.signal)
      .then(result => {
        if (!controller.signal.aborted) setWorkflowOptions(result)
      })
      .catch(reason => {
        if (!controller.signal.aborted) {
          setOptionsError(reason instanceof Error ? reason.message : 'Server choices are unavailable; enter a value manually.')
        }
      })
      .finally(() => {
        if (!controller.signal.aborted) setOptionsLoading(false)
      })
    return () => controller.abort()
  }, [expanded, selectedId])

  const chooseWorkflow = (workflowId: string) => {
    setSelectedId(workflowId)
    const workflow = available.find(item => item.id === workflowId)
    if (workflow) {
      setFieldBindings(autoMap(fields, workflow))
      setInputDefaults(initialDefaults(workflow.bindings))
    } else {
      setFieldBindings({})
      setInputDefaults({})
    }
    setValidation(null)
    setError(null)
  }

  const setMapping = (field: string, value: string) => {
    setFieldBindings(current => ({ ...current, [field]: value }))
    setValidation(null)
  }

  const setDefault = (binding: api.ComfyUIBinding, value: api.ComfyUIScalar | undefined) => {
    setInputDefaults(current => {
      const next = { ...current }
      if (value === undefined) delete next[binding.name]
      else next[binding.name] = value
      return next
    })
    setValidation(null)
  }

  const effectiveDefaults = (): Record<string, api.ComfyUIScalar> => {
    if (!selected) return {}
    const output: Record<string, api.ComfyUIScalar> = {}
    for (const binding of selected.bindings) {
      if (mappedNames.has(binding.name) || isMediaBinding(binding)) continue
      const value = inputDefaults[binding.name] ?? binding.default
      if (value !== undefined) output[binding.name] = value
    }
    return output
  }

  const verifySelected = async () => {
    if (!selected || !canConfigure) return null
    setValidating(true)
    setError(null)
    try {
      const inputs = effectiveDefaults()
      for (const target of Object.values(fieldBindings)) {
        if (!target || inputs[target] !== undefined) continue
        const binding = selected.bindings.find(item => item.name === target)
        if (binding) inputs[target] = validationValue(binding)
      }
      const result = await api.validateComfyUIWorkflow(selected.id, inputs)
      setValidation(result)
      if (!result.valid) {
        const detail = [...(result.errors ?? []), ...(result.missing_nodes ?? []), ...(result.missing_models ?? [])]
          .map(validationMessage)
          .filter(Boolean)
          .slice(0, 4)
          .join('; ')
        setError(detail ? `Resolve workflow validation before enabling it: ${detail}` : 'Resolve workflow validation before enabling it.')
        return null
      }
      return { inputs, result }
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'Could not validate the selected ComfyUI workflow.')
      return null
    } finally {
      setValidating(false)
    }
  }

  const saveEnabledRoute = async () => {
    if (!selected || !canConfigure || saving || validating) return
    setSaving(true)
    setError(null)
    try {
      const verified = await verifySelected()
      if (!verified) return
      const fieldBindings = Object.fromEntries(Object.entries(fieldBindingsState()).filter(([, target]) => Boolean(target)))
      const saved = await api.saveComfyUIRoute(modelType, {
        workflow_id: selected.id,
        enabled: true,
        field_bindings: fieldBindings,
        input_defaults: effectiveDefaults(),
        kind,
      })
      setRoute(saved)
      setStatus(current => current ? { ...current, routes: { ...(current.routes ?? {}), [modelType]: saved } } : current)
      setError(null)
      setExpanded(false)
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'Could not save this ComfyUI route.')
    } finally {
      setSaving(false)
    }
  }

  const fieldBindingsState = () => fieldBindings

  const switchToNative = async () => {
    if (!route || !route.workflow_id || saving) return
    setSaving(true)
    setError(null)
    try {
      const saved = await api.saveComfyUIRoute(modelType, {
        workflow_id: route.workflow_id,
        enabled: false,
        field_bindings: route.field_bindings ?? {},
        input_defaults: route.input_defaults ?? {},
        kind: route.kind ?? kind,
      })
      setRoute(saved)
      setStatus(current => current ? { ...current, routes: { ...(current.routes ?? {}), [modelType]: saved } } : current)
      setError(null)
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'Could not switch this model to Native.')
    } finally {
      setSaving(false)
    }
  }

  if (loading || !status?.enabled) return null

  return <section className="rounded-xl border border-border bg-bg-secondary">
    <button
      type="button"
      aria-expanded={expanded}
      aria-controls={panelId}
      onClick={() => setExpanded(value => !value)}
      className="flex w-full items-center gap-2 px-3 py-2.5 text-left"
    >
      {expanded ? <ChevronDown size={13} /> : <ChevronRight size={13} />}
      <span className="min-w-0 flex-1">
        <span className="block text-xs font-medium text-text-primary">Generation engine</span>
        <span className="mt-0.5 block truncate text-[10px] text-text-muted">
          {routeEnabled ? `${routeWorkflowName}${routeKindMismatch ? ` · configured for ${route?.kind}` : ''}` : 'Native'}
        </span>
      </span>
      <span className={`rounded-full border px-2 py-0.5 text-[9px] ${routeEnabled ? 'border-accent-blue/30 text-accent-blue' : 'border-border text-text-muted'}`}>
        {routeEnabled ? 'ComfyUI' : 'Native'}
      </span>
    </button>

    {expanded && <div id={panelId} className="space-y-3 border-t border-border px-3 py-3">
      {error && <div role="alert" className="rounded-lg border border-red-500/30 bg-red-500/10 px-2.5 py-2 text-[10px] text-chip-red">{error}</div>}
      {routeKindMismatch && <p role="status" className="text-[10px] text-indicator-warning">This model has a {route?.kind} workflow route, which does not match the current {kind} mode. Native generations are rejected until you choose Native or save a matching workflow.</p>}
      {status.connected === false && <p className="text-[10px] text-text-muted">This control does not test the ComfyUI connection. Native remains available while ComfyUI is unreachable.</p>}
      <div className="flex flex-wrap items-center gap-2">
        <button
          type="button"
          aria-pressed={!routeEnabled}
          onClick={() => void switchToNative()}
          disabled={!routeEnabled || saving}
          className={`rounded-lg border px-2.5 py-2 text-[10px] ${!routeEnabled ? 'border-border bg-bg-active text-text-primary' : 'border-border bg-bg-primary text-text-secondary hover:text-text-primary'} disabled:opacity-50`}
        >
          {saving && routeEnabled ? <Loader2 size={11} className="mr-1 inline animate-spin" /> : null}Use Native
        </button>
        {routeEnabled && <span className="inline-flex items-center gap-1 text-[10px] text-indicator-success"><Check size={12} /> Routing to {routeWorkflowName}</span>}
        <ComfyUIWorkflowButton label="Open workflow library" className="rounded-lg border border-border bg-bg-primary px-2.5 py-2 text-[10px] text-text-secondary hover:text-text-primary" />
      </div>

      <div className="space-y-2 border-t border-border pt-3">
        <div>
          <label htmlFor={`${panelId}-workflow`} className="mb-1 block text-[10px] font-medium text-text-secondary">Verified {kind} workflow</label>
          <select
            id={`${panelId}-workflow`}
            aria-label="Verified workflow"
            value={selectedId}
            onChange={event => chooseWorkflow(event.target.value)}
            className="w-full rounded-lg border border-border bg-bg-primary px-2.5 py-2 text-xs text-text-primary outline-none focus:border-accent-blue"
          >
            <option value="">Choose a {kind} workflow</option>
            {available.map(workflow => <option key={workflow.id} value={workflow.id}>{workflow.name}</option>)}
          </select>
          {!available.length && <p className="mt-1 text-[9px] text-text-muted">Import or save a {kind} workflow in the ComfyUI workflow library first.</p>}
        </div>

        {selected && <>
          <p className="text-[9px] leading-relaxed text-text-muted">Native remains the default. Map the Native controls you want to pass through; required workflow inputs and preflight are checked before saving.</p>
          {optionsLoading && <p role="status" className="text-[9px] text-text-muted">Loading choices from the selected ComfyUI server…</p>}
          {optionsError && <p role="status" className="text-[9px] text-indicator-warning">Server choices are unavailable. You can still enter values manually. {optionsError}</p>}
          <p className="text-[9px] leading-relaxed text-text-muted">The selected workflow controls its checkpoint, sampler, and performance. Maestro native LoRAs, finishing, and multi-window options are not translated; unsupported options are rejected, with no fallback.</p>
          {kind === 'video' && <p className="text-[9px] leading-relaxed text-text-muted">For driving-audio clips, map Audio guide and one Driving audio offset field. A nonzero audio offset without a workflow mapping is rejected.</p>}

          <fieldset className="space-y-1.5">
            <legend className="mb-1 text-[10px] font-semibold text-text-secondary">Map Native fields</legend>
            {kind !== 'audio' && <p className="mb-1 text-[9px] text-text-muted">The first two ordered image references can be mapped here. Configure additional reference slots through the route API.</p>}
            {fields.map(field => {
              const usedByOther = new Set(Object.entries(fieldBindings).filter(([name, target]) => name !== field.name && target).map(([, target]) => target))
              const options = selected.bindings.filter(binding => !usedByOther.has(binding.name) && bindingMatchesField(field, binding))
              return <div key={field.name} className="grid grid-cols-[minmax(7rem,1fr)_minmax(9rem,1.4fr)] items-center gap-2">
                <label htmlFor={`${panelId}-map-${field.name.replace(/[^a-zA-Z0-9_-]/g, '-')}`} className="text-[10px] text-text-secondary">{field.label}</label>
                <select
                  id={`${panelId}-map-${field.name.replace(/[^a-zA-Z0-9_-]/g, '-')}`}
                  aria-label={`Native field mapping: ${field.label}`}
                  value={fieldBindings[field.name] ?? ''}
                  onChange={event => setMapping(field.name, event.target.value)}
                  className="min-w-0 rounded-md border border-border bg-bg-primary px-2 py-1.5 text-[10px] text-text-primary outline-none focus:border-accent-blue"
                >
                  <option value="">Not mapped</option>
                  {options.map(binding => <option key={binding.name} value={binding.name}>{binding.label || binding.name}</option>)}
                </select>
              </div>
            })}
          </fieldset>

          {unmappedBindings.length > 0 && <fieldset className="space-y-2 border-t border-border pt-3">
            <legend className="mb-1 text-[10px] font-semibold text-text-secondary">Workflow defaults</legend><p className="-mt-1 text-[9px] text-text-muted">These values belong to this ComfyUI workflow. Required values and server preflight must pass before routing can be saved.</p>
            {unmappedBindings.map(binding => {
              const safeName = binding.name.replace(/[^a-zA-Z0-9_-]/g, '-')
              const value = inputDefaults[binding.name] ?? binding.default
              const label = binding.label || binding.name
              const liveChoiceOptions = (binding.options ?? []).concat(workflowOptions?.choices[binding.name] ?? [])
                .filter((option, index, options) => options.indexOf(option) === index)
              const staleChoice = binding.type === 'choice' && value !== undefined && !liveChoiceOptions.includes(value)
              const choiceOptions = staleChoice ? [value, ...liveChoiceOptions] : liveChoiceOptions
              if (isMediaBinding(binding)) return <div key={binding.name} className="grid grid-cols-[minmax(7rem,1fr)_minmax(9rem,1.4fr)] items-center gap-2">
                <span className="text-[10px] text-text-secondary">{label}{binding.required ? <span className="ml-0.5 text-chip-red">*</span> : null}</span>
                <p className="text-[9px] text-text-muted">Map a Native media field above. Maestro uploads media for each run.</p>
              </div>
              const updateNumber = (event: ChangeEvent<HTMLInputElement>) => {
                if (!event.target.value) return setDefault(binding, undefined)
                const parsed = Number(event.target.value)
                if (Number.isFinite(parsed)) setDefault(binding, binding.type === 'integer' ? Math.trunc(parsed) : parsed)
              }
              return <div key={binding.name} className="grid grid-cols-[minmax(7rem,1fr)_minmax(9rem,1.4fr)] items-center gap-2">
                <label htmlFor={`${panelId}-default-${safeName}`} className="text-[10px] text-text-secondary">
                  {label}{binding.required ? <span className="ml-0.5 text-chip-red">*</span> : null}
                </label>
                {binding.type === 'boolean'
                  ? <label className="flex items-center gap-2 text-[10px] text-text-primary"><input id={`${panelId}-default-${safeName}`} type="checkbox" checked={value === true} onChange={event => setDefault(binding, event.target.checked)} /> Enabled</label>
                  : binding.type === 'integer' || binding.type === 'number'
                    ? <input id={`${panelId}-default-${safeName}`} aria-label={`Default for ${label}`} type="number" step={binding.type === 'integer' ? 1 : 'any'} min={binding.min} max={binding.max} value={typeof value === 'number' ? value : ''} onChange={updateNumber} className="min-w-0 rounded-md border border-border bg-bg-primary px-2 py-1.5 text-[10px] text-text-primary outline-none focus:border-accent-blue" />
                    : binding.type === 'choice' && choiceOptions.length > 0
                      ? <select id={panelId + '-default-' + safeName} aria-label={'Default for ' + label} value={value === undefined ? '-1' : String(choiceOptions.findIndex(option => option === value))} onChange={event => { const index = Number(event.target.value); setDefault(binding, index < 0 ? undefined : choiceOptions[index]) }} className="min-w-0 rounded-md border border-border bg-bg-primary px-2 py-1.5 text-[10px] text-text-primary outline-none focus:border-accent-blue">
                        {value === undefined && <option value="-1">Choose…</option>}
                        {choiceOptions.map((option, index) => <option key={binding.name + '-' + index} value={index}>{String(option)}{staleChoice && option === value ? ' · saved, no longer listed' : ''}</option>)}
                      </select>
                      : binding.type === 'string' && (workflowOptions?.choices[binding.name]?.length ?? 0) > 0
                        ? <div className="min-w-0">
                          <input id={panelId + '-default-' + safeName} aria-label={'Default for ' + label} list={panelId + '-choices-' + safeName} type="text" value={defaultText(value)} onChange={event => setDefault(binding, event.target.value)} placeholder="Choose a server value or enter one" className="min-w-0 w-full rounded-md border border-border bg-bg-primary px-2 py-1.5 text-[10px] text-text-primary outline-none placeholder:text-text-muted focus:border-accent-blue" />
                          <datalist id={panelId + '-choices-' + safeName}>{workflowOptions?.choices[binding.name]?.map((option, index) => <option key={binding.name + '-' + index} value={String(option)} />)}</datalist>
                          {workflowOptions && value !== undefined && !(workflowOptions.choices[binding.name] ?? []).some(option => String(option) === String(value)) && <p className="text-[8px] text-indicator-warning">Saved value is not in the server list; it is kept as a custom value.</p>}
                          <p className="mt-0.5 text-[8px] text-text-muted">ComfyUI server choices · custom values are allowed</p>
                        </div>
                        : <input id={panelId + '-default-' + safeName} aria-label={'Default for ' + label} type="text" value={defaultText(value)} onChange={event => setDefault(binding, event.target.value)} placeholder={binding.type === 'image' || binding.type === 'video' || binding.type === 'audio' ? 'Maestro media reference' : 'Fixed input value'} className="min-w-0 rounded-md border border-border bg-bg-primary px-2 py-1.5 text-[10px] text-text-primary outline-none placeholder:text-text-muted focus:border-accent-blue" />}
              </div>
            })}
          </fieldset>}

          {(missingRequiredDefaults.length > 0 || !promptMapped || duplicateTargets) && <div role="status" className="rounded-lg border border-amber-500/30 bg-amber-500/10 px-2.5 py-2 text-[9px] text-text-secondary">
            {!promptMapped && <p>Map Native Prompt to a workflow text binding.</p>}
            {duplicateTargets && <p>Each workflow binding can receive only one Native field.</p>}
            {missingRequiredDefaults.map(binding => <p key={binding.name}>Set a fixed value or Native mapping for required input {binding.label || binding.name}.</p>)}
          </div>}
          {validation?.valid && <p role="status" className="inline-flex items-center gap-1 text-[9px] text-indicator-success"><Check size={11} /> Workflow preflight passed. Saving routing does not start generation.</p>}
        </>}

        <div className="flex flex-wrap gap-2 border-t border-border pt-3">
          <button type="button" onClick={() => void verifySelected()} disabled={!canConfigure || validating || saving} className="rounded-lg border border-border bg-bg-primary px-2.5 py-2 text-[10px] text-text-secondary hover:text-text-primary disabled:opacity-50">
            {validating ? <Loader2 size={11} className="mr-1 inline animate-spin" /> : null}Validate workflow
          </button>
          <button type="button" onClick={() => void saveEnabledRoute()} disabled={!canConfigure || saving || validating} className="inline-flex items-center gap-1.5 rounded-lg bg-accent-blue px-2.5 py-2 text-[10px] font-medium text-white hover:brightness-110 disabled:opacity-50">
            {saving ? <Loader2 size={11} className="animate-spin" /> : <Save size={11} />}Save routing
          </button>
        </div>
        <p className="text-[9px] text-text-muted">Routing applies only to this native model. Saving or switching engines does not submit a workflow.</p>
      </div>
    </div>}
  </section>
}
