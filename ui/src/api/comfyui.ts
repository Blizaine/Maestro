/** Typed API client for the optional ComfyUI integration. */

export type ComfyUIKind = 'image' | 'video' | 'audio'
export type ComfyUIBindingType = 'string' | 'integer' | 'number' | 'boolean' | 'choice' | 'image' | 'video' | 'audio'
export type ComfyUIScalar = string | number | boolean

export const COMFYUI_STATE_CHANGED_EVENT = 'maestro:comfyui-state-changed'
export const COMFYUI_OPEN_WORKFLOW_EVENT = 'maestro:comfyui-open-workflow'

export interface ComfyUIWorkflowLaunchRequest {
  workflowId: string
  inputs: Record<string, ComfyUIScalar>
}

export function notifyComfyUIStateChanged(): void {
  if (typeof window !== 'undefined') window.dispatchEvent(new Event(COMFYUI_STATE_CHANGED_EVENT))
}

export function requestOpenComfyUIWorkflow(workflowId: string, inputs: Record<string, ComfyUIScalar>): void {
  if (typeof window !== 'undefined') {
    window.dispatchEvent(new CustomEvent<ComfyUIWorkflowLaunchRequest>(COMFYUI_OPEN_WORKFLOW_EVENT, {
      detail: { workflowId, inputs },
    }))
  }
}

export interface ComfyUIBinding {
  name: string
  label?: string
  node_id: string
  input: string
  type: ComfyUIBindingType
  required?: boolean
  default?: ComfyUIScalar
  options?: ComfyUIScalar[]
  min?: number
  max?: number
}

export interface ComfyUIWorkflow {
  id: string
  name: string
  kind: ComfyUIKind
  description?: string
  workflow: Record<string, unknown>
  bindings: ComfyUIBinding[]
  template?: boolean
}

export interface ComfyUIRouteStatus {
  workflow_id: string
  enabled: boolean
  kind?: ComfyUIKind
  name?: string
  field_bindings?: Record<string, string>
  input_defaults?: Record<string, ComfyUIScalar>
}

export interface SaveComfyUIRouteInput {
  workflow_id: string
  enabled: boolean
  field_bindings: Record<string, string>
  input_defaults: Record<string, ComfyUIScalar>
  kind: ComfyUIKind
}

export interface ComfyUIPendingSubmission {
  submission_id: string
  job_id?: string | null
  state: string
  error?: string | null
}

export interface ComfyUIStatus {
  enabled: boolean
  endpoint: string
  dedicated?: boolean
  execution_timeout?: number
  connected?: boolean
  connection_error?: string | null
  installations?: ComfyUIInstallation[]
  routes?: Record<string, ComfyUIRouteStatus>
  templates?: ComfyUIWorkflow[]
  pending_submissions?: ComfyUIPendingSubmission[]
}

export interface ComfyUIConnectionTest {
  connected: boolean
  error?: string
  connection_error?: string | null
  node_count?: number
  system?: Record<string, unknown>
  missing?: string[]
}

export interface ComfyUIEndpointCandidate {
  endpoint: string
  connected: boolean
  node_count?: number
  error?: string
}

export interface ComfyUIEndpointDiscovery {
  candidates: ComfyUIEndpointCandidate[]
}

export interface ComfyUIWorkflowOptions {
  workflow_id: string
  choices: Record<string, ComfyUIScalar[]>
}

export interface ComfyUIModelPath {
  category: string
  path: string
}

export interface ComfyUIInstallation {
  id: string
  name: string
  path: string
  model_paths: ComfyUIModelPath[]
  available: boolean
}

export interface ComfyUIModelFile {
  id: string
  path: string
  filename: string
  category: string
  size: number
  compatibility: string | boolean
  reason?: string
  relative_path?: string
}

export interface ComfyUIModelDiscovery {
  installations: ComfyUIInstallation[]
}

export interface ComfyUIModelScan {
  roots: string[]
  files: ComfyUIModelFile[]
  truncated: boolean
}

export interface ComfyUIModelLinkResult {
  linked_paths: string[]
}

export interface ComfyUIWorkflowValidation {
  valid: boolean
  errors: unknown[]
  missing_nodes: unknown[]
  missing_models: unknown[]
}

export interface ComfyUIJobReceipt {
  submission_id?: string
  client_submission_id?: string
  job_id?: string | null
  prompt_id?: string | null
  status?: string
  state?: string
  detail?: string | null
  error?: string | null
}

async function requestJson<T>(path: string, init: RequestInit = {}): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: {
      ...(init.body ? { 'Content-Type': 'application/json' } : {}),
      ...init.headers,
    },
  })
  if (!response.ok) {
    const error = await response.json().catch(() => ({})) as { detail?: string; error?: string; message?: string }
    throw new Error(error.detail || error.error || error.message || `ComfyUI request failed (${response.status}).`)
  }
  return response.json() as Promise<T>
}

export function fetchComfyUIStatus(): Promise<ComfyUIStatus> {
  return requestJson('/api/v1/comfyui/status')
}

export async function saveComfyUISettings(settings: { enabled: boolean; endpoint: string; dedicated?: boolean }): Promise<ComfyUIStatus> {
  const result = await requestJson<ComfyUIStatus>('/api/v1/comfyui/settings', {
    method: 'PUT',
    body: JSON.stringify(settings),
  })
  notifyComfyUIStateChanged()
  return result
}

export async function testComfyUIConnection(): Promise<ComfyUIConnectionTest> {
  const result = await requestJson<ComfyUIConnectionTest>('/api/v1/comfyui/connection/test', {
    method: 'POST',
    body: JSON.stringify({}),
  })
  notifyComfyUIStateChanged()
  return result
}

export function discoverComfyUIConnections(signal?: AbortSignal): Promise<ComfyUIEndpointDiscovery> {
  return requestJson('/api/v1/comfyui/connection/discover', {
    method: 'POST',
    body: JSON.stringify({}),
    signal,
  })
}

export function discoverComfyUIModels(paths?: string[]): Promise<ComfyUIModelDiscovery> {
  return requestJson('/api/v1/comfyui/models/discover', {
    method: 'POST',
    body: JSON.stringify(paths?.length ? { paths } : {}),
  })
}

export function scanComfyUIModels(paths: string[]): Promise<ComfyUIModelScan> {
  return requestJson('/api/v1/comfyui/models/scan', {
    method: 'POST',
    body: JSON.stringify({ paths }),
  })
}

export async function linkComfyUIModels(paths: string[]): Promise<ComfyUIModelLinkResult> {
  const result = await requestJson<ComfyUIModelLinkResult>('/api/v1/comfyui/models/link', {
    method: 'POST',
    body: JSON.stringify({ paths }),
  })
  notifyComfyUIStateChanged()
  return result
}

export async function fetchComfyUIWorkflows(): Promise<{ workflows: ComfyUIWorkflow[]; templates: ComfyUIWorkflow[] }> {
  const result = await requestJson<{ workflows?: ComfyUIWorkflow[]; templates?: ComfyUIWorkflow[] }>('/api/v1/comfyui/workflows')
  return { workflows: result.workflows ?? [], templates: result.templates ?? [] }
}

export async function createComfyUIWorkflow(input: {
  name: string
  kind: ComfyUIKind
  workflow: Record<string, unknown>
  bindings: ComfyUIBinding[]
}): Promise<ComfyUIWorkflow> {
  const result = await requestJson<ComfyUIWorkflow>('/api/v1/comfyui/workflows', {
    method: 'POST',
    body: JSON.stringify(input),
  })
  notifyComfyUIStateChanged()
  return result
}

export async function deleteComfyUIWorkflow(id: string): Promise<{ deleted: boolean }> {
  const result = await requestJson<{ deleted: boolean }>('/api/v1/comfyui/workflows/' + encodeURIComponent(id), { method: 'DELETE' })
  notifyComfyUIStateChanged()
  return result
}

export function fetchComfyUIWorkflowOptions(id: string, signal?: AbortSignal): Promise<ComfyUIWorkflowOptions> {
  return requestJson('/api/v1/comfyui/workflows/' + encodeURIComponent(id) + '/options', { signal })
}

export function validateComfyUIWorkflow(
  id: string,
  inputs: Record<string, ComfyUIScalar> = {},
): Promise<ComfyUIWorkflowValidation> {
  return requestJson(`/api/v1/comfyui/workflows/${encodeURIComponent(id)}/validate`, {
    method: 'POST',
    body: JSON.stringify({ inputs }),
  })
}

export function runComfyUIWorkflow(id: string, input: {
  client_submission_id: string
  inputs: Record<string, ComfyUIScalar>
  workspace: string
}): Promise<ComfyUIJobReceipt> {
  return requestJson(`/api/v1/comfyui/workflows/${encodeURIComponent(id)}/run`, {
    method: 'POST',
    body: JSON.stringify(input),
  })
}

export function getComfyUIJobBySubmission(submissionId: string): Promise<ComfyUIJobReceipt> {
  return requestJson('/api/v1/comfyui/jobs/by-submission/' + encodeURIComponent(submissionId))
}

export async function resumeComfyUIJobBySubmission(submissionId: string): Promise<ComfyUIJobReceipt> {
  const result = await requestJson<ComfyUIJobReceipt>('/api/v1/comfyui/jobs/by-submission/' + encodeURIComponent(submissionId) + '/resume', {
    method: 'POST',
    body: JSON.stringify({}),
  })
  notifyComfyUIStateChanged()
  return result
}

export async function abandonComfyUISubmission(submissionId: string): Promise<{ abandoned: boolean }> {
  const result = await requestJson<{ abandoned: boolean }>('/api/v1/comfyui/jobs/by-submission/' + encodeURIComponent(submissionId) + '/abandon', {
    method: 'POST',
    body: JSON.stringify({}),
  })
  notifyComfyUIStateChanged()
  return result
}

export async function saveComfyUIRoute(modelType: string, route: SaveComfyUIRouteInput): Promise<ComfyUIRouteStatus> {
  const result = await requestJson<ComfyUIRouteStatus>('/api/v1/comfyui/routes/' + encodeURIComponent(modelType), {
    method: 'PUT',
    body: JSON.stringify(route),
  })
  notifyComfyUIStateChanged()
  return result
}
