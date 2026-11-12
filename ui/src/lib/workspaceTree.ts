/** Nested output workspaces.
 *
 *  A workspace name is "default" (the output folder itself) or a folder path
 *  below it: "ClientA", "ClientA/Shoot-1". The server lists parents before
 *  their children; older servers (and tests) may send only `name`, so every
 *  derived field falls back to parsing the name. */

export const WORKSPACE_DEFAULT = 'default'
export const WORKSPACE_MAX_DEPTH = 8
/** Fired on window after a folder is renamed or moved: detail {from, to}. */
export const WORKSPACE_RELOCATED_EVENT = 'maestro:workspace-relocated'
/** Folder names the server accepts for new, renamed and moved folders. */
export const WORKSPACE_SEGMENT_PATTERN = /^[A-Za-z0-9][A-Za-z0-9_-]*$/

export interface WorkspaceLike {
  name: string
  file_count?: number
  parent?: string | null
  depth?: number
  label?: string
  reserved?: boolean
}

export interface WorkspaceNode {
  name: string
  label: string
  parent: string | null
  depth: number
  fileCount: number
  reserved: boolean
  /** Direct sub-folders, in server order. */
  children: string[]
}

export function workspaceSegments(name: string): string[] {
  return !name || name === WORKSPACE_DEFAULT ? [] : name.split('/')
}

export function workspaceLabel(name: string): string {
  const segments = workspaceSegments(name)
  return segments.length ? segments[segments.length - 1] : WORKSPACE_DEFAULT
}

export function workspaceParent(name: string): string | null {
  const segments = workspaceSegments(name)
  if (!segments.length) return null
  return segments.length === 1 ? WORKSPACE_DEFAULT : segments.slice(0, -1).join('/')
}

export function joinWorkspace(parent: string | null | undefined, label: string): string {
  return !parent || parent === WORKSPACE_DEFAULT ? label : `${parent}/${label}`
}

/** True when `name` is `ancestor` or a folder below it ("A/B" is within "A"). */
export function isWithinWorkspace(name: string | undefined | null, ancestor: string | undefined | null): boolean {
  if (!name || !ancestor) return false
  if (ancestor === WORKSPACE_DEFAULT) return true
  return name === ancestor || name.startsWith(`${ancestor}/`)
}

/** Rename `old` (and everything below it) to `next`; other names are unchanged. */
export function rebaseWorkspaceName(name: string, old: string, next: string): string {
  return isWithinWorkspace(name, old) && old !== WORKSPACE_DEFAULT ? next + name.slice(old.length) : name
}

/** Rebase the `workspace` query of a Maestro file URL ("/api/v1/file/x.mp4?workspace=A"). */
export function rebaseWorkspaceUrl(url: string, old: string, next: string): string {
  const query = url.indexOf('?')
  if (query < 0 || !url.includes('workspace=')) return url
  const params = new URLSearchParams(url.slice(query + 1))
  const current = params.get('workspace')
  if (!current || !isWithinWorkspace(current, old) || old === WORKSPACE_DEFAULT) return url
  params.set('workspace', rebaseWorkspaceName(current, old, next))
  return `${url.slice(0, query)}?${params.toString()}`
}

/** Index the flat server list into nodes keyed by name, parents first. */
export function buildWorkspaceTree(workspaces: WorkspaceLike[]): Map<string, WorkspaceNode> {
  const nodes = new Map<string, WorkspaceNode>()
  for (const ws of workspaces) {
    const segments = workspaceSegments(ws.name)
    nodes.set(ws.name, {
      name: ws.name,
      label: ws.label || workspaceLabel(ws.name),
      parent: ws.parent !== undefined ? ws.parent : workspaceParent(ws.name),
      depth: ws.depth ?? segments.length,
      fileCount: ws.file_count ?? 0,
      reserved: Boolean(ws.reserved),
      children: [],
    })
  }
  for (const node of nodes.values()) {
    if (node.parent && node.parent !== WORKSPACE_DEFAULT) nodes.get(node.parent)?.children.push(node.name)
  }
  return nodes
}

/** Every folder below `name` (not including it). */
export function workspaceDescendants(nodes: Map<string, WorkspaceNode>, name: string): string[] {
  const out: string[] = []
  const stack = [...(nodes.get(name)?.children ?? [])].reverse()
  while (stack.length) {
    const next = stack.pop()!
    out.push(next)
    stack.push(...[...(nodes.get(next)?.children ?? [])].reverse())
  }
  return out
}

/** Deepest nesting below `name`, counted from `name` (0 when it has no sub-folders). */
function subtreeHeight(nodes: Map<string, WorkspaceNode>, name: string): number {
  const base = workspaceSegments(name).length
  return workspaceDescendants(nodes, name).reduce(
    (height, child) => Math.max(height, workspaceSegments(child).length - base), 0)
}

/** Why `source` cannot be dropped into `target`, or null when it can. */
export function workspaceDropProblem(
  nodes: Map<string, WorkspaceNode>,
  source: string,
  target: string,
): string | null {
  const sourceNode = nodes.get(source)
  if (!sourceNode || source === WORKSPACE_DEFAULT || sourceNode.reserved) return 'This folder cannot be moved.'
  if (target !== WORKSPACE_DEFAULT && !nodes.has(target)) return 'That folder no longer exists.'
  if (nodes.get(target)?.reserved) return 'That folder is managed by Maestro.'
  if (isWithinWorkspace(target, source) && target !== WORKSPACE_DEFAULT) return 'A folder cannot go inside itself.'
  if ((workspaceParent(source) ?? WORKSPACE_DEFAULT) === target) return 'It is already there.'
  const label = workspaceLabel(source)
  const destination = joinWorkspace(target, label)
  if (nodes.has(destination)) return `"${label}" already exists there.`
  if (workspaceSegments(destination).length + subtreeHeight(nodes, source) > WORKSPACE_MAX_DEPTH) {
    return `Folders can be nested at most ${WORKSPACE_MAX_DEPTH} deep.`
  }
  return null
}

/** Client-side check of a new folder name; null when the server should accept it. */
export function workspaceNameProblem(label: string): string | null {
  if (!label) return 'Enter a folder name.'
  if (label.length > 120) return 'That name is too long.'
  if (!WORKSPACE_SEGMENT_PATTERN.test(label)) return 'Use letters, numbers, hyphens and underscores.'
  if (label.toLowerCase() === WORKSPACE_DEFAULT) return '"default" is reserved.'
  return null
}

/** Typed text → folder name, the way "New Workspace" always has (spaces become hyphens). */
export function cleanWorkspaceLabel(text: string): string {
  return text.trim().replace(/\s+/g, '-')
}
