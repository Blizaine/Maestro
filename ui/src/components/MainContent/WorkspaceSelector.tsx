import { useCallback, useEffect, useMemo, useRef, useState, type DragEvent, type MouseEvent } from 'react'
import { Check, ChevronDown, ChevronRight, FolderOpen, FolderPlus, Loader2, Pencil, Plus, Trash2, Upload, X } from 'lucide-react'
import { useStore } from '../../stores/useStore'
import {
  WORKSPACE_DEFAULT,
  buildWorkspaceTree,
  cleanWorkspaceLabel,
  workspaceDescendants,
  workspaceDropProblem,
  workspaceLabel,
  workspaceNameProblem,
  workspaceParent,
} from '../../lib/workspaceTree'

const DRAG_TYPE = 'application/x-maestro-workspace'
const EXPANDED_STORAGE_KEY = 'maestro_workspace_tree_expanded'
const INDENT_PX = 14
// Touch screens have no hover: keep row actions visible there.
const ROW_ACTION = 'opacity-0 group-hover:opacity-100 focus-visible:opacity-100 [@media(hover:none)]:opacity-100'

type Editing = { kind: 'create'; parent: string } | { kind: 'rename'; name: string } | null

function loadExpanded(): Set<string> {
  try {
    const saved = JSON.parse(localStorage.getItem(EXPANDED_STORAGE_KEY) || '[]')
    return new Set(Array.isArray(saved) ? saved.filter((name): name is string => typeof name === 'string') : [])
  } catch {
    return new Set()
  }
}

function saveExpanded(expanded: Set<string>) {
  try { localStorage.setItem(EXPANDED_STORAGE_KEY, JSON.stringify([...expanded])) } catch { /* private mode */ }
}

function ancestorsOf(name: string): string[] {
  const out: string[] = []
  let parent = workspaceParent(name)
  while (parent && parent !== WORKSPACE_DEFAULT) {
    out.push(parent)
    parent = workspaceParent(parent)
  }
  return out
}

function plural(count: number, word: string) {
  return `${count} ${word}${count === 1 ? '' : 's'}`
}

export function WorkspaceSelector() {
  const workspaces = useStore(s => s.workspaces)
  const activeWorkspace = useStore(s => s.activeWorkspace)
  const browsingAllFolders = useStore(s => s.browsingAllFolders)
  const browsingUploads = useStore(s => s.browsingUploads)
  const switchWorkspace = useStore(s => s.switchWorkspace)
  const createWorkspace = useStore(s => s.createWorkspace)
  const deleteWorkspace = useStore(s => s.deleteWorkspace)
  const renameWorkspace = useStore(s => s.renameWorkspace)
  const moveWorkspace = useStore(s => s.moveWorkspace)
  const [open, setOpen] = useState(false)
  const [editing, setEditing] = useState<Editing>(null)
  const [draft, setDraft] = useState('')
  const [pending, setPending] = useState<string | null>(null)
  const [confirmDelete, setConfirmDelete] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [expanded, setExpanded] = useState<Set<string>>(loadExpanded)
  const [dragging, setDragging] = useState<string | null>(null)
  const [dropTarget, setDropTarget] = useState<string | null>(null)
  const dropdownRef = useRef<HTMLDivElement>(null)
  const expandTimer = useRef<number | null>(null)
  const errorTimer = useRef<number | null>(null)

  const nodes = useMemo(() => buildWorkspaceTree(workspaces), [workspaces])
  const viewingFolder = !browsingAllFolders && !browsingUploads

  const showError = useCallback((message: string) => {
    setError(message)
    if (errorTimer.current) window.clearTimeout(errorTimer.current)
    errorTimer.current = window.setTimeout(() => setError(null), 6000)
  }, [])

  const setExpandedAndSave = useCallback((update: (current: Set<string>) => Set<string>) => {
    setExpanded(current => {
      const next = update(current)
      saveExpanded(next)
      return next
    })
  }, [])

  const expand = useCallback((names: string[]) => {
    setExpandedAndSave(current => {
      if (names.every(name => current.has(name))) return current
      const next = new Set(current)
      names.forEach(name => next.add(name))
      return next
    })
  }, [setExpandedAndSave])

  const toggleExpanded = (name: string) => {
    setExpandedAndSave(current => {
      const next = new Set(current)
      if (next.has(name)) next.delete(name)
      else next.add(name)
      return next
    })
  }

  const close = useCallback(() => {
    setOpen(false)
    setEditing(null)
    setConfirmDelete(null)
  }, [])

  // Opening the list reveals where generations are saving.
  useEffect(() => {
    if (open && activeWorkspace !== WORKSPACE_DEFAULT) expand(ancestorsOf(activeWorkspace))
  }, [open, activeWorkspace, expand])

  // Close on outside click
  useEffect(() => {
    if (!open) return
    const handler = (e: globalThis.MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target as Node)) close()
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [open, close])

  useEffect(() => () => {
    if (expandTimer.current) window.clearTimeout(expandTimer.current)
    if (errorTimer.current) window.clearTimeout(errorTimer.current)
  }, [])

  // Rows whose ancestors are all expanded, in server (depth-first) order.
  const visible = useMemo(() => workspaces.filter(ws => ancestorsOf(ws.name).every(name => expanded.has(name))),
    [workspaces, expanded])

  const startCreate = (parent: string, e?: MouseEvent) => {
    e?.stopPropagation()
    setEditing({ kind: 'create', parent })
    setDraft('')
    setConfirmDelete(null)
    if (parent !== WORKSPACE_DEFAULT) expand([parent, ...ancestorsOf(parent)])
  }

  const startRename = (name: string, e: MouseEvent) => {
    e.stopPropagation()
    setEditing({ kind: 'rename', name })
    setDraft(workspaceLabel(name))
    setConfirmDelete(null)
  }

  const commitEdit = async () => {
    if (!editing || pending) return
    const label = cleanWorkspaceLabel(draft)
    const problem = workspaceNameProblem(label)
    if (problem) {
      showError(problem)
      return
    }
    const target = editing.kind === 'rename' ? editing.name : editing.parent
    setPending(target)
    try {
      if (editing.kind === 'create') {
        await createWorkspace(label, editing.parent === WORKSPACE_DEFAULT ? undefined : editing.parent)
        setDraft('')
        close()
      } else {
        if (label !== workspaceLabel(editing.name)) await renameWorkspace(editing.name, label)
        setEditing(null)
      }
    } catch (err) {
      showError(err instanceof Error ? err.message : String(err))
    } finally {
      setPending(null)
    }
  }

  const handleDelete = async (name: string, e: MouseEvent) => {
    e.stopPropagation()
    if (confirmDelete !== name) {
      setConfirmDelete(name)
      setTimeout(() => setConfirmDelete(c => (c === name ? null : c)), 4000)
      return
    }
    setConfirmDelete(null)
    setPending(name)
    try {
      await deleteWorkspace(name)
    } catch (err) {
      showError(err instanceof Error ? err.message : String(err))
    } finally {
      setPending(null)
    }
  }

  const deleteSummary = (name: string) => {
    const descendants = workspaceDescendants(nodes, name)
    const files = [name, ...descendants].reduce((sum, folder) => sum + (nodes.get(folder)?.fileCount ?? 0), 0)
    return descendants.length
      ? `${plural(descendants.length, 'sub-folder')} and ${plural(files, 'file')}`
      : plural(files, 'file')
  }

  // --- Drag a folder onto another to make it a sub-folder ----------------
  const clearDrag = () => {
    if (expandTimer.current) window.clearTimeout(expandTimer.current)
    expandTimer.current = null
    setDragging(null)
    setDropTarget(null)
  }

  const handleDragStart = (name: string, e: DragEvent) => {
    e.dataTransfer.setData(DRAG_TYPE, name)
    e.dataTransfer.effectAllowed = 'move'
    setDragging(name)
    setEditing(null)
    setConfirmDelete(null)
  }

  const handleDragOver = (target: string, e: DragEvent) => {
    if (!dragging || workspaceDropProblem(nodes, dragging, target)) return
    e.preventDefault()
    e.dataTransfer.dropEffect = 'move'
    if (dropTarget === target) return
    setDropTarget(target)
    if (expandTimer.current) window.clearTimeout(expandTimer.current)
    expandTimer.current = null
    // Hovering a closed folder opens it so a deeper target can be reached.
    if (target !== WORKSPACE_DEFAULT && nodes.get(target)?.children.length && !expanded.has(target)) {
      expandTimer.current = window.setTimeout(() => expand([target]), 700)
    }
  }

  const handleDragLeave = (target: string, e: DragEvent) => {
    if (e.currentTarget.contains(e.relatedTarget as Node | null)) return
    if (dropTarget === target) setDropTarget(null)
  }

  const handleDrop = async (target: string, e: DragEvent) => {
    e.preventDefault()
    const source = dragging || e.dataTransfer.getData(DRAG_TYPE)
    clearDrag()
    if (!source) return
    const problem = workspaceDropProblem(nodes, source, target)
    if (problem) {
      showError(problem)
      return
    }
    setPending(source)
    try {
      await moveWorkspace(source, target)
      if (target !== WORKSPACE_DEFAULT) expand([target, ...ancestorsOf(target)])
    } catch (err) {
      showError(err instanceof Error ? err.message : String(err))
    } finally {
      setPending(null)
    }
  }

  const dropProps = (target: string) => ({
    onDragOver: (e: DragEvent) => handleDragOver(target, e),
    onDragLeave: (e: DragEvent) => handleDragLeave(target, e),
    onDrop: (e: DragEvent) => { void handleDrop(target, e) },
  })

  const nameInput = (depth: number, placeholder: string, label: string) => (
    <div className="flex items-center gap-1 py-1 pr-2" style={{ paddingLeft: 8 + depth * INDENT_PX }}>
      <input
        type="text"
        value={draft}
        onChange={e => setDraft(e.target.value)}
        onKeyDown={e => {
          if (e.key === 'Enter') void commitEdit()
          else if (e.key === 'Escape') { e.stopPropagation(); setEditing(null) }
        }}
        onClick={e => e.stopPropagation()}
        // Renaming starts from the current name, ready to be replaced.
        onFocus={e => e.currentTarget.select()}
        placeholder={placeholder}
        aria-label={label}
        disabled={pending !== null}
        className="min-w-0 flex-1 bg-bg-tertiary border border-border rounded px-2 py-1 text-xs text-text-primary focus:outline-none focus:border-accent-blue"
        autoFocus
      />
      <button
        onClick={e => { e.stopPropagation(); void commitEdit() }}
        disabled={!draft.trim() || pending !== null}
        className="p-1 rounded text-accent-blue hover:bg-bg-hover disabled:opacity-50"
        title="Save"
        aria-label="Save folder name"
      >
        {pending !== null ? <Loader2 size={12} className="animate-spin" /> : <Check size={12} />}
      </button>
      <button
        onClick={e => { e.stopPropagation(); setEditing(null) }}
        className="p-1 rounded text-text-muted hover:text-text-primary hover:bg-bg-hover"
        title="Cancel"
        aria-label="Cancel"
      >
        <X size={12} />
      </button>
    </div>
  )

  const triggerLabel = browsingAllFolders ? 'All folders' : browsingUploads ? 'Uploads' : workspaceLabel(activeWorkspace)

  return (
    <div className="relative" ref={dropdownRef}>
      <button
        onClick={() => (open ? close() : setOpen(true))}
        className="flex items-center gap-1.5 px-2 py-1 rounded-md text-xs text-text-secondary hover:text-text-primary hover:bg-bg-hover transition-colors border border-border"
        title="Switch workspace"
      >
        <FolderOpen size={12} />
        <span className="max-w-[120px] truncate" title={viewingFolder && activeWorkspace.includes('/') ? activeWorkspace : undefined}>{triggerLabel}</span>
      </button>

      {open && (
        <div className="absolute right-0 top-full mt-1 w-72 max-w-[calc(100vw-1rem)] bg-bg-secondary border border-border rounded-lg shadow-lg z-50 overflow-hidden">
          {/* While a sub-folder is dragged the header doubles as the "top
              level" target: the default row can be scrolled out of view. */}
          {dragging && workspaceParent(dragging) !== WORKSPACE_DEFAULT ? (
            <div
              {...dropProps(WORKSPACE_DEFAULT)}
              className={`px-2 py-1.5 border-b flex items-center gap-1.5 text-[10px] uppercase tracking-wider ${
                dropTarget === WORKSPACE_DEFAULT
                  ? 'border-accent-blue bg-accent-blue/15 text-accent-blue'
                  : 'border-border text-text-muted'
              }`}
            >
              <FolderOpen size={10} /> Drop here for the top level
            </div>
          ) : (
            <div className="px-2 py-1.5 border-b border-border flex items-center justify-between gap-2">
              <span className="text-[10px] text-text-muted uppercase tracking-wider">Workspaces</span>
              {dragging && <span className="text-[10px] text-text-muted truncate">Drop onto a folder to nest it</span>}
            </div>
          )}
          <button
            onClick={() => { switchWorkspace('__all__'); close() }}
            className={`w-full text-left px-3 py-2 text-xs flex items-center justify-between hover:bg-bg-hover ${browsingAllFolders ? 'text-accent-blue' : 'text-text-secondary'}`}
            title="Browse and search every output folder. New generations keep their current destination."
          >
            <span className="flex items-center gap-1.5"><FolderOpen size={12} /> All folders</span>
            {browsingAllFolders && <Check size={12} />}
          </button>
          <div className="max-h-[320px] overflow-y-auto" role="tree" aria-label="Workspaces">
            {visible.map(ws => {
              const node = nodes.get(ws.name)
              const isDefault = ws.name === WORKSPACE_DEFAULT
              const depth = isDefault ? 0 : Math.max(0, (node?.depth ?? 1) - 1)
              const hasChildren = !isDefault && (node?.children.length ?? 0) > 0
              const isExpanded = expanded.has(ws.name)
              const isActive = viewingFolder && ws.name === activeWorkspace
              const reserved = Boolean(node?.reserved)
              const movable = !isDefault && !reserved && pending === null && editing === null
              const isPending = pending === ws.name
              const label = node?.label ?? ws.name
              const renaming = editing?.kind === 'rename' && editing.name === ws.name
              const isDropTarget = dropTarget === ws.name
              return (
                <div key={ws.name} role="none">
                  {renaming ? nameInput(depth, 'folder-name', `Rename ${label}`) : (
                    <div
                      role="treeitem"
                      aria-level={depth + 1}
                      aria-expanded={hasChildren ? isExpanded : undefined}
                      aria-selected={isActive}
                      draggable={movable}
                      onDragStart={movable ? e => handleDragStart(ws.name, e) : undefined}
                      onDragEnd={clearDrag}
                      {...dropProps(ws.name)}
                      className={`flex items-center group transition-colors ${
                        isDropTarget ? 'bg-accent-blue/15 ring-1 ring-inset ring-accent-blue' : 'hover:bg-bg-hover'
                      } ${dragging === ws.name ? 'opacity-50' : ''}`}
                      style={{ paddingLeft: 4 + depth * INDENT_PX }}
                    >
                      {hasChildren ? (
                        <button
                          onClick={e => { e.stopPropagation(); toggleExpanded(ws.name) }}
                          className="p-1 shrink-0 text-text-muted hover:text-text-primary"
                          aria-label={`${isExpanded ? 'Collapse' : 'Expand'} ${label}`}
                          title={isExpanded ? 'Collapse' : 'Expand'}
                        >
                          {isExpanded ? <ChevronDown size={11} /> : <ChevronRight size={11} />}
                        </button>
                      ) : <span className="w-[19px] shrink-0" aria-hidden="true" />}
                      <button
                        onClick={() => { switchWorkspace(ws.name); close() }}
                        className={`flex-1 min-w-0 text-left pr-2 py-2 text-xs flex items-center justify-between ${
                          isActive ? 'text-accent-blue' : 'text-text-secondary'
                        }`}
                        title={movable ? `${ws.name} — drag onto another folder to move it` : ws.name}
                      >
                        <span className="truncate">{label}</span>
                        {isActive && <Check size={12} className="shrink-0" />}
                      </button>
                      {!isDefault && !reserved && (
                        <>
                          <button
                            onClick={e => startCreate(ws.name, e)}
                            disabled={pending !== null}
                            className={`px-1.5 py-2 shrink-0 text-text-muted hover:text-accent-blue transition-colors ${ROW_ACTION}`}
                            title={`New sub-folder in "${label}"`}
                            aria-label={`New sub-folder in ${label}`}
                          >
                            <FolderPlus size={12} />
                          </button>
                          <button
                            onClick={e => startRename(ws.name, e)}
                            disabled={pending !== null}
                            className={`px-1.5 py-2 shrink-0 text-text-muted hover:text-text-primary transition-colors ${ROW_ACTION}`}
                            title={`Rename "${label}"`}
                            aria-label={`Rename ${label}`}
                          >
                            <Pencil size={12} />
                          </button>
                        </>
                      )}
                      {/* default IS the outputs folder itself — not deletable */}
                      {!isDefault && (
                        <button
                          onClick={e => handleDelete(ws.name, e)}
                          disabled={isPending}
                          className={`px-2 py-2 shrink-0 transition-colors ${
                            confirmDelete === ws.name
                              ? 'text-red-400 bg-red-500/15'
                              : isPending
                                ? 'text-text-muted cursor-wait'
                                : `text-text-muted hover:text-red-400 ${ROW_ACTION}`
                          }`}
                          title={confirmDelete === ws.name
                            ? `Click again to permanently delete "${ws.name}" and its ${deleteSummary(ws.name)}`
                            : `Delete workspace (${deleteSummary(ws.name)})`}
                          aria-label={`Delete ${label}`}
                        >
                          {isPending ? <Loader2 size={12} className="animate-spin" /> : <Trash2 size={12} />}
                        </button>
                      )}
                    </div>
                  )}
                  {editing?.kind === 'create' && editing.parent === ws.name && !isDefault &&
                    nameInput(depth + 1, 'sub-folder-name', `New sub-folder in ${label}`)}
                </div>
              )
            })}
          </div>
          {error && (
            <div className="px-3 py-1.5 text-[10px] text-red-400 border-t border-border leading-snug" role="alert">{error}</div>
          )}
          {/* Virtual Uploads view — browse user-uploaded media (read-only;
              generations keep saving to the real active workspace). */}
          <div className="border-t border-border">
            <button
              onClick={() => { switchWorkspace('__uploads__'); close() }}
              className={`w-full text-left px-3 py-2 text-xs flex items-center justify-between hover:bg-bg-hover transition-colors ${
                browsingUploads ? 'text-accent-blue' : 'text-text-secondary'
              }`}
              title="Browse media you've uploaded — reuse as inputs"
            >
              <span className="flex items-center gap-1.5"><Upload size={12} /> Uploads</span>
              {browsingUploads && <Check size={12} />}
            </button>
          </div>
          <div className="border-t border-border p-2">
            {editing?.kind === 'create' && editing.parent === WORKSPACE_DEFAULT ? (
              <div className="-mx-2 -my-1">{nameInput(0, 'workspace-name', 'New workspace name')}</div>
            ) : (
              <button
                onClick={() => startCreate(WORKSPACE_DEFAULT)}
                className="w-full text-left px-1 py-1 text-xs text-accent-blue hover:text-accent-blue-hover flex items-center gap-1"
              >
                <Plus size={12} /> New Workspace
              </button>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
