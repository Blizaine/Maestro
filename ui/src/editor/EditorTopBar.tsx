import { useEffect, useRef, useState } from 'react'
import {
  Check,
  ChevronDown,
  Download,
  FilePlus2,
  Loader2,
  Redo2,
  Save,
  Settings,
  Trash2,
  Undo2,
} from 'lucide-react'
import { GlobalQueuePopover } from '../components/GlobalQueuePopover'
import { useIsMobile } from '../lib/useIsMobile'
import { useStore } from '../stores/useStore'
import { useEditorStore } from './useEditorStore'

const CANVAS_PRESETS = [
  { label: '16:9 · 1080p', width: 1920, height: 1080 },
  { label: '9:16 · 1080p', width: 1080, height: 1920 },
  { label: '1:1 · 1080p', width: 1080, height: 1080 },
  { label: '4:3 · 1080p', width: 1440, height: 1080 },
]

function ProjectNameField({
  projectName,
  onCommit,
}: {
  projectName: string
  onCommit: (name: string) => void
}) {
  const [draft, setDraft] = useState(projectName)
  const commit = () => {
    const next = draft.trim()
    if (next && next !== projectName) onCommit(next)
    else setDraft(projectName)
  }

  return (
    <input
      value={draft}
      onChange={event => setDraft(event.target.value)}
      onBlur={commit}
      onKeyDown={event => {
        if (event.key === 'Enter') event.currentTarget.blur()
        if (event.key === 'Escape') {
          setDraft(projectName)
          event.currentTarget.blur()
        }
      }}
      className="min-w-0 flex-1 bg-transparent px-2.5 py-1.5 text-xs text-text-primary outline-none"
      aria-label="Project name"
    />
  )
}

export function EditorTopBar() {
  const isMobile = useIsMobile()
  const rootRef = useRef<HTMLDivElement>(null)
  const [projectMenuOpen, setProjectMenuOpen] = useState(false)
  const [deleteConfirmId, setDeleteConfirmId] = useState<string | null>(null)
  const project = useEditorStore(state => state.project)
  const projects = useEditorStore(state => state.projects)
  const dirty = useEditorStore(state => state.dirty)
  const saving = useEditorStore(state => state.saving)
  const history = useEditorStore(state => state.history)
  const future = useEditorStore(state => state.future)
  const exportJobId = useEditorStore(state => state.exportJobId)
  const exportProgress = useEditorStore(state => state.exportProgress)
  const loadProject = useEditorStore(state => state.loadProject)
  const createProject = useEditorStore(state => state.createProject)
  const deleteProject = useEditorStore(state => state.deleteProject)
  const renameProject = useEditorStore(state => state.renameProject)
  const saveProject = useEditorStore(state => state.saveProject)
  const setCanvas = useEditorStore(state => state.setCanvas)
  const undo = useEditorStore(state => state.undo)
  const redo = useEditorStore(state => state.redo)
  const exportProject = useEditorStore(state => state.exportProject)
  const setSidebarMode = useStore(state => state.setSidebarMode)
  const toggleSettings = useStore(state => state.toggleSettings)

  useEffect(() => {
    if (!projectMenuOpen) return
    const close = (event: MouseEvent) => {
      if (rootRef.current && !rootRef.current.contains(event.target as Node)) {
        setProjectMenuOpen(false)
        setDeleteConfirmId(null)
      }
    }
    document.addEventListener('mousedown', close)
    return () => document.removeEventListener('mousedown', close)
  }, [projectMenuOpen])

  return (
    <header className={`flex shrink-0 items-center gap-2 border-b border-border bg-bg-secondary px-2.5 md:h-14 md:px-4 ${isMobile ? 'h-auto flex-wrap py-2' : 'h-12'}`}>
      <div className="flex shrink-0 items-center gap-2">
        <img src="/maestro-home-icon-orange.png" alt="" className="h-7 w-7 rounded-[7px] md:h-8 md:w-8" />
        {!isMobile && <span className="text-sm font-semibold tracking-tight text-text-primary">Maestro</span>}
      </div>

      <div className="flex shrink-0 rounded-lg border border-border bg-bg-tertiary p-0.5">
        {(['director', 'studio', 'editor'] as const).map(mode => (
          <button
            key={mode}
            type="button"
            onClick={() => setSidebarMode(mode)}
            className={`rounded-md px-2 py-1 text-[10px] capitalize transition-colors md:px-3 md:text-xs ${
              mode === 'editor'
                ? 'bg-toggle-active text-white shadow-accent-glow'
                : 'text-text-secondary hover:bg-bg-hover hover:text-text-primary'
            }`}
          >
            {mode}
          </button>
        ))}
      </div>

      <div ref={rootRef} className={`relative min-w-0 md:ml-2 md:max-w-[440px] ${isMobile ? 'order-3 w-full flex-none' : 'flex-1'}`}>
        <div className="flex min-w-0 items-center rounded-lg border border-border bg-bg-tertiary focus-within:border-accent-blue/60">
          <ProjectNameField
            key={project?.id || 'no-project'}
            projectName={project?.name || ''}
            onCommit={renameProject}
          />
          <button
            type="button"
            onClick={() => setProjectMenuOpen(open => {
              if (open) setDeleteConfirmId(null)
              return !open
            })}
            className="border-l border-border p-1.5 text-text-muted hover:bg-bg-hover hover:text-text-primary"
            title="Projects"
          >
            <ChevronDown size={14} />
          </button>
        </div>
        {projectMenuOpen && (
          <div className="absolute left-0 top-full z-[90] mt-1.5 w-[min(330px,calc(100vw-1rem))] overflow-hidden rounded-xl border border-border bg-bg-secondary shadow-2xl">
            <div className="flex items-center justify-between border-b border-border px-3 py-2">
              <span className="text-[10px] font-medium uppercase tracking-wider text-text-muted">Editor projects</span>
              <button
                type="button"
                onClick={() => { void createProject(); setProjectMenuOpen(false); setDeleteConfirmId(null) }}
                className="flex items-center gap-1 rounded-md bg-accent-blue/10 px-2 py-1 text-[10px] text-accent-blue hover:bg-accent-blue/20"
              >
                <FilePlus2 size={11} /> New
              </button>
            </div>
            <div className="max-h-64 overflow-y-auto p-1.5">
              {projects.map(summary => (
                <div key={summary.id} className="group flex items-center gap-1 rounded-lg hover:bg-bg-hover">
                  <button
                    type="button"
                    onClick={() => { void loadProject(summary.id); setProjectMenuOpen(false); setDeleteConfirmId(null) }}
                    className="min-w-0 flex-1 px-2 py-2 text-left"
                  >
                    <div className="flex items-center gap-1.5 text-xs text-text-primary">
                      {summary.id === project?.id && <Check size={11} className="shrink-0 text-accent-blue" />}
                      <span className="truncate">{summary.name}</span>
                    </div>
                    <div className="mt-0.5 text-[9px] text-text-muted">
                      {summary.asset_count} assets · {Math.max(0, summary.duration).toFixed(1)}s
                    </div>
                  </button>
                  <button
                    type="button"
                    onClick={() => {
                      if (deleteConfirmId === summary.id) {
                        void deleteProject(summary.id)
                        setDeleteConfirmId(null)
                      } else {
                        setDeleteConfirmId(summary.id)
                      }
                    }}
                    className={`mr-1 rounded p-1.5 opacity-0 group-hover:opacity-100 focus-visible:opacity-100 ${deleteConfirmId === summary.id ? 'bg-red-500/15 text-red-400 opacity-100' : 'text-text-muted hover:bg-red-500/10 hover:text-red-400'}`}
                    title={deleteConfirmId === summary.id ? 'Click again to confirm deletion' : 'Delete project'}
                  >
                    <Trash2 size={11} />
                  </button>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {!isMobile && project && (
        <select
          value={`${project.canvas.width}x${project.canvas.height}`}
          onChange={event => {
            const selected = CANVAS_PRESETS.find(preset => `${preset.width}x${preset.height}` === event.target.value)
            if (selected) setCanvas({ width: selected.width, height: selected.height })
          }}
          className="hidden rounded-lg border border-border bg-bg-tertiary px-2 py-1.5 text-[10px] text-text-secondary outline-none lg:block"
          title="Canvas size"
        >
          {CANVAS_PRESETS.map(preset => (
            <option key={preset.label} value={`${preset.width}x${preset.height}`}>{preset.label}</option>
          ))}
        </select>
      )}

      <div className={`ml-auto flex shrink-0 items-center gap-0.5 md:gap-1 ${isMobile ? 'order-2' : ''}`}>
        {!isMobile && (
          <>
            <button type="button" onClick={undo} disabled={history.length === 0} className="rounded-lg p-2 text-text-secondary hover:bg-bg-hover hover:text-text-primary disabled:opacity-25" title="Undo (Ctrl+Z)">
              <Undo2 size={15} />
            </button>
            <button type="button" onClick={redo} disabled={future.length === 0} className="rounded-lg p-2 text-text-secondary hover:bg-bg-hover hover:text-text-primary disabled:opacity-25" title="Redo (Ctrl+Shift+Z)">
              <Redo2 size={15} />
            </button>
          </>
        )}
        <button
          type="button"
          onClick={() => void saveProject()}
          disabled={saving || !project}
          className={`rounded-lg p-2 transition-colors ${dirty ? 'text-accent-blue hover:bg-accent-blue/10' : 'text-text-muted hover:bg-bg-hover'} disabled:opacity-40`}
          title={dirty ? 'Save project (Ctrl+S)' : 'Project saved'}
        >
          {saving ? <Loader2 size={15} className="animate-spin" /> : <Save size={15} />}
        </button>
        <button
          type="button"
          onClick={() => void exportProject()}
          disabled={!project || Boolean(exportJobId)}
          className="relative flex h-8 items-center gap-1.5 overflow-hidden rounded-lg bg-cta px-2.5 text-[10px] font-semibold text-white shadow-accent-glow disabled:opacity-50 md:px-3 md:text-xs"
          title="Export finished video"
        >
          {exportJobId ? <Loader2 size={13} className="animate-spin" /> : <Download size={13} />}
          <span>{exportJobId ? `${Math.round(exportProgress * 100)}%` : 'Export'}</span>
          {exportJobId && <span className="absolute inset-x-0 bottom-0 h-0.5 bg-white/20"><span className="block h-full bg-white" style={{ width: `${exportProgress * 100}%` }} /></span>}
        </button>
        <GlobalQueuePopover />
        <button type="button" onClick={toggleSettings} className="rounded-lg p-2 text-text-secondary hover:bg-bg-hover hover:text-text-primary" title="Settings">
          <Settings size={16} />
        </button>
      </div>
    </header>
  )
}
