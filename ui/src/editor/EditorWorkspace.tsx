import { useEffect } from 'react'
import { FolderOpen, Layers3, SlidersHorizontal } from 'lucide-react'
import { useIsMobile } from '../lib/useIsMobile'
import { useStore } from '../stores/useStore'
import { EditorInspector } from './EditorInspector'
import { EditorMediaBin } from './EditorMediaBin'
import { EditorPreview } from './EditorPreview'
import { EditorTimeline } from './EditorTimeline'
import { EditorTopBar } from './EditorTopBar'
import { useEditorStore } from './useEditorStore'

export function EditorWorkspace() {
  const isMobile = useIsMobile()
  const activeWorkspace = useStore(state => state.activeWorkspace)
  const initialize = useEditorStore(state => state.initialize)
  const loading = useEditorStore(state => state.loading)
  const dirty = useEditorStore(state => state.dirty)
  const error = useEditorStore(state => state.error)
  const mobilePanel = useEditorStore(state => state.mobilePanel)
  const setMobilePanel = useEditorStore(state => state.setMobilePanel)
  const saveProject = useEditorStore(state => state.saveProject)
  const undo = useEditorStore(state => state.undo)
  const redo = useEditorStore(state => state.redo)
  const splitSelected = useEditorStore(state => state.splitSelected)
  const deleteSelected = useEditorStore(state => state.deleteSelected)
  const playing = useEditorStore(state => state.playing)
  const setPlaying = useEditorStore(state => state.setPlaying)

  useEffect(() => {
    void initialize(activeWorkspace)
  }, [activeWorkspace, initialize])

  useEffect(() => {
    if (!dirty) return
    const timer = window.setTimeout(() => void saveProject(), 1100)
    return () => window.clearTimeout(timer)
  }, [dirty, saveProject])

  useEffect(() => {
    const keyboard = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null
      const editingText = target?.tagName === 'INPUT' || target?.tagName === 'TEXTAREA' || target?.tagName === 'SELECT' || target?.isContentEditable
      const command = event.ctrlKey || event.metaKey
      if (command && event.key.toLowerCase() === 's') {
        event.preventDefault()
        void saveProject()
      } else if (command && event.key.toLowerCase() === 'z') {
        event.preventDefault()
        if (event.shiftKey) redo()
        else undo()
      } else if (command && event.key.toLowerCase() === 'y') {
        event.preventDefault()
        redo()
      } else if (!editingText && event.code === 'Space') {
        event.preventDefault()
        setPlaying(!playing)
      } else if (!editingText && event.key.toLowerCase() === 's') {
        event.preventDefault()
        splitSelected()
      } else if (!editingText && (event.key === 'Delete' || event.key === 'Backspace')) {
        event.preventDefault()
        deleteSelected()
      }
    }
    window.addEventListener('keydown', keyboard)
    return () => window.removeEventListener('keydown', keyboard)
  }, [deleteSelected, playing, redo, saveProject, setPlaying, splitSelected, undo])

  return (
    <main className="relative flex h-full w-full min-w-0 flex-col overflow-hidden bg-bg-primary text-text-primary">
      <EditorTopBar />

      {isMobile ? (
        <div className="flex min-h-0 flex-1 flex-col">
          <div className="min-h-[220px] flex-[0.9]">
            <EditorPreview />
          </div>
          <div className="min-h-0 flex-[1.1] border-t border-border">
            {mobilePanel === 'media' && <EditorMediaBin compact />}
            {mobilePanel === 'timeline' && <EditorTimeline compact />}
            {mobilePanel === 'inspector' && <EditorInspector compact />}
          </div>
          <nav className="grid h-13 shrink-0 grid-cols-3 border-t border-border bg-bg-secondary px-4 pb-[env(safe-area-inset-bottom)]">
            {([
              ['media', FolderOpen, 'Media'],
              ['timeline', Layers3, 'Timeline'],
              ['inspector', SlidersHorizontal, 'Adjust'],
            ] as const).map(([panel, Icon, label]) => (
              <button
                key={panel}
                type="button"
                onClick={() => setMobilePanel(panel)}
                className={`flex flex-col items-center justify-center gap-0.5 text-[9px] ${mobilePanel === panel ? 'text-accent-blue' : 'text-text-muted'}`}
              >
                <Icon size={16} /> {label}
              </button>
            ))}
          </nav>
        </div>
      ) : (
        <div className="flex min-h-0 flex-1">
          <EditorMediaBin />
          <div className="flex min-w-0 flex-1 flex-col">
            <EditorPreview />
            <EditorTimeline />
          </div>
          <EditorInspector />
        </div>
      )}

      {loading && (
        <div className="absolute inset-0 z-[100] flex items-center justify-center bg-bg-primary/75 backdrop-blur-sm">
          <div className="flex items-center gap-2 rounded-xl border border-border bg-bg-secondary px-4 py-3 text-xs text-text-secondary shadow-2xl">
            <span className="h-3 w-3 animate-spin rounded-full border-2 border-accent-blue border-t-transparent" />
            Opening Editor project…
          </div>
        </div>
      )}
      {error && (
        <div className="pointer-events-none absolute bottom-4 left-1/2 z-[110] max-w-[min(520px,calc(100vw-2rem))] -translate-x-1/2 rounded-lg border border-red-500/30 bg-red-950/90 px-3 py-2 text-[10px] text-red-200 shadow-xl">
          {error}
        </div>
      )}
    </main>
  )
}
