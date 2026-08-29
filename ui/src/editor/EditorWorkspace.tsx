import { useEffect } from 'react'
import { useIsMobile } from '../lib/useIsMobile'
import { useStore } from '../stores/useStore'
import { EditorLeftPanel } from './EditorLeftPanel'
import { EditorPreview } from './EditorPreview'
import { EditorTimeline } from './EditorTimeline'
import { EditorTopBar } from './EditorTopBar'
import { editorProjectDuration } from './editorUtils'
import { useEditorStore } from './useEditorStore'

export function EditorWorkspace() {
  const isMobile = useIsMobile()
  const activeWorkspace = useStore(state => state.activeWorkspace)
  const initialize = useEditorStore(state => state.initialize)
  const loading = useEditorStore(state => state.loading)
  const dirty = useEditorStore(state => state.dirty)
  const project = useEditorStore(state => state.project)
  const error = useEditorStore(state => state.error)
  const saveProject = useEditorStore(state => state.saveProject)
  const undo = useEditorStore(state => state.undo)
  const redo = useEditorStore(state => state.redo)
  const splitSelected = useEditorStore(state => state.splitSelected)
  const deleteSelected = useEditorStore(state => state.deleteSelected)
  const duplicateSelected = useEditorStore(state => state.duplicateSelected)
  const copySelected = useEditorStore(state => state.copySelected)
  const cutSelected = useEditorStore(state => state.cutSelected)
  const pasteClipboard = useEditorStore(state => state.pasteClipboard)
  const jumpToEdit = useEditorStore(state => state.jumpToEdit)
  const addMarker = useEditorStore(state => state.addMarker)
  const selectItem = useEditorStore(state => state.selectItem)
  const playhead = useEditorStore(state => state.playhead)
  const setPlayhead = useEditorStore(state => state.setPlayhead)
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
      } else if (command && !editingText && event.key.toLowerCase() === 'c') {
        event.preventDefault()
        copySelected()
      } else if (command && !editingText && event.key.toLowerCase() === 'x') {
        event.preventDefault()
        cutSelected()
      } else if (command && !editingText && event.key.toLowerCase() === 'v') {
        event.preventDefault()
        pasteClipboard()
      } else if (command && !editingText && event.key.toLowerCase() === 'd') {
        event.preventDefault()
        duplicateSelected()
      } else if (!editingText && event.code === 'Space') {
        event.preventDefault()
        setPlaying(!playing)
      } else if (!editingText && event.key.toLowerCase() === 's') {
        event.preventDefault()
        splitSelected()
      } else if (!editingText && event.key.toLowerCase() === 'm') {
        event.preventDefault()
        addMarker(playhead)
      } else if (!editingText && event.key.toLowerCase() === 'j') {
        event.preventDefault()
        setPlaying(false)
        setPlayhead(Math.max(0, playhead - (event.shiftKey ? 5 : 1)))
      } else if (!editingText && event.key.toLowerCase() === 'k') {
        event.preventDefault()
        setPlaying(false)
      } else if (!editingText && event.key.toLowerCase() === 'l') {
        event.preventDefault()
        if (playhead >= editorProjectDuration(project)) setPlayhead(0)
        setPlaying(true)
      } else if (!editingText && (event.key === 'Delete' || event.key === 'Backspace')) {
        event.preventDefault()
        deleteSelected()
      } else if (!editingText && event.key === 'ArrowLeft') {
        event.preventDefault()
        const amount = event.shiftKey ? 1 : 1 / Math.max(1, project?.canvas.fps || 30)
        setPlaying(false)
        setPlayhead(playhead - amount)
      } else if (!editingText && event.key === 'ArrowRight') {
        event.preventDefault()
        const amount = event.shiftKey ? 1 : 1 / Math.max(1, project?.canvas.fps || 30)
        setPlaying(false)
        setPlayhead(Math.min(editorProjectDuration(project), playhead + amount))
      } else if (!editingText && event.key === 'ArrowUp') {
        event.preventDefault()
        jumpToEdit(-1)
      } else if (!editingText && event.key === 'ArrowDown') {
        event.preventDefault()
        jumpToEdit(1)
      } else if (!editingText && event.key === 'Home') {
        event.preventDefault()
        setPlaying(false)
        setPlayhead(0)
      } else if (!editingText && event.key === 'End') {
        event.preventDefault()
        setPlaying(false)
        setPlayhead(editorProjectDuration(project))
      } else if (!editingText && event.key === 'Escape') {
        selectItem(null, null)
      }
    }
    window.addEventListener('keydown', keyboard)
    return () => window.removeEventListener('keydown', keyboard)
  }, [addMarker, copySelected, cutSelected, deleteSelected, duplicateSelected, jumpToEdit, pasteClipboard, playhead, playing, project, redo, saveProject, selectItem, setPlayhead, setPlaying, splitSelected, undo])

  return (
    <main className="relative flex h-full w-full min-w-0 flex-col overflow-hidden bg-bg-primary text-text-primary">
      <EditorTopBar />

      {isMobile ? (
        <>
          <EditorLeftPanel mobile />
          <div className="flex min-h-0 flex-1 flex-col">
            <div className="min-h-[220px] flex-[0.9]">
              <EditorPreview />
            </div>
            <div className="min-h-0 flex-[1.1] border-t border-border">
              <EditorTimeline compact />
            </div>
          </div>
        </>
      ) : (
        <div className="flex min-h-0 flex-1">
          <EditorLeftPanel />
          <div className="flex min-w-0 flex-1 flex-col">
            <EditorPreview />
            <EditorTimeline />
          </div>
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
