import { FolderOpen, SlidersHorizontal, X } from 'lucide-react'
import { AppModeToggle, MaestroBrand } from '../components/AppModeNavigation'
import { useStore } from '../stores/useStore'
import { EditorInspector } from './EditorInspector'
import { EditorMediaBin } from './EditorMediaBin'
import { useEditorStore } from './useEditorStore'

type EditorSideTab = 'media' | 'inspector'

export function EditorLeftPanel({ mobile = false }: { mobile?: boolean }) {
  const sidebarOpen = useStore(state => state.sidebarOpen)
  const setSidebarOpen = useStore(state => state.setSidebarOpen)
  const mobilePanel = useEditorStore(state => state.mobilePanel)
  const setMobilePanel = useEditorStore(state => state.setMobilePanel)
  const tab: EditorSideTab = mobilePanel === 'inspector' ? 'inspector' : 'media'

  const content = (
    <aside
      className={mobile
        ? `fixed inset-y-0 left-0 z-[150] flex w-[380px] max-w-[85vw] flex-col border-r border-border bg-bg-secondary shadow-2xl transition-transform duration-300 ease-in-out ${sidebarOpen ? 'translate-x-0' : '-translate-x-full'}`
        : 'flex h-full w-[420px] shrink-0 flex-col border-r border-border bg-bg-secondary'}
      aria-label="Editor media and settings"
    >
      {mobile && (
        <div className="flex h-14 shrink-0 items-center justify-between border-b border-border px-4">
          <MaestroBrand />
          <div className="flex items-center gap-1.5">
            <AppModeToggle size="sm" />
            <button
              type="button"
              onClick={() => setSidebarOpen(false)}
              className="rounded-lg p-1.5 text-text-secondary transition-colors hover:bg-bg-hover hover:text-text-primary"
              title="Close Editor panel"
            >
              <X size={16} />
            </button>
          </div>
        </div>
      )}

      <div className="grid h-11 shrink-0 grid-cols-2 gap-1 border-b border-border bg-bg-secondary p-1.5">
        {([
          ['media', FolderOpen, 'Media'],
          ['inspector', SlidersHorizontal, 'Adjust'],
        ] as const).map(([value, Icon, label]) => (
          <button
            key={value}
            type="button"
            onClick={() => setMobilePanel(value)}
            className={`flex items-center justify-center gap-1.5 rounded-lg text-[10px] font-medium transition-colors ${tab === value ? 'bg-bg-active text-text-primary shadow-sm' : 'text-text-muted hover:bg-bg-hover hover:text-text-secondary'}`}
          >
            <Icon size={12} className={tab === value ? 'text-accent-blue' : ''} />
            {label}
          </button>
        ))}
      </div>

      <div className="min-h-0 flex-1">
        {tab === 'media' ? <EditorMediaBin compact /> : <EditorInspector compact />}
      </div>
    </aside>
  )

  if (!mobile) return content
  return (
    <>
      {sidebarOpen && (
        <button
          type="button"
          aria-label="Close Editor panel"
          className="fixed inset-0 z-[140] cursor-default bg-black/45"
          onClick={() => setSidebarOpen(false)}
        />
      )}
      {content}
    </>
  )
}
