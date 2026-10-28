import { useEffect, useRef } from 'react'
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
  const drawerRef = useRef<HTMLElement>(null)
  const wasDrawerOpen = useRef(false)
  const drawerOpener = useRef<HTMLElement | null>(null)
  const tab: EditorSideTab = mobilePanel === 'inspector' ? 'inspector' : 'media'

  useEffect(() => {
    const drawer = drawerRef.current
    if (!mobile) return
    if (!sidebarOpen) {
      if (wasDrawerOpen.current) {
        wasDrawerOpen.current = false
        const activeElement = document.activeElement
        const backdrop = document.querySelector('[data-testid="editor-mobile-drawer-backdrop"]')
        if (drawer?.contains(activeElement) || activeElement === backdrop || activeElement === document.body) drawerOpener.current?.focus()
        drawerOpener.current = null
      }
      return
    }
    if (!wasDrawerOpen.current) {
      wasDrawerOpen.current = true
      drawerOpener.current = document.activeElement instanceof HTMLElement ? document.activeElement : null
      drawer?.querySelector<HTMLElement>('button[aria-pressed="true"]')?.focus()
    }

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        event.preventDefault()
        event.stopPropagation()
        setSidebarOpen(false)
        return
      }
      if (event.key !== 'Tab' || !drawer) return
      const controls = Array.from(drawer.querySelectorAll<HTMLElement>(
        'a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])',
      )).filter(element => element.offsetParent !== null)
      if (!controls.length) return
      const first = controls[0]
      const last = controls[controls.length - 1]
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault()
        last.focus()
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault()
        first.focus()
      }
    }
    document.addEventListener('keydown', handleKeyDown)
    return () => document.removeEventListener('keydown', handleKeyDown)
  }, [mobile, setSidebarOpen, sidebarOpen])

  if (!mobile) {
    return (
      <aside className="h-full min-h-0 w-full min-w-0 overflow-hidden border-r border-border bg-bg-secondary" aria-label="Editor media">
        <EditorMediaBin compact />
      </aside>
    )
  }

  return (
    <>
      {sidebarOpen && (
        <button
          type="button"
          aria-label="Close Editor panel"
          className="fixed inset-x-0 top-0 z-[140] cursor-default bg-black/50"
          style={{ bottom: 'calc(3.5rem + env(safe-area-inset-bottom))' }}
          data-testid="editor-mobile-drawer-backdrop"
          onClick={() => setSidebarOpen(false)}
        />
      )}
      <aside
        ref={drawerRef}
        role="dialog"
        aria-modal={sidebarOpen ? true : undefined}
        aria-label={tab === 'media' ? 'Editor media panel' : 'Editor adjustment panel'}
        aria-hidden={!sidebarOpen}
        inert={!sidebarOpen}
        data-testid="editor-mobile-drawer"
        className={`fixed left-0 top-0 z-[150] flex flex-col border-r border-border bg-bg-secondary shadow-2xl transition-transform duration-300 ease-in-out ${sidebarOpen ? 'translate-x-0' : '-translate-x-full'}`}
        style={{ width: 'min(380px, calc(100vw - 2.75rem))', bottom: 'calc(3.5rem + env(safe-area-inset-bottom))' }}
      >
        <div className="flex h-14 shrink-0 items-center justify-between border-b border-border px-4">
          <MaestroBrand />
          <div className="flex items-center gap-1.5">
            <AppModeToggle size="sm" />
            <button
              type="button"
              onClick={() => setSidebarOpen(false)}
              className="grid h-10 w-10 place-items-center rounded-lg text-text-secondary transition-colors hover:bg-bg-hover hover:text-text-primary focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent-blue"
              aria-label="Close Editor panel"
              title="Close Editor panel"
            >
              <X size={17} />
            </button>
          </div>
        </div>

        <div className="grid h-14 shrink-0 grid-cols-2 gap-1 border-b border-border bg-bg-secondary p-1.5" aria-label="Editor panel views">
          {([
            ['media', FolderOpen, 'Media'],
            ['inspector', SlidersHorizontal, 'Adjust'],
          ] as const).map(([value, Icon, label]) => (
            <button
              key={value}
              type="button"
              onClick={() => setMobilePanel(value)}
              aria-pressed={tab === value}
              className={`flex min-h-11 items-center justify-center gap-2 rounded-lg px-3 text-xs font-medium transition-colors focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent-blue ${tab === value ? 'bg-bg-active text-text-primary shadow-sm' : 'text-text-muted hover:bg-bg-hover hover:text-text-secondary'}`}
            >
              <Icon size={15} className={tab === value ? 'text-accent-blue' : ''} />
              {label}
            </button>
          ))}
        </div>

        <div className="min-h-0 flex-1">
          {tab === 'media' ? <EditorMediaBin compact /> : <EditorInspector compact />}
        </div>
      </aside>
    </>
  )
}
