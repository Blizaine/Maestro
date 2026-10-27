import { Settings, X } from 'lucide-react'
import { useStore } from '../../stores/useStore'
import { AppModeToggle, MaestroBrand, WorldsButton } from '../AppModeNavigation'

export function SidebarHeader({
  mobile = false,
  onOpenWorlds,
  worldsOpen = false,
  onSettings,
  onClose,
}: {
  mobile?: boolean
  onOpenWorlds?: () => void
  worldsOpen?: boolean
  onSettings: () => void
  onClose: () => void
}) {
  const worldsEnabled = useStore(s => s.servicesConfig?.worlds_enabled === true)
  const showWorlds = worldsEnabled && !!onOpenWorlds
  const size = mobile ? 'sm' : 'md'

  return (
    <header data-testid="sidecar-header" className="shrink-0 border-b border-border px-4 py-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <MaestroBrand compact={mobile} />
        <div className="flex max-w-full flex-wrap items-center gap-2">
          {!showWorlds && <AppModeToggle size={size} />}
          <button
            type="button"
            onClick={mobile ? onClose : onSettings}
            aria-label={mobile ? 'Close sidecar' : 'Settings'}
            title={mobile ? 'Close sidecar' : 'Settings'}
            className="shrink-0 rounded-lg p-1.5 text-text-secondary transition-colors hover:bg-bg-hover hover:text-text-primary"
          >
            {mobile ? <X size={16} /> : <Settings size={16} />}
          </button>
        </div>
      </div>
      {showWorlds && (
        <div className="mt-2 flex flex-wrap items-center gap-2">
          <AppModeToggle size={size} />
          <WorldsButton size="sm" open={worldsOpen} onClick={onOpenWorlds} />
        </div>
      )}
    </header>
  )
}
