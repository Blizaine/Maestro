import { X } from 'lucide-react'
import { useStore } from '../../stores/useStore'
import { SystemSettingsPanel } from './SystemSettingsPanel'
import { ServicesSettingsPanel } from './ServicesSettingsPanel'

/**
 * Settings drawer — global panel for hardware/perf and external-service
 * configuration. Two tabs in both Studio and Director modes:
 *
 *   Performance    — VRAM coefficient, profile, hardware tier, etc.
 *                    (mounts <SystemSettingsPanel />)
 *   Integrations   — LLM provider, API keys, NSFW master gate, etc.
 *                    (mounts <ServicesSettingsPanel />)
 *
 * Director-mode-specific controls used to live in a third "Parameters"
 * tab here, but everything in that tab was either:
 *   - a duplicate of Studio's selection (image/video model, LoRAs)
 *   - a duplicate of Integrations (LLM model + device)
 *   - or a post-processing knob that's now in the Director chat sidebar
 *     under the "Advanced" accordion.
 *
 * Removing the tab makes Settings mode-agnostic — same layout in Studio
 * and Director — which matches the user's mental model of Settings as
 * "global preferences" vs Director's per-shoot setup which lives in
 * the chat sidebar where the work is happening.
 */
export function SettingsDrawer() {
  const settingsOpen = useStore(s => s.settingsOpen)
  const setSettingsOpen = useStore(s => s.setSettingsOpen)
  const settingsTab = useStore(s => s.settingsTab)
  const setSettingsTab = useStore(s => s.setSettingsTab)

  const tabs = [
    { id: 'performance' as const, label: 'Performance' },
    { id: 'integrations' as const, label: 'Integrations' },
  ]

  return (
    <>
      {/* Backdrop */}
      {settingsOpen && (
        <div
          className="fixed inset-0 bg-black/40 z-40"
          onClick={() => setSettingsOpen(false)}
        />
      )}

      {/* Drawer */}
      <div className={`fixed top-0 left-0 h-full w-full md:w-[420px] bg-bg-secondary border-r border-border z-50 transform transition-transform duration-300 ease-in-out ${
        settingsOpen ? 'translate-x-0' : '-translate-x-full'
      }`}>
        {/* Header */}
        <div className="px-5 py-3 border-b border-border flex items-center justify-between">
          <h2 className="font-semibold text-sm">Settings</h2>
          <button
            onClick={() => setSettingsOpen(false)}
            className="p-1.5 rounded-lg hover:bg-bg-hover text-text-secondary hover:text-text-primary transition-colors"
          >
            <X size={16} />
          </button>
        </div>

        {/* Tab Bar */}
        <div className="px-5 pt-3">
          <div className="flex bg-bg-tertiary rounded-lg p-0.5 border border-border">
            {tabs.map(tab => (
              <button
                key={tab.id}
                onClick={() => setSettingsTab(tab.id)}
                className={`flex-1 text-xs py-1.5 rounded-md transition-all ${
                  settingsTab === tab.id
                    ? 'bg-bg-active text-text-primary'
                    : 'text-text-secondary hover:text-text-primary'
                }`}
              >
                {tab.label}
              </button>
            ))}
          </div>
        </div>

        {/* Content */}
        <div className="overflow-y-auto h-[calc(100%-96px)] px-5 py-4 space-y-5">
          {settingsTab === 'performance' && (
            <SystemSettingsPanel />
          )}

          {settingsTab === 'integrations' && (
            <ServicesSettingsPanel />
          )}
        </div>
      </div>
    </>
  )
}
