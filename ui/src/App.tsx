import { useEffect } from 'react'
import { Menu, Settings } from 'lucide-react'
import { Sidebar } from './components/Sidebar/Sidebar'
import { MainContent } from './components/MainContent/MainContent'
import { SettingsDrawer } from './components/SettingsDrawer/SettingsDrawer'
import { LoraBrowser } from './components/LoraBrowser/LoraBrowser'
import { DirectorDashboard } from './components/DirectorDashboard/DirectorDashboard'
import { StorageDashboard } from './components/StorageDashboard/StorageDashboard'
import { RetakeDialog } from './components/RetakeDialog'
import { OomRecoveryBanner } from './components/OomRecoveryBanner'
import { DownloadStatusBanner } from './components/DownloadStatusBanner'
import { PreflightBanner } from './components/PreflightBanner'
import { WelcomeModal } from './components/WelcomeModal'
import { RecipesOverlay } from './components/Recipes/RecipesOverlay'
import { useStore } from './stores/useStore'
import { useIsMobile } from './lib/useIsMobile'

function App() {
  const loadModels = useStore(s => s.loadModels)
  const loadOutputs = useStore(s => s.loadOutputs)
  const loadWorkspaces = useStore(s => s.loadWorkspaces)
  const reconnectJobs = useStore(s => s.reconnectJobs)
  const loadSystemConfig = useStore(s => s.loadSystemConfig)
  const loadServicesConfig = useStore(s => s.loadServicesConfig)
  const loadLlmStatus = useStore(s => s.loadLlmStatus)
  const loadLlmModels = useStore(s => s.loadLlmModels)
  const toggleSidebar = useStore(s => s.toggleSidebar)
  const setSidebarOpen = useStore(s => s.setSidebarOpen)
  const toggleSettings = useStore(s => s.toggleSettings)
  const appVersion = useStore(s => s.systemConfig?.app_version)
  const isMobile = useIsMobile()

  useEffect(() => {
    loadModels()
    loadWorkspaces()
    loadOutputs()
    loadSystemConfig()
    loadServicesConfig()
    loadLlmStatus()
    loadLlmModels()
    reconnectJobs()
  }, [loadModels, loadWorkspaces, loadOutputs, loadSystemConfig, loadServicesConfig, loadLlmStatus, loadLlmModels, reconnectJobs])

  // Poll LLM status to stay in sync with backend auto-load/unload
  useEffect(() => {
    const interval = setInterval(loadLlmStatus, 15000)
    return () => clearInterval(interval)
  }, [loadLlmStatus])

  return (
    <div className="flex flex-col md:flex-row h-full w-full bg-bg-primary">
      {/* Mobile header */}
      {isMobile && (
        <header className="h-12 shrink-0 px-4 border-b border-border flex items-center justify-between bg-bg-secondary">
          <button
            onClick={toggleSidebar}
            className="p-2 rounded-lg hover:bg-bg-hover text-text-secondary hover:text-text-primary transition-colors"
          >
            <Menu size={20} />
          </button>
          <div className="flex items-center gap-2">
            <div className="w-7 h-7 rounded-lg bg-accent-blue flex items-center justify-center text-white font-bold text-sm">
              M
            </div>
            <span className="font-semibold text-sm">Maestro</span>
            {appVersion && <span className="text-[10px] text-text-muted font-normal mt-0.5">v{appVersion}</span>}
          </div>
          <button
            onClick={() => { setSidebarOpen(false); toggleSettings() }}
            className="p-2 rounded-lg hover:bg-bg-hover text-text-secondary hover:text-text-primary transition-colors"
          >
            <Settings size={20} />
          </button>
        </header>
      )}

      <Sidebar />
      <MainContent />
      <SettingsDrawer />
      <LoraBrowser />
      <DirectorDashboard />
      <StorageDashboard />
      <RecipesOverlay />
      <RetakeDialog />
      {/* OomRecoveryBanner is a fixed-position overlay — renders nothing
          unless the latest job/pipeline failure has oom_info attached.
          Lives at the App root so it floats above whichever screen the
          user is looking at when their generation OOMs. */}
      <OomRecoveryBanner />
      {/* PreflightBanner — fixed top overlay shown once on startup if the
          environment is missing ffmpeg / CUDA or low on disk. Renders
          nothing when everything checks out. */}
      <PreflightBanner />
      {/* DownloadStatusBanner — fixed bottom-right overlay, polls
          /api/v1/downloads/active every 2s. Renders nothing unless
          a model file is being downloaded. Highlights stalled
          downloads in amber so users know the system is recovering
          rather than frozen. */}
      <DownloadStatusBanner />
      {/* WelcomeModal — one-time first-run orientation (localStorage-gated). */}
      <WelcomeModal />
    </div>
  )
}

export default App
