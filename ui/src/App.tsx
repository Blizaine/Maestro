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
import { GlobalQueuePopover } from './components/GlobalQueuePopover'
import { NotificationCoordinator } from './components/NotificationCoordinator'
import { NotificationToastHost } from './components/NotificationToastHost'
import { EditorWorkspace } from './editor/EditorWorkspace'
import { AppModeToggle, MaestroBrand } from './components/AppModeNavigation'
import { EditorRoundTripBanner } from './editor/EditorRoundTripBanner'
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
  const loadPipelineList = useStore(s => s.loadPipelineList)
  const toggleSidebar = useStore(s => s.toggleSidebar)
  const setSidebarOpen = useStore(s => s.setSidebarOpen)
  const toggleSettings = useStore(s => s.toggleSettings)
  const sidebarMode = useStore(s => s.sidebarMode)
  const isMobile = useIsMobile()
  const isEditor = sidebarMode === 'editor'

  useEffect(() => {
    loadModels()
    loadWorkspaces()
    loadOutputs()
    loadSystemConfig()
    loadServicesConfig()
    loadLlmStatus()
    loadLlmModels()
    loadPipelineList()
    reconnectJobs()
  }, [loadModels, loadWorkspaces, loadOutputs, loadSystemConfig, loadServicesConfig, loadLlmStatus, loadLlmModels, loadPipelineList, reconnectJobs])

  // Poll LLM status to stay in sync with backend auto-load/unload
  useEffect(() => {
    const interval = setInterval(loadLlmStatus, 15000)
    return () => clearInterval(interval)
  }, [loadLlmStatus])

  return (
    <div className="flex flex-col md:flex-row h-full w-full bg-bg-primary">
      {/* Mobile header */}
      {isMobile && !isEditor && (
        <header className="h-12 shrink-0 gap-1 px-2 border-b border-border flex items-center bg-bg-secondary">
          <button
            onClick={toggleSidebar}
            className="p-2 rounded-lg hover:bg-bg-hover text-text-secondary hover:text-text-primary transition-colors"
          >
            <Menu size={20} />
          </button>
          <MaestroBrand compact />
          <AppModeToggle size="sm" />
          <div className="ml-auto flex items-center gap-0.5">
            <GlobalQueuePopover iconSize={20} panelAlign="header-edge" />
            <button
              onClick={() => { setSidebarOpen(false); toggleSettings() }}
              className="p-2 rounded-lg hover:bg-bg-hover text-text-secondary hover:text-text-primary transition-colors"
              title="Settings"
            >
              <Settings size={20} />
            </button>
          </div>
        </header>
      )}

      {isEditor ? (
        <EditorWorkspace />
      ) : (
        <>
          <Sidebar />
          <MainContent />
        </>
      )}
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
      {/* One observer covers Studio, Director, and the universal queue.
          Toasts remain useful even when browser notifications are disabled. */}
      <NotificationCoordinator />
      <NotificationToastHost />
      <EditorRoundTripBanner />
    </div>
  )
}

export default App
