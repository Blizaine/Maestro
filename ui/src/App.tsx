import { lazy, Suspense, useEffect, useState } from 'react'
import { Menu, Settings } from 'lucide-react'
import { Sidebar } from './components/Sidebar/Sidebar'
import { MainContent } from './components/MainContent/MainContent'
import { OomRecoveryBanner } from './components/OomRecoveryBanner'
import { DownloadStatusBanner } from './components/DownloadStatusBanner'
import { PreflightBanner } from './components/PreflightBanner'
import { WelcomeModal } from './components/WelcomeModal'
import { GlobalQueuePopover } from './components/GlobalQueuePopover'
import { NotificationCoordinator } from './components/NotificationCoordinator'
import { NotificationToastHost } from './components/NotificationToastHost'
import { MaestroBrand } from './components/AppModeNavigation'
import { EditorRoundTripBanner } from './editor/EditorRoundTripBanner'
import { useStore } from './stores/useStore'
import { useIsMobileSidecar } from './lib/useIsMobile'
import { DeferredPanel } from './components/DeferredPanel'
import { createAppStartup, RETRY_STARTUP_EVENT } from './lib/appStartup'

const SettingsDrawer = lazy(() => import('./components/SettingsDrawer/SettingsDrawer').then(module => ({default: module.SettingsDrawer})))
const LoraBrowser = lazy(() => import('./components/LoraBrowser/LoraBrowser').then(module => ({default: module.LoraBrowser})))
const DirectorDashboard = lazy(() => import('./components/DirectorDashboard/DirectorDashboard').then(module => ({default: module.DirectorDashboard})))
const StorageDashboard = lazy(() => import('./components/StorageDashboard/StorageDashboard').then(module => ({default: module.StorageDashboard})))
const RetakeDialog = lazy(() => import('./components/RetakeDialog').then(module => ({default: module.RetakeDialog})))
const RecipesOverlay = lazy(() => import('./components/Recipes/RecipesOverlay').then(module => ({default: module.RecipesOverlay})))
const EditorWorkspace = lazy(() => import('./editor/EditorWorkspace').then(module => ({default: module.EditorWorkspace})))
const WorldExplorer = lazy(() => import('./components/Worlds/WorldExplorer').then(module => ({default: module.WorldExplorer})))

function App() {
  const [worldsOpen, setWorldsOpen] = useState(false)
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
  const settingsOpen = useStore(s => s.settingsOpen)
  const loraBrowserOpen = useStore(s => s.loraBrowserOpen)
  const dashboardOpen = useStore(s => s.dashboardOpen)
  const storageOpen = useStore(s => s.storageDashboardOpen)
  const recipesOpen = useStore(s => s.recipesOpen)
  const retakeOpen = useStore(s => s.retakeDialogOpen)
  const controlsReady = useStore(s => s.modelsLoaded && !s.modelsLoading && !s.modelsLoadError)
  const isMobile = useIsMobileSidecar()
  const isEditor = sidebarMode === 'editor'

  const openWorlds = () => {
    if (isMobile) setSidebarOpen(false)
    setWorldsOpen(true)
  }

  useEffect(() => {
    const startup = createAppStartup({
      loadControls: async cancelled => {
        useStore.setState({modelsLoading: true, modelsLoadError: null})
        await loadServicesConfig()
        if (cancelled()) return
        if (!useStore.getState().servicesConfig) {
          useStore.setState({modelsLoading: false})
          throw new Error('Maestro is not ready yet. Reconnecting…')
        }
        await loadModels({recoverOnly: true})
      },
      controlsReady: () => useStore.getState().modelsLoaded && !useStore.getState().modelsLoadError,
      onError: message => useStore.setState({modelsLoading: false, modelsLoadError: useStore.getState().modelsLoadError || message}),
      loadBackground: () => {
        // Restore the saved folder before listing media; optional scans and
        // remote writer discovery never gate the model controls.
        void (async () => {
          try {
            await loadWorkspaces()
            await loadOutputs()
          } finally {
            useStore.setState({galleryInitializing: false})
          }
        })()
        void loadSystemConfig()
        void loadLlmStatus()
        void loadLlmModels()
        void loadPipelineList()
      },
    })
    void startup.run()
    const recover = () => {
      if (!useStore.getState().modelsLoaded || useStore.getState().modelsLoadError) void startup.retry()
    }
    const visible = () => { if (document.visibilityState === 'visible') recover() }
    window.addEventListener('online', recover)
    window.addEventListener('pageshow', recover)
    window.addEventListener(RETRY_STARTUP_EVENT, recover)
    document.addEventListener('visibilitychange', visible)
    // Queue ownership/status remains immediate even while controls reconnect.
    reconnectJobs()
    return () => {
      startup.dispose()
      window.removeEventListener('online', recover)
      window.removeEventListener('pageshow', recover)
      window.removeEventListener(RETRY_STARTUP_EVENT, recover)
      document.removeEventListener('visibilitychange', visible)
    }
  }, [loadModels, loadWorkspaces, loadOutputs, loadSystemConfig, loadServicesConfig, loadLlmStatus, loadLlmModels, loadPipelineList, reconnectJobs])

  // Poll LLM status to stay in sync with backend auto-load/unload
  useEffect(() => {
    const interval = setInterval(loadLlmStatus, 15000)
    return () => clearInterval(interval)
  }, [loadLlmStatus])

  return (
    <div className={`flex ${isMobile && !isEditor ? 'flex-col' : 'flex-col md:flex-row'} h-full w-full bg-bg-primary`}>
      {/* Mobile header */}
      {isMobile && !isEditor && (
        <header className="h-12 shrink-0 grid grid-cols-[1fr_auto_1fr] items-center gap-1 px-2 border-b border-border bg-bg-secondary">
          <button
            onClick={toggleSidebar}
            aria-label="Open sidecar"
            className="justify-self-start p-2 rounded-lg hover:bg-bg-hover text-text-secondary hover:text-text-primary transition-colors"
          >
            <Menu size={20} />
          </button>
          <MaestroBrand />
          <div className="justify-self-end flex items-center gap-0.5">
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
        <Suspense fallback={<p role="status" className="m-auto text-sm text-text-muted">Loading editor…</p>}><EditorWorkspace /></Suspense>
      ) : (
        <>
          <Sidebar onOpenWorlds={openWorlds} worldsOpen={worldsOpen} />
          <MainContent />
        </>
      )}
      <DeferredPanel active={settingsOpen}><SettingsDrawer /></DeferredPanel>
      <DeferredPanel active={loraBrowserOpen}><LoraBrowser /></DeferredPanel>
      <DeferredPanel active={dashboardOpen}><DirectorDashboard /></DeferredPanel>
      <DeferredPanel active={storageOpen}><StorageDashboard /></DeferredPanel>
      <DeferredPanel active={recipesOpen}><RecipesOverlay /></DeferredPanel>
      <DeferredPanel active={retakeOpen}><RetakeDialog /></DeferredPanel>
      {worldsOpen && (
        <button
          type="button"
          aria-label="Close World Explorer"
          onClick={() => setWorldsOpen(false)}
          className="fixed inset-0 z-[84] cursor-default bg-black/40"
        />
      )}
      <DeferredPanel active={worldsOpen}>
        <WorldExplorer open={worldsOpen} onClose={() => setWorldsOpen(false)} />
      </DeferredPanel>
      {/* OomRecoveryBanner is a fixed-position overlay — renders nothing
          unless the latest job/pipeline failure has oom_info attached.
          Lives at the App root so it floats above whichever screen the
          user is looking at when their generation OOMs. */}
      <OomRecoveryBanner />
      {/* PreflightBanner — fixed top overlay shown once on startup if the
          environment is missing ffmpeg / CUDA or low on disk. Renders
          nothing when everything checks out. */}
      <DeferredPanel active={controlsReady}><PreflightBanner /></DeferredPanel>
      {/* DownloadStatusBanner — fixed bottom-right overlay, long-polls
          /api/v1/downloads/active. Renders nothing unless
          a model file is being downloaded. Highlights stalled
          downloads in amber so users know the system is recovering
          rather than frozen. */}
      <DeferredPanel active={controlsReady}><DownloadStatusBanner /></DeferredPanel>
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
