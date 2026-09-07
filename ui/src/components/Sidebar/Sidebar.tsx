import { Settings, X } from 'lucide-react'
import { useEffect, useMemo, useState, type CSSProperties } from 'react'
import { useStore } from '../../stores/useStore'
import { ViggleControls } from './ViggleControls'
import { useIsMobile } from '../../lib/useIsMobile'
import { GenerationModeSelector } from './GenerationModeSelector'
import { InputsPanel } from './InputsPanel'
import { OmniReferenceSection } from './OmniReferenceSection'
import { PromptInput } from './PromptInput'
import { ImageRefSection } from './ImageRefSection'
import { AudioModeSection } from './AudioModeSection'
import { MusicControls } from './MusicControls'
import { AudioSubModeToggle } from './AudioSubModeToggle'
import { SfxControls } from './SfxControls'
import { MixerControls } from './MixerControls'
import { StudioFooter } from './StudioFooter'
import { MultiClipEditor } from './MultiClipEditor'
import { DirectorChat } from './DirectorChat'
import { RestyleControls } from './RestyleControls'
import { InpaintControls } from './InpaintControls'
import { OutpaintControls } from './OutpaintControls'
import { RetakeControls } from './RetakeControls'
import { EditAnythingControls } from './EditAnythingControls'
import { RecastControls } from './RecastControls'
import { BlendControls } from './BlendControls'
import { AnchorReturnBanner } from './AnchorReturnBanner'
import { VoiceRefSection } from './VoiceRefSection'
import { ToolsPanel } from './ToolsPanel'
import { HardwareStatusBar } from './HardwareStatusBar'
import { VideoWorkflowSelector } from './VideoWorkflowSelector'
import { ImageWorkflowSelector } from './ImageWorkflowSelector'
import { ImageWorkflowControls } from './ImageWorkflowControls'
import { AppModeToggle, MaestroBrand } from '../AppModeNavigation'
import { CharacterToolbarContext, PromptDock, SidebarLayoutContext } from './SidebarPanels'

export function Sidebar() {
  const toggleSettings = useStore(s => s.toggleSettings)
  const generationMode = useStore(s => s.generationMode)
  const imageMode = useStore(s => s.params.image_mode)
  const modelOptions = useStore(s => s.modelOptions)
  const sidebarOpen = useStore(s => s.sidebarOpen)
  const setSidebarOpen = useStore(s => s.setSidebarOpen)
  const sidebarMode = useStore(s => s.sidebarMode)
  const editSubMode = useStore(s => s.editSubMode)
  const selectedModel = useStore(s => s.models.find(model => model.model_type === s.params.model_type))
  const isMobile = useIsMobile()
  const [characterSlot, setCharacterSlot] = useState<HTMLDivElement | null>(null)
  const [sidebarElement, setSidebarElement] = useState<HTMLElement | null>(null)
  const [settingsElement, setSettingsElement] = useState<HTMLDivElement | null>(null)
  const layout = useMemo(() => ({ sidebar: sidebarElement, settings: settingsElement }), [sidebarElement, settingsElement])
  const [viewportHeight, setViewportHeight] = useState<number | undefined>(undefined)
  useEffect(() => {
    if (!isMobile || !window.visualViewport) return
    const viewport = window.visualViewport
    const resize = () => setViewportHeight(viewport.height)
    resize()
    viewport.addEventListener('resize', resize)
    return () => viewport.removeEventListener('resize', resize)
  }, [isMobile])

  const isVideo = generationMode === 'video'
  const isImage = generationMode === 'image'
  const isAudio = generationMode === 'audio'
  const audioSubMode = useStore(s => s.audioSubMode)
  const isEdit = generationMode === 'avatar'
  const isTools = generationMode === 'tools'
  const toolsTool = useStore(s => s.toolsTool)
  const toolsUpscaleMedia = useStore(s => s.toolsUpscaleMedia)
  const videoWorkflow = useStore(s => s.studioVideoWorkflow)
  const imageWorkflow = useStore(s => s.studioImageWorkflow)
  const isUpscale = isTools && toolsTool === 'upscale'
  const isImageUpscale = isUpscale && toolsUpscaleMedia === 'image'
  const isVideoUpscale = isUpscale && toolsUpscaleMedia === 'video'
  const isFilmGrain = isTools && toolsTool === 'film_grain'
  const isRevoice = (isTools && toolsTool === 'revoice') || (isAudio && audioSubMode === 'revoice')
  const isVideoWorkspace = isVideo || isEdit || isVideoUpscale || isFilmGrain
  const isImageWorkspace = isImage || isImageUpscale
  const isAudioWorkspace = isAudio || isRevoice
  const isStandaloneTool = isUpscale || isFilmGrain || isRevoice
  const isRetake = isEdit && editSubMode === 'retake'
  const isRestyle = isEdit && editSubMode === 'restyle'
  const isInpaint = isEdit && editSubMode === 'inpaint'
  const isOutpaint = isEdit && editSubMode === 'outpaint'
  const isEditAnything = isEdit && editSubMode === 'edit_anything'
  const isRecast = isEdit && editSubMode === 'recast'
  const isOmniReference = isVideo && Boolean(
    selectedModel?.omni_reference
    || selectedModel?.director?.video_strategy === 'omni_reference'
    || selectedModel?.model_type.toLowerCase().startsWith('minimax_h3_ref2va'),
  )
  const isFramesWorkflow = isVideo && Number(imageMode) === 0 && videoWorkflow === 'frames'
  const isAnimate = isVideo && videoWorkflow === 'animate'
  const isReferencesWorkflow = isVideo && Number(imageMode) === 0 && videoWorkflow === 'references'
  const isMultiClip = isVideo && imageMode === 2
  const isContinue = isVideo && imageMode === 3
  const isBlend = isVideo && imageMode === 4
  const isDirector = sidebarMode === 'director'
  const isI2vOnly = modelOptions?.i2v_class && !modelOptions?.t2v_class
  const hasPrompt = !isStandaloneTool && !isAnimate && !(isAudio && ['sfx', 'mixer', 'music'].includes(audioSubMode))

  // Video Transform controls backed by the legacy edit-mode engines.
  const editControls = (
    <>
      {isRetake && (
        <>
          <RetakeControls />
        </>
      )}
      {isInpaint && (
        <>
          <InpaintControls />
        </>
      )}
      {isOutpaint && (
        <>
          <OutpaintControls />
        </>
      )}
      {isRestyle && (
        <>
          <RestyleControls />
        </>
      )}
      {isEditAnything && (
        <>
          <EditAnythingControls />
        </>
      )}
      {isRecast && (
        <>
          <RecastControls />
        </>
      )}
    </>
  )

  const studioControls = (
    <SidebarLayoutContext.Provider value={layout}>
    <CharacterToolbarContext.Provider value={characterSlot}>
      {/* Prompt Edit/Recast → Image Mode round-trip banner. Visible while
          a boundary anchor or Recast reference is being edited; null otherwise. */}
      <AnchorReturnBanner />

      <div data-testid="studio-workflow-header" className="studio-workflow-header flex shrink-0 flex-col gap-2 px-3 pb-2 pt-3">
        <GenerationModeSelector />

        {/* Studio's user-facing hierarchy is media first, workflow second.
            The workflow selectors route into the legacy video/avatar/tools
            engines so saved jobs and API behavior remain compatible. */}
        {isVideoWorkspace && <VideoWorkflowSelector />}
        {isImageWorkspace && <ImageWorkflowSelector />}
        {isAudioWorkspace && <AudioSubModeToggle />}
      </div>
      <div className="studio-composition mx-3 flex min-h-0 flex-1 flex-col rounded-2xl border border-border bg-bg-primary/30" data-has-prompt={hasPrompt}>
      <div data-testid="studio-controls-scroll" className={`${hasPrompt ? 'studio-inputs shrink-0' : 'flex-1'} flex min-h-0 flex-col gap-3 overflow-y-auto overflow-x-hidden overscroll-contain p-2.5 [&>*]:shrink-0`}>
        {isUpscale ? (
          <ToolsPanel forcedTool="upscale" mediaKind={toolsUpscaleMedia} embedded />
        ) : isFilmGrain ? (
          <ToolsPanel forcedTool="film_grain" mediaKind="video" embedded />
        ) : isRevoice ? (
          <ToolsPanel forcedTool="revoice" embedded />
        ) : (
        <>
        {/* Video Transform workflows use the established Edit engines. */}
        {isEdit && editControls}

        {/* Blend mode manages its own duration (overlap_sec) and its own
            start/end anchors — so the generic Duration slider and
            start/end ImageUpload don't apply there. */}
        {isAnimate && <ViggleControls />}
        {/* Frames (image_mode 0) AND Extend (image_mode 3) both use the unified
            InputsPanel. In Extend mode its first tile is the source video to
            continue from; otherwise it's the start frame. */}
        {isVideo && !isMultiClip && !isBlend && (isFramesWorkflow || isContinue) && (
          <div>
            {isI2vOnly && !isContinue && (
              <div className="text-[10px] text-indicator-warning bg-amber-500/10 border border-amber-500/20 rounded-lg px-3 py-1.5 mb-2">
                This model requires a start image to generate video.
              </div>
            )}
            <InputsPanel />
          </div>
        )}
        {isReferencesWorkflow && <OmniReferenceSection />}
        {isBlend && <BlendControls />}

        {/* Image workflows expose only the inputs their native pipeline uses. */}
        {isImage && <ImageWorkflowControls />}
        {isImage && (imageWorkflow === 'generate' || !!modelOptions?.image_ref_choices) && <ImageRefSection />}

        {/* Video/Image mode: audio controls (soundtrack, control video, etc.).
            In Frames mode (video, image_mode 0) the unified InputsPanel routes
            audio/control-video via tiles instead, so the dropdown is hidden
            there. Other video sub-modes + image mode keep AudioModeSection. */}
        {!isEdit && !isAudio && !(isVideo && (imageMode === 0 || imageMode === 3)) && modelOptions?.audio_prompt_type_sources && <AudioModeSection />}

        {/* Audio mode: workflow-specific controls */}
        {isAudio && audioSubMode === 'speech' && modelOptions?.audio_only && <AudioModeSection />}
        {isAudio && audioSubMode === 'sfx' && <SfxControls />}
        {isAudio && audioSubMode === 'mixer' && <MixerControls />}
        {isAudio && audioSubMode === 'music' && <MusicControls />}

        {/* Video: reference images below prompt. In Frames mode the InputsPanel
            renders them as ordered tiles instead. */}
        {isVideo && !isOmniReference && imageMode !== 0 && imageMode !== 3 && modelOptions?.image_ref_choices && <ImageRefSection />}

        {/* LTX Voice Reference (ID-LoRA) — gated by Video Frames →
            Advanced. VoiceRefSection also verifies the active LTX model. */}
        {isVideo && !isDirector && !isOmniReference && imageMode !== 0 && imageMode !== 3 && <VoiceRefSection />}
        </>
        )}
      </div>

      {hasPrompt && (
        <PromptDock>{isMultiClip ? <MultiClipEditor /> : <PromptInput />}</PromptDock>
      )}
      </div>
      {!isStandaloneTool && <StudioFooter onCharacterSlot={setCharacterSlot} onAnchor={setSettingsElement} />}
    </CharacterToolbarContext.Provider>
    </SidebarLayoutContext.Provider>
  )

  // Mobile: overlay drawer
  if (isMobile) {
    return (
      <>
        {sidebarOpen && (
          <div
            className="fixed inset-0 bg-black/40 z-40"
            onClick={() => setSidebarOpen(false)}
          />
        )}
        <aside ref={setSidebarElement} style={{ height: viewportHeight, '--studio-visual-viewport-height': viewportHeight ? `${viewportHeight}px` : undefined } as CSSProperties} inert={!sidebarOpen} aria-hidden={!sidebarOpen}
          data-keyboard-open={viewportHeight != null && viewportHeight < window.innerHeight - 100}
          className={`maestro-sidebar fixed top-0 h-dvh w-[380px] max-w-[94vw] bg-bg-secondary border-r border-border z-50 flex flex-col transition-[left] duration-300 ease-in-out ${sidebarOpen ? 'left-0' : '-left-full'}`}>
          {/* Header */}
          <div className="shrink-0 px-4 py-3 border-b border-border flex items-center justify-between">
            <MaestroBrand compact />
            <div className="flex items-center gap-1.5">
              <AppModeToggle size="sm" />
              <button
                onClick={() => setSidebarOpen(false)}
                className="p-1.5 rounded-lg hover:bg-bg-hover text-text-secondary hover:text-text-primary transition-colors"
              >
                <X size={16} />
              </button>
            </div>
          </div>
          {isDirector ? <DirectorChat /> : studioControls}
          <div className="studio-hardware shrink-0"><HardwareStatusBar /></div>
        </aside>
      </>
    )
  }

  // Desktop: static sidebar
  return (
    <aside ref={setSidebarElement} className="maestro-sidebar w-[420px] h-full bg-bg-secondary border-r border-border flex flex-col shrink-0">
      {/* Header */}
      <div className="flex h-14 items-center justify-between border-b border-border px-4">
        <MaestroBrand />
        <div className="flex items-center gap-2">
          <AppModeToggle />
          <button
            onClick={toggleSettings}
            className="p-1.5 rounded-lg hover:bg-bg-hover text-text-secondary hover:text-text-primary transition-colors"
            title="Settings"
          >
            <Settings size={16} />
          </button>
        </div>
      </div>
      {isDirector ? <DirectorChat /> : studioControls}
      <div className="studio-hardware shrink-0"><HardwareStatusBar /></div>
    </aside>
  )
}
