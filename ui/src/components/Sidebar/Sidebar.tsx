import { Settings, X, Globe, BookMarked } from 'lucide-react'
import { useStore } from '../../stores/useStore'
import { useIsMobile } from '../../lib/useIsMobile'
import { GenerationModeSelector } from './GenerationModeSelector'
import { ImageUpload } from './ImageUpload'
import { InputsPanel } from './InputsPanel'
import { PromptInput } from './PromptInput'
import { ImageRefSection } from './ImageRefSection'
import { AudioModeSection } from './AudioModeSection'
import { MusicControls } from './MusicControls'
import { AudioSubModeToggle } from './AudioSubModeToggle'
import { SfxControls } from './SfxControls'
import { MixerControls } from './MixerControls'
import { ModeToggle } from './ModeToggle'
import { DurationSlider } from './DurationSlider'
import { AdvancedSettings } from './AdvancedSettings'
import { GenerateButton } from './GenerateButton'
import { ModelSelector } from './ModelSelector'
import { MultiClipEditor } from './MultiClipEditor'
import { DirectorChat } from './DirectorChat'
import { EditSubModeToggle } from './EditSubModeToggle'
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

export function Sidebar() {
  const toggleSettings = useStore(s => s.toggleSettings)
  const generationMode = useStore(s => s.generationMode)
  const imageMode = useStore(s => s.params.image_mode)
  const modelOptions = useStore(s => s.modelOptions)
  const sidebarOpen = useStore(s => s.sidebarOpen)
  const appVersion = useStore(s => s.systemConfig?.app_version)
  const setSidebarOpen = useStore(s => s.setSidebarOpen)
  const sidebarMode = useStore(s => s.sidebarMode)
  const setSidebarMode = useStore(s => s.setSidebarMode)
  const editSubMode = useStore(s => s.editSubMode)
  const modelType = useStore(s => s.params.model_type)
  const openLoraBrowser = useStore(s => s.setLoraBrowserOpen)
  const isMobile = useIsMobile()

  const isVideo = generationMode === 'video'
  const isImage = generationMode === 'image'
  const isAudio = generationMode === 'audio'
  const audioSubMode = useStore(s => s.audioSubMode)
  const isEdit = generationMode === 'avatar'
  const isTools = generationMode === 'tools'
  const isRetake = isEdit && editSubMode === 'retake'
  const isRestyle = isEdit && editSubMode === 'restyle'
  const isInpaint = isEdit && editSubMode === 'inpaint'
  const isOutpaint = isEdit && editSubMode === 'outpaint'
  const isEditAnything = isEdit && editSubMode === 'edit_anything'
  const isRecast = isEdit && editSubMode === 'recast'
  const isMultiClip = isVideo && imageMode === 2
  const isContinue = isVideo && imageMode === 3
  const isBlend = isVideo && imageMode === 4
  const isDirector = sidebarMode === 'director'
  const isI2vOnly = modelOptions?.i2v_class && !modelOptions?.t2v_class

  const modeToggle = (size: 'sm' | 'md') => (
    <div className="flex bg-bg-tertiary rounded-lg p-0.5 border border-border">
      <button
        onClick={() => setSidebarMode('director')}
        className={`${size === 'sm' ? 'px-2 py-1 text-[11px]' : 'px-3 py-1 text-xs'} rounded-md transition-all ${
          // bg-toggle-active is flat accent-blue in the default theme
          // (preserves the original blue pill) and a red→orange sunset
          // gradient in Golden Hour. shadow-accent-glow is empty in
          // default and a warm bloom in Golden Hour.
          isDirector ? 'bg-toggle-active shadow-accent-glow text-white' : 'text-text-secondary hover:text-text-primary'
        }`}
      >
        Director
      </button>
      <button
        onClick={() => setSidebarMode('studio')}
        className={`${size === 'sm' ? 'px-2 py-1 text-[11px]' : 'px-3 py-1 text-xs'} rounded-md transition-all ${
          // Studio active intentionally uses bg-toggle-active too so the
          // currently-active mode reads with the same prominence in
          // Golden Hour as the reference render. Default theme: flat
          // accent-blue (was bg-bg-active dark elevation — small change
          // that brings the two buttons into visual parity).
          !isDirector ? 'bg-toggle-active shadow-accent-glow text-white' : 'text-text-secondary hover:text-text-primary'
        }`}
      >
        Studio
      </button>
    </div>
  )

  // Edit mode sub-controls based on sub-mode
  const editControls = (
    <>
      {isRetake && (
        <>
          <RetakeControls />
          <PromptInput />
        </>
      )}
      {isInpaint && (
        <>
          <InpaintControls />
          <PromptInput />
        </>
      )}
      {isOutpaint && (
        <>
          <OutpaintControls />
          <PromptInput />
        </>
      )}
      {isRestyle && (
        <>
          <RestyleControls />
          <DurationSlider />
          <ImageUpload />
          <PromptInput />
        </>
      )}
      {isEditAnything && (
        <>
          <EditAnythingControls />
          <PromptInput />
        </>
      )}
      {isRecast && (
        <>
          <RecastControls />
          <PromptInput />
        </>
      )}
    </>
  )

  const studioControls = (
    <>
      {/* Edit Anything → Image Mode round-trip banner. Visible whenever
          the user is in the middle of editing boundary anchors via the
          Image Mode workflow; null otherwise. */}
      <AnchorReturnBanner />

      {/* [&>*]:shrink-0 — keep every section at its natural height and let
          the column SCROLL when space is tight (e.g. ID-LoRA voice section
          added + hardware bar expanded), instead of letting flex-shrink
          crush sections into each other. */}
      <div className="flex-1 overflow-y-auto px-4 py-4 flex flex-col gap-4 min-h-0 [&>*]:shrink-0">
        <GenerationModeSelector />

        {/* Tools mode: standalone post-processing (upscale / revoice) on any
            existing clip. Renders in place of the generation controls. */}
        {isTools ? <ToolsPanel /> : (
        <>
        {/* Edit mode: sub-mode toggle + sub-controls */}
        {isEdit && <EditSubModeToggle />}
        {isEdit && editControls}

        {/* Video mode */}
        {isVideo && <ModeToggle />}
        {/* Blend mode manages its own duration (overlap_sec) and its own
            start/end anchors — so the generic Duration slider and
            start/end ImageUpload don't apply there. */}
        {isVideo && !isBlend && <DurationSlider />}
        {/* Frames (image_mode 0) AND Extend (image_mode 3) both use the unified
            InputsPanel. In Extend mode its first tile is the source video to
            continue from; otherwise it's the start frame. */}
        {isVideo && !isMultiClip && !isBlend && (
          <div>
            {isI2vOnly && !isContinue && (
              <div className="text-[10px] text-indicator-warning bg-amber-500/10 border border-amber-500/20 rounded-lg px-3 py-1.5 mb-2">
                This model requires a start image to generate video.
              </div>
            )}
            <InputsPanel />
          </div>
        )}
        {isBlend && <BlendControls />}

        {/* Image mode: reference images */}
        {isImage && modelOptions?.image_ref_choices && <ImageRefSection />}

        {/* Video/Image mode: audio controls (soundtrack, control video, etc.).
            In Frames mode (video, image_mode 0) the unified InputsPanel routes
            audio/control-video via tiles instead, so the dropdown is hidden
            there. Other video sub-modes + image mode keep AudioModeSection. */}
        {!isEdit && !isAudio && !(isVideo && (imageMode === 0 || imageMode === 3)) && modelOptions?.audio_prompt_type_sources && <AudioModeSection />}

        {/* Audio mode: sub-mode toggle + mode-specific controls */}
        {isAudio && <AudioSubModeToggle />}
        {isAudio && audioSubMode === 'speech' && modelOptions?.audio_prompt_type_sources && <AudioModeSection />}
        {isAudio && audioSubMode === 'sfx' && <SfxControls />}
        {isAudio && audioSubMode === 'mixer' && <MixerControls />}
        {isAudio && audioSubMode === 'music' && <MusicControls />}

        {/* Prompt area (non-edit modes, skip for SFX/Mixer/Music which have their own UI) */}
        {!isEdit && !(isAudio && (audioSubMode === 'sfx' || audioSubMode === 'mixer' || audioSubMode === 'music')) && (isMultiClip ? <MultiClipEditor /> : <PromptInput />)}

        {/* Video: reference images below prompt. In Frames mode the InputsPanel
            renders them as ordered tiles instead. */}
        {isVideo && imageMode !== 0 && imageMode !== 3 && modelOptions?.image_ref_choices && <ImageRefSection />}

        {/* Voice Reference (ID-LoRA) — gated by Settings → Services
            toggle (`voice_reference_enabled`). VoiceRefSection internally
            no-ops when the toggle is off. We render it for Studio Video
            mode (basic, multi-clip, continue, blend) — it's the same
            generation path that consumes `directorVoiceRef` server-side.
            Director mode renders its own copy in DirectorChat. */}
        {isVideo && !isDirector && imageMode !== 0 && imageMode !== 3 && <VoiceRefSection />}
        </>
        )}
      </div>

      {/* Bottom Bar: Advanced + LoRA Browser + Model + Generate.
          Hidden in Tools mode — ToolsPanel has its own Run button and
          owns no model. */}
      {!isTools && (
      <div className="px-3 py-2.5 border-t border-border">
        <div className="flex items-center gap-2">
          <AdvancedSettings />
          <button
            onClick={() => useStore.getState().setRecipesOpen(true)}
            className="p-2 rounded-lg bg-bg-tertiary border border-border hover:border-border-light text-text-secondary hover:text-accent-blue transition-colors shrink-0"
            title="Recipes — one-click presets"
          >
            <BookMarked size={14} />
          </button>
          <button
            onClick={() => openLoraBrowser(true, modelType)}
            className="p-2 rounded-lg bg-bg-tertiary border border-border hover:border-border-light text-text-secondary hover:text-accent-blue transition-colors shrink-0"
            title="Browse LoRAs on CivitAI"
          >
            <Globe size={14} />
          </button>
          <div className="flex-1 min-w-0">
            <ModelSelector />
          </div>
          <div className="shrink-0">
            <GenerateButton />
          </div>
        </div>
      </div>
      )}
    </>
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
        <aside className={`fixed top-0 left-0 h-full w-[380px] max-w-[85vw] bg-bg-secondary border-r border-border z-50 flex flex-col transform transition-transform duration-300 ease-in-out ${
          sidebarOpen ? 'translate-x-0' : '-translate-x-full'
        }`}>
          {/* Header */}
          <div className="px-4 py-3 border-b border-border flex items-center justify-between">
            <div className="flex items-center gap-2">
              <div className="w-7 h-7 rounded-lg bg-accent-blue flex items-center justify-center text-white font-bold text-sm">
                M
              </div>
              <span className="font-semibold text-sm">Maestro</span>
              {appVersion && <span className="text-[10px] text-text-muted font-normal mt-0.5">v{appVersion}</span>}
            </div>
            <div className="flex items-center gap-1.5">
              {modeToggle('sm')}
              <button
                onClick={() => setSidebarOpen(false)}
                className="p-1.5 rounded-lg hover:bg-bg-hover text-text-secondary hover:text-text-primary transition-colors"
              >
                <X size={16} />
              </button>
            </div>
          </div>
          {isDirector ? <DirectorChat /> : studioControls}
          <HardwareStatusBar />
        </aside>
      </>
    )
  }

  // Desktop: static sidebar
  return (
    <aside className="w-[420px] h-full bg-bg-secondary border-r border-border flex flex-col shrink-0">
      {/* Header */}
      <div className="px-4 py-3 border-b border-border flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div className="w-7 h-7 rounded-lg bg-accent-blue flex items-center justify-center text-white font-bold text-sm">
            M
          </div>
          <span className="font-semibold text-sm">Maestro</span>
              {appVersion && <span className="text-[10px] text-text-muted font-normal mt-0.5">v{appVersion}</span>}
        </div>
        <div className="flex items-center gap-2">
          {modeToggle('md')}
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
      <HardwareStatusBar />
    </aside>
  )
}
