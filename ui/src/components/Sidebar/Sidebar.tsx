import { lazy, Suspense, useEffect, useLayoutEffect, useMemo, useState, type CSSProperties } from 'react'
import { useStore } from '../../stores/useStore'
import { AvatarControls } from './AvatarControls'
import { ViggleControls } from './ViggleControls'
import { useIsMobileSidecar } from '../../lib/useIsMobile'
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
import { ControlVideoSection } from './ControlVideoSection'
import { SidebarHeader } from './SidebarHeader'
import { CharacterToolbarContext, PromptDock, SidebarLayoutContext } from './SidebarPanels'
import { StartupStatus } from '../StartupStatus'

const DirectorChat = lazy(() => import('./DirectorChat').then(module => ({default: module.DirectorChat})))
const StudioTitlesPanel = lazy(() => import('../../editor/StudioTitlesPanel').then(module => ({
  default: module.StudioTitlesPanel,
})))

function isKeyboardInput(element: Element | null): boolean {
  if (element instanceof HTMLTextAreaElement) return !element.disabled && !element.readOnly
  if (element instanceof HTMLInputElement) {
    return !element.disabled && !element.readOnly
      && ['text', 'search', 'email', 'number', 'password', 'tel', 'url'].includes(element.type)
  }
  return element instanceof HTMLElement && element.isContentEditable
}

function inputRevealBounds(input: HTMLElement) {
  if (input instanceof HTMLTextAreaElement) {
    const mirror = input.parentElement?.querySelector('[data-prompt-mirror]')?.firstChild
    if (mirror instanceof Text) {
      // The prompt grows with its text. Reveal the actual caret line instead
      // of jumping to the top of a textarea that can span several screens.
      const offset = input.selectionDirection === 'backward' ? input.selectionStart : input.selectionEnd
      const range = document.createRange()
      range.setStart(mirror, Math.min(offset, mirror.length - 1))
      range.setEnd(mirror, Math.min(offset + 1, mirror.length))
      return range.getBoundingClientRect()
    }
  }
  return input.getBoundingClientRect()
}

export function Sidebar({ onOpenWorlds, embedded = false }: { onOpenWorlds?: () => void; embedded?: boolean } = {}) {
  const toggleSettings = useStore(s => s.toggleSettings)
  const generationMode = useStore(s => s.generationMode)
  const imageMode = useStore(s => s.params.image_mode)
  const modelOptions = useStore(s => s.modelOptions)
  const sidebarOpen = useStore(s => s.sidebarOpen)
  const setSidebarOpen = useStore(s => s.setSidebarOpen)
  const sidebarMode = useStore(s => s.sidebarMode)
  const editSubMode = useStore(s => s.editSubMode)
  const selectedModel = useStore(s => s.models.find(model => model.model_type === s.params.model_type))
  const isMobile = useIsMobileSidecar()
  const [characterSlot, setCharacterSlot] = useState<HTMLDivElement | null>(null)
  const [enhancementSlot, setEnhancementSlot] = useState<HTMLDivElement | null>(null)
  const [titlesFooterHost, setTitlesFooterHost] = useState<HTMLDivElement | null>(null)
  const [sidebarElement, setSidebarElement] = useState<HTMLElement | null>(null)
  const [settingsElement, setSettingsElement] = useState<HTMLDivElement | null>(null)
  const layout = useMemo(() => ({ sidebar: sidebarElement, settings: settingsElement }), [sidebarElement, settingsElement])
  const [visibleViewport, setVisibleViewport] = useState<{ height: number; top: number; keyboardOpen: boolean } | null>(null)
  useEffect(() => {
    if (!isMobile || !window.visualViewport) return
    const viewport = window.visualViewport
    let fullHeight = viewport.height
    let width = window.innerWidth
    let focusFrame = 0
    const resize = () => {
      const input = document.activeElement
      const editing = sidebarOpen && !!sidebarElement?.contains(input) && isKeyboardInput(input)
      // Browser chrome and restored tabs can leave innerHeight much taller
      // than the visible page without a keyboard. Measure the actual viewport
      // while idle, retaining that baseline only during text entry. This also
      // handles iOS shrinking innerHeight together with an actual keyboard.
      const unzoomed = Math.abs(viewport.scale - 1) < 0.01
      fullHeight = width !== window.innerWidth || !editing || !unzoomed
        ? viewport.height : Math.max(fullHeight, viewport.height)
      width = window.innerWidth
      const keyboardOpen = editing && unzoomed && viewport.height < fullHeight - 100
      setVisibleViewport(previous => (
        previous?.height === viewport.height && previous.top === viewport.offsetTop && previous.keyboardOpen === keyboardOpen
          ? previous : { height: viewport.height, top: viewport.offsetTop, keyboardOpen }
      ))
    }
    const focusChanged = () => {
      // Focusout precedes focusin when using the keyboard's Next button. Wait
      // until the new target is focused before resetting the height baseline.
      cancelAnimationFrame(focusFrame)
      focusFrame = requestAnimationFrame(resize)
    }
    const restore = () => {
      if (document.visibilityState !== 'visible') return
      fullHeight = viewport.height
      width = window.innerWidth
      resize()
    }
    resize()
    viewport.addEventListener('resize', resize)
    viewport.addEventListener('scroll', resize)
    window.addEventListener('resize', resize)
    window.addEventListener('pageshow', restore)
    document.addEventListener('visibilitychange', restore)
    document.addEventListener('focusin', focusChanged)
    document.addEventListener('focusout', focusChanged)
    return () => {
      cancelAnimationFrame(focusFrame)
      viewport.removeEventListener('resize', resize)
      viewport.removeEventListener('scroll', resize)
      window.removeEventListener('resize', resize)
      window.removeEventListener('pageshow', restore)
      document.removeEventListener('visibilitychange', restore)
      document.removeEventListener('focusin', focusChanged)
      document.removeEventListener('focusout', focusChanged)
    }
  }, [isMobile, sidebarOpen, sidebarElement])
  useEffect(() => {
    if (!isMobile || !sidebarOpen) return
    // iOS can scroll the document to an old caret position while opening the
    // keyboard. Lock the gallery underneath the drawer, retaining its position.
    const body = document.body
    const { position, top, left, width } = body.style
    const { scrollX, scrollY } = window
    Object.assign(body.style, { position: 'fixed', top: `${-scrollY}px`, left: `${-scrollX}px`, width: '100%' })
    return () => {
      Object.assign(body.style, { position, top, left, width })
      window.scrollTo(scrollX, scrollY)
    }
  }, [isMobile, sidebarOpen])
  useLayoutEffect(() => {
    if (!sidebarElement) return
    let frame = 0
    const revealInput = () => {
      const input = document.activeElement
      if (!(input instanceof HTMLElement) || !sidebarElement.contains(input)
        || !input.matches('textarea, input:not([type="range"]):not([type="checkbox"]):not([type="file"]), [contenteditable="true"]')) return
      // Scroll only inside this drawer, never the gallery/document underneath.
      // Both the main composer and Animate's embedded text fields need this.
      for (let parent = input.parentElement; parent && parent !== sidebarElement; parent = parent.parentElement) {
        if (parent.scrollHeight <= parent.clientHeight || !['auto', 'scroll'].includes(getComputedStyle(parent).overflowY)) continue
        const bounds = parent.getBoundingClientRect()
        const field = inputRevealBounds(input)
        const visibleHeight = Math.min(field.height, Math.max(0, parent.clientHeight - 16))
        const delta = field.top < bounds.top + 8 ? field.top - bounds.top - 8
          : field.top + visibleHeight > bounds.bottom - 8 ? field.top + visibleHeight - bounds.bottom + 8 : 0
        if (Math.abs(delta) > 1) parent.scrollTop += delta
      }
    }
    const scheduleReveal = () => {
      cancelAnimationFrame(frame)
      frame = requestAnimationFrame(revealInput)
    }
    // Re-evaluate after viewport changes (not on focus/typing). Pointer focus
    // already means the caret is visible; repeatedly revealing the whole field
    // moves long text away from the line the user clicked. Mobile keyboard
    // opening updates visibleViewport and schedules this layout pass.
    scheduleReveal()
    window.addEventListener('resize', scheduleReveal)
    return () => {
      cancelAnimationFrame(frame)
      window.removeEventListener('resize', scheduleReveal)
    }
  }, [isMobile, sidebarElement, visibleViewport])

  const isVideo = generationMode === 'video'
  const isImage = generationMode === 'image'
  const isAudio = generationMode === 'audio'
  const audioSubMode = useStore(s => s.audioSubMode)
  const isEdit = generationMode === 'avatar'
  const isTools = generationMode === 'tools'
  const toolsTool = useStore(s => s.toolsTool)
  const toolsUpscaleMedia = useStore(s => s.toolsUpscaleMedia)
  const videoWorkflow = useStore(s => s.studioVideoWorkflow)
  const isTitles = isVideo && videoWorkflow === 'titles'
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
  const isAvatarWorkflow = isVideo && videoWorkflow === 'avatar'
  const isReferencesWorkflow = isVideo && Number(imageMode) === 0 && videoWorkflow === 'references'
  const isMultiClip = isVideo && imageMode === 2
  const isContinue = isVideo && imageMode === 3
  const isBlend = isVideo && imageMode === 4
  const isDirector = sidebarMode === 'director'
  const isI2vOnly = modelOptions?.i2v_class && !modelOptions?.t2v_class
  const hasPrompt = !isStandaloneTool && !isAnimate && !isTitles && !(isAudio && ['sfx', 'mixer', 'music'].includes(audioSubMode))

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
      <>
      <div data-testid="studio-body-scroll" className="studio-scroll-body flex min-h-0 flex-1 flex-col">
      {/* Prompt Edit/Recast → Image Mode round-trip banner. Visible while
          a boundary anchor or Recast reference is being edited; null otherwise. */}
      {!isTitles && <AnchorReturnBanner />}

      {!embedded && <div data-testid="studio-workflow-header" className="studio-workflow-header flex shrink-0 flex-col gap-2 px-3 pb-2 pt-3">
        <GenerationModeSelector />

        {/* Studio's user-facing hierarchy is media first, workflow second.
            The workflow selectors route into the legacy video/avatar/tools
            engines so saved jobs and API behavior remain compatible. */}
        {isVideoWorkspace && <VideoWorkflowSelector onOpenWorlds={onOpenWorlds} />}
        {isImageWorkspace && <ImageWorkflowSelector />}
        {isAudioWorkspace && <AudioSubModeToggle />}
      </div>}
      {isTitles ? (
        <Suspense fallback={<p role="status" className="m-3 text-xs text-text-muted">Loading Titles…</p>}>
          <StudioTitlesPanel footerHost={titlesFooterHost} />
        </Suspense>
      ) : (
      <div className="studio-composition mx-3 flex grow shrink-0 basis-auto flex-col rounded-2xl border border-border bg-bg-primary/30" data-has-prompt={hasPrompt}>
      <div data-testid="studio-controls-scroll" className={`${hasPrompt ? 'studio-inputs shrink-0' : 'grow shrink-0'} flex min-w-0 flex-col gap-3 p-2.5 [&>*]:shrink-0`}>
        {isUpscale ? (
          <ToolsPanel forcedTool="upscale" mediaKind={toolsUpscaleMedia} embedded />
        ) : isFilmGrain ? (
          <ToolsPanel forcedTool="film_grain" mediaKind="video" embedded />
        ) : isRevoice ? (
          <ToolsPanel forcedTool="revoice" embedded />
        ) : (
        <>
        {(isImage || isVideo) && !isAnimate && !isAvatarWorkflow && !isOmniReference && !modelOptions?.minimax_h3_media_sources && (
          <ControlVideoSection galleryOnly />
        )}
        {/* Video Transform workflows use the established Edit engines. */}
        {isEdit && editControls}

        {/* Blend mode manages its own duration (overlap_sec) and its own
            start/end anchors — so the generic Duration slider and
            start/end ImageUpload don't apply there. */}
        {isAnimate && <ViggleControls />}
        {isAvatarWorkflow && <AvatarControls />}
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
        {isImage && (imageWorkflow === 'generate' || !!modelOptions?.image_ref_inpaint) && <ImageRefSection />}

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
        <PromptDock toolbarHost={enhancementSlot}>{isMultiClip ? <MultiClipEditor /> : <PromptInput />}</PromptDock>
      )}
      </div>
      )}
      </div>
      {/* Keep generation controls outside the scrolling prompt/input body. */}
      {!isStandaloneTool && !isTitles && <StudioFooter onCharacterSlot={setCharacterSlot} onEnhancementSlot={hasPrompt && !isMultiClip ? setEnhancementSlot : undefined} onAnchor={setSettingsElement} />}
      <div ref={setTitlesFooterHost} data-testid="titles-footer-host" className="studio-footer shrink-0" hidden={!isTitles} />
      </>
    </CharacterToolbarContext.Provider>
    </SidebarLayoutContext.Provider>
  )

  if (embedded) return (
    <div ref={setSidebarElement} className="maestro-sidebar flex h-full min-h-0 w-full min-w-0 flex-col bg-bg-secondary" data-testid="editor-ai-controls">
      <StartupStatus />
      {studioControls}
    </div>
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
        <aside ref={setSidebarElement} style={{ top: visibleViewport?.top, height: visibleViewport?.height, '--studio-visual-viewport-height': visibleViewport ? `${visibleViewport.height}px` : undefined, '--studio-visual-viewport-top': `${visibleViewport?.top || 0}px` } as CSSProperties} inert={!sidebarOpen} aria-hidden={!sidebarOpen}
          data-keyboard-open={visibleViewport?.keyboardOpen ?? false}
          className={`maestro-sidebar fixed top-0 h-dvh w-[380px] max-w-[94vw] bg-bg-secondary border-r border-border z-50 flex flex-col transition-[left] duration-300 ease-in-out ${sidebarOpen ? 'left-0' : '-left-full'}`}>
          {/* Header */}
          <SidebarHeader mobile onSettings={toggleSettings} onClose={() => setSidebarOpen(false)} />
          <StartupStatus />
          {isDirector ? <Suspense fallback={<p role="status" className="m-3 text-xs text-text-muted">Loading Director…</p>}><DirectorChat /></Suspense> : studioControls}
          <div className="studio-hardware shrink-0"><HardwareStatusBar /></div>
        </aside>
      </>
    )
  }

  // Desktop: static sidebar
  return (
    <aside ref={setSidebarElement} className="maestro-sidebar w-[420px] h-full bg-bg-secondary border-r border-border flex flex-col shrink-0">
      {/* Header */}
      <SidebarHeader onSettings={toggleSettings} onClose={() => setSidebarOpen(false)} />
      <StartupStatus />
      {isDirector ? <Suspense fallback={<p role="status" className="m-3 text-xs text-text-muted">Loading Director…</p>}><DirectorChat /></Suspense> : studioControls}
      <div className="studio-hardware shrink-0"><HardwareStatusBar /></div>
    </aside>
  )
}
