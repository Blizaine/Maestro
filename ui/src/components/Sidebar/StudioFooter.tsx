import { BookMarked, Globe } from 'lucide-react'
import { useStore } from '../../stores/useStore'
import { AdvancedSettings } from './AdvancedSettings'
import { GenerateButton } from './GenerateButton'
import { ModelSelector } from './ModelSelector'
import { OutputFormatControls } from './OutputFormatControls'

export function StudioFooter({onCharacterSlot, onEnhancementSlot, onAnchor}: {
  onCharacterSlot: (node: HTMLDivElement | null) => void
  onEnhancementSlot?: (node: HTMLDivElement | null) => void
  onAnchor: (node: HTMLDivElement | null) => void
}) {
  const modelType = useStore(s => s.params.model_type)
  const isOutpaint = useStore(s => s.generationMode === 'avatar' && s.editSubMode === 'outpaint')
  return <div ref={onAnchor} data-studio-footer className="studio-footer shrink-0 bg-bg-secondary px-3 pb-2 pt-2">
    <div data-testid="studio-settings-strip" className="studio-settings-row flex min-w-0 items-center gap-1 pb-2">
      <div ref={onCharacterSlot} className="shrink-0 empty:hidden" />
      <div data-testid="studio-output-settings" className="flex min-w-0 items-center gap-1">
        <OutputFormatControls />
        <AdvancedSettings compact />
      </div>
      {onEnhancementSlot && <div ref={onEnhancementSlot} role="group" aria-label="Prompt controls"
        className="studio-footer-enhancer ml-auto flex shrink-0 items-center rounded-lg border border-border bg-bg-tertiary" />}
    </div>
    <div data-testid="studio-generate-bar" className="flex min-w-0 items-center gap-1.5">
      <button type="button" aria-label="Open recipes" aria-haspopup="dialog" title="Recipes"
        onClick={() => useStore.getState().setRecipesOpen(true)}
        className="studio-footer-library flex min-h-11 w-8 shrink-0 items-center justify-center rounded-lg border border-border text-text-secondary hover:bg-bg-hover">
        <BookMarked size={16}/>
      </button>
      {!isOutpaint && <button type="button" aria-label="Open model browser" aria-haspopup="dialog" title="Model Browser"
        onClick={() => useStore.getState().setLoraBrowserOpen(true, modelType)} className="studio-footer-library flex min-h-11 w-8 shrink-0 items-center justify-center rounded-lg border border-border text-text-secondary hover:bg-bg-hover">
        <Globe size={17}/>
      </button>}
      <ModelSelector placement="footer" />
      <div className="min-w-[132px] flex-1"><GenerateButton stretch /></div>
    </div>
  </div>
}
