import { BookMarked, Globe } from 'lucide-react'
import { useStore } from '../../stores/useStore'
import { AdvancedSettings } from './AdvancedSettings'
import { GenerateButton } from './GenerateButton'
import { ModelSelector } from './ModelSelector'
import { OutputFormatControls } from './OutputFormatControls'

export function StudioFooter({onCharacterSlot, onAnchor}: {
  onCharacterSlot: (node: HTMLDivElement | null) => void
  onAnchor: (node: HTMLDivElement | null) => void
}) {
  const modelType = useStore(s => s.params.model_type)
  const isOutpaint = useStore(s => s.generationMode === 'avatar' && s.editSubMode === 'outpaint')
  return <div ref={onAnchor} data-studio-footer className="studio-footer shrink-0 bg-bg-secondary px-3 pb-2 pt-2">
    <div data-testid="studio-settings-strip" className="flex min-w-0 items-center gap-1 pb-2">
      <div ref={onCharacterSlot} className="shrink-0 empty:hidden" />
      <button type="button" aria-label="Open recipes" aria-haspopup="dialog" title="Recipes"
        onClick={() => useStore.getState().setRecipesOpen(true)}
        className="studio-setting-chip border-border bg-bg-tertiary text-text-secondary hover:border-border-light">
        <BookMarked size={15}/>
      </button>
      <div data-testid="studio-output-settings" className="ml-auto flex min-w-0 flex-wrap items-center justify-end gap-1">
        <OutputFormatControls />
        <AdvancedSettings compact />
      </div>
    </div>
    <div data-testid="studio-generate-bar" className="flex min-w-0 items-center gap-1.5">
      {!isOutpaint && <button type="button" aria-label="Open model browser" aria-haspopup="dialog" title="Model Browser"
        onClick={() => useStore.getState().setLoraBrowserOpen(true, modelType)} className="flex min-h-11 w-8 shrink-0 items-center justify-center rounded-lg border border-border text-text-secondary hover:bg-bg-hover">
        <Globe size={17}/>
      </button>}
      <ModelSelector placement="footer" />
      <div className="min-w-[132px] flex-1"><GenerateButton stretch /></div>
    </div>
  </div>
}
