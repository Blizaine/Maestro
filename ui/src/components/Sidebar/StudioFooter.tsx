import { useState } from 'react'
import { BookMarked, Globe, MoreHorizontal } from 'lucide-react'
import { useStore } from '../../stores/useStore'
import { AdvancedSettings } from './AdvancedSettings'
import { GenerateButton } from './GenerateButton'
import { ModelSelector } from './ModelSelector'
import { OutputFormatControls } from './OutputFormatControls'
import { SidebarDialog } from './SidebarPanels'

export function StudioFooter({onCharacterSlot, onAnchor}: {
  onCharacterSlot: (node: HTMLDivElement | null) => void
  onAnchor: (node: HTMLDivElement | null) => void
}) {
  const [librariesOpen, setLibrariesOpen] = useState(false)
  const modelType = useStore(s => s.params.model_type)
  const isOutpaint = useStore(s => s.generationMode === 'avatar' && s.editSubMode === 'outpaint')
  return <div ref={onAnchor} data-studio-footer className="studio-footer shrink-0 bg-bg-secondary px-3 pb-2 pt-2">
    <div data-testid="studio-settings-strip" className="flex min-w-0 items-center gap-1 pb-2">
      <div ref={onCharacterSlot} className="shrink-0 empty:hidden" />
      <div data-testid="studio-output-settings" className="ml-auto flex min-w-0 flex-wrap items-center justify-end gap-1">
        <OutputFormatControls />
        <AdvancedSettings compact />
      </div>
    </div>
    <div data-testid="studio-generate-bar" className="flex min-w-0 items-center gap-1.5">
      <button type="button" aria-label="Recipes and model browser" aria-expanded={librariesOpen} title="Recipes and Model Browser"
        onClick={() => setLibrariesOpen(value => !value)} className="flex min-h-11 w-8 shrink-0 items-center justify-center rounded-lg border border-border text-text-secondary hover:bg-bg-hover">
        <MoreHorizontal size={17}/>
      </button>
      <ModelSelector placement="footer" />
      <div className="min-w-[132px] flex-1"><GenerateButton stretch /></div>
    </div>
    <SidebarDialog open={librariesOpen} title="Studio libraries" variant="settings" onClose={() => setLibrariesOpen(false)}>
      <div className="space-y-2">
        <button type="button" aria-label="Open recipes" onClick={() => { setLibrariesOpen(false); useStore.getState().setRecipesOpen(true) }}
          className="flex min-h-11 w-full items-center gap-3 rounded-xl border border-border px-3 text-left text-xs text-text-secondary hover:bg-bg-hover"><BookMarked size={16}/><span>Recipes<span className="block text-[10px] text-text-muted">Load a saved generation setup</span></span></button>
        {!isOutpaint && <button type="button" aria-label="Open model browser" onClick={() => { setLibrariesOpen(false); useStore.getState().setLoraBrowserOpen(true, modelType) }}
          className="flex min-h-11 w-full items-center gap-3 rounded-xl border border-border px-3 text-left text-xs text-text-secondary hover:bg-bg-hover"><Globe size={16}/><span>Model Browser<span className="block text-[10px] text-text-muted">CivitAI, Hugging Face, URLs and characters</span></span></button>}
      </div>
    </SidebarDialog>
  </div>
}
