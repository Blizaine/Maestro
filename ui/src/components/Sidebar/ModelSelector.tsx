import { ChevronDown, Check, Plus } from 'lucide-react'
import { useState, useRef, useEffect } from 'react'
import { useStore, getFamiliesForMode, getModelsForFamily } from '../../stores/useStore'

export function ModelSelector() {
  const models = useStore(s => s.models)
  const families = useStore(s => s.families)
  const enabledModels = useStore(s => s.enabledModels)
  const generationMode = useStore(s => s.generationMode)
  const editSubMode = useStore(s => s.editSubMode)
  const currentModelType = useStore(s => s.params.model_type)
  const selectModel = useStore(s => s.selectModel)
  const openModelVisibility = useStore(s => s.openModelVisibility)
  // Mature Mode gate: models with nsfw_only flag are hidden from the
  // selector unless servicesConfig.nsfw_mode is enabled. Backend always
  // ships the entry (so the toggle can show/hide without a model reload)
  // but the UI clamps visibility here.
  const nsfwMode = useStore(s => s.servicesConfig?.nsfw_mode ?? false)

  const [open, setOpen] = useState(false)
  const containerRef = useRef<HTMLDivElement>(null)

  // Close on click outside
  useEffect(() => {
    if (!open) return
    function handleClick(e: MouseEvent) {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) {
        setOpen(false)
      }
    }
    document.addEventListener('mousedown', handleClick)
    return () => document.removeEventListener('mousedown', handleClick)
  }, [open])

  const audioSubMode = useStore(s => s.audioSubMode)

  const currentModel = models.find(m => m.model_type === currentModelType)
  const effectiveSubMode = generationMode === 'avatar' ? editSubMode : undefined
  const effectiveAudioSubMode = generationMode === 'audio' ? audioSubMode : undefined
  const modeFamilies = getFamiliesForMode(generationMode, families, effectiveSubMode, effectiveAudioSubMode)

  // Build grouped model list, filtered by:
  //   1. enabledModels (Settings → System → Model Visibility),
  //   2. nsfw_only gate (Mature Mode must be on for those to appear).
  const groups = modeFamilies.map(family => ({
    family,
    models: getModelsForFamily(family.id, models, generationMode, effectiveSubMode)
      .filter(m => enabledModels.has(m.model_type))
      .filter(m => !m.nsfw_only || nsfwMode),
  })).filter(g => g.models.length > 0)

  // How many models are available for this mode but NOT enabled — powers the
  // "+N" hint that nudges users toward Settings → Enabled Models.
  const disabledCount = modeFamilies.reduce((n, family) => {
    const avail = getModelsForFamily(family.id, models, generationMode, effectiveSubMode)
      .filter(m => !m.nsfw_only || nsfwMode)
    return n + avail.filter(m => !enabledModels.has(m.model_type)).length
  }, 0)

  return (
    <div className="relative flex-1 min-w-0" ref={containerRef}>
      {/* Trigger button */}
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center gap-1.5 bg-bg-tertiary border border-border rounded-lg px-2.5 py-2 text-left hover:border-border-light transition-colors"
      >
        <span className="flex-1 min-w-0 truncate text-xs text-text-primary">
          {currentModel?.name ?? 'Select model'}
        </span>
        <ChevronDown size={14} className={`shrink-0 text-text-muted transition-transform ${open ? 'rotate-180' : ''}`} />
      </button>

      {/* Dropdown (opens upward) */}
      {open && (
        <div className="absolute bottom-full left-0 mb-1 w-[360px] max-w-[90vw] bg-bg-secondary border border-border rounded-lg shadow-xl overflow-hidden z-50">
          {/* Enable-more entry — sits above the enabled model list; opens
              Settings → Enabled Models expanded to this mode. */}
          {disabledCount > 0 && (
            <button
              onClick={() => { openModelVisibility(generationMode); setOpen(false) }}
              className="w-full flex items-center gap-2 px-3 py-2 text-left border-b border-border text-text-secondary hover:bg-bg-hover hover:text-accent-blue transition-colors"
            >
              <Plus size={13} className="shrink-0" />
              <span className="flex-1 text-xs">Enable more models</span>
              <span className="text-[10px] text-text-muted shrink-0">{disabledCount} available</span>
            </button>
          )}
          <div className="max-h-[360px] overflow-y-auto py-1">
            {groups.map(({ family, models: famModels }) => (
              <div key={family.id}>
                {/* Family header */}
                <div className="px-3 pt-2 pb-1 text-[10px] text-text-muted uppercase tracking-wider font-medium">
                  {family.label}
                </div>
                {/* Models in family */}
                {famModels.map(model => {
                  const isSelected = model.model_type === currentModelType
                  return (
                    <button
                      key={model.model_type}
                      onClick={() => {
                        selectModel(model.model_type)
                        setOpen(false)
                      }}
                      className={`w-full px-3 py-1.5 flex items-center gap-2 text-left transition-colors ${
                        isSelected
                          ? 'bg-accent-blue/10 text-text-primary'
                          : 'hover:bg-bg-hover text-text-secondary hover:text-text-primary'
                      }`}
                    >
                      <span className="flex-1 min-w-0 text-xs truncate">{model.name}</span>
                      <ModelBadges model={model} />
                      {isSelected && <Check size={12} className="shrink-0 text-accent-blue" />}
                    </button>
                  )
                })}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

function ModelBadges({ model }: {
  model: { is_i2v: boolean; is_t2v: boolean; supports_end_frame?: boolean; supports_audio?: boolean; supports_ref_images?: boolean }
}) {
  const badges: string[] = []
  if (model.is_i2v && model.supports_end_frame) badges.push('S/E Frame')
  else if (model.is_i2v) badges.push('I2V')
  if (model.supports_audio) badges.push('Audio')
  if (model.supports_ref_images) badges.push('Ref')
  if (badges.length === 0) return null
  return (
    <span className="flex gap-0.5 shrink-0">
      {badges.map(b => (
        <span key={b} className="text-[9px] px-1 py-0.5 rounded bg-bg-tertiary text-text-muted leading-none">
          {b}
        </span>
      ))}
    </span>
  )
}
