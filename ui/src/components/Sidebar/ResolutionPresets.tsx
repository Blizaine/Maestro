import { useStore } from '../../stores/useStore'
import type { ResolutionPreset } from '../../types'

export function ResolutionPresets() {
  const resolutionPreset = useStore(s => s.resolutionPreset)
  const setResolutionPreset = useStore(s => s.setResolutionPreset)
  const generationMode = useStore(s => s.generationMode)
  const modelOptions = useStore(s => s.modelOptions)
  const isEdit = generationMode === 'avatar'

  const isImage = generationMode === 'image'
  // Model-specific lists take precedence so a family can label its native
  // tier and clearly identify higher-cost experimental canvases.
  const presets: ResolutionPreset[] = modelOptions?.resolution_preset_order?.length
    ? modelOptions.resolution_preset_order
    : (isEdit || isImage)
      ? ['auto', '480p', '540p', '720p', '1080p']
      : ['480p', '540p', '720p', '1080p']
  const selectedModelPreset = modelOptions?.resolution_presets?.[resolutionPreset]

  return (
    <div>
      <label className="text-[11px] text-text-muted uppercase tracking-wider mb-1.5 block">Resolution</label>
      <div className="flex bg-bg-tertiary rounded-lg p-0.5 border border-border">
        {presets.map(p => (
          <button
            key={p}
            onClick={() => setResolutionPreset(p)}
            className={`flex-1 text-xs py-1.5 rounded-md transition-all capitalize ${
              resolutionPreset === p
                ? 'bg-bg-active text-text-primary'
                : 'text-text-secondary hover:text-text-primary'
            }`}
          >
            {p === 'auto'
              ? 'Auto'
              : modelOptions?.resolution_presets?.[p]?.label || p}
          </button>
        ))}
      </div>
      {resolutionPreset === 'auto' && (
        <p className="text-[9px] text-text-muted mt-0.5">
          {isEdit ? 'Uses source clip resolution' : isImage ? 'Matches reference image aspect ratio' : 'Auto resolution'}
        </p>
      )}
      {selectedModelPreset?.hint && (
        <p className={`mt-1 text-[9px] leading-relaxed ${
          selectedModelPreset.experimental ? 'text-indicator-warning' : 'text-text-muted'
        }`}>
          {selectedModelPreset.hint}
        </p>
      )}
    </div>
  )
}
