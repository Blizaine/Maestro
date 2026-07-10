import { useStore } from '../../stores/useStore'
import type { ResolutionPreset } from '../../types'

export function ResolutionPresets() {
  const resolutionPreset = useStore(s => s.resolutionPreset)
  const setResolutionPreset = useStore(s => s.setResolutionPreset)
  const generationMode = useStore(s => s.generationMode)
  const isEdit = generationMode === 'avatar'

  const isImage = generationMode === 'image'
  // Edit and Image modes get Auto option
  const presets: ResolutionPreset[] = (isEdit || isImage)
    ? ['auto', '480p', '540p', '720p', '1080p']
    : ['480p', '540p', '720p', '1080p']

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
            {p === 'auto' ? 'Auto' : p}
          </button>
        ))}
      </div>
      {resolutionPreset === 'auto' && (
        <p className="text-[9px] text-text-muted mt-0.5">
          {isEdit ? 'Uses source clip resolution' : isImage ? 'Matches reference image aspect ratio' : 'Auto resolution'}
        </p>
      )}
    </div>
  )
}
