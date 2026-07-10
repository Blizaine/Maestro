import { useCallback } from 'react'
import { Upload, X } from 'lucide-react'
import { useStore } from '../../stores/useStore'

export function ImageUpload() {
  const { startImage, endImage, setStartImage, setEndImage } = useStore()
  const supportsEndFrame = useStore(s => s.modelOptions?.supports_end_frame ?? false)
  const strengthLabel = useStore(s => s.modelOptions?.input_video_strength_label ?? '')
  const inputVideoStrength = useStore(s => s.params.input_video_strength ?? 1.0)
  const setParam = useStore(s => s.setParam)
  const generationMode = useStore(s => s.generationMode)

  const isImage = generationMode === 'image'
  const startLabel = isImage ? 'Input Image' : 'Start Frame'
  const endLabel = 'End Frame'

  const handleDrop = useCallback((e: React.DragEvent, target: 'start' | 'end') => {
    e.preventDefault()
    const file = e.dataTransfer.files[0]
    if (file && file.type.startsWith('image/')) {
      target === 'start' ? setStartImage(file) : setEndImage(file)
    }
  }, [setStartImage, setEndImage])

  const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>, target: 'start' | 'end') => {
    const file = e.target.files?.[0]
    if (file) {
      target === 'start' ? setStartImage(file) : setEndImage(file)
    }
  }, [setStartImage, setEndImage])

  return (
    <div className="space-y-2">
      <div className="flex gap-2">
        <DropZone
          label={startLabel}
          sublabel={isImage ? 'Optional' : undefined}
          file={startImage}
          onDrop={e => handleDrop(e, 'start')}
          onSelect={e => handleFileSelect(e, 'start')}
          onClear={() => setStartImage(null)}
        />
        {supportsEndFrame && (
          <DropZone
            label={endLabel}
            file={endImage}
            onDrop={e => handleDrop(e, 'end')}
            onSelect={e => handleFileSelect(e, 'end')}
            onClear={() => setEndImage(null)}
          />
        )}
      </div>
      {strengthLabel && startImage && (
        <div className="space-y-1">
          <div className="flex items-center justify-between">
            <label className="text-[11px] text-text-secondary">Image Strength</label>
            <span className="text-[11px] text-text-muted tabular-nums">{inputVideoStrength.toFixed(2)}</span>
          </div>
          <input
            type="range"
            min={0}
            max={1}
            step={0.01}
            value={inputVideoStrength}
            onChange={e => setParam('input_video_strength', parseFloat(e.target.value))}
            className="w-full h-1 accent-accent-blue"
          />
          <p className="text-[9px] text-text-muted">Lower values can increase motion</p>
        </div>
      )}
    </div>
  )
}

function DropZone({ label, sublabel, file, onDrop, onSelect, onClear }: {
  label: string
  sublabel?: string
  file: File | null
  onDrop: (e: React.DragEvent) => void
  onSelect: (e: React.ChangeEvent<HTMLInputElement>) => void
  onClear: () => void
}) {
  return (
    <div
      className="relative flex-1 border border-dashed border-border rounded-lg p-3 flex flex-col items-center justify-center gap-1 cursor-pointer hover:border-border-light transition-colors min-h-[80px]"
      onDrop={onDrop}
      onDragOver={e => e.preventDefault()}
      onClick={() => {
        const input = document.createElement('input')
        input.type = 'file'
        input.accept = 'image/*'
        input.onchange = (e) => onSelect(e as unknown as React.ChangeEvent<HTMLInputElement>)
        input.click()
      }}
    >
      {file ? (
        <>
          <img
            src={URL.createObjectURL(file)}
            alt={label}
            className="w-full h-full object-cover rounded absolute inset-0"
          />
          <button
            onClick={e => { e.stopPropagation(); onClear() }}
            className="absolute top-1 right-1 bg-bg-primary/80 rounded-full p-0.5 hover:bg-bg-hover z-10"
          >
            <X size={12} />
          </button>
        </>
      ) : (
        <>
          <Upload size={16} className="text-text-muted" />
          <span className="text-[11px] text-text-muted text-center">{label}</span>
          {sublabel && <span className="text-[9px] text-text-muted">{sublabel}</span>}
        </>
      )}
    </div>
  )
}
