import { useState, useCallback, useEffect } from 'react'
import { Upload, X } from 'lucide-react'
import { useStore } from '../../stores/useStore'

export function ImageRefSection() {
  const modelOptions = useStore(s => s.modelOptions)
  const imageRefs = useStore(s => s.imageRefs)
  const imageRefType = useStore(s => s.imageRefType)
  const removeBackgroundRefs = useStore(s => s.removeBackgroundRefs)
  const addImageRef = useStore(s => s.addImageRef)
  const removeImageRef = useStore(s => s.removeImageRef)
  const reorderImageRefs = useStore(s => s.reorderImageRefs)
  const setImageRefType = useStore(s => s.setImageRefType)
  const setRemoveBackgroundRefs = useStore(s => s.setRemoveBackgroundRefs)
  const [dragOverIndex, setDragOverIndex] = useState<number | null>(null)

  const config = modelOptions?.image_ref_choices
  const bgLabel = modelOptions?.background_removal_label

  // Determine available modes from choices
  const hasLandscapeMode = config?.choices?.some(([, v]: [string, string]) => v.includes('K')) ?? false
  const hasPeopleMode = config?.choices?.some(([, v]: [string, string]) => v === 'I') ?? false
  const defaultRefType = hasLandscapeMode ? 'KI' : hasPeopleMode ? 'I' : ''

  // Auto-set ref type when images are added/removed
  useEffect(() => {
    if (!config) return
    if (imageRefs.length > 0 && imageRefType === '') {
      setImageRefType(defaultRefType)
    } else if (imageRefs.length === 0 && imageRefType !== '') {
      setImageRefType('')
    }
  }, [config, defaultRefType, imageRefs.length, imageRefType, setImageRefType])

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    // If it's a reorder drag (has our index data), ignore — handled by item onDrop
    if (e.dataTransfer.getData('ref-index')) return
    const files = Array.from(e.dataTransfer.files).filter(f => f.type.startsWith('image/'))
    files.forEach(addImageRef)
  }, [addImageRef])

  const handleFileSelect = useCallback(() => {
    const input = document.createElement('input')
    input.type = 'file'
    input.accept = '.png,.jpg,.jpeg,.webp,.bmp'
    input.multiple = true
    input.onchange = () => {
      const files = Array.from(input.files || [])
      files.forEach(addImageRef)
    }
    input.click()
  }, [addImageRef])

  if (!config) return null

  return (
    <div className="space-y-2">
      <label className="text-[11px] text-text-muted uppercase tracking-wider block">
        Reference Images
      </label>

      {/* Thumbnails + add button in a unified row */}
      <div className="flex flex-wrap gap-1.5">
        {imageRefs.map((file, i) => (
          <div
            key={`${i}-${file.name}`}
            draggable
            onDragStart={e => {
              e.dataTransfer.setData('ref-index', String(i))
              e.dataTransfer.effectAllowed = 'move'
            }}
            onDragOver={e => {
              e.preventDefault()
              e.dataTransfer.dropEffect = 'move'
              setDragOverIndex(i)
            }}
            onDragLeave={() => setDragOverIndex(null)}
            onDrop={e => {
              e.preventDefault()
              e.stopPropagation()
              setDragOverIndex(null)
              const from = parseInt(e.dataTransfer.getData('ref-index'), 10)
              if (!isNaN(from) && from !== i) reorderImageRefs(from, i)
            }}
            className={`relative w-[90px] h-[90px] rounded-lg overflow-hidden border group cursor-grab active:cursor-grabbing transition-colors ${
              dragOverIndex === i ? 'border-accent-blue border-2' : 'border-border'
            }`}
          >
            <img
              src={URL.createObjectURL(file)}
              alt={`Ref ${i + 1}`}
              className="w-full h-full object-cover pointer-events-none"
            />
            {i === 0 && imageRefs.length > 1 && hasLandscapeMode && imageRefType === 'KI' && (
              <div className="absolute bottom-0 left-0 right-0 bg-black/60 text-[8px] text-white text-center py-0.5">
                Main
              </div>
            )}
            {/* Position number */}
            <span className="absolute top-0.5 left-0.5 bg-black/60 text-white text-[8px] px-1 rounded pointer-events-none">
              {i + 1}
            </span>
            <button
              onClick={() => removeImageRef(i)}
              className="absolute top-0.5 right-0.5 bg-bg-primary/80 rounded-full p-0.5 opacity-0 group-hover:opacity-100 transition-opacity hover:bg-bg-hover"
            >
              <X size={10} />
            </button>
          </div>
        ))}

        {/* Add button / drop zone */}
        <div
          className="w-[90px] h-[90px] border border-dashed border-border rounded-lg flex flex-col items-center justify-center gap-0.5 cursor-pointer hover:border-border-light transition-colors"
          onDrop={handleDrop}
          onDragOver={e => e.preventDefault()}
          onClick={handleFileSelect}
        >
          <Upload size={14} className="text-text-muted" />
          <span className="text-[8px] text-text-muted">Add</span>
        </div>
      </div>

      {/* Focus mode toggle — only when images present and model supports both modes */}
      {imageRefs.length > 0 && hasLandscapeMode && hasPeopleMode && (
        <div className="flex bg-bg-tertiary rounded-lg p-0.5 border border-border">
          <button
            onClick={() => setImageRefType('KI')}
            className={`flex-1 text-[10px] py-1.5 rounded-md transition-all ${
              imageRefType === 'KI'
                ? 'bg-bg-active text-text-primary'
                : 'text-text-secondary hover:text-text-primary'
            }`}
          >
            Subject / Landscape
          </button>
          <button
            onClick={() => setImageRefType('I')}
            className={`flex-1 text-[10px] py-1.5 rounded-md transition-all ${
              imageRefType === 'I'
                ? 'bg-bg-active text-text-primary'
                : 'text-text-secondary hover:text-text-primary'
            }`}
          >
            People / Objects
          </button>
        </div>
      )}

      {/* Hint text */}
      {imageRefs.length > 0 && hasLandscapeMode && imageRefType === 'KI' && (
        <p className="text-[10px] text-text-muted">
          First image is the main subject/landscape. Additional images are people/objects to inject. Drag to reorder.
        </p>
      )}

      {/* Background removal toggle */}
      {imageRefs.length > 0 && bgLabel && (
        <label className="flex items-start gap-2 cursor-pointer">
          <input
            type="checkbox"
            checked={removeBackgroundRefs}
            onChange={e => setRemoveBackgroundRefs(e.target.checked)}
            className="mt-0.5 w-3.5 h-3.5 rounded border-border bg-bg-tertiary accent-accent-blue shrink-0"
          />
          <span className="text-[10px] text-text-secondary leading-tight">{bgLabel}</span>
        </label>
      )}
    </div>
  )
}
