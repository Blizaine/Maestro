import { useState, useCallback, useEffect } from 'react'
import { Image as ImageIcon, X } from 'lucide-react'
import { useStore } from '../../stores/useStore'
import * as api from '../../api/client'
import { CharacterImagePickerButton } from '../Characters/CharacterImagePicker'
import { CharacterToolbarItem } from './SidebarPanels'
import { MediaAddTile } from './MediaInputCard'

export function ImageRefSection() {
  const modelOptions = useStore(s => s.modelOptions)
  const generationMode = useStore(s => s.generationMode)
  const imageWorkflow = useStore(s => s.studioImageWorkflow)
  const imageMode = useStore(s => Number(s.params.image_mode ?? 1))
  const imageRefs = useStore(s => s.imageRefs)
  const imageRefType = useStore(s => s.imageRefType)
  const removeBackgroundRefs = useStore(s => s.removeBackgroundRefs)
  const addImageRef = useStore(s => s.addImageRef)
  const removeImageRef = useStore(s => s.removeImageRef)
  const reorderImageRefs = useStore(s => s.reorderImageRefs)
  const setImageRefType = useStore(s => s.setImageRefType)
  const setRemoveBackgroundRefs = useStore(s => s.setRemoveBackgroundRefs)
  const outputs = useStore(s => s.outputs)
  const selectedOutput = useStore(s => s.selectedOutput)
  const selectedGalleryImage = outputs[selectedOutput]?.type === 'image'
    ? outputs[selectedOutput]
    : null
  const [dragOverIndex, setDragOverIndex] = useState<number | null>(null)
  const [copyingGalleryImage, setCopyingGalleryImage] = useState(false)

  const config = modelOptions?.image_ref_choices
  const isAdaptiveImageGenerate = generationMode === 'image' && imageWorkflow === 'generate'
  const bgLabel = modelOptions?.background_removal_label
  // max_image_refs is the model's total conditioning-image budget. In Edit
  // mode the uploaded source already consumes one slot.
  const configuredMaxRefs = modelOptions?.max_image_refs ?? null
  const maxRefs = configuredMaxRefs == null ? null : Math.max(0, configuredMaxRefs - (imageMode === 2 ? 1 : 0))
  const canAddMore = maxRefs == null || imageRefs.length < maxRefs

  const addFiles = useCallback((files: File[]) => {
    const room = maxRefs == null ? files.length : Math.max(0, maxRefs - imageRefs.length)
    files.slice(0, room).forEach(addImageRef)
  }, [addImageRef, imageRefs.length, maxRefs])

  // Determine available modes from choices
  const hasLandscapeMode = config?.choices?.some(([, v]: [string, string]) => v.includes('K')) ?? false
  const hasPeopleMode = config?.choices?.some(([, v]: [string, string]) => v === 'I') ?? false
  const defaultRefType = hasLandscapeMode ? 'KI' : hasPeopleMode ? 'I' : ''

  // Auto-set ref type when images are added/removed
  useEffect(() => {
    if (!config) return
    const validRefTypes = new Set(config.choices?.map(([, value]) => value) ?? [])
    if (imageRefs.length > 0 && (imageRefType === '' || !validRefTypes.has(imageRefType))) {
      setImageRefType(defaultRefType)
    } else if (imageRefs.length === 0 && imageRefType !== '') {
      setImageRefType('')
    }
  }, [config, defaultRefType, imageRefs.length, imageRefType, setImageRefType])

  const addSelectedGalleryImage = useCallback(async () => {
    if (!selectedGalleryImage || !canAddMore) return
    setCopyingGalleryImage(true)
    try {
      const response = await fetch(selectedGalleryImage.url)
      if (!response.ok) throw new Error(`Gallery image returned ${response.status}`)
      const blob = await response.blob()
      addFiles([
        new File(
          [blob],
          selectedGalleryImage.name,
          { type: blob.type || 'image/png' },
        ),
      ])
    } catch (error) {
      console.error('Could not add selected gallery image:', error)
    } finally {
      setCopyingGalleryImage(false)
    }
  }, [addFiles, canAddMore, selectedGalleryImage])

  // Image Generate must always expose its optional source picker. Starting
  // from a T2I model is valid; adding the first image immediately filters and
  // switches the selector to a compatible I2I/edit model.
  if (!config && !isAdaptiveImageGenerate) return null

  return (
    <div className="space-y-2">
      <label className="text-[11px] text-text-muted uppercase tracking-wider block">
        {isAdaptiveImageGenerate ? 'Source / Reference Images (Optional)' : 'Reference Images'}
      </label>

      {/* Thumbnails + add button in a unified row */}
      <div className="grid grid-cols-2 gap-2">
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
            className={`relative h-[128px] min-w-0 rounded-xl overflow-hidden border group cursor-grab active:cursor-grabbing transition-colors ${
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
            {i === 0 && isAdaptiveImageGenerate && (
              <div className="absolute bottom-0 left-0 right-0 bg-black/60 text-[8px] text-white text-center py-0.5">
                Source
              </div>
            )}
            {/* Position number */}
            <span className="absolute top-0.5 left-0.5 bg-black/60 text-white text-[8px] px-1 rounded pointer-events-none">
              {i + 1}
            </span>
            <button
              type="button" aria-label={`Remove reference image ${i + 1}`}
              onClick={() => removeImageRef(i)}
              className="absolute top-1 right-1 rounded-full bg-black/65 p-2 text-white hover:bg-black/85"
            >
              <X size={10} />
            </button>
          </div>
        ))}

        {/* Add button / drop zone */}
        {canAddMore && (
          <MediaAddTile label={isAdaptiveImageGenerate && !imageRefs.length ? 'Add source image' : 'Add reference'} hint="Drop or choose an image"
            accept=".png,.jpg,.jpeg,.webp,.bmp" onFiles={files => addFiles(files.filter(file => file.type.startsWith('image/') || /\.(png|jpe?g|webp|bmp)$/i.test(file.name)))}/>
        )}
      </div>

      {generationMode === 'image' && <CharacterToolbarItem><CharacterImagePickerButton label="Characters" disabled={!canAddMore}
        maxImages={maxRefs == null ? null : Math.max(0, maxRefs - imageRefs.length)}
        onSelect={async (character, images) => {
          const files = await Promise.all(images.map(image => api.characterImageFile(character, image)))
          addFiles(files)
        }}/></CharacterToolbarItem>}

      {isAdaptiveImageGenerate && (
        <button
          type="button"
          onClick={() => void addSelectedGalleryImage()}
          disabled={!selectedGalleryImage || !canAddMore || copyingGalleryImage}
          className="flex w-full items-center justify-center gap-1.5 rounded-md border border-border bg-bg-tertiary py-1.5 text-[11px] text-text-secondary transition-colors hover:text-text-primary disabled:cursor-not-allowed disabled:opacity-40"
        >
          <ImageIcon size={12} />
          {copyingGalleryImage
            ? 'Adding selected image…'
            : selectedGalleryImage
              ? 'Add selected gallery image'
              : 'Select an image in the gallery to add it'}
        </button>
      )}

      {maxRefs != null && (
        <p className="text-[9px] text-text-muted">
          Up to {maxRefs} reference image{maxRefs === 1 ? '' : 's'}.
        </p>
      )}

      {isAdaptiveImageGenerate && imageRefs.length > 0 && (
        <p className="text-[10px] text-text-muted">
          Describe the finished image below. Add more images when the model supports multi-reference editing.
        </p>
      )}

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
