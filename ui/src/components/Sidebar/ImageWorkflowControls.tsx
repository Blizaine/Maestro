import { useRef, useState, type RefObject } from 'react'
import { Image as ImageIcon, Upload, X } from 'lucide-react'
import { useStore } from '../../stores/useStore'
import * as api from '../../api/client'

type SourceKind = 'source' | 'mask'

const OUTPAINT_SIDES = [
  ['top', 'Top'],
  ['bottom', 'Bottom'],
  ['left', 'Left'],
  ['right', 'Right'],
] as const

export function ImageWorkflowControls() {
  const workflow = useStore(state => state.studioImageWorkflow)
  const sourcePath = useStore(state => state.imageWorkflowSourcePath)
  const sourceUrl = useStore(state => state.imageWorkflowSourceUrl)
  const maskPath = useStore(state => state.imageWorkflowMaskPath)
  const maskUrl = useStore(state => state.imageWorkflowMaskUrl)
  const setSource = useStore(state => state.setImageWorkflowSource)
  const setMask = useStore(state => state.setImageWorkflowMask)
  const padding = useStore(state => state.imageOutpaintPadding)
  const setPadding = useStore(state => state.setImageOutpaintPadding)
  const outputs = useStore(state => state.outputs)
  const selectedOutput = useStore(state => state.selectedOutput)
  const current = outputs[selectedOutput]
  const currentIsImage = current?.type === 'image'
  const sourceInput = useRef<HTMLInputElement>(null)
  const maskInput = useRef<HTMLInputElement>(null)
  const [uploading, setUploading] = useState<SourceKind | null>(null)
  const [error, setError] = useState<string | null>(null)

  if (workflow === 'new' || workflow === 'edit' || workflow === 'upscale') return null

  const upload = async (kind: SourceKind, file: File) => {
    setUploading(kind)
    setError(null)
    try {
      const result = await api.uploadImage(file)
      const item = { file, path: result.path, url: result.url || URL.createObjectURL(file) }
      if (kind === 'source') setSource(item)
      else setMask(item)
    } catch (uploadError) {
      console.error('Image workflow upload failed:', uploadError)
      setError(`Could not upload the ${kind} image.`)
    } finally {
      setUploading(null)
    }
  }

  const applySelectedImage = async () => {
    if (!currentIsImage || !current) return
    setUploading('source')
    setError(null)
    try {
      // Gallery outputs live in workspace folders, whereas the generic image
      // generation loader consumes upload paths. Copy through the upload API
      // so the same source contract works for default and named workspaces.
      const response = await fetch(current.url)
      if (!response.ok) throw new Error(`Gallery image returned ${response.status}`)
      const blob = await response.blob()
      const file = new File([blob], current.name, { type: blob.type || 'image/png' })
      const result = await api.uploadImage(file)
      setSource({ file, path: result.path, url: result.url || current.url })
    } catch (galleryError) {
      console.error('Could not copy gallery image into Image workflow:', galleryError)
      setError('Could not use the selected gallery image.')
    } finally {
      setUploading(null)
    }
  }

  const renderInput = (
    kind: SourceKind,
    path: string,
    url: string,
    inputRef: RefObject<HTMLInputElement | null>,
  ) => {
    const isMask = kind === 'mask'
    const clear = () => isMask ? setMask(null) : setSource(null)
    return (
      <div>
        <label className="mb-1.5 block text-[11px] uppercase tracking-wider text-text-muted">
          {isMask ? 'Edit Mask' : 'Source Image'}
        </label>
        {path ? (
          <div className="relative overflow-hidden rounded-lg border border-border bg-bg-tertiary">
            {url && (
              <img
                src={url}
                alt={isMask ? 'Edit mask' : 'Source'}
                className={`max-h-52 w-full object-contain ${isMask ? 'bg-black' : 'bg-bg-primary'}`}
              />
            )}
            <button
              type="button"
              onClick={clear}
              title={`Remove ${isMask ? 'mask' : 'source'}`}
              className="absolute right-1.5 top-1.5 rounded-full bg-black/65 p-1 text-white/80 transition-colors hover:text-white"
            >
              <X size={13} />
            </button>
          </div>
        ) : (
          <div className="space-y-2">
            <button
              type="button"
              onClick={() => inputRef.current?.click()}
              className="w-full rounded-lg border-2 border-dashed border-border p-4 text-center transition-colors hover:border-accent-blue/60 hover:bg-bg-hover/30"
            >
              <Upload size={18} className="mx-auto mb-1.5 text-text-muted" />
              <span className="text-[11px] text-text-secondary">
                {uploading === kind ? 'Uploading…' : `Upload ${isMask ? 'a black-and-white mask' : 'an image'}`}
              </span>
            </button>
            <input
              ref={inputRef}
              type="file"
              accept="image/png,image/jpeg,image/webp,image/bmp"
              className="hidden"
              onChange={event => {
                const file = event.target.files?.[0]
                if (file) void upload(kind, file)
                event.currentTarget.value = ''
              }}
            />
            {!isMask && (
              <button
                type="button"
                disabled={!currentIsImage}
                onClick={() => void applySelectedImage()}
                className="flex w-full items-center justify-center gap-1.5 rounded-md border border-border bg-bg-tertiary py-1.5 text-[11px] text-text-secondary transition-colors hover:text-text-primary disabled:cursor-not-allowed disabled:opacity-40"
              >
                <ImageIcon size={12} />
                {currentIsImage ? 'Use selected gallery image' : 'Select an image in the gallery first'}
              </button>
            )}
          </div>
        )}
      </div>
    )
  }

  return (
    <div className="space-y-3">
      {renderInput('source', sourcePath, sourceUrl, sourceInput)}

      {workflow === 'inpaint' ? (
        <>
          {renderInput('mask', maskPath, maskUrl, maskInput)}
          <p className="text-[10px] leading-snug text-text-muted">
            White areas are regenerated from your prompt. Black areas stay unchanged.
          </p>
        </>
      ) : (
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <label className="text-[11px] uppercase tracking-wider text-text-muted">Expand Canvas</label>
            <div className="flex gap-1">
              <button
                type="button"
                onClick={() => {
                  setPadding('top', 0); setPadding('bottom', 0)
                  setPadding('left', 35); setPadding('right', 35)
                }}
                className="rounded border border-border px-1.5 py-1 text-[9px] text-text-secondary hover:text-text-primary"
              >
                Wide
              </button>
              <button
                type="button"
                onClick={() => {
                  setPadding('top', 35); setPadding('bottom', 35)
                  setPadding('left', 0); setPadding('right', 0)
                }}
                className="rounded border border-border px-1.5 py-1 text-[9px] text-text-secondary hover:text-text-primary"
              >
                Tall
              </button>
              <button
                type="button"
                onClick={() => OUTPAINT_SIDES.forEach(([side]) => setPadding(side, 25))}
                className="rounded border border-border px-1.5 py-1 text-[9px] text-text-secondary hover:text-text-primary"
              >
                All
              </button>
            </div>
          </div>
          <div className="grid grid-cols-2 gap-x-3 gap-y-2 rounded-lg border border-border bg-bg-tertiary p-2.5">
            {OUTPAINT_SIDES.map(([side, label]) => (
              <label key={side} className="space-y-1">
                <span className="flex justify-between text-[10px] text-text-secondary">
                  <span>{label}</span><span>{padding[side]}%</span>
                </span>
                <input
                  type="range"
                  min={0}
                  max={100}
                  step={5}
                  value={padding[side]}
                  onChange={event => setPadding(side, Number(event.target.value))}
                  className="w-full"
                />
              </label>
            ))}
          </div>
          <p className="text-[10px] leading-snug text-text-muted">
            The source stays protected while the model creates the new surrounding area.
          </p>
        </div>
      )}

      {error && (
        <div className="rounded border border-red-500/20 bg-red-500/10 px-2 py-1.5 text-[10px] text-red-400">
          {error}
        </div>
      )}
    </div>
  )
}
