import { useState, useCallback } from 'react'
import { Upload, X, ImageIcon } from 'lucide-react'
import { useStore } from '../../stores/useStore'
import * as api from '../../api/client'

export function VideoEditControls() {
  const params = useStore(s => s.params)
  const setParam = useStore(s => s.setParam)
  const [sourceVideo, setSourceVideo] = useState<File | null>(null)
  const [, setSourceVideoPath] = useState<string | null>(null)
  const [refImage, setRefImage] = useState<File | null>(null)
  const [uploading, setUploading] = useState(false)

  const handleVideoUpload = useCallback(async (file: File) => {
    setUploading(true)
    try {
      // Upload as video guide (same path as control video)
      const result = await api.uploadImage(file)
      setSourceVideo(file)
      setSourceVideoPath(result.path)
      setParam('video_guide', result.path)
      // Set the video prompt type to include V for video guide
      const vpt = params.video_prompt_type || ''
      if (!vpt.includes('V')) setParam('video_prompt_type', 'V' + vpt)
    } catch (e) {
      console.error('Video upload failed:', e)
    } finally {
      setUploading(false)
    }
  }, [params.video_prompt_type, setParam])

  const handleRefUpload = useCallback(async (file: File) => {
    try {
      const result = await api.uploadImage(file)
      setRefImage(file)
      setParam('image_refs', [result.path])
      // Add K and I to video_prompt_type for reference image
      const vpt = params.video_prompt_type || ''
      let newVpt = vpt
      if (!newVpt.includes('K')) newVpt += 'K'
      if (!newVpt.includes('I')) newVpt += 'I'
      setParam('video_prompt_type', newVpt)
    } catch (e) {
      console.error('Reference upload failed:', e)
    }
  }, [params.video_prompt_type, setParam])

  return (
    <div className="space-y-3">
      {/* Source Video */}
      <div>
        <label className="text-[11px] text-text-muted uppercase tracking-wider mb-1.5 block">
          Source Video
        </label>
        <div
          className={`relative border border-dashed rounded-lg overflow-hidden transition-colors cursor-pointer
            ${sourceVideo ? 'border-accent-blue' : 'border-border hover:border-border-light'}`}
          onDrop={e => {
            e.preventDefault()
            const f = e.dataTransfer.files[0]
            if (f && f.type.startsWith('video/')) handleVideoUpload(f)
          }}
          onDragOver={e => e.preventDefault()}
          onClick={() => {
            const input = document.createElement('input')
            input.type = 'file'
            input.accept = '.mp4,.webm,.avi,.mov'
            input.onchange = (ev) => {
              const f = (ev.target as HTMLInputElement).files?.[0]
              if (f) handleVideoUpload(f)
            }
            input.click()
          }}
        >
          {sourceVideo ? (
            <div className="flex items-center gap-2 px-3 py-2.5">
              <span className="text-xs text-text-primary truncate flex-1">{sourceVideo.name}</span>
              <button
                onClick={e => {
                  e.stopPropagation()
                  setSourceVideo(null)
                  setSourceVideoPath(null)
                  setParam('video_guide', undefined)
                }}
                className="p-0.5 rounded hover:bg-bg-hover text-text-muted"
              >
                <X size={12} />
              </button>
            </div>
          ) : (
            <div className="flex flex-col items-center justify-center gap-1 py-4 text-text-muted">
              <Upload size={18} />
              <span className="text-[10px]">{uploading ? 'Uploading...' : 'Drop video to edit'}</span>
            </div>
          )}
        </div>
      </div>

      {/* Reference Image (optional) */}
      <div>
        <label className="text-[11px] text-text-muted uppercase tracking-wider mb-1.5 block">
          Reference Image <span className="text-text-muted font-normal">(optional)</span>
        </label>
        <div
          className={`relative border border-dashed rounded-lg overflow-hidden transition-colors cursor-pointer min-h-[48px]
            ${refImage ? 'border-accent-blue' : 'border-border hover:border-border-light'}`}
          onDrop={e => {
            e.preventDefault()
            const f = e.dataTransfer.files[0]
            if (f && f.type.startsWith('image/')) handleRefUpload(f)
          }}
          onDragOver={e => e.preventDefault()}
          onClick={() => {
            const input = document.createElement('input')
            input.type = 'file'
            input.accept = '.png,.jpg,.jpeg,.webp'
            input.onchange = (ev) => {
              const f = (ev.target as HTMLInputElement).files?.[0]
              if (f) handleRefUpload(f)
            }
            input.click()
          }}
        >
          {refImage ? (
            <div className="flex items-center gap-2 px-3 py-2">
              <ImageIcon size={14} className="text-accent-blue shrink-0" />
              <span className="text-xs text-text-primary truncate flex-1">{refImage.name}</span>
              <button
                onClick={e => {
                  e.stopPropagation()
                  setRefImage(null)
                  setParam('image_refs', undefined)
                }}
                className="p-0.5 rounded hover:bg-bg-hover text-text-muted"
              >
                <X size={12} />
              </button>
            </div>
          ) : (
            <div className="flex items-center justify-center gap-1 py-2 text-text-muted">
              <ImageIcon size={14} />
              <span className="text-[10px]">Drop style/object reference</span>
            </div>
          )}
        </div>
        <p className="text-[9px] text-text-muted mt-1">
          For style transfer or object reference. Leave empty for instruction-only edits.
        </p>
      </div>
    </div>
  )
}
