import { useState, useCallback } from 'react'
import { Upload, X } from 'lucide-react'
import { useStore } from '../../stores/useStore'
import * as api from '../../api/client'

const controlTypes = [
  { value: 'DVG', label: 'Depth', desc: 'Preserve 3D spatial layout' },
  { value: 'PVG', label: 'Motion', desc: 'Transfer human poses/movement' },
  { value: 'EVG', label: 'Edges', desc: 'Preserve shapes and outlines' },
]

export function RestyleControls() {
  const setParam = useStore(s => s.setParam)
  const [sourceVideo, setSourceVideo] = useState<File | null>(null)
  const [controlType, setControlType] = useState('DVG')
  const [uploading, setUploading] = useState(false)

  const handleVideoUpload = useCallback(async (file: File) => {
    setUploading(true)
    try {
      const result = await api.uploadImage(file)
      setSourceVideo(file)
      setParam('video_guide', result.path)
      setParam('video_prompt_type', controlType)
    } catch (e) {
      console.error('Video upload failed:', e)
    } finally {
      setUploading(false)
    }
  }, [controlType, setParam])

  const handleControlTypeChange = (value: string) => {
    setControlType(value)
    if (sourceVideo) {
      setParam('video_prompt_type', value)
    }
  }

  return (
    <div className="space-y-3">
      {/* Control Type */}
      <div>
        <label className="text-[11px] text-text-muted uppercase tracking-wider mb-1.5 block">
          Control Type
        </label>
        <div className="flex bg-bg-tertiary rounded-lg p-0.5 border border-border">
          {controlTypes.map(ct => (
            <button
              key={ct.value}
              onClick={() => handleControlTypeChange(ct.value)}
              className={`flex-1 text-[10px] py-1.5 rounded-md transition-all ${
                controlType === ct.value
                  ? 'bg-bg-active text-text-primary'
                  : 'text-text-secondary hover:text-text-primary'
              }`}
              title={ct.desc}
            >
              {ct.label}
            </button>
          ))}
        </div>
        <p className="text-[9px] text-text-muted mt-1">
          {controlTypes.find(ct => ct.value === controlType)?.desc}
        </p>
      </div>

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
                  setParam('video_guide', undefined)
                  setParam('video_prompt_type', '')
                }}
                className="p-0.5 rounded hover:bg-bg-hover text-text-muted"
              >
                <X size={12} />
              </button>
            </div>
          ) : (
            <div className="flex flex-col items-center justify-center gap-1 py-4 text-text-muted">
              <Upload size={18} />
              <span className="text-[10px]">{uploading ? 'Uploading...' : 'Drop video to restyle'}</span>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
