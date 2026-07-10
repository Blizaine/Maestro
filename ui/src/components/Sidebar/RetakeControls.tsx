import { useRef, useCallback } from 'react'
import { Upload, X } from 'lucide-react'
import { useStore } from '../../stores/useStore'
import { VideoTimelineSelector } from '../shared/VideoTimelineSelector'
import * as api from '../../api/client'

export function RetakeControls() {
  const editVideoFile = useStore(s => s.editVideoFile)
  const editVideoPath = useStore(s => s.editVideoPath)
  const editVideoUrl = useStore(s => s.editVideoUrl)
  const editVideoDuration = useStore(s => s.editVideoDuration)
  const editStartTime = useStore(s => s.editStartTime)
  const editEndTime = useStore(s => s.editEndTime)
  const editRetakeStrength = useStore(s => s.editRetakeStrength)
  const editRetakeEngine = useStore(s => s.editRetakeEngine)
  const editRegenerateAudio = useStore(s => s.editRegenerateAudio)
  const setEditVideo = useStore(s => s.setEditVideo)
  const clearEditVideo = useStore(s => s.clearEditVideo)
  const fileRef = useRef<HTMLInputElement>(null)

  const handleUpload = useCallback(async (file: File) => {
    try {
      const result = await api.uploadImage(file)
      const url = URL.createObjectURL(file)
      const video = document.createElement('video')
      video.src = url
      video.onloadedmetadata = () => {
        const duration = video.duration && isFinite(video.duration) ? video.duration : 0
        const resolution = `${video.videoWidth}x${video.videoHeight}`
        setEditVideo(file, result.path, url, duration, resolution)
      }
    } catch {
      console.error('Failed to upload video')
    }
  }, [setEditVideo])

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    const file = e.dataTransfer.files[0]
    if (file && file.type.startsWith('video/')) handleUpload(file)
  }, [handleUpload])

  return (
    <div className="space-y-3">
      {/* Video Upload or Timeline */}
      {!editVideoFile ? (
        <div
          onDragOver={e => e.preventDefault()}
          onDrop={handleDrop}
          onClick={() => fileRef.current?.click()}
          className="border-2 border-dashed border-border rounded-lg p-6 text-center cursor-pointer hover:border-accent-blue/50 hover:bg-bg-hover/30 transition-all"
        >
          <Upload size={24} className="mx-auto mb-2 text-text-muted" />
          <p className="text-xs text-text-secondary">Drop a video or click to upload</p>
          <p className="text-[9px] text-text-muted mt-1">Select the part you want to edit, then describe the change</p>
          <input ref={fileRef} type="file" accept="video/*" className="hidden"
            onChange={e => { if (e.target.files?.[0]) handleUpload(e.target.files[0]) }} />
        </div>
      ) : (
        <div className="relative">
          <button onClick={clearEditVideo}
            className="absolute top-1.5 right-1.5 z-20 p-1 rounded-full bg-black/60 text-white/80 hover:text-white hover:bg-black/80 transition-colors">
            <X size={14} />
          </button>
          <VideoTimelineSelector
            videoUrl={editVideoUrl}
            duration={editVideoDuration}
            startTime={editStartTime}
            endTime={editEndTime}
            onStartChange={t => useStore.setState({ editStartTime: t })}
            onEndChange={t => useStore.setState({ editEndTime: t })}
          />
          <p className="text-[9px] text-text-muted mt-1 truncate">{editVideoFile.name}</p>
        </div>
      )}

      {/* Regenerate Audio toggle — native engine only */}
      {editRetakeEngine === 'native' && editVideoPath && (
        <label className="flex items-center gap-2 cursor-pointer">
          <input type="checkbox" checked={editRegenerateAudio}
            onChange={e => useStore.setState({ editRegenerateAudio: e.target.checked })}
            className="w-3.5 h-3.5 rounded border-border accent-accent-blue" />
          <span className="text-[10px] text-text-secondary">Regenerate Audio</span>
          <span className="text-[9px] text-text-muted ml-auto">
            {editRegenerateAudio ? 'New audio' : 'Keep source'}
          </span>
        </label>
      )}

      {/* Strength — legacy engine only */}
      {editRetakeEngine === 'legacy' && (
        <div>
          <div className="flex items-center justify-between mb-1">
            <label className="text-[10px] text-text-muted uppercase tracking-wider">Retake Strength</label>
            <span className="text-[10px] text-text-secondary">{editRetakeStrength.toFixed(2)}</span>
          </div>
          <input type="range" min={0.1} max={1} step={0.05} value={editRetakeStrength}
            onChange={e => useStore.setState({ editRetakeStrength: parseFloat(e.target.value) })} className="w-full" />
        </div>
      )}

      {/* Engine toggle */}
      <div>
        <label className="text-[10px] text-text-muted uppercase tracking-wider mb-1 block">Retake Engine</label>
        <div className="flex gap-1">
          <button onClick={() => useStore.setState({ editRetakeEngine: 'native' })}
            className={`flex-1 px-2 py-1.5 text-[10px] rounded transition-colors ${
              editRetakeEngine === 'native' ? 'bg-accent-blue text-white' : 'bg-bg-tertiary text-text-secondary hover:text-text-primary'
            }`}>
            Native
          </button>
          <button onClick={() => useStore.setState({ editRetakeEngine: 'legacy' })}
            className={`flex-1 px-2 py-1.5 text-[10px] rounded transition-colors ${
              editRetakeEngine === 'legacy' ? 'bg-accent-blue text-white' : 'bg-bg-tertiary text-text-secondary hover:text-text-primary'
            }`}>
            Legacy
          </button>
        </div>
        <p className="text-[9px] text-text-muted mt-0.5">
          {editRetakeEngine === 'native'
            ? 'Lightricks denoise_mask — preserves source identity'
            : 'MaskInjection — strength-controlled blending'}
        </p>
      </div>
    </div>
  )
}
