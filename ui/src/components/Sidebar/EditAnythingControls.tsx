import { useRef, useCallback, useEffect, useState } from 'react'
import { Upload, X, Sparkles } from 'lucide-react'
import { useStore } from '../../stores/useStore'
import { VideoTimelineSelector } from '../shared/VideoTimelineSelector'
import * as api from '../../api/client'

/**
 * Edit Anything sub-mode — prompt-driven video edit via the
 * Alissonerdx/LTX-LoRAs "Edit Anything" LoRA. No SAM mask required;
 * the LoRA interprets Add / Remove / Replace / Style patterns directly.
 */
export function EditAnythingControls() {
  const editVideoFile = useStore(s => s.editVideoFile)
  const editVideoUrl = useStore(s => s.editVideoUrl)
  const editVideoDuration = useStore(s => s.editVideoDuration)
  const editStartTime = useStore(s => s.editStartTime)
  const editEndTime = useStore(s => s.editEndTime)
  const loraStrength = useStore(s => s.editAnythingLoraStrength)
  const retakeStrength = useStore(s => s.editRetakeStrength)
  const setEditVideo = useStore(s => s.setEditVideo)
  const clearEditVideo = useStore(s => s.clearEditVideo)
  const ensureEditAnythingLora = useStore(s => s.ensureEditAnythingLora)

  const [showAdvanced, setShowAdvanced] = useState(false)
  const fileRef = useRef<HTMLInputElement>(null)

  // On mount, make sure the Edit Anything LoRA is downloaded. Idempotent.
  useEffect(() => {
    void ensureEditAnythingLora()
  }, [ensureEditAnythingLora])

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
      {/* Header hint */}
      <div className="flex items-start gap-2 bg-accent-blue/10 border border-accent-blue/20 rounded-lg px-2.5 py-2">
        <Sparkles size={12} className="text-accent-blue mt-0.5 shrink-0" />
        <p className="text-[10px] text-text-secondary leading-snug">
          Prompt-driven edit — no mask needed. Write one of:
          <br />
          <span className="text-text-muted">• </span><span className="text-text-primary">Add</span> a [thing] [location]
          <br />
          <span className="text-text-muted">• </span><span className="text-text-primary">Remove</span> the [thing]
          <br />
          <span className="text-text-muted">• </span><span className="text-text-primary">Replace</span> the [X] with a [Y]
          <br />
          <span className="text-text-muted">• </span><span className="text-text-primary">Convert</span> the video into a [style] style
        </p>
      </div>

      {/* Video upload or timeline */}
      {!editVideoFile ? (
        <div
          onDragOver={e => e.preventDefault()}
          onDrop={handleDrop}
          onClick={() => fileRef.current?.click()}
          className="border-2 border-dashed border-border rounded-lg p-6 text-center cursor-pointer hover:border-accent-blue/50 hover:bg-bg-hover/30 transition-all"
        >
          <Upload size={24} className="mx-auto mb-2 text-text-muted" />
          <p className="text-xs text-text-secondary">Drop a video or click to upload</p>
          <p className="text-[9px] text-text-muted mt-1">Optional: select a time range below to edit only part of the clip</p>
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

      {/* Advanced knobs */}
      <button onClick={() => setShowAdvanced(!showAdvanced)}
        className="text-[10px] text-text-muted hover:text-text-primary transition-colors">
        {showAdvanced ? '▾' : '▸'} Advanced
      </button>
      {showAdvanced && (
        <div className="space-y-3 pl-2 border-l border-border/50">
          <div>
            <div className="flex items-center justify-between mb-1">
              <label className="text-[10px] text-text-muted uppercase tracking-wider">LoRA Strength</label>
              <span className="text-[10px] text-text-secondary">{loraStrength.toFixed(2)}</span>
            </div>
            <input
              type="range" min={0.5} max={1.5} step={0.05}
              value={loraStrength}
              onChange={e => useStore.setState({ editAnythingLoraStrength: parseFloat(e.target.value) })}
              className="w-full"
            />
            <p className="text-[9px] text-text-muted mt-0.5">
              Per the LoRA card: start at 1.0. Bump to 1.2 if the edit is too weak; drop below if it distorts unrelated content.
            </p>
          </div>
          <div>
            <div className="flex items-center justify-between mb-1">
              <label className="text-[10px] text-text-muted uppercase tracking-wider">Source Preservation</label>
              <span className="text-[10px] text-text-secondary">{retakeStrength.toFixed(2)}</span>
            </div>
            <input
              type="range" min={0.3} max={1.0} step={0.05}
              value={retakeStrength}
              onChange={e => useStore.setState({ editRetakeStrength: parseFloat(e.target.value) })}
              className="w-full"
            />
            <p className="text-[9px] text-text-muted mt-0.5">
              How much of the source's structure is regenerated. Lower = more source preserved (subtle edit), higher = more aggressive.
            </p>
          </div>
        </div>
      )}
    </div>
  )
}
