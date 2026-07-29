import { useRef, useCallback, useState } from 'react'
import { Upload, X, UserRoundPen, Loader2, Eye } from 'lucide-react'
import { useStore } from '../../stores/useStore'
import { VideoTimelineSelector } from '../shared/VideoTimelineSelector'
import * as api from '../../api/client'

/**
 * Recast sub-mode — SCAIL-2 Replace: swap a person in an existing video
 * for the character in a reference image. Maestro builds the tracking
 * mask automatically from the "who to replace" keyword (SAM3), so no
 * manual masking is needed. The prompt below is optional; describing the
 * new character in the scene helps identity.
 */
export function RecastControls() {
  const editVideoFile = useStore(s => s.editVideoFile)
  const editVideoPath = useStore(s => s.editVideoPath)
  const editVideoUrl = useStore(s => s.editVideoUrl)
  const editVideoDuration = useStore(s => s.editVideoDuration)
  const editStartTime = useStore(s => s.editStartTime)
  const editEndTime = useStore(s => s.editEndTime)
  const setEditVideo = useStore(s => s.setEditVideo)
  const clearEditVideo = useStore(s => s.clearEditVideo)
  const recastTarget = useStore(s => s.editRecastTarget)
  const refKeyword = useStore(s => {
    const custom = s.params.custom_settings as Record<string, unknown> | undefined
    return typeof custom?.image_ref_keyword_content === 'string'
      ? custom.image_ref_keyword_content
      : 'human character'
  })
  const setParam = useStore(s => s.setParam)
  const refFile = useStore(s => s.editRecastRefFile)
  const refUrl = useStore(s => s.editRecastRefUrl)
  const setEditRecastRef = useStore(s => s.setEditRecastRef)

  const [previewImg, setPreviewImg] = useState<string | null>(null)
  const [previewState, setPreviewState] = useState<'idle' | 'loading' | 'found' | 'notfound' | 'error'>('idle')
  const videoFileRef = useRef<HTMLInputElement>(null)
  const refFileRef = useRef<HTMLInputElement>(null)

  const handleVideoUpload = useCallback(async (file: File) => {
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
      setPreviewImg(null)
      setPreviewState('idle')
    } catch {
      console.error('Failed to upload video')
    }
  }, [setEditVideo])

  const handleRefUpload = useCallback(async (file: File) => {
    try {
      const result = await api.uploadImage(file)
      setEditRecastRef(file, result.path, URL.createObjectURL(file))
    } catch {
      console.error('Failed to upload reference image')
    }
  }, [setEditRecastRef])

  const handlePreview = useCallback(async () => {
    if (!editVideoPath) return
    setPreviewState('loading')
    try {
      const res = await api.recastPreview({
        video_path: editVideoPath,
        target: recastTarget || 'person',
        time: editStartTime,
      })
      setPreviewImg(res.preview)
      setPreviewState(res.found ? 'found' : 'notfound')
    } catch (e) {
      console.error('Recast preview failed:', e)
      setPreviewState('error')
    }
  }, [editVideoPath, recastTarget, editStartTime])

  return (
    <div className="space-y-3">
      {/* Header hint */}
      <div className="flex items-start gap-2 bg-accent-blue/10 border border-accent-blue/20 rounded-lg px-2.5 py-2">
        <UserRoundPen size={12} className="text-accent-blue mt-0.5 shrink-0" />
        <p className="text-[10px] text-text-secondary leading-snug">
          Replace a person in the video with your reference character
          (SCAIL-2). The mask is built automatically — just say who to
          replace. Scene, camera, and audio are preserved.
        </p>
      </div>

      {/* Source video upload or timeline */}
      {!editVideoFile ? (
        <div
          onDragOver={e => e.preventDefault()}
          onDrop={e => {
            e.preventDefault()
            const file = e.dataTransfer.files[0]
            if (file && file.type.startsWith('video/')) handleVideoUpload(file)
          }}
          onClick={() => videoFileRef.current?.click()}
          className="border-2 border-dashed border-border rounded-lg p-6 text-center cursor-pointer hover:border-accent-blue/50 hover:bg-bg-hover/30 transition-all"
        >
          <Upload size={24} className="mx-auto mb-2 text-text-muted" />
          <p className="text-xs text-text-secondary">Drop a video or click to upload</p>
          <p className="text-[9px] text-text-muted mt-1">Optional: trim to a time range below</p>
          <input ref={videoFileRef} type="file" accept="video/*" className="hidden"
            onChange={e => { if (e.target.files?.[0]) handleVideoUpload(e.target.files[0]) }} />
        </div>
      ) : (
        <div className="relative">
          <button onClick={() => { clearEditVideo(); setPreviewImg(null); setPreviewState('idle') }}
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

      {/* Who to replace + preview */}
      <div>
        <label className="text-[10px] text-text-muted uppercase tracking-wider mb-1 block">Who to replace</label>
        <div className="flex gap-1.5">
          <input
            type="text"
            value={recastTarget}
            onChange={e => useStore.setState({ editRecastTarget: e.target.value })}
            placeholder="person"
            className="flex-1 bg-bg-tertiary border border-border rounded px-2 py-1.5 text-xs text-text-primary placeholder:text-text-muted focus:outline-none focus:border-accent-blue"
          />
          <button
            onClick={handlePreview}
            disabled={!editVideoPath || previewState === 'loading'}
            className="px-2 py-1.5 rounded bg-bg-tertiary border border-border text-text-secondary hover:text-text-primary hover:border-accent-blue/50 transition-colors disabled:opacity-40"
            title="Preview which person will be replaced (first use loads the detector, ~15s)"
          >
            {previewState === 'loading' ? <Loader2 size={13} className="animate-spin" /> : <Eye size={13} />}
          </button>
        </div>
        <p className="text-[9px] text-text-muted mt-0.5">
          A short description works best: "woman", "man in red", "the singer".
        </p>
        {previewImg && (
          <div className="mt-1.5">
            <img src={previewImg} alt="Target preview" className="w-full rounded border border-border" />
            {previewState === 'notfound' && (
              <p className="text-[9px] text-status-warning mt-0.5">Nothing matched — try a different description.</p>
            )}
          </div>
        )}
        {previewState === 'error' && (
          <p className="text-[9px] text-status-error mt-0.5">Preview failed — check the server log.</p>
        )}
      </div>

      {/* The source-video target and reference-image target are separate
          detector prompts. Keep them together so the distinction is visible
          before the user uploads either asset. */}
      <div className="bg-bg-tertiary/50 border border-border/70 rounded-lg p-2.5">
        <label className="text-[10px] text-text-muted uppercase tracking-wider mb-1 block">Reference image contains</label>
        <input
          type="text"
          value={refKeyword}
          onChange={e => {
            const custom = { ...(useStore.getState().params.custom_settings || {}) } as Record<string, unknown>
            custom.image_ref_keyword_content = e.target.value
            setParam('custom_settings', custom)
          }}
          placeholder="human character"
          className="w-full bg-bg-tertiary border border-border rounded px-2 py-1.5 text-xs text-text-primary placeholder:text-text-muted focus:outline-none focus:border-accent-blue"
        />
        <p className="text-[9px] text-text-muted mt-1">
          Describe the subject in the new-character image, e.g. &quot;red panda&quot;.
        </p>
      </div>

      {/* Reference character image */}
      <div>
        <label className="text-[10px] text-text-muted uppercase tracking-wider mb-1 block">New character</label>
        {!refFile ? (
          <div
            onDragOver={e => e.preventDefault()}
            onDrop={e => {
              e.preventDefault()
              const file = e.dataTransfer.files[0]
              if (file && file.type.startsWith('image/')) handleRefUpload(file)
            }}
            onClick={() => refFileRef.current?.click()}
            className="border-2 border-dashed border-border rounded-lg p-4 text-center cursor-pointer hover:border-accent-blue/50 hover:bg-bg-hover/30 transition-all"
          >
            <Upload size={18} className="mx-auto mb-1 text-text-muted" />
            <p className="text-[10px] text-text-secondary">Drop the character image</p>
            <p className="text-[9px] text-text-muted mt-0.5">A clear full/half-body shot works best</p>
            <input ref={refFileRef} type="file" accept="image/*" className="hidden"
              onChange={e => { if (e.target.files?.[0]) handleRefUpload(e.target.files[0]) }} />
          </div>
        ) : (
          <div className="relative inline-block">
            <img src={refUrl} alt="Reference character" className="h-24 rounded border border-border" />
            <button onClick={() => setEditRecastRef(null, '', '')}
              className="absolute top-1 right-1 p-0.5 rounded-full bg-black/60 text-white/80 hover:text-white hover:bg-black/80 transition-colors">
              <X size={12} />
            </button>
          </div>
        )}
      </div>

      <p className="text-[9px] text-text-muted">
        The prompt below is optional — describing the new character in the
        scene (clothes, hair, lighting) improves identity.
      </p>
    </div>
  )
}
