import { useState, useRef, useEffect, useCallback } from 'react'

interface Props {
  videoUrl: string
  duration: number
  startTime: number
  endTime: number
  onStartChange: (t: number) => void
  onEndChange: (t: number) => void
  height?: number
}

/**
 * Visual timeline selector with thumbnail filmstrip and draggable handles.
 * Video scrubs to the handle position as you drag.
 */
export function VideoTimelineSelector({
  videoUrl, duration, startTime, endTime,
  onStartChange, onEndChange, height = 56,
}: Props) {
  const videoRef = useRef<HTMLVideoElement>(null)
  const trackRef = useRef<HTMLDivElement>(null)
  const [thumbnails, setThumbnails] = useState<string[]>([])
  const [dragging, setDragging] = useState<'start' | 'end' | null>(null)
  const [previewTime, setPreviewTime] = useState<number | null>(null)
  const thumbCount = 10

  // Generate thumbnail filmstrip
  useEffect(() => {
    if (!videoUrl || duration <= 0) return
    const canvas = document.createElement('canvas')
    const ctx = canvas.getContext('2d')
    const video = document.createElement('video')
    video.crossOrigin = 'anonymous'
    video.muted = true
    video.src = videoUrl

    const frames: string[] = []
    let idx = 0

    video.onloadeddata = () => {
      canvas.width = 96
      canvas.height = Math.round(96 * (video.videoHeight / video.videoWidth))
      captureNext()
    }

    function captureNext() {
      if (idx >= thumbCount) {
        setThumbnails(frames)
        return
      }
      const t = (idx / (thumbCount - 1)) * duration
      video.currentTime = t
      video.onseeked = () => {
        ctx!.drawImage(video, 0, 0, canvas.width, canvas.height)
        frames.push(canvas.toDataURL('image/jpeg', 0.6))
        idx++
        captureNext()
      }
    }
  }, [videoUrl, duration])

  // Scrub video preview to handle position
  useEffect(() => {
    const v = videoRef.current
    if (v && previewTime !== null && isFinite(previewTime)) {
      v.currentTime = previewTime
    }
  }, [previewTime])

  const getTimeFromX = useCallback((clientX: number) => {
    const track = trackRef.current
    if (!track || duration <= 0) return 0
    const rect = track.getBoundingClientRect()
    const pct = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width))
    return Math.round(pct * duration * 10) / 10
  }, [duration])

  const handlePointerDown = useCallback((handle: 'start' | 'end', e: React.PointerEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setDragging(handle)
    const t = getTimeFromX(e.clientX)
    if (handle === 'start') {
      onStartChange(Math.min(t, endTime - 0.1))
      setPreviewTime(t)
    } else {
      onEndChange(Math.max(t, startTime + 0.1))
      setPreviewTime(t)
    }
  }, [getTimeFromX, startTime, endTime, onStartChange, onEndChange])

  useEffect(() => {
    if (!dragging) return
    const handleMove = (e: PointerEvent) => {
      const t = getTimeFromX(e.clientX)
      if (dragging === 'start') {
        const clamped = Math.max(0, Math.min(t, endTime - 0.1))
        onStartChange(clamped)
        setPreviewTime(clamped)
      } else {
        const clamped = Math.min(duration, Math.max(t, startTime + 0.1))
        onEndChange(clamped)
        setPreviewTime(clamped)
      }
    }
    const handleUp = () => {
      setDragging(null)
      setPreviewTime(null)
    }
    window.addEventListener('pointermove', handleMove)
    window.addEventListener('pointerup', handleUp)
    return () => {
      window.removeEventListener('pointermove', handleMove)
      window.removeEventListener('pointerup', handleUp)
    }
  }, [dragging, duration, startTime, endTime, getTimeFromX, onStartChange, onEndChange])

  const startPct = duration > 0 ? (startTime / duration) * 100 : 0
  const endPct = duration > 0 ? (endTime / duration) * 100 : 100

  return (
    <div className="space-y-2">
      {/* Video preview — scrubs to handle position */}
      <div className="rounded-lg overflow-hidden bg-black aspect-video">
        <video ref={videoRef} src={videoUrl} className="w-full h-full object-contain" muted />
      </div>

      {/* Timeline bar */}
      <div className="text-[9px] text-text-muted flex justify-between">
        <span>{formatTime(startTime)}</span>
        <span className="text-accent-blue">{formatTime(endTime - startTime)} selected</span>
        <span>{formatTime(duration)}</span>
      </div>

      {/* Filmstrip + handles */}
      <div
        ref={trackRef}
        className="relative select-none touch-none"
        style={{ height }}
      >
        {/* Thumbnail filmstrip */}
        <div className="absolute inset-0 flex rounded-md overflow-hidden">
          {thumbnails.length > 0 ? thumbnails.map((src, i) => (
            <img key={i} src={src} className="h-full flex-1 object-cover" draggable={false} />
          )) : (
            <div className="w-full h-full bg-bg-tertiary flex items-center justify-center">
              <span className="text-[9px] text-text-muted">Loading thumbnails...</span>
            </div>
          )}
        </div>

        {/* Dimmed regions outside selection */}
        <div className="absolute inset-y-0 left-0 bg-black/60 rounded-l-md pointer-events-none"
          style={{ width: `${startPct}%` }} />
        <div className="absolute inset-y-0 right-0 bg-black/60 rounded-r-md pointer-events-none"
          style={{ width: `${100 - endPct}%` }} />

        {/* Selection border */}
        <div className="absolute inset-y-0 border-2 border-accent-blue rounded-sm pointer-events-none"
          style={{ left: `${startPct}%`, width: `${endPct - startPct}%` }} />

        {/* Start handle */}
        <div
          className={`absolute inset-y-0 w-4 cursor-col-resize flex items-center justify-center z-10 ${
            dragging === 'start' ? 'bg-accent-blue/40' : 'hover:bg-accent-blue/20'
          }`}
          style={{ left: `calc(${startPct}% - 8px)` }}
          onPointerDown={e => handlePointerDown('start', e)}
        >
          <div className="w-1 h-8 bg-accent-blue rounded-full" />
        </div>

        {/* End handle */}
        <div
          className={`absolute inset-y-0 w-4 cursor-col-resize flex items-center justify-center z-10 ${
            dragging === 'end' ? 'bg-accent-blue/40' : 'hover:bg-accent-blue/20'
          }`}
          style={{ left: `calc(${endPct}% - 8px)` }}
          onPointerDown={e => handlePointerDown('end', e)}
        >
          <div className="w-1 h-8 bg-accent-blue rounded-full" />
        </div>
      </div>
    </div>
  )
}

function formatTime(seconds: number): string {
  const m = Math.floor(seconds / 60)
  const s = Math.floor(seconds % 60)
  const ms = Math.floor((seconds % 1) * 10)
  return m > 0 ? `${m}:${s.toString().padStart(2, '0')}.${ms}` : `${s}.${ms}s`
}
