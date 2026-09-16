import { useState, useRef, useEffect, useCallback } from 'react'

interface Props {
  videoUrl: string
  duration: number
  startTime: number
  endTime: number
  onStartChange: (t: number) => void
  onEndChange: (t: number) => void
  height?: number
  playheadTime?: number
  onPlayheadChange?: (t: number) => void
  onMetadata?: (duration: number) => void
  disabled?: boolean
}

/**
 * Visual timeline selector with thumbnail filmstrip and draggable handles.
 * Video scrubs to the handle position as you drag.
 */
export function VideoTimelineSelector({
  videoUrl, duration, startTime, endTime,
  onStartChange, onEndChange, height = 56,
  playheadTime, onPlayheadChange, onMetadata, disabled = false,
}: Props) {
  const videoRef = useRef<HTMLVideoElement>(null)
  const trackRef = useRef<HTMLDivElement>(null)
  const [filmstrip, setFilmstrip] = useState<{url: string; frames: string[]}>({url: '', frames: []})
  const thumbnails = filmstrip.url === videoUrl ? filmstrip.frames : []
  const [dragging, setDragging] = useState<'start' | 'end' | 'frame' | null>(null)
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
    video.playsInline = true
    video.preload = 'auto'
    video.src = videoUrl
    let cancelled = false

    const frames: string[] = []
    let idx = 0

    video.onloadeddata = () => {
      canvas.width = 96
      canvas.height = Math.round(96 * (video.videoHeight / video.videoWidth))
      captureNext()
    }

    function captureNext() {
      if (idx >= thumbCount) {
        if (!cancelled) setFilmstrip({url: videoUrl, frames})
        return
      }
      const t = (idx / (thumbCount - 1)) * Math.max(0, duration - 1 / 24)
      const capture = () => {
        if (cancelled || !ctx) return
        try {
          ctx.drawImage(video, 0, 0, canvas.width, canvas.height)
          frames.push(canvas.toDataURL('image/jpeg', 0.6))
        } catch { return }
        idx++
        captureNext()
      }
      if (Math.abs(video.currentTime - t) < 0.001) capture()
      else { video.onseeked = capture; video.currentTime = t }
    }
    return () => {
      cancelled = true
      video.onloadeddata = video.onseeked = null
      video.removeAttribute('src')
      video.load()
    }
  }, [videoUrl, duration])

  // Scrub video preview to handle position
  useEffect(() => {
    const v = videoRef.current
    if (v && previewTime !== null && isFinite(previewTime)) {
      v.currentTime = previewTime
    }
  }, [previewTime])

  useEffect(() => {
    const video = videoRef.current
    if (video && video.paused && playheadTime != null && Number.isFinite(playheadTime)
      && Math.abs(video.currentTime - playheadTime) > 0.001) video.currentTime = playheadTime
  }, [playheadTime, videoUrl])

  const getTimeFromX = useCallback((clientX: number) => {
    const track = trackRef.current
    if (!track || duration <= 0) return 0
    const rect = track.getBoundingClientRect()
    if (rect.width <= 0) return 0
    const pct = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width))
    const precision = onPlayheadChange ? 100 : 10
    return Math.round(pct * duration * precision) / precision
  }, [duration, onPlayheadChange])

  const handlePointerDown = useCallback((handle: 'start' | 'end' | 'frame', e: React.PointerEvent) => {
    if (disabled || duration <= 0) return
    e.preventDefault()
    e.stopPropagation()
    videoRef.current?.pause()
    setDragging(handle)
    const t = getTimeFromX(e.clientX)
    if (handle === 'frame') {
      onPlayheadChange?.(Math.max(startTime, Math.min(t, endTime - 1 / 24)))
    } else if (handle === 'start') {
      const clamped = Math.max(0, Math.min(t, endTime - 0.1))
      onStartChange(clamped)
      if (!onPlayheadChange) setPreviewTime(clamped)
    } else {
      const clamped = Math.min(duration, Math.max(t, startTime + 0.1))
      onEndChange(clamped)
      if (!onPlayheadChange) setPreviewTime(clamped)
    }
  }, [getTimeFromX, startTime, endTime, onStartChange, onEndChange, onPlayheadChange, disabled, duration])

  useEffect(() => {
    if (!dragging) return
    const handleMove = (e: PointerEvent) => {
      const t = getTimeFromX(e.clientX)
      if (dragging === 'frame') {
        onPlayheadChange?.(Math.max(startTime, Math.min(t, endTime - 1 / 24)))
      } else if (dragging === 'start') {
        const clamped = Math.max(0, Math.min(t, endTime - 0.1))
        onStartChange(clamped)
        if (!onPlayheadChange) setPreviewTime(clamped)
      } else {
        const clamped = Math.min(duration, Math.max(t, startTime + 0.1))
        onEndChange(clamped)
        if (!onPlayheadChange) setPreviewTime(clamped)
      }
    }
    const handleUp = () => {
      setDragging(null)
      setPreviewTime(null)
    }
    window.addEventListener('pointermove', handleMove)
    window.addEventListener('pointerup', handleUp)
    window.addEventListener('pointercancel', handleUp)
    return () => {
      window.removeEventListener('pointermove', handleMove)
      window.removeEventListener('pointerup', handleUp)
      window.removeEventListener('pointercancel', handleUp)
    }
  }, [dragging, duration, startTime, endTime, getTimeFromX, onStartChange, onEndChange, onPlayheadChange])

  const startPct = duration > 0 ? (startTime / duration) * 100 : 0
  const endPct = duration > 0 ? (endTime / duration) * 100 : 100

  return (
    <div className="space-y-2">
      {/* Video preview — scrubs to handle position */}
      <div className="rounded-lg overflow-hidden bg-black aspect-video">
        <video ref={videoRef} src={videoUrl} className="w-full h-full object-contain" muted playsInline
          controls={!!onPlayheadChange && !disabled} preload="metadata" aria-label="Control video preview"
          onLoadedMetadata={event => {
            onMetadata?.(event.currentTarget.duration)
            if (playheadTime != null) event.currentTarget.currentTime = playheadTime
          }} onTimeUpdate={event => {
            if (!onPlayheadChange || dragging || disabled || duration <= 0) return
            const video = event.currentTarget
            const time = Math.max(startTime, Math.min(video.currentTime, Math.max(startTime, endTime - 1 / 24)))
            if (video.currentTime >= endTime || video.currentTime < startTime) {
              video.pause()
              video.currentTime = time
            }
            if (Math.abs(time - (playheadTime ?? 0)) > 0.001) onPlayheadChange(time)
          }} />
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
        className={`relative select-none touch-none ${onPlayheadChange ? 'mt-4 cursor-crosshair' : ''} ${disabled ? 'pointer-events-none opacity-50' : ''}`}
        aria-label="Video trim timeline"
        onPointerDown={onPlayheadChange ? event => handlePointerDown('frame', event) : undefined}
        style={{ height }}
      >
        {/* Thumbnail filmstrip */}
        <div className="absolute inset-0 flex rounded-md overflow-hidden">
          {thumbnails.length > 0 ? thumbnails.map((src, i) => (
            <img key={i} src={src} alt="" className="h-full min-w-0 flex-1 object-cover" draggable={false} />
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
          role="slider" aria-label="Trim start" aria-valuemin={0} aria-valuemax={Math.max(0, endTime - 0.1)} aria-valuenow={startTime}
          tabIndex={disabled ? -1 : 0}
          onKeyDown={event => {
            if (disabled) return
            const delta = event.key === 'ArrowRight' ? 0.1 : event.key === 'ArrowLeft' ? -0.1 : 0
            if (delta) {event.preventDefault(); onStartChange(Math.max(0, Math.min(endTime - 0.1, startTime + delta)))}
          }}
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
          role="slider" aria-label="Trim end" aria-valuemin={startTime + 0.1} aria-valuemax={duration} aria-valuenow={endTime}
          tabIndex={disabled ? -1 : 0}
          onKeyDown={event => {
            if (disabled) return
            const delta = event.key === 'ArrowRight' ? 0.1 : event.key === 'ArrowLeft' ? -0.1 : 0
            if (delta) {event.preventDefault(); onEndChange(Math.min(duration, Math.max(startTime + 0.1, endTime + delta)))}
          }}
          className={`absolute inset-y-0 w-4 cursor-col-resize flex items-center justify-center z-10 ${
            dragging === 'end' ? 'bg-accent-blue/40' : 'hover:bg-accent-blue/20'
          }`}
          style={{ left: `calc(${endPct}% - 8px)` }}
          onPointerDown={e => handlePointerDown('end', e)}
        >
          <div className="w-1 h-8 bg-accent-blue rounded-full" />
        </div>
        {onPlayheadChange && <div className="pointer-events-none absolute inset-y-0 z-20 w-0.5 bg-white shadow-[0_0_2px_1px_#000]"
          style={{left: `${duration > 0 ? ((playheadTime ?? startTime) / duration) * 100 : 0}%`}}>
          <div role="slider" aria-label="Frame to edit" aria-valuemin={startTime}
            aria-valuemax={Math.max(startTime, endTime - 1 / 24)} aria-valuenow={playheadTime ?? startTime}
            aria-valuetext={formatTime(playheadTime ?? startTime)} tabIndex={disabled ? -1 : 0}
            className="pointer-events-auto absolute -top-3 -left-3 flex h-6 w-6 cursor-col-resize items-start justify-center touch-none"
            onPointerDown={event => handlePointerDown('frame', event)}
            onKeyDown={event => {
              if (disabled) return
              const current = playheadTime ?? startTime
              const next = event.key === 'Home' ? startTime : event.key === 'End' ? endTime - 1 / 24
                : event.key === 'ArrowLeft' ? current - 1 / 24 : event.key === 'ArrowRight' ? current + 1 / 24 : null
              if (next != null) {event.preventDefault(); videoRef.current?.pause(); onPlayheadChange(Math.max(startTime, Math.min(endTime - 1 / 24, next)))}
            }}><span className="h-3 w-3 rounded-sm border border-black/60 bg-white" /></div>
        </div>}
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
