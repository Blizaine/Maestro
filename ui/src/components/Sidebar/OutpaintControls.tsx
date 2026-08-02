import { useState, useRef, useCallback, useEffect, useMemo } from 'react'
import { Upload, X } from 'lucide-react'
import { useStore } from '../../stores/useStore'
import { VideoTimelineSelector } from '../shared/VideoTimelineSelector'
import { OutpaintCanvas } from './OutpaintCanvas'
import * as api from '../../api/client'

/**
 * Outpaint mode controls.
 *
 * Layout when a clip is loaded:
 *   1. Film-strip timeline with start/end trim markers (same component
 *      as inpaint mode for consistency).
 *   2. Interactive canvas composer (OutpaintCanvas) — black canvas at
 *      the chosen aspect ratio with the source clip draggable +
 *      resizable on top. Embeds its own 6-button aspect picker.
 *   3. Output Quality button group (replaces the legacy dropdown).
 *   4. Advanced section (collapsed by default) — mask preservation,
 *      preserve audio, and sliding-window cleanup.
 *
 * The previous version exposed per-side padding sliders + a separate
 * aspect-ratio dropdown that drove those sliders. The interactive
 * canvas replaces both: the user drags / resizes the video inside the
 * canvas and the pads are derived at submit time (see useStore.submit
 * outpaint branch). Backwards-compat: the legacy `outpaintPadding`
 * field in the store is still mirrored on submit so older sidecars
 * keep round-tripping.
 */

const QUALITY_OPTIONS = [
  { v: 'auto', label: 'Auto', hint: 'Native source resolution after padding' },
  { v: '480p', label: '480p', hint: 'Lowest VRAM' },
  { v: '540p', label: '540p', hint: 'Lower VRAM' },
  { v: '720p', label: '720p', hint: '' },
  { v: '1080p', label: '1080p', hint: 'Highest quality' },
] as const

type Quality = (typeof QUALITY_OPTIONS)[number]['v']

export function OutpaintControls() {
  const editVideoFile = useStore(s => s.editVideoFile)
  const editVideoUrl = useStore(s => s.editVideoUrl)
  const editVideoDuration = useStore(s => s.editVideoDuration)
  const editVideoResolution = useStore(s => s.editVideoResolution)
  const setEditVideo = useStore(s => s.setEditVideo)
  const clearEditVideo = useStore(s => s.clearEditVideo)

  // Trim window — outpaint has its own trim state separate from the
  // inpaint flow's editStartTime/editEndTime so the two modes don't
  // step on each other when a user toggles between them.
  const trimStart = useStore(s => s.outpaintTrimStart)
  const trimEnd = useStore(s => s.outpaintTrimEnd)
  const setTrimStart = useStore(s => s.setOutpaintTrimStart)
  const setTrimEnd = useStore(s => s.setOutpaintTrimEnd)

  const outpaintResolutionPreset = useStore(s => s.outpaintResolutionPreset)
  const setOutpaintResolutionPreset = useStore(s => s.setOutpaintResolutionPreset)
  const outpaintMaskPreserving = useStore(s => s.outpaintMaskPreserving)
  const setOutpaintMaskPreserving = useStore(s => s.setOutpaintMaskPreserving)
  const outpaintPreserveSourceAudio = useStore(s => s.outpaintPreserveSourceAudio)
  const setOutpaintPreserveSourceAudio = useStore(s => s.setOutpaintPreserveSourceAudio)
  const outpaintTrimSmear = useStore(s => s.outpaintTrimSmear)
  const setOutpaintTrimSmear = useStore(s => s.setOutpaintTrimSmear)
  const windowSize = useStore(s => s.slidingWindowSeconds)
  const setWindowSize = useStore(s => s.setSlidingWindowSeconds)
  const windowLocked = useStore(s => s.slidingWindowLocked)

  const [error, setError] = useState<string | null>(null)
  const [showAdvanced, setShowAdvanced] = useState(false)
  const fileRef = useRef<HTMLInputElement>(null)

  // Source pixel dimensions parsed from editVideoResolution (e.g. "1280x720").
  // OutpaintCanvas needs these to:
  //   - render the canvas in canvas-pixel-space (so output dims match aspect)
  //   - letterbox-fit the source video into the canvas on aspect change
  // Falls back to {0,0} when no resolution string is set yet (placeholder
  // dimensions OutpaintCanvas treats as "not ready" and renders nothing).
  const { srcW, srcH } = useMemo(() => {
    if (!editVideoResolution) return { srcW: 0, srcH: 0 }
    const m = /^(\d+)x(\d+)$/.exec(editVideoResolution)
    if (!m) return { srcW: 0, srcH: 0 }
    return { srcW: parseInt(m[1], 10), srcH: parseInt(m[2], 10) }
  }, [editVideoResolution])

  const handleUpload = useCallback(async (file: File) => {
    setError(null)
    try {
      const result = await api.uploadImage(file)
      const url = URL.createObjectURL(file)
      const isVideo = file.type.startsWith('video/')
      if (isVideo) {
        const video = document.createElement('video')
        video.src = url
        video.onloadedmetadata = () => {
          const duration = video.duration && isFinite(video.duration) ? video.duration : 0
          const resolution = `${video.videoWidth}x${video.videoHeight}`
          setEditVideo(file, result.path, url, duration, resolution)
          // Reset trim window to the full clip so the timeline starts
          // wide-open. User can narrow with the markers if they want a
          // partial-window outpaint.
          setTrimStart(0)
          setTrimEnd(duration)
        }
      } else {
        const img = new window.Image()
        img.src = url
        img.onload = () => {
          setEditVideo(file, result.path, url, 0, `${img.naturalWidth}x${img.naturalHeight}`)
        }
      }
    } catch {
      setError('Failed to upload')
    }
  }, [setEditVideo, setTrimStart, setTrimEnd])

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    const file = e.dataTransfer.files[0]
    if (file && (file.type.startsWith('video/') || file.type.startsWith('image/'))) handleUpload(file)
  }, [handleUpload])

  const isVideoFile = editVideoFile?.type.startsWith('video/')
  const selectedDuration = trimEnd > trimStart
    ? trimEnd - trimStart
    : editVideoDuration

  // Match Studio Frames mode: an unlocked window follows a short clip with
  // a one-second quantization buffer, up to the model's ~20-second limit.
  // Outpaint has a trim timeline instead of DurationSlider, so it needs the
  // same tracking behavior here.
  useEffect(() => {
    if (windowLocked || !isVideoFile || selectedDuration <= 0) return
    if (selectedDuration <= 20) {
      const nextWindow = Math.min(21, Math.ceil(selectedDuration) + 1)
      if (windowSize !== nextWindow) setWindowSize(nextWindow)
    } else if (windowSize < 10) {
      setWindowSize(20)
    }
  }, [isVideoFile, selectedDuration, windowLocked]) // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div className="space-y-3">
      {/* Upload zone — shown only when no clip loaded. Same dashed-border
          + drop-target treatment as the other Edit-mode flows. */}
      {!editVideoFile ? (
        <div
          onDragOver={e => e.preventDefault()}
          onDrop={handleDrop}
          onClick={() => fileRef.current?.click()}
          className="border-2 border-dashed border-border rounded-lg p-6 text-center cursor-pointer hover:border-accent-blue transition-colors"
        >
          <Upload size={24} className="mx-auto mb-2 text-text-muted" />
          <p className="text-xs text-text-secondary">Drop a video or image to outpaint</p>
          <p className="text-[10px] text-text-muted mt-1">or click to browse</p>
          <input
            ref={fileRef}
            type="file"
            accept="video/*,image/*"
            className="hidden"
            onChange={e => { const f = e.target.files?.[0]; if (f) handleUpload(f) }}
          />
        </div>
      ) : (
        <>
          {/* Header: filename + remove button. The X mirrors the inpaint
              flow — clears the loaded clip and resets the canvas state
              implicitly (OutpaintCanvas will re-fit on next mount when
              srcW/srcH return to 0). */}
          <div className="flex items-center justify-between gap-2">
            <p className="text-[10px] text-text-muted truncate flex-1">{editVideoFile.name}</p>
            <button
              onClick={() => clearEditVideo()}
              className="shrink-0 p-1 rounded text-text-muted hover:bg-bg-hover hover:text-red-400 transition-colors"
              title="Remove clip"
            >
              <X size={14} />
            </button>
          </div>

          {/* Film-strip timeline with start/end trim markers — only meaningful
              for actual videos. Images skip this since there's no time axis. */}
          {isVideoFile && editVideoDuration > 0 && (
            <VideoTimelineSelector
              videoUrl={editVideoUrl}
              duration={editVideoDuration}
              startTime={trimStart}
              endTime={trimEnd}
              onStartChange={setTrimStart}
              onEndChange={setTrimEnd}
            />
          )}

          {/* Interactive composer — embeds its own 6-button aspect picker
              (16:9 / 9:16 / 1:1 / 4:3 / 3:4 / source). Drives outpaintAspect
              and outpaintVideoBox in the store; useStore.submit reads both
              at submit time and derives pad_top/bottom/left/right pixels. */}
          {srcW > 0 && srcH > 0 && editVideoUrl && (
            <OutpaintCanvas
              videoUrl={editVideoUrl}
              srcW={srcW}
              srcH={srcH}
            />
          )}
        </>
      )}

      {error && <p className="text-[10px] text-red-400">{error}</p>}

      {/* Output Quality button group (replaces the legacy dropdown).
          Always shown — the chosen quality applies to whatever clip is
          loaded next, so it's useful to set even before upload. */}
      <div>
        <label className="text-[11px] text-text-muted uppercase tracking-wider mb-1.5 block">
          Output Quality
        </label>
        <div className="grid grid-cols-5 gap-1">
          {QUALITY_OPTIONS.map(q => (
            <button
              key={q.v}
              onClick={() => setOutpaintResolutionPreset(q.v as Quality)}
              title={q.hint}
              className={`px-1 py-1.5 rounded text-[10px] border transition-colors ${
                outpaintResolutionPreset === q.v
                  ? 'border-accent-blue bg-accent-blue/15 text-accent-blue'
                  : 'border-border text-text-muted hover:border-border-light hover:text-text-secondary'
              }`}
            >
              {q.label}
            </button>
          ))}
        </div>
        <p className="text-[10px] text-text-muted mt-1">
          Auto keeps full source resolution after padding. Pick a preset to scale down for lower-VRAM models.
        </p>
      </div>

      {/* Advanced remains collapsed so the default workflow stays simple. */}
      <button
        onClick={() => setShowAdvanced(!showAdvanced)}
        className="text-[10px] text-text-muted hover:text-text-primary transition-colors"
      >
        {showAdvanced ? '▾' : '▸'} Advanced
      </button>
      {showAdvanced && (
        <div className="space-y-3 pl-2 border-l border-border/50">
          <label
            className="flex items-start gap-2 cursor-pointer"
            title="Uses LTX-2.3's binary-mask conditioning and a soft multiscale boundary blend. Disable only to compare with Maestro's legacy Outpaint path."
          >
            <input
              type="checkbox"
              checked={outpaintMaskPreserving}
              onChange={e => setOutpaintMaskPreserving(e.target.checked)}
              className="w-3 h-3 mt-0.5 rounded border-border accent-accent-blue shrink-0"
            />
            <div className="flex-1">
              <span className="text-[10px] text-text-secondary">
                Preserve original scene
              </span>
              <span className="ml-1 text-[9px] text-accent-blue">
                Recommended
              </span>
              <p className="text-[9px] text-text-muted mt-0.5">
                Protects the source and softly blends the new area.
              </p>
            </div>
          </label>

          <label className="flex items-start gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={outpaintPreserveSourceAudio}
              onChange={e => setOutpaintPreserveSourceAudio(e.target.checked)}
              className="w-3 h-3 mt-0.5 rounded border-border accent-accent-blue shrink-0"
            />
            <div className="flex-1">
              <span className="text-[10px] text-text-secondary">Preserve source audio</span>
              <p className="text-[9px] text-text-muted mt-0.5">
                Re-mux the original soundtrack into the outpainted clip. The model would otherwise generate fresh audio that doesn't match the source.
              </p>
            </div>
          </label>

          <label className="flex items-start gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={outpaintTrimSmear}
              onChange={e => setOutpaintTrimSmear(e.target.checked)}
              className="w-3 h-3 mt-0.5 rounded border-border accent-accent-blue shrink-0"
            />
            <div className="flex-1">
              <span className="text-[10px] text-text-secondary">Trim window smear</span>
              <p className="text-[9px] text-text-muted mt-0.5">
                Trim the last few frames of each sliding-window segment where the model occasionally smears the boundary.
              </p>
            </div>
          </label>
        </div>
      )}
    </div>
  )
}
