import { useState, useRef, useCallback, useEffect } from 'react'
import { Upload, X, Eye, AlertTriangle, Download } from 'lucide-react'
import { useStore } from '../../stores/useStore'
import { VideoTimelineSelector } from '../shared/VideoTimelineSelector'
import * as api from '../../api/client'

export function InpaintControls() {
  const editVideoFile = useStore(s => s.editVideoFile)
  const editVideoPath = useStore(s => s.editVideoPath)
  const editVideoUrl = useStore(s => s.editVideoUrl)
  const editVideoDuration = useStore(s => s.editVideoDuration)
  const editStartTime = useStore(s => s.editStartTime)
  const editEndTime = useStore(s => s.editEndTime)
  const editMasksPath = useStore(s => s.editMasksPath)
  const editMaskPreview = useStore(s => s.editMaskPreview)
  const editDetectedTarget = useStore(s => s.editDetectedTarget)
  const setEditVideo = useStore(s => s.setEditVideo)
  const clearEditVideo = useStore(s => s.clearEditVideo)

  const samTarget = useStore(s => s.editSamTarget)
  const setSamTarget = (v: string) => useStore.setState({ editSamTarget: v })
  const retakeStrength = useStore(s => s.editRetakeStrength)
  const setRetakeStrength = (v: number) => useStore.setState({ editRetakeStrength: v })
  const promptStrength = useStore(s => s.editPromptStrength)
  const setPromptStrength = (v: number) => useStore.setState({ editPromptStrength: v })
  const [previewing, setPreviewing] = useState(false)
  const [error, setError] = useState<string | null>(null)
  // Richer SAM status state — 'not_installed' is distinct from generic
  // 'unavailable' because the user-action is different (run the
  // Pinokio menu install vs check service health). 'samStatusError'
  // holds the friendly message returned by the backend when applicable.
  const [samStatus, setSamStatus] = useState<'unknown' | 'available' | 'not_installed' | 'unavailable'>('unknown')
  const [samStatusError, setSamStatusError] = useState<string | null>(null)
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [showingMask, setShowingMask] = useState(false)
  const fileRef = useRef<HTMLInputElement>(null)

  // Reusable status check — used both on mount AND after upload so the
  // user gets the install prompt immediately when they switch to
  // Inpaint mode, not only after they upload a video.
  const refreshSamStatus = useCallback(async () => {
    try {
      const status = await api.samServiceStatus()
      const s = status.status
      if (s === 'available' || s === 'ready' || s === 'installed') {
        setSamStatus('available')
        setSamStatusError(null)
      } else if (s === 'not_installed') {
        setSamStatus('not_installed')
        setSamStatusError(status.error || null)
      } else {
        setSamStatus('unavailable')
        setSamStatusError(status.error || null)
      }
    } catch {
      setSamStatus('unavailable')
      setSamStatusError(null)
    }
  }, [])

  // Check on mount so the "needs install" banner appears the moment
  // the user enters Inpaint mode — they shouldn't have to upload a
  // video first to discover that SAM isn't installed.
  useEffect(() => {
    refreshSamStatus()
  }, [refreshSamStatus])

  const handleUpload = useCallback(async (file: File) => {
    setError(null)
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
      // Re-check after upload in case the user just installed SAM in
      // another tab while this Inpaint session was open.
      refreshSamStatus()
    } catch {
      setError('Failed to upload video')
    }
  }, [setEditVideo, refreshSamStatus])

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    const file = e.dataTransfer.files[0]
    if (file && file.type.startsWith('video/')) handleUpload(file)
  }, [handleUpload])

  // Preview Mask: single-frame only (fast, for visual feedback)
  const handlePreviewMask = async () => {
    if (!editVideoPath || !samTarget.trim()) return
    setPreviewing(true)
    setError(null)
    try {
      const result = await api.segmentPreview({
        video_path: editVideoPath,
        text: samTarget.trim(),
        start_time: editStartTime,
        end_time: editEndTime,
        full_video: false,
        invert_mask: useStore.getState().editInvertMask,
      })
      useStore.setState({
        editMaskPreview: result.mask_preview,
        editDetectedTarget: result.target || samTarget.trim(),
      })
      setShowingMask(true)
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setPreviewing(false)
    }
  }

  // Full video segmentation happens on Generate (via store's startGeneration)
  // Store the SAM target so startGeneration can use it
  // We pass samTarget via a store field

  return (
    <div className="space-y-3">
      {/* SAM not installed — prominent banner shown the moment the user
          enters Inpaint mode (not gated on upload). Tells them exactly
          which Pinokio menu item to click. Inpaint won't work at all
          without SAM so this needs to be unmissable. */}
      {samStatus === 'not_installed' && (
        <div className="flex items-start gap-2 text-[11px] text-amber-300 bg-amber-500/10 border border-amber-500/40 rounded-lg px-3 py-2.5">
          <Download size={14} className="shrink-0 mt-0.5" />
          <div className="space-y-1">
            <p className="font-medium text-amber-200">Inpaint requires SAM 3.1 (not installed)</p>
            <p className="text-amber-300/80">
              {samStatusError || "Open the Pinokio menu and click 'Install Inpaint Support (SAM 3.1)' to enable inpainting. ~5 min one-time setup."}
            </p>
          </div>
        </div>
      )}
      {/* SAM installed but service unhealthy — only show after the
          user uploaded a video and we tried to use it. Less alarming
          than the not-installed case because the on-demand startup
          might still recover. */}
      {samStatus === 'unavailable' && editVideoPath && !editMasksPath && (
        <div className="flex items-start gap-2 text-[10px] text-amber-400 bg-amber-500/10 border border-amber-500/20 rounded-lg px-3 py-2">
          <AlertTriangle size={14} className="shrink-0 mt-0.5" />
          <div>
            <p className="font-medium">SAM service not available</p>
            <p className="text-amber-400/70 mt-0.5">Check SAM installation or run Preview Mask to start it on demand.</p>
          </div>
        </div>
      )}

      {/* Video Upload */}
      {!editVideoFile ? (
        <div
          onDragOver={e => e.preventDefault()}
          onDrop={handleDrop}
          onClick={() => fileRef.current?.click()}
          className="border-2 border-dashed border-border rounded-lg p-6 text-center cursor-pointer hover:border-accent-blue/50 hover:bg-bg-hover/30 transition-all"
        >
          <Upload size={24} className="mx-auto mb-2 text-text-muted" />
          <p className="text-xs text-text-secondary">Drop a video or click to upload</p>
          <p className="text-[9px] text-text-muted mt-1">Describe what to change — AI segments and replaces</p>
          <input ref={fileRef} type="file" accept="video/*" className="hidden"
            onChange={e => { if (e.target.files?.[0]) handleUpload(e.target.files[0]) }} />
        </div>
      ) : (
        <div className="relative">
          <button onClick={() => { clearEditVideo(); setShowingMask(false); setSamTarget('') }}
            className="absolute top-1.5 right-1.5 z-20 p-1 rounded-full bg-black/60 text-white/80 hover:text-white hover:bg-black/80 transition-colors">
            <X size={14} />
          </button>

          {/* Mask preview overlay or timeline */}
          {showingMask && editMaskPreview ? (
            <div>
              <div className="rounded-lg overflow-hidden bg-black aspect-video relative">
                <img src={`data:image/png;base64,${editMaskPreview}`} alt="Mask preview"
                  className="w-full h-full object-contain" />
                <button onClick={() => setShowingMask(false)}
                  className="absolute bottom-2 left-2 px-2 py-1 rounded bg-black/70 text-[10px] text-white/80 hover:text-white">
                  Show timeline
                </button>
              </div>
              {editDetectedTarget && (
                <p className="text-[9px] text-accent-blue mt-1">SAM target: {editDetectedTarget}</p>
              )}
            </div>
          ) : (
            <div>
              <VideoTimelineSelector
                videoUrl={editVideoUrl}
                duration={editVideoDuration}
                startTime={editStartTime}
                endTime={editEndTime}
                onStartChange={t => useStore.setState({ editStartTime: t })}
                onEndChange={t => useStore.setState({ editEndTime: t })}
              />
              {editMaskPreview && (
                <button onClick={() => setShowingMask(true)}
                  className="text-[9px] text-accent-blue hover:text-accent-blue/80 mt-1">
                  Show mask preview
                </button>
              )}
            </div>
          )}
          <p className="text-[9px] text-text-muted mt-1 truncate">{editVideoFile.name}</p>
        </div>
      )}

      {/* SAM Target — what to segment */}
      <div>
        <label className="text-[10px] text-text-muted uppercase tracking-wider mb-1 block">
          What to select <span className="normal-case text-text-muted/50">(for SAM segmentation)</span>
        </label>
        <input
          type="text"
          value={samTarget}
          onChange={e => setSamTarget(e.target.value)}
          placeholder='e.g. "the woman", "his hands", "the sky"'
          className="w-full bg-bg-tertiary border border-border rounded px-2.5 py-1.5 text-xs text-text-primary placeholder:text-text-muted/40 focus:outline-none focus:border-accent-blue"
        />
        <label className="flex items-center gap-2 mt-1.5 cursor-pointer">
          <input type="checkbox"
            checked={useStore.getState().editInvertMask}
            onChange={e => useStore.setState({ editInvertMask: e.target.checked, editMasksPath: null, editMaskPreview: null })}
            className="w-3 h-3 rounded border-border accent-accent-blue" />
          <span className="text-[10px] text-text-secondary">Invert mask</span>
          <span className="text-[9px] text-text-muted/50 ml-auto">Edit everything except selection</span>
        </label>
      </div>

      {/* Preview Mask + cached mask indicator */}
      <div className="flex items-center gap-2">
        <button onClick={handlePreviewMask}
          disabled={previewing || !editVideoPath || !samTarget.trim()}
          className="flex-1 py-1.5 rounded-lg text-[10px] font-medium border border-border text-text-secondary hover:text-text-primary hover:bg-bg-hover transition-colors disabled:opacity-40 disabled:cursor-not-allowed flex items-center justify-center gap-1.5">
          <Eye size={12} />
          {previewing ? 'Segmenting...' : 'Preview Mask'}
        </button>
        {editMasksPath && (
          <div className="flex items-center gap-1 text-[9px] text-green-400 shrink-0">
            <div className="w-1.5 h-1.5 rounded-full bg-green-400" />
            Cached
          </div>
        )}
      </div>

      {/* Advanced */}
      <button onClick={() => setShowAdvanced(!showAdvanced)}
        className="text-[10px] text-text-muted hover:text-text-primary transition-colors">
        {showAdvanced ? '▾' : '▸'} Advanced
      </button>
      {showAdvanced && (
        <div className="space-y-3 pl-2 border-l border-border/50">
          <div>
            <div className="flex items-center justify-between mb-1">
              <label className="text-[10px] text-text-muted uppercase tracking-wider">Prompt Strength</label>
              <span className="text-[10px] text-text-secondary">{promptStrength.toFixed(1)}</span>
            </div>
            <input
              type="range"
              min={1.0} max={7.0} step={0.5}
              value={promptStrength}
              onChange={e => setPromptStrength(parseFloat(e.target.value))}
              className="w-full"
            />
            <p className="text-[9px] text-text-muted/60 mt-0.5">
              CFG — how hard the model follows your prompt inside the masked region.
              1.0 ≈ prompt ignored (output looks like original). 3–4 ≈ balanced. 5+ ≈ strong, may distort.
            </p>
          </div>
          <div>
            <div className="flex items-center justify-between mb-1">
              <label className="text-[10px] text-text-muted uppercase tracking-wider">Retake Strength</label>
              <span className="text-[10px] text-text-secondary">{retakeStrength.toFixed(2)}</span>
            </div>
            <input
              type="range"
              min={0.1} max={1.0} step={0.05}
              value={retakeStrength}
              onChange={e => setRetakeStrength(parseFloat(e.target.value))}
              className="w-full"
            />
            <p className="text-[9px] text-text-muted/60 mt-0.5">
              How aggressively the masked region is re-generated. 0.1–0.4 = subtle tweak, keeps source look.
              0.7–0.9 = full replacement with prompt content.
            </p>
          </div>
        </div>
      )}

      {/* Status */}
      {error && <div className="text-[10px] text-red-400 bg-red-500/10 border border-red-500/20 rounded px-2 py-1.5">{error}</div>}

      <p className="text-[9px] text-text-muted/60 text-center">
        Preview Mask checks targeting. Use the prompt below for what to generate, then click Generate.
      </p>
    </div>
  )
}
