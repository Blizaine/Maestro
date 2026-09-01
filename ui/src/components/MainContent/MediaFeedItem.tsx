import { useState, useRef, useEffect, useCallback, type CSSProperties } from 'react'
import { Play, Pencil, RefreshCw, Copy, Trash2, Check, Combine, Loader2, Heart, ArrowLeftToLine, Download, FolderInput, Scissors, FastForward, BookMarked, Info, ChevronDown, ChevronUp, MoreHorizontal } from 'lucide-react'
import { SaveRecipeDialog } from '../Recipes/SaveRecipeDialog'
import { useStore } from '../../stores/useStore'
import { getUploadUrl, fetchOutputMetadata, getFileUrl, moveOutput, uploadImage } from '../../api/client'
import type { OutputFile, OutputMetadata } from '../../types'
import { formatGenerationDuration } from '../../lib/format'
import { formatDuration } from '../../lib/durationPlanning'
import { modelDisplayName } from '../../lib/modelDisplay'

interface Props {
  file: OutputFile
  index: number
  isActive: boolean
  onActivate: (index: number) => void
  onPlaybackStart: (index: number, media: HTMLMediaElement) => void
  onMeasured: (index: number, height: number) => void
  style?: CSSProperties
}

/** Image component that retries loading if the file isn't fully written yet.
 *
 * Backstops the backend's atomic image-write guarantee in two ways:
 *   1. onError — fires when the request fails outright (404 during the
 *      tiny window between job-complete signal and file existence).
 *   2. onLoad with naturalWidth === 0 — fires when the backend returned
 *      bytes the browser couldn't decode (truncated/corrupt body that
 *      still produced a 200 OK with matching Content-Length). The
 *      browser silently shows an empty box in this case; without the
 *      check the user sees a half-image and feels they need to refresh
 *      the page (which loses Studio prompts/settings/reference images).
 */
function RetryImage({ url, alt }: { url: string; alt: string }) {
  const [src, setSrc] = useState(url)
  const retries = useRef(0)
  const maxRetries = 5

  const scheduleRetry = useCallback(() => {
    if (retries.current < maxRetries) {
      retries.current++
      setTimeout(() => {
        setSrc(`${url}${url.includes('?') ? '&' : '?'}t=${Date.now()}`)
      }, 800 * retries.current)
    }
  }, [url])

  const handleError = useCallback(() => {
    scheduleRetry()
  }, [scheduleRetry])

  const handleLoad = useCallback((e: React.SyntheticEvent<HTMLImageElement>) => {
    // Truncated body that decoded to nothing — browser fired onLoad
    // (Content-Length matched) but produced a 0×0 image. Treat as
    // failure and retry with a cache-busted URL.
    const img = e.currentTarget
    if (img.naturalWidth === 0 || img.naturalHeight === 0) {
      scheduleRetry()
    }
  }, [scheduleRetry])

  return (
    <img
      key={src}
      src={src}
      alt={alt}
      className="w-full h-full object-contain"
      onError={handleError}
      onLoad={handleLoad}
    />
  )
}

export function MediaFeedItem({ file, index, isActive, onActivate, onPlaybackStart, onMeasured, style }: Props) {
  const setSelectedOutput = useStore(s => s.setSelectedOutput)
  const loadSettingsFromOutput = useStore(s => s.loadSettingsFromOutput)
  const rerollGeneration = useStore(s => s.rerollGeneration)
  const deleteOutput = useStore(s => s.deleteSelectedOutput)
  const rejoinClipGroup = useStore(s => s.rejoinClipGroup)
  const toggleFavorite = useStore(s => s.toggleFavorite)
  const setStartImage = useStore(s => s.setStartImage)
  const addImageRef = useStore(s => s.addImageRef)
  const setContinueVideo = useStore(s => s.setContinueVideo)
  const setParam = useStore(s => s.setParam)
  const openRetakeDialog = useStore(s => s.openRetakeDialog)
  const generationMode = useStore(s => s.generationMode)
  const workspaces = useStore(s => s.workspaces)
  const activeWorkspace = useStore(s => s.activeWorkspace)
  // Virtual Uploads view: browse-only. Move/favorite/delete resolve
  // against the active OUTPUT workspace server-side, so they can't act
  // on upload files — hide them. Download + send-to-input still work
  // (serve_file falls back to the uploads folder).
  const browsingUploads = useStore(s => s.browsingUploads)
  // Used to translate the raw model_type slug (e.g.
  // "ltx2_22B_distilled_1_1") in the per-clip metadata bar into the
  // human-readable display name (e.g. "LTX-2.3 Distilled 1.1 22B")
  // via modelDisplayName().
  const models = useStore(s => s.models)

  const saveRecipeFromOutput = useStore(s => s.saveRecipeFromOutput)
  const nsfwMode = useStore(s => !!s.servicesConfig?.nsfw_mode)

  const [meta, setMeta] = useState<OutputMetadata | null>(null)
  const [metaLoaded, setMetaLoaded] = useState(false)
  const [confirmDelete, setConfirmDelete] = useState(false)
  const [showSaveRecipe, setShowSaveRecipe] = useState(false)
  const confirmRef = useRef(false)
  const timeoutRef = useRef<ReturnType<typeof setTimeout>>(undefined)
  const [copied, setCopied] = useState(false)
  const [rejoining, setRejoining] = useState(false)
  const [sentToInput, setSentToInput] = useState(false)
  const [showActionMenu, setShowActionMenu] = useState(false)
  const [actionMenuOpensDown, setActionMenuOpensDown] = useState(false)
  const [showMoveMenu, setShowMoveMenu] = useState(false)
  const [showDetails, setShowDetails] = useState(false)
  const [moving, setMoving] = useState(false)
  const actionMenuRef = useRef<HTMLDivElement>(null)
  const itemRef = useRef<HTMLDivElement>(null)
  const videoRef = useRef<HTMLVideoElement>(null)

  // Measure actual height and report to parent
  useEffect(() => {
    const el = itemRef.current
    if (!el) return
    const ro = new ResizeObserver((entries) => {
      const height = entries[0].borderBoxSize?.[0]?.blockSize ?? entries[0].contentRect.height
      onMeasured(index, height)
    })
    ro.observe(el)
    return () => ro.disconnect()
  }, [index, onMeasured])

  // Lazy load metadata when first visible
  useEffect(() => {
    if (metaLoaded) return
    const el = itemRef.current
    if (!el) return
    const observer = new IntersectionObserver(
      (entries) => {
        if (entries[0].isIntersecting) {
          setMetaLoaded(true)
          fetchOutputMetadata(file.name).then(setMeta).catch(() => {})
        }
      },
      { threshold: 0.1 }
    )
    observer.observe(el)
    return () => observer.disconnect()
  }, [file.name, metaLoaded])

  // Pause video when scrolled out of view (but don't auto-play when scrolled in)
  useEffect(() => {
    if (!videoRef.current) return
    if (!isActive) {
      videoRef.current.pause()
    }
  }, [isActive])

  const params = meta?.params as Record<string, unknown> | null
  const uploadFilenames = meta?.upload_filenames as Record<string, string> | undefined

  const prompt = (params?._tts_original_prompt as string) || (params?.prompt as string) || ''
  const originalPrompt = (params?._h3_original_prompt as string) || ''
  const h3WindowPlan = (
    params?.h3_window_plan && typeof params.h3_window_plan === 'object'
      ? params.h3_window_plan as Record<string, unknown>
      : null
  )
  const ltxWindowPlan = (
    params?.ltx_window_plan && typeof params.ltx_window_plan === 'object'
      ? params.ltx_window_plan as Record<string, unknown>
      : null
  )
  const effectiveWindowPrompts = (() => {
    for (const direct of [params?.h3_window_prompts, params?.ltx_window_prompts]) {
      if (Array.isArray(direct)) {
        const prompts = direct.map(value => String(value || '').trim()).filter(Boolean)
        if (prompts.length > 0) return prompts
      }
    }
    for (const planned of [h3WindowPlan?.window_prompts, ltxWindowPlan?.window_prompts]) {
      if (Array.isArray(planned)) {
        const prompts = planned.map(value => String(value || '').trim()).filter(Boolean)
        if (prompts.length > 0) return prompts
      }
    }
    return []
  })()
  const effectivePromptPlan = h3WindowPlan || ltxWindowPlan
  const effectivePromptPlannedBy = String(
    effectivePromptPlan?.planned_by || '',
  ).trim().toLowerCase()
  const effectivePromptMode = String(
    params?.minimax_h3_sequence_prompt_mode
      || params?.ltx_window_prompt_mode
      || '',
  ).trim().toLowerCase()
  const generatedWindowPrompts = (
    effectiveWindowPrompts.length > 0
    && effectivePromptPlannedBy !== 'manual'
    && effectivePromptMode !== 'manual'
  )
  const cardPrompt = effectiveWindowPrompts[0] || prompt
  const windowPromptsHeading = generatedWindowPrompts
    ? 'Generated window prompts'
    : 'Effective window prompts'
  const h3PlanningWarnings = Array.isArray(h3WindowPlan?.planning_warnings)
    ? h3WindowPlan.planning_warnings.map(value => String(value || '').trim()).filter(Boolean)
    : []
  const h3PlanningDiagnostics = Array.isArray(h3WindowPlan?.planning_diagnostics)
    ? h3WindowPlan.planning_diagnostics.map(value => String(value || '').trim()).filter(Boolean)
    : []
  const h3PlanningNotes = Array.isArray(h3WindowPlan?.planning_notes)
    ? h3WindowPlan.planning_notes.map(value => String(value || '').trim()).filter(Boolean)
    : []
  const modelType = (params?.model_type as string) || ''
  const modelLabel = modelDisplayName(modelType, models)
  const isAudio = file.type === 'audio'
  const resolution = isAudio ? '' : ((params?.resolution as string) || '')
  const seed = params?.seed as number | undefined
  const generationTime = meta?.generation_time
  const inferenceSteps = params?.num_inference_steps as number | undefined
  const guidanceScale = params?.guidance_scale as number | undefined
  const activeLoras = (() => {
    const value = params?.activated_loras
    if (Array.isArray(value)) return value.map(item => String(item)).filter(Boolean)
    return typeof value === 'string' && value ? [value] : []
  })()
  const loraWeights = (() => {
    const value = params?.loras_multipliers
    if (Array.isArray(value)) return value.map(item => String(item))
    if (typeof value === 'string' && value) return value.split(/[;,]/).map(item => item.trim())
    return []
  })()
  const turboEnabled = params?.minimax_h3_turbo_mode === true
  const turboPreset = String(params?.minimax_h3_turbo_preset || '')
  const pddEnabled = turboEnabled && (
    turboPreset.toLowerCase().includes('pdd')
    || activeLoras.some(item => item.toLowerCase().includes('pdd') || item.toLowerCase().includes('-acc-'))
  )
  const firstBlockEnabled = params?.skip_steps_cache_type === 'first_block'
  const solEnabled = String(params?.override_attention || '').toLowerCase() === 'sol'
  const h3Workflow = modelType.includes('minimax_h3')
    ? (modelType.includes('ref2va') ? 'Omni / Ref2VA' : 'First / Last / FL2VA')
    : ''
  const optimizationLabels = [
    ...(turboEnabled ? ['Turbo'] : []),
    ...(pddEnabled ? ['PDD'] : []),
    ...(solEnabled ? ['Sol Engine'] : []),
    ...(firstBlockEnabled ? ['First Block Cache'] : []),
  ]

  const timedWindowSeconds = Array.isArray(meta?.multi_window_timing?.window_generation_seconds)
    ? meta.multi_window_timing.window_generation_seconds
      .map(value => Number(value))
      .filter(value => Number.isFinite(value) && value >= 0)
    : []
  const numberValue = (value: unknown) => {
    const parsed = Number(value)
    return Number.isFinite(parsed) && parsed > 0 ? parsed : 0
  }
  const windowCount = Math.max(
    1,
    Math.round(numberValue(meta?.multi_window_timing?.window_count)),
    Math.round(numberValue(h3WindowPlan?.window_count)),
    Math.round(numberValue(ltxWindowPlan?.window_count)),
    effectiveWindowPrompts.length,
  )
  const isMultiWindow = windowCount > 1
  const explicitSceneDuration = numberValue(meta?.multi_window_timing?.scene_duration_seconds)
    || numberValue(params?.duration_seconds)
    || numberValue(params?.audio_duration_seconds)
  const frameCount = numberValue(params?.video_length)
  const explicitFps = numberValue(params?.fps)
  const inferredFps = explicitFps || (
    modelType.includes('ltx') ? 25 : modelType.includes('minimax_h3') ? 24 : 0
  )
  const frameSceneDuration = (
    frameCount > 0 && inferredFps > 0 ? frameCount / inferredFps : 0
  )
  const sceneDurationSeconds = (
    file.type === 'video' ? frameSceneDuration : explicitSceneDuration
  ) || explicitSceneDuration || frameSceneDuration
  const totalWindowGenerationSeconds = numberValue(
    meta?.multi_window_timing?.total_generation_seconds,
  ) || (isMultiWindow ? numberValue(generationTime) : 0)
  const completedWindowCount = Math.min(
    windowCount,
    Math.max(
      timedWindowSeconds.length,
      Math.round(numberValue(meta?.multi_window_timing?.completed_windows)),
    ),
  )

  const multiClipInfo = params?.multi_clip_info as { group_id: string; index: number; total: number } | undefined
  const groupId = multiClipInfo?.group_id
  const clipIndex = multiClipInfo?.index
  const clipTotal = multiClipInfo?.total

  const rawStart = uploadFilenames?.image_start
  const rawEnd = uploadFilenames?.image_end
  const imageStartFile = Array.isArray(rawStart) ? (rawStart.find((f: string) => f) || null) : rawStart
  const imageEndFile = Array.isArray(rawEnd) ? (rawEnd.find((f: string) => f) || null) : rawEnd

  const handleSelect = useCallback(() => {
    onActivate(index)
  }, [index, onActivate])

  const handlePlaybackStart = useCallback((event: React.SyntheticEvent<HTMLMediaElement>) => {
    // A direct Play action is the strongest possible selection signal. Unmute
    // immediately (before React's active-card rerender) so native controls and
    // fullscreen playback both begin with sound.
    event.currentTarget.muted = false
    onPlaybackStart(index, event.currentTarget)
  }, [index, onPlaybackStart])

  const handleLoadSettings = useCallback(() => {
    setSelectedOutput(index)
    setTimeout(() => loadSettingsFromOutput(), 50)
  }, [index, setSelectedOutput, loadSettingsFromOutput])

  const handleReroll = useCallback(() => {
    setSelectedOutput(index)
    setTimeout(() => rerollGeneration(), 50)
  }, [index, setSelectedOutput, rerollGeneration])

  const handleCopyPrompt = () => {
    if (!prompt) return
    // navigator.clipboard requires secure context; fallback to execCommand
    if (navigator.clipboard?.writeText) {
      navigator.clipboard.writeText(prompt).then(() => {
        setCopied(true)
        setTimeout(() => setCopied(false), 1500)
      }).catch(() => {
        // Fallback
        const ta = document.createElement('textarea')
        ta.value = prompt
        ta.style.position = 'fixed'
        ta.style.opacity = '0'
        document.body.appendChild(ta)
        ta.select()
        document.execCommand('copy')
        document.body.removeChild(ta)
        setCopied(true)
        setTimeout(() => setCopied(false), 1500)
      })
    } else {
      const ta = document.createElement('textarea')
      ta.value = prompt
      ta.style.position = 'fixed'
      ta.style.opacity = '0'
      document.body.appendChild(ta)
      ta.select()
      document.execCommand('copy')
      document.body.removeChild(ta)
      setCopied(true)
      setTimeout(() => setCopied(false), 1500)
    }
  }

  const handleDelete = async () => {
    if (!confirmRef.current) {
      confirmRef.current = true
      setConfirmDelete(true)
      clearTimeout(timeoutRef.current)
      timeoutRef.current = setTimeout(() => {
        confirmRef.current = false
        setConfirmDelete(false)
      }, 3000)
      return
    }
    clearTimeout(timeoutRef.current)
    confirmRef.current = false
    setConfirmDelete(false)
    // Release video element src to unlock the file on Windows
    if (videoRef.current) {
      videoRef.current.pause()
      videoRef.current.removeAttribute('src')
      videoRef.current.load()
    }
    setSelectedOutput(index)
    // Small delay to let the browser release the file handle
    setTimeout(() => deleteOutput(), 200)
  }

  const handleRejoin = async () => {
    if (!groupId) return
    setRejoining(true)
    try {
      await rejoinClipGroup(groupId)
    } finally {
      setRejoining(false)
    }
  }

  // Keep the labeled action popover transient. Workspace choices live inside
  // the same popover, so one outside-click boundary handles both levels.
  useEffect(() => {
    if (!showActionMenu) return
    const handler = (e: MouseEvent) => {
      if (actionMenuRef.current && !actionMenuRef.current.contains(e.target as Node)) {
        setShowActionMenu(false)
        setShowMoveMenu(false)
      }
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [showActionMenu])

  const handleMove = async (targetWs: string) => {
    setMoving(true)
    setShowMoveMenu(false)
    try {
      await moveOutput(file.name, targetWs)
      // Immediately remove from local state (source may still exist during deferred cleanup)
      const store = useStore.getState()
      const filtered = store.outputs.filter(o => o.name !== file.name)
      useStore.setState({ outputs: filtered, selectedOutput: Math.min(store.selectedOutput, Math.max(0, filtered.length - 1)) })
      setShowActionMenu(false)
    } catch (e) {
      console.error('Move failed:', e)
    } finally {
      setMoving(false)
    }
  }

  const handleSendToInput = async () => {
    if (file.type !== 'image') return
    try {
      const res = await fetch(getFileUrl(file.name))
      const blob = await res.blob()
      const imageFile = new File([blob], file.name, { type: blob.type || 'image/png' })
      if (generationMode === 'image') {
        addImageRef(imageFile)
      } else {
        setStartImage(imageFile)
      }
      setSentToInput(true)
      setTimeout(() => setSentToInput(false), 2000)
    } catch (e) {
      console.error('Failed to send image to input:', e)
    }
  }

  // Capture the frame the video preview is currently SHOWING (canvas grab
  // of the <video> element at its currentTime — same-origin, so no taint)
  // and append it to the Reference tiles. Pairs with SCAIL-2: scrub to the
  // pose you want, one click, it's your character reference.
  const handleSendFrameToRefs = async () => {
    if (file.type !== 'video') return
    try {
      let video = videoRef.current
      if (!video || video.videoWidth === 0) {
        // Preview not loaded (never hovered) — decode frame 0 offscreen.
        video = document.createElement('video')
        video.src = getFileUrl(file.name)
        video.muted = true
        await new Promise<void>((resolve, reject) => {
          video!.onloadeddata = () => resolve()
          video!.onerror = () => reject(new Error('video load failed'))
        })
      }
      const canvas = document.createElement('canvas')
      canvas.width = video.videoWidth
      canvas.height = video.videoHeight
      const ctx = canvas.getContext('2d')
      if (!ctx) throw new Error('canvas unavailable')
      ctx.drawImage(video, 0, 0)
      const blob: Blob = await new Promise((resolve, reject) =>
        canvas.toBlob(b => (b ? resolve(b) : reject(new Error('frame capture failed'))), 'image/png')
      )
      const stem = file.name.replace(/\.[^.]+$/, '')
      const frameFile = new File([blob], `${stem}_t${video.currentTime.toFixed(2)}s.png`, { type: 'image/png' })
      addImageRef(frameFile)
      setSentToInput(true)
      setTimeout(() => setSentToInput(false), 2000)
    } catch (e) {
      console.error('Failed to capture video frame:', e)
    }
  }

  const handleContinueFrom = async () => {
    if (file.type !== 'video') return
    try {
      const res = await fetch(getFileUrl(file.name))
      const blob = await res.blob()
      const videoFile = new File([blob], file.name, { type: blob.type || 'video/mp4' })
      const url = URL.createObjectURL(videoFile)
      const video = document.createElement('video')
      video.src = url
      video.onloadedmetadata = async () => {
        const duration = video.duration && isFinite(video.duration) ? video.duration : 0
        const uploaded = await uploadImage(videoFile)
        // Switch sub-mode FIRST: the switch stashes the current sub-mode's
        // working set and opens Extend's own slate. Setting the source
        // after keeps it from being wiped by that swap.
        setParam('image_mode', 3)
        setContinueVideo(videoFile, uploaded.path, url, duration)
      }
    } catch (e) {
      console.error('Failed to load video for continuation:', e)
    }
  }

  const handleDownload = () => {
    const link = document.createElement('a')
    link.href = getFileUrl(file.name)
    link.download = file.name
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
  }

  return (
    <div
      ref={itemRef}
      data-feed-index={index}
      style={style}
      className={`rounded-xl border-2 overflow-visible transition-colors ${showActionMenu ? 'z-40' : 'z-0'} ${
        // Active frame: theme-aware bezel via frame-active-gradient.
        //
        // Default theme: linear gradient with both stops set to
        // accent-blue → reads as a flat 2px blue ring (preserves
        // prior visual exactly).
        //
        // Golden Hour: a conic-gradient override (see index.css)
        // sweeps "spotlight stops" around the perimeter — bright
        // orange / gold / ember at three asymmetric angles, with
        // bg-primary in between so those sections of the border
        // blend into the surrounding panel. The effect reads as
        // "stage lights catching the edge of the asset at random
        // points" rather than a uniform halo or solid line.
        //
        // shadow-active-ring is now minimal (just a 6px / 15% wash)
        // because the visual character lives ON the bezel itself,
        // not as an outward glow.
        isActive
          ? 'border-transparent frame-active-gradient shadow-active-ring'
          : 'border-border bg-bg-tertiary'
      }`}
      onClick={handleSelect}
    >
      {/* Media player — bg-media-canvas keeps the letterbox dark even on light themes */}
      <div className="relative flex aspect-video w-full items-center justify-center overflow-hidden rounded-t-[10px] bg-media-canvas">
        {file.type === 'video' ? (
          <video
            ref={videoRef}
            key={file.url}
            src={file.url}
            controls
            loop
            playsInline
            className="w-full h-full object-contain"
            muted={!isActive}
            data-gallery-media="true"
            onPlay={handlePlaybackStart}
          />
        ) : file.type === 'audio' ? (
          <div className="flex flex-col items-center gap-4">
            <div className="w-16 h-16 rounded-2xl bg-bg-active flex items-center justify-center">
              <Play size={24} className="text-text-muted" />
            </div>
            <p className="text-xs text-text-muted mb-2">{file.name}</p>
            <audio
              key={file.url}
              src={file.url}
              controls
              className="w-64"
              data-gallery-media="true"
              onPlay={handlePlaybackStart}
            />
          </div>
        ) : (
          <RetryImage key={file.url} url={file.url} alt={file.name} />
        )}
      </div>

      {/* Inline info bar */}
      <div className="px-3 py-2 flex items-center gap-2 min-h-[40px]">
        {imageStartFile && (
          <img
            src={getUploadUrl(imageStartFile)}
            alt="Start"
            className="w-7 h-7 rounded border border-border object-cover shrink-0"
            title="Start image"
          />
        )}
        {imageEndFile && (
          <img
            src={getUploadUrl(imageEndFile)}
            alt="End"
            className="w-7 h-7 rounded border border-border object-cover shrink-0"
            title="End image"
          />
        )}

        <div className="flex-1 min-w-0">
          {params ? (
            <>
              <div className="text-xs text-text-secondary truncate">
                {modelLabel && <span className="font-medium" title={modelType}>{modelLabel}</span>}
                {resolution && <span className="text-text-muted"> &middot; {resolution}</span>}
                {seed != null && seed >= 0 && <span className="text-text-muted"> &middot; seed {seed}</span>}
                {generationTime != null && (
                  <span
                  className="text-text-muted"
                  title={meta?.generation_time_basis === 'active'
                    ? 'Generation time (excluding queue wait and model loading)'
                    : 'Recorded generation time'}
                  >
                    {' '}&middot; {formatGenerationDuration(generationTime)}
                  </span>
                )}
                {clipIndex != null && clipTotal != null && (
                  <span className="text-accent-blue"> &middot; clip {clipIndex + 1}/{clipTotal}</span>
                )}
              </div>
              {cardPrompt && (
                <div className="text-[11px] text-text-muted truncate mt-0.5" title={cardPrompt}>
                  {effectiveWindowPrompts.length > 0 && (
                    <span className="text-accent-blue">
                      {generatedWindowPrompts ? 'AI window 1' : 'Window 1'} &middot;{' '}
                    </span>
                  )}
                  {cardPrompt}
                </div>
              )}
            </>
          ) : metaLoaded ? (
            <div className="text-[11px] text-text-muted truncate">{file.name}</div>
          ) : (
            <div className="text-[11px] text-text-muted animate-pulse">Loading...</div>
          )}
        </div>

        {/* Four persistent controls; secondary actions are labeled in More. */}
        <div ref={actionMenuRef} className="relative flex shrink-0 items-center gap-0.5" onClick={e => e.stopPropagation()}>
          {params && (
            <button
              onClick={() => {
                onActivate(index)
                setShowDetails(value => !value)
                setShowActionMenu(false)
                setShowMoveMenu(false)
              }}
              className={`rounded-lg p-1.5 transition-colors ${
                showDetails
                  ? 'bg-bg-active text-accent-blue'
                  : 'text-text-secondary hover:bg-bg-hover hover:text-text-primary'
              }`}
              title={showDetails ? 'Hide generation details' : 'Show generation details'}
              aria-label={showDetails ? 'Hide generation details' : 'Show generation details'}
              aria-expanded={showDetails}
            >
              <span className="flex items-center gap-0.5">
                <Info size={14} />
                {showDetails ? <ChevronUp size={10} /> : <ChevronDown size={10} />}
              </span>
            </button>
          )}
          {params && (
            <button
              onClick={() => {
                setShowActionMenu(false)
                setShowMoveMenu(false)
                handleLoadSettings()
              }}
              className="rounded-lg p-1.5 text-text-secondary transition-colors hover:bg-bg-hover hover:text-text-primary"
              title="Load settings"
              aria-label="Load settings"
            >
              <Pencil size={14} />
            </button>
          )}
          {!browsingUploads && (
            <button
              onClick={() => toggleFavorite(file.name)}
              className={`rounded-lg p-1.5 transition-colors ${
                file.favorite
                  ? 'text-red-400 hover:text-red-300'
                  : 'text-text-secondary hover:bg-bg-hover hover:text-red-400'
              }`}
              title={file.favorite ? 'Remove from favorites' : 'Add to favorites'}
              aria-label={file.favorite ? 'Remove from favorites' : 'Add to favorites'}
            >
              <Heart size={14} fill={file.favorite ? 'currentColor' : 'none'} />
            </button>
          )}
          <button
            onClick={() => {
              onActivate(index)
              const rect = actionMenuRef.current?.getBoundingClientRect()
              if (rect) {
                const spaceAbove = rect.top - 8
                const spaceBelow = window.innerHeight - rect.bottom - 8
                setActionMenuOpensDown(spaceBelow > spaceAbove)
              }
              setShowActionMenu(value => !value)
              setShowMoveMenu(false)
            }}
            className={`rounded-lg p-1.5 transition-colors ${
              showActionMenu
                ? 'bg-bg-active text-accent-blue'
                : 'text-text-secondary hover:bg-bg-hover hover:text-text-primary'
            }`}
            title="More clip actions"
            aria-label="More clip actions"
            aria-haspopup="menu"
            aria-expanded={showActionMenu}
          >
            <MoreHorizontal size={15} />
          </button>

          {showActionMenu && (
            <div
              role="menu"
              aria-label="Clip actions"
              className={`absolute right-0 z-50 max-h-[min(420px,65vh)] w-64 max-w-[calc(100vw-2rem)] overflow-y-auto rounded-xl border border-border bg-bg-secondary p-1.5 shadow-2xl ${
                actionMenuOpensDown ? 'top-full mt-2' : 'bottom-full mb-2'
              }`}
            >
              <div className="px-2 py-1.5 text-[10px] font-semibold uppercase tracking-wider text-text-muted">
                Clip actions
              </div>
              {params && (
                <button
                  role="menuitem"
                  onClick={() => {
                    setShowActionMenu(false)
                    setShowSaveRecipe(true)
                  }}
                  className="flex w-full items-center gap-2.5 rounded-lg px-2.5 py-2 text-left text-xs text-text-secondary transition-colors hover:bg-bg-hover hover:text-text-primary"
                >
                  <BookMarked size={14} className="text-accent-blue" />
                  <span>Save as Recipe</span>
                </button>
              )}
              {params && (
                <button
                  role="menuitem"
                  onClick={() => {
                    setShowActionMenu(false)
                    handleReroll()
                  }}
                  className="flex w-full items-center gap-2.5 rounded-lg px-2.5 py-2 text-left text-xs text-text-secondary transition-colors hover:bg-bg-hover hover:text-text-primary"
                >
                  <RefreshCw size={14} />
                  <span>Regenerate with same settings</span>
                </button>
              )}
              {params && file.type === 'video' && (
                <button
                  role="menuitem"
                  onClick={() => {
                    setShowActionMenu(false)
                    openRetakeDialog(file.name)
                  }}
                  className="flex w-full items-center gap-2.5 rounded-lg px-2.5 py-2 text-left text-xs text-text-secondary transition-colors hover:bg-bg-hover hover:text-indicator-warning"
                >
                  <Scissors size={14} />
                  <span>Retake a time region</span>
                </button>
              )}
              {params && file.type === 'video' && (
                <button
                  role="menuitem"
                  onClick={() => {
                    setShowActionMenu(false)
                    handleContinueFrom()
                  }}
                  className="flex w-full items-center gap-2.5 rounded-lg px-2.5 py-2 text-left text-xs text-text-secondary transition-colors hover:bg-bg-hover hover:text-accent-blue"
                >
                  <FastForward size={14} />
                  <span>Extend this video</span>
                </button>
              )}
              {groupId && (
                <button
                  role="menuitem"
                  onClick={async () => {
                    await handleRejoin()
                    setShowActionMenu(false)
                  }}
                  disabled={rejoining}
                  className="flex w-full items-center gap-2.5 rounded-lg px-2.5 py-2 text-left text-xs text-accent-blue transition-colors hover:bg-bg-hover disabled:opacity-50"
                >
                  {rejoining ? <Loader2 size={14} className="animate-spin" /> : <Combine size={14} />}
                  <span>Rejoin all {clipTotal} clips</span>
                </button>
              )}
              {params && prompt && (
                <button
                  role="menuitem"
                  onClick={handleCopyPrompt}
                  className="flex w-full items-center gap-2.5 rounded-lg px-2.5 py-2 text-left text-xs text-text-secondary transition-colors hover:bg-bg-hover hover:text-text-primary"
                >
                  {copied ? <Check size={14} className="text-accent-green" /> : <Copy size={14} />}
                  <span>{copied ? 'Prompt copied' : 'Copy prompt'}</span>
                </button>
              )}
              {file.type === 'image' && (
                <button
                  role="menuitem"
                  onClick={handleSendToInput}
                  className="flex w-full items-center gap-2.5 rounded-lg px-2.5 py-2 text-left text-xs text-text-secondary transition-colors hover:bg-bg-hover hover:text-accent-blue"
                >
                  {sentToInput ? <Check size={14} className="text-accent-green" /> : <ArrowLeftToLine size={14} />}
                  <span>{generationMode === 'image' ? 'Use as input image' : 'Use as start frame'}</span>
                </button>
              )}
              {file.type === 'video' && (
                <button
                  role="menuitem"
                  onClick={handleSendFrameToRefs}
                  className="flex w-full items-center gap-2.5 rounded-lg px-2.5 py-2 text-left text-xs text-text-secondary transition-colors hover:bg-bg-hover hover:text-accent-blue"
                >
                  {sentToInput ? <Check size={14} className="text-accent-green" /> : <ArrowLeftToLine size={14} />}
                  <span>Use current frame as reference</span>
                </button>
              )}
              <button
                role="menuitem"
                onClick={() => {
                  handleDownload()
                  setShowActionMenu(false)
                }}
                className="flex w-full items-center gap-2.5 rounded-lg px-2.5 py-2 text-left text-xs text-text-secondary transition-colors hover:bg-bg-hover hover:text-text-primary"
              >
                <Download size={14} />
                <span>Download</span>
              </button>
              {!browsingUploads && (
                <>
                  <button
                    role="menuitem"
                    onClick={() => setShowMoveMenu(value => !value)}
                    disabled={moving}
                    className="flex w-full items-center gap-2.5 rounded-lg px-2.5 py-2 text-left text-xs text-text-secondary transition-colors hover:bg-bg-hover hover:text-text-primary disabled:opacity-50"
                    aria-expanded={showMoveMenu}
                  >
                    {moving ? <Loader2 size={14} className="animate-spin text-accent-blue" /> : <FolderInput size={14} />}
                    <span className="flex-1">Move to workspace</span>
                    {showMoveMenu ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
                  </button>
                  {showMoveMenu && (
                    <div className="mx-2 mb-1 overflow-hidden rounded-lg border border-border bg-bg-tertiary">
                      {workspaces.filter(ws => ws.name !== activeWorkspace).map(ws => (
                        <button
                          key={ws.name}
                          role="menuitem"
                          onClick={() => handleMove(ws.name)}
                          className="w-full px-3 py-2 text-left text-[11px] text-text-secondary transition-colors hover:bg-bg-hover hover:text-text-primary"
                        >
                          {ws.name}
                        </button>
                      ))}
                      {workspaces.filter(ws => ws.name !== activeWorkspace).length === 0 && (
                        <div className="px-3 py-2 text-[10px] text-text-muted">No other workspaces</div>
                      )}
                    </div>
                  )}
                </>
              )}
              {!browsingUploads && (
                <button
                  role="menuitem"
                  onClick={() => {
                    const alreadyConfirmed = confirmRef.current
                    handleDelete()
                    if (alreadyConfirmed) setShowActionMenu(false)
                  }}
                  className={`flex w-full items-center gap-2.5 rounded-lg px-2.5 py-2 text-left text-xs transition-colors ${
                    confirmDelete
                      ? 'bg-red-500/15 text-red-400 hover:bg-red-500/25'
                      : 'text-text-secondary hover:bg-bg-hover hover:text-red-400'
                  }`}
                >
                  <Trash2 size={14} />
                  <span>{confirmDelete ? 'Click again to delete' : 'Delete output'}</span>
                </button>
              )}
            </div>
          )}
        </div>
      </div>
      {showDetails && params && (
        <div
          className="rounded-b-[10px] border-t border-border bg-bg-secondary/70 px-3 py-3"
          onClick={event => event.stopPropagation()}
        >
          <div className="flex flex-wrap gap-1.5 mb-3">
            {h3Workflow && (
              <span className="rounded-full border border-border bg-bg-tertiary px-2 py-0.5 text-[10px] text-text-secondary">
                {h3Workflow}
              </span>
            )}
            {resolution && (
              <span className="rounded-full border border-border bg-bg-tertiary px-2 py-0.5 text-[10px] text-text-secondary">
                {resolution}
              </span>
            )}
            {inferenceSteps != null && (
              <span className="rounded-full border border-border bg-bg-tertiary px-2 py-0.5 text-[10px] text-text-secondary">
                {inferenceSteps} steps
              </span>
            )}
            {isMultiWindow && (
              <span className="rounded-full border border-border bg-bg-tertiary px-2 py-0.5 text-[10px] text-text-secondary">
                {windowCount} windows
              </span>
            )}
            {isMultiWindow && sceneDurationSeconds > 0 && (
              <span className="rounded-full border border-border bg-bg-tertiary px-2 py-0.5 text-[10px] text-text-secondary">
                {formatDuration(sceneDurationSeconds, true)} scene
              </span>
            )}
            {optimizationLabels.map(label => (
              <span
                key={label}
                className="rounded-full border border-accent-blue/30 bg-accent-blue/10 px-2 py-0.5 text-[10px] text-accent-blue"
              >
                {label}
              </span>
            ))}
          </div>

          <dl className="grid grid-cols-[auto_minmax(0,1fr)] gap-x-3 gap-y-1 text-[11px]">
            <dt className="text-text-muted">Model</dt>
            <dd className="text-text-secondary break-words">{modelLabel || modelType || 'Unknown'}</dd>
            {h3Workflow && (
              <>
                <dt className="text-text-muted">Workflow</dt>
                <dd className="text-text-secondary">{h3Workflow}</dd>
              </>
            )}
            {resolution && (
              <>
                <dt className="text-text-muted">Resolution</dt>
                <dd className="text-text-secondary">{resolution}</dd>
              </>
            )}
            {inferenceSteps != null && (
              <>
                <dt className="text-text-muted">Sampling</dt>
                <dd className="text-text-secondary">
                  {inferenceSteps} steps{guidanceScale != null ? ` · guidance ${guidanceScale}` : ''}
                </dd>
              </>
            )}
            {isMultiWindow && (
              <>
                <dt className="text-text-muted">Sequence</dt>
                <dd className="text-text-secondary">
                  {windowCount} windows
                  {completedWindowCount > 0 && completedWindowCount < windowCount
                    ? ` · ${completedWindowCount} completed`
                    : ''}
                </dd>
              </>
            )}
            {isMultiWindow && sceneDurationSeconds > 0 && (
              <>
                <dt className="text-text-muted">Scene duration</dt>
                <dd className="text-text-secondary">
                  {formatDuration(sceneDurationSeconds, true)}
                </dd>
              </>
            )}
            {isMultiWindow && totalWindowGenerationSeconds > 0 && (
              <>
                <dt className="text-text-muted">Total render</dt>
                <dd className="text-text-secondary">
                  {formatGenerationDuration(totalWindowGenerationSeconds)}
                </dd>
              </>
            )}
            {seed != null && seed >= 0 && (
              <>
                <dt className="text-text-muted">Seed</dt>
                <dd className="text-text-secondary">{seed}</dd>
              </>
            )}
            {optimizationLabels.length > 0 && (
              <>
                <dt className="text-text-muted">Optimizations</dt>
                <dd className="text-text-secondary">{optimizationLabels.join(' · ')}</dd>
              </>
            )}
            {turboEnabled && turboPreset && (
              <>
                <dt className="text-text-muted">Turbo preset</dt>
                <dd className="break-words text-text-secondary">{turboPreset}</dd>
              </>
            )}
            {firstBlockEnabled && (
              <>
                <dt className="text-text-muted">Cache tuning</dt>
                <dd className="text-text-secondary">
                  threshold {String(params.skip_steps_multiplier ?? 'default')}
                  {params.skip_steps_start_step_perc != null
                    ? ` · starts at ${String(params.skip_steps_start_step_perc)}%`
                    : ''}
                </dd>
              </>
            )}
          </dl>

          {isMultiWindow && (
            <div className="mt-3 rounded-lg border border-border bg-bg-tertiary/70 p-2.5">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div className="text-[10px] font-medium uppercase tracking-wide text-text-muted">
                  Window timing
                </div>
                <div className="text-[10px] text-text-muted">
                  {windowCount} windows
                  {sceneDurationSeconds > 0
                    ? ` · ${formatDuration(sceneDurationSeconds, true)} scene`
                    : ''}
                  {totalWindowGenerationSeconds > 0
                    ? ` · ${formatGenerationDuration(totalWindowGenerationSeconds)} render`
                    : ''}
                </div>
              </div>
              {timedWindowSeconds.length > 0 ? (
                <div className="mt-2 grid grid-cols-2 gap-1.5 sm:grid-cols-3">
                  {timedWindowSeconds.map((seconds, windowIndex) => (
                    <div
                      key={`window-timing-${windowIndex}`}
                      className="flex items-center justify-between gap-2 rounded-md border border-border/70 bg-bg-secondary px-2 py-1.5 text-[10px]"
                    >
                      <span className="text-text-muted">Window {windowIndex + 1}</span>
                      <span className="font-medium text-text-secondary">
                        {formatGenerationDuration(seconds)}
                      </span>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="mt-2 text-[10px] leading-relaxed text-text-muted">
                  Per-window completion times are recorded for new multi-window generations.
                </div>
              )}
            </div>
          )}

          {activeLoras.length > 0 && (
            <div className="mt-3">
              <div className="mb-1 text-[10px] font-medium uppercase tracking-wide text-text-muted">Active LoRAs</div>
              <div className="space-y-1">
                {activeLoras.map((lora, loraIndex) => (
                  <div key={`${lora}-${loraIndex}`} className="flex gap-2 text-[11px]">
                    <span className="min-w-0 flex-1 break-all text-text-secondary">{lora}</span>
                    {loraWeights[loraIndex] && (
                      <span className="shrink-0 text-text-muted">{loraWeights[loraIndex]}x</span>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          {prompt && (
            <div className="mt-3">
              <div className="mb-1 flex items-center justify-between gap-2">
                <span className="text-[10px] font-medium uppercase tracking-wide text-text-muted">
                  {effectiveWindowPrompts.length > 0 ? 'Source prompt' : 'Prompt'}
                </span>
                <button
                  type="button"
                  onClick={handleCopyPrompt}
                  className="flex items-center gap-1 rounded px-1.5 py-0.5 text-[10px] text-text-muted hover:bg-bg-hover hover:text-text-primary"
                >
                  {copied ? <Check size={10} className="text-accent-green" /> : <Copy size={10} />}
                  {copied ? 'Copied' : 'Copy'}
                </button>
              </div>
              <div className="max-h-40 overflow-y-auto whitespace-pre-wrap break-words rounded-lg border border-border bg-bg-tertiary p-2 text-[11px] leading-relaxed text-text-secondary">
                {prompt}
              </div>
            </div>
          )}
          {originalPrompt && originalPrompt.trim() !== prompt.trim() && (
            <div className="mt-3">
              <div className="mb-1 text-[10px] font-medium uppercase tracking-wide text-text-muted">Original prompt</div>
              <div className="max-h-28 overflow-y-auto whitespace-pre-wrap break-words rounded-lg border border-border bg-bg-tertiary p-2 text-[11px] leading-relaxed text-text-secondary">
                {originalPrompt}
              </div>
            </div>
          )}
          {effectiveWindowPrompts.length > 0 && (
            <div className="mt-3">
              <div className="mb-1 text-[10px] font-medium uppercase tracking-wide text-text-muted">
                {windowPromptsHeading} ({effectiveWindowPrompts.length})
              </div>
              <div className="space-y-2">
                {effectiveWindowPrompts.map((windowPrompt, windowIndex) => (
                  <details
                    key={`effective-window-${windowIndex}`}
                    className="overflow-hidden rounded-lg border border-border bg-bg-tertiary"
                    open={effectiveWindowPrompts.length <= 2}
                  >
                    <summary className="cursor-pointer select-none px-2 py-1.5 text-[11px] font-medium text-text-secondary hover:bg-bg-hover">
                      Window {windowIndex + 1}
                    </summary>
                    <div className="whitespace-pre-wrap break-words border-t border-border px-2 py-2 text-[11px] leading-relaxed text-text-secondary">
                      {windowPrompt}
                    </div>
                  </details>
                ))}
              </div>
            </div>
          )}
          {h3PlanningWarnings.length > 0 && (
            <div className="mt-3 rounded-lg border border-amber-500/25 bg-amber-500/10 p-2">
              <div className="mb-1 text-[10px] font-medium uppercase tracking-wide text-amber-300">
                Planning notes
              </div>
              {h3PlanningWarnings.map((warning, warningIndex) => (
                <div key={`h3-planning-warning-${warningIndex}`} className="text-[11px] leading-relaxed text-text-secondary">
                  {warning}
                </div>
              ))}
              {h3PlanningDiagnostics.length > 0 && (
                <details className="mt-1 text-[10px] text-text-muted">
                  <summary className="cursor-pointer select-none">Why repair was needed</summary>
                  <ul className="mt-1 list-disc space-y-0.5 pl-4">
                    {h3PlanningDiagnostics.map((diagnostic, diagnosticIndex) => (
                      <li key={`h3-planning-diagnostic-${diagnosticIndex}`}>{diagnostic}</li>
                    ))}
                  </ul>
                </details>
              )}
            </div>
          )}
          {h3PlanningNotes.length > 0 && (
            <div className="mt-3 rounded-lg border border-border bg-bg-tertiary p-2">
              <div className="mb-1 text-[10px] font-medium uppercase tracking-wide text-text-muted">
                H3 timing notes
              </div>
              {h3PlanningNotes.map((note, noteIndex) => (
                <div key={`h3-planning-note-${noteIndex}`} className="text-[11px] leading-relaxed text-text-secondary">
                  {note}
                </div>
              ))}
            </div>
          )}
        </div>
      )}
      {showSaveRecipe && (
        <SaveRecipeDialog
          defaultNsfw={nsfwMode}
          onCancel={() => setShowSaveRecipe(false)}
          onSave={async (name, description, nsfw) => {
            await saveRecipeFromOutput(file.name, name, description, nsfw)
            setShowSaveRecipe(false)
          }}
        />
      )}
    </div>
  )
}
