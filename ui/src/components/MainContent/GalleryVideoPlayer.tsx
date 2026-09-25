import { forwardRef, useCallback, useEffect, useImperativeHandle, useLayoutEffect, useRef, useState } from 'react'
import { Pause, Play, Volume2, VolumeX } from 'lucide-react'

export interface GalleryVideoPlayerHandle {
  revealControls: () => void
  togglePlayback: () => void
  prepareSource: (src: string) => Promise<GalleryVideoPreparationResult>
  cancelPreparedSource: () => void
}

export type GalleryVideoPreparationResult = 'ready' | 'error' | 'timeout' | 'cancelled'

export interface GalleryVideoPlayerProps {
  src?: string
  name: string
  initialTime?: number
  onFrameReady?: (src: string) => void
}

type AutoplayFallback = 'sound' | 'play' | null
type FrozenFrame = { src: string; url: string }
type StagedNavigation = {
  src: string
  previousSrc?: string
  previousTime: number
  wasPlaying: boolean
  previousMuted: boolean
}

interface PendingPreparation extends StagedNavigation {
  generation: number
  promise: Promise<GalleryVideoPreparationResult>
  resolve: (result: GalleryVideoPreparationResult) => void
  cancel: (result: GalleryVideoPreparationResult) => void
  timer: ReturnType<typeof setTimeout>
  onLoadedData: () => void
  onError: () => void
  capturingFrame: boolean
}

const PREPARATION_TIMEOUT_MS = 3500

function formatTime(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) return '0:00'
  const wholeSeconds = Math.floor(seconds)
  const hours = Math.floor(wholeSeconds / 3600)
  const minutes = Math.floor((wholeSeconds % 3600) / 60)
  const remainder = wholeSeconds % 60
  if (hours > 0) return `${hours}:${String(minutes).padStart(2, '0')}:${String(remainder).padStart(2, '0')}`
  return `${minutes}:${String(remainder).padStart(2, '0')}`
}

function stopPropagation(event: { stopPropagation: () => void }) {
  event.stopPropagation()
}

function stopGalleryArrowNavigation(event: React.KeyboardEvent) {
  if (event.key === 'ArrowUp' || event.key === 'ArrowDown' || event.key === 'ArrowLeft' || event.key === 'ArrowRight') {
    event.stopPropagation()
  }
}

function videoHasSource(video: HTMLVideoElement, src: string): boolean {
  if (video.getAttribute('src') === src) return true
  try {
    return video.src === new URL(src, document.baseURI).href
  } catch {
    return false
  }
}

function isAtRequestedTime(video: HTMLVideoElement, requestedTime: number | undefined): boolean {
  if (!Number.isFinite(requestedTime) || (requestedTime ?? 0) <= 0) return true
  const target = Math.max(0, Math.min(
    requestedTime ?? 0,
    Number.isFinite(video.duration) ? video.duration : (requestedTime ?? 0),
  ))
  return Math.abs(video.currentTime - target) < 0.2
}

function drawVideoFrame(video: HTMLVideoElement, canvas: HTMLCanvasElement): boolean {
  if (video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA || video.videoWidth <= 0 || video.videoHeight <= 0) return false
  try {
    const scale = Math.min(1, 1920 / Math.max(video.videoWidth, video.videoHeight))
    canvas.width = Math.max(1, Math.round(video.videoWidth * scale))
    canvas.height = Math.max(1, Math.round(video.videoHeight * scale))
    const context = canvas.getContext('2d')
    if (!context) return false
    context.drawImage(video, 0, 0, canvas.width, canvas.height)
    return true
  } catch {
    // A tainted or unavailable canvas must not prevent normal video navigation.
    return false
  }
}

async function captureVideoFrame(video: HTMLVideoElement): Promise<string | null> {
  const canvas = document.createElement('canvas')
  if (!drawVideoFrame(video, canvas)) return null
  try {
    const blob = await new Promise<Blob | null>(resolve => canvas.toBlob(resolve, 'image/webp', 0.82))
    return blob ? URL.createObjectURL(blob) : null
  } catch {
    // A tainted or unavailable canvas must not prevent normal video navigation.
    return null
  }
}

async function waitForImageDecode(src: string): Promise<void> {
  const image = new Image()
  image.src = src
  if (typeof image.decode === 'function') {
    await image.decode().catch(() => undefined)
    return
  }
  await new Promise<void>(resolve => {
    if (image.complete) return resolve()
    image.onload = () => resolve()
    image.onerror = () => resolve()
    window.setTimeout(resolve, 500)
  })
}

export const GalleryVideoPlayer = forwardRef<GalleryVideoPlayerHandle, GalleryVideoPlayerProps>(function GalleryVideoPlayer(
  { src, name, initialTime, onFrameReady },
  ref,
) {
  const videoRef = useRef<HTMLVideoElement>(null)
  const outgoingCanvasRef = useRef<HTMLCanvasElement>(null)
  const initialSeekDoneRef = useRef(false)
  const activeMediaRef = useRef(false)
  const activeSrcRef = useRef<string | undefined>(undefined)
  const desiredSrcRef = useRef(src)
  const mediaGenerationRef = useRef(0)
  const preparationGenerationRef = useRef(0)
  const preparedSrcRef = useRef<string | undefined>(undefined)
  const stagedNavigationRef = useRef<StagedNavigation | null>(null)
  const pendingPreparationRef = useRef<PendingPreparation | null>(null)
  const frameCallbackRef = useRef<{ video: HTMLVideoElement; id: number; kind: 'video' | 'animation'; fallbackId?: number } | null>(null)
  const preparedFrameRef = useRef<FrozenFrame | null>(null)
  const onFrameReadyRef = useRef(onFrameReady)
  const readyReportedSrcRef = useRef<string | undefined>(undefined)
  const userMutedRef = useRef(false)
  const ignoreCleanupPauseRef = useRef(false)
  const controlsVisibleRef = useRef(false)
  const controlsFocusedRef = useRef(false)
  const pointerInteractionRef = useRef(false)
  const revealTriggerFocusedRef = useRef(false)
  const scrubbingRef = useRef(false)
  const hideTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  const [stateSrc, setStateSrc] = useState(src)
  const [controlsVisible, setControlsVisible] = useState(false)
  const [isPlaying, setIsPlaying] = useState(false)
  const [isMuted, setIsMuted] = useState(false)
  const [preferredMuted, setPreferredMuted] = useState(false)
  const [currentTime, setCurrentTime] = useState(0)
  const [duration, setDuration] = useState(0)
  const [autoplayFallback, setAutoplayFallback] = useState<AutoplayFallback>(null)
  const [mediaLoadError, setMediaLoadError] = useState(false)
  const [frameReady, setFrameReady] = useState(false)
  const [outgoingFrameVisible, setOutgoingFrameVisible] = useState(false)
  const [preparedFrame, setPreparedFrame] = useState<FrozenFrame | null>(null)

  if (stateSrc !== src) {
    setStateSrc(src)
    setControlsVisible(false)
    setIsPlaying(false)
    setIsMuted(preferredMuted)
    setCurrentTime(0)
    setDuration(0)
    setAutoplayFallback(null)
    setMediaLoadError(false)
    setFrameReady(false)
  }

  useLayoutEffect(() => {
    // Commit the destination before the outgoing source's passive cleanup.
    desiredSrcRef.current = src
    readyReportedSrcRef.current = undefined
  }, [src])

  useLayoutEffect(() => {
    onFrameReadyRef.current = onFrameReady
  }, [onFrameReady])

  const revokeFrameLater = useCallback((previous: FrozenFrame | null, next: FrozenFrame | null) => {
    if (previous && previous.url !== next?.url) {
      window.requestAnimationFrame(() => URL.revokeObjectURL(previous.url))
    }
  }, [])

  const hideOutgoingFrame = useCallback(() => {
    setOutgoingFrameVisible(false)
    const canvas = outgoingCanvasRef.current
    if (!canvas) return
    canvas.style.visibility = 'hidden'
    const context = canvas.getContext('2d')
    if (context && canvas.width > 0 && canvas.height > 0) context.clearRect(0, 0, canvas.width, canvas.height)
  }, [])

  const updatePreparedFrame = useCallback((next: FrozenFrame | null) => {
    const previous = preparedFrameRef.current
    preparedFrameRef.current = next
    setPreparedFrame(next)
    revokeFrameLater(previous, next)
  }, [revokeFrameLater])

  const cancelFrameReadyCallback = useCallback(() => {
    const pending = frameCallbackRef.current
    frameCallbackRef.current = null
    if (!pending) return
    if (pending.fallbackId !== undefined) window.cancelAnimationFrame(pending.fallbackId)
    if (pending.kind === 'animation') {
      window.cancelAnimationFrame(pending.id)
      return
    }
    const frameVideo = pending.video as HTMLVideoElement & { cancelVideoFrameCallback?: (id: number) => void }
    frameVideo.cancelVideoFrameCallback?.(pending.id)
  }, [])

  const markFrameReady = useCallback((video: HTMLVideoElement, source: string, generation: number) => {
    if (readyReportedSrcRef.current === source) return
    cancelFrameReadyCallback()
    const mark = () => {
      cancelFrameReadyCallback()
      frameCallbackRef.current = null
      if (!activeMediaRef.current || activeSrcRef.current !== source
        || mediaGenerationRef.current !== generation || readyReportedSrcRef.current === source
        || !videoHasSource(video, source) || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA || video.seeking) return
      readyReportedSrcRef.current = source
      setFrameReady(true)
      onFrameReadyRef.current?.(source)
      if (preparedFrameRef.current?.src === source) updatePreparedFrame(null)
    }
    const frameVideo = video as HTMLVideoElement & {
      requestVideoFrameCallback?: (callback: (now: number, metadata: unknown) => void) => number
    }
    if (video.paused) {
      if (video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA || video.seeking) return
      let fallbackId: number | undefined
      const id = window.requestAnimationFrame(() => {
        fallbackId = window.requestAnimationFrame(mark)
        if (frameCallbackRef.current?.id === id) frameCallbackRef.current.fallbackId = fallbackId
      })
      frameCallbackRef.current = { video, id, kind: 'animation' }
      return
    }
    if (typeof frameVideo.requestVideoFrameCallback === 'function') {
      const id = frameVideo.requestVideoFrameCallback(() => mark())
      frameCallbackRef.current = { video, id, kind: 'video' }
    } else {
      if (video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA || video.seeking) return
      let fallbackId: number | undefined
      const id = window.requestAnimationFrame(() => {
        fallbackId = window.requestAnimationFrame(mark)
        if (frameCallbackRef.current?.id === id) frameCallbackRef.current.fallbackId = fallbackId
      })
      frameCallbackRef.current = { video, id, kind: 'animation' }
    }
  }, [cancelFrameReadyCallback, updatePreparedFrame])

  const clearHideTimer = useCallback(() => {
    if (hideTimerRef.current !== null) {
      clearTimeout(hideTimerRef.current)
      hideTimerRef.current = null
    }
  }, [])

  const scheduleHide = useCallback(() => {
    clearHideTimer()
    const video = videoRef.current
    if (!video || video.paused || controlsFocusedRef.current || revealTriggerFocusedRef.current || scrubbingRef.current) return
    hideTimerRef.current = setTimeout(() => {
      hideTimerRef.current = null
      const currentVideo = videoRef.current
      if (!currentVideo || currentVideo.paused || controlsFocusedRef.current || revealTriggerFocusedRef.current || scrubbingRef.current) return
      controlsVisibleRef.current = false
      setControlsVisible(false)
    }, 3000)
  }, [clearHideTimer])

  const revealControls = useCallback(() => {
    controlsVisibleRef.current = true
    setControlsVisible(true)
    scheduleHide()
  }, [scheduleHide])

  const keepControlsVisible = useCallback((event: React.FocusEvent<HTMLElement>) => {
    const focusVisible = !pointerInteractionRef.current
      && event.target instanceof HTMLElement
      && event.target.matches(':focus-visible')
    controlsFocusedRef.current = focusVisible
    controlsVisibleRef.current = true
    setControlsVisible(true)
    if (focusVisible) clearHideTimer()
    else scheduleHide()
  }, [clearHideTimer, scheduleHide])

  const markPointerInteraction = useCallback(() => {
    pointerInteractionRef.current = true
    controlsFocusedRef.current = false
    revealTriggerFocusedRef.current = false
  }, [])

  const handleControlsPointerDown = useCallback((event: React.PointerEvent<HTMLElement>) => {
    markPointerInteraction()
    event.stopPropagation()
  }, [markPointerInteraction])

  const releaseControlsFocus = useCallback((event: React.FocusEvent<HTMLElement>) => {
    const nextTarget = event.relatedTarget
    if (nextTarget instanceof Node && event.currentTarget.contains(nextTarget)) return
    controlsFocusedRef.current = false
    scheduleHide()
  }, [scheduleHide])

  const setScrubbing = useCallback((value: boolean) => {
    scrubbingRef.current = value
    if (value) {
      controlsVisibleRef.current = true
      clearHideTimer()
      setControlsVisible(true)
    } else {
      scheduleHide()
    }
  }, [clearHideTimer, scheduleHide])

  const seekToInitialTime = useCallback((video: HTMLVideoElement) => {
    if (initialSeekDoneRef.current) return
    const requestedTime = initialTime
    if (!Number.isFinite(requestedTime) || (requestedTime ?? 0) <= 0) {
      initialSeekDoneRef.current = true
      return
    }

    const targetTime = Math.max(0, Math.min(
      requestedTime ?? 0,
      Number.isFinite(video.duration) ? video.duration : (requestedTime ?? 0),
    ))
    try {
      video.currentTime = targetTime
      initialSeekDoneRef.current = true
    } catch {
      // Some streaming formats do not accept a seek until their first frame is ready.
    }
  }, [initialTime])

  const handleLoadedMetadata = useCallback((event: React.SyntheticEvent<HTMLVideoElement>) => {
    if (!activeMediaRef.current || !activeSrcRef.current) return
    const video = event.currentTarget
    setDuration(Number.isFinite(video.duration) ? video.duration : 0)
    seekToInitialTime(video)
  }, [seekToInitialTime])

  const handleLoadedData = useCallback((event: React.SyntheticEvent<HTMLVideoElement>) => {
    const source = activeSrcRef.current
    if (!activeMediaRef.current || !source) return
    setMediaLoadError(false)
    if (isAtRequestedTime(event.currentTarget, initialTime)) markFrameReady(event.currentTarget, source, mediaGenerationRef.current)
  }, [initialTime, markFrameReady])

  const handleSeeked = useCallback((event: React.SyntheticEvent<HTMLVideoElement>) => {
    const source = activeSrcRef.current
    if (!activeMediaRef.current || !source || !isAtRequestedTime(event.currentTarget, initialTime)) return
    markFrameReady(event.currentTarget, source, mediaGenerationRef.current)
  }, [initialTime, markFrameReady])

  const handlePlay = useCallback((event: React.SyntheticEvent<HTMLVideoElement>) => {
    const source = activeSrcRef.current
    if (!activeMediaRef.current || !source) return
    setIsPlaying(true)
    setIsMuted(event.currentTarget.muted)
    if (isAtRequestedTime(event.currentTarget, initialTime)) markFrameReady(event.currentTarget, source, mediaGenerationRef.current)
    if (controlsVisibleRef.current) scheduleHide()
  }, [initialTime, markFrameReady, scheduleHide])

  const handlePause = useCallback(() => {
    if (ignoreCleanupPauseRef.current) {
      ignoreCleanupPauseRef.current = false
      return
    }
    if (!activeMediaRef.current) return
    setIsPlaying(false)
    controlsVisibleRef.current = true
    setControlsVisible(true)
    clearHideTimer()
  }, [clearHideTimer])

  const handleError = useCallback(() => {
    if (!activeMediaRef.current) return
    setIsPlaying(false)
    setMediaLoadError(true)
    setAutoplayFallback('play')
    clearHideTimer()
  }, [clearHideTimer])

  const cancelPreparedSource = useCallback(() => {
    const pending = pendingPreparationRef.current
    if (pending) pending.cancel('cancelled')
    preparationGenerationRef.current += 1

    const staged = stagedNavigationRef.current
    if (!staged) return
    if (desiredSrcRef.current === staged.src) {
      stagedNavigationRef.current = null
      preparedSrcRef.current = undefined
      return
    }

    stagedNavigationRef.current = null
    preparedSrcRef.current = undefined
    cancelFrameReadyCallback()
    hideOutgoingFrame()
    updatePreparedFrame(null)
    const video = videoRef.current
    if (!video) return

    const restoreSrc = staged.previousSrc
    const restoreGeneration = mediaGenerationRef.current + 1
    mediaGenerationRef.current = restoreGeneration
    activeMediaRef.current = Boolean(restoreSrc)
    activeSrcRef.current = restoreSrc
    readyReportedSrcRef.current = undefined
    setFrameReady(false)
    setIsPlaying(false)
    if (!restoreSrc) {
      video.pause()
      video.removeAttribute('src')
      video.load()
      return
    }

    video.muted = staged.previousMuted
    if (!videoHasSource(video, restoreSrc)) {
      video.src = restoreSrc
      video.load()
    }
    initialSeekDoneRef.current = true
    const restorePlayback = () => {
      if (mediaGenerationRef.current !== restoreGeneration || activeSrcRef.current !== restoreSrc) return
      if (Number.isFinite(staged.previousTime) && staged.previousTime > 0) {
        try { video.currentTime = staged.previousTime } catch { /* The source may not be seekable yet. */ }
      }
      if (!staged.wasPlaying) return
      try {
        void video.play().then(() => {
          if (mediaGenerationRef.current === restoreGeneration && activeSrcRef.current === restoreSrc) {
            setIsPlaying(!video.paused)
            setIsMuted(video.muted)
          }
        }).catch(() => {
          if (mediaGenerationRef.current === restoreGeneration && activeSrcRef.current === restoreSrc) {
            setAutoplayFallback('play')
          }
        })
      } catch {
        setAutoplayFallback('play')
      }
    }
    if (video.readyState >= HTMLMediaElement.HAVE_METADATA) restorePlayback()
    else video.addEventListener('loadedmetadata', restorePlayback, { once: true })
  }, [cancelFrameReadyCallback, hideOutgoingFrame, updatePreparedFrame])

  const prepareSource = useCallback(async (targetSrc: string): Promise<GalleryVideoPreparationResult> => {
    const video = videoRef.current
    if (!video || !targetSrc) return 'error'
    if (activeSrcRef.current === targetSrc && video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) return 'ready'
    if (pendingPreparationRef.current?.src === targetSrc) return pendingPreparationRef.current.promise
    if (stagedNavigationRef.current?.src === targetSrc) {
      if (video.error) return 'error'
      if (video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) return 'ready'
    }

    cancelPreparedSource()
    const generation = preparationGenerationRef.current + 1
    preparationGenerationRef.current = generation
    const staged: StagedNavigation = {
      src: targetSrc,
      previousSrc: activeSrcRef.current ?? desiredSrcRef.current,
      previousTime: video.currentTime,
      wasPlaying: !video.paused,
      previousMuted: video.muted,
    }
    stagedNavigationRef.current = staged
    const outgoingCanvas = outgoingCanvasRef.current
    const hasFrozenOutgoingFrame = Boolean(staged.previousSrc && outgoingCanvas && drawVideoFrame(video, outgoingCanvas))
    if (outgoingCanvas) outgoingCanvas.style.visibility = hasFrozenOutgoingFrame ? 'visible' : 'hidden'
    setOutgoingFrameVisible(hasFrozenOutgoingFrame)

    activeMediaRef.current = false
    activeSrcRef.current = undefined
    cancelFrameReadyCallback()
    video.pause()
    clearHideTimer()
    controlsVisibleRef.current = false
    setControlsVisible(false)
    setIsPlaying(false)
    setFrameReady(false)
    readyReportedSrcRef.current = undefined

    let resolvePromise!: (result: GalleryVideoPreparationResult) => void
    const promise = new Promise<GalleryVideoPreparationResult>(resolve => { resolvePromise = resolve })
    const finish = (result: GalleryVideoPreparationResult) => {
      if (pendingPreparationRef.current !== pending) return
      window.clearTimeout(pending.timer)
      video.removeEventListener('loadeddata', pending.onLoadedData)
      video.removeEventListener('error', pending.onError)
      pendingPreparationRef.current = null
      resolvePromise(result)
    }
    const onLoadedData = () => {
      if (pendingPreparationRef.current !== pending || pending.capturingFrame
        || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) return
      pending.capturingFrame = true
      void captureVideoFrame(video).then(targetFrameUrl => {
        const decodedFrame = targetFrameUrl ? waitForImageDecode(targetFrameUrl) : Promise.resolve()
        void decodedFrame.then(() => {
          if (pendingPreparationRef.current !== pending
            || preparationGenerationRef.current !== generation || stagedNavigationRef.current !== staged) {
            if (targetFrameUrl) URL.revokeObjectURL(targetFrameUrl)
            return
          }
          if (targetFrameUrl) updatePreparedFrame({ src: targetSrc, url: targetFrameUrl })
          finish('ready')
        })
      })
    }
    const onError = () => finish('error')
    const timer = window.setTimeout(() => finish('timeout'), PREPARATION_TIMEOUT_MS)
    const pending: PendingPreparation = {
      ...staged,
      generation,
      promise,
      resolve: resolvePromise,
      cancel: finish,
      timer,
      onLoadedData,
      onError,
      capturingFrame: false,
    }
    pendingPreparationRef.current = pending
    video.addEventListener('loadeddata', onLoadedData)
    video.addEventListener('error', onError)
    preparedSrcRef.current = targetSrc
    video.muted = staged.previousMuted
    if (!videoHasSource(video, targetSrc)) {
      video.src = targetSrc
      video.load()
    } else if (video.readyState === 0) {
      video.load()
    }
    if (video.error) onError()
    else if (video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) onLoadedData()
    return promise
  }, [cancelFrameReadyCallback, cancelPreparedSource, clearHideTimer, updatePreparedFrame])

  /* eslint-disable react-hooks/set-state-in-effect -- Synchronize the controls and frame overlays with the external video element, including an already-decoded prepared source. */
  useEffect(() => {
    const video = videoRef.current
    if (!video) return

    let alive = true
    const preparedHandoff = Boolean(src && preparedSrcRef.current === src && videoHasSource(video, src))
    const mediaGeneration = mediaGenerationRef.current + 1
    mediaGenerationRef.current = mediaGeneration
    activeMediaRef.current = Boolean(src)
    activeSrcRef.current = src
    initialSeekDoneRef.current = false
    readyReportedSrcRef.current = undefined
    setFrameReady(false)
    controlsVisibleRef.current = false
    controlsFocusedRef.current = false
    revealTriggerFocusedRef.current = false
    scrubbingRef.current = false
    clearHideTimer()

    video.muted = userMutedRef.current
    if (src) {
      if (preparedHandoff) {
        preparedSrcRef.current = undefined
        stagedNavigationRef.current = null
        hideOutgoingFrame()
        if (preparedFrameRef.current && preparedFrameRef.current.src !== src) updatePreparedFrame(null)
      } else if (!videoHasSource(video, src)) {
        video.src = src
        video.load()
      } else if (video.readyState === 0) {
        video.load()
      }

      if (video.readyState >= HTMLMediaElement.HAVE_METADATA) {
        setDuration(Number.isFinite(video.duration) ? video.duration : 0)
        seekToInitialTime(video)
        setCurrentTime(video.currentTime)
      }
      if (video.error) {
        setIsPlaying(false)
        setMediaLoadError(true)
        setAutoplayFallback('play')
      } else if (video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA && isAtRequestedTime(video, initialTime)) {
        markFrameReady(video, src, mediaGeneration)
      }

      const isCurrent = () => alive
        && mediaGenerationRef.current === mediaGeneration
        && activeSrcRef.current === src

      const tryAutoplay = async () => {
        try {
          await video.play()
          if (!isCurrent()) return
          setIsPlaying(!video.paused)
          setIsMuted(video.muted)
        } catch {
          if (!isCurrent()) return
          video.muted = true
          setIsMuted(true)
          try {
            await video.play()
            if (!isCurrent()) return
            setIsPlaying(!video.paused)
            setAutoplayFallback(userMutedRef.current ? null : 'sound')
          } catch {
            if (!isCurrent()) return
            setIsPlaying(false)
            setAutoplayFallback('play')
          }
        }
      }

      if (!video.error) void tryAutoplay()
    } else {
      hideOutgoingFrame()
      updatePreparedFrame(null)
      setFrameReady(false)
    }

    return () => {
      alive = false
      activeMediaRef.current = false
      activeSrcRef.current = undefined
      if (mediaGenerationRef.current === mediaGeneration) mediaGenerationRef.current += 1
      clearHideTimer()
      const preservingPreparedSource = Boolean(preparedSrcRef.current
        && preparedSrcRef.current === desiredSrcRef.current)
      cancelFrameReadyCallback()
      if (!video.paused) {
        ignoreCleanupPauseRef.current = true
        video.pause()
      }
      if (!preservingPreparedSource) {
        video.removeAttribute('src')
        video.load()
      }
    }
  }, [cancelFrameReadyCallback, clearHideTimer, hideOutgoingFrame, initialTime, markFrameReady, seekToInitialTime, src, updatePreparedFrame])
  /* eslint-enable react-hooks/set-state-in-effect */

  useEffect(() => () => {
    clearHideTimer()
    pendingPreparationRef.current?.cancel('cancelled')
    pendingPreparationRef.current = null
    preparationGenerationRef.current += 1
    preparedSrcRef.current = undefined
    stagedNavigationRef.current = null
    cancelFrameReadyCallback()
    const frames = [preparedFrameRef.current].filter((frame): frame is FrozenFrame => Boolean(frame))
    preparedFrameRef.current = null
    frames.forEach(frame => URL.revokeObjectURL(frame.url))
  }, [cancelFrameReadyCallback, clearHideTimer])

  const seek = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    const video = videoRef.current
    const nextTime = Number(event.currentTarget.value)
    if (!video || !activeMediaRef.current || !Number.isFinite(nextTime)) return
    video.currentTime = nextTime
    setCurrentTime(nextTime)
  }, [])

  const togglePlayback = useCallback(() => {
    const video = videoRef.current
    if (!video || !activeMediaRef.current) return
    if (!video.paused) {
      video.pause()
      setIsPlaying(false)
      controlsVisibleRef.current = true
      setControlsVisible(true)
      clearHideTimer()
      return
    }

    controlsVisibleRef.current = true
    setControlsVisible(true)
    const mediaGeneration = mediaGenerationRef.current
    try {
      const result = video.play()
      void Promise.resolve(result).then(() => {
        if (!activeMediaRef.current || mediaGenerationRef.current !== mediaGeneration) return
        setIsPlaying(!video.paused)
        setIsMuted(video.muted)
        setAutoplayFallback(video.muted && !userMutedRef.current ? 'sound' : null)
        setMediaLoadError(false)
        if (!video.paused) scheduleHide()
      }).catch(() => {
        if (!activeMediaRef.current || mediaGenerationRef.current !== mediaGeneration) return
        setIsPlaying(false)
        setAutoplayFallback('play')
      })
    } catch {
      setIsPlaying(false)
      setAutoplayFallback('play')
    }
  }, [clearHideTimer, scheduleHide])

  useImperativeHandle(ref, () => ({ revealControls, togglePlayback, prepareSource, cancelPreparedSource }),
    [cancelPreparedSource, prepareSource, revealControls, togglePlayback])

  const toggleMute = useCallback(() => {
    const video = videoRef.current
    if (!video || !activeMediaRef.current) return
    if (!video.muted) {
      userMutedRef.current = true
      setPreferredMuted(true)
      video.muted = true
      setIsMuted(true)
      setAutoplayFallback(null)
      return
    }

    const mediaGeneration = mediaGenerationRef.current
    userMutedRef.current = false
    setPreferredMuted(false)
    video.muted = false
    setIsMuted(false)
    if (video.paused) {
      setAutoplayFallback(null)
      return
    }
    try {
      const result = video.play()
      void Promise.resolve(result).then(() => {
        if (!activeMediaRef.current || mediaGenerationRef.current !== mediaGeneration) return
        setIsMuted(video.muted)
        setAutoplayFallback(null)
      }).catch(() => {
        if (!activeMediaRef.current || mediaGenerationRef.current !== mediaGeneration) return
        video.muted = true
        setIsMuted(true)
        setAutoplayFallback('sound')
      })
    } catch {
      video.muted = true
      setIsMuted(true)
      setAutoplayFallback('sound')
    }
  }, [])

  const playFromFallback = useCallback(() => {
    const video = videoRef.current
    if (!video || !activeMediaRef.current) return
    const mediaGeneration = mediaGenerationRef.current
    video.muted = userMutedRef.current
    setIsMuted(userMutedRef.current)
    try {
      const result = video.play()
      void Promise.resolve(result).then(() => {
        if (!activeMediaRef.current || mediaGenerationRef.current !== mediaGeneration) return
        setIsPlaying(!video.paused)
        setAutoplayFallback(video.muted && !userMutedRef.current ? 'sound' : null)
      }).catch(async () => {
        if (!activeMediaRef.current || mediaGenerationRef.current !== mediaGeneration) return
        video.muted = true
        setIsMuted(true)
        try {
          await video.play()
          if (!activeMediaRef.current || mediaGenerationRef.current !== mediaGeneration) return
          setIsPlaying(!video.paused)
          setAutoplayFallback(userMutedRef.current ? null : 'sound')
          setMediaLoadError(false)
        } catch {
          if (!activeMediaRef.current || mediaGenerationRef.current !== mediaGeneration) return
          setIsPlaying(false)
          setAutoplayFallback('play')
        }
      })
    } catch {
      video.muted = true
      setIsMuted(true)
      setAutoplayFallback('play')
    }
  }, [])

  const unmuteFromFallback = useCallback(() => {
    const video = videoRef.current
    if (!video || !activeMediaRef.current) return
    const mediaGeneration = mediaGenerationRef.current
    userMutedRef.current = false
    setPreferredMuted(false)
    video.muted = false
    setIsMuted(false)
    if (video.paused) {
      setAutoplayFallback(null)
      return
    }
    try {
      const result = video.play()
      void Promise.resolve(result).then(() => {
        if (!activeMediaRef.current || mediaGenerationRef.current !== mediaGeneration) return
        setAutoplayFallback(null)
        setIsMuted(video.muted)
      }).catch(() => {
        if (!activeMediaRef.current || mediaGenerationRef.current !== mediaGeneration) return
        video.muted = true
        setIsMuted(true)
        setAutoplayFallback('sound')
      })
    } catch {
      video.muted = true
      setIsMuted(true)
      setAutoplayFallback('sound')
    }
  }, [])

  const visibleFallback = autoplayFallback && !controlsVisible
  const visibleFrozenFrame = preparedFrame?.src === src ? preparedFrame : null

  return (
    <div
      className="relative h-full w-full min-h-0 overflow-hidden bg-black"
      onPointerDown={markPointerInteraction}
      onKeyDownCapture={() => { pointerInteractionRef.current = false }}
    >
      <video
        ref={videoRef}
        data-gallery-video-active={Boolean(src)}
        data-gallery-video-ready={frameReady ? 'true' : 'false'}
        aria-label={name || 'Gallery video'}
        className="h-full w-full object-contain"
        controls={false}
        playsInline
        loop
        preload="auto"
        tabIndex={-1}
        onLoadedMetadata={handleLoadedMetadata}
        onLoadedData={handleLoadedData}
        onSeeked={handleSeeked}
        onTimeUpdate={event => {
          if (activeMediaRef.current) setCurrentTime(event.currentTarget.currentTime)
        }}
        onDurationChange={event => {
          if (activeMediaRef.current) setDuration(Number.isFinite(event.currentTarget.duration) ? event.currentTarget.duration : 0)
        }}
        onPlay={handlePlay}
        onPause={handlePause}
        onVolumeChange={event => {
          if (activeMediaRef.current) setIsMuted(event.currentTarget.muted)
        }}
        onError={handleError}
      />

      <canvas
        ref={outgoingCanvasRef}
        aria-hidden="true"
        data-gallery-video-frozen-frame
        className="pointer-events-none absolute inset-0 z-10 h-full w-full select-none object-contain"
        style={{ visibility: outgoingFrameVisible ? 'visible' : 'hidden' }}
      />

      {visibleFrozenFrame && (
        <img
          src={visibleFrozenFrame.url}
          alt=""
          draggable={false}
          data-gallery-video-prepared-frame
          className="pointer-events-none absolute inset-0 z-10 h-full w-full select-none object-contain"
        />
      )}

      <button
        type="button"
        className="sr-only focus:not-sr-only focus:absolute focus:left-3 focus:top-3 focus:z-30 focus:rounded-full focus:bg-black/80 focus:px-3 focus:py-2 focus:text-xs focus:text-white focus:ring-2 focus:ring-white"
        aria-label={`Show video controls for ${name}`}
        onFocus={() => {
          revealTriggerFocusedRef.current = true
          revealControls()
          clearHideTimer()
        }}
        onBlur={() => {
          revealTriggerFocusedRef.current = false
          scheduleHide()
        }}
        onClick={revealControls}
        onKeyDown={stopGalleryArrowNavigation}
      >
        Show video controls
      </button>

      {visibleFallback && (
        <div
          className="absolute z-20"
          style={{
            bottom: 'max(0.75rem, env(safe-area-inset-bottom))',
            right: 'max(0.75rem, env(safe-area-inset-right))',
          }}
          data-gallery-gesture-ignore
          onPointerDown={handleControlsPointerDown}
          onPointerUp={stopPropagation}
          onClick={stopPropagation}
        >
          {mediaLoadError && <span className="mb-1 block rounded bg-black/75 px-2.5 py-1.5 text-right text-[11px] text-white/85">Video could not be loaded.</span>}
          {autoplayFallback === 'sound' ? (
            <button
              type="button"
              aria-label="Tap for sound"
              className="flex h-9 items-center gap-2 rounded-full border border-white/20 bg-black/70 px-3 text-xs font-medium text-white shadow-lg backdrop-blur-sm hover:bg-black/85 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-white"
              onClick={() => {
                revealControls()
                unmuteFromFallback()
              }}
            >
              <Volume2 size={15} />
              Tap for sound
            </button>
          ) : (
            <button
              type="button"
              aria-label="Play video"
              className="flex h-9 items-center gap-2 rounded-full border border-white/20 bg-black/70 px-3 text-xs font-medium text-white shadow-lg backdrop-blur-sm hover:bg-black/85 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-white"
              onClick={() => {
                revealControls()
                playFromFallback()
              }}
            >
              <Play size={15} />
              Play video
            </button>
          )}
        </div>
      )}

      <div
        data-gallery-gesture-ignore
        data-gallery-playback-controls
        aria-hidden={!controlsVisible}
        inert={!controlsVisible}
        className={`absolute z-20 transition-opacity duration-150 ${controlsVisible ? 'pointer-events-auto opacity-100' : 'pointer-events-none opacity-0'}`}
        style={{
          bottom: 'max(0.75rem, env(safe-area-inset-bottom))',
          left: 'max(0.75rem, env(safe-area-inset-left))',
          right: 'max(0.75rem, env(safe-area-inset-right))',
        }}
        onPointerDown={handleControlsPointerDown}
        onPointerUp={stopPropagation}
        onPointerCancel={stopPropagation}
        onClick={stopPropagation}
        onKeyDown={stopGalleryArrowNavigation}
        onFocus={keepControlsVisible}
        onBlur={releaseControlsFocus}
      >
        <div className="flex min-h-12 items-center gap-2 rounded-xl border border-white/15 bg-black/70 px-2.5 py-2 text-white shadow-lg backdrop-blur-sm sm:gap-3 sm:px-3">
          <button
            type="button"
            aria-label={isPlaying ? 'Pause video' : 'Play video'}
            title={isPlaying ? 'Pause video' : 'Play video'}
            tabIndex={controlsVisible ? 0 : -1}
            className="grid h-8 w-8 shrink-0 place-items-center rounded-full text-white/90 hover:bg-white/10 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-white"
            onFocus={keepControlsVisible}
            onBlur={releaseControlsFocus}
            onClick={togglePlayback}
          >
            {isPlaying ? <Pause size={17} /> : <Play size={17} />}
          </button>

          <span className="w-10 shrink-0 text-right text-[11px] tabular-nums text-white/75">{formatTime(currentTime)}</span>
          <input
            type="range"
            min={0}
            max={duration > 0 ? duration : 0}
            step={0.01}
            value={duration > 0 ? Math.min(currentTime, duration) : 0}
            disabled={duration <= 0}
            aria-label="Video position"
            aria-valuetext={`${formatTime(currentTime)} of ${formatTime(duration)}`}
            tabIndex={controlsVisible ? 0 : -1}
            className="h-1.5 min-w-0 flex-1 cursor-pointer accent-white disabled:cursor-default disabled:opacity-50"
            onFocus={keepControlsVisible}
            onBlur={releaseControlsFocus}
            onPointerDown={() => setScrubbing(true)}
            onPointerUp={() => setScrubbing(false)}
            onPointerCancel={() => setScrubbing(false)}
            onChange={seek}
          />
          <span className="w-10 shrink-0 text-[11px] tabular-nums text-white/75">{formatTime(duration)}</span>

          <button
            type="button"
            aria-label={isMuted ? 'Unmute video' : 'Mute video'}
            title={isMuted ? 'Unmute video' : 'Mute video'}
            tabIndex={controlsVisible ? 0 : -1}
            className="grid h-8 w-8 shrink-0 place-items-center rounded-full text-white/90 hover:bg-white/10 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-white"
            onFocus={keepControlsVisible}
            onBlur={releaseControlsFocus}
            onClick={toggleMute}
          >
            {isMuted ? <VolumeX size={17} /> : <Volume2 size={17} />}
          </button>
        </div>
      </div>
    </div>
  )
})
