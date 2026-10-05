import { useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react'
import { Pause, Play } from 'lucide-react'
import type { GenerationPreview as GenerationPreviewData } from '../../types'

export function GenerationPreview({
  preview,
  initiallyPaused = false,
  onPausedChange,
}: {
  preview: GenerationPreviewData
  initiallyPaused?: boolean
  onPausedChange?: (paused: boolean) => void
}) {
  const videoRef = useRef<HTMLVideoElement>(null)
  const playbackRequest = useRef(0)
  const playbackAttemptIdentityRef = useRef<string | null>(null)
  const mountedRef = useRef(false)
  const [paused, setPaused] = useState(initiallyPaused)
  const [blockedPreviewIdentity, setBlockedPreviewIdentity] = useState<string | null>(null)
  const [failedPreview, setFailedPreview] = useState<string | null>(null)
  const previewIdentity = `${preview.kind}:${preview.url}:${preview.revision}`
  const previewIdentityRef = useRef(previewIdentity)
  const playbackPaused = paused || blockedPreviewIdentity === previewIdentity
  const mediaError = failedPreview === previewIdentity

  const setVideoRef = useCallback((video: HTMLVideoElement | null) => {
    videoRef.current = video
    if (video) video.defaultMuted = true
  }, [])
  const invalidatePlaybackRequest = useCallback(() => {
    playbackRequest.current++
  }, [])

  useLayoutEffect(() => {
    previewIdentityRef.current = previewIdentity
  }, [previewIdentity])

  useEffect(() => {
    mountedRef.current = true
    return () => {
      mountedRef.current = false
      invalidatePlaybackRequest()
      playbackAttemptIdentityRef.current = null
    }
  }, [invalidatePlaybackRequest])

  const requestPlayback = useCallback((video: HTMLVideoElement, identity: string) => {
    const requestId = ++playbackRequest.current
    const isCurrentRequest = () => (
      requestId === playbackRequest.current
      && identity === previewIdentityRef.current
      && video === videoRef.current
      && mountedRef.current
    )
    const handleFailure = (error: unknown) => {
      if (!isCurrentRequest()) return
      const name = error && typeof error === 'object' && 'name' in error
        ? String((error as {name?: unknown}).name ?? '')
        : ''
      if (name === 'NotAllowedError' || name === 'AbortError') {
        // Keep the media element and its Play control available so a tap can
        // retry from a user gesture. This state belongs only to this revision.
        setBlockedPreviewIdentity(identity)
        return
      }
      setFailedPreview(identity)
    }

    try {
      // Invoke play synchronously so this same function can safely run inside
      // the Play button's gesture handler, even on Safari.
      void Promise.resolve(video.play()).catch(handleFailure)
    } catch (error) {
      handleFailure(error)
    }
  }, [])

  useEffect(() => {
    const video = videoRef.current
    if (!video || preview.kind !== 'video') return

    if (playbackPaused) {
      invalidatePlaybackRequest()
      video.pause()
      return
    }

    // Start as soon as the element exists. Waiting for loadeddata can stall on
    // iOS, where muted inline autoplay is allowed before data is ready.
    if (playbackAttemptIdentityRef.current === previewIdentity) return
    playbackAttemptIdentityRef.current = previewIdentity
    requestPlayback(video, previewIdentity)
  }, [invalidatePlaybackRequest, playbackPaused, preview.kind, preview.revision, preview.url, previewIdentity, requestPlayback])

  const togglePlayback = () => {
    const video = videoRef.current
    if (!video) return
    if (playbackPaused) {
      onPausedChange?.(false)
      setPaused(false)
      setBlockedPreviewIdentity(null)
      playbackAttemptIdentityRef.current = previewIdentity
      // This call stays directly in the gesture handler for Safari's playback
      // policy; the effect above skips this already-started attempt.
      requestPlayback(video, previewIdentity)
    } else {
      onPausedChange?.(true)
      // A late play() resolution must not undo the user's pause.
      invalidatePlaybackRequest()
      video.pause()
      setPaused(true)
    }
  }

  return (
    <div className="absolute inset-0 overflow-hidden bg-black" data-generation-preview="true">
      {!mediaError && preview.kind === 'image' && (
        <img
          src={preview.url}
          alt={`Generation preview, clip ${preview.clip}, window ${preview.window}`}
          className="h-full w-full object-contain"
          onError={() => setFailedPreview(previewIdentity)}
        />
      )}
      {!mediaError && preview.kind === 'video' && (
        <video
          ref={setVideoRef}
          src={preview.url}
          muted
          loop
          playsInline
          autoPlay={!playbackPaused}
          preload="auto"
          aria-label={`Generation preview, clip ${preview.clip}, window ${preview.window}`}
          title={playbackPaused ? 'Click to resume live preview' : 'Click to pause live preview'}
          className="h-full w-full cursor-pointer object-contain"
          onClick={togglePlayback}
          onError={() => setFailedPreview(previewIdentity)}
        />
      )}

      {mediaError ? (
        <div className="absolute inset-0 flex items-center justify-center px-4 text-center text-xs text-text-muted" role="status">
          Preview unavailable
        </div>
      ) : (
        <>
          <div className="absolute left-2 top-2 rounded bg-black/65 px-2 py-1 text-[10px] text-white">
            Clip {preview.clip}/{preview.total_clips} · Window {preview.window}/{preview.total_windows}
          </div>
          {preview.kind === 'video' && (
            <button
              type="button"
              onClick={event => { event.stopPropagation(); togglePlayback() }}
              aria-label={playbackPaused ? 'Play live preview' : 'Pause live preview'}
              title={playbackPaused ? 'Play live preview' : 'Pause live preview'}
              className="absolute right-2 top-2 rounded-full bg-black/70 p-2 text-white hover:bg-black/90 focus-visible:outline focus-visible:outline-2 focus-visible:outline-white"
            >
              {playbackPaused ? <Play size={14} /> : <Pause size={14} />}
            </button>
          )}
        </>
      )}
    </div>
  )
}
