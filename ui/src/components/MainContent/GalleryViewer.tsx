import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState, type KeyboardEvent as ReactKeyboardEvent } from 'react'
import { createPortal } from 'react-dom'
import { ChevronDown, Heart, Loader2, SlidersHorizontal, X } from 'lucide-react'
import type { OutputFile } from '../../types'
import { outputIdentity } from '../../lib/galleryIdentity'
import { ImageComparison, type GalleryImageChoice } from './ImageComparison'
import { dismissGalleryFullscreenHelp, type createGalleryViewerSurface } from '../../lib/galleryFullscreen'
import { GallerySwipeDeck, type GallerySwipeDeckHandle } from './GallerySwipeDeck'
import { GalleryVideoPlayer, type GalleryVideoPlayerHandle } from './GalleryVideoPlayer'
import { GalleryZoomImage } from './GalleryZoomImage'
import { getVideoPosterUrl, requestThumbnail, type VideoPosterSize } from '../../lib/thumbnailCache'
import { useVideoPosterSize } from '../../lib/useVideoPosterSize'

export type { GalleryImageChoice } from './ImageComparison'

export interface GalleryViewerProps {
  surface: ReturnType<typeof createGalleryViewerSurface>
  /** The stable gallery session, containing only image and video outputs. */
  items: OutputFile[]
  /** outputIdentity() for the output that opened the viewer. */
  initialId: string
  initialCompare?: boolean
  /** Initial playback position in seconds; applied only to initialId. */
  initialTime?: number
  /** Ordered image inputs from the active output's sidecar. */
  sourceImages: GalleryImageChoice[]
  /** Images in the current gallery, with workspace-aware display names. */
  comparisonImages: GalleryImageChoice[]
  /** False for the Uploads view. */
  allowFavorite: boolean
  onFavorite: (file: OutputFile) => Promise<void>
  onClose: (activeId: string) => void
  /** Optional pagination for galleries that can load more outputs. */
  hasMore?: boolean
  onLoadMore?: () => Promise<void>
}

const KEYBOARD_IGNORE = 'input,select,textarea,video,audio,[contenteditable="true"],[role="slider"]'

function GalleryPreview({ file, posterSize, onThumbnailReady }: {
  file: OutputFile
  posterSize: VideoPosterSize | null
  onThumbnailReady?: (identity: string, thumbnail: string) => void
}) {
  const [thumbnail, setThumbnail] = useState<string | null>(null)
  useEffect(() => {
    if (file.type !== 'video' || !posterSize) return
    let active = true
    const identity = outputIdentity(file)
    void requestThumbnail(file.url, file.url, posterSize).then(url => {
      if (active) setThumbnail(url)
      if (url) onThumbnailReady?.(identity, url)
    })
    return () => { active = false }
  }, [file, file.type, file.url, posterSize, onThumbnailReady])
  const url = file.type === 'image' ? file.url : thumbnail
  return <div className="flex h-full w-full items-center justify-center overflow-hidden bg-black">
    {url ? <img src={url} alt="" draggable={false} className="h-full w-full select-none object-contain" />
      : <span className="px-5 text-center text-sm text-white/60">{file.name}</span>}
  </div>
}

function isFocusTarget(element: Element): element is HTMLElement {
  return element instanceof HTMLElement
    && !element.hasAttribute('disabled')
    && element.tabIndex >= 0
    && !element.closest('[inert],[aria-hidden="true"]')
    && element.getClientRects().length > 0
}

export function GalleryViewer({
  surface,
  items,
  initialId,
  initialCompare = false,
  initialTime,
  sourceImages,
  comparisonImages,
  allowFavorite,
  onFavorite,
  onClose,
  hasMore = false,
  onLoadMore,
}: GalleryViewerProps) {
  const viewerRef = useRef<HTMLDivElement>(null)
  const posterSize = useVideoPosterSize(viewerRef, initialId)
  const swipeDeckRef = useRef<GallerySwipeDeckHandle>(null)
  const playerRef = useRef<GalleryVideoPlayerHandle>(null)
  const lastComparisonImageRef = useRef<GalleryImageChoice | null>(null)
  const previousFocusRef = useRef<HTMLElement | null>(null)
  const focusReturnFrameRef = useRef<number | null>(null)
  const loadMoreInFlightRef = useRef(false)
  const lastLoadAttemptLengthRef = useRef<number | null>(null)
  const pendingNextFromRef = useRef<number | null>(null)
  const pendingNextOriginRef = useRef<string | null>(null)
  const pendingNextAutomaticRef = useRef(false)
  const imageAdvanceTimerRef = useRef<number | null>(null)
  const autoAdvanceRef = useRef(false)
  const automaticAdvanceRef = useRef(false)
  const activeIdentityRef = useRef('')
  const overlayPointerRef = useRef(false)
  const previewTargetIdsRef = useRef<Set<string>>(new Set())

  const galleryItems = useMemo(() => items.filter(item => item.type === 'image' || item.type === 'video'), [items])
  const portalHost = surface.host
  const [activeId, setActiveId] = useState(initialId)
  const [compareMode, setCompareMode] = useState(initialCompare)
  const [favoriteBusy, setFavoriteBusy] = useState(false)
  const [favoriteError, setFavoriteError] = useState('')
  const [mediaLoadError, setMediaLoadError] = useState('')
  const [fullscreenError, setFullscreenError] = useState('')
  const [loadingMore, setLoadingMore] = useState(false)
  const [loadMoreError, setLoadMoreError] = useState('')
  const [imageInteraction, setImageInteraction] = useState({ id: '', blocked: false })
  const [previewThumbnails, setPreviewThumbnails] = useState<Record<string, string>>({})
  const [readyVideoSrc, setReadyVideoSrc] = useState('')
  const [autoAdvance, setAutoAdvance] = useState(false)
  const [imageDuration, setImageDuration] = useState(3)
  const [loadedImageKey, setLoadedImageKey] = useState('')
  const [tabVisible, setTabVisible] = useState(() => typeof document === 'undefined' || !document.hidden)
  const [deckBusy, setDeckBusy] = useState(false)
  const [autoAdvanceWaiting, setAutoAdvanceWaiting] = useState(false)
  const [endedVideoSrc, setEndedVideoSrc] = useState<string | null>(null)
  const [videoControls, setVideoControls] = useState({ src: '', visible: false })
  const [pinnedControlsSrc, setPinnedControlsSrc] = useState('')

  const activeIndex = galleryItems.findIndex(item => outputIdentity(item) === activeId)
  const resolvedIndex = activeIndex >= 0 ? activeIndex : 0
  const currentItem = galleryItems[resolvedIndex]
  const currentIdentity = currentItem ? outputIdentity(currentItem) : activeId
  const mediaKey = currentItem ? `${currentIdentity}\u0000${currentItem.url}\u0000${currentItem.type}` : ''
  const viewerControlsVisible = currentItem?.type !== 'video'
    || (videoControls.src === currentItem.url && videoControls.visible)
  const overlayVisibility = viewerControlsVisible ? 'pointer-events-auto opacity-100' : 'pointer-events-none opacity-0'
  autoAdvanceRef.current = autoAdvance
  activeIdentityRef.current = currentIdentity
  const currentVideoPoster = currentItem?.type === 'video'
    ? (posterSize && getVideoPosterUrl(currentItem.url, posterSize)) || previewThumbnails[currentIdentity] || null
    : null
  const hasPrevious = resolvedIndex > 0
  const hasNext = resolvedIndex >= 0 && resolvedIndex < galleryItems.length - 1
  previewTargetIdsRef.current = new Set([
    ...(currentItem ? [currentIdentity] : []),
    ...(hasPrevious ? [outputIdentity(galleryItems[resolvedIndex - 1])] : []),
    ...(hasNext ? [outputIdentity(galleryItems[resolvedIndex + 1])] : []),
  ])

  const clearImageAdvanceTimer = useCallback(() => {
    if (imageAdvanceTimerRef.current === null) return
    window.clearTimeout(imageAdvanceTimerRef.current)
    imageAdvanceTimerRef.current = null
  }, [])

  const cancelPendingAdvance = useCallback((automaticOnly = false, cancelMotion = false) => {
    const pendingAutomatic = pendingNextAutomaticRef.current
    const activeAutomatic = automaticAdvanceRef.current
    if (automaticOnly && !pendingAutomatic && !activeAutomatic) return

    if (!automaticOnly || pendingAutomatic) {
      pendingNextFromRef.current = null
      pendingNextOriginRef.current = null
      pendingNextAutomaticRef.current = false
    }
    if (pendingAutomatic || activeAutomatic) {
      automaticAdvanceRef.current = false
      setAutoAdvanceWaiting(false)
      if (cancelMotion) {
        swipeDeckRef.current?.cancelGesture()
        playerRef.current?.cancelPreparedSource()
      }
    }
  }, [])

  const comparisonVisible = compareMode && currentItem?.type === 'image'
  const imageGestureBlocked = currentItem?.type === 'image' && imageInteraction.id === mediaKey && imageInteraction.blocked
  const handleImageInteraction = useCallback((blocked: boolean) => {
    if (blocked) {
      clearImageAdvanceTimer()
      cancelPendingAdvance(false, true)
      swipeDeckRef.current?.cancelGesture()
    }
    setImageInteraction(previous => previous.id === mediaKey && previous.blocked === blocked ? previous : { id: mediaKey, blocked })
  }, [cancelPendingAdvance, clearImageAdvanceTimer, mediaKey])
  const rememberPreviewThumbnail = useCallback((identity: string, thumbnail: string) => {
    if (!previewTargetIdsRef.current.has(identity)) return
    setPreviewThumbnails(current => {
      if (current[identity] === thumbnail) return current
      const entries = Object.entries(current).filter(([key]) => key !== identity)
      entries.push([identity, thumbnail])
      return Object.fromEntries(entries.slice(-3))
    })
  }, [])
  const handleDeckBusyChange = useCallback((busy: boolean) => {
    setDeckBusy(current => current === busy ? current : busy)
  }, [])
  const handleVideoFrameReady = useCallback((src: string) => setReadyVideoSrc(src), [])
  const handleControlsVisibility = useCallback((src: string, visible: boolean) => {
    setVideoControls(current => current.src === src && current.visible === visible ? current : { src, visible })
  }, [])

  const revealViewerControls = () => {
    if (currentItem?.type === 'video') playerRef.current?.revealControls()
  }
  const overlayInteraction = {
    onPointerDown: (event: React.PointerEvent<HTMLDivElement>) => {
      overlayPointerRef.current = true
      setPinnedControlsSrc(event.target instanceof Element && event.target.closest('select') ? currentItem?.url ?? '' : '')
      revealViewerControls()
    },
    onKeyDown: () => {
      overlayPointerRef.current = false
      setPinnedControlsSrc(currentItem?.url ?? '')
      revealViewerControls()
    },
    onFocus: (event: React.FocusEvent<HTMLDivElement>) => {
      if (event.target instanceof Element && (event.target.matches('select')
        || (!overlayPointerRef.current && event.target.matches(':focus-visible')))) {
        setPinnedControlsSrc(currentItem?.url ?? '')
      }
      revealViewerControls()
    },
    onBlur: (event: React.FocusEvent<HTMLDivElement>) => {
      if (!(event.relatedTarget instanceof Node) || !event.currentTarget.contains(event.relatedTarget)) setPinnedControlsSrc('')
    },
  }

  useEffect(() => {
    if (!viewerControlsVisible && document.activeElement === document.body) viewerRef.current?.focus()
  }, [viewerControlsVisible])

  useEffect(() => {
    const updateVisibility = () => setTabVisible(!document.hidden)
    document.addEventListener('visibilitychange', updateVisibility)
    return () => document.removeEventListener('visibilitychange', updateVisibility)
  }, [])

  useLayoutEffect(() => {
    const viewer = viewerRef.current
    const viewport = window.visualViewport
    if (!viewer || !viewport) return
    const resize = () => {
      // Safari's browser bars can cover part of the layout viewport. Keep the
      // whole deck, including its bottom controls, in the visible viewport.
      // Leave native pinch zoom alone rather than fitting the image again.
      if (viewport.scale > 1) return
      Object.assign(viewer.style, {
        height: `${viewport.height}px`, width: `${viewport.width}px`,
        top: `${viewport.offsetTop}px`, left: `${viewport.offsetLeft}px`,
      })
    }
    resize()
    viewport.addEventListener('resize', resize)
    viewport.addEventListener('scroll', resize)
    window.addEventListener('resize', resize)
    return () => {
      viewport.removeEventListener('resize', resize)
      viewport.removeEventListener('scroll', resize)
      window.removeEventListener('resize', resize)
      for (const property of ['height', 'width', 'top', 'left']) viewer.style.removeProperty(property)
    }
  }, [])

  useEffect(() => {
    if (focusReturnFrameRef.current !== null) {
      window.cancelAnimationFrame(focusReturnFrameRef.current)
      focusReturnFrameRef.current = null
    }
    previousFocusRef.current = surface.returnFocus
    // Child effects may already have started the fullscreen player. Pause only
    // the underlying gallery, never the player inside this viewer's surface.
    document.querySelectorAll<HTMLMediaElement>('video, audio').forEach(media => {
      if (!surface.host.contains(media)) media.pause()
    })
    return () => {
      const previousFocus = previousFocusRef.current
      previousFocusRef.current = null
      if (previousFocus) {
        focusReturnFrameRef.current = window.requestAnimationFrame(() => {
          focusReturnFrameRef.current = null
          previousFocus.focus({ preventScroll: true })
        })
      }
    }
  }, [surface])

  useEffect(() => {
    if (!portalHost) return
    const body = document.body
    const root = document.documentElement
    const oldBodyOverflow = body.style.overflow
    const oldRootOverflow = root.style.overflow
    const oldBodyTouchAction = body.style.touchAction
    const oldRootTouchAction = root.style.touchAction
    const hiddenSiblings = Array.from(body.children)
      .filter((element): element is HTMLElement => element instanceof HTMLElement && element !== portalHost)
      .map(element => ({
        element,
        inert: element.hasAttribute('inert'),
        inertValue: element.getAttribute('inert'),
        ariaHidden: element.getAttribute('aria-hidden'),
      }))

    body.style.overflow = 'hidden'
    root.style.overflow = 'hidden'
    body.style.touchAction = 'none'
    root.style.touchAction = 'none'
    for (const sibling of hiddenSiblings) {
      sibling.element.setAttribute('inert', '')
      sibling.element.setAttribute('aria-hidden', 'true')
    }

    return () => {
      body.style.overflow = oldBodyOverflow
      root.style.overflow = oldRootOverflow
      body.style.touchAction = oldBodyTouchAction
      root.style.touchAction = oldRootTouchAction
      for (const sibling of hiddenSiblings) {
        if (sibling.inert) sibling.element.setAttribute('inert', sibling.inertValue ?? '')
        else sibling.element.removeAttribute('inert')
        if (sibling.ariaHidden === null) sibling.element.removeAttribute('aria-hidden')
        else sibling.element.setAttribute('aria-hidden', sibling.ariaHidden)
      }
    }
  }, [portalHost])

  useEffect(() => {
    if (!currentItem) return
    if (currentIdentity !== activeId) setActiveId(currentIdentity)
  }, [activeId, currentIdentity, currentItem])

  useEffect(() => {
    setFavoriteError('')
    setMediaLoadError('')
  }, [currentIdentity, mediaKey])

  useEffect(() => {
    viewerRef.current?.focus()
  }, [portalHost])

  useEffect(() => {
    let active = true
    void surface.fullscreenRequest.then(message => {
      if (!active) return
      setFullscreenError(message || '')
    })
    return () => { active = false }
  }, [surface])

  const close = useCallback(() => {
    onClose(currentItem ? outputIdentity(currentItem) : activeId || initialId)
  }, [activeId, currentItem, initialId, onClose])

  useEffect(() => {
    // Touch browsers can leave focus on body after a previously focused media
    // element is hidden. Escape must still dismiss the active modal.
    const handleEscape = (event: KeyboardEvent) => {
      if (event.key !== 'Escape') return
      event.preventDefault()
      close()
    }
    document.addEventListener('keydown', handleEscape)
    return () => document.removeEventListener('keydown', handleEscape)
  }, [close])

  const requestMore = useCallback(async (manual = false, advance = false, automatic = false) => {
    const requestedLength = galleryItems.length
    if (advance) {
      pendingNextFromRef.current = requestedLength
      pendingNextOriginRef.current = currentIdentity
      pendingNextAutomaticRef.current = automatic
    }
    if (!hasMore || !onLoadMore) {
      if (advance) cancelPendingAdvance(automatic)
      return
    }
    if (loadMoreInFlightRef.current) return
    if (!manual && lastLoadAttemptLengthRef.current === requestedLength) {
      if (advance) cancelPendingAdvance(automatic)
      return
    }

    loadMoreInFlightRef.current = true
    lastLoadAttemptLengthRef.current = requestedLength
    setLoadingMore(true)
    setLoadMoreError('')
    try {
      await onLoadMore()
    } catch (error) {
      setLoadMoreError(error instanceof Error ? error.message : 'Could not load more images.')
      if (advance) cancelPendingAdvance(automatic)
    } finally {
      loadMoreInFlightRef.current = false
      setLoadingMore(false)
    }
  }, [cancelPendingAdvance, currentIdentity, galleryItems.length, hasMore, onLoadMore])

  useEffect(() => {
    const requestedLength = pendingNextFromRef.current
    if (requestedLength === null) return
    const automatic = pendingNextAutomaticRef.current
    if (activeIdentityRef.current !== pendingNextOriginRef.current || (automatic && !autoAdvanceRef.current)) {
      cancelPendingAdvance(automatic)
      return
    }
    if (automatic && (!tabVisible || deckBusy || comparisonVisible || imageGestureBlocked)) return
    if (galleryItems.length > requestedLength) {
      pendingNextFromRef.current = null
      pendingNextOriginRef.current = null
      pendingNextAutomaticRef.current = false
      if (!automatic && (comparisonVisible || imageGestureBlocked)) {
        setReadyVideoSrc('')
        setActiveId(outputIdentity(galleryItems[requestedLength]))
        automaticAdvanceRef.current = false
        setAutoAdvanceWaiting(false)
      } else {
        swipeDeckRef.current?.navigate(1)
      }
    } else if (!hasMore && !loadingMore) {
      cancelPendingAdvance(automatic)
    }
  }, [cancelPendingAdvance, comparisonVisible, deckBusy, galleryItems, hasMore, imageGestureBlocked, loadingMore, tabVisible])

  useEffect(() => {
    if (galleryItems.length > 0 && resolvedIndex >= galleryItems.length - 2 && hasMore && onLoadMore) {
      void requestMore()
    }
  }, [galleryItems.length, hasMore, onLoadMore, requestMore, resolvedIndex])

  const navigate = useCallback((direction: -1 | 1, manualLoad = false, automatic = false) => {
    if (!automatic) {
      clearImageAdvanceTimer()
      cancelPendingAdvance(false, true)
      setEndedVideoSrc(null)
    }
    if (galleryItems.length === 0) return
    const nextIndex = resolvedIndex + direction
    if (nextIndex >= 0 && nextIndex < galleryItems.length) {
      if (comparisonVisible || imageGestureBlocked) {
        setReadyVideoSrc('')
        setActiveId(outputIdentity(galleryItems[nextIndex]))
        if (automatic) {
          automaticAdvanceRef.current = false
          setAutoAdvanceWaiting(false)
        }
      } else swipeDeckRef.current?.navigate(direction)
      return
    }
    if (direction > 0 && hasMore && onLoadMore) void requestMore(manualLoad, true, automatic)
    else if (automatic) {
      automaticAdvanceRef.current = false
      setAutoAdvanceWaiting(false)
    }
  }, [cancelPendingAdvance, clearImageAdvanceTimer, comparisonVisible, galleryItems, hasMore, imageGestureBlocked, onLoadMore, requestMore, resolvedIndex])

  const startAutomaticAdvance = useCallback(() => {
    if (!autoAdvanceRef.current || deckBusy || (!hasNext && (!hasMore || !onLoadMore))) return
    automaticAdvanceRef.current = true
    setAutoAdvanceWaiting(true)
    navigate(1, false, true)
  }, [deckBusy, hasMore, hasNext, navigate, onLoadMore])

  const handleVideoEnded = useCallback((src: string) => {
    setEndedVideoSrc(src)
  }, [])

  useEffect(() => {
    if (!endedVideoSrc) return
    if (!autoAdvance || currentItem?.type !== 'video' || currentItem.url !== endedVideoSrc) {
      setEndedVideoSrc(null)
      return
    }
    // The store's pagination flag can update before the viewer receives the
    // appended items. Keep the completed clip pending through that handoff.
    if (!tabVisible || deckBusy || (!hasNext && loadingMore)) return
    const video = viewerRef.current?.querySelector<HTMLVideoElement>('video[data-gallery-video-active="true"]')
    if (!video?.ended) {
      setEndedVideoSrc(null)
      return
    }
    setEndedVideoSrc(null)
    startAutomaticAdvance()
  }, [autoAdvance, currentItem, deckBusy, endedVideoSrc, hasNext, loadingMore, startAutomaticAdvance, tabVisible])

  useEffect(() => {
    if (!autoAdvance || currentItem?.type !== 'image' || loadedImageKey !== mediaKey
      || comparisonVisible || imageGestureBlocked || deckBusy
      || !tabVisible || autoAdvanceWaiting || (!hasNext && !hasMore)) return

    const identity = currentIdentity
    const timer = window.setTimeout(() => {
      if (imageAdvanceTimerRef.current === timer) imageAdvanceTimerRef.current = null
      if (!autoAdvanceRef.current || activeIdentityRef.current !== identity || document.hidden || deckBusy) return
      startAutomaticAdvance()
    }, imageDuration * 1000)
    imageAdvanceTimerRef.current = timer
    return () => {
      window.clearTimeout(timer)
      if (imageAdvanceTimerRef.current === timer) imageAdvanceTimerRef.current = null
    }
  }, [autoAdvance, autoAdvanceWaiting, comparisonVisible, currentIdentity, currentItem, deckBusy, hasMore, hasNext, imageDuration, imageGestureBlocked, loadedImageKey, mediaKey, startAutomaticAdvance, tabVisible])

  const commitNavigation = useCallback((direction: -1 | 1) => {
    const next = galleryItems[resolvedIndex + direction]
    if (next) {
      setReadyVideoSrc('')
      setActiveId(outputIdentity(next))
      automaticAdvanceRef.current = false
      setAutoAdvanceWaiting(false)
    }
  }, [galleryItems, resolvedIndex])

  const prepareNavigation = useCallback(async (direction: -1 | 1) => {
    const next = galleryItems[resolvedIndex + direction]
    if (next?.type !== 'video') return true
    const result = await playerRef.current?.prepareSource(next.url)
    return result !== 'cancelled'
  }, [galleryItems, resolvedIndex])

  const cancelPreparedNavigation = useCallback(() => {
    playerRef.current?.cancelPreparedSource()
    automaticAdvanceRef.current = false
    setAutoAdvanceWaiting(false)
  }, [])

  const handleKeyDown = (event: ReactKeyboardEvent<HTMLDivElement>) => {
    if (event.key === 'Tab') {
      const focusable = viewerRef.current
        ? Array.from(viewerRef.current.querySelectorAll<HTMLElement>('button:not([disabled]),a[href],input:not([disabled]),select:not([disabled]),textarea:not([disabled]),[tabindex]:not([tabindex="-1"])')).filter(isFocusTarget)
        : []
      if (focusable.length === 0) {
        event.preventDefault()
        viewerRef.current?.focus()
        return
      }
      const first = focusable[0]
      const last = focusable[focusable.length - 1]
      if (event.shiftKey && (document.activeElement === first || !viewerRef.current?.contains(document.activeElement))) {
        event.preventDefault()
        last.focus()
      } else if (!event.shiftKey && (document.activeElement === last || !viewerRef.current?.contains(document.activeElement))) {
        event.preventDefault()
        first.focus()
      }
      return
    }

    if (event.target instanceof Element && event.target.closest(KEYBOARD_IGNORE)) return
    if (event.key === 'ArrowUp' || event.key === 'ArrowLeft') {
      event.preventDefault()
      navigate(-1)
    } else if (event.key === 'ArrowDown' || event.key === 'ArrowRight') {
      event.preventDefault()
      navigate(1, true)
    }
  }

  const toggleFavorite = async () => {
    if (!currentItem || !allowFavorite || favoriteBusy) return
    setFavoriteBusy(true)
    setFavoriteError('')
    try {
      await onFavorite(currentItem)
    } catch (error) {
      setFavoriteError(error instanceof Error ? error.message : 'Could not update favorite.')
    } finally {
      setFavoriteBusy(false)
    }
  }

  const toggleAutoAdvance = () => {
    if (autoAdvance) {
      clearImageAdvanceTimer()
      cancelPendingAdvance(true, true)
      setEndedVideoSrc(null)
    }
    setAutoAdvance(!autoAdvance)
  }

  const currentComparisonImage = useMemo<GalleryImageChoice | null>(() => currentItem?.type === 'image'
    ? { id: currentIdentity, name: currentItem.name, url: currentItem.url }
    : null, [currentIdentity, currentItem])
  if (currentComparisonImage) lastComparisonImageRef.current = currentComparisonImage
  const comparisonImage = currentComparisonImage || lastComparisonImageRef.current

  return createPortal(
    <div
      ref={viewerRef}
      role="dialog"
      aria-modal="true"
      aria-label="Gallery viewer"
      aria-describedby="gallery-viewer-description"
      tabIndex={-1}
      className="fixed left-0 top-0 z-[130] flex h-[100dvh] w-full min-h-0 flex-col overflow-hidden bg-black text-white outline-none"
      onKeyDown={handleKeyDown}
    >
      <p id="gallery-viewer-description" className="sr-only" aria-live="polite">
        {currentItem?.name}. {galleryItems.length ? `${resolvedIndex + 1} of ${galleryItems.length}.` : ''}
        Swipe up or down, or use the arrow keys to browse. Tap a video to pause or resume playback.
        Pinch an image to zoom, then drag to pan. Reset zoom to swipe to another item.
      </p>
      <div data-gallery-viewer-actions aria-hidden={!viewerControlsVisible} inert={!viewerControlsVisible}
        {...overlayInteraction}
        className={`absolute z-30 flex items-center gap-2 transition-opacity duration-150 ${overlayVisibility}`}
        style={{ top: 'max(0.75rem, env(safe-area-inset-top))', right: 'max(0.75rem, env(safe-area-inset-right))' }}>
        {currentItem?.type === 'image' && (
          <button
            type="button"
            onClick={() => setCompareMode(value => !value)}
            aria-pressed={compareMode}
            className={`flex h-11 w-11 shrink-0 items-center justify-center rounded-full shadow-md transition-colors focus-visible:ring-2 focus-visible:ring-white ${compareMode ? 'bg-accent-blue text-white' : 'bg-black/40 text-white hover:bg-black/65'}`}
            aria-label={compareMode ? 'Close comparison' : 'Compare images'}
            title={compareMode ? 'Close comparison' : 'Compare images'}
          >
            <SlidersHorizontal size={16} />
          </button>
        )}

        <button
          type="button"
          onClick={() => void toggleFavorite()}
          disabled={!allowFavorite || favoriteBusy || !currentItem}
          aria-pressed={Boolean(currentItem?.favorite)}
          aria-label={currentItem?.favorite ? 'Remove from favorites' : 'Add to favorites'}
          title={allowFavorite ? (currentItem?.favorite ? 'Remove from favorites' : 'Add to favorites') : 'Favorites are unavailable for Uploads'}
          className={`grid h-11 w-11 shrink-0 place-items-center rounded-full bg-black/40 shadow-md transition-colors hover:bg-black/65 focus-visible:ring-2 focus-visible:ring-white disabled:cursor-not-allowed disabled:opacity-35 ${currentItem?.favorite ? 'text-rose-400' : 'text-white'}`}
        >
          {favoriteBusy ? <Loader2 size={20} className="animate-spin" /> : <Heart size={20} fill={currentItem?.favorite ? 'currentColor' : 'none'} />}
        </button>

        <button
          type="button"
          data-gallery-close
          onClick={close}
          className="grid h-11 w-11 shrink-0 place-items-center rounded-full bg-black/40 text-white shadow-md hover:bg-black/65 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-white"
          aria-label="Close viewer"
          title="Close (Esc)"
        >
          <X size={22} />
        </button>
      </div>

      <div data-gallery-auto-advance aria-hidden={!viewerControlsVisible} inert={!viewerControlsVisible}
        {...overlayInteraction}
        className={`absolute z-30 flex items-center gap-1.5 transition-opacity duration-150 ${overlayVisibility}`}
        style={{ top: 'max(0.75rem, env(safe-area-inset-top))', left: 'max(0.75rem, env(safe-area-inset-left))' }}>
        <button
          type="button"
          onClick={toggleAutoAdvance}
          aria-label="Auto advance"
          aria-pressed={autoAdvance}
          title={autoAdvance ? 'Turn off auto advance' : 'Turn on auto advance'}
          className={`h-11 rounded-full px-3 text-xs font-medium shadow-md transition-colors focus-visible:ring-2 focus-visible:ring-white ${autoAdvance ? 'bg-accent-blue text-white' : 'bg-black/40 text-white hover:bg-black/65'}`}
        >
          <span className="sm:hidden">Auto</span><span className="hidden sm:inline">Auto advance</span>
        </button>
        {autoAdvance && (
          <select
            aria-label="Image duration"
            title="How long to show each image"
            value={imageDuration}
            onChange={event => { setImageDuration(Number(event.currentTarget.value)); setPinnedControlsSrc('') }}
            className="h-11 w-16 rounded-full border-0 bg-black/60 px-2 text-xs text-white shadow-md outline-none focus-visible:ring-2 focus-visible:ring-white [&>option]:bg-neutral-900"
          >
            {Array.from({ length: 10 }, (_, index) => index + 1).map(seconds => (
              <option key={seconds} value={seconds}>{seconds}s</option>
            ))}
          </select>
        )}
      </div>

      {favoriteError && <p role="alert" aria-hidden={!viewerControlsVisible} className={`absolute right-3 z-40 rounded-md bg-rose-950/90 px-3 py-2 text-xs text-rose-100 transition-opacity duration-150 ${overlayVisibility}`} style={{ top: 'calc(max(0.75rem, env(safe-area-inset-top)) + 3.25rem)' }}>{favoriteError}</p>}
      {fullscreenError && <div role="status" aria-hidden={!viewerControlsVisible} inert={!viewerControlsVisible}
        {...overlayInteraction}
        className={`absolute inset-x-3 z-30 mx-auto flex max-w-md items-start gap-2 rounded-lg bg-black/75 px-3 py-2 text-xs text-white/80 transition-opacity duration-150 ${overlayVisibility}`} style={{ top: 'calc(max(0.75rem, env(safe-area-inset-top)) + 3.25rem)' }}>
        <p className="flex-1">{fullscreenError}</p>
        <button type="button" aria-label="Dismiss fullscreen help" className="shrink-0 p-1" onClick={() => {
          dismissGalleryFullscreenHelp(fullscreenError)
          setFullscreenError('')
        }}><X size={14} /></button>
      </div>}

      <main className="relative flex min-h-0 flex-1 flex-col overflow-hidden">
        {compareMode && comparisonImage && (
          <div
            className={currentItem?.type === 'image' ? 'flex min-h-0 flex-1 flex-col' : 'hidden'}
            aria-hidden={currentItem?.type !== 'image'}
            style={{ paddingTop: 'calc(max(0.75rem, env(safe-area-inset-top)) + 3.25rem)', paddingBottom: 'env(safe-area-inset-bottom)' }}
          >
            <ImageComparison
              currentImage={comparisonImage}
              sourceImages={sourceImages}
              comparisonImages={comparisonImages}
            />
          </div>
        )}
        <div className={comparisonVisible ? 'hidden' : 'flex min-h-0 flex-1 flex-col'} aria-hidden={comparisonVisible}>
        <GallerySwipeDeck ref={swipeDeckRef} activeId={currentIdentity} enabled={!comparisonVisible && !imageGestureBlocked}
          onNavigate={commitNavigation}
          onPrepareNavigate={prepareNavigation}
          onCancelNavigation={cancelPreparedNavigation}
          onBusyChange={handleDeckBusyChange}
          onTap={() => { if (currentItem?.type === 'video') playerRef.current?.togglePlayback() }}
          previous={hasPrevious ? <GalleryPreview key={outputIdentity(galleryItems[resolvedIndex - 1])} file={galleryItems[resolvedIndex - 1]} posterSize={posterSize} onThumbnailReady={rememberPreviewThumbnail} /> : undefined}
          next={hasNext ? <GalleryPreview key={outputIdentity(galleryItems[resolvedIndex + 1])} file={galleryItems[resolvedIndex + 1]} posterSize={posterSize} onThumbnailReady={rememberPreviewThumbnail} /> : undefined}>
        {/* Safari authorizes sound per media element. Keep this player mounted
            through video and image navigation, unloading its source on images. */}
        <div className={currentItem?.type === 'video' ? 'relative h-full w-full min-h-0' : 'hidden'}
          aria-hidden={currentItem?.type !== 'video'} inert={currentItem?.type !== 'video'}>
          <GalleryVideoPlayer ref={playerRef} src={currentItem?.type === 'video' ? currentItem.url : undefined} name={currentItem?.name ?? ''}
            initialTime={currentIdentity === initialId ? initialTime : undefined} loop={!autoAdvance}
            onFrameReady={handleVideoFrameReady} onEnded={handleVideoEnded}
            controlsPinned={currentItem?.type === 'video' && pinnedControlsSrc === currentItem.url}
            onControlsVisibilityChange={handleControlsVisibility} />
          {currentItem?.type === 'video' && readyVideoSrc !== currentItem.url && (
            <div data-gallery-video-preview={currentIdentity}
              className="pointer-events-none absolute inset-0 z-10 flex items-center justify-center overflow-hidden bg-black">
              {currentVideoPoster
                ? <img src={currentVideoPoster} alt="" draggable={false} className="h-full w-full select-none object-contain" />
                : <span className="px-5 text-center text-sm text-white/60">{currentItem.name}</span>}
            </div>
          )}
        </div>
        {currentItem?.type === 'image' ? (
          <div className="flex min-h-0 flex-1 items-center justify-center overflow-hidden bg-black">
            <GalleryZoomImage
              key={`${mediaKey}:${comparisonVisible}`}
              src={currentItem.url}
              name={currentItem.name}
              onInteractionChange={handleImageInteraction}
              onLoad={() => { setMediaLoadError(''); setLoadedImageKey(mediaKey) }}
              onError={() => { setLoadedImageKey(''); setMediaLoadError('This image could not be loaded.') }}
            />
            {mediaLoadError && <p role="alert" className="pointer-events-none absolute bottom-4 left-1/2 z-20 -translate-x-1/2 rounded bg-black/85 px-3 py-2 text-center text-xs text-rose-200">{mediaLoadError}</p>}
          </div>
        ) : !currentItem ? (
          <div className="flex min-h-0 flex-1 items-center justify-center text-sm text-white/60">No images or videos to show.</div>
        ) : null}
        </GallerySwipeDeck>
        </div>
      </main>

      {(loadMoreError || (!hasNext && hasMore && onLoadMore)) && (
        <div aria-hidden={!viewerControlsVisible} inert={!viewerControlsVisible} {...overlayInteraction}
          className={`absolute inset-x-3 z-30 flex flex-col items-center gap-2 text-xs transition-opacity duration-150 ${overlayVisibility}`} style={{ bottom: 'calc(max(0.75rem, env(safe-area-inset-bottom)) + 4rem)' }}>
          {loadMoreError && <p role="status" className="rounded-md bg-black/75 px-3 py-2 text-rose-200">{loadMoreError}</p>}
          {hasMore && onLoadMore && (
            <button
              type="button"
              onClick={() => void requestMore(true)}
              disabled={loadingMore}
              className="flex h-10 items-center gap-1.5 rounded-full bg-black/65 px-3 text-white hover:bg-black/85 disabled:opacity-50"
            >
              {loadingMore ? <Loader2 size={13} className="animate-spin" /> : <ChevronDown size={14} />}
              <span>{loadingMore ? 'Loading' : 'Load more'}</span>
            </button>
          )}
        </div>
      )}
    </div>,
    portalHost,
  )
}
