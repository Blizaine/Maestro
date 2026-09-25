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
import { getVideoPosterUrl, requestThumbnail } from '../../lib/thumbnailCache'

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

function GalleryPreview({ file, onThumbnailReady }: {
  file: OutputFile
  onThumbnailReady?: (identity: string, thumbnail: string) => void
}) {
  const [thumbnail, setThumbnail] = useState<string | null>(null)
  useEffect(() => {
    if (file.type !== 'video') return
    let active = true
    const identity = outputIdentity(file)
    void requestThumbnail(file.url, file.url).then(url => {
      if (active) setThumbnail(url)
      if (url) onThumbnailReady?.(identity, url)
    })
    return () => { active = false }
  }, [file, file.type, file.url, onThumbnailReady])
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
  const swipeDeckRef = useRef<GallerySwipeDeckHandle>(null)
  const playerRef = useRef<GalleryVideoPlayerHandle>(null)
  const lastComparisonImageRef = useRef<GalleryImageChoice | null>(null)
  const previousFocusRef = useRef<HTMLElement | null>(null)
  const focusReturnFrameRef = useRef<number | null>(null)
  const loadMoreInFlightRef = useRef(false)
  const lastLoadAttemptLengthRef = useRef<number | null>(null)
  const pendingNextFromRef = useRef<number | null>(null)
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

  const activeIndex = galleryItems.findIndex(item => outputIdentity(item) === activeId)
  const resolvedIndex = activeIndex >= 0 ? activeIndex : 0
  const currentItem = galleryItems[resolvedIndex]
  const currentIdentity = currentItem ? outputIdentity(currentItem) : activeId
  const mediaKey = currentItem ? `${currentIdentity}\u0000${currentItem.url}\u0000${currentItem.type}` : ''
  const currentVideoPoster = currentItem?.type === 'video'
    ? previewThumbnails[currentIdentity] || getVideoPosterUrl(currentItem.url)
    : null
  const hasPrevious = resolvedIndex > 0
  const hasNext = resolvedIndex >= 0 && resolvedIndex < galleryItems.length - 1
  previewTargetIdsRef.current = new Set([
    ...(currentItem ? [currentIdentity] : []),
    ...(hasPrevious ? [outputIdentity(galleryItems[resolvedIndex - 1])] : []),
    ...(hasNext ? [outputIdentity(galleryItems[resolvedIndex + 1])] : []),
  ])
  const comparisonVisible = compareMode && currentItem?.type === 'image'
  const imageGestureBlocked = currentItem?.type === 'image' && imageInteraction.id === mediaKey && imageInteraction.blocked
  const handleImageInteraction = useCallback((blocked: boolean) => {
    if (blocked) swipeDeckRef.current?.cancelGesture()
    setImageInteraction(previous => previous.id === mediaKey && previous.blocked === blocked ? previous : { id: mediaKey, blocked })
  }, [mediaKey])
  const rememberPreviewThumbnail = useCallback((identity: string, thumbnail: string) => {
    if (!previewTargetIdsRef.current.has(identity)) return
    setPreviewThumbnails(current => {
      if (current[identity] === thumbnail) return current
      const entries = Object.entries(current).filter(([key]) => key !== identity)
      entries.push([identity, thumbnail])
      return Object.fromEntries(entries.slice(-3))
    })
  }, [])
  const handleVideoFrameReady = useCallback((src: string) => setReadyVideoSrc(src), [])

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
    if (document.activeElement === document.body) viewerRef.current?.focus()
    const closeButton = viewerRef.current?.querySelector<HTMLButtonElement>('[data-gallery-close]')
    closeButton?.focus()
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

  const requestMore = useCallback(async (manual = false, advance = false) => {
    const requestedLength = galleryItems.length
    if (advance) pendingNextFromRef.current = requestedLength
    if (!hasMore || !onLoadMore) return
    if (loadMoreInFlightRef.current) return
    if (!manual && lastLoadAttemptLengthRef.current === requestedLength) return

    loadMoreInFlightRef.current = true
    lastLoadAttemptLengthRef.current = requestedLength
    setLoadingMore(true)
    setLoadMoreError('')
    try {
      await onLoadMore()
    } catch (error) {
      setLoadMoreError(error instanceof Error ? error.message : 'Could not load more images.')
    } finally {
      loadMoreInFlightRef.current = false
      setLoadingMore(false)
    }
  }, [galleryItems.length, hasMore, onLoadMore])

  useEffect(() => {
    const requestedLength = pendingNextFromRef.current
    if (requestedLength === null) return
    if (galleryItems.length > requestedLength) {
      setReadyVideoSrc('')
      setActiveId(outputIdentity(galleryItems[requestedLength]))
      pendingNextFromRef.current = null
    } else if (!hasMore) {
      pendingNextFromRef.current = null
    }
  }, [galleryItems, hasMore])

  useEffect(() => {
    if (galleryItems.length > 0 && resolvedIndex >= galleryItems.length - 2 && hasMore && onLoadMore) {
      void requestMore()
    }
  }, [galleryItems.length, hasMore, onLoadMore, requestMore, resolvedIndex])

  const navigate = useCallback((direction: -1 | 1, manualLoad = false) => {
    if (galleryItems.length === 0) return
    const nextIndex = resolvedIndex + direction
    if (nextIndex >= 0 && nextIndex < galleryItems.length) {
      if (comparisonVisible || imageGestureBlocked) {
        setReadyVideoSrc('')
        setActiveId(outputIdentity(galleryItems[nextIndex]))
      } else swipeDeckRef.current?.navigate(direction)
      return
    }
    if (direction > 0 && hasMore && onLoadMore) void requestMore(manualLoad, true)
  }, [comparisonVisible, galleryItems, hasMore, imageGestureBlocked, onLoadMore, requestMore, resolvedIndex])

  const commitNavigation = useCallback((direction: -1 | 1) => {
    const next = galleryItems[resolvedIndex + direction]
    if (next) {
      setReadyVideoSrc('')
      setActiveId(outputIdentity(next))
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
      <div data-gallery-viewer-actions className="absolute z-30 flex items-center gap-2"
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

      {favoriteError && <p role="alert" className="absolute right-3 z-40 rounded-md bg-rose-950/90 px-3 py-2 text-xs text-rose-100" style={{ top: 'calc(max(0.75rem, env(safe-area-inset-top)) + 3.25rem)' }}>{favoriteError}</p>}
      {fullscreenError && <div role="status" className="absolute inset-x-3 z-30 mx-auto flex max-w-md items-start gap-2 rounded-lg bg-black/75 px-3 py-2 text-xs text-white/80" style={{ top: 'calc(max(0.75rem, env(safe-area-inset-top)) + 3.25rem)' }}>
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
          onTap={() => { if (currentItem?.type === 'video') playerRef.current?.togglePlayback() }}
          previous={hasPrevious ? <GalleryPreview key={outputIdentity(galleryItems[resolvedIndex - 1])} file={galleryItems[resolvedIndex - 1]} onThumbnailReady={rememberPreviewThumbnail} /> : undefined}
          next={hasNext ? <GalleryPreview key={outputIdentity(galleryItems[resolvedIndex + 1])} file={galleryItems[resolvedIndex + 1]} onThumbnailReady={rememberPreviewThumbnail} /> : undefined}>
        {/* Safari authorizes sound per media element. Keep this player mounted
            through video and image navigation, unloading its source on images. */}
        <div className={currentItem?.type === 'video' ? 'relative h-full w-full min-h-0' : 'hidden'}
          aria-hidden={currentItem?.type !== 'video'} inert={currentItem?.type !== 'video'}>
          <GalleryVideoPlayer ref={playerRef} src={currentItem?.type === 'video' ? currentItem.url : undefined} name={currentItem?.name ?? ''}
            initialTime={currentIdentity === initialId ? initialTime : undefined} onFrameReady={handleVideoFrameReady} />
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
              onLoad={() => setMediaLoadError('')}
              onError={() => setMediaLoadError('This image could not be loaded.')}
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
        <div className="absolute inset-x-3 z-30 flex flex-col items-center gap-2 text-xs" style={{ bottom: 'calc(max(0.75rem, env(safe-area-inset-bottom)) + 4rem)' }}>
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
