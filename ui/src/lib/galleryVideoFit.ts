import { useCallback, useLayoutEffect, useState, type RefObject, type SyntheticEvent } from 'react'
import type { OutputFile, OutputMediaInfo } from '../types'

const DEFAULT_GALLERY_VIDEO_ASPECT = 16 / 9
const GALLERY_VIDEO_ASPECT_TOLERANCE = 0.05

type VideoDimensions = { width: number; height: number }
type IdentifiedDimensions = VideoDimensions & { identity: string }

/**
 * A gallery video can use its native frame ratio only when it is close enough
 * to 16:9 that the card's surrounding feed layout remains consistent.
 */
export function getGalleryVideoFrameAspect(dimensions?: VideoDimensions | null): string | undefined {
  if (!dimensions || !Number.isFinite(dimensions.width) || !Number.isFinite(dimensions.height)) return undefined
  if (dimensions.width <= 0 || dimensions.height <= 0) return undefined

  const aspect = dimensions.width / dimensions.height
  if (Math.abs(aspect / DEFAULT_GALLERY_VIDEO_ASPECT - 1) > GALLERY_VIDEO_ASPECT_TOLERANCE) return undefined
  return `${dimensions.width} / ${dimensions.height}`
}

/** Metadata dimensions are a seed only when the sidecar describes this exact file size. */
export function getTrustedGalleryVideoDimensions(
  mediaInfo: OutputMediaInfo | undefined,
  fileSize: number,
): VideoDimensions | null {
  if (!mediaInfo || mediaInfo.size_bytes !== fileSize) return null
  const { width, height } = mediaInfo
  if (typeof width !== 'number' || typeof height !== 'number') return null
  if (!Number.isFinite(width) || !Number.isFinite(height) || width <= 0 || height <= 0) return null
  return { width, height }
}

/** Include the file's path and stat-like identity so virtualized rows cannot reuse old dimensions. */
export function getGalleryVideoIdentity(file: Pick<OutputFile, 'id' | 'workspace' | 'name' | 'url' | 'size' | 'created_at'>): string {
  return JSON.stringify([
    file.id || '',
    file.workspace || '',
    file.name,
    file.url,
    file.size,
    file.created_at,
  ])
}

/**
 * Prefer dimensions from the loaded media element, using matching sidecar
 * dimensions as an early seed while the browser is still loading metadata.
 */
export function useGalleryVideoFrameAspect(
  identity: string,
  mediaInfo: OutputMediaInfo | undefined,
  fileSize: number,
  videoRef: RefObject<HTMLVideoElement | null>,
): { frameAspect: string | undefined; onLoadedMetadata: (event: SyntheticEvent<HTMLVideoElement>) => void } {
  const [browserDimensions, setBrowserDimensions] = useState<IdentifiedDimensions | null>(null)
  const seededDimensions = getTrustedGalleryVideoDimensions(mediaInfo, fileSize)
  const currentBrowserDimensions = browserDimensions?.identity === identity ? browserDimensions : null
  const dimensions = currentBrowserDimensions || seededDimensions

  const onLoadedMetadata = useCallback((event: SyntheticEvent<HTMLVideoElement>) => {
    const { videoWidth: width, videoHeight: height } = event.currentTarget
    if (width <= 0 || height <= 0) return
    setBrowserDimensions({ identity, width, height })
  }, [identity])

  // `loadedmetadata` may have fired before this component's handler was
  // attached (for example, when the browser has the resource cached).
  useLayoutEffect(() => {
    const video = videoRef.current
    if (!video || video.videoWidth <= 0 || video.videoHeight <= 0) return
    setBrowserDimensions({ identity, width: video.videoWidth, height: video.videoHeight })
  }, [identity, videoRef])

  return { frameAspect: getGalleryVideoFrameAspect(dimensions), onLoadedMetadata }
}
