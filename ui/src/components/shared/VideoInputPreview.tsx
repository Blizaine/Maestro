import { useState } from 'react'
import { getVideoPosterUrl } from '../../lib/thumbnailCache'

function posterUrlForSource(src: string): string | null {
  const direct = getVideoPosterUrl(src)
  if (direct) return direct

  // The generic upload endpoint returns /api/v1/uploads/<name>, while the
  // thumbnail route resolves the same file through the gallery file path.
  try {
    const url = new URL(src, window.location.href)
    if (url.origin !== window.location.origin || !url.pathname.startsWith('/api/v1/uploads/')) return null
    url.pathname = url.pathname.replace('/api/v1/uploads/', '/api/v1/file/')
    if (!url.searchParams.has('workspace')) url.searchParams.set('workspace', '__uploads__')
    return getVideoPosterUrl(url.toString())
  } catch {
    return null
  }
}

/** Use the server's bounded first-frame poster for managed files, with a
 * browser video preview as a fallback for blob URLs and unavailable posters. */
export function VideoInputPreview({ src, alt, className }: {
  src: string
  alt: string
  className?: string
}) {
  const posterUrl = posterUrlForSource(src)
  const [failedPosterUrl, setFailedPosterUrl] = useState<string | null>(null)

  if (posterUrl && failedPosterUrl !== posterUrl) {
    return <img src={posterUrl} alt={alt} draggable={false} loading="lazy" className={className}
      onError={() => setFailedPosterUrl(posterUrl)} />
  }

  return <video src={`${src}${src.includes('#') ? '&' : '#'}t=0.1`} muted playsInline preload="metadata"
    aria-label={alt} className={className} />
}
