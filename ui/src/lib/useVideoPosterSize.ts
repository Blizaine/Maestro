import { useLayoutEffect, useState, type RefObject } from 'react'
import type { VideoPosterSize } from './thumbnailCache'

/** Match the displayed surface, including Retina screens, without making tiny
 * thumbnails or ordinary phone cards fetch desktop-sized images. */
export function useVideoPosterSize(ref: RefObject<HTMLElement | null>, mediaKey: string): VideoPosterSize | null {
  const [size, setSize] = useState<VideoPosterSize | null>(null)
  useLayoutEffect(() => {
    const element = ref.current
    if (!element || !mediaKey) return
    const measure = () => {
      const { width, height } = element.getBoundingClientRect()
      if (width <= 0 || height <= 0) return
      const pixels = Math.max(width, height) * Math.min(2, window.devicePixelRatio || 1)
      setSize(pixels <= 480 ? 480 : pixels <= 960 ? 960 : 1920)
    }
    const observer = new ResizeObserver(measure)
    observer.observe(element)
    // The first measurement happens before a poster URL is requested. Resizing
    // a card, rotating a phone, or opening fullscreen selects the proper tier.
    const frame = window.requestAnimationFrame(measure)
    window.addEventListener('resize', measure)
    return () => {
      observer.disconnect()
      window.cancelAnimationFrame(frame)
      window.removeEventListener('resize', measure)
    }
  }, [ref, mediaKey])
  return size
}
