import { useEffect, useState } from 'react'
import * as api from '../api/client'
import type { EditorAsset, EditorGraphic } from '../types'
import { editorGraphicPreviewAssets, editorGraphicPreviewKey } from './editorGraphics'

const GRAPHIC_PREVIEW_CACHE_LIMIT = 24
const graphicPreviewCache = new Map<string, Blob>()

function cacheGraphicPreview(key: string, blob: Blob): void {
  graphicPreviewCache.delete(key)
  graphicPreviewCache.set(key, blob)
  while (graphicPreviewCache.size > GRAPHIC_PREVIEW_CACHE_LIMIT) {
    const oldest = graphicPreviewCache.keys().next().value
    if (oldest === undefined) break
    graphicPreviewCache.delete(oldest)
  }
}

export function useEditorGraphicPreview(
  graphic: EditorGraphic,
  width: number,
  height: number,
  assets: Record<string, EditorAsset>,
  workspace: string,
): { url: string | null; error: string | null; loading: boolean } {
  const key = editorGraphicPreviewKey(graphic, width, height, assets, workspace)
  const [result, setResult] = useState<{
    key: string
    url: string | null
    error: string | null
    loading: boolean
  }>({ key: '', url: null, error: null, loading: true })

  useEffect(() => {
    let objectUrl: string | null = null
    const cached = graphicPreviewCache.get(key)
    if (cached) {
      graphicPreviewCache.delete(key)
      graphicPreviewCache.set(key, cached)
      objectUrl = URL.createObjectURL(cached)
      let released = false
      queueMicrotask(() => {
        if (!released && objectUrl) setResult({ key, url: objectUrl, error: null, loading: false })
      })
      return () => {
        released = true
        if (objectUrl) URL.revokeObjectURL(objectUrl)
      }
    }

    const controller = new AbortController()
    const timer = window.setTimeout(() => {
      const previewAssets = editorGraphicPreviewAssets(graphic, assets)
      void api.previewEditorGraphic({ graphic, width, height, assets: previewAssets, workspace }, controller.signal)
        .then(blob => {
          if (controller.signal.aborted) return
          cacheGraphicPreview(key, blob)
          objectUrl = URL.createObjectURL(blob)
          setResult({ key, url: objectUrl, error: null, loading: false })
        })
        .catch(error => {
          if (controller.signal.aborted) return
          setResult({
            key,
            url: null,
            error: error instanceof Error ? error.message : 'Unable to render this graphic overlay.',
            loading: false,
          })
        })
    }, 200)

    return () => {
      window.clearTimeout(timer)
      controller.abort()
      if (objectUrl) URL.revokeObjectURL(objectUrl)
    }
  }, [assets, graphic, height, key, width, workspace])

  return result.key === key
    ? { url: result.url, error: result.error, loading: result.loading }
    : { url: null, error: null, loading: true }
}
