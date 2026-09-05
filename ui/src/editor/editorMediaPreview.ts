import { useEffect, useMemo, useState } from 'react'
import * as api from '../api/client'
import type { EditorAsset, EditorMediaPreview } from '../types'

interface PreviewState {
  data: EditorMediaPreview | null
  loading: boolean
  failed: boolean
}

type EditorProxyProfile = 'auto' | 'mobile'

const previewCache = new Map<string, EditorMediaPreview>()
const previewPromises = new Map<string, Promise<EditorMediaPreview>>()
const previewFailures = new Set<string>()

function previewKey(
  asset: EditorAsset,
  workspace: string,
  includeProxy: boolean,
  proxyProfile: EditorProxyProfile,
): string {
  return [
    workspace,
    asset.id,
    asset.path || asset.name,
    asset.size || 0,
    asset.duration || 0,
    includeProxy ? `proxy-${proxyProfile}` : 'analysis',
  ].join('|')
}

function requestPreview(
  asset: EditorAsset,
  workspace: string,
  includeProxy: boolean,
  proxyProfile: EditorProxyProfile,
): Promise<EditorMediaPreview> {
  const key = previewKey(asset, workspace, includeProxy, proxyProfile)
  const cached = previewCache.get(key)
  if (cached) return Promise.resolve(cached)
  const pending = previewPromises.get(key)
  if (pending) return pending
  const request = api.fetchEditorMediaPreview(asset, workspace, includeProxy, proxyProfile)
    .then(result => {
      previewCache.set(key, result)
      previewFailures.delete(key)
      return result
    })
    .finally(() => previewPromises.delete(key))
  previewPromises.set(key, request)
  return request
}

export function invalidateEditorMediaPreview(assetId: string): void {
  Array.from(previewCache.keys()).forEach(key => {
    if (key.includes(`|${assetId}|`)) previewCache.delete(key)
  })
  Array.from(previewFailures).forEach(key => {
    if (key.includes(`|${assetId}|`)) previewFailures.delete(key)
  })
}

export function useEditorMediaPreview(
  asset: EditorAsset | undefined,
  workspace: string,
  includeProxy = false,
  proxyProfile: EditorProxyProfile = 'auto',
): PreviewState {
  const key = useMemo(
    () => asset ? previewKey(asset, workspace, includeProxy, proxyProfile) : '',
    [asset, includeProxy, proxyProfile, workspace],
  )
  const [result, setResult] = useState<{
    key: string
    data: EditorMediaPreview | null
    failed: boolean
  }>(() => ({
    key,
    data: key ? previewCache.get(key) || null : null,
    failed: Boolean(key && previewFailures.has(key)),
  }))

  useEffect(() => {
    let active = true
    if (!asset || !key || asset.missing) return () => { active = false }
    const cached = previewCache.get(key)
    if (cached) return () => { active = false }
    void requestPreview(asset, workspace, includeProxy, proxyProfile).then(result => {
      if (active) setResult({ key, data: result, failed: false })
    }).catch(() => {
      previewFailures.add(key)
      if (active) setResult({ key, data: null, failed: true })
    })
    return () => { active = false }
  }, [asset, includeProxy, key, proxyProfile, workspace])

  if (!asset || !key) return { data: null, loading: false, failed: false }
  if (asset.missing) return { data: null, loading: false, failed: true }
  const cached = previewCache.get(key)
  const data = cached || (result.key === key ? result.data : null)
  const failed = previewFailures.has(key) || (result.key === key && result.failed)
  return { data, loading: !data && !failed, failed }
}
