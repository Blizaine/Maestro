import { useEffect, useMemo, useState } from 'react'
import type { EditorAsset, EditorTimelineItem, EditorTrackType } from '../types'
import { useEditorMediaPreview } from './editorMediaPreview'

const videoStripCache = new Map<string, string[]>()
const videoStripPromises = new Map<string, Promise<string[]>>()
const waveformCache = new Map<string, number[]>()
const waveformPromises = new Map<string, Promise<number[]>>()

function waitForEvent(target: EventTarget, success: string, failure: string): Promise<void> {
  return new Promise((resolve, reject) => {
    const cleanup = () => {
      target.removeEventListener(success, onSuccess)
      target.removeEventListener(failure, onFailure)
    }
    const onSuccess = () => { cleanup(); resolve() }
    const onFailure = () => { cleanup(); reject(new Error(`Media ${failure}`)) }
    target.addEventListener(success, onSuccess, { once: true })
    target.addEventListener(failure, onFailure, { once: true })
  })
}

async function generateVideoStrip(
  asset: EditorAsset,
  item: EditorTimelineItem,
  count: number,
): Promise<string[]> {
  const key = `${asset.url}|${item.source_in.toFixed(3)}|${item.duration.toFixed(3)}|${item.speed.toFixed(3)}|${count}`
  const cached = videoStripCache.get(key)
  if (cached) return cached
  const pending = videoStripPromises.get(key)
  if (pending) return pending
  const promise = (async () => {
    const video = document.createElement('video')
    video.muted = true
    video.playsInline = true
    video.preload = 'metadata'
    video.src = asset.url
    if (video.readyState < 1) await waitForEvent(video, 'loadedmetadata', 'error')
    if (video.readyState < 2) await waitForEvent(video, 'loadeddata', 'error')
    const sourceDuration = Math.max(0.04, item.duration * Math.max(0.1, item.speed))
    const maximumTime = Math.max(0, (Number.isFinite(video.duration) ? video.duration : asset.duration) - 0.03)
    const canvas = document.createElement('canvas')
    canvas.width = 144
    canvas.height = 82
    const context = canvas.getContext('2d', { alpha: false })
    if (!context) return []
    const frames: string[] = []
    for (let index = 0; index < count; index += 1) {
      const fraction = count === 1 ? 0.5 : index / Math.max(1, count - 1)
      const time = Math.min(maximumTime, Math.max(0, item.source_in + sourceDuration * fraction))
      if (Math.abs(video.currentTime - time) > 0.01) {
        video.currentTime = time
        await waitForEvent(video, 'seeked', 'error')
      }
      context.fillStyle = '#111827'
      context.fillRect(0, 0, canvas.width, canvas.height)
      const sourceWidth = Math.max(1, video.videoWidth)
      const sourceHeight = Math.max(1, video.videoHeight)
      const scale = Math.max(canvas.width / sourceWidth, canvas.height / sourceHeight)
      const width = sourceWidth * scale
      const height = sourceHeight * scale
      context.drawImage(video, (canvas.width - width) / 2, (canvas.height - height) / 2, width, height)
      frames.push(canvas.toDataURL('image/jpeg', 0.62))
    }
    video.removeAttribute('src')
    video.load()
    videoStripCache.set(key, frames)
    return frames
  })().catch(() => []).finally(() => videoStripPromises.delete(key))
  videoStripPromises.set(key, promise)
  return promise
}

async function generateWaveform(url: string): Promise<number[]> {
  const cached = waveformCache.get(url)
  if (cached) return cached
  const pending = waveformPromises.get(url)
  if (pending) return pending
  const promise = (async () => {
    const response = await fetch(url)
    if (!response.ok) throw new Error('Waveform source unavailable')
    const AudioContextClass = window.AudioContext
      || (window as typeof window & { webkitAudioContext?: typeof AudioContext }).webkitAudioContext
    if (!AudioContextClass) return []
    const context = new AudioContextClass()
    try {
      const decoded = await context.decodeAudioData(await response.arrayBuffer())
      const channel = decoded.getChannelData(0)
      const bins = 240
      const block = Math.max(1, Math.floor(channel.length / bins))
      const values = Array.from({ length: bins }, (_, index) => {
        const start = index * block
        const end = Math.min(channel.length, start + block)
        let peak = 0
        for (let sample = start; sample < end; sample += Math.max(1, Math.floor(block / 48))) {
          peak = Math.max(peak, Math.abs(channel[sample] || 0))
        }
        return peak
      })
      const maximum = Math.max(0.001, ...values)
      const normalized = values.map(value => Math.max(0.06, Math.min(1, value / maximum)))
      waveformCache.set(url, normalized)
      return normalized
    } finally {
      void context.close()
    }
  })().catch(() => []).finally(() => waveformPromises.delete(url))
  waveformPromises.set(url, promise)
  return promise
}

function useVideoStrip(asset: EditorAsset | undefined, item: EditorTimelineItem, width: number, enabled = true): string[] {
  const count = Math.max(1, Math.min(10, Math.ceil(width / 82)))
  const key = asset ? `${asset.url}|${item.source_in.toFixed(3)}|${item.duration.toFixed(3)}|${item.speed.toFixed(3)}|${count}` : ''
  const [result, setResult] = useState<{ key: string; frames: string[] }>(() => ({
    key,
    frames: videoStripCache.get(key) || [],
  }))
  useEffect(() => {
    let active = true
    if (!enabled || !asset || asset.type !== 'video' || width < 36) return () => { active = false }
    void generateVideoStrip(asset, item, count).then(frames => {
      if (active) setResult({ key, frames })
    })
    return () => { active = false }
  }, [asset, count, enabled, item, key, width])
  if (!enabled || !asset || asset.type !== 'video' || width < 36) return []
  return result.key === key ? result.frames : videoStripCache.get(key) || []
}

function useWaveform(asset: EditorAsset | undefined): number[] {
  const key = asset?.has_audio ? asset.url : ''
  const [result, setResult] = useState<{ key: string; values: number[] }>(() => ({
    key,
    values: key ? waveformCache.get(key) || [] : [],
  }))
  useEffect(() => {
    let active = true
    if (!key) return () => { active = false }
    void generateWaveform(key).then(values => {
      if (active) setResult({ key, values })
    })
    return () => { active = false }
  }, [key])
  if (!key) return []
  return result.key === key ? result.values : waveformCache.get(key) || []
}

export function EditorClipMedia({
  asset,
  item,
  trackType,
  width,
  workspace,
}: {
  asset?: EditorAsset
  item: EditorTimelineItem
  trackType: EditorTrackType
  width: number
  workspace: string
}) {
  const serverPreview = useEditorMediaPreview(asset, workspace)
  const frames = useVideoStrip(
    trackType === 'video' ? asset : undefined,
    item,
    width,
    serverPreview.failed || (!serverPreview.loading && !serverPreview.data?.thumbnail_url),
  )
  const browserWaveform = useWaveform(
    trackType === 'audio'
      && (serverPreview.failed || (!serverPreview.loading && !serverPreview.data?.waveform?.length))
      ? asset
      : undefined,
  )
  const waveform = serverPreview.data?.waveform?.length
    ? serverPreview.data.waveform
    : browserWaveform
  const visibleWaveform = useMemo(() => {
    if (waveform.length === 0) return []
    const count = Math.max(12, Math.min(120, Math.floor(width / 3)))
    const sourceStart = asset?.duration
      ? Math.max(0, Math.min(1, item.source_in / asset.duration))
      : 0
    const sourceEnd = asset?.duration
      ? Math.max(sourceStart, Math.min(1, (item.source_in + item.duration * item.speed) / asset.duration))
      : 1
    return Array.from({ length: count }, (_, index) => {
      const fraction = count === 1 ? 0 : index / (count - 1)
      const sourceIndex = Math.min(
        waveform.length - 1,
        Math.floor((sourceStart + (sourceEnd - sourceStart) * fraction) * waveform.length),
      )
      return waveform[sourceIndex] || 0.06
    })
  }, [asset?.duration, item.duration, item.source_in, item.speed, waveform, width])

  if (asset?.missing) {
    return (
      <span className="pointer-events-none absolute inset-0 grid place-items-center bg-red-950/55 text-[8px] font-semibold uppercase tracking-wide text-red-200">
        Media offline
      </span>
    )
  }

  if (trackType === 'audio') {
    return (
      <span className="pointer-events-none absolute inset-0 flex items-center gap-px overflow-hidden px-1 opacity-55">
        {(visibleWaveform.length > 0 ? visibleWaveform : [0.2, 0.5, 0.32, 0.72, 0.4, 0.62, 0.25]).map((level, index) => (
          <i key={index} className="min-w-px flex-1 rounded-full bg-current" style={{ height: `${Math.max(8, level * 82)}%` }} />
        ))}
      </span>
    )
  }
  if (asset?.type === 'image') {
    return <img src={asset.url} alt="" className="pointer-events-none absolute inset-0 h-full w-full object-cover opacity-55" draggable={false} />
  }
  if (serverPreview.data?.thumbnail_url) {
    return (
      <span
        className="pointer-events-none absolute inset-0 bg-center bg-no-repeat opacity-60"
        style={{
          backgroundImage: `url(${serverPreview.data.thumbnail_url})`,
          backgroundSize: '100% 100%',
        }}
      />
    )
  }
  if (frames.length > 0) {
    return (
      <span className="pointer-events-none absolute inset-0 flex overflow-hidden opacity-55">
        {frames.map((frame, index) => <img key={`${frame.slice(-18)}-${index}`} src={frame} alt="" className="h-full min-w-0 flex-1 object-cover" draggable={false} />)}
      </span>
    )
  }
  return null
}
