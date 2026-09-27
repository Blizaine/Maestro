import { useEffect, useId, useRef, useState, type KeyboardEvent, type PointerEvent } from 'react'
import { createPortal } from 'react-dom'
import { Loader2, Pause, Play, Scissors, X } from 'lucide-react'
import { fetchEditorMediaPreview, getFileUrl, getUploadUrl, probeEditorMedia, trimGalleryMedia } from '../../api/client'
import { useGalleryInputs } from '../../lib/galleryInputs'
import { getVideoPosterUrl } from '../../lib/thumbnailCache'
import type { EditorAsset, EditorMediaPreview } from '../../types'

interface Props {
  source: EditorAsset
  targetId: string
  targetLabel: string
  onUse: (file: File) => Promise<void>
  onClose: () => void
}

const clock = (seconds: number) => {
  if (!Number.isFinite(seconds)) return '—'
  const hundredths = Math.round(Math.max(0, seconds) * 100)
  return `${Math.floor(hundredths / 6000)}:${(Math.floor(hundredths / 100) % 60).toString().padStart(2, '0')}.${(hundredths % 100).toString().padStart(2, '0')}`
}

/** The same selection gesture for gallery soundtracks, voice samples and videos. */
export function MediaTrimDialog({ source, targetId, targetLabel, onUse, onClose }: Props) {
  const titleId = useId()
  const dialog = useRef<HTMLDivElement>(null)
  const media = useRef<HTMLMediaElement | null>(null)
  const track = useRef<HTMLDivElement>(null)
  const drag = useRef<'start' | 'end' | 'playhead' | null>(null)
  const submitting = useRef(false)
  const busyRef = useRef(false)
  const closeRef = useRef(onClose)
  const [duration, setDuration] = useState(0)
  const [start, setStart] = useState(0)
  const [end, setEnd] = useState(0)
  const [playhead, setPlayhead] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [preview, setPreview] = useState<EditorMediaPreview | null>(null)
  const [loading, setLoading] = useState(true)
  const [previewLoading, setPreviewLoading] = useState(true)
  const [previewError, setPreviewError] = useState('')
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const [zoom, setZoom] = useState<[number, number] | null>(null)
  const target = useGalleryInputs(s => s.targets.find(item => item.id === targetId))
  const receiving = useGalleryInputs(s => s.receiving)
  const destinationError = !target || target.label !== targetLabel || target.kind !== source.type
    ? 'This input is no longer available. Close this preview and choose an input in the current mode.' : target.disabledReason
  const valid = Number.isFinite(start) && Number.isFinite(end) && start >= 0 && end <= duration && end > start && end - start >= Math.min(0.05, duration)
  const fullSelection = valid && start === 0 && end === duration
  const [viewStart, viewEnd] = zoom || [0, duration]
  const span = viewEnd - viewStart || 1
  const percent = (time: number) => Math.max(0, Math.min(100, (time - viewStart) / span * 100))

  useEffect(() => { closeRef.current = onClose; busyRef.current = busy }, [onClose, busy])

  useEffect(() => {
    const previousFocus = document.activeElement as HTMLElement | null
    const previousOverflow = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    dialog.current?.focus()
    const keydown = (event: globalThis.KeyboardEvent) => {
      if (event.key === 'Escape') {
        event.preventDefault()
        event.stopImmediatePropagation()
        if (!busyRef.current && !submitting.current) closeRef.current()
      }
      if (event.key === 'Tab') {
        const elements = Array.from(dialog.current?.querySelectorAll<HTMLElement>('button:not(:disabled), input:not(:disabled), [tabindex="0"]') || [])
          .filter(element => element.getClientRects().length > 0)
        const first = elements[0], last = elements.at(-1)
        if (event.shiftKey && (document.activeElement === first || document.activeElement === dialog.current)) {
          event.preventDefault(); last?.focus()
        } else if (!event.shiftKey && (document.activeElement === last || document.activeElement === dialog.current)) {
          event.preventDefault(); first?.focus()
        }
      }
    }
    document.addEventListener('keydown', keydown, true)
    return () => {
      document.body.style.overflow = previousOverflow
      document.removeEventListener('keydown', keydown, true)
      previousFocus?.focus()
    }
  }, [])

  useEffect(() => {
    const controller = new AbortController()
    void (async () => {
      try {
        const info = await probeEditorMedia(source, source.workspace, controller.signal)
        if (controller.signal.aborted) return
        if (!Number.isFinite(info.duration) || info.duration <= 0) throw new Error('The clip has no readable duration. You can still use the full clip.')
        setDuration(info.duration)
        setEnd(info.duration)
        setLoading(false)
        try {
          const result = await fetchEditorMediaPreview({ ...source, ...info, type: source.type }, source.workspace || '', false, 'auto', controller.signal)
          if (!controller.signal.aborted) setPreview(result)
        } catch {
          if (!controller.signal.aborted) setPreviewError('Timeline preview unavailable. You can still select times and preview the clip.')
        }
      } catch (cause) {
        if (!controller.signal.aborted) setError(cause instanceof Error ? cause.message : 'Could not inspect this clip.')
      } finally {
        if (!controller.signal.aborted) { setLoading(false); setPreviewLoading(false) }
      }
    })()
    return () => controller.abort()
  }, [source])

  const seek = (time: number) => {
    setPlayhead(time)
    if (media.current?.readyState) media.current.currentTime = time
  }
  const updateHandle = (handle: 'start' | 'end' | 'playhead', time: number) => {
    media.current?.pause()
    const gap = Math.min(0.05, duration)
    if (handle === 'start') {
      const value = Math.max(0, Math.min(time, (Number.isFinite(end) ? end : duration) - gap))
      setStart(value); seek(value)
    } else if (handle === 'end') {
      const value = Math.min(duration, Math.max(time, (Number.isFinite(start) ? start : 0) + gap))
      setEnd(value); seek(value)
    } else seek(Math.max(0, Math.min(duration, time)))
  }
  const pointerTime = (event: PointerEvent) => {
    const bounds = track.current!.getBoundingClientRect()
    return Math.round((viewStart + Math.max(0, Math.min(1, (event.clientX - bounds.left) / bounds.width)) * span) * 100) / 100
  }
  const pointerDown = (event: PointerEvent<HTMLDivElement>, handle: 'start' | 'end' | 'playhead') => {
    if (busy || duration <= 0) return
    event.preventDefault(); event.stopPropagation()
    event.currentTarget.setPointerCapture(event.pointerId)
    drag.current = handle
    updateHandle(handle, pointerTime(event))
  }
  const keyboardHandle = (event: KeyboardEvent, handle: 'start' | 'end' | 'playhead', current: number) => {
    let time = current
    if (event.key === 'ArrowLeft' || event.key === 'ArrowDown') time -= event.shiftKey ? 1 : 0.1
    else if (event.key === 'ArrowRight' || event.key === 'ArrowUp') time += event.shiftKey ? 1 : 0.1
    else if (event.key === 'Home') time = viewStart
    else if (event.key === 'End') time = viewEnd
    else return
    event.preventDefault(); event.stopPropagation()
    if (!busy) updateHandle(handle, Math.round(time * 100) / 100)
  }
  const togglePreview = async () => {
    const element = media.current
    if (!element || !valid) return
    if (!element.paused) { element.pause(); return }
    if (element.currentTime < start || element.currentTime >= end - 0.025) element.currentTime = start
    try { await element.play() } catch { setPreviewError('Playback is unavailable in this browser. You can still use this clip.') }
  }
  const onTimeUpdate = () => {
    const element = media.current
    if (!element) return
    if (!element.paused && valid && element.currentTime >= end) {
      element.pause(); element.currentTime = start
    }
    setPlayhead(element.currentTime)
  }
  const submitClip = async (full: boolean) => {
    if (submitting.current || receiving || destinationError || (!full && !valid)) return
    submitting.current = true; setBusy(true); setError(''); media.current?.pause()
    try {
      const checkDestination = () => {
        const current = useGalleryInputs.getState().targets.find(item => item.id === targetId)
        if (!current || current.label !== targetLabel || current.kind !== source.type || current.disabledReason) {
          throw new Error(current?.disabledReason || 'The input changed. Close this preview and choose a destination again.')
        }
      }
      checkDestination()
      let url = source.url
      let name = source.name
      let mime = ''
      if (!full) {
        const result = await trimGalleryMedia({ name: source.name, origin: source.origin, workspace: source.workspace }, start, end)
        url = getUploadUrl(result.filename)
        name = `${source.name.replace(/\.[^.]+$/, '').slice(0, 120)}_excerpt_${start.toFixed(2)}-${end.toFixed(2)}.${result.media_type === 'audio' ? 'wav' : 'mp4'}`
        mime = result.mime_type
      }
      // Avoid silently sending into a different mode after a long trim/upload.
      checkDestination()
      const response = await fetch(url)
      if (!response.ok) throw new Error(`Could not read this clip (${response.status}).`)
      const blob = await response.blob()
      checkDestination()
      const extension = name.split('.').pop()?.toLowerCase() || ''
      const fallback: Record<string, string> = { mp3: 'audio/mpeg', wav: 'audio/wav', flac: 'audio/flac', ogg: 'audio/ogg', opus: 'audio/ogg', m4a: 'audio/mp4', aac: 'audio/aac', mp4: 'video/mp4', mov: 'video/quicktime', webm: `${source.type}/webm`, mkv: 'video/x-matroska' }
      await onUse(new File([blob], name, { type: mime || (blob.type.startsWith(`${source.type}/`) ? blob.type : fallback[extension] || `${source.type}/${extension || 'wav'}`) }))
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Could not use this clip.')
    } finally { submitting.current = false; setBusy(false) }
  }

  const waveform = preview?.waveform || []
  const waveStart = Math.floor(viewStart / (duration || 1) * waveform.length)
  const waveEnd = Math.max(waveStart + 1, Math.ceil(viewEnd / (duration || 1) * waveform.length))
  const peaks = waveform.slice(waveStart, waveEnd)
  const mutedClass = 'disabled:opacity-40 disabled:cursor-not-allowed'
  const smallButton = `rounded-lg border border-border px-2.5 py-1.5 text-xs text-text-secondary hover:bg-bg-hover ${mutedClass}`
  const handleProps = (handle: 'start' | 'end', time: number) => ({
    role: 'slider', tabIndex: busy ? -1 : 0,
    'aria-label': handle === 'start' ? 'Trim start' : 'Trim end',
    'aria-valuemin': handle === 'end' ? (Number.isFinite(start) ? start : 0) : 0,
    'aria-valuemax': handle === 'start' ? (Number.isFinite(end) ? end : duration) : duration,
    'aria-valuenow': Number.isFinite(time) ? time : 0, 'aria-valuetext': clock(time), 'aria-disabled': busy,
    onPointerDown: (event: PointerEvent<HTMLDivElement>) => pointerDown(event, handle),
    onKeyDown: (event: KeyboardEvent) => keyboardHandle(event, handle, time),
  })

  return createPortal(
    <div className="fixed inset-0 z-[200] flex items-center justify-center bg-black/75 p-3 sm:p-5" onClick={event => { event.stopPropagation(); if (event.target === event.currentTarget && !busy && !submitting.current) onClose() }}>
      <div ref={dialog} role="dialog" aria-modal="true" aria-labelledby={titleId} tabIndex={-1}
        className="max-h-[92dvh] w-full max-w-2xl overflow-y-auto overscroll-contain rounded-xl border border-border bg-bg-secondary p-4 text-text-primary shadow-2xl outline-none sm:p-5">
        <div className="mb-2 flex items-center gap-2">
          <Scissors size={17} className="shrink-0 text-accent-blue" />
          <h2 id={titleId} className="flex-1 text-sm font-semibold">Use {source.type} as {targetLabel}</h2>
          <button aria-label="Close trim preview" disabled={busy} onClick={onClose} className={`rounded-lg p-2 hover:bg-bg-hover ${mutedClass}`}><X size={18} /></button>
        </div>
        <p className="truncate text-xs text-text-secondary" title={source.name}>{source.name}</p>
        <p className="mb-3 mt-1 text-xs text-text-muted">Use the full clip or drag the handles to choose an excerpt. The original stays unchanged.</p>
        {source.type === 'video'
          ? <video ref={element => { media.current = element }} src={source.url} poster={getVideoPosterUrl(getFileUrl(source.name, source.workspace)) || undefined} playsInline preload="metadata" className="mb-3 max-h-[30dvh] w-full rounded-lg bg-black object-contain" onTimeUpdate={onTimeUpdate} onPlay={() => setPlaying(true)} onPause={() => setPlaying(false)} onEnded={() => setPlaying(false)} onError={() => setPreviewError('This browser cannot preview the video. You can still select and use an excerpt.')} onClick={() => void togglePreview()} />
          : <audio ref={element => { media.current = element }} src={source.url} preload="metadata" onTimeUpdate={onTimeUpdate} onPlay={() => setPlaying(true)} onPause={() => setPlaying(false)} onEnded={() => setPlaying(false)} onError={() => setPreviewError('This browser cannot preview the audio. You can still select and use an excerpt.')} />}
        {loading && <p role="status" className="my-4 flex items-center gap-2 text-xs text-text-secondary"><Loader2 size={14} className="animate-spin" /> Reading clip…</p>}
        {duration > 0 && <>
          <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
            <button onClick={() => void togglePreview()} disabled={busy || !valid} className={`flex items-center gap-1.5 ${smallButton}`}>
              {playing ? <Pause size={14} /> : <Play size={14} />} {playing ? 'Pause preview' : 'Preview selection'}
            </button>
            <span className="text-xs tabular-nums text-text-muted">{clock(playhead)} / {clock(duration)}</span>
          </div>
          <div ref={track} className="relative mx-3 h-20 touch-none select-none rounded-md border border-border bg-bg-tertiary"
            onPointerDown={event => pointerDown(event, 'playhead')}
            onPointerMove={event => { if (drag.current && !busy) updateHandle(drag.current, pointerTime(event)) }}
            onPointerUp={() => { drag.current = null }} onPointerCancel={() => { drag.current = null }} onLostPointerCapture={() => { drag.current = null }}>
            {preview?.thumbnail_url && source.type === 'video' && <div aria-hidden="true" className="pointer-events-none absolute inset-0 rounded-md opacity-70" style={{ backgroundImage: `url("${preview.thumbnail_url}")`, backgroundRepeat: 'no-repeat', backgroundSize: `${duration / span * 100}% 100%`, backgroundPosition: `${duration > span ? viewStart / (duration - span) * 100 : 0}% 0` }} />}
            {source.type === 'audio' && peaks.length > 0 && <svg aria-hidden="true" className="pointer-events-none absolute inset-0 h-full w-full text-accent-blue" viewBox={`0 0 ${peaks.length} 100`} preserveAspectRatio="none">
              <path d={peaks.map((peak, index) => `M${index + 0.5},${50 - Math.max(1, peak * 44)}V${50 + Math.max(1, peak * 44)}`).join(' ')} stroke="currentColor" strokeWidth="0.75" />
            </svg>}
            {previewLoading && <div role="status" className="pointer-events-none absolute inset-0 flex items-center justify-center gap-2 text-xs text-text-muted"><Loader2 size={13} className="animate-spin" /> Loading {source.type === 'audio' ? 'waveform' : 'filmstrip'}…</div>}
            <div className="pointer-events-none absolute inset-y-0 left-0 rounded-l-md bg-black/60" style={{ width: `${percent(Number.isFinite(start) ? start : 0)}%` }} />
            <div className="pointer-events-none absolute inset-y-0 right-0 rounded-r-md bg-black/60" style={{ width: `${100 - percent(Number.isFinite(end) ? end : duration)}%` }} />
            {valid && <div className="pointer-events-none absolute inset-y-0 border-y-2 border-accent-blue" style={{ left: `${percent(start)}%`, right: `${100 - percent(end)}%` }} />}
            <div {...handleProps('start', start)} className="absolute inset-y-0 z-10 flex w-6 -translate-x-full cursor-ew-resize items-center justify-center rounded-l-md border-2 border-accent-blue bg-accent-blue/80 outline-offset-2 focus-visible:outline focus-visible:outline-2 focus-visible:outline-white" style={{ left: `${percent(Number.isFinite(start) ? start : 0)}%` }}><span className="h-6 border-l-2 border-white" /></div>
            <div {...handleProps('end', end)} className="absolute inset-y-0 z-10 flex w-6 cursor-ew-resize items-center justify-center rounded-r-md border-2 border-accent-blue bg-accent-blue/80 outline-offset-2 focus-visible:outline focus-visible:outline-2 focus-visible:outline-white" style={{ left: `${percent(Number.isFinite(end) ? end : duration)}%` }}><span className="h-6 border-l-2 border-white" /></div>
            <div aria-hidden="true" className="pointer-events-none absolute inset-y-0 border-l-2 border-white" style={{ left: `${percent(playhead)}%` }} />
          </div>
          <div className="mx-3 mt-1 flex justify-between text-[10px] tabular-nums text-text-muted"><span>{clock(viewStart)}</span><span>{clock(viewEnd)}</span></div>
          <div className="mt-3 grid grid-cols-2 gap-3">
            <label className="text-xs text-text-secondary">Start (seconds)<input type="number" min={0} max={duration} step="0.01" value={Number.isFinite(start) ? Number(start.toFixed(3)) : ''} disabled={busy}
              onChange={event => { media.current?.pause(); setStart(event.target.value === '' ? NaN : Number(event.target.value)) }}
              className="mt-1 w-full rounded-lg border border-border bg-bg-tertiary px-3 py-2 text-sm text-text-primary" /></label>
            <label className="text-xs text-text-secondary">End (seconds)<input type="number" min={0} max={duration} step="0.01" value={Number.isFinite(end) ? Number(end.toFixed(3)) : ''} disabled={busy}
              onChange={event => { media.current?.pause(); setEnd(event.target.value === '' ? NaN : Number(event.target.value)) }}
              className="mt-1 w-full rounded-lg border border-border bg-bg-tertiary px-3 py-2 text-sm text-text-primary" /></label>
          </div>
          <div className="my-3 flex flex-wrap items-center gap-2">
            {[3, 5, 10].map(length => <button key={length} disabled={busy || !Number.isFinite(start) || start < 0 || start >= duration} className={smallButton}
              onClick={() => { media.current?.pause(); setEnd(Math.min(duration, start + length)); seek(start); setZoom(null) }}>{length}s</button>)}
            <button disabled={busy || !valid} className={smallButton} onClick={() => { const padding = Math.max(0.5, (end - start) * 0.25); setZoom([Math.max(0, start - padding), Math.min(duration, end + padding)]) }}>Zoom to selection</button>
            {zoom && <button disabled={busy} className={smallButton} onClick={() => setZoom(null)}>Show full timeline</button>}
          </div>
          <p className={`text-xs ${valid ? 'text-text-secondary' : 'text-red-400'}`} role={valid ? undefined : 'alert'}>{valid ? `Selected ${clock(start)}–${clock(end)} · ${(end - start).toFixed(2)} seconds` : 'Choose a start and end within the clip, at least 0.05 seconds apart.'}</p>
        </>}
        {previewError && <p className="mt-2 text-xs text-text-muted">{previewError}</p>}
        {(error || destinationError) && <p role="alert" className="mt-3 text-xs text-red-400">{destinationError || error}</p>}
        <div className="mt-4 flex flex-wrap justify-end gap-2 border-t border-border pt-3">
          <button disabled={busy} onClick={onClose} className={smallButton}>Cancel</button>
          <button disabled={busy || receiving || !!destinationError} onClick={() => void submitClip(true)} className={smallButton}>Use full clip</button>
          <button disabled={busy || receiving || !valid || !!destinationError} onClick={() => void submitClip(fullSelection)} className={`flex items-center gap-2 rounded-lg bg-accent-blue px-4 py-2 text-xs text-white hover:bg-accent-blue-hover ${mutedClass}`}>
            {busy && <Loader2 size={14} className="animate-spin" />}{busy ? 'Preparing clip…' : fullSelection ? 'Use clip' : 'Use selection'}
          </button>
        </div>
      </div>
    </div>, document.body,
  )
}
