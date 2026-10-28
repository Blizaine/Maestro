import { useEffect, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import { Plus, X } from 'lucide-react'
import type { EditorAsset } from '../types'
import * as api from '../api/client'
import { formatEditorTime } from './editorUtils'
import { useEditorStore } from './useEditorStore'

export function EditorSourceViewer({ asset, onClose }: { asset: EditorAsset; onClose: () => void }) {
  const media = useRef<HTMLMediaElement | null>(null)
  const closeButton = useRef<HTMLButtonElement>(null)
  const [duration, setDuration] = useState(asset.type === 'image' ? 3600 : asset.duration)
  const [sourceIn, setSourceIn] = useState(0)
  const [sourceOut, setSourceOut] = useState(asset.type === 'image' ? 5 : asset.duration)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const workspace = useEditorStore(state => state.project?.workspace || 'default')
  const addMedia = useEditorStore(state => state.addMedia)
  const trackId = useEditorStore(state => state.selectedTrackId)
  useEffect(() => {
    const focused = document.activeElement as HTMLElement | null
    closeButton.current?.focus()
    return () => focused?.focus()
  }, [])
  useEffect(() => {
    if (asset.type === 'image' || asset.duration > 0) return
    let active = true
    void api.probeEditorMedia(asset, workspace).then(probe => {
      if (!active || !(probe.duration > 0)) return
      setDuration(probe.duration)
      setSourceOut(probe.duration)
    }).catch(() => { if (active) setError('Source details could not be read. Loading the media may still provide its duration.') })
    return () => { active = false }
  }, [asset, workspace])
  const loaded = () => {
    const seconds = media.current?.duration
    if (seconds && Number.isFinite(seconds)) {
      setDuration(seconds)
      setSourceOut(previous => previous > 0 ? Math.min(previous, seconds) : seconds)
    }
  }
  const valid = Number.isFinite(sourceIn) && Number.isFinite(sourceOut)
    && sourceIn >= 0 && sourceOut > sourceIn && sourceOut <= duration
  const insert = async () => {
    if (!valid) return
    setBusy(true)
    try {
      await addMedia(asset, undefined, trackId || undefined, { sourceIn, sourceOut })
      const issue = useEditorStore.getState().error
      if (issue) setError(issue)
      else onClose()
    } finally { setBusy(false) }
  }
  return createPortal(
    <div className="fixed inset-0 z-[240] flex items-center justify-center bg-black/65 p-3 backdrop-blur-sm"
      onKeyDown={event => {
        event.stopPropagation()
        if (event.key === 'Escape') onClose()
        if (event.key === 'Tab') {
          const controls = [...event.currentTarget.querySelectorAll<HTMLElement>('button:not([disabled]), input:not([disabled]), video[controls], audio[controls]')]
          const first = controls[0], last = controls.at(-1)
          if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last?.focus() }
          else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first?.focus() }
        }
      }}>
      <section role="dialog" aria-modal="true" aria-label="Source preview"
        className="flex max-h-[95dvh] w-full max-w-3xl flex-col overflow-hidden rounded-xl border border-border bg-bg-secondary shadow-2xl">
        <header className="flex shrink-0 items-center justify-between gap-3 border-b border-border px-4 py-3">
          <div className="min-w-0"><h2 className="truncate text-sm font-semibold">{asset.display_name || asset.name}</h2>
            <p className="mt-1 text-xs text-text-muted">Source preview · {asset.width > 0 ? `${asset.width} × ${asset.height} · ` : ''}{asset.fps > 0 ? `${asset.fps} fps · ` : ''}{asset.type}</p></div>
          <button ref={closeButton} type="button" onClick={onClose} aria-label="Close source preview" className="rounded-lg p-2 text-text-secondary hover:bg-bg-hover"><X size={18} /></button>
        </header>
        <div className="flex min-h-0 flex-1 items-center justify-center overflow-hidden bg-media-canvas p-3">
          {asset.type === 'image' ? <img src={asset.url} alt={asset.name} className="max-h-[55dvh] max-w-full object-contain" />
            : asset.type === 'video' ? <video ref={element => { media.current = element }} src={asset.url} controls playsInline preload="metadata" onLoadedMetadata={loaded} className="max-h-[55dvh] w-full" />
              : <audio ref={element => { media.current = element }} src={asset.url} controls preload="metadata" onLoadedMetadata={loaded} className="my-12 w-full" />}
        </div>
        <div className="shrink-0 space-y-3 border-t border-border p-4">
          <div className="flex flex-wrap items-end gap-3">
            <label className="flex-1 text-xs text-text-secondary">In · seconds
              <input aria-label="Source in" type="number" min={0} max={sourceOut} step={0.001} value={sourceIn}
                onChange={event => setSourceIn(Number(event.target.value))} disabled={asset.type === 'image'}
                className="mt-1 w-full rounded-lg border border-border bg-bg-tertiary px-3 py-2 text-sm disabled:opacity-40" />
            </label>
            <label className="flex-1 text-xs text-text-secondary">{asset.type === 'image' ? 'Duration' : 'Out'} · seconds
              <input aria-label="Source out" type="number" min={sourceIn} max={duration || undefined} step={0.001} value={sourceOut}
                onChange={event => setSourceOut(Number(event.target.value))}
                className="mt-1 w-full rounded-lg border border-border bg-bg-tertiary px-3 py-2 text-sm" />
            </label>
            {asset.type !== 'image' && <div className="flex gap-2">
              <button type="button" onClick={() => setSourceIn(media.current?.currentTime || 0)} className="rounded-lg border border-border px-3 py-2 text-xs hover:bg-bg-hover">Mark In</button>
              <button type="button" onClick={() => setSourceOut(media.current?.currentTime ?? duration)} className="rounded-lg border border-border px-3 py-2 text-xs hover:bg-bg-hover">Mark Out</button>
            </div>}
          </div>
          {error && <p role="alert" className="text-xs text-red-400">{error}</p>}
          <footer className="flex items-center justify-between gap-3">
            <span className="text-xs text-text-muted">{valid ? `${formatEditorTime(sourceOut - sourceIn)} selected` : 'Choose an In/Out range inside the source.'}</span>
            <button type="button" onClick={() => void insert()} disabled={!valid || busy}
              className="flex items-center gap-2 rounded-lg bg-accent-blue px-4 py-2 text-xs font-semibold text-white disabled:opacity-40"><Plus size={14} />{busy ? 'Adding…' : 'Insert at playhead'}</button>
          </footer>
        </div>
      </section>
    </div>, document.body,
  )
}
