import { useEffect, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import { Check, Download, Images, Loader2, RefreshCw, Star, X } from 'lucide-react'
import * as api from '../../api/client'
import type { SavedOmniCharacter } from '../../types'
import { characterDisplayName } from '../../lib/characters'

export function CharacterImagesButton({ character }: { character: SavedOmniCharacter }) {
  const [open, setOpen] = useState(false)
  return <>
    <button type="button" onClick={() => setOpen(true)} aria-label={`Images for ${characterDisplayName(character.name)}`}
      className="inline-flex min-h-9 items-center gap-1.5 rounded-lg border border-border px-2.5 text-xs text-accent-blue hover:bg-bg-hover">
      <Images size={14} />{character.image_views ? `${character.image_views.items.length} ${character.image_views.items.length === 1 ? 'image' : 'images'}` : character.refmod ? 'Recover images' : 'Choose images'}
    </button>
    {open && <CharacterImages character={character} onClose={() => setOpen(false)} />}
  </>
}

export function CharacterImages({ character: initial, onClose }: { character: SavedOmniCharacter; onClose: () => void }) {
  const [character, setCharacter] = useState(initial)
  const [selected, setSelected] = useState(initial.image_views?.selected_ids || [])
  const [cover, setCover] = useState(initial.image_views?.cover_id || '')
  const [preview, setPreview] = useState(initial.image_views?.cover_id || '')
  const [busy, setBusy] = useState(false)
  const [message, setMessage] = useState('')
  const [error, setError] = useState('')
  const dialog = useRef<HTMLDivElement>(null)
  const alive = useRef(true)
  const started = useRef(false)
  const views = character.image_views
  const current = views?.items.find(item => item.id === preview) || views?.items[0]
  const apply = (result: SavedOmniCharacter) => {
    if (!alive.current) return
    setCharacter(result)
    setSelected(result.image_views?.selected_ids || [])
    setCover(result.image_views?.cover_id || '')
    setPreview(result.image_views?.cover_id || '')
  }
  const recover = async (force = false) => {
    setBusy(true); setError(''); setMessage('Preparing images…')
    try {
      apply(await api.recoverCharacterImages(initial.id, text => { if (alive.current) setMessage(text) }, force))
    } catch (err) {
      if (alive.current) setError(err instanceof Error ? err.message : 'Could not recover character images.')
    } finally {
      if (alive.current) { setBusy(false); setMessage('') }
    }
  }
  useEffect(() => {
    alive.current = true
    const trigger = document.activeElement as HTMLElement | null
    dialog.current?.focus()
    if (!started.current && !initial.image_views) { started.current = true; void recover() }
    return () => { alive.current = false; trigger?.focus() }
    // Recovery is a single background operation for this opened character.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])
  const toggle = (id: string) => {
    const next = selected.includes(id) ? selected.filter(value => value !== id) : [...selected, id]
    setSelected(next)
    if (!next.includes(cover)) setCover(next[0] || '')
  }
  const save = async (download = false) => {
    setBusy(true); setError(''); setMessage('Saving selection…')
    try {
      const result = await api.selectCharacterImages(initial.id, selected, cover)
      apply(result)
      if (download) {
        const link = document.createElement('a')
        link.href = api.characterImagesDownloadUrl(initial.id)
        link.download = ''
        document.body.appendChild(link); link.click(); link.remove()
      } else if (alive.current) onClose()
    } catch (err) {
      if (alive.current) setError(err instanceof Error ? err.message : 'Could not save image selection.')
    } finally { if (alive.current) { setBusy(false); setMessage('') } }
  }
  return createPortal(<div className="fixed inset-0 z-[10000] flex items-center justify-center bg-black/75 p-2 sm:p-6"
    onClick={event => { if (event.target === event.currentTarget) onClose() }}>
    <div ref={dialog} tabIndex={-1} role="dialog" aria-modal="true" aria-labelledby="character-images-title"
      className="flex max-h-[94dvh] w-full max-w-4xl min-w-0 flex-col overflow-hidden rounded-2xl border border-border bg-bg-primary shadow-2xl outline-none"
      onKeyDown={event => {
        if (event.key === 'Escape') { event.stopPropagation(); onClose() }
        if (event.key === 'Tab') {
          const nodes = dialog.current?.querySelectorAll<HTMLElement>('button:not(:disabled), input:not(:disabled), a[href], [tabindex="0"]')
          if (!nodes?.length) return
          const first = nodes[0], last = nodes[nodes.length - 1]
          if (event.shiftKey && (document.activeElement === first || document.activeElement === dialog.current)) { event.preventDefault(); last.focus() }
          else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus() }
        }
      }}>
      <div className="flex shrink-0 items-center gap-3 border-b border-border px-4 py-3">
        <Images size={21} className="shrink-0 text-accent-blue" />
        <div className="min-w-0 flex-1"><h2 id="character-images-title" className="text-base font-semibold text-text-primary break-words">{characterDisplayName(character.name)} · Images</h2>
          <p className="text-xs text-text-muted">Choose a cover and useful reference views.</p></div>
        <button type="button" aria-label="Close character images" onClick={onClose} className="p-2 text-text-secondary"><X size={20} /></button>
      </div>
      <div className="min-h-0 overflow-y-auto overscroll-contain p-3 sm:p-5 space-y-4">
        {busy && <p role="status" className="flex items-center gap-2 text-sm text-text-secondary"><Loader2 size={16} className="animate-spin shrink-0" />{message}</p>}
        {error && <p role="alert" className="text-sm text-indicator-error break-words">{error}</p>}
        {views && current && <>
          <div className="space-y-2">
            <img src={current.url} alt={`Preview ${Number(current.id.slice(-4))} for ${characterDisplayName(character.name)}`}
              className="max-h-[40dvh] w-full rounded-xl bg-black object-contain" />
            <div className="flex flex-wrap items-center gap-2 text-xs">
              <span className="mr-auto text-text-muted">{current.width} × {current.height} · PNG</span>
              <button type="button" disabled={busy} aria-pressed={cover === current.id} onClick={() => {
                setCover(current.id); if (!selected.includes(current.id)) setSelected([...selected, current.id])
              }} className="inline-flex min-h-10 items-center gap-1.5 rounded-lg border border-border px-3 text-text-primary disabled:opacity-40">
                <Star size={14} />{cover === current.id ? 'Cover image' : 'Use as cover'}
              </button>
              <a href={`${current.url}&download=true`} download className="inline-flex min-h-10 items-center gap-1.5 rounded-lg border border-border px-3 text-accent-blue"><Download size={14} />PNG</a>
            </div>
          </div>
          <p className="text-xs text-text-muted">{views.source === 'refmod' ? 'Recovered at the stored resolution, without added detail.' : 'Taken from the saved original media.'} Select images to include in downloads and character exports.</p>
          {views.notes?.map((note, index) => <p key={index} className="rounded-lg bg-bg-secondary p-2.5 text-xs text-text-secondary">{note}</p>)}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
            {views.items.map((item, index) => <div key={item.id} className={`min-w-0 overflow-hidden rounded-xl border ${preview === item.id ? 'border-accent-blue' : 'border-border'} bg-bg-secondary`}>
              <button type="button" onClick={() => setPreview(item.id)} aria-label={`Preview image ${index + 1}`} aria-pressed={preview === item.id} className="block w-full bg-black">
                <img src={item.url} alt="" loading="lazy" className="aspect-square w-full object-contain" />
              </button>
              <label className="flex min-h-11 cursor-pointer items-center gap-2 px-2 text-xs text-text-primary">
                <input type="checkbox" aria-label={`Select image ${index + 1}`} checked={selected.includes(item.id)} disabled={busy} onChange={() => toggle(item.id)} className="h-4 w-4 accent-blue-500" />
                <span>View {index + 1}</span>{cover === item.id && <Star size={12} className="ml-auto text-accent-blue" aria-label="Cover" />}
              </label>
            </div>)}
          </div>
        </>}
        <button type="button" disabled={busy} onClick={() => void recover(Boolean(views))} className="inline-flex min-h-10 items-center gap-1.5 text-xs text-text-muted hover:text-text-primary disabled:opacity-40">
          <RefreshCw size={13} />{views ? 'Recover again' : 'Recover images'}
        </button>
      </div>
      <div className="flex shrink-0 flex-wrap items-center justify-end gap-2 border-t border-border p-3">
        <span className="mr-auto text-xs text-text-muted">{selected.length} selected</span>
        <button type="button" disabled={busy || !selected.length} onClick={() => void save(true)} className="inline-flex min-h-10 items-center gap-1.5 rounded-lg border border-border px-3 text-xs text-text-primary disabled:opacity-40"><Download size={14} />Download selected</button>
        <button type="button" disabled={busy || !selected.length} onClick={() => void save()} className="inline-flex min-h-10 items-center gap-1.5 rounded-lg bg-accent-blue px-3 text-xs text-white disabled:opacity-40"><Check size={14} />Save selection</button>
      </div>
    </div>
  </div>, document.body)
}
