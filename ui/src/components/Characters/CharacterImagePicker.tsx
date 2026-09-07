import { useEffect, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import { ArrowLeft, Check, Loader2, Search, UserRound, X } from 'lucide-react'
import * as api from '../../api/client'
import type { CharacterImageViews, SavedOmniCharacter } from '../../types'
import { characterDisplayName } from '../../lib/characters'
import { ImportCharacterButton } from './CharacterFileActions'

type ImageView = CharacterImageViews['items'][number]
type Props = {
  maxImages?: number | null
  disabled?: boolean
  label?: string
  onSelect: (character: SavedOmniCharacter, images: ImageView[]) => Promise<void>
}

export function CharacterImagePickerButton({maxImages, disabled, label = 'Add character', onSelect}: Props) {
  const [open, setOpen] = useState(false)
  return <>
    <button type="button" disabled={disabled || maxImages === 0} onClick={() => setOpen(true)}
      className="flex min-h-10 items-center justify-center gap-2 rounded-xl border border-border px-3 text-xs text-text-secondary hover:border-accent-blue disabled:opacity-40">
      <UserRound size={15}/>{label}
    </button>
    {open && <CharacterImagePicker maxImages={maxImages} onSelect={onSelect} onClose={() => setOpen(false)}/>}
  </>
}

function CharacterImagePicker({maxImages, onSelect, onClose}: Props & {onClose: () => void}) {
  const [characters, setCharacters] = useState<SavedOmniCharacter[]>([])
  const [character, setCharacter] = useState<SavedOmniCharacter | null>(null)
  const [selected, setSelected] = useState<string[]>([])
  const [query, setQuery] = useState('')
  const [busy, setBusy] = useState(false)
  const [message, setMessage] = useState('Loading characters…')
  const [error, setError] = useState('')
  const dialog = useRef<HTMLDivElement>(null)
  const alive = useRef(true)
  const limit = maxImages == null ? 128 : Math.max(0, maxImages)
  useEffect(() => {
    alive.current = true
    const trigger = document.activeElement as HTMLElement | null
    dialog.current?.focus()
    const refresh = () => void api.fetchCharacters().then(items => {
      if (alive.current) {setCharacters(items); setMessage('')}
    }).catch(err => {if (alive.current) {setError(String(err)); setMessage('')}})
    refresh()
    window.addEventListener('maestro-characters-changed', refresh)
    return () => {alive.current = false; window.removeEventListener('maestro-characters-changed', refresh); trigger?.focus()}
  }, [])
  const choose = async (item: SavedOmniCharacter) => {
    setBusy(true); setError(''); setMessage('Preparing character images…')
    try {
      const ready = item.image_views ? item : await api.recoverCharacterImages(item.id, text => {if (alive.current) setMessage(text)})
      if (alive.current) {
        setCharacter(ready)
        setSelected([ready.image_views?.cover_id || ready.image_views?.items[0]?.id || ''].filter(Boolean))
      }
    } catch (err) {if (alive.current) setError(err instanceof Error ? err.message : 'Could not prepare character images.')}
    finally {if (alive.current) {setBusy(false); setMessage('')}}
  }
  const add = async () => {
    if (!character?.image_views || !selected.length || selected.length > limit) return
    setBusy(true); setError(''); setMessage('Adding character images…')
    try {
      // Preserve the visible selection order; the cover is the default first ref.
      await onSelect(character, selected.map(id => character.image_views!.items.find(item => item.id === id)!))
      if (alive.current) onClose()
    } catch (err) {if (alive.current) setError(err instanceof Error ? err.message : 'Could not add this character.')}
    finally {if (alive.current) {setBusy(false); setMessage('')}}
  }
  const visible = characters.filter(item => `${item.name} ${characterDisplayName(item.name)}`.toLocaleLowerCase().includes(query.toLocaleLowerCase()))
  return createPortal(<div className="fixed inset-0 z-[10000] flex items-center justify-center bg-black/75 p-2 sm:p-6"
    onClick={event => {if (event.target === event.currentTarget) onClose()}}>
    <div role="dialog" aria-modal="true" aria-labelledby="character-image-picker-title" tabIndex={-1} ref={dialog}
      className="flex max-h-[90dvh] w-full max-w-2xl min-w-0 flex-col overflow-hidden rounded-2xl border border-border bg-bg-primary shadow-2xl outline-none"
      onKeyDown={event => {
        if (event.key === 'Escape') {event.stopPropagation(); onClose()}
        if (event.key === 'Tab') {
          const nodes = dialog.current?.querySelectorAll<HTMLElement>('button:not(:disabled), input:not(:disabled), a[href]')
          if (!nodes?.length) return
          const first = nodes[0], last = nodes[nodes.length - 1]
          if (event.shiftKey && (document.activeElement === first || document.activeElement === dialog.current)) {event.preventDefault(); last.focus()}
          else if (!event.shiftKey && document.activeElement === last) {event.preventDefault(); first.focus()}
        }
      }}>
      <div className="flex shrink-0 items-center gap-2 border-b border-border p-3">
        {character && <button type="button" disabled={busy} aria-label="Back to characters" onClick={() => {setCharacter(null); setError('')}} className="p-2 text-text-secondary"><ArrowLeft size={18}/></button>}
        <h2 id="character-image-picker-title" className="min-w-0 flex-1 break-words text-sm font-semibold text-text-primary">{character ? characterDisplayName(character.name) : 'Choose a character'}</h2>
        <button type="button" aria-label="Close character picker" onClick={onClose} className="p-2 text-text-secondary"><X size={20}/></button>
      </div>
      <div className="min-h-0 overflow-y-auto overscroll-contain space-y-3 p-3 sm:p-4">
        {message && <p role="status" className="flex items-center gap-2 text-xs text-text-secondary"><Loader2 size={14} className="animate-spin"/>{message}</p>}
        {error && <p role="alert" className="text-xs text-indicator-error">{error}</p>}
        {character?.image_views ? <>
          <p className="text-xs text-text-muted">{limit === 1 ? 'Choose the character image to use.' : `Choose reference views${limit < 128 ? ` (up to ${limit})` : ''}.`} These images supply appearance; describe clothing and other changes in your prompt.</p>
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
            {character.image_views.items.map((item, index) => <button type="button" key={item.id} aria-pressed={selected.includes(item.id)}
              aria-label={`Use character view ${index + 1}`} disabled={busy || (!selected.includes(item.id) && selected.length >= limit && limit !== 1)}
              onClick={() => setSelected(limit === 1 ? [item.id] : selected.includes(item.id) ? selected.filter(id => id !== item.id) : [...selected, item.id])}
              className={`relative overflow-hidden rounded-xl border bg-bg-secondary text-text-primary disabled:opacity-40 ${selected.includes(item.id) ? 'border-accent-blue' : 'border-border'}`}>
              <img src={item.url} alt="" loading="lazy" className="aspect-square w-full object-contain bg-black"/>
              <span className="flex min-h-10 items-center justify-between px-2 text-xs">View {index + 1}{selected.includes(item.id) && <Check size={14} className="text-accent-blue"/>}</span>
            </button>)}
          </div>
        </> : <>
          <div className="relative"><Search size={14} className="absolute left-3 top-3 text-text-muted"/><input aria-label="Search characters for image" value={query} onChange={event => setQuery(event.target.value)} placeholder="Search characters…" className="min-h-11 w-full rounded-xl border border-border bg-bg-secondary pl-9 pr-3 text-base sm:text-sm text-text-primary"/></div>
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
            {visible.map(item => <button key={item.id} type="button" disabled={busy} onClick={() => void choose(item)} aria-label={`Choose ${characterDisplayName(item.name)}`}
              className="min-w-0 overflow-hidden rounded-xl border border-border bg-bg-secondary text-left hover:border-accent-blue disabled:opacity-40">
              <img src={item.visual.thumbnail_url || item.visual.url} alt="" loading="lazy" className="aspect-[4/3] w-full object-contain bg-black"/>
              <span className="block break-words p-2 text-xs text-text-primary">{characterDisplayName(item.name)}</span>
            </button>)}
          </div>
          {!message && !visible.length && <p className="text-xs text-text-muted">No matching characters. Save one in Reference mode or import a character file.</p>}
          <ImportCharacterButton disabled={busy}/>
        </>}
      </div>
      {character && <div className="flex shrink-0 items-center justify-between gap-2 border-t border-border p-3">
        <span className="text-xs text-text-muted">{selected.length} selected</span>
        <button type="button" disabled={busy || !selected.length || selected.length > limit} onClick={() => void add()} className="min-h-11 rounded-xl bg-accent-blue px-4 text-xs text-white disabled:opacity-40">Use {selected.length === 1 ? 'this image' : `${selected.length} images`}</button>
      </div>}
    </div>
  </div>, document.body)
}
