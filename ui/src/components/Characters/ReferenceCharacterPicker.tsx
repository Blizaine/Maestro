import { useEffect, useRef, useState } from 'react'
import { Check, Mic, MoreHorizontal, Plus, Search, Trash2, UserRound, X } from 'lucide-react'
import type { SavedOmniCharacter } from '../../types'
import { characterDisplayName } from '../../lib/characters'
import { ExportCharacterButton } from './CharacterFileActions'

interface Props {
  characters: SavedOmniCharacter[]
  addedIds: string[]
  disabled: boolean
  onAdd: (character: SavedOmniCharacter) => void
  onDelete: (character: SavedOmniCharacter) => void
  scroll?: boolean
}

export function ReferenceCharacterPicker({ characters, addedIds, disabled, onAdd, onDelete, scroll = true }: Props) {
  const [query, setQuery] = useState('')
  const [optionsId, setOptionsId] = useState<string | null>(null)
  const optionsCard = useRef<HTMLElement>(null)

  useEffect(() => {
    if (!optionsId) return
    const onPointerDown = (event: PointerEvent) => {
      if (!optionsCard.current?.contains(event.target as Node)) setOptionsId(null)
    }
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key !== 'Escape') return
      optionsCard.current?.querySelector<HTMLButtonElement>('[data-character-options]')?.focus()
      setOptionsId(null)
    }
    document.addEventListener('pointerdown', onPointerDown)
    document.addEventListener('keydown', onKeyDown)
    return () => {
      document.removeEventListener('pointerdown', onPointerDown)
      document.removeEventListener('keydown', onKeyDown)
    }
  }, [optionsId])

  const search = query.trim().toLocaleLowerCase()
  const visible = characters.filter(character => (
    `${characterDisplayName(character.name)} ${character.name} ${character.description || ''}`.toLocaleLowerCase().includes(search)
  ))

  if (characters.length === 0) return (
    <div className="rounded-xl border border-dashed border-border px-4 py-6 text-center">
      <UserRound size={24} className="mx-auto mb-2 text-text-muted" />
      <p className="text-xs font-medium text-text-primary">Build your character library</p>
      <p className="mt-1 text-[11px] leading-relaxed text-text-muted">Save a person or import a character to use them in your scenes.</p>
    </div>
  )

  return <div className="space-y-3">
    <div className="relative">
      <Search size={14} className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-text-muted" />
      <input value={query} onChange={event => { setQuery(event.target.value); setOptionsId(null) }}
        aria-label="Search reference characters" placeholder="Find a character…"
        className="w-full min-w-0 rounded-lg border border-border bg-bg-primary py-2 pl-9 pr-9 text-base text-text-primary placeholder:text-text-muted focus:border-accent-blue focus:outline-none sm:text-xs" />
      {query && <button type="button" aria-label="Clear character search" onClick={() => setQuery('')}
        className="absolute right-0 top-0 flex h-full w-9 items-center justify-center text-text-muted hover:text-text-primary">
        <X size={14} />
      </button>}
    </div>

    <div role="group" aria-label="Saved character choices" tabIndex={0}
      className={`${scroll ? 'max-h-[min(26rem,50dvh)] overflow-y-auto' : ''} rounded-xl p-0.5 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent-blue/50`}>
      <div className="grid grid-cols-2 gap-2.5">
        {visible.map(character => {
          const name = characterDisplayName(character.name)
          const added = addedIds.includes(character.id)
          const optionsOpen = optionsId === character.id
          const thumbnail = character.visual.thumbnail_url || character.visual.url
          return <article key={character.id} ref={optionsOpen ? optionsCard : undefined}
            className={`relative min-w-0 rounded-xl ${optionsOpen ? 'z-10' : ''}`}>
            <button type="button" disabled={disabled || added}
              aria-label={added ? `${name} added to references` : `Add ${name} to references`}
              onClick={() => { setOptionsId(null); onAdd(character) }}
              className={`group block h-full w-full overflow-hidden rounded-xl border text-left transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent-blue focus-visible:ring-offset-2 focus-visible:ring-offset-bg-secondary ${
                added ? 'border-accent-blue bg-accent-blue/10 ring-1 ring-accent-blue/30'
                  : 'border-border bg-bg-primary enabled:hover:border-accent-blue/60 enabled:hover:bg-bg-hover'
              } ${disabled ? 'opacity-50' : ''}`}>
              <span className="relative block aspect-[5/4] overflow-hidden bg-bg-tertiary">
                <span className="absolute inset-0 flex items-center justify-center text-text-muted"><UserRound size={36} /></span>
                <img key={thumbnail} src={thumbnail} alt="" loading="lazy"
                  onError={event => { event.currentTarget.style.visibility = 'hidden' }}
                  className="relative h-full w-full object-cover object-[50%_30%] transition-transform duration-300 motion-safe:group-enabled:group-hover:scale-[1.03]" />
                <span className="pointer-events-none absolute inset-x-0 bottom-0 h-16 bg-gradient-to-t from-black/70 to-transparent" />
                {added && <span className="absolute left-2 top-2 flex h-6 w-6 items-center justify-center rounded-full bg-accent-blue text-white shadow-sm"><Check size={14} strokeWidth={3} /></span>}
                <span className="absolute bottom-2 left-2 inline-flex max-w-[calc(100%-1rem)] items-center gap-1 rounded-md bg-black/40 px-1.5 py-1 text-[10px] font-medium leading-none text-white backdrop-blur-sm">
                  {character.voice && <Mic size={11} className="shrink-0" />}{character.voice ? 'With voice' : 'Visual only'}
                </span>
              </span>
              <span className="block px-2.5 pb-2.5 pt-2">
                <span title={name} className="line-clamp-2 min-h-9 break-words text-xs font-semibold leading-[18px] text-text-primary">{name}</span>
                <span className="mt-1.5 flex items-center gap-1 text-[11px] font-medium text-accent-blue">
                  {added ? <Check size={13} /> : <Plus size={13} />}{added ? 'Added' : 'Add'}
                </span>
              </span>
            </button>

            <button type="button" data-character-options aria-label={`Options for ${name}`}
              aria-expanded={optionsOpen} aria-controls={`character-actions-${character.id}`} disabled={disabled}
              onClick={() => setOptionsId(current => current === character.id ? null : character.id)}
              className={`absolute right-1.5 top-1.5 flex h-9 w-9 items-center justify-center rounded-lg border border-white/15 text-white shadow-sm backdrop-blur-md transition-colors hover:bg-black/80 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent-blue disabled:opacity-40 ${optionsOpen ? 'bg-black/90' : 'bg-black/50'}`}>
              <MoreHorizontal size={18} />
            </button>
            {optionsOpen && <div id={`character-actions-${character.id}`} role="group" aria-label={`Actions for ${name}`}
              className="absolute inset-x-1.5 top-12 z-20 space-y-0.5 rounded-lg border border-border bg-bg-secondary p-1 shadow-xl">
              <ExportCharacterButton character={character} disabled={disabled}
                className="flex min-h-10 w-full items-center gap-2 rounded-md px-2 text-xs text-text-primary hover:bg-bg-hover disabled:opacity-40" />
              <button type="button" disabled={disabled} onClick={() => { setOptionsId(null); onDelete(character) }}
                className="flex min-h-10 w-full items-center gap-2 rounded-md px-2 text-xs text-indicator-error hover:bg-bg-hover disabled:opacity-40">
                <Trash2 size={13} className="shrink-0" /> Delete
              </button>
            </div>}
          </article>
        })}
      </div>
      {visible.length === 0 && <p role="status" className="px-3 py-7 text-center text-xs text-text-muted">No matching characters. Try another name.</p>}
    </div>
  </div>
}
