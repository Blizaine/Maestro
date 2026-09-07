import { useEffect, useState } from 'react'
import { BookUser, ChevronDown, Loader2, Plus, RefreshCw, UserPlus } from 'lucide-react'
import * as api from '../../api/client'
import type { SavedOmniCharacter } from '../../types'
import { characterDisplayName } from '../../lib/characters'
import { ExportCharacterButton, ImportCharacterButton } from '../Characters/CharacterFileActions'

export function TtsCharacterLibrary({ onSelect, onCharactersChange, selectedIds, canAdd, disabled, switchToClone }: {
  onSelect: (character: SavedOmniCharacter) => void
  onCharactersChange: (characters: SavedOmniCharacter[]) => void
  selectedIds: string[]
  canAdd: boolean
  disabled: boolean
  switchToClone: boolean
}) {
  const [characters, setCharacters] = useState<SavedOmniCharacter[]>([])
  const [open, setOpen] = useState(true)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [refresh, setRefresh] = useState(0)
  const [formOpen, setFormOpen] = useState(false)
  const [name, setName] = useState('')
  const [visual, setVisual] = useState<File | null>(null)
  const [voice, setVoice] = useState<File | null>(null)
  const [saving, setSaving] = useState(false)

  useEffect(() => {
    let cancelled = false
    setLoading(true)
    void api.fetchCharacters().then(items => {
      if (cancelled) return
      setCharacters(items)
      onCharactersChange(items)
      setError('')
    }).catch(err => {
      if (!cancelled) setError(err instanceof Error ? err.message : 'Could not load saved characters.')
    }).finally(() => { if (!cancelled) setLoading(false) })
    return () => { cancelled = true }
  }, [refresh, onCharactersChange])

  useEffect(() => {
    const reload = () => setRefresh(value => value + 1)
    window.addEventListener('focus', reload)
    window.addEventListener('maestro-characters-changed', reload)
    return () => {
      window.removeEventListener('focus', reload)
      window.removeEventListener('maestro-characters-changed', reload)
    }
  }, [])

  const save = async () => {
    const isVideo = Boolean(visual && /\.(mp4|mov|mkv|webm|avi|m4v)$/i.test(visual.name))
    if (!name.trim() || !visual || (!isVideo && !/\.(png|jpe?g|webp|bmp|tiff?)$/i.test(visual.name))) {
      setError('Enter a name and choose a character image or video.')
      return
    }
    if (!voice && !isVideo) {
      setError('Choose a voice reference for this character.')
      return
    }
    setSaving(true)
    setError('')
    try {
      const visualUpload = await api.uploadImage(visual)
      const voiceUpload = voice ? await api.uploadAudio(voice) : null
      const character = await api.createCharacter({
        name: name.trim(), visual_path: visualUpload.path, visual_type: isVideo ? 'video' : 'image',
        voice_path: voiceUpload?.path, use_video_voice: isVideo && !voiceUpload,
      })
      const next = [...characters, character]
      setCharacters(next)
      onCharactersChange(next)
      setName('')
      setVisual(null)
      setVoice(null)
      setFormOpen(false)
      // Saving remains useful even when the active model has no free slot.
      if (canAdd) onSelect(character)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not save this character.')
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="rounded-lg border border-border bg-bg-tertiary/50 overflow-hidden">
      <button type="button" aria-expanded={open} onClick={() => setOpen(value => !value)}
        className="w-full flex items-center justify-between px-2.5 py-2 text-left hover:bg-bg-tertiary">
        <span className="flex items-center gap-1.5 text-[10px] font-medium text-text-primary">
          <BookUser size={13} className="text-accent-blue" /> Saved characters
          <span className="text-text-muted">{characters.length}</span>
        </span>
        <ChevronDown size={13} className={`text-text-muted ${open ? 'rotate-180' : ''}`} />
      </button>
      {open && (
        <div className="border-t border-border p-2 space-y-2">
          <p className="text-[10px] text-text-muted">
            Use a character's saved voice from Reference mode. Write their name before each line of dialogue, or use Speaker 1 and Speaker 2.
          </p>
          {switchToClone && <p className="text-[10px] text-text-secondary">Saved voices use Qwen3 Voice Cloning. Choosing a character switches to that model and keeps your script.</p>}
          {loading && <p className="text-[10px] text-text-muted">Loading characters…</p>}
          {!loading && characters.length === 0 && <p className="text-[10px] text-text-muted">No saved characters yet. Save one below or in Reference mode.</p>}
          <div className="space-y-1.5 max-h-64 overflow-y-auto">
            {characters.map(character => {
              const selected = selectedIds.includes(character.id)
              return (
                <div key={character.id} className="flex items-center gap-2 rounded-md border border-border p-1.5">
                  <img src={character.visual.thumbnail_url || character.visual.url} alt="" loading="lazy" className="w-8 h-8 shrink-0 rounded object-cover" />
                  <div className="min-w-0 flex-1">
                    <div className="truncate text-[10px] font-medium text-text-primary" title={characterDisplayName(character.name)}>{characterDisplayName(character.name)}</div>
                    <div className="text-[9px] text-text-muted">{character.voice ? 'Saved voice' : 'No saved voice reference'}</div>
                    <ExportCharacterButton character={character} />
                  </div>
                  <button type="button" aria-label={`${switchToClone ? 'Use Qwen3 Voice Cloning with' : 'Add voice for'} ${characterDisplayName(character.name)}`}
                    disabled={disabled || saving || selected || !character.voice?.path || !canAdd}
                    onClick={() => onSelect(character)}
                    title={switchToClone ? 'Switch to Qwen3 Voice Cloning and use this character' : `Use ${characterDisplayName(character.name)}'s saved voice`}
                    className="text-[10px] text-accent-blue px-1.5 py-1 rounded hover:bg-bg-hover disabled:opacity-40 disabled:cursor-not-allowed flex items-center gap-1">
                    {!selected && <Plus size={11} />}{selected ? 'Added' : switchToClone ? 'Use with clone' : 'Add voice'}
                  </button>
                </div>
              )
            })}
          </div>
          <ImportCharacterButton disabled={disabled || saving} />
          {!canAdd && !switchToClone && <p className="text-[9px] text-text-muted">Replace a character in a voice slot below, or remove a voice to add another.</p>}
          <div className="flex items-center justify-between gap-2">
            <button type="button" disabled={disabled || saving} onClick={() => setFormOpen(value => !value)}
              className="flex items-center gap-1 text-[10px] text-accent-blue disabled:opacity-40">
              <UserPlus size={12} />{formOpen ? 'Close new character' : 'Save a new character'}
            </button>
            <button type="button" aria-label="Refresh saved characters" disabled={loading || saving} onClick={() => setRefresh(value => value + 1)} className="text-text-muted hover:text-text-primary">
              <RefreshCw size={12} />
            </button>
          </div>
          {formOpen && (
            <fieldset disabled={disabled || saving} className="space-y-2 rounded-md bg-bg-primary p-2 disabled:opacity-50">
              <input aria-label="New character name" placeholder="Character name" value={name} onChange={event => setName(event.target.value)}
                className="w-full rounded border border-border bg-bg-tertiary px-2 py-1 text-[11px] text-text-primary" />
              <label className="block text-[10px] text-text-secondary">Image or video
                <input aria-label="New character image or video" type="file" onChange={event => setVisual(event.target.files?.[0] ?? null)} className="block w-full mt-1 text-[10px]" />
              </label>
              <label className="block text-[10px] text-text-secondary">Voice audio or video
                <input aria-label="New character voice" type="file" onChange={event => setVoice(event.target.files?.[0] ?? null)} className="block w-full mt-1 text-[10px]" />
              </label>
              <p className="text-[9px] text-text-muted">Use at least two seconds of clear speech. A character video supplies its own audio when no separate voice file is selected. Saved characters are also available in Reference mode.</p>
              <button type="button" onClick={() => void save()} className="w-full flex items-center justify-center gap-1 rounded bg-accent-blue py-1.5 text-[10px] text-white">
                {saving && <Loader2 size={11} className="animate-spin" />}{saving ? 'Saving…' : canAdd ? 'Save and add voice' : 'Save character'}
              </button>
            </fieldset>
          )}
          {error && <p role="alert" className="text-[10px] text-indicator-error">{error}</p>}
        </div>
      )}
    </div>
  )
}
