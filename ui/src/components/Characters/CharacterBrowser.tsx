import { useCallback, useEffect, useState } from 'react'
import { Loader2, RefreshCw, Search, Trash2 } from 'lucide-react'
import * as api from '../../api/client'
import type { SavedOmniCharacter } from '../../types'
import { useStore } from '../../stores/useStore'
import { characterDisplayName } from '../../lib/characters'
import { CharacterVoiceButton, ExportCharacterButton, ImportCharacterButton } from './CharacterFileActions'
import { CharacterImagesButton } from './CharacterImages'

const REFMOD_COLLECTION_URL = 'https://huggingface.co/malcolmrey/minimaxh3/tree/main'

export function CharacterBrowser() {
  const [characters, setCharacters] = useState<SavedOmniCharacter[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [query, setQuery] = useState('')
  const [filter, setFilter] = useState('all')
  const [url, setUrl] = useState('')
  const [files, setFiles] = useState<string[]>([])
  const [fileSearch, setFileSearch] = useState('')
  const [filename, setFilename] = useState('')
  const [finding, setFinding] = useState(false)
  const [importing, setImporting] = useState(false)
  const [message, setMessage] = useState('')
  const [remoteError, setRemoteError] = useState('')
  const [confirmDelete, setConfirmDelete] = useState<string | null>(null)
  const refresh = useCallback(() => {
    setLoading(true)
    setError('')
    void api.fetchCharacters().then(setCharacters).catch(err => setError(String(err))).finally(() => setLoading(false))
  }, [])
  useEffect(() => {
    refresh()
    window.addEventListener('maestro-characters-changed', refresh)
    window.addEventListener('focus', refresh)
    return () => {
      window.removeEventListener('maestro-characters-changed', refresh)
      window.removeEventListener('focus', refresh)
    }
  }, [refresh])

  const changeUrl = (value: string) => {
    setUrl(value)
    setFiles([])
    setFilename('')
    setFileSearch('')
    setMessage('')
    setRemoteError('')
  }
  const find = async (sourceUrl = url) => {
    setFinding(true)
    setRemoteError('')
    setMessage('')
    setFiles([])
    setFilename('')
    try {
      const result = await api.findCharacterFiles(sourceUrl)
      setFiles(result.files)
      setFilename(result.files.length === 1 ? result.files[0] : '')
    } catch (err) {
      setRemoteError(err instanceof Error ? err.message : 'Could not list character files.')
    } finally { setFinding(false) }
  }
  const importRemote = async () => {
    setImporting(true)
    setRemoteError('')
    try {
      const character = await api.importCharacterUrl(url, filename, setMessage)
      setMessage(`Imported ${characterDisplayName(character.name)}${character.voice ? ' with voice' : ''}.`)
      refresh()
    } catch (err) {
      setRemoteError(err instanceof Error ? err.message : 'Character import failed.')
      setMessage('')
    } finally { setImporting(false) }
  }
  const remove = async (character: SavedOmniCharacter) => {
    const state = useStore.getState()
    if ((state.params.minimax_h3_references ?? []).some(reference => reference.library_character_id === character.id)
        || state.directorH3References.some(reference => reference.library_character_id === character.id)
        || state.ttsVoices.some(voice => voice.characterId === character.id)) {
      setError(`Remove ${characterDisplayName(character.name)} from the current references and TTS voices before deleting it.`)
      return
    }
    if (confirmDelete !== character.id) { setConfirmDelete(character.id); return }
    try {
      await api.deleteCharacter(character.id)
      window.dispatchEvent(new Event('maestro-characters-changed'))
    } catch (err) { setError(err instanceof Error ? err.message : 'Could not delete character.') }
    finally { setConfirmDelete(null) }
  }
  const visible = characters.filter(character => {
    if (filter === 'voice' && !character.voice) return false
    if (filter === 'refmod' && !character.refmod) return false
    if (filter === 'no_voice' && character.voice) return false
    return `${characterDisplayName(character.name)} ${character.name} ${character.description || ''}`.toLowerCase().includes(query.toLowerCase())
  })
  const matchingFiles = files.filter(file => `${characterDisplayName(file)} ${file}`.toLowerCase().includes(fileSearch.toLowerCase()))

  return <section aria-label="Character browser" tabIndex={0} className="h-full min-h-0 overflow-y-auto overscroll-contain focus:outline-none">
  <div className="mx-auto max-w-6xl p-3 sm:p-5 space-y-5">
    <div>
      <h2 className="text-base font-semibold text-text-primary">Characters & H3 RefMods</h2>
      <p className="mt-1 text-xs text-text-muted">Import a shared character, add a voice, or export your own. Saved characters appear in Reference mode and their voices are available in TTS.</p>
    </div>
    <div className="rounded-xl border border-border bg-bg-secondary p-3 space-y-3">
      <ImportCharacterButton />
      <p className="text-[11px] text-text-muted">Share one file such as blaine.maestro.safetensors with appearance and any saved voice embedded. Standard H3 RefMods are also supported.</p>
      <div className="border-t border-border pt-3 space-y-2">
        <label htmlFor="character-import-url" className="block text-xs text-text-secondary">Import from Hugging Face</label>
        <div className="flex flex-wrap items-center gap-2 text-[11px]">
          <span className="text-text-muted">Saved collection</span>
          <button type="button" disabled={finding || importing} title={REFMOD_COLLECTION_URL}
            onClick={() => { changeUrl(REFMOD_COLLECTION_URL); void find(REFMOD_COLLECTION_URL) }}
            className="rounded-md border border-border px-2 py-1.5 text-accent-blue hover:bg-bg-hover disabled:opacity-40">
            malcolmrey / MiniMax H3
          </button>
        </div>
        <div className="flex gap-2 min-w-0">
          <input id="character-import-url" value={url} disabled={finding || importing}
            onChange={event => changeUrl(event.target.value)}
            onKeyDown={event => { if (event.key === 'Enter' && url.trim() && !finding && !importing) void find() }}
            placeholder="Repository or .safetensors file URL" className="min-w-0 flex-1 rounded-lg border border-border bg-bg-primary px-2 py-2 text-base sm:text-xs text-text-primary" />
          <button type="button" onClick={() => void find()} disabled={!url.trim() || finding || importing}
            className="shrink-0 rounded-lg bg-accent-blue px-3 text-xs text-white disabled:opacity-40">
            {finding ? <Loader2 size={14} className="animate-spin" /> : 'Find files'}
          </button>
        </div>
        {files.length > 0 && <div className="space-y-2">
          {files.length > 1 && <input aria-label="Filter remote character files" value={fileSearch} onChange={event => setFileSearch(event.target.value)}
            placeholder={`Search ${files.length} files…`} className="w-full min-w-0 rounded border border-border bg-bg-primary px-2 py-1.5 text-base sm:text-xs text-text-primary" />}
          <select aria-label="Character file to import" value={filename} disabled={importing}
            onChange={event => setFilename(event.target.value)} className="w-full min-w-0 rounded border border-border bg-bg-primary px-2 py-2 text-base sm:text-xs text-text-primary">
            <option value="">Choose a character / RefMod file</option>
            {[...new Set([...(filename ? [filename] : []), ...matchingFiles])].map(file => <option value={file} key={file} title={file}>{characterDisplayName(file)}</option>)}
          </select>
          <button type="button" onClick={() => void importRemote()} disabled={!filename || importing}
            className="inline-flex gap-1.5 items-center rounded bg-accent-blue px-3 py-2 text-xs text-white disabled:opacity-40">
            {importing && <Loader2 size={12} className="animate-spin" />}Import selected character
          </button>
        </div>}
        {message && <p role="status" className="text-xs text-text-secondary break-words">{message}</p>}
        {remoteError && <p role="alert" className="text-xs text-indicator-error break-words">{remoteError}</p>}
      </div>
    </div>
    <div className="flex flex-wrap items-center gap-2">
      <div className="relative min-w-0 flex-1 basis-44">
        <Search size={13} className="absolute left-2 top-2.5 text-text-muted" />
        <input aria-label="Search saved characters" value={query} onChange={event => setQuery(event.target.value)} placeholder="Search characters…"
          className="w-full rounded-lg border border-border bg-bg-secondary py-2 pl-7 pr-2 text-base sm:text-xs text-text-primary" />
      </div>
      <select aria-label="Character filter" value={filter} onChange={event => setFilter(event.target.value)}
        className="rounded-lg border border-border bg-bg-secondary px-2 py-2 text-xs text-text-primary">
        <option value="all">All characters</option><option value="voice">With voice</option>
        <option value="refmod">H3 RefMods</option><option value="no_voice">Without voice</option>
      </select>
      <button type="button" onClick={refresh} aria-label="Refresh characters" className="p-2 text-text-muted"><RefreshCw size={14} /></button>
    </div>
    {error && <p role="alert" className="text-xs text-indicator-error">{error}</p>}
    {loading && characters.length === 0 && <p role="status" className="text-xs text-text-muted">Loading characters…</p>}
    {!loading && visible.length === 0 && <p className="text-xs text-text-muted">{characters.length ? 'No characters match this filter.' : 'No saved characters yet. Import a file above or save a character in Reference mode.'}</p>}
    <div className="grid grid-cols-1 min-[420px]:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-3">
      {visible.map(character => <article key={character.id} className="min-w-0 overflow-hidden rounded-xl border border-border bg-bg-secondary">
        {character.visual.type === 'image'
          ? <img src={character.visual.thumbnail_url || character.visual.url} alt={characterDisplayName(character.name)} className="w-full aspect-video object-contain bg-bg-primary" loading="lazy" />
          : <video src={character.visual.url} poster={character.visual.thumbnail_url} aria-label={`${characterDisplayName(character.name)} video preview`} controls muted playsInline preload="none" className="w-full aspect-video bg-bg-primary" />}
        <div className="p-3 space-y-2">
          <div className="flex gap-2 items-start"><h3 className="min-w-0 flex-1 text-sm font-medium text-text-primary break-words">{characterDisplayName(character.name)}</h3>
            <button type="button" onClick={() => void remove(character)} aria-label={`Delete ${characterDisplayName(character.name)}`} className="text-text-muted hover:text-indicator-error text-[10px]">
              {confirmDelete === character.id ? 'Confirm delete' : <Trash2 size={12} />}
            </button>
          </div>
          <p className="text-[10px] text-text-muted">{character.refmod ? 'H3 RefMod' : 'Saved reference'} · {character.voice ? 'With voice' : 'No voice'}</p>
          <CharacterImagesButton character={character} />
          {character.description && <p className="text-[11px] text-text-secondary line-clamp-3 break-words">{character.description}</p>}
          {character.voice && <audio key={character.updated_at} controls preload="none" src={character.voice.url} aria-label={`${characterDisplayName(character.name)} voice preview`} className="w-full h-8" />}
          <div className="flex flex-wrap items-start gap-2"><ExportCharacterButton character={character} /><CharacterVoiceButton character={character} /></div>
        </div>
      </article>)}
    </div>
  </div>
  </section>
}
