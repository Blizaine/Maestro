import { useCallback, useEffect, useState } from 'react'
import { Loader2, Search, X } from 'lucide-react'
import { fetchMusicStyles } from '../../api/musicTraining'
import type { MusicStyle } from '../../api/musicTraining'
import { deselectMusicStyle, selectMusicStyle, selectedMusicStyles, setMusicStyleStrength, toggleMusicStyle } from '../../lib/musicStyles'
import { useStore } from '../../stores/useStore'
import { LoraSortToggle } from '../SettingsDrawer/LoraSelector'
import { MyMusicDialog } from './MyMusicDialog'

export function Yue2LoraSelector() {
  const params = useStore(s => s.params)
  const instrumental = useStore(s => s.musicInstrumental)
  const sort = useStore(s => s.loraPickerSort)
  const setSort = useStore(s => s.setLoraPickerSort)
  const [styles, setStyles] = useState<MusicStyle[]>([])
  const [open, setOpen] = useState(false)
  const [search, setSearch] = useState('')
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const refresh = useCallback(() => {
    void fetchMusicStyles(true).then(value => {setStyles(value.styles); setError('')})
      .catch(reason => setError(reason instanceof Error ? reason.message : 'Could not load music LoRAs'))
      .finally(() => setLoading(false))
  }, [])
  useEffect(() => {
    refresh()
    window.addEventListener('maestro:music-styles-changed', refresh)
    window.addEventListener('focus', refresh)
    return () => {
      window.removeEventListener('maestro:music-styles-changed', refresh)
      window.removeEventListener('focus', refresh)
    }
  }, [refresh])
  const selected = selectedMusicStyles(params.custom_settings)
  const ids = new Set(selected.map(item => item.id))
  // Keep an adapter loaded from a job or training visible until unchecked,
  // even if it has not been added to the user's shortlist.
  const available = styles.filter(style => (!style.archived && style.in_selector) || ids.has(style.id))
  const filtered = available.filter(style => `${style.name} ${style.trigger}`.toLowerCase().includes(search.trim().toLowerCase()))
    .sort((a, b) => (sort === 'newest' ? (b.created_at || 0) - (a.created_at || 0) : 0) || a.name.localeCompare(b.name))
  const knownPairs = new Set(styles.filter(style => ids.has(style.id) && style.tokenizer_revision).map(style => style.tokenizer_pair || 'v4'))
  return <div className="min-w-0" aria-label="YuE2 music LoRA">
    <div className="mb-2 flex items-center justify-between gap-2">
      <span className="text-[10px] uppercase tracking-wider text-text-muted">LoRAs</span>
      <div className="flex items-center gap-3">
        <LoraSortToggle sort={sort} onChange={setSort}/>
        <button type="button" className="text-[10px] text-text-muted transition-colors hover:text-accent-blue"
          onClick={() => setOpen(true)}>My music</button>
      </div>
    </div>
    <div className="relative mb-2">
      <Search size={12} aria-hidden="true" className="absolute left-2.5 top-1/2 -translate-y-1/2 text-text-muted"/>
      <input type="search" aria-label="Search available music LoRAs" placeholder="Search LoRAs..." value={search}
        onChange={event => setSearch(event.target.value)}
        className="w-full rounded-lg border border-border bg-bg-tertiary py-1.5 pl-7 pr-3 text-xs text-text-primary placeholder:text-text-muted focus:border-accent-blue focus:outline-none"/>
    </div>
    {instrumental && <p className="mb-2 text-xs text-accent-blue">Instrumental LoRA · active at strength 1</p>}
    <fieldset disabled={instrumental} className="min-w-0 disabled:opacity-50">
    <div role="group" aria-label="Available music LoRAs" className="max-h-[120px] overflow-y-auto rounded-lg border border-border bg-bg-tertiary">
      {filtered.map(style => <label key={style.id} title={style.name}
        className={`flex w-full cursor-pointer items-center gap-2 px-2.5 py-1.5 text-left text-xs transition-colors hover:bg-bg-hover ${ids.has(style.id) ? 'text-accent-blue' : 'text-text-secondary'}`}>
        <input type="checkbox" checked={ids.has(style.id)} onChange={event => toggleMusicStyle(style, event.target.checked)}
          className="h-3.5 w-3.5 shrink-0 rounded border-border accent-accent-blue"/>
        <span className="min-w-0 flex-1 truncate">{style.name}</span>
      </label>)}
      {!filtered.length && <div className="flex items-center justify-center gap-2 px-3 py-3 text-center text-xs text-text-muted">
        {loading ? <><Loader2 size={12} className="animate-spin"/> Loading LoRAs...</> : search ? 'No matches' : 'Add your favorites here from My music → Saved LoRAs.'}
      </div>}
    </div>
    <p className="mt-2 text-[10px] text-text-muted">{instrumental ? 'Artist LoRAs are paused while Instrumental is on. Your selections are kept.' : 'Check one or more LoRAs to activate them.'}</p>
    {!instrumental && selected.length > 1 && <p className="mt-2 text-[11px] text-text-muted"><span className="text-accent-blue">Experimental mix.</span> LoRAs apply across the whole song. Describe who raps or sings each section in Music Style; voices may blend instead of switching.</p>}
    {!instrumental && knownPairs.size > 1 && <p role="alert" className="mt-2 text-xs text-red-400">Choose LoRAs trained with the same tokenizer version. v4 and v9 cannot be combined.</p>}
    {selected.length > 0 && <div className="mt-3 space-y-2">
      <div className="flex items-center justify-between">
        <span className="text-[10px] uppercase tracking-wider text-text-muted">Selected ({selected.length})</span>
        <button type="button" onClick={() => selectMusicStyle()} className="text-[10px] text-text-muted transition-colors hover:text-red-400">Clear all</button>
      </div>
      {selected.map(item => {
        const style = styles.find(style => style.id === item.id)
        const name = style?.name || `Music LoRA ${item.id}`
        return <div key={item.id} role="group" aria-label={`Selected ${name}`} className="rounded-lg border border-border bg-bg-tertiary px-2.5 py-2">
        <div className="mb-1.5 flex items-center justify-between gap-2">
          <p className="min-w-0 flex-1 truncate text-xs text-text-primary" title={name}>{name}</p>
          <button type="button" aria-label={`Disable ${name}`} title="Disable this LoRA" onClick={() => deselectMusicStyle(item.id)}
            className="shrink-0 rounded p-0.5 text-text-muted hover:text-red-400"><X size={12}/></button>
        </div>
        <div className="flex items-center gap-2">
          <input aria-label="Music LoRA strength" type="range" min={0} max={1.5} step={0.05}
            value={item.strength} className="min-w-0 flex-1"
            onChange={event => setMusicStyleStrength(item.id, Number(event.target.value))}/>
          <span className="w-8 shrink-0 text-right text-[10px] text-text-muted">{item.strength.toFixed(2)}</span>
        </div>
        {style?.trigger && <p className="mt-2 break-words text-[11px] text-text-muted">Trigger added automatically: <span className="text-text-primary">{style.trigger}</span></p>}
        {style && (!style.in_selector || style.archived) && <p className="mt-2 text-[10px] text-text-muted">Loaded for this job. Add it to your selector in My music to keep it available after unchecking.</p>}
      </div>})}
    </div>}
    </fieldset>
    {error && <p role="alert" className="mt-2 text-xs text-red-400">{error}</p>}
    {open && <MyMusicDialog onClose={() => {setOpen(false); refresh()}} onSelect={selectMusicStyle}/>}
  </div>
}
