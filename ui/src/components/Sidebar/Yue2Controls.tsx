import { useEffect, useRef, useState } from 'react'
import { useStore } from '../../stores/useStore'
import { uploadAudio } from '../../api/client'
import { AutoGrowTextarea } from './MusicControls'
import { MyMusicDialog } from './MyMusicDialog'
import { fetchMusicStyles } from '../../api/musicTraining'
import type { MusicStyle } from '../../api/musicTraining'

export function Yue2Controls() {
  const params = useStore(s => s.params)
  const setParam = useStore(s => s.setParam)
  const input = useRef<HTMLInputElement>(null)
  const [uploading, setUploading] = useState(false)
  const [error, setError] = useState('')
  const [myMusic, setMyMusic] = useState(false)
  const [styles, setStyles] = useState<MusicStyle[]>([])
  useEffect(() => {void fetchMusicStyles().then(value => setStyles(value.styles)).catch(() => {})}, [myMusic])
  const selectStyle = (style?: MusicStyle) => {
    setParam('custom_settings', {...params.custom_settings, artist_id: style?.id || '', artist_strength: 1, abc: ''})
    if (style) {setParam('model_mode', 2); setParam('audio_prompt_type', ''); setParam('audio_guide', undefined)}
  }
  const mode = params.model_mode ?? 2
  useEffect(() => {
    // Keep the displayed default in the request, including before model
    // options load. Zero is an explicit Melody and chords selection.
    if (params.model_mode == null) setParam('model_mode', 2)
  }, [params.model_mode, setParam])
  const source = params.audio_prompt_type === 'A' ? params.audio_guide : ''
  return <div className="space-y-2 border-t border-border pt-3">
    <div className="flex items-end gap-2">
      <label className="min-w-0 flex-1 text-[11px] text-text-muted">Music style
        <select aria-label="Saved music style" className="mt-1 w-full rounded-lg border border-border bg-bg-tertiary p-2 text-xs text-text-primary"
          value={String(params.custom_settings?.artist_id || '')} onChange={event => selectStyle(styles.find(style => style.id === event.target.value))}>
          <option value="">Base YuE2</option>{styles.map(style => <option key={style.id} value={style.id}>{style.name}</option>)}
        </select>
      </label>
      <button type="button" title="Experimental music-style training" className="rounded-lg border border-border px-3 py-2 text-xs" onClick={() => setMyMusic(true)}>My music</button>
    </div>
    {!!params.custom_settings?.artist_id && <label className="block text-[11px] text-text-muted">Style strength · {Number(params.custom_settings?.artist_strength ?? 1).toFixed(2)}
      <input aria-label="Music style strength" type="range" min={0} max={1.5} step={0.05} value={Number(params.custom_settings?.artist_strength ?? 1)} className="mt-1 w-full"
        onChange={event => setParam('custom_settings', {...params.custom_settings, artist_strength: Number(event.target.value)})}/>
    </label>}
    {myMusic && <MyMusicDialog onClose={() => setMyMusic(false)} onSelect={selectStyle}/>}
    <label className="block text-[11px] text-text-muted">Composition planning
      <select aria-label="Composition planning" value={mode} disabled={!!params.custom_settings?.artist_id}
        onChange={event => {
          const next = Number(event.target.value); setParam('model_mode', next)
          if (next === 2) {
            setParam('audio_prompt_type', '')
            setParam('audio_guide', undefined)
            setParam('custom_settings', {...params.custom_settings, abc: ''})
          }
        }} className="mt-1 w-full rounded-lg border border-border bg-bg-tertiary p-2 text-xs text-text-primary">
        <option value={2}>Direct generation</option><option value={0}>Melody and chords</option><option value={1}>Melody only</option>
      </select>
    </label>
    <details className="text-xs text-text-secondary">
      <summary className="cursor-pointer py-1">Score and source song</summary>
      <div className="space-y-2 pt-2">
        <p className="text-[10px] text-text-muted">Extract notes from a recording or enter an ABC score. Supply the lyrics separately. This creates a new recording.</p>
        <input ref={input} hidden type="file" accept="audio/*" onChange={async event => {
          const file = event.target.files?.[0]; if (!file) return
          setUploading(true); setError('')
          try {const result = await uploadAudio(file); setParam('audio_guide', result.path); setParam('audio_prompt_type', 'A')}
          catch (reason) {setError(reason instanceof Error ? reason.message : 'Upload failed')}
          finally {setUploading(false); event.target.value = ''}
        }} />
        <button type="button" disabled={mode === 2 || uploading} onClick={() => input.current?.click()}
          className="rounded border border-border px-2 py-1 disabled:opacity-40">{uploading ? 'Uploading…' : source ? 'Replace source song' : 'Choose source song'}</button>
        {source && <div className="flex items-center gap-2"><span className="min-w-0 truncate">{source.split(/[\\/]/).pop()}</span>
          <button type="button" onClick={() => {setParam('audio_guide', undefined); setParam('audio_prompt_type', '')}}>Remove</button></div>}
        {!source && mode !== 2 && <AutoGrowTextarea value={String(params.custom_settings?.abc || '')}
          onChange={abc => setParam('custom_settings', {...params.custom_settings, abc})} placeholder="Optional ABC score with Vocal and Ins voices" />}
        {mode === 2 && <p className="text-[10px] text-text-muted">Choose a planning mode to use a score or source song.</p>}
        {error && <p role="alert" className="text-red-400">{error}</p>}
      </div>
    </details>
    <p className="text-[10px] text-text-muted">48 kHz stereo · CC BY-NC 4.0 model weights</p>
  </div>
}
