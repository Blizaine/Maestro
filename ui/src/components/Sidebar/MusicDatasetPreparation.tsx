import { useRef, useState } from 'react'
import { cancelJob } from '../../api/client'
import { analyzeMusicSongs, buildMusicDataset, musicPreparationAudioUrl, saveMusicPreparation, selectMusicVoices, rescanMusicSong } from '../../api/musicTraining'
import type { MusicPreparedClip, MusicPreparedSong, MusicProject } from '../../api/musicTraining'

const field = 'w-full rounded-lg border border-border bg-bg-tertiary p-2 text-sm text-text-primary'
const button = 'rounded-lg border border-border px-3 py-2 text-xs hover:bg-bg-tertiary disabled:opacity-40'
const time = (seconds: number) => `${Math.floor(seconds / 60)}:${String(Math.floor(seconds % 60)).padStart(2, '0')}`
type Run = (action: () => Promise<void>) => Promise<void>

function RangePlayer({url, start, end, label}: {url: string; start: number; end: number; label: string}) {
  const ref = useRef<HTMLAudioElement>(null)
  const [error, setError] = useState('')
  return <div className="space-y-1">
    <button type="button" className={button} onClick={() => {
      const player = ref.current
      if (!player) return
      setError(''); player.currentTime = start
      void player.play().catch(() => setError('Could not play this preview. Try again when analysis finishes.'))
    }}>{label} · {time(start)}–{time(end)}</button>
    <audio ref={ref} controls preload="metadata" src={url} className="h-9 w-full min-w-0" aria-label={label}
      onPlay={() => {const p = ref.current; if (p && (p.currentTime < start || p.currentTime >= end)) p.currentTime = start}}
      onTimeUpdate={() => {const p = ref.current; if (p && p.currentTime >= end) p.pause()}}
      onError={() => setError('Audio preview could not be loaded.')}/>
    {error && <p role="alert" className="text-xs text-amber-400">{error}</p>}
  </div>
}

function SongReview({project, trackId, song, busy, run, onDirty}: {project: MusicProject; trackId: string; song: MusicPreparedSong; busy: boolean; run: Run; onDirty: (dirty: boolean) => void}) {
  const [clips, setClips] = useState(song.clips)
  const [voices, setVoices] = useState(song.selected_speakers)
  const [holdout, setHoldout] = useState(song.holdout)
  const [dirty, setDirty] = useState(false)
  const [base, setBase] = useState({revision: project.preparation!.revision, song: JSON.stringify(song)})
  const [description, setDescription] = useState('')
  const [language, setLanguage] = useState(song.language_override || '')
  const serialized = JSON.stringify(song)
  const revision = project.preparation!.revision
  // A changed server song replaces local state only after Save/Discard. Polls
  // with the same content and changes to another song never erase local edits.
  if (!dirty && serialized !== base.song) {
    setClips(song.clips); setVoices(song.selected_speakers); setHoldout(song.holdout)
    setBase({revision, song: serialized})
  }
  const expectedRevision = serialized === base.song ? revision : base.revision
  const mark = (value: boolean) => {setDirty(value); onDirty(value)}
  const edit = (id: string, change: Partial<MusicPreparedClip>) => {
    setClips(rows => rows.map(clip => clip.id === id ? {...clip, reviewed: false, ...change} : clip)); mark(true)
  }
  const reset = () => {setClips(song.clips); setVoices(song.selected_speakers); setHoldout(song.holdout); mark(false)}
  const url = musicPreparationAudioUrl(project.id, trackId)
  const selectedChanged = JSON.stringify([...voices].sort()) !== JSON.stringify([...song.selected_speakers].sort())
  return <div className="space-y-3">
    {song.warnings.map((warning, i) => <p key={i} className="text-xs leading-relaxed text-text-muted">{warning}</p>)}
    <details className="space-y-2 rounded-lg border border-border p-3">
      <summary className="cursor-pointer text-xs">Full song &amp; transcription</summary>
      <audio controls preload="metadata" src={url} className="w-full min-w-0" aria-label="Full source song"/>
      <p className="text-xs text-text-muted">Detected language: {song.language || 'unknown'}. {song.recognized_words ?? song.speakers.reduce((sum, voice) => sum + voice.word_count, 0)} words recognized. A long intro or backing vocals can confuse recognition.</p>
      <label className="block text-xs">Lyrics language<input aria-label="Lyrics language" className={`${field} mt-1`} value={language} maxLength={3} onChange={event => setLanguage(event.target.value.toLowerCase())} placeholder="Automatic, or en for English"/></label>
      <p className="text-xs text-text-muted">Rescan keeps reviewed and manually added excerpts. New suggestions need review. Save current edits first.</p>
      <button type="button" className={button} disabled={busy || dirty || !!language && !/^[a-z]{2,3}$/.test(language)} onClick={() => void run(async () => {await rescanMusicSong(project.id, trackId, revision, language)})}>Rescan lyrics · keep reviewed clips</button>
    </details>
    <p className="text-sm font-medium">Voices</p>
    <p className="text-xs text-text-muted">The voice with the most confidently recognized words is selected. Listen to each voice; you can select several if the same performer was split into different groups.</p>
    <div className="space-y-2">{song.speakers.map(voice => <div key={voice.id} className="space-y-2 rounded-lg border border-border p-3">
      <label className="flex items-start gap-2 text-xs"><input type="checkbox" checked={voices.includes(voice.id)} disabled={busy} onChange={e => {setVoices(v => e.target.checked ? [...v, voice.id] : v.filter(id => id !== voice.id)); mark(true)}}/>
        <span>{voice.label} · {voice.word_count} words ({voice.confident_words} confident)</span></label>
      <RangePlayer url={url} start={voice.start} end={voice.end} label={`Listen to ${voice.label.toLowerCase()}`}/>
    </div>)}</div>
    {selectedChanged && <div className="space-y-2 rounded-lg border border-border p-3">
      <p className="text-xs text-amber-400">Suggesting again replaces this song’s clip edits. Other songs stay as they are.</p>
      <button type="button" className={button} disabled={busy || !voices.length} onClick={() => void run(async () => {
        await selectMusicVoices(project.id, trackId, expectedRevision, voices); mark(false)
      })}>Suggest clips for selected voices</button>
    </div>}
    <label className="flex items-center gap-2 text-xs"><input type="checkbox" checked={holdout} disabled={busy} onChange={e => {setHoldout(e.target.checked); mark(true)}}/>Use this song for checking only (held out)</label>
    <p className="text-xs text-text-muted">The model never learns from check-only clips. They help compare checkpoints on a song it has not practiced. Choose a few complete phrases of the same target performer; several short fragments give a weaker check.</p>
    <p className="text-sm font-medium">Suggested clips · {clips.filter(c => c.included).length} included</p>
    <p className="text-xs text-text-muted">Listen, correct the lyrics, and check the start and end. Suggestions aim for complete 20–30 second phrases. Exclude other singers, uncertain passages, and clips you don’t want to learn.</p>
    <details className="rounded-lg border border-border p-3">
      <summary className="cursor-pointer text-xs">Describe this song for all clips</summary>
      <div className="mt-2 space-y-2"><input aria-label="Description for all clips" className={field} value={description} maxLength={1800} onChange={e => setDescription(e.target.value)} placeholder="Genre, instruments, mood, vocal delivery…"/>
        <button className={button} type="button" disabled={busy || !description.trim()} onClick={() => {setClips(rows => rows.map(c => ({...c, style: description.trim(), reviewed: false}))); mark(true)}}>Apply description</button></div>
    </details>
    {clips.map((clip, index) => <details key={clip.id} className={`rounded-lg border border-border p-3 ${!clip.included ? 'opacity-60' : ''}`}>
      <summary className="cursor-pointer text-xs">Clip {index + 1} · {time(clip.start)}–{time(clip.end)} · {clip.included ? clip.reviewed ? 'Reviewed' : 'Needs review' : 'Excluded'}</summary>
      <div className="mt-3 space-y-3">
        <RangePlayer url={url} start={clip.start} end={clip.end} label={`Play clip ${index + 1}`}/>
        <label className="flex items-center gap-2 text-xs"><input type="checkbox" checked={clip.included} disabled={busy} onChange={e => edit(clip.id, {included: e.target.checked})}/>Include this clip</label>
        <div className="grid grid-cols-2 gap-2">
          <label className="text-xs">Start (seconds)<input type="number" min={0} max={song.duration} step="0.1" className={`${field} mt-1`} value={clip.start} disabled={busy} onChange={e => edit(clip.id, {start: Number(e.target.value)})}/></label>
          <label className="text-xs">End (seconds)<input type="number" min={0} max={song.duration} step="0.1" className={`${field} mt-1`} value={clip.end} disabled={busy} onChange={e => edit(clip.id, {end: Number(e.target.value)})}/></label>
        </div>
        <label className="block text-xs">Lyrics<textarea aria-label="Lyrics" className={`${field} mt-1 min-h-28`} value={clip.lyrics} maxLength={40000} disabled={busy} onChange={e => edit(clip.id, {lyrics: e.target.value})}/></label>
        <label className="block text-xs">Music description<textarea aria-label="Music description" className={`${field} mt-1`} value={clip.style} maxLength={1800} disabled={busy} onChange={e => edit(clip.id, {style: e.target.value})}/></label>
        <label className="block text-xs">Vocal delivery<select aria-label="Vocal delivery" className={`${field} mt-1`} value={clip.delivery} disabled={busy} onChange={e => edit(clip.id, {delivery: e.target.value as MusicPreparedClip['delivery']})}>
          <option value="unspecified">Unspecified</option><option value="rap">Rap</option><option value="sung">Sung</option><option value="mixed">Rap and singing</option></select></label>
        {clip.warnings.map((warning, i) => <p key={i} className="text-xs text-amber-400">{warning}</p>)}
        <label className="flex items-start gap-2 text-xs"><input type="checkbox" checked={clip.reviewed} disabled={busy || !clip.included} onChange={e => edit(clip.id, {reviewed: e.target.checked})}/>I checked this clip’s voice, lyrics, description, and boundaries</label>
      </div>
    </details>)}
    <button type="button" className={button} disabled={busy} onClick={() => {
      setClips(rows => [...rows, {id: `manual-${Date.now()}-${Math.random().toString(36).slice(2)}`, start: 0, end: Math.min(25, song.duration), lyrics: '',
        style: description || 'Music with lead vocals and instrumental accompaniment', delivery: 'unspecified', included: true, reviewed: false, warnings: []}]); mark(true)
    }}>Add excerpt manually</button>
    <label className="flex items-start gap-2 text-xs"><input type="checkbox" checked={clips.some(c => c.included) && clips.filter(c => c.included).every(c => c.reviewed)} disabled={busy || !clips.some(c => c.included)} onChange={e => {setClips(rows => rows.map(c => c.included ? {...c, reviewed: e.target.checked} : c)); mark(true)}}/>I have listened to and checked all included clips in this song</label>
    {serialized !== base.song && dirty && <p role="alert" className="text-xs text-amber-400">This song changed elsewhere. Discard local edits to load its latest review.</p>}
    <div className="flex flex-wrap gap-2">
      <button type="button" className={button} disabled={busy || !dirty || selectedChanged} onClick={() => void run(async () => {
        await saveMusicPreparation(project.id, trackId, expectedRevision, clips, holdout); mark(false)
      })}>Save song review</button>
      {dirty && <button type="button" className={button} disabled={busy} onClick={reset}>Discard unsaved edits</button>}
    </div>
  </div>
}

export function MusicDatasetPreparation({project, busy, run, onOpenProject}: {project: MusicProject; busy: boolean; run: Run; onOpenProject: (id: string) => void}) {
  const [trackId, setTrackId] = useState(project.tracks[0]?.id || '')
  const [dirty, setDirty] = useState(false)
  const prep = project.preparation!
  const running = ['queued', 'preparing'].includes(project.status)
  const song = prep.songs[trackId]
  const songs = Object.values(prep.songs)
  const clips = songs.flatMap(s => s.clips.filter(c => c.included).map(c => ({...c, holdout: s.holdout})))
  const seconds = (held: boolean) => clips.filter(c => c.holdout === held).reduce((sum, c) => sum + c.end - c.start, 0)
  const ready = songs.length === project.tracks.length && clips.length >= 2 && clips.every(c => c.reviewed) && seconds(false) > 0 && seconds(true) > 0
  return <div className="space-y-4">
    <div className="space-y-2 rounded-lg border border-border p-3">
      <p className="text-sm font-medium">{project.name} · Song preparation</p>
      <p role="status" className="break-words text-xs">{project.message}</p>
      <p className="text-xs text-text-muted">{songs.length}/{project.tracks.length} songs analyzed · {project.tokenizer_pair || 'v4'} tokenizer</p>
      {running && <progress aria-label="Song preparation progress" className="w-full accent-accent-orange" max={100} value={project.progress}/>}
      {running ? <button type="button" className={button} disabled={busy} onClick={() => void run(async () => {if (project.job_id) await cancelJob(project.job_id)})}>Stop preparation</button>
        : songs.length < project.tracks.length && <button type="button" className={button} disabled={busy || dirty} onClick={() => void run(async () => {await analyzeMusicSongs(project.id)})}>{songs.length ? 'Resume song analysis' : 'Analyze songs'}</button>}
      {!running && songs.some(s => s.diarization_failed || s.warnings.some(w => w.startsWith('Voices could not be separated automatically.'))) && <div className="space-y-2">
        <p className="text-xs text-amber-400">Retrying voice analysis replaces clip edits only for songs whose voice analysis failed. Cached lyrics and vocal separation are reused.</p>
        <button type="button" className={button} disabled={busy || dirty} onClick={() => void run(async () => {await analyzeMusicSongs(project.id, true)})}>Retry voice analysis</button>
      </div>}
    </div>
    <label className="block text-xs">Song to review<select aria-label="Song to review" className={`${field} mt-1`} value={trackId} disabled={dirty} onChange={e => setTrackId(e.target.value)}>{project.tracks.map(track => <option key={track.id} value={track.id}>{track.name}{prep.songs[track.id || ''] ? '' : ' · awaiting analysis'}</option>)}</select></label>
    {dirty && <p className="text-xs text-text-muted">Save or discard this song’s edits before switching songs or creating the dataset.</p>}
    {song && <SongReview key={trackId} project={project} trackId={trackId} song={song} busy={busy || running} run={run} onDirty={setDirty}/>}
    <div className="space-y-2 rounded-lg border border-border p-3">
      <p className="text-sm">Dataset · {clips.length} included clips</p>
      <p className="text-xs text-text-muted">{time(seconds(false))} for learning · {time(seconds(true))} for checking only · {clips.filter(c => c.reviewed).length}/{clips.length} reviewed</p>
      {seconds(true) > 0 && seconds(true) < 20.5 && <p className="text-xs text-amber-400">The check-only audio is under 20.5 seconds. It can provide a limited check; a few longer, complete phrases are more informative.</p>}
      <p className="text-xs text-text-muted">Creates a new training project from the original full mixes. Every excerpt of a song stays on the same side of the split. The originals and existing checkpoints are preserved.</p>
      {prep.dataset_project_id ? <button type="button" className={button} disabled={busy || dirty} onClick={() => onOpenProject(prep.dataset_project_id!)}>Open training dataset</button>
        : <button type="button" className={button} disabled={busy || running || dirty || !ready} onClick={() => void run(async () => {await buildMusicDataset(project.id)})}>Create training dataset</button>}
      {!ready && <p className="text-xs text-text-muted">Review included clips and keep at least one training song and a separate held-out song.</p>}
    </div>
  </div>
}
