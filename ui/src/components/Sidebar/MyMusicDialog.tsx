import { useCallback, useEffect, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import { Check, Loader2, X } from 'lucide-react'
import { uploadAudio } from '../../api/client'
import { createMusicProject, fetchMusicProjects, fetchMusicStyles, importMusicStyle,
  musicStyleExportUrl, musicRecordingUrl, createMusicDraft, analyzeMusicSongs, setMusicStyleArchived, setMusicStyleListed, startAutoMusicTraining } from '../../api/musicTraining'
import type { MusicProject, MusicStyle, MusicTrack } from '../../api/musicTraining'
import { MusicDataReview, MusicTrackFields } from './MusicDataReview'
import { MusicDatasetPreparation } from './MusicDatasetPreparation'
import { MusicTrainingWorkflow } from './MusicTrainingWorkflow'
import { AutoMusicTraining, AutoTrainingTargets } from './AutoMusicTraining'
import { validAutoSteps } from '../../lib/musicTraining'
import { deselectMusicStyle } from '../../lib/musicStyles'

const field = 'w-full rounded-lg border border-border bg-bg-tertiary p-2 text-sm text-text-primary'
const button = 'rounded-lg border border-border px-3 py-2 text-xs hover:bg-bg-tertiary disabled:opacity-40'
const active = (project: MusicProject) => ['queued', 'preparing', 'training', 'auditioning'].includes(project.status)

export function MyMusicDialog({onClose, onSelect}: {onClose: () => void; onSelect: (style: MusicStyle) => void}) {
  const dialog = useRef<HTMLDialogElement>(null)
  const [tab, setTab] = useState<'library' | 'train' | 'import'>('library')
  const [styles, setStyles] = useState<MusicStyle[]>([])
  const [search, setSearch] = useState('')
  const [showRemoved, setShowRemoved] = useState(false)
  const [removed, setRemoved] = useState<MusicStyle | null>(null)
  const [projects, setProjects] = useState<MusicProject[]>([])
  const [selectedId, setSelectedId] = useState('')
  const [name, setName] = useState('')
  const [trigger, setTrigger] = useState('')
  const [tracks, setTracks] = useState<MusicTrack[]>([])
  const [pair, setPair] = useState<'v4' | 'v9'>('v9')
  const [automatic, setAutomatic] = useState(true)
  const [autoMode, setAutoMode] = useState(false)
  const [voiceSteps, setVoiceSteps] = useState(100)
  const [styleSteps, setStyleSteps] = useState(200)
  const [autoSetupId, setAutoSetupId] = useState('')
  const [manualProjectId, setManualProjectId] = useState('')
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const [ar, setAr] = useState<File | null>(null)
  const [nar, setNar] = useState<File | null>(null)
  const project = projects.find(item => item.id === selectedId)
  const refresh = useCallback(async () => {
    const [library, training] = await Promise.all([fetchMusicStyles(true), fetchMusicProjects()])
    setStyles(library.styles); setProjects(training.projects)
  }, [])
  useEffect(() => {
    const element = dialog.current
    element?.showModal()
    void refresh().catch(reason => setError(String(reason.message || reason)))
    let pending = false
    const interval = window.setInterval(() => {
      if (pending) return
      pending = true
      void refresh().catch(() => {}).finally(() => {pending = false})
    }, 2500)
    return () => {window.clearInterval(interval); element?.close()}
  }, [refresh])
  const run = async (action: () => Promise<void>) => {
    setBusy(true); setError('')
    try {await action(); await refresh()}
    catch (reason) {setError(reason instanceof Error ? reason.message : 'Music request failed')}
    finally {setBusy(false)}
  }
  const editTrack = (index: number, changes: Partial<MusicTrack>) => setTracks(previous => previous.map((track, at) => at === index ? {...track, ...changes} : track))
  const editDataset = (values: MusicTrack[]) => {
    if (!project) return
    setName(project.name.slice(0, 75) + ' · reviewed'); setTrigger(project.trigger)
    setTracks(values.map(track => ({...track, preview_url: track.id ? musicRecordingUrl(project.id, track.id) : undefined})))
    setPair(project.tokenizer_pair || 'v4')
    setAutomatic(false)
    setAutoMode(false)
    setSelectedId('')
  }

  return createPortal(<dialog ref={dialog} onCancel={onClose} aria-labelledby="my-music-title" aria-describedby="my-music-description"
    className="m-auto max-h-[90dvh] w-[680px] max-w-[calc(100vw-24px)] overflow-y-auto rounded-xl border border-border bg-bg-secondary p-0 text-text-primary shadow-2xl backdrop:bg-black/70">
    <div className="sticky top-0 z-10 flex items-center justify-between border-b border-border bg-bg-secondary p-4">
      <div className="flex flex-wrap items-center gap-2">
        <h2 id="my-music-title" className="text-base font-semibold">My music</h2>
        <span className="rounded-full border border-accent-orange/30 bg-accent-orange/10 px-2 py-0.5 text-[10px] font-medium text-accent-orange">Experimental</span>
      </div>
      <button type="button" onClick={onClose} aria-label="Close My music" className="p-1"><X size={18}/></button>
    </div>
    <div className="space-y-4 p-4">
      <p id="my-music-description" className="text-xs leading-relaxed text-text-muted">Train and manage YuE2 music LoRAs from your recordings. Each saved LoRA keeps its matching sound decoder. Voice resemblance remains experimental.</p>
      <nav className="flex gap-2" aria-label="My music sections">
        {([['library', 'Saved LoRAs'], ['train', 'Train a style'], ['import', 'Import']] as const).map(([value, label]) =>
          <button key={value} type="button" aria-pressed={tab === value} onClick={() => {setTab(value); setError('')}}
            className={`${button} ${tab === value ? 'bg-bg-tertiary text-accent-orange' : ''}`}>{label}</button>)}
      </nav>
      {error && <p role="alert" className="text-sm text-red-400">{error}</p>}
      {busy && <p role="status" className="flex items-center gap-2 text-xs text-text-muted"><Loader2 size={14} className="animate-spin"/> Working…</p>}
      {tab === 'library' && <div className="space-y-3">
        <p className="text-xs text-text-muted">Choose which LoRAs appear in Advanced → LoRAs &amp; presets. Turn them on there with a checkbox and adjust strength. Adding one to the selector does not activate it.</p>
        <label className="block text-xs">Search music LoRAs<input type="search" className={`${field} mt-1`} value={search} onChange={event => setSearch(event.target.value)} placeholder="Name, trigger or checkpoint"/></label>
        <label className="flex items-center gap-2 text-xs text-text-muted"><input type="checkbox" checked={showRemoved} onChange={event => setShowRemoved(event.target.checked)}/>Show removed LoRAs</label>
        {removed && <div role="status" className="rounded-lg border border-border p-3 text-xs text-text-muted">
          <p className="break-words">Removed {removed.name} from the library. Songs and training checkpoints are kept. You can restore it any time.</p>
          <button type="button" disabled={busy} className={`${button} mt-2`} onClick={() => void run(async () => {
            await setMusicStyleArchived(removed.id, false); setRemoved(null)
          })}>Undo removal</button>
        </div>}
        {styles.length === 0 && <p className="py-6 text-center text-sm text-text-muted">No saved music LoRAs yet. Train a style or import a compatible YuE2 adapter.</p>}
        {styles.length > 0 && !styles.some(style => (showRemoved || !style.archived) && `${style.name} ${style.trigger} ${style.training?.checkpoint || ''}`.toLowerCase().includes(search.trim().toLowerCase())) &&
          <p className="py-4 text-center text-xs text-text-muted">No matching LoRAs. Try another search or show removed entries.</p>}
        {styles.filter(style => (showRemoved || !style.archived) && `${style.name} ${style.trigger} ${style.training?.checkpoint || ''}`.toLowerCase().includes(search.trim().toLowerCase())).map(style =>
          <article key={style.id} aria-label={style.name} className="space-y-3 rounded-lg border border-border p-3">
            <div className="min-w-0"><p className="break-words text-sm">{style.name}{style.archived && <span className="ml-2 text-xs text-text-muted">Removed</span>}</p>
              <p className="break-words text-xs text-text-muted">{style.trigger ? `Auto trigger: ${style.trigger}` : 'No trigger required'}</p>
              {style.adapted_pair && <p className="text-[11px] text-text-muted">Sound adaptation: {style.adapted_pair.step} steps</p>}
            </div>
            <div className="flex flex-wrap items-center gap-2">
              {!style.archived && <label className="flex cursor-pointer items-center gap-2 text-xs text-text-secondary">
                <span className="relative inline-flex shrink-0">
                  <input type="checkbox" role="switch" checked={!!style.in_selector} disabled={busy}
                    aria-label={`Show ${style.name} in LoRA selector`}
                    className="peer h-5 w-9 cursor-pointer appearance-none rounded-full border border-border bg-bg-tertiary checked:border-accent-blue checked:bg-accent-blue disabled:cursor-wait disabled:opacity-40 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent-blue"
                    onChange={event => {const listed = event.target.checked; void run(async () => {
                      await setMusicStyleListed(style.id, listed)
                      if (!listed) deselectMusicStyle(style.id)
                    })}}/>
                  <span aria-hidden="true" className="pointer-events-none absolute left-0.5 top-0.5 h-4 w-4 rounded-full bg-text-primary shadow-sm transition-transform peer-checked:translate-x-4 peer-checked:bg-white peer-disabled:opacity-40"/>
                </span>
                Show in LoRA selector
              </label>}
              <a className={button} href={musicStyleExportUrl(style.id)} download>Export</a>
              <button type="button" className={`${button} ml-auto`} disabled={busy} onClick={() => void run(async () => {
                await setMusicStyleArchived(style.id, !style.archived)
                if (!style.archived) {deselectMusicStyle(style.id); setRemoved(style)} else setRemoved(null)
              })}>{style.archived ? 'Restore' : 'Remove'}</button>
            </div>
          </article>)}
        <p className="text-[10px] text-text-muted">Remove hides a LoRA from the library and keeps its files for queued jobs and restoration.</p>
      </div>}
      {tab === 'import' && <div className="space-y-3 text-sm">
        <p className="text-xs text-text-muted">Import a Maestro style ZIP, a combined AI-Toolkit YuE2 LoRA, or a Mothersuperior v4 AR adapter. Combined LoRAs include both parts. Separate AR files use the v4 audio adapter unless you supply one.</p>
        <label className="block">Style name<input className={field} value={name} maxLength={100} onChange={event => setName(event.target.value)}/></label>
        <label className="block">Style trigger<input className={field} value={trigger} maxLength={200} onChange={event => setTrigger(event.target.value)} placeholder="A distinctive name used when training"/></label>
        <label className="block">Style ZIP or YuE2 adapter<input className={`${field} mt-1`} type="file" accept=".zip,.pt,.safetensors" onChange={event => {setAr(event.target.files?.[0] || null); setNar(null)}}/></label>
        {!ar?.name.toLowerCase().endsWith('.zip') && <label className="block">Separate audio adapter (optional, AR-only files)<input className={`${field} mt-1`} type="file" accept=".pt,.safetensors" onChange={event => setNar(event.target.files?.[0] || null)}/></label>}
        <button type="button" disabled={busy || !ar || (!name.trim() && !ar.name.toLowerCase().endsWith('.zip'))} className={button} onClick={() => void run(async () => {
          if (!ar) return
          const form = new FormData(); form.set('name', name); form.set('trigger', trigger); form.set('ar', ar)
          if (nar) form.set('nar', nar)
          await importMusicStyle(form); setTab('library')
        })}>Import style</button>
      </div>}
      {tab === 'train' && <div className="space-y-4">
        <p className="text-xs text-text-muted">Learn voice &amp; sound, then song style. Use Auto to prepare and train in one run, or Guided to review each stage. Training needs at least 20 GB VRAM and waits for other GPU jobs.</p>
        <label className="block text-xs">Training project<select aria-label="Training project" className={`${field} mt-1`} value={selectedId} onChange={event => {
          setSelectedId(event.target.value)
          setManualProjectId(''); setAutoSetupId('')
          if (!event.target.value) {setPair('v9'); setAutomatic(true)}
        }}><option value="">New project</option>{projects.map(item => <option key={item.id} value={item.id}>{item.name} · {item.status}</option>)}</select></label>
        {!project ? <div className="space-y-3">
          <div role="group" aria-label="Training mode" className="flex gap-2">
            {(['Auto', 'Guided'] as const).map(mode => {
              const selected = autoMode === (mode === 'Auto')
              return <button key={mode} type="button" aria-pressed={selected}
                className={`flex min-w-24 items-center justify-center gap-2 rounded-lg border px-3 py-2 text-xs font-semibold transition-colors focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent-blue ${selected ? 'border-accent-blue bg-accent-blue/20 text-text-primary' : 'border-border bg-bg-primary text-text-secondary hover:bg-bg-tertiary'}`}
                onClick={() => {setAutoMode(mode === 'Auto'); if (mode === 'Auto') setAutomatic(true)}}>
                <Check size={14} aria-hidden="true" className={selected ? 'text-accent-blue' : 'invisible'}/>{mode}
              </button>
            })}
          </div>
          <p className="text-xs text-text-muted">{autoMode ? 'No lyrics or individual clip reviews required. Maestro uses the main detected voice in each song, prepares clips, trains both stages and saves your LoRA. You can review the choices later.' : automatic ? 'No lyrics needed to start. Review the suggested voices, lyrics and excerpts before training.' : 'Use prepared recordings with lyrics and descriptions you supply.'}</p>
          {autoMode && <AutoTrainingTargets voice={voiceSteps} style={styleSteps} onVoice={setVoiceSteps} onStyle={setStyleSteps} disabled={busy}/>}
          <details className="space-y-3 rounded-lg border border-border p-3">
            <summary className="cursor-pointer text-xs">Expert setup</summary>
            <label className="flex items-start gap-2 text-xs"><input type="checkbox" checked={automatic} disabled={autoMode} onChange={e => setAutomatic(e.target.checked)}/>Prepare full songs automatically: transcribe, find voices, and suggest clips</label>
            <label className="block text-xs">Audio tokenizer and decoder<select aria-label="Audio tokenizer and decoder" className={`${field} mt-1`} value={pair} onChange={event => setPair(event.target.value as 'v4' | 'v9')}><option value="v9">v9 · recommended for new projects</option><option value="v4">v4 · legacy comparison</option></select></label>
          </details>
          <label className="block text-xs">Project name<input className={`${field} mt-1`} value={name} maxLength={100} onChange={event => setName(event.target.value)}/></label>
          <label className="block text-xs">Style trigger<input className={`${field} mt-1`} value={trigger} maxLength={200} placeholder="e.g. My midnight acoustic sessions" onChange={event => setTrigger(event.target.value)}/></label>
          <label className="block text-xs">Recordings<input className={`${field} mt-1`} type="file" accept="audio/*" multiple disabled={busy} onChange={event => {
            const files = Array.from(event.target.files || []); event.target.value = ''
            void run(async () => {
              if (tracks.length + files.length > 50) throw new Error('Choose up to 50 recordings')
              for (const file of files) {
                const uploaded = await uploadAudio(file)
                setTracks(previous => [...previous, {audio_path: uploaded.path, preview_url: uploaded.url, name: file.name, lyrics: '', style: '', holdout: previous.length === 1}])
              }
            })
          }}/></label>
          <p className="text-xs text-text-muted">Use 2–50 distinct songs{automatic ? ', up to 20 minutes each' : ''}. Choose at least one for checking only (held out). The model learns from the others; this separate song helps check whether the learning works beyond its practice material. It is not a voice reference fed into new songs.</p>
          {tracks.map((track, index) => <div key={`${track.audio_path}-${index}`} className="space-y-2 rounded-lg border border-border p-3">
            <div className="flex items-center justify-between gap-2"><span className="min-w-0 break-words text-xs">{track.name}</span><button type="button" className="text-xs text-text-muted" onClick={() => setTracks(previous => previous.filter((_, at) => at !== index))}>Remove</button></div>
            {automatic ? <div className="space-y-2">
              {track.preview_url && <audio controls preload="metadata" src={track.preview_url} className="h-10 w-full min-w-0" aria-label={`Preview ${track.name}`}/>}
              <label className="block text-xs">Music description (optional)<input className={`${field} mt-1`} value={track.style} maxLength={1800} placeholder="Genre, instruments, mood, rap or singing…" onChange={e => editTrack(index, {style: e.target.value})}/></label>
              <label className="flex items-center gap-2 text-xs"><input type="checkbox" checked={track.holdout} onChange={e => editTrack(index, {holdout: e.target.checked})}/>Check only — do not train on this song (held out)</label>
            </div> : <MusicTrackFields track={track} onChange={changes => editTrack(index, changes)}/>}
          </div>)}
          <button type="button" className={button} disabled={busy || !name.trim() || !trigger.trim() || tracks.length < 2 || (autoMode && !validAutoSteps(voiceSteps, styleSteps))} onClick={() => void run(async () => {
            const created = await (automatic ? createMusicDraft : createMusicProject)(name, trigger, tracks, pair)
            setProjects(previous => [created, ...previous]); setSelectedId(created.id); setTracks([])
            if (autoMode) {setAutoSetupId(created.id); await startAutoMusicTraining(created.id, voiceSteps, styleSteps)}
            else if (automatic) await analyzeMusicSongs(created.id)
          })}>{autoMode ? 'Start Auto training' : automatic ? 'Analyze songs' : 'Create project'}</button>
        </div> : <>
        {(project.auto_training || project.auto_training_root || autoSetupId === project.id) && manualProjectId === project.id && <button type="button" className={button} onClick={() => {setSelectedId(project.auto_training_root || project.id); setManualProjectId('')}}>Back to Auto overview</button>}
        {!project.auto_training && project.auto_training_root && manualProjectId !== project.id && <button type="button" className={button} onClick={() => {setSelectedId(project.auto_training_root!); setManualProjectId('')}}>Open Auto overview</button>}
        {(project.auto_training || autoSetupId === project.id) && manualProjectId !== project.id ? <AutoMusicTraining key={project.id} project={project} styles={styles} busy={busy} run={run} initialVoice={voiceSteps} initialStyle={styleSteps}
          onManual={id => {setSelectedId(id); setManualProjectId(id)}} onSelect={style => {onSelect(style); onClose()}}/>
        : project.preparation_draft ? <>
          {!project.auto_training && <button type="button" className={button} disabled={busy || active(project)} onClick={() => {setAutoSetupId(project.id); setManualProjectId('')}}>Continue automatically</button>}
          <MusicDatasetPreparation key={project.id} project={project} busy={busy} run={run} onOpenProject={setSelectedId}/>
        </> : <div className="space-y-3">
          {!project.auto_training && !project.auto_training_root && !project.adapted_pair && <button type="button" className={button} disabled={busy || active(project)} onClick={() => {setAutoSetupId(project.id); setManualProjectId('')}}>Continue automatically</button>}
          <div className="rounded-lg border border-border p-3">
            <p className="text-sm">{project.name}</p><p className="mt-1 text-xs text-text-muted">{project.tracks.length} recordings · {project.tracks.filter(track => track.holdout).length} held out</p>
            <p role="status" className="mt-2 break-words text-xs">{active(project) || ['failed', 'cancelled', 'interrupted'].includes(project.status) ? project.message : project.checkpoints.length ? 'Song style is ready to try.' : project.pair_completed_steps ? 'Voice & sound saved. Continue to song style below.' : 'Ready for the next training step.'}</p>
            <p className="mt-1 text-xs text-text-muted">Check-only recordings are used to measure progress, never to teach the model.</p>
            {active(project) && <progress aria-label="Music job progress" max={100} value={project.progress} className="mt-2 w-full accent-accent-orange"/>}
          </div>
          <MusicDataReview project={project} busy={busy} run={run} onEdit={editDataset}/>
          <MusicTrainingWorkflow key={project.id} project={project} projects={projects} styles={styles} busy={busy} run={run}
            onOpenProject={setSelectedId} onSelect={style => {onSelect(style); onClose()}}/>
        </div>}</>}
      </div>}
      <p className="border-t border-border pt-3 text-[10px] text-text-muted">YuE2 and the real-audio tokenizer use CC BY-NC 4.0 weights. Your recordings, prepared data and checkpoints stay local.</p>
    </div>
  </dialog>, document.body)
}
