import { useCallback, useEffect, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import { Loader2, X } from 'lucide-react'
import { cancelJob, uploadAudio } from '../../api/client'
import { auditionMusicCheckpoint, createMusicProject, fetchMusicProjects, fetchMusicStyles, importMusicStyle,
  musicStyleExportUrl, startMusicPreparation, startMusicTraining, adaptMusicSound, alignMusicLyrics,
  auditionAudioCheckpoint, forkMusicProject, prepareMusicSound, musicRecordingUrl } from '../../api/musicTraining'
import type { MusicAuditionSettings, MusicProject, MusicStyle, MusicTrack } from '../../api/musicTraining'
import { MusicDataReview, MusicTrackFields } from './MusicDataReview'
import { MusicAuditions } from './MusicAuditions'

const field = 'w-full rounded-lg border border-border bg-bg-tertiary p-2 text-sm text-text-primary'
const button = 'rounded-lg border border-border px-3 py-2 text-xs hover:bg-bg-tertiary disabled:opacity-40'
const active = (project: MusicProject) => ['queued', 'preparing', 'training', 'auditioning'].includes(project.status)

export function MyMusicDialog({onClose, onSelect}: {onClose: () => void; onSelect: (style: MusicStyle) => void}) {
  const dialog = useRef<HTMLDialogElement>(null)
  const [tab, setTab] = useState<'library' | 'train' | 'import'>('library')
  const [styles, setStyles] = useState<MusicStyle[]>([])
  const [projects, setProjects] = useState<MusicProject[]>([])
  const [selectedId, setSelectedId] = useState('')
  const [name, setName] = useState('')
  const [trigger, setTrigger] = useState('')
  const [tracks, setTracks] = useState<MusicTrack[]>([])
  const [steps, setSteps] = useState(800)
  const [rank, setRank] = useState(64)
  const [aligned, setAligned] = useState(false)
  const [audioSteps, setAudioSteps] = useState(200)
  const [conditioning, setConditioning] = useState('')
  const [auditionDraft, setAuditionDraft] = useState<{projectId: string; settings: MusicAuditionSettings} | null>(null)
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const [ar, setAr] = useState<File | null>(null)
  const [nar, setNar] = useState<File | null>(null)
  const project = projects.find(item => item.id === selectedId)
  const audition = auditionDraft && auditionDraft.projectId === project?.id ? auditionDraft.settings : project?.audition_settings || {enabled: false}
  const setAudition = (settings: MusicAuditionSettings) => setAuditionDraft({projectId: project?.id || '', settings})
  const refresh = useCallback(async () => {
    const [library, training] = await Promise.all([fetchMusicStyles(), fetchMusicProjects()])
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
    setSelectedId(''); setAligned(false); setConditioning(''); setSteps(800); setAudioSteps(200)
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
      <p id="my-music-description" className="text-xs leading-relaxed text-text-muted">Train reusable music styles from your recordings. Results vary; matching a specific singer’s voice is not guaranteed.</p>
      <nav className="flex gap-2" aria-label="My music sections">
        {([['library', 'Saved styles'], ['train', 'Train a style'], ['import', 'Import']] as const).map(([value, label]) =>
          <button key={value} type="button" aria-pressed={tab === value} onClick={() => {setTab(value); setError('')}}
            className={`${button} ${tab === value ? 'bg-bg-tertiary text-accent-orange' : ''}`}>{label}</button>)}
      </nav>
      {error && <p role="alert" className="text-sm text-red-400">{error}</p>}
      {busy && <p role="status" className="flex items-center gap-2 text-xs text-text-muted"><Loader2 size={14} className="animate-spin"/> Working…</p>}
      {tab === 'library' && <div className="space-y-3">
        <p className="text-xs text-text-muted">Apply a style, then generate a song with new lyrics. Use the same seed and lyrics to compare it with base YuE2.</p>
        {styles.length === 0 && <p className="py-6 text-center text-sm text-text-muted">No saved styles yet. Train a style or import a compatible YuE2 adapter.</p>}
        {styles.map(style => <div key={style.id} className="flex flex-wrap items-center gap-3 rounded-lg border border-border p-3">
          <div className="min-w-0 flex-1"><p className="text-sm">{style.name}</p><p className="break-words text-xs text-text-muted">{style.trigger}</p></div>
          <button type="button" className={button} onClick={() => {onSelect(style); onClose()}}>Use style</button>
          <a className={button} href={musicStyleExportUrl(style.id)} download>Export</a>
        </div>)}
      </div>}
      {tab === 'import' && <div className="space-y-3 text-sm">
        <p className="text-xs text-text-muted">Import a Maestro music-style ZIP or a compatible Mothersuperior v4 AR adapter. A raw AR adapter uses the paired v4 NAR adapter unless you supply one.</p>
        <label className="block">Style name<input className={field} value={name} maxLength={100} onChange={event => setName(event.target.value)}/></label>
        <label className="block">Style trigger<input className={field} value={trigger} maxLength={200} onChange={event => setTrigger(event.target.value)} placeholder="A distinctive name used when training"/></label>
        <label className="block">Style bundle or AR adapter<input className={`${field} mt-1`} type="file" accept=".zip,.pt,.safetensors" onChange={event => setAr(event.target.files?.[0] || null)}/></label>
        {!ar?.name.toLowerCase().endsWith('.zip') && <label className="block">NAR adapter (optional)<input className={`${field} mt-1`} type="file" accept=".pt,.safetensors" onChange={event => setNar(event.target.files?.[0] || null)}/></label>}
        <button type="button" disabled={busy || !ar || (!name.trim() && !ar.name.toLowerCase().endsWith('.zip'))} className={button} onClick={() => void run(async () => {
          if (!ar) return
          const form = new FormData(); form.set('name', name); form.set('trigger', trigger); form.set('ar', ar)
          if (nar) form.set('nar', nar)
          await importMusicStyle(form); setTab('library')
        })}>Import style</button>
      </div>}
      {tab === 'train' && <div className="space-y-4">
        <p className="text-xs text-text-muted">Prepare recordings with the real-audio tokenizer, then train a YuE2 style adapter. Optional assets download on first use. Training needs at least 20 GB VRAM and waits for other GPU jobs.</p>
        <label className="block text-xs">Training project<select className={`${field} mt-1`} value={selectedId} onChange={event => {
          setSelectedId(event.target.value)
          const item = projects.find(value => value.id === event.target.value)
          if (item?.training_options) setRank(item.training_options.rank)
          setAligned(!!item?.training_options?.lyric_alignment)
          setConditioning(item?.audio_training_options?.conditioning_checkpoint || '')
          setAudioSteps(Math.min(1600, Math.max(200, (item?.audio_completed_steps || 0) + 100)))
          setSteps(Math.min(1600, Math.max(800, (item?.completed_steps || 0) + 200)))
        }}><option value="">New project</option>{projects.map(item => <option key={item.id} value={item.id}>{item.name} · {item.status}</option>)}</select></label>
        {!project ? <div className="space-y-3">
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
          <p className="text-xs text-text-muted">Use 2–50 distinct songs, 1 second to 10 minutes each. Keep at least one song held out: it is evaluated but never used for training. Originals are preserved.</p>
          {tracks.map((track, index) => <div key={`${track.audio_path}-${index}`} className="space-y-2 rounded-lg border border-border p-3">
            <div className="flex items-center justify-between gap-2"><span className="min-w-0 break-words text-xs">{track.name}</span><button type="button" className="text-xs text-text-muted" onClick={() => setTracks(previous => previous.filter((_, at) => at !== index))}>Remove</button></div>
            <MusicTrackFields track={track} onChange={changes => editTrack(index, changes)}/>
          </div>)}
          <button type="button" className={button} disabled={busy || !name.trim() || !trigger.trim() || tracks.length < 2} onClick={() => void run(async () => {
            const created = await createMusicProject(name, trigger, tracks); setSelectedId(created.id); setTracks([])
          })}>Create project</button>
        </div> : <div className="space-y-3">
          <div className="rounded-lg border border-border p-3">
            <p className="text-sm">{project.name}</p><p className="mt-1 text-xs text-text-muted">{project.tracks.length} recordings · {project.tracks.filter(track => track.holdout).length} held out</p>
            <p role="status" className="mt-2 break-words text-xs">{project.message}</p>
            {!active(project) && <button type="button" className={`${button} mt-2`} disabled={busy} onClick={() => void run(async () => {
              const next = await forkMusicProject(project.id); setSelectedId(next.id); setAligned(false); setConditioning(''); setSteps(800); setAudioSteps(200)
            })}>New experiment with these recordings</button>}
            {active(project) && <progress aria-label="Music job progress" max={100} value={project.progress} className="mt-2 w-full accent-accent-orange"/>}
          </div>
          <MusicDataReview project={project} busy={busy} run={run} onEdit={editDataset}/>
          {(project.prepared || project.checkpoints.length > 0 || !!project.audio_checkpoints?.length) && <MusicAuditions project={project} settings={audition} onChange={setAudition} busy={busy} run={run}/>}
          {active(project) ? <button type="button" className={button} disabled={busy} onClick={() => void run(async () => {if (project.job_id) await cancelJob(project.job_id)})}>Stop after current step</button>
            : <div className="space-y-3">
              <button type="button" className={button} disabled={busy} onClick={() => void run(async () => {await startMusicPreparation(project.id)})}>{project.prepared ? 'Check prepared audio' : 'Prepare audio tokens'}</button>
              {project.prepared && <>
                <div className="space-y-2 rounded-lg border border-border p-3">
                  <p className="text-sm">Lyric timing</p>
                  <p className="text-xs text-text-muted">Align sung words with the supplied lyrics. Confident word timings can guide new music training; instrumentals and silent gaps are excluded.</p>
                  <button type="button" className={button} disabled={busy} onClick={() => void run(async () => {await alignMusicLyrics(project.id)})}>{project.alignment ? 'Check lyric alignment' : 'Align lyrics'}</button>
                  {project.alignment?.tracks.map(track => <p key={track.track_id} className="text-xs text-text-muted">{project.tracks.find(t => t.id === track.track_id)?.name}: {track.status === 'instrumental' ? 'Instrumental' : `${Math.round(track.coverage * 100)}% confident words${track.status === 'needs_review' ? ' · review supplied lyrics' : ''}`}</p>)}
                  {!!project.alignment && !project.alignment.ready && <p className="text-xs text-amber-400">Review the recordings, captions and lyrics above. Check intros, repeated sections and omitted verses before making an edited copy.</p>}
                  <label className="flex items-center gap-2 text-xs"><input type="checkbox" checked={aligned} disabled={!project.alignment?.ready || !!project.resume_available} onChange={event => setAligned(event.target.checked)}/>Use lyric timing when training</label>
                  {project.resume_available && !project.training_options?.lyric_alignment && <p className="text-xs text-text-muted">Start a new experiment to add lyric timing to an existing training run.</p>}
                </div>
                <div className="grid grid-cols-2 gap-3">
                  <label className="text-xs">Total training steps<input type="number" min={1} max={1600} className={`${field} mt-1`} value={steps} onChange={event => setSteps(Number(event.target.value))}/></label>
                  <label className="text-xs">Adapter rank<select className={`${field} mt-1`} value={rank} disabled={!!project.training_options} onChange={event => setRank(Number(event.target.value))}>{[4, 8, 16, 32, 64].map(value => <option key={value}>{value}</option>)}</select></label>
                </div>
                <p className="text-xs text-text-muted">Saves a checkpoint every 200 steps and when stopped. Compare held-out results before training longer.</p>
                <button type="button" className={button} disabled={busy || steps <= (project.completed_steps || 0)} onClick={() => void run(async () => {
                  await startMusicTraining(project.id, {steps, rank, seed: project.training_options?.seed ?? 22005,
                    learning_rate: project.training_options?.learning_rate ?? 0.0001, resume: !!project.resume_available,
                    lyric_alignment: project.resume_available ? !!project.training_options?.lyric_alignment : aligned, audition})
                })}>{project.resume_available ? 'Resume training' : 'Start training'}</button>
                <details className="space-y-3 rounded-lg border border-border p-3">
                  <summary className="cursor-pointer text-sm">Adapt source sound · experimental</summary>
                  <p className="text-xs text-text-muted">Train the audio decoder against the actual recordings. This uses separate checkpoints and keeps the tokenizer fixed. Audition a short run before extending it; voice similarity is not guaranteed.</p>
                  <button type="button" className={button} disabled={busy} onClick={() => void run(async () => {await prepareMusicSound(project.id)})}>{project.audio_prepared ? 'Check source sound' : 'Prepare source sound'}</button>
                  {project.audio_prepared && <>
                    <label className="block text-xs">Total audio training steps<input type="number" min={1} max={1600} className={`${field} mt-1`} value={audioSteps} onChange={event => setAudioSteps(Number(event.target.value))}/></label>
                    <label className="block text-xs">Music conditioning<select className={`${field} mt-1`} disabled={!!project.audio_resume_available} value={conditioning} onChange={event => setConditioning(event.target.value)}><option value="">Base YuE2</option>{project.checkpoints.map(row => <option key={row.file} value={row.file}>Music checkpoint {row.step}</option>)}</select></label>
                    <p className="text-xs text-text-muted">The saved audio style uses the same music checkpoint it learned with. Saves every 100 steps and when stopped.</p>
                    <button type="button" className={button} disabled={busy || audioSteps <= (project.audio_completed_steps || 0)} onClick={() => void run(async () => {
                      await adaptMusicSound(project.id, {steps: audioSteps, seed: project.audio_training_options?.seed ?? 22005,
                        learning_rate: project.audio_training_options?.learning_rate ?? .00005,
                        conditioning_checkpoint: project.audio_training_options?.conditioning_checkpoint ?? conditioning,
                        resume: !!project.audio_resume_available, audition})
                    })}>{project.audio_resume_available ? 'Resume audio adaptation' : 'Adapt source sound'}</button>
                  </>}
                </details>
              </>}
            </div>}
          {project.checkpoints.length > 0 && <div className="space-y-2">
            <p className="text-sm">Saved checkpoints</p>
            {project.checkpoints.map(item => <div key={item.file} className="flex items-center justify-between gap-2 rounded border border-border p-2">
              <div className="text-xs">Step {item.step}<p className="text-[10px] text-text-muted">{typeof item.scores.heldout === 'number' ? `Held-out loss ${item.scores.heldout.toFixed(3)} · control ${item.scores.minted_val?.toFixed(3)}` : 'Evaluation pending'}{project.baseline?.heldout ? ` (base ${project.baseline.heldout.toFixed(3)})` : ''}</p></div>
              <button type="button" className={button} disabled={busy} onClick={() => void run(async () => {const style = await auditionMusicCheckpoint(project.id, item.file); onSelect(style); onClose()})}>Use for audition</button>
            </div>)}
            <p className="text-xs text-text-muted">Audition on new lyrics with a fixed seed. Listening quality matters more than loss alone.</p>
          </div>}
          {!!project.audio_checkpoints?.length && <div className="space-y-2">
            <p className="text-sm">Audio checkpoints</p>
            {project.audio_checkpoints.map(item => <div key={item.file} className="flex items-center justify-between gap-2 rounded border border-border p-2">
              <div className="text-xs">Audio step {item.step}<p className="text-[10px] text-text-muted">{typeof item.scores.heldout === 'number' ? `Held-out flow loss ${item.scores.heldout.toFixed(3)} · control ${item.scores.minted_val?.toFixed(3)}` : 'Evaluation pending'}{project.audio_baseline?.heldout ? ` (base ${project.audio_baseline.heldout.toFixed(3)})` : ''}</p></div>
              <button type="button" className={button} disabled={busy} onClick={() => void run(async () => {const style = await auditionAudioCheckpoint(project.id, item.file); onSelect(style); onClose()})}>Use for audition</button>
            </div>)}
          </div>}
        </div>}
      </div>}
      <p className="border-t border-border pt-3 text-[10px] text-text-muted">YuE2 and the real-audio tokenizer use CC BY-NC 4.0 weights. Your recordings, prepared data and checkpoints stay local.</p>
    </div>
  </dialog>, document.body)
}
