import { useState } from 'react'
import type { ReactNode } from 'react'
import { adaptMusicSound, alignMusicLyrics, auditionAudioCheckpoint, auditionMusicCheckpoint, prepareMusicSound, startMusicPreparation, startMusicTraining } from '../../api/musicTraining'
import type { MusicAuditionSettings, MusicProject, MusicStyle } from '../../api/musicTraining'
import { MusicAuditions } from './MusicAuditions'
import { JointMusicTraining } from './JointMusicTraining'

const field = 'w-full rounded-lg border border-border bg-bg-tertiary p-2 text-sm text-text-primary'
const button = 'rounded-lg border border-border px-3 py-2 text-xs hover:bg-bg-tertiary disabled:opacity-40'
const primary = `${button} border-accent-orange/50 bg-accent-orange/10 text-accent-orange`

export function SongStyleTraining({project, styles, advanced, expertTools, busy, run, onSelect}: {
  project: MusicProject; styles: MusicStyle[]; advanced: boolean; expertTools: ReactNode; busy: boolean
  run: (action: () => Promise<void>) => Promise<void>; onSelect: (style: MusicStyle) => void
}) {
  const [stepTarget, setStepTarget] = useState<number | null>(null)
  const [newRank, setRank] = useState(64)
  const [newAligned, setAligned] = useState(false)
  const [audioSteps, setAudioSteps] = useState(Math.min(1600, Math.max(200, (project.audio_completed_steps || 0) + 100)))
  const [conditioning, setConditioning] = useState(project.audio_training_options?.conditioning_checkpoint || '')
  const [auditionDraft, setAudition] = useState<MusicAuditionSettings | null>(null)
  const [expertOpen, setExpertOpen] = useState(advanced)
  const audition = auditionDraft ?? project.audition_settings ?? {enabled: false}
  const completed = project.completed_steps || 0
  const savedTarget = project.training_options?.steps || 0
  const steps = stepTarget ?? (savedTarget > completed ? savedTarget : Math.min(1600, Math.max(200, completed + 200)))
  const rank = project.training_options?.rank ?? newRank
  const aligned = project.resume_available ? !!project.training_options?.lyric_alignment : newAligned
  const checkpoints = [...project.checkpoints].sort((a, b) => b.step - a.step)
  const latest = checkpoints[0]
  const active = ['queued', 'preparing', 'training', 'auditioning'].includes(project.status)
  const disabled = busy || active
  const validTarget = Number.isInteger(steps) && steps > completed && steps <= 1600
  const train = () => void run(async () => {
    await startMusicTraining(project.id, {steps, rank, seed: project.training_options?.seed ?? 22005,
      learning_rate: project.training_options?.learning_rate ?? 0.0001, resume: !!project.resume_available,
      lyric_alignment: aligned, audition})
    setStepTarget(null)
  })
  const selectCheckpoint = (file: string) => void run(async () => {onSelect(await auditionMusicCheckpoint(project.id, file))})
  return <div className="space-y-3">
    <section aria-label="Learn song style" className="space-y-3 rounded-lg border border-accent-orange/40 p-3">
      <h3 className="text-sm font-medium">{latest ? '4. Try a song' : '3. Learn song style'}</h3>
      <p className="text-xs leading-relaxed text-text-muted">{project.adapted_pair
        ? `Voice training is already included (sound step ${project.adapted_pair.step}). This stage learns musical structure, phrasing and style using that saved sound.`
        : 'This project learns musical structure, phrasing and style with its existing sound decoder.'}</p>
      <p className="text-xs text-text-muted">Your supplied lyrics are always used for song-style training. There is no need to enable an extra lyric option.</p>
      {latest ? <>
        <p className="text-xs">Song style saved at {latest.step} steps. Try it with new lyrics before deciding whether to train longer.</p>
        <button type="button" className={primary} disabled={busy} onClick={() => selectCheckpoint(latest.file)}>Use LoRA &amp; try a song</button>
        <p className="text-xs text-text-muted">Returns to Studio with this LoRA selected. Your training trigger is added automatically; enter new lyrics and generate.</p>
      </> : !project.prepared ? <>
        <p className="text-xs text-text-muted">Prepare the recordings for this stage once. Maestro uses the correct saved voice version automatically.</p>
        <button type="button" className={primary} disabled={disabled} onClick={() => void run(async () => {await startMusicPreparation(project.id)})}>Prepare song-style training</button>
      </> : <>
        <p className="text-xs text-text-muted">{project.resume_available ? `Continue from ${completed} to ${steps} steps.` : `Start with ${steps} steps, then listen to a new song.`} You can continue training later in Expert settings.</p>
        <button type="button" className={primary} disabled={disabled || !validTarget} onClick={train}>{project.resume_available ? 'Resume song-style training' : 'Train song style'}</button>
      </>}
    </section>
    {(project.prepared || latest || !!project.audio_checkpoints?.length) && <details className="space-y-3 rounded-lg border border-border p-3">
      <summary className="cursor-pointer text-sm">Test songs &amp; comparisons</summary>
      <MusicAuditions project={project} settings={audition} onChange={setAudition} busy={busy} run={run}/>
    </details>}
    <details open={expertOpen} onToggle={event => setExpertOpen(event.currentTarget.open)} className="space-y-3 rounded-lg border border-border p-3">
      <summary className="cursor-pointer text-sm">Expert settings</summary>
      {expertTools}
      <div className="grid grid-cols-2 gap-3">
        <label className="min-w-0 text-xs">Total training steps<input type="number" min={1} max={1600} disabled={disabled} className={`${field} mt-1`} value={steps} onChange={event => setStepTarget(Number(event.target.value))}/></label>
        <label className="min-w-0 text-xs">Adapter rank<select className={`${field} mt-1`} value={rank} disabled={disabled || !!project.training_options} onChange={event => setRank(Number(event.target.value))}>{[4, 8, 16, 32, 64].map(value => <option key={value}>{value}</option>)}</select></label>
      </div>
      <p className="text-xs text-text-muted">Steps are the total target, including completed training. Rank is the adapter’s learning capacity: higher uses more memory and does not guarantee a better voice. The song-style default is 64; voice/sound adaptation uses 32. Existing runs keep their settings.</p>
      <p className="text-xs text-text-muted">Saves every 200 steps and when stopped. Compare new songs before training longer.</p>
      {!!latest && project.prepared && <button type="button" className={button} disabled={disabled || !validTarget} onClick={train}>Continue song-style training</button>}
      {project.prepared && <details className="space-y-2 rounded-lg border border-border p-3">
        <summary className="cursor-pointer text-xs">Extra word-timing guidance · experimental</summary>
        <p className="text-xs text-text-muted">Lyrics are already included. This adds a separate training objective for when each word occurs. It is off by default and is not required to learn lyrics or vocals.</p>
        <button type="button" className={button} disabled={disabled} onClick={() => void run(async () => {await alignMusicLyrics(project.id)})}>{project.alignment ? 'Check lyric alignment' : 'Align lyrics'}</button>
        {project.alignment?.tracks.map(track => <p key={track.track_id} className="text-xs text-text-muted">{project.tracks.find(t => t.id === track.track_id)?.name}: {track.status === 'instrumental' ? 'Instrumental' : `${Math.round(track.coverage * 100)}% confident words${track.status === 'needs_review' ? ' · review supplied lyrics' : ''}`}</p>)}
        {!!project.alignment && !project.alignment.ready && <p className="text-xs text-amber-400">Review the recordings, captions and lyrics above. Check intros, repeated sections and omitted verses before making an edited copy.</p>}
        <label className="flex items-center gap-2 text-xs"><input type="checkbox" checked={aligned} disabled={disabled || !project.alignment?.ready || !!project.resume_available} onChange={event => setAligned(event.target.checked)}/>Add extra word-timing guidance</label>
        {project.resume_available && <p className="text-xs text-text-muted">This run keeps word-timing guidance {aligned ? 'on' : 'off'}. Use a new experiment to change it.</p>}
      </details>}
      {project.prepared && <button type="button" className={button} disabled={disabled} onClick={() => void run(async () => {await startMusicPreparation(project.id)})}>Check prepared audio</button>}
      {!!checkpoints.length && <details className="space-y-2 border-t border-border pt-3">
        <summary className="cursor-pointer text-xs">Saved song-style versions</summary>
        {checkpoints.map(item => <div key={item.file} className="flex flex-wrap items-center justify-between gap-2 rounded border border-border p-2">
          <div className="text-xs">Step {item.step}<p className="text-[10px] text-text-muted">{typeof item.scores.heldout === 'number' ? `Held-out loss ${item.scores.heldout.toFixed(3)}` : 'Evaluation pending'}</p></div>
          <button type="button" className={button} disabled={busy} onClick={() => selectCheckpoint(item.file)}>Use this LoRA</button>
        </div>)}
      </details>}
      {advanced && project.prepared && <>
        <JointMusicTraining key={project.id} project={project} styles={styles.filter(style => !style.archived)} audition={audition} busy={busy} run={run} onSelect={onSelect}/>
        <details className="space-y-3 rounded-lg border border-border p-3">
          <summary className="cursor-pointer text-sm">Adapt source sound · decoder-only experiment</summary>
          <p className="text-xs text-text-muted">An alternate method that keeps the tokenizer fixed. This is separate from the guided voice &amp; sound stage and is not needed to complete it.</p>
          <button type="button" className={button} disabled={disabled} onClick={() => void run(async () => {await prepareMusicSound(project.id)})}>{project.audio_prepared ? 'Check source sound' : 'Prepare source sound'}</button>
          {project.audio_prepared && <>
            <label className="block text-xs">Total audio training steps<input type="number" min={1} max={1600} disabled={disabled} className={`${field} mt-1`} value={audioSteps} onChange={event => setAudioSteps(Number(event.target.value))}/></label>
            <label className="block text-xs">Music conditioning<select className={`${field} mt-1`} disabled={disabled || !!project.audio_resume_available} value={conditioning} onChange={event => setConditioning(event.target.value)}><option value="">Base YuE2</option>{project.checkpoints.map(row => <option key={row.file} value={row.file}>Music checkpoint {row.step}</option>)}</select></label>
            <p className="text-xs text-text-muted">The saved LoRA uses the same music checkpoint it learned with. Saves every 100 steps and when stopped.</p>
            <button type="button" className={button} disabled={disabled || !Number.isInteger(audioSteps) || audioSteps <= (project.audio_completed_steps || 0) || audioSteps > 1600} onClick={() => void run(async () => {
              await adaptMusicSound(project.id, {steps: audioSteps, seed: project.audio_training_options?.seed ?? 22005,
                learning_rate: project.audio_training_options?.learning_rate ?? .00005,
                conditioning_checkpoint: project.audio_training_options?.conditioning_checkpoint ?? conditioning,
                resume: !!project.audio_resume_available, audition})
            })}>{project.audio_resume_available ? 'Resume audio adaptation' : 'Adapt source sound'}</button>
          </>}
          {project.audio_checkpoints?.map(item => <div key={item.file} className="flex flex-wrap items-center justify-between gap-2 rounded border border-border p-2">
            <p className="text-xs">Audio step {item.step}</p>
            <button type="button" className={button} disabled={busy} onClick={() => void run(async () => {onSelect(await auditionAudioCheckpoint(project.id, item.file))})}>Use this audio LoRA</button>
          </div>)}
        </details>
      </>}
    </details>
  </div>
}
