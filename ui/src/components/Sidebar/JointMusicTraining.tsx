import { useState } from 'react'
import { auditionJointCheckpoint, prepareMusicSound, trainJointMusic } from '../../api/musicTraining'
import type { MusicAuditionSettings, MusicProject, MusicStyle } from '../../api/musicTraining'

const button = 'rounded-lg border border-border px-3 py-2 text-xs hover:bg-bg-tertiary disabled:opacity-40'
const field = 'w-full rounded-lg border border-border bg-bg-tertiary p-2 text-sm text-text-primary'

export function JointMusicTraining({project, styles, audition, busy, run, onSelect}: {
  project: MusicProject; styles: MusicStyle[]; audition: MusicAuditionSettings; busy: boolean
  run: (action: () => Promise<void>) => Promise<void>; onSelect: (style: MusicStyle) => void
}) {
  const [steps, setSteps] = useState(Math.max(200, (project.joint_completed_steps || 0) + 100))
  const [initialStyle, setInitialStyle] = useState('')
  const active = ['queued', 'preparing', 'training', 'auditioning'].includes(project.status)
  const saved = project.joint_training_options
  const compatible = styles.filter(style => style.tokenizer_revision && style.trigger === project.trigger
    && (style.tokenizer_pair || 'v4') === (project.tokenizer_pair || 'v4')
    && (project.adapted_pair ? style.tokenizer_revision === project.prepared?.tokenizer_revision : !style.tokenizer_revision.startsWith('local-')))
  const initialId = project.joint_resume_available ? saved?.initialization?.style_id || '' : initialStyle
  return <details className="space-y-3 rounded-lg border border-border p-3">
    <summary className="cursor-pointer text-sm">Train music and sound together · experimental</summary>
    <p className="text-xs text-text-muted">Learn the song structure and audio rendering in one run. Saves a matched pair every 100 steps. Uses separate checkpoints, leaving your existing training intact.</p>
    {!active && <>
      {!project.audio_prepared && <button type="button" className={button} disabled={busy} onClick={() => void run(async () => {await prepareMusicSound(project.id)})}>Prepare source sound</button>}
      {project.audio_prepared && <>
        {project.joint_resume_available ? <p className="text-xs text-text-muted">Started from {saved?.initialization?.style_name || 'the base model'}. Resume keeps the same starting style.</p> : <>
          <label className="block text-xs">Joint training starting point<select aria-label="Joint training starting point" className={`${field} mt-1`} value={initialStyle} disabled={busy} onChange={event => setInitialStyle(event.target.value)}>
            <option value="">Base model</option>
            {compatible.map(style => <option key={style.id} value={style.id}>{style.name}</option>)}
          </select></label>
          <p className="text-xs text-text-muted">Saved styles must match this project's tokenizer and exact style trigger. Continuing a style preserves its current sound, saves a baseline, and starts new training checkpoints.</p>
        </>}
        <label className="block text-xs">Total joint training steps<input type="number" min={1} max={1600} className={`${field} mt-1`} value={steps} onChange={event => setSteps(Number(event.target.value))}/></label>
        <p className="text-xs text-text-muted">{initialId ? 'Preserves saved adapter ranks' : `Rank ${saved?.rank ?? 32}`} · up to 60-second audio windows · direct composition. Compare fixed auditions before extending a run.</p>
        <button type="button" className={button} disabled={busy || steps <= (project.joint_completed_steps || 0)} onClick={() => void run(async () => {
          await trainJointMusic(project.id, {steps, rank: saved?.rank ?? 32, seed: saved?.seed ?? 22005,
            learning_rate: saved?.learning_rate ?? (initialId ? .00002 : .0001), window_frames: saved?.window_frames ?? 1500,
            initial_style_id: initialId, resume: !!project.joint_resume_available, audition})
        })}>{project.joint_resume_available ? 'Resume joint training' : 'Start joint training'}</button>
      </>}
    </>}
    {project.joint_checkpoints?.map(row => <div key={row.file} className="flex items-center justify-between gap-2 rounded border border-border p-2">
      <span className="text-xs">{row.step === 0 ? 'Joint baseline' : `Joint step ${row.step}`}</span>
      <button type="button" className={button} disabled={busy || active} onClick={() => void run(async () => {onSelect(await auditionJointCheckpoint(project.id, row.file))})}>Use for audition</button>
    </div>)}
  </details>
}
