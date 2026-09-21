import { useState } from 'react'
import type { ReactNode } from 'react'
import { adaptMusicPair, compareMusicPair, prepareMusicPair, selectMusicPair } from '../../api/musicTraining'
import type { MusicProject } from '../../api/musicTraining'

const button = 'rounded-lg border border-border px-3 py-2 text-xs hover:bg-bg-tertiary disabled:opacity-40'
const primary = `${button} border-accent-orange/50 bg-accent-orange/10 text-accent-orange`
const field = 'w-full rounded-lg border border-border bg-bg-tertiary p-2 text-sm text-text-primary'

export function AuthorMusicTraining({project, sourceProject, styleProjects = [], busy, run, onOpenProject, expertTools}: {
  project: MusicProject; sourceProject?: MusicProject; styleProjects?: MusicProject[]; busy: boolean
  run: (action: () => Promise<void>) => Promise<void>; onOpenProject: (id: string) => void; expertTools?: ReactNode
}) {
  const [stepTarget, setStepTarget] = useState<number | null>(null)
  const [comparison, setComparison] = useState('')
  const completed = project.pair_completed_steps || 0
  const savedTarget = project.pair_training_options?.steps || 0
  const steps = stepTarget ?? (savedTarget > completed ? savedTarget : Math.min(1600, Math.max(100, completed + 100)))
  const checkpoints = [...(project.pair_checkpoints || [])].sort((a, b) => b.step - a.step)
  const latest = checkpoints.find(row => row.step > 0)
  const active = ['queued', 'preparing', 'training', 'auditioning'].includes(project.status)
  const disabled = busy || active
  const validTarget = Number.isInteger(steps) && steps > completed && steps <= 1600
  const train = () => void run(async () => {
    await adaptMusicPair(project.id, steps, !!project.pair_resume_available, project.pair_training_options?.seed ?? 22005)
    setStepTarget(null)
  })
  const continueWith = (checkpoint: NonNullable<MusicProject['pair_checkpoints']>[number]) => void run(async () => {
    // Reopening a stage resumes its existing project instead of duplicating
    // the same sound checkpoint every time the user clicks Continue.
    const existing = styleProjects.filter(item => item.adapted_pair?.source_project === project.id && item.adapted_pair.step === checkpoint.step)
      .sort((a, b) => (b.completed_steps || 0) - (a.completed_steps || 0))[0]
    const next = existing || await selectMusicPair(project.id, checkpoint.file)
    onOpenProject(next.id)
  })
  if (sourceProject && project.adapted_pair && !project.pair_checkpoints?.length && !project.pair_resume_available) {
    return <div className="space-y-3 rounded-lg border border-accent-orange/40 p-3">
      <p className="text-sm font-medium">Voice training is already included</p>
      <p className="text-xs leading-relaxed text-text-muted">This project uses voice &amp; sound step {project.adapted_pair.step}. To train the voice further, return to its original project. Your song-style training stays saved here.</p>
      <p className="break-words text-xs text-text-muted">{sourceProject.name}{typeof sourceProject.pair_completed_steps === 'number' && ` · ${sourceProject.pair_completed_steps} sound steps completed`}</p>
      <button type="button" className={button} disabled={disabled} onClick={() => onOpenProject(sourceProject.id)}>Open original sound training</button>
    </div>
  }
  return <div className="space-y-3">
    <section aria-label="Learn voice and sound" className="space-y-3 rounded-lg border border-accent-orange/40 p-3">
      <h3 className="text-sm font-medium">2. Learn voice &amp; sound</h3>
      <p className="text-xs leading-relaxed text-text-muted">Learn vocal character, instruments and production from your recordings. Song structure and lyrics come next.</p>
      {latest ? <>
        <p className="text-xs">Voice &amp; sound saved at {latest.step} steps. Continue with this version to learn song style.</p>
        <button type="button" className={primary} disabled={disabled} onClick={() => continueWith(latest)}>Continue to song style</button>
        <p className="text-xs text-text-muted">Maestro keeps the voice and song-style parts together. Earlier versions and further voice training are in Expert settings.</p>
      </> : !project.pair_prepared ? <>
        <p className="text-xs text-text-muted">Prepare the recordings once before training. This downloads about 1.1 GB of reference audio to help preserve general music ability, plus any missing models.</p>
        <button type="button" className={primary} disabled={disabled} onClick={() => void run(async () => {await prepareMusicPair(project.id)})}>Prepare voice training</button>
      </> : <>
        <p className="text-xs text-text-muted">{project.pair_resume_available ? `Continue from ${completed} to ${steps} steps.` : `Start with ${steps} steps, then train song style and try a song.`} More training is available later; more steps do not always sound better.</p>
        <button type="button" className={primary} disabled={disabled || !validTarget} onClick={train}>{project.pair_resume_available ? 'Resume voice & sound' : 'Train voice & sound'}</button>
      </>}
      {!!project.pair_skipped_tracks?.length && <p className="text-xs text-amber-400">{project.pair_skipped_tracks.length} excerpts were too short for voice training. They are still available for song-style training.</p>}
    </section>
    {comparison && <p role="status" className="text-xs text-text-muted">Sound comparison queued ({comparison}). Original, before and after audio will appear in the current gallery folder.</p>}
    <details className="space-y-3 rounded-lg border border-border p-3">
      <summary className="cursor-pointer text-sm">Expert settings</summary>
      {expertTools}
      <label className="block text-xs">Sound adaptation steps<input aria-label="Sound adaptation steps" type="number" min={1} max={1600} disabled={disabled} className={`${field} mt-1`} value={steps} onChange={event => setStepTarget(Number(event.target.value))}/></label>
      <p className="text-xs text-text-muted">Total target, including completed steps. Saves every 25 steps. {project.pair_resume_available ? `Resumes from ${completed} with the original settings.` : '100 steps is a short first comparison, not a guarantee of voice likeness.'}</p>
      {latest && project.pair_prepared && <button type="button" className={button} disabled={disabled || !validTarget} onClick={train}>Resume voice &amp; sound adaptation</button>}
      <p className="text-xs text-text-muted">Uses the author’s tokenizer + audio decoder method. The decoder adapter has rank 32; the separate song-style adapter defaults to rank 64. Rank controls learning capacity, not a quality score.</p>
      <p className="text-xs text-text-muted">Voice training needs excerpts of at least 20.5 seconds. Check-only excerpts need at least 5.2 seconds. Several complete phrases make a more useful comparison.</p>
      {!!checkpoints.length && <details className="space-y-2 border-t border-border pt-3">
        <summary className="cursor-pointer text-xs">Saved sound versions &amp; comparisons</summary>
        {checkpoints.map(row => <div key={row.file} className="space-y-2 rounded border border-border p-2">
          <p className="text-xs">{row.step ? `Sound pair · step ${row.step}` : 'Original sound pair · baseline'}</p>
          {typeof row.scores.heldout_spectral === 'number' && <p className="text-[10px] text-text-muted">Check-only audio difference: {row.scores.heldout_spectral.toFixed(3)}. Lower is closer in this measurement; listen before choosing.</p>}
          <div className="flex flex-wrap gap-2">
            {row.step > 0 && <button type="button" className={button} disabled={disabled} onClick={() => void run(async () => {
              const result = await compareMusicPair(project.id, row.file); setComparison(result.job_id)
            })}>Compare original / before / after</button>}
            <button type="button" className={button} disabled={disabled} onClick={() => continueWith(row)}>Train song style with this version</button>
          </div>
        </div>)}
      </details>}
    </details>
  </div>
}
