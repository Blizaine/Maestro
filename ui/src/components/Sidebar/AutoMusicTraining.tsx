import { useState } from 'react'
import { cancelJob } from '../../api/client'
import { startAutoMusicTraining } from '../../api/musicTraining'
import type { MusicProject, MusicStyle } from '../../api/musicTraining'
import { validAutoSteps } from '../../lib/musicTraining'

const field = 'w-full rounded-lg border border-border bg-bg-tertiary p-2 text-sm text-text-primary'
const button = 'rounded-lg border border-border px-3 py-2 text-xs hover:bg-bg-tertiary disabled:opacity-40'
const primary = `${button} border-accent-orange/50 bg-accent-orange/10 text-accent-orange`

export function AutoTrainingTargets({voice, style, onVoice, onStyle, disabled = false}: {
  voice: number; style: number; onVoice: (steps: number) => void; onStyle: (steps: number) => void; disabled?: boolean
}) {
  return <div className="space-y-2">
    <div className="grid grid-cols-2 gap-3">
      <label className="min-w-0 text-xs">Voice &amp; sound steps<input type="number" min={1} max={1600} className={`${field} mt-1`} disabled={disabled} value={voice} onChange={event => onVoice(Number(event.target.value))}/></label>
      <label className="min-w-0 text-xs">Song-style steps<input type="number" min={1} max={1600} className={`${field} mt-1`} disabled={disabled} value={style} onChange={event => onStyle(Number(event.target.value))}/></label>
    </div>
    <p className="text-xs text-text-muted">100 / 200 is a short first test. You can train either stage further after listening.</p>
  </div>
}

const stages = [
  {label: 'Prepare recordings', keys: ['analyze-songs', 'build-dataset']},
  {label: 'Learn voice & sound', keys: ['prepare-pair', 'adapt-pair']},
  {label: 'Learn song style', keys: ['select-pair', 'prepare', 'train']},
  {label: 'Save LoRA', keys: ['publish']},
]

export function AutoMusicTraining({project, styles, busy, run, onManual, onSelect, initialVoice = 100, initialStyle = 200}: {
  project: MusicProject; styles: MusicStyle[]; busy: boolean; run: (action: () => Promise<void>) => Promise<void>
  onManual: (id: string) => void; onSelect: (style: MusicStyle) => void; initialVoice?: number; initialStyle?: number
}) {
  const plan = project.auto_training
  const [voice, setVoice] = useState(initialVoice)
  const [style, setStyle] = useState(initialStyle)
  const [stopping, setStopping] = useState(false)
  const running = plan ? ['queued', 'running'].includes(plan.status) : ['queued', 'preparing', 'training', 'auditioning'].includes(project.status)
  const complete = plan?.status === 'completed'
  const saved = styles.find(item => item.id === plan?.style_id)
  const summary = plan?.selection_summary
  return <section aria-label="Auto music training" className="space-y-4 rounded-lg border border-accent-orange/40 p-3">
    <div><h3 className="text-sm font-medium">Auto training</h3><p className="mt-1 break-words text-xs text-text-muted">{project.name}</p></div>
    {!plan ? <>
      <p className="text-xs text-text-muted">Use the main detected voice and suggested excerpts, prepare the dataset, train both stages and save a LoRA. No clip-by-clip review or stage approvals. Automatic voice choices and transcripts can be imperfect; you can review them afterward.</p>
      <AutoTrainingTargets voice={voice} style={style} onVoice={setVoice} onStyle={setStyle} disabled={busy || running}/>
      <button type="button" className={primary} disabled={busy || running || !validAutoSteps(voice, style)} onClick={() => void run(async () => {await startAutoMusicTraining(project.id, voice, style)})}>Start Auto training</button>
    </> : <>
      <p className="text-xs text-text-muted">Voice &amp; sound: {plan.voice_steps} steps · Song style: {plan.style_steps} steps</p>
      <ol aria-label="Auto training stages" className="grid grid-cols-2 gap-2 text-xs">
        {stages.map((stage, index) => {
          const current = !complete && stage.keys.includes(plan.stage)
          const done = complete || stage.keys.every(key => plan.done?.includes(key))
          return <li key={stage.label} aria-current={current ? 'step' : undefined} className={`rounded-lg border p-2 ${current ? 'border-accent-orange/50 text-accent-orange' : 'border-border text-text-muted'}`}>
            {index + 1}. {stage.label}{done ? ' ✓' : ''}
          </li>
        })}
      </ol>
      <p role="status" className={`break-words text-xs ${plan.status === 'failed' ? 'text-amber-400' : ''}`}>{stopping && running ? 'Stopping after the current step is saved…' : project.message}</p>
      {running && <>
        <progress aria-label="Auto training progress" max={100} value={project.progress} className="w-full accent-accent-orange"/>
        <button type="button" className={button} disabled={busy || stopping || !project.job_id} onClick={() => void run(async () => {if (project.job_id) {await cancelJob(project.job_id); setStopping(true)}})}>Stop Auto training</button>
      </>}
      {!running && !complete && <button type="button" className={primary} disabled={busy} onClick={() => void run(async () => {
        await startAutoMusicTraining(project.id, plan.voice_steps, plan.style_steps); setStopping(false)
      })}>Resume Auto training</button>}
      {complete && saved && !saved.archived && <button type="button" className={primary} disabled={busy} onClick={() => onSelect(saved)}>Use LoRA &amp; try a song</button>}
      {complete && <p className="text-xs text-text-muted">Your LoRA is in Saved LoRAs. Try a song with new lyrics, or open either training stage below to continue it or compare earlier versions.</p>}
    </>}
    <p className="text-xs text-text-muted">Runs in Maestro’s GPU queue and continues if you close this panel or browser. Keep Maestro running. After an app restart, use Resume Auto training.</p>
    {summary && <details className="space-y-2 border-t border-border pt-3 text-xs text-text-muted">
      <summary className="cursor-pointer">Dataset: {summary.training_clips} training clips · {summary.check_clips} check-only clips</summary>
      <p>{summary.unreviewed_clips} clips were selected without human review. Check-only recordings never teach the model.</p>
      {summary.warnings.map((warning, index) => <p key={index} className="break-words">{warning}</p>)}
    </details>}
    <div className="flex flex-wrap gap-2 border-t border-border pt-3">
      {project.preparation_draft && <button type="button" className={button} disabled={busy || running} onClick={() => onManual(project.id)}>Open prepared clips</button>}
      {plan?.sound_project_id && <button type="button" className={button} disabled={busy || running} onClick={() => onManual(plan.sound_project_id!)}>Open voice training</button>}
      {plan?.style_project_id && <button type="button" className={button} disabled={busy || running} onClick={() => onManual(plan.style_project_id!)}>Open song-style training</button>}
    </div>
  </section>
}
