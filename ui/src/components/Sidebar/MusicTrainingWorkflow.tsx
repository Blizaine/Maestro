import { useState } from 'react'
import { cancelJob } from '../../api/client'
import { forkMusicProject } from '../../api/musicTraining'
import type { MusicProject, MusicStyle } from '../../api/musicTraining'
import { AuthorMusicTraining } from './AuthorMusicTraining'
import { SongStyleTraining } from './SongStyleTraining'

const field = 'w-full rounded-lg border border-border bg-bg-tertiary p-2 text-sm text-text-primary'
const button = 'rounded-lg border border-border px-3 py-2 text-xs hover:bg-bg-tertiary disabled:opacity-40'

export function MusicTrainingWorkflow({project, projects, styles, busy, run, onOpenProject, onSelect}: {
  project: MusicProject; projects: MusicProject[]; styles: MusicStyle[]; busy: boolean
  run: (action: () => Promise<void>) => Promise<void>; onOpenProject: (id: string) => void; onSelect: (style: MusicStyle) => void
}) {
  const [choice, setChoice] = useState<'author' | 'style' | 'advanced' | null>(null)
  const stored = project.training_workflow || (project.completed_steps || project.checkpoints.length ? 'style' : 'author')
  // A paired song-style project has already completed voice learning. Never
  // offer a fresh voice run here; its resumable state belongs to the parent.
  const path = choice || (project.adapted_pair && stored === 'author' ? 'style' : stored)
  const active = ['queued', 'preparing', 'training', 'auditioning'].includes(project.status)
  const disabled = busy || active
  const sourceProject = projects.find(item => item.id === project.adapted_pair?.source_project)
  const step = path === 'author' ? 1 : project.checkpoints.length ? 3 : 2
  const expertTools = <div className="space-y-3 border-b border-border pb-3">
    <label className="block text-xs">Training method<select aria-label="Training method" className={`${field} mt-1`} value={path} disabled={disabled} onChange={event => setChoice(event.target.value as 'author' | 'style' | 'advanced')}>
      {!project.adapted_pair && <option value="author">Voice &amp; sound, then song style · guided</option>}
      <option value="style">Song style{project.adapted_pair ? ' · keep learned voice & sound' : ' only · existing decoder'}</option>
      <option value="advanced">Other training experiments</option>
    </select></label>
    {sourceProject && <div className="space-y-2 text-xs text-text-muted">
      <p>Voice training: {sourceProject.name} · step {project.adapted_pair?.step}. Further voice training belongs to that original project.</p>
      <button type="button" className={button} disabled={disabled} onClick={() => onOpenProject(sourceProject.id)}>Open original sound training</button>
    </div>}
    <p className="text-xs text-text-muted">Tokenizer/decoder {project.tokenizer_pair || 'v4'} · fixed for this project.</p>
    <div className="flex flex-wrap gap-2">
      {(project.tokenizer_pair || 'v4') === 'v4' && <button type="button" className={button} disabled={disabled} onClick={() => void run(async () => {onOpenProject((await forkMusicProject(project.id, 'v9')).id)})}>Compare v9 in a new project</button>}
      <button type="button" className={button} disabled={disabled} onClick={() => void run(async () => {onOpenProject((await forkMusicProject(project.id)).id)})}>New experiment with these recordings</button>
    </div>
  </div>
  return <div className="space-y-3">
    <ol aria-label="Training stages" className="grid grid-cols-2 gap-2 text-xs sm:grid-cols-4">
      {['Review recordings', 'Learn voice & sound', 'Learn song style', 'Try a song'].map((label, index) => <li key={label}
        aria-current={step === index ? 'step' : undefined} className={`rounded-lg border p-2 ${step === index ? 'border-accent-orange/50 bg-accent-orange/10 text-accent-orange' : 'border-border text-text-muted'}`}>
        {index + 1}. {label}
      </li>)}
    </ol>
    {path === 'author' ? <AuthorMusicTraining project={project} busy={busy} run={run} sourceProject={sourceProject} styleProjects={projects} onOpenProject={onOpenProject} expertTools={expertTools}/>
      : <SongStyleTraining project={project} styles={styles} advanced={path === 'advanced'} expertTools={expertTools} busy={busy} run={run} onSelect={onSelect}/>}
    {active && <button type="button" className={button} disabled={busy || !project.job_id} onClick={() => void run(async () => {if (project.job_id) await cancelJob(project.job_id)})}>Stop after current step</button>}
  </div>
}
