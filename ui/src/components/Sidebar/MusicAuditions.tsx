import { musicAuditionUrl, renderMusicAudition } from '../../api/musicTraining'
import type { MusicAuditionSettings, MusicProject } from '../../api/musicTraining'

const field = 'w-full rounded-lg border border-border bg-bg-tertiary p-2 text-sm text-text-primary'
const button = 'rounded-lg border border-border px-3 py-2 text-xs hover:bg-bg-tertiary disabled:opacity-40'

export function MusicAuditions({project, settings, onChange, busy, run}: {
  project: MusicProject; settings: MusicAuditionSettings; onChange: (settings: MusicAuditionSettings) => void
  busy: boolean; run: (action: () => Promise<void>) => Promise<void>
}) {
  const active = ['queued', 'preparing', 'training', 'auditioning'].includes(project.status)
  const disabled = busy || active
  const edit = (value: Partial<MusicAuditionSettings>) => onChange({...settings, ...value})
  const valid = !!settings.style?.trim() && !!settings.lyrics?.trim()
  const checkpoints = [...project.checkpoints.map(row => ({...row, branch: 'style' as const})),
    ...(project.audio_checkpoints || []).map(row => ({...row, branch: 'audio' as const})),
    ...(project.joint_checkpoints || []).map(row => ({...row, branch: 'joint' as const}))]
  const label = (branch: string) => branch === 'joint' ? 'Music + sound' : branch === 'audio' ? 'Sound adaptation' : 'Style'
  return <div className="space-y-3 rounded-lg border border-border p-3">
    <label className="flex items-center gap-2 text-sm"><input type="checkbox" checked={settings.enabled} disabled={disabled} onChange={event => edit({enabled: event.target.checked})}/>Automatically audition checkpoints</label>
    <p className="text-xs text-text-muted">Save, render the same short test song, then resume training. Uses extra GPU time every 200 style steps or 100 audio/joint steps, and at the final checkpoint. Preview audio stays with this project.</p>
    {settings.enabled && <fieldset disabled={disabled} className="space-y-3 disabled:opacity-60">
      <label className="block text-xs">Audition style caption<textarea rows={2} className={`${field} mt-1`} value={settings.style || ''} maxLength={2000} onChange={event => edit({style: event.target.value})} placeholder="Describe the new test song, voice and instruments"/></label>
      <label className="block text-xs">Audition lyrics<textarea rows={4} className={`${field} mt-1`} value={settings.lyrics || ''} maxLength={4000} onChange={event => edit({lyrics: event.target.value})} placeholder="A short verse of new lyrics, or [Instrumental]"/></label>
      <div className="grid grid-cols-3 gap-2">
        <label className="min-w-0 text-xs">Max seconds<input aria-label="Maximum seconds" className={`${field} mt-1`} type="number" min={8} max={60} value={settings.seconds ?? 30} onChange={event => edit({seconds: Number(event.target.value)})}/></label>
        <label className="min-w-0 text-xs">Fixed seed<input className={`${field} mt-1`} type="number" min={0} max={4294967295} value={settings.seed ?? 22005} onChange={event => edit({seed: Number(event.target.value)})}/></label>
        <label className="min-w-0 text-xs">Style strength<input className={`${field} mt-1`} type="number" min={0} max={1.5} step={0.1} value={settings.strength ?? 1} onChange={event => edit({strength: Number(event.target.value)})}/></label>
      </div>
      <p className="text-[11px] text-text-muted">Direct mode · 32 decoder steps · temperature 0.8 · top-p 0.95 · top-k 64. YuE2 can end early or reach the time limit. Keep this request unchanged to compare checkpoints; it does not use the current Studio prompt.</p>
      {!!checkpoints.length && <label className="block text-xs">Render a saved checkpoint now<select aria-label="Render saved checkpoint" value="" className={`${field} mt-1`} disabled={disabled || !valid} onChange={event => {
        const row = checkpoints[Number(event.target.value)]
        if (row) void run(async () => {await renderMusicAudition(project.id, row.branch, row.file, settings)})
      }}><option value="">Choose checkpoint…</option>{checkpoints.map((row, index) => <option key={`${row.branch}-${row.step}`} value={index}>{label(row.branch)} · step {row.step}</option>)}</select></label>}
    </fieldset>}
    {!!project.auditions?.length && <div className="space-y-3 border-t border-border pt-3">
      <p className="text-sm">Checkpoint comparisons</p>
      <p className="text-[11px] text-text-muted">Compare samples with the same request ID and generation engine. Style and sound-adaptation checkpoints are separate comparisons; listening quality matters more than a lower loss.</p>
      {project.auditions.map(sample => <div key={sample.id} className="space-y-2 rounded border border-border p-2">
        <p className="text-xs">{label(sample.branch)} · step {sample.step} · {sample.status === 'completed' ? `${sample.seconds?.toFixed(1)}s` : sample.status}</p>
        {sample.status === 'completed' && <audio controls preload="none" className="w-full" aria-label={`${sample.branch} step ${sample.step} audition`} src={musicAuditionUrl(project.id, sample.id)}/>}
        {sample.metadata?.truncated?.semantic && <p className="text-[11px] text-amber-400">Reached the sample time limit; the ending may be cut off.</p>}
        {sample.error && <p className="break-words text-xs text-amber-400">{sample.error}</p>}
        <details className="text-[11px] text-text-muted"><summary className="cursor-pointer">Request {sample.request_id} · seed {sample.settings.seed} · strength {sample.settings.strength}{sample.engine ? ` · ${sample.engine}` : ''}</summary>
          <p className="mt-2 whitespace-pre-wrap">{sample.settings.style}</p><p className="mt-2 whitespace-pre-wrap">{sample.settings.lyrics}</p>
        </details>
        {['failed', 'cancelled'].includes(sample.status) && <button type="button" className={button} disabled={disabled} onClick={() => void run(async () => {await renderMusicAudition(project.id, sample.branch, sample.checkpoint, sample.settings)})}>Retry this sample</button>}
      </div>)}
    </div>}
  </div>
}
