import { useEffect, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import { ScanFace, X, Upload, Loader2, Play } from 'lucide-react'
import { useStore } from '../../stores/useStore'
import * as api from '../../api/client'
import { characterDisplayName } from '../../lib/characters'
import type { FaceRefinerOptions, FaceRefinerAnalysis, FaceRefinerAssignment, SavedOmniCharacter } from '../../types'

export const defaultFaceRefiner: FaceRefinerOptions = {
  enabled: false, face_count: 0, strength: 0.75, steps: 4,
  window_frames: 243, model: 'auto', character_ids: [],
}
const inputClass = 'w-full min-w-0 rounded-lg border border-border bg-bg-tertiary px-3 py-2 text-sm text-text-primary'
type Source = { path: string; name: string; url: string | null }

function FaceSettings({ value, onChange, disabled = false }: {
  value: FaceRefinerOptions; onChange: (value: FaceRefinerOptions) => void; disabled?: boolean
}) {
  const [characters, setCharacters] = useState<SavedOmniCharacter[]>([])
  const [error, setError] = useState('')
  useEffect(() => { let active = true
    api.fetchCharacters().then(list => { if (active) setCharacters(list) }).catch(e => { if (active) setError(e.message) })
    return () => { active = false }
  }, [])
  const patch = (key: keyof FaceRefinerOptions, next: unknown) => onChange({ ...value, [key]: next })
  return <fieldset disabled={disabled} className="min-w-0 space-y-3 disabled:opacity-60">
    <div className="grid grid-cols-2 gap-3">
      <label className="space-y-1 text-xs text-text-secondary">Faces to refine
        <select aria-label="Faces to refine" className={inputClass} value={value.face_count} onChange={e => patch('face_count', Number(e.target.value))}>
          <option value={0}>Auto · up to 5</option>{[1, 2, 3, 4, 5].map(n => <option key={n} value={n}>{n} {n === 1 ? 'face' : 'faces'}</option>)}
        </select>
      </label>
      <label className="space-y-1 text-xs text-text-secondary">Refinement · {Math.round(value.strength * 100)}%
        <input aria-label="Face refinement strength" className="w-full mt-3 accent-blue-500" type="range" min={0.1} max={1} step={0.05}
          value={value.strength} onChange={e => patch('strength', Number(e.target.value))} />
      </label>
    </div>
    <details className="rounded-lg border border-border p-3">
      <summary className="cursor-pointer text-xs text-text-secondary">Character matching & advanced settings</summary>
      <div className="mt-3 space-y-3">
        <p className="text-xs text-text-muted">Saved characters from the video's generation are matched automatically. Select up to five here to use a different set. Unmatched faces keep their original identity.</p>
        {!!characters.length && <div className="max-h-48 overflow-y-auto grid grid-cols-1 sm:grid-cols-2 gap-1.5">
          {characters.map(c => <label key={c.id} className="flex min-w-0 items-center gap-2 rounded-lg bg-bg-tertiary p-2 text-xs cursor-pointer">
            <input type="checkbox" checked={value.character_ids.includes(c.id)} disabled={disabled || (!value.character_ids.includes(c.id) && value.character_ids.length >= 5)}
              onChange={e => patch('character_ids', e.target.checked ? [...value.character_ids, c.id] : value.character_ids.filter(id => id !== c.id))} />
            <img src={c.visual.thumbnail_url || c.visual.url} alt="" className="h-9 w-9 shrink-0 rounded-md object-cover" loading="lazy" />
            <span className="truncate">{characterDisplayName(c.name)}</span>
          </label>)}
        </div>}
        {error && <p role="alert" className="text-xs text-red-400">{error}</p>}
        <label className="block text-xs text-text-secondary">Refinement model
          <select aria-label="Face refinement model" className={inputClass} value={value.model} onChange={e => patch('model', e.target.value)}>
            <option value="auto">Auto · installed Fused, otherwise Pruned</option><option value="fused">H3 Fused · 4-step</option><option value="pruned">H3 Pruned + LightX2V</option>
          </select>
        </label>
        <div className="grid grid-cols-2 gap-3">
          <label className="text-xs text-text-secondary">Steps
            <select aria-label="Face refinement steps" className={inputClass} value={value.steps} onChange={e => patch('steps', Number(e.target.value))}>{[4, 5, 6, 7, 8].map(n => <option key={n}>{n}</option>)}</select>
          </label>
          <label className="text-xs text-text-secondary">Window length
            <select aria-label="Face refinement window" className={inputClass} value={value.window_frames} onChange={e => patch('window_frames', Number(e.target.value))}>
              <option value={124}>5.2s · less VRAM</option><option value={243}>10.1s</option><option value={345}>14.4s</option>
            </select>
            <span className="mt-1 block text-[10px] text-text-muted">Times shown at 24 fps.</span>
          </label>
        </div>
      </div>
    </details>
  </fieldset>
}

export function AutomaticFaceRefiner() {
  const options = useStore(s => s.params.face_refiner)
  const setParam = useStore(s => s.setParam)
  const value = { ...defaultFaceRefiner, ...options }
  return <div className="rounded-xl border border-border bg-bg-secondary p-3 space-y-3">
    <label className="flex items-center gap-2.5 cursor-pointer">
      <ScanFace size={18} className="text-accent-blue shrink-0" /><span className="flex-1 text-sm text-text-primary">Refine faces after generation</span>
      <input aria-label="Refine faces after generation" type="checkbox" checked={value.enabled} onChange={e => setParam('face_refiner', { ...value, enabled: e.target.checked })} className="accent-blue-500 h-4 w-4" />
    </label>
    {value.enabled && <><p className="text-xs text-text-muted">H3 restores facial detail and follows each identity across the clip. Saves a refined copy at the same size with the original soundtrack.</p>
      <FaceSettings value={value} onChange={next => setParam('face_refiner', next)} /></>}
  </div>
}

export function FaceRefinerButton({ source, className = '' }: { source?: Source; className?: string }) {
  const [open, setOpen] = useState(false)
  return <>
    <button type="button" onClick={() => setOpen(true)} title="Detect, map and refine faces in this video" className={`inline-flex items-center justify-center gap-1.5 rounded-lg border border-border px-2.5 py-1.5 text-xs text-text-secondary hover:text-accent-blue hover:bg-bg-hover ${className}`}>
      <ScanFace size={15} /><span>Refine faces</span>
    </button>
    {open && <FaceRefinerDialog initialSource={source} onClose={() => setOpen(false)} />}
  </>
}

export function FaceRefinerDialog({ initialSource, onClose }: { initialSource?: Source; onClose: () => void }) {
  const workspace = useStore(s => s.activeWorkspace)
  const [source, setSource] = useState<Source | undefined>(initialSource)
  const [options, setOptions] = useState<FaceRefinerOptions>({ ...defaultFaceRefiner, enabled: true })
  const [characters, setCharacters] = useState<SavedOmniCharacter[]>([])
  const [analysis, setAnalysis] = useState<FaceRefinerAnalysis | null>(null)
  const [assignments, setAssignments] = useState<FaceRefinerAssignment[]>([])
  const [busy, setBusy] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [status, setStatus] = useState('')
  const [error, setError] = useState('')
  const [resultUrl, setResultUrl] = useState('')
  const [jobId, setJobId] = useState('')
  const file = useRef<HTMLInputElement>(null)
  const dialog = useRef<HTMLDivElement>(null)
  const alive = useRef(true)
  useEffect(() => {
    alive.current = true
    const previousFocus = document.activeElement as HTMLElement | null
    dialog.current?.focus()
    api.fetchCharacters().then(list => { if (alive.current) setCharacters(list) }).catch(e => { if (alive.current) setError(e.message) })
    return () => { alive.current = false; previousFocus?.focus() }
  }, [])
  const resetAnalysis = () => { setAnalysis(null); setAssignments([]); setResultUrl(''); setError(''); setStatus('') }
  const changeOptions = (next: FaceRefinerOptions) => {
    if (next.face_count !== options.face_count || JSON.stringify(next.character_ids) !== JSON.stringify(options.character_ids)) resetAnalysis()
    setOptions(next)
  }
  const upload = async (selected: File) => {
    setUploading(true); setError('')
    try { const result = await api.uploadImage(selected); setSource({ path: result.path, name: selected.name, url: result.url }); resetAnalysis() }
    catch (e) { setError(e instanceof Error ? e.message : 'Upload failed') }
    finally { setUploading(false) }
  }
  const run = async (analyzeOnly: boolean) => {
    if (!source || busy) return
    setBusy(true); setError(''); setStatus('Queued'); setResultUrl('')
    let id = ''
    try {
      const submitted = await api.submitFaceRefiner({ video_path: source.path, workspace, options, analyze_only: analyzeOnly,
        ...(!analyzeOnly && analysis ? { analysis_id: analysis.id, assignments } : {}) })
      id = submitted.job_id; setJobId(id)
      useStore.setState(s => ({ isGenerating: true, jobs: [...s.jobs, { id, status: 'queued', progress: 0, step: 0, totalSteps: 0, phase: '',
        message: analyzeOnly ? 'Detecting faces...' : 'Refining faces...', outputFiles: [], error: null, oomInfo: null }] }))
      for (;;) {
        await new Promise(resolve => setTimeout(resolve, 1500))
        const result = await api.fetchJobStatus(id)
        useStore.setState(s => ({ jobs: s.jobs.map(j => j.id === id ? { ...j, status: result.status, progress: result.progress / 100,
          step: result.step, totalSteps: result.total_steps, phase: result.phase, message: result.message, outputFiles: result.output_files, error: result.error } : j) }))
        if (alive.current) setStatus(result.message)
        if (result.status === 'failed') throw new Error(result.error || 'Face refinement failed')
        if (result.status === 'cancelled') { if (alive.current) setStatus('Cancelled'); break }
        if (result.status !== 'completed') continue
        if (analyzeOnly && submitted.analysis_id) {
          const found = await api.fetchFaceAnalysis(submitted.analysis_id)
          if (alive.current) { setAnalysis(found); setAssignments(found.faces.map(face => ({ track_id: face.track_id, character_id: face.character_id, skip: false }))) }
        } else {
          await useStore.getState().loadOutputs()
          if (alive.current && result.output_files[0]) setResultUrl(`/api/v1/file/${encodeURIComponent(result.output_files[0])}?workspace=${encodeURIComponent(workspace)}`)
        }
        useStore.setState(s => ({ jobs: s.jobs.filter(j => j.id !== id) }))
        break
      }
    } catch (e) { if (alive.current) setError(e instanceof Error ? e.message : 'Face refinement failed') }
    finally {
      useStore.setState(s => ({ isGenerating: s.jobs.some(j => ['running', 'queued'].includes(j.status)) }))
      if (alive.current) { setBusy(false); setJobId('') }
    }
  }
  const cancel = async () => { if (jobId) { try { await api.cancelJob(jobId); setStatus('Cancelling...') } catch (e) { setError(String(e)) } } }
  return createPortal(<div className="fixed inset-0 z-[10000] flex items-center justify-center bg-black/70 p-2 sm:p-6" onClick={e => { if (e.target === e.currentTarget) onClose() }}>
    <div ref={dialog} role="dialog" aria-modal="true" aria-labelledby="face-refiner-title" tabIndex={-1}
      className="flex w-full max-w-3xl max-h-[94dvh] min-w-0 flex-col overflow-hidden rounded-2xl border border-border bg-bg-primary shadow-2xl"
      onKeyDown={e => {
        if (e.key === 'Escape') onClose()
        if (e.key === 'Tab') {
          const nodes = dialog.current?.querySelectorAll<HTMLElement>('button:not(:disabled), input:not(:disabled), select:not(:disabled), summary, [tabindex="0"]')
          if (!nodes?.length) return
          const first = nodes[0], last = nodes[nodes.length - 1]
          if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus() }
          else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus() }
        }
      }}>
      <div className="flex shrink-0 items-center gap-3 border-b border-border px-4 py-4 sm:px-6">
        <div className="rounded-xl bg-accent-blue/10 p-2.5 text-accent-blue"><ScanFace size={23} /></div>
        <div className="min-w-0 flex-1"><h2 id="face-refiner-title" className="text-lg font-semibold text-text-primary">Refine faces</h2><p className="text-xs text-text-muted">Restore facial detail with H3 · keep the performance</p></div>
        <button aria-label="Close Face Refiner" onClick={onClose} className="p-2 text-text-secondary hover:text-text-primary"><X size={20} /></button>
      </div>
      <div className="min-h-0 overflow-y-auto overscroll-contain px-4 py-4 sm:px-6 space-y-5">
        {source ? <div className="rounded-xl border border-border bg-bg-secondary p-3 space-y-2">
          {source.url && <video src={source.url} controls playsInline className="max-h-48 w-full rounded-lg bg-black" />}
          <div className="flex gap-2 items-center"><span className="min-w-0 flex-1 truncate text-xs text-text-secondary">{source.name}</span><button disabled={busy || uploading} className="shrink-0 text-xs text-accent-blue" onClick={() => file.current?.click()}>Change video</button></div>
        </div> : <button disabled={busy || uploading} onClick={() => file.current?.click()} className="flex w-full items-center justify-center gap-2 rounded-xl border border-dashed border-border p-8 text-sm text-text-secondary hover:border-accent-blue">
          {uploading ? <Loader2 className="animate-spin" size={20} /> : <Upload size={20} />} Upload a video
        </button>}
        <input ref={file} type="file" accept="video/*" hidden onChange={e => { if (e.target.files?.[0]) void upload(e.target.files[0]); e.target.value = '' }} />
        <FaceSettings value={options} onChange={changeOptions} disabled={busy} />
        <div className="space-y-3">
          <div className="flex flex-wrap gap-2 items-center justify-between"><div><h3 className="text-sm font-medium text-text-primary">Face mapping</h3><p className="text-xs text-text-muted">Optional · preview identities before refining</p></div>
            <button disabled={!source || busy || uploading} onClick={() => void run(true)} className="rounded-lg border border-border px-3 py-2 text-xs text-text-primary disabled:opacity-40 hover:bg-bg-hover">{analysis ? 'Detect again' : 'Detect & map faces'}</button>
          </div>
          {analysis && !analysis.faces.length && <p className="rounded-lg bg-bg-secondary p-3 text-sm text-text-secondary">No relevant faces found. Try selecting a specific face count to include smaller or briefly visible faces.</p>}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            {analysis?.faces.map(face => {
              const mapping = assignments.find(a => a.track_id === face.track_id)
              const chosen = characters.find(c => c.id === mapping?.character_id)
              return <div key={face.track_id} className={`flex min-w-0 gap-3 rounded-xl border border-border bg-bg-secondary p-3 ${mapping?.skip ? 'opacity-50' : ''}`}>
                <img src={face.thumbnail_url} alt={`Detected face ${face.track_id}`} className="h-20 w-20 shrink-0 rounded-lg object-cover" />
                <div className="min-w-0 flex-1 space-y-1.5"><div className="text-sm font-medium text-text-primary">Face {face.track_id}</div>
                  <p className="text-[11px] text-text-muted">First seen {face.first_seen_seconds}s · {Math.round(face.presence * 100)}% of clip</p>
                  <select aria-label={`Character for face ${face.track_id}`} disabled={busy} className={`${inputClass} !px-2 !py-1.5 !text-xs`} value={mapping?.skip ? '__skip' : mapping?.character_id || ''}
                    onChange={e => setAssignments(items => items.map(item => item.track_id === face.track_id ? { track_id: face.track_id, skip: e.target.value === '__skip', character_id: e.target.value && e.target.value !== '__skip' ? e.target.value : null } : item))}>
                    <option value="">Keep original identity</option><option value="__skip">Skip this face</option>{characters.map(c => <option key={c.id} value={c.id}>{characterDisplayName(c.name)}</option>)}
                  </select>
                  {chosen && <div className="flex items-center gap-1.5 text-[10px] text-text-muted"><img alt="" src={chosen.visual.thumbnail_url || chosen.visual.url} className="h-6 w-6 rounded object-cover" />Saved character reference</div>}
                </div>
              </div>
            })}
          </div>
          {analysis?.warnings.map(warning => <p key={warning} className="text-xs text-indicator-warning">{warning}</p>)}
        </div>
        <p className="text-xs leading-relaxed text-text-muted">Tracks up to five identities, including faces that leave and return. Only face regions are blended back. The source stays in your gallery, and the copy keeps its resolution, frame rate and soundtrack. Detection and H3 models download on first use.</p>
        {resultUrl && <video aria-label="Refined video" src={resultUrl} controls playsInline className="w-full rounded-xl bg-black" />}
        {error && <p role="alert" className="rounded-lg bg-red-500/10 p-3 text-sm text-red-400">{error}</p>}
      </div>
      <div className="shrink-0 border-t border-border px-4 py-3 sm:px-6 space-y-2">
        {status && <p role="status" className="text-xs text-text-secondary break-words">{status}</p>}
        <div className="flex items-center justify-between gap-3"><span className="text-[11px] text-text-muted">{busy ? 'You can close this dialog; the job stays in the queue.' : 'Saves a new copy'}</span>
          {busy ? <button onClick={() => void cancel()} className="shrink-0 rounded-lg border border-border px-4 py-2 text-sm text-text-primary"><Loader2 className="mr-2 inline animate-spin" size={14} />Cancel</button>
            : <button disabled={!source || uploading || (analysis !== null && (!assignments.length || assignments.every(a => a.skip)))} onClick={() => void run(false)} className="shrink-0 inline-flex items-center gap-2 rounded-lg bg-accent-blue px-4 py-2.5 text-sm font-medium text-white disabled:opacity-40"><Play size={15} />Refine faces</button>}
        </div>
      </div>
    </div>
  </div>, document.body)
}
