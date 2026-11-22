import { useEffect, useState } from 'react'
import { cancelJob, fetchCharacters, fetchJobStatus, getFileUrl, submitToolRevoice } from '../../api/client'
import { characterDisplayName } from '../../lib/characters'
import { useStore } from '../../stores/useStore'
import type { SavedOmniCharacter } from '../../types'

export function DirectorSingingControls() {
  const sourcePath = useStore(s => s.directorAudioPath)
  const workspace = useStore(s => s.activeWorkspace)
  const uploadAndAnalyze = useStore(s => s.directorUploadAndAnalyze)
  const [open, setOpen] = useState(false)
  const [characters, setCharacters] = useState<SavedOmniCharacter[]>([])
  const [first, setFirst] = useState('')
  const [second, setSecond] = useState('')
  const [submission, setSubmission] = useState<{ id: string; workspace: string; source: string } | null>(null)
  const [output, setOutput] = useState<{ name: string; workspace: string; source: string } | null>(null)
  const [status, setStatus] = useState('')
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)

  useEffect(() => {
    if (!open) return
    let active = true
    fetchCharacters().then(items => {
      if (active) setCharacters(items.filter(character => character.voice?.path))
    }).catch(cause => {
      if (active) setError(cause instanceof Error ? cause.message : String(cause))
    })
    return () => { active = false }
  }, [open])

  useEffect(() => {
    if (!submission) return
    let active = true
    let timer: ReturnType<typeof setTimeout>
    const poll = async () => {
      try {
        const job = await fetchJobStatus(submission.id)
        if (!active) return
        if (job.status === 'completed') {
          const name = job.output_files?.[0]
          if (!name) throw new Error('The singing conversion returned no soundtrack')
          setOutput({ name, workspace: submission.workspace, source: submission.source })
          setStatus('')
          setSubmission(null)
        } else if (job.status === 'failed' || job.status === 'cancelled') {
          throw new Error(job.error || job.message || 'Singing conversion did not finish')
        } else {
          setStatus(job.message || 'Converting singing')
          timer = setTimeout(poll, 1200)
        }
      } catch (cause) {
        if (!active) return
        setError(cause instanceof Error ? cause.message : String(cause))
        setStatus('')
        setSubmission(null)
      }
    }
    void poll()
    return () => { active = false; clearTimeout(timer) }
  }, [submission])

  const convert = async () => {
    if (!sourcePath || !first || second === first) return
    const voices = [first, second].filter(Boolean).map(id => characters.find(item => item.id === id)?.voice?.path)
    if (!voices.every((voice): voice is string => !!voice)) return
    setBusy(true)
    setError('')
    setOutput(null)
    try {
      const job = await submitToolRevoice({ video_path: sourcePath, voice_ref_paths: voices, singing: true, workspace })
      setSubmission({ id: job.job_id, workspace, source: sourcePath })
      setStatus('Queued')
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause))
    } finally {
      setBusy(false)
    }
  }

  const applySoundtrack = async () => {
    if (!output || output.source !== sourcePath) return
    setBusy(true)
    setError('')
    try {
      const response = await fetch(getFileUrl(output.name, output.workspace))
      if (!response.ok) throw new Error('Could not open the converted soundtrack')
      if (useStore.getState().directorAudioPath !== output.source) return
      const blob = await response.blob()
      if (useStore.getState().directorAudioPath !== output.source) return
      await uploadAndAnalyze(new File([blob], output.name, { type: 'audio/wav' }))
      setOutput(null)
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause))
    } finally {
      setBusy(false)
    }
  }

  if (!sourcePath) return null
  const working = !!submission || busy
  const control = 'mt-1 w-full rounded-md border border-border bg-bg-tertiary p-2 text-xs text-text-primary'
  const availableOutput = output?.source === sourcePath ? output : null
  return (
    <details className="mt-2 rounded-md border border-border px-2.5 py-2 text-xs" onToggle={e => setOpen(e.currentTarget.open)}>
      <summary className="cursor-pointer font-medium text-text-primary">Sung by</summary>
      <div className="mt-2 space-y-2">
        <div className="grid gap-2 sm:grid-cols-2">
          <label className="text-text-secondary">Singer
            <select className={control} value={first} disabled={working} onChange={e => setFirst(e.target.value)}>
              <option value="">Choose a saved voice</option>
              {characters.map(item => <option key={item.id} value={item.id} disabled={second === item.id}>{characterDisplayName(item.name)}</option>)}
            </select>
          </label>
          <label className="text-text-secondary">Second singer
            <select className={control} value={second} disabled={working} onChange={e => setSecond(e.target.value)}>
              <option value="">None</option>
              {characters.map(item => <option key={item.id} value={item.id} disabled={first === item.id}>{characterDisplayName(item.name)}</option>)}
            </select>
          </label>
        </div>
        {second && <p className="text-text-muted">Select duet voices in the order they first sing.</p>}
        {open && !characters.length && <p className="text-text-muted">Add a voice sample to a saved character to use it here.</p>}
        <div className="flex items-center gap-2">
          <button type="button" disabled={!first || working} onClick={() => void convert()}
            className="rounded-md border border-border px-2 py-1.5 text-text-primary disabled:opacity-40">Convert singing</button>
          {submission && <button type="button" onClick={() => void cancelJob(submission.id).catch(cause => setError(String(cause)))}
            className="rounded-md px-2 py-1.5 text-text-secondary">Cancel</button>}
        </div>
        {status && <div role="status" className="text-text-secondary">{status}</div>}
        {availableOutput && <div className="space-y-2 rounded-md border border-border p-2">
          <audio className="w-full" controls src={getFileUrl(availableOutput.name, availableOutput.workspace)} aria-label="Preview converted song" />
          <button type="button" disabled={working} onClick={() => void applySoundtrack()}
            className="rounded-md bg-bg-hover px-2 py-1.5 text-text-primary disabled:opacity-40">Use this soundtrack</button>
        </div>}
        {error && <p role="alert" className="text-red-400">{error}</p>}
      </div>
    </details>
  )
}
