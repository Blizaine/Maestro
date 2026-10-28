import { useEffect, useMemo, useState } from 'react'
import { AudioLines, Check, Loader2, Mic2, SlidersHorizontal, Volume2, X } from 'lucide-react'
import type { CaptionCue } from './editorCaptions'
import { useEditorStore } from './useEditorStore'
import { useEditorDialog } from './useEditorDialog'

interface AudioCapabilities {
  transcription_available: boolean
  reason: string
}

interface AudioAnalysis {
  integrated_lufs: number
  peak_db?: number
  true_peak_db?: number
  true_peak_dbtp?: number
  gain_db: number
  gain: number
  reason?: string
}

interface TranscriptionResult {
  cues: Array<CaptionCue & { words?: CaptionCue[] }>
  offset: number
}

const inputClass = 'w-full rounded-md border border-border bg-bg-tertiary px-2 py-1.5 text-[12px] text-text-primary outline-none focus:border-accent-blue/60'

async function responseJson<T>(response: Response): Promise<T> {
  const result = await response.json().catch(() => ({})) as T & { detail?: string; error?: string }
  if (!response.ok) throw new Error(result.detail || result.error || `Request failed (${response.status}).`)
  return result
}

export function EditorAudioTools({ open, onClose }: { open: boolean; onClose: () => void }) {
  const dialogRef = useEditorDialog(open, onClose)
  const project = useEditorStore(state => state.project)
  const selectedItemId = useEditorStore(state => state.selectedItemId)
  const executeEditOperations = useEditorStore(state => state.executeEditOperations)
  const [capabilities, setCapabilities] = useState<AudioCapabilities | null>(null)
  const [capabilityLoading, setCapabilityLoading] = useState(false)
  const [narrationTrackId, setNarrationTrackId] = useState('')
  const [duckDb, setDuckDb] = useState(12)
  const [attack, setAttack] = useState(0.2)
  const [release, setRelease] = useState(0.45)
  const [busy, setBusy] = useState<'duck' | 'analyze' | 'transcribe' | null>(null)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const [analysis, setAnalysis] = useState<AudioAnalysis | null>(null)
  const [analysisTarget, setAnalysisTarget] = useState('')
  const [transcript, setTranscript] = useState<TranscriptionResult | null>(null)
  const [transcriptTarget, setTranscriptTarget] = useState('')
  const [transcriptGrouping, setTranscriptGrouping] = useState<'phrase' | 'word'>('phrase')

  const selected = useMemo(() => {
    if (!project || !selectedItemId) return null
    for (const track of project.tracks) {
      const item = track.items.find(candidate => candidate.id === selectedItemId)
      if (item) return { item, track }
    }
    return null
  }, [project, selectedItemId])
  const selectedAsset = selected?.item.asset_id ? project?.assets[selected.item.asset_id] : undefined
  const selectedHasAudio = Boolean(selected && (selected.track.type === 'audio' || (selectedAsset?.type === 'video' && selectedAsset.has_audio)))
  const selectedKey = selected && project ? JSON.stringify([project.id, selected.item.id, selected.item.asset_id, selectedAsset?.path, selected.item.start, selected.item.source_in, selected.item.duration, selected.item.speed]) : ''
  const narrationTracks = useMemo(() => project?.tracks.filter(track => (
    track.id !== selected?.track.id
    && (track.type === 'audio' || (track.type === 'video' && track.items.some(item => project.assets[item.asset_id || '']?.has_audio)))
  )) || [], [project, selected?.track.id])

  useEffect(() => {
    if (!open) return
    const controller = new AbortController()
    setCapabilities(null)
    setCapabilityLoading(true)
    void fetch('/api/v1/editor/audio/capabilities', { signal: controller.signal })
      .then(response => responseJson<AudioCapabilities>(response))
      .then(result => setCapabilities({ transcription_available: Boolean(result.transcription_available), reason: result.reason || '' }))
      .catch(reason => {
        if (!controller.signal.aborted) setCapabilities({ transcription_available: false, reason: reason instanceof Error ? reason.message : 'Audio capabilities are unavailable.' })
      })
      .finally(() => { if (!controller.signal.aborted) setCapabilityLoading(false) })
    return () => controller.abort()
  }, [open])

  useEffect(() => {
    if (narrationTracks.some(track => track.id === narrationTrackId)) return
    setNarrationTrackId(narrationTracks[0]?.id || '')
  }, [narrationTrackId, narrationTracks])

  useEffect(() => {
    if (!open) return
    const onKeyDown = (event: KeyboardEvent) => { if (event.key === 'Escape') onClose() }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [onClose, open])

  if (!open || !project) return null

  const requestBase = {
    workspace: project.workspace,
    project_id: project.id,
    ...(project.revision !== undefined ? { revision: project.revision } : {}),
  }

  const applyDuck = () => {
    if (!selected || !narrationTrackId) return
    setError(''); setNotice('')
    setBusy('duck')
    const succeeded = executeEditOperations([{
      type: 'duck-audio',
      itemId: selected.item.id,
      narrationTrackId,
      duckDb,
      attack,
      release,
    }])
    setBusy(null)
    if (succeeded) setNotice(`Added a ${duckDb} dB duck envelope to ${selected.item.name}. Adjust it in the Motion controls.`)
    else setError(useEditorStore.getState().error || 'The duck envelope could not be added.')
  }

  const analyze = async () => {
    if (!selected || !selectedHasAudio) return
    setBusy('analyze'); setError(''); setNotice(''); setAnalysis(null)
    try {
      const saved = await useEditorStore.getState().saveProject()
      if (!saved) throw new Error(useEditorStore.getState().error || 'Save this project before analyzing its audio.')
      const result = await responseJson<AudioAnalysis>(await fetch('/api/v1/editor/audio/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ...requestBase, revision: saved.revision, item_id: selected.item.id }),
      }))
      const measuredPeak = result.true_peak_db ?? result.true_peak_dbtp ?? result.peak_db
      if (![result.integrated_lufs, measuredPeak, result.gain_db, result.gain].every(Number.isFinite)) throw new Error('The analysis response did not include valid loudness measurements.')
      setAnalysis(result)
      setAnalysisTarget(selectedKey)
      setNotice('Measured the selected clip’s source range. This is source analysis, not a live playback meter.')
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'Audio analysis failed.')
    } finally { setBusy(null) }
  }

  const applyNormalization = () => {
    if (!selected || !analysis || analysisTarget !== selectedKey) return
    const proposedGain = analysis.gain
    if (!Number.isFinite(proposedGain) || proposedGain < 0) {
      setError('The analysis did not return a usable gain recommendation.')
      return
    }
    const newVolume = Math.min(4, Math.max(0, proposedGain))
    const envelopeGain = selected.item.volume > 1e-8 ? newVolume / selected.item.volume : newVolume
    setError(''); setNotice('')
    const succeeded = executeEditOperations([{
      type: 'set-volume',
      itemId: selected.item.id,
      volume: newVolume,
      envelopeGain,
    }])
    if (succeeded) setNotice(`Applied the recommended clip gain (${analysis.gain_db >= 0 ? '+' : ''}${analysis.gain_db.toFixed(1)} dB) to the clip and its volume envelope.`)
    else setError(useEditorStore.getState().error || 'The gain could not be applied.')
  }

  const transcribe = async () => {
    if (!selected || !selectedHasAudio || !capabilities?.transcription_available) return
    setBusy('transcribe'); setError(''); setNotice(''); setTranscript(null)
    try {
      const saved = await useEditorStore.getState().saveProject()
      if (!saved) throw new Error(useEditorStore.getState().error || 'Save this project before transcribing its audio.')
      const result = await responseJson<TranscriptionResult>(await fetch('/api/v1/editor/audio/transcribe', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ...requestBase, revision: saved.revision, item_id: selected.item.id }),
      }))
      if (!Array.isArray(result.cues) || !Number.isFinite(result.offset)) throw new Error('The transcription response is incomplete.')
      setTranscript({ cues: result.cues, offset: result.offset })
      setTranscriptTarget(selectedKey)
      setTranscriptGrouping('phrase')
      setNotice(`Transcribed ${result.cues.length} cues. Review them before adding captions.`)
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'Transcription failed.')
    } finally { setBusy(null) }
  }

  const addTranscript = () => {
    if (!transcript?.cues.length || transcriptTarget !== selectedKey) return
    const captionCues: CaptionCue[] = [...(transcriptGrouping === 'word'
      ? transcript.cues.flatMap(cue => cue.words?.length ? cue.words : [cue])
      : transcript.cues)]
    captionCues.sort((left, right) => left.start - right.start || left.end - right.end)
    const succeeded = executeEditOperations([{
      type: 'import-captions',
      cues: captionCues,
      offset: transcript.offset,
      name: 'Captions',
    }])
    if (succeeded) setNotice(`Added ${captionCues.length} ${transcriptGrouping === 'word' ? 'word' : 'transcript'} cues to the Captions track in one undo step.`)
    else setError(useEditorStore.getState().error || 'The transcript cues could not be added.')
  }

  const narrationTrack = narrationTracks.find(track => track.id === narrationTrackId)
  const locked = Boolean(selected?.track.locked || narrationTrack?.locked)
  const analysisMatchesSelection = Boolean(analysis && analysisTarget === selectedKey)
  const transcriptMatchesSelection = Boolean(transcript && transcriptTarget === selectedKey)
  const transcriptPreview: CaptionCue[] = (transcript && transcriptGrouping === 'word'
    ? transcript.cues.flatMap(cue => cue.words?.length ? cue.words : [cue])
    : transcript?.cues || []).slice().sort((left, right) => left.start - right.start || left.end - right.end)
  const measuredPeak = analysis?.true_peak_db ?? analysis?.true_peak_dbtp ?? analysis?.peak_db

  return (
    <div ref={dialogRef} className="fixed inset-0 z-[160] flex h-[100dvh] items-center justify-center overflow-hidden bg-black/65 px-3 py-3 backdrop-blur-sm" onMouseDown={event => { if (event.target === event.currentTarget) onClose() }}>
      <section role="dialog" aria-modal="true" aria-label="Audio tools" className="flex max-h-full w-full max-w-[680px] flex-col overflow-hidden rounded-2xl border border-border bg-bg-secondary shadow-2xl">
        <header className="flex shrink-0 items-center justify-between border-b border-border px-4 py-3">
          <div className="flex items-center gap-2.5"><span className="grid h-8 w-8 place-items-center rounded-lg bg-accent-blue/10 text-accent-blue"><AudioLines size={15} /></span><div><h2 className="text-sm font-semibold text-text-primary">Audio tools</h2><p className="text-[10px] text-text-muted">Duck music, inspect source loudness, or transcribe selected audio.</p></div></div>
          <button type="button" onClick={onClose} aria-label="Close audio tools" className="rounded-lg p-2 text-text-muted hover:bg-bg-hover hover:text-text-primary"><X size={16} /></button>
        </header>
        <div className="min-h-0 flex-1 space-y-4 overflow-y-auto p-4">
          <section className="space-y-3 rounded-xl border border-border bg-bg-tertiary p-3">
            <div className="flex items-start gap-2"><Volume2 size={15} className="mt-0.5 text-accent-blue" /><div><h3 className="text-[12px] font-semibold text-text-primary">Music ducking</h3><p className="mt-0.5 text-[10px] leading-relaxed text-text-muted">Adds volume keyframes to lower the selected music clip while narration overlaps.</p></div></div>
            <div className="rounded-lg border border-border bg-bg-secondary px-3 py-2 text-[11px] text-text-secondary">Selected clip: <strong className="font-medium text-text-primary">{selected?.item.name || 'Choose a clip in the timeline'}</strong>{selected && <span className="ml-1 text-text-muted">· {selected.track.name}{selected.track.locked ? ' · locked' : ''}</span>}</div>
            <div className="grid gap-2 sm:grid-cols-2">
              <label className="space-y-1 text-[10px] text-text-muted">Narration track<select aria-label="Narration track" value={narrationTrackId} onChange={event => setNarrationTrackId(event.target.value)} className={inputClass}><option value="">Choose a track</option>{narrationTracks.map(track => <option key={track.id} value={track.id}>{track.name}{track.locked ? ' · locked' : ''}</option>)}</select></label>
              <label className="space-y-1 text-[10px] text-text-muted">Duck depth (dB)<input aria-label="Duck depth in dB" type="number" min="0" max="36" step="1" value={duckDb} onChange={event => setDuckDb(Number(event.target.value))} className={inputClass} /></label>
              <label className="space-y-1 text-[10px] text-text-muted">Attack (seconds)<input aria-label="Duck attack seconds" type="number" min="0" max="10" step="0.05" value={attack} onChange={event => setAttack(Number(event.target.value))} className={inputClass} /></label>
              <label className="space-y-1 text-[10px] text-text-muted">Release (seconds)<input aria-label="Duck release seconds" type="number" min="0" max="10" step="0.05" value={release} onChange={event => setRelease(Number(event.target.value))} className={inputClass} /></label>
            </div>
            <button type="button" onClick={applyDuck} disabled={!selectedHasAudio || !narrationTrackId || locked || busy !== null || !Number.isFinite(duckDb + attack + release)} className="inline-flex w-full items-center justify-center gap-1.5 rounded-lg bg-cta px-3 py-2 text-[11px] font-semibold text-white disabled:cursor-not-allowed disabled:opacity-40">{busy === 'duck' ? <Loader2 size={13} className="animate-spin" /> : <SlidersHorizontal size={13} />} Apply duck envelope</button>
            {!selectedHasAudio && <p className="text-[10px] text-text-muted">Select an audio clip or a video clip with audio. The resulting envelope is editable in Motion controls.</p>}
          </section>

          <section className="space-y-3 rounded-xl border border-border bg-bg-tertiary p-3">
            <div className="flex items-start gap-2"><AudioLines size={15} className="mt-0.5 text-accent-blue" /><div><h3 className="text-[12px] font-semibold text-text-primary">Source loudness</h3><p className="mt-0.5 text-[10px] leading-relaxed text-text-muted">Analyze the selected clip’s source range and optionally apply the peak-protected gain recommendation.</p></div></div>
            <button type="button" onClick={() => void analyze()} disabled={!selectedHasAudio || Boolean(selected?.track.locked) || busy !== null} className="inline-flex items-center gap-1.5 rounded-lg border border-border bg-bg-secondary px-3 py-2 text-[11px] text-text-secondary hover:bg-bg-hover disabled:opacity-40">{busy === 'analyze' ? <Loader2 size={13} className="animate-spin" /> : <Volume2 size={13} />} Analyze selected source range</button>
            {analysisMatchesSelection && analysis && <div className="rounded-lg border border-border bg-bg-secondary p-3"><div className="grid grid-cols-3 gap-2 text-center"><div><div className="font-mono text-[13px] text-text-primary">{analysis.integrated_lufs.toFixed(1)} LUFS</div><div className="text-[9px] text-text-muted">Integrated source</div></div><div><div className="font-mono text-[13px] text-text-primary">{(measuredPeak ?? 0).toFixed(1)} {analysis.true_peak_db !== undefined || analysis.true_peak_dbtp !== undefined ? 'dBTP' : 'dBFS'}</div><div className="text-[9px] text-text-muted">{analysis.true_peak_db !== undefined || analysis.true_peak_dbtp !== undefined ? 'Source true peak' : 'Source peak'}</div></div><div><div className="font-mono text-[13px] text-accent-blue">{analysis.gain_db >= 0 ? '+' : ''}{analysis.gain_db.toFixed(1)} dB</div><div className="text-[9px] text-text-muted">Peak-protected gain</div></div></div>{analysis.reason && <p className="mt-2 text-[10px] text-text-muted">{analysis.reason}</p>}<button type="button" onClick={applyNormalization} disabled={Boolean(selected?.track.locked)} className="mt-3 inline-flex w-full items-center justify-center gap-1.5 rounded-lg bg-accent-blue px-3 py-2 text-[11px] font-semibold text-white disabled:opacity-40"><Check size={13} /> Apply clip gain and envelope gain</button></div>}
          </section>

          <section className="space-y-3 rounded-xl border border-border bg-bg-tertiary p-3">
            <div className="flex items-start gap-2"><Mic2 size={15} className="mt-0.5 text-accent-blue" /><div><h3 className="text-[12px] font-semibold text-text-primary">Transcription</h3><p className="mt-0.5 text-[10px] leading-relaxed text-text-muted">Transcription runs only after you press the button. Cues are shown for review before import.</p></div></div>
            {capabilityLoading ? <p className="inline-flex items-center gap-1.5 text-[10px] text-text-muted"><Loader2 size={11} className="animate-spin" /> Checking installed transcription support…</p> : !capabilities?.transcription_available ? <p className="text-[10px] text-text-muted">{capabilities?.reason || 'Transcription is not available on this installation.'}</p> : <button type="button" onClick={() => void transcribe()} disabled={!selectedHasAudio || Boolean(selected?.track.locked) || busy !== null} className="inline-flex items-center gap-1.5 rounded-lg border border-border bg-bg-secondary px-3 py-2 text-[11px] text-text-secondary hover:bg-bg-hover disabled:opacity-40">{busy === 'transcribe' ? <Loader2 size={13} className="animate-spin" /> : <Mic2 size={13} />} Transcribe selected clip</button>}
            {transcriptMatchesSelection && transcript && <div className="space-y-2 rounded-lg border border-border bg-bg-secondary p-3"><div className="flex flex-wrap items-center justify-between gap-2 text-[10px] text-text-secondary"><span>Transcript preview · {transcriptPreview.length} {transcriptGrouping === 'word' ? 'word captions' : 'phrase cues'}</span><span>Offset {transcript.offset.toFixed(3)} s</span></div>{transcript.cues.some(cue => cue.words?.length) && <label className="flex items-center gap-2 text-[10px] text-text-muted">Caption grouping<select aria-label="Transcript caption grouping" value={transcriptGrouping} onChange={event => setTranscriptGrouping(event.target.value as 'phrase' | 'word')} className="rounded-md border border-border bg-bg-tertiary px-2 py-1.5 text-[11px] text-text-primary"><option value="phrase">Phrase cues</option><option value="word">Word-by-word</option></select></label>}<ol className="max-h-40 space-y-1 overflow-y-auto">{transcriptPreview.map((cue, index) => <li key={`${cue.start}-${index}`} className="flex gap-2 text-[10px]"><span className="shrink-0 font-mono text-text-muted">{cue.start.toFixed(2)}–{cue.end.toFixed(2)}s</span><span className="whitespace-pre-wrap text-text-primary">{cue.text}</span></li>)}</ol><button type="button" onClick={addTranscript} className="w-full rounded-lg bg-accent-blue px-3 py-2 text-[11px] font-semibold text-white">Add reviewed transcript as captions</button></div>}
          </section>
          <p className="text-[9px] leading-relaxed text-text-muted">The source analysis displays measured LUFS and peak values. Its recommendation is not a live meter.</p>
          {error && <p role="alert" className="rounded-lg border border-red-500/25 bg-red-500/10 px-3 py-2 text-[11px] text-red-200">{error}</p>}
          {notice && <p role="status" className="rounded-lg border border-accent-blue/20 bg-accent-blue/5 px-3 py-2 text-[11px] text-accent-blue">{notice}</p>}
        </div>
      </section>
    </div>
  )
}
