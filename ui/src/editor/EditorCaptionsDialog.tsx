import { useEffect, useId, useMemo, useState } from 'react'
import { Download, FileText, Plus, Trash2, Upload, X } from 'lucide-react'
import { useEditorStore } from './useEditorStore'
import { useEditorDialog } from './useEditorDialog'
import { captionTracks, collectCaptionTrackCues, exportSrt, exportWebVtt, parseCaptionFile, type CaptionCue } from './editorCaptions'

const fieldClass = 'w-full rounded-md border border-border bg-bg-tertiary px-2 py-1.5 text-[12px] text-text-primary outline-none focus:border-accent-blue/60'

function downloadText(filename: string, mimeType: string, contents: string): void {
  const url = URL.createObjectURL(new Blob([contents], { type: `${mimeType};charset=utf-8` }))
  const anchor = document.createElement('a')
  anchor.href = url
  anchor.download = filename
  anchor.click()
  URL.revokeObjectURL(url)
}

export function EditorCaptionsDialog({ open, onClose }: { open: boolean; onClose: () => void }) {
  const projectId = useEditorStore(state => state.project?.id)
  if (!projectId) return null
  return <EditorCaptionsDialogContent key={projectId} open={open} onClose={onClose} />
}

function EditorCaptionsDialogContent({ open, onClose }: { open: boolean; onClose: () => void }) {
  const dialogRef = useEditorDialog(open, onClose)
  const id = useId()
  const project = useEditorStore(state => state.project)
  const executeEditOperations = useEditorStore(state => state.executeEditOperations)
  const [cues, setCues] = useState<CaptionCue[]>([])
  const [offset, setOffset] = useState(0)
  const [trackName, setTrackName] = useState('Captions')
  const [exportSource, setExportSource] = useState('draft')
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  useEffect(() => {
    if (!open) return
    const onKeyDown = (event: KeyboardEvent) => { if (event.key === 'Escape') onClose() }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [onClose, open])
  const tracks = useMemo(() => project ? captionTracks(project) : [], [project])
  const selectedExportSource = exportSource === 'draft' || tracks.some(track => track.id === exportSource)
    ? exportSource
    : 'draft'
  const exportCues = useMemo(() => {
    if (!project) return []
    return selectedExportSource === 'draft'
      ? cues.map(cue => ({ ...cue, start: cue.start + offset, end: cue.end + offset }))
      : collectCaptionTrackCues(project, selectedExportSource)
  }, [cues, offset, project, selectedExportSource])

  if (!open || !project) return null

  const readFile = async (file?: File) => {
    if (!file) return
    try {
      const parsed = parseCaptionFile(await file.text(), file.name)
      setCues(parsed)
      setExportSource('draft')
      setError('')
      setNotice(`Loaded ${parsed.length} cues from ${file.name}.`)
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'This subtitle file could not be read.')
      setNotice('')
    }
  }

  const updateCue = (index: number, patch: Partial<CaptionCue>) => {
    setCues(previous => previous.map((cue, cueIndex) => cueIndex === index ? { ...cue, ...patch } : cue))
    setError('')
  }

  const importCues = () => {
    setError('')
    setNotice('')
    const succeeded = executeEditOperations([{ type: 'import-captions', cues, offset, name: trackName }])
    if (succeeded) setNotice(`Added ${cues.length} captions to “${trackName.trim() || 'Captions'}” in one undo step.`)
    else setError(useEditorStore.getState().error || 'The caption cues could not be added.')
  }

  const download = (format: 'srt' | 'vtt') => {
    try {
      const content = format === 'srt' ? exportSrt(exportCues) : exportWebVtt(exportCues)
      downloadText(`${project.name.replace(/[\\/:*?"<>|]+/g, ' ').trim() || 'captions'}.${format}`, format === 'srt' ? 'application/x-subrip' : 'text/vtt', content)
      setError('')
      setNotice(`Downloaded ${selectedExportSource === 'draft' ? 'draft cues' : 'timeline captions'} as ${format.toUpperCase()}.`)
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'No valid cues are available to export.')
      setNotice('')
    }
  }

  return (
    <div ref={dialogRef} className="fixed inset-0 z-[160] flex h-[100dvh] items-center justify-center overflow-hidden bg-black/65 px-3 py-3 backdrop-blur-sm" onMouseDown={event => { if (event.target === event.currentTarget) onClose() }}>
      <section role="dialog" aria-modal="true" aria-label="Captions" className="flex max-h-full w-full max-w-[760px] flex-col overflow-hidden rounded-2xl border border-border bg-bg-secondary shadow-2xl">
        <header className="flex shrink-0 items-center justify-between border-b border-border px-4 py-3">
          <div className="flex items-center gap-2.5">
            <span className="grid h-8 w-8 place-items-center rounded-lg bg-accent-blue/10 text-accent-blue"><FileText size={15} /></span>
            <div><h2 className="text-sm font-semibold text-text-primary">Captions</h2><p className="text-[10px] text-text-muted">Import, edit, and export SRT or WebVTT cues.</p></div>
          </div>
          <button type="button" onClick={onClose} aria-label="Close captions" className="rounded-lg p-2 text-text-muted hover:bg-bg-hover hover:text-text-primary"><X size={16} /></button>
        </header>

        <div className="min-h-0 flex-1 space-y-4 overflow-y-auto p-4">
          <div className="flex flex-wrap items-end gap-2 rounded-xl border border-border bg-bg-tertiary p-3">
            <label htmlFor={`${id}-file`} className="inline-flex cursor-pointer items-center gap-1.5 rounded-lg border border-border bg-bg-secondary px-3 py-2 text-[11px] font-medium text-text-secondary hover:bg-bg-hover"><Upload size={13} /> Import SRT / VTT</label>
            <input id={`${id}-file`} type="file" accept=".srt,.vtt,application/x-subrip,text/vtt" className="sr-only" onChange={event => { void readFile(event.target.files?.[0]); event.currentTarget.value = '' }} />
            <label className="min-w-[130px] flex-1 space-y-1 text-[10px] text-text-muted">Track name<input aria-label="Caption track name" value={trackName} onChange={event => setTrackName(event.target.value)} className={fieldClass} /></label>
            <label className="w-[130px] space-y-1 text-[10px] text-text-muted">Import offset (seconds)<input aria-label="Caption offset" type="number" step="0.001" value={offset} onChange={event => setOffset(Number(event.target.value))} className={fieldClass} /></label>
            <button type="button" onClick={() => { setCues(previous => [...previous, { start: previous.length ? previous[previous.length - 1].end : 0, end: (previous.length ? previous[previous.length - 1].end : 0) + 2, text: '' }]); setExportSource('draft') }} className="inline-flex items-center gap-1 rounded-lg px-3 py-2 text-[11px] text-accent-blue hover:bg-accent-blue/10"><Plus size={13} /> Add cue</button>
          </div>

          <div className="rounded-xl border border-border bg-bg-tertiary p-3">
            <div className="mb-2 flex items-center justify-between gap-3"><div><h3 className="text-[12px] font-semibold text-text-primary">Draft cues <span className="font-normal text-text-muted">({cues.length})</span></h3><p className="text-[10px] text-text-muted">The default title style is a centered, high-contrast lower third.</p></div><button type="button" onClick={importCues} disabled={!cues.length || !trackName.trim()} className="shrink-0 rounded-lg bg-cta px-3 py-2 text-[11px] font-semibold text-white disabled:cursor-not-allowed disabled:opacity-40">Add to timeline</button></div>
            {!cues.length ? <p className="rounded-lg border border-dashed border-border px-3 py-8 text-center text-[11px] text-text-muted">Choose a subtitle file or add a cue to start.</p> : (
              <ol className="max-h-[min(45vh,420px)] space-y-2 overflow-y-auto">
                {cues.map((cue, index) => (
                  <li key={index} className="grid gap-2 rounded-lg border border-border bg-bg-secondary p-2 sm:grid-cols-[100px_100px_1fr_28px] sm:items-start">
                    <label className="space-y-1 text-[9px] text-text-muted">Start (s)<input aria-label={`Cue ${index + 1} start`} type="number" min="0" step="0.001" value={cue.start} onChange={event => updateCue(index, { start: Number(event.target.value) })} className={fieldClass} /></label>
                    <label className="space-y-1 text-[9px] text-text-muted">End (s)<input aria-label={`Cue ${index + 1} end`} type="number" min="0" step="0.001" value={cue.end} onChange={event => updateCue(index, { end: Number(event.target.value) })} className={fieldClass} /></label>
                    <label className="space-y-1 text-[9px] text-text-muted">Text<textarea aria-label={`Cue ${index + 1} text`} rows={2} value={cue.text} onChange={event => updateCue(index, { text: event.target.value })} className={`${fieldClass} resize-y`} /></label>
                    <button type="button" aria-label={`Delete cue ${index + 1}`} onClick={() => setCues(previous => previous.filter((_, cueIndex) => cueIndex !== index))} className="mt-4 grid h-7 w-7 place-items-center rounded-md text-text-muted hover:bg-red-500/10 hover:text-red-300"><Trash2 size={13} /></button>
                  </li>
                ))}
              </ol>
            )}
          </div>

          <div className="rounded-xl border border-border bg-bg-tertiary p-3">
            <div className="flex flex-wrap items-end justify-between gap-3">
              <div className="min-w-[180px] flex-1 space-y-1"><label htmlFor={`${id}-export-source`} className="text-[10px] font-medium text-text-secondary">Export source</label><select id={`${id}-export-source`} value={selectedExportSource} onChange={event => setExportSource(event.target.value)} className={fieldClass}><option value="draft">Draft cues (includes import offset)</option>{tracks.map(track => <option key={track.id} value={track.id}>{track.name} · {track.count} timeline cues</option>)}</select></div>
              <div className="flex gap-2"><button type="button" onClick={() => download('srt')} disabled={!exportCues.length} className="inline-flex items-center gap-1.5 rounded-lg border border-border bg-bg-secondary px-3 py-2 text-[11px] text-text-secondary hover:bg-bg-hover disabled:opacity-40"><Download size={12} /> SRT</button><button type="button" onClick={() => download('vtt')} disabled={!exportCues.length} className="inline-flex items-center gap-1.5 rounded-lg border border-border bg-bg-secondary px-3 py-2 text-[11px] text-text-secondary hover:bg-bg-hover disabled:opacity-40"><Download size={12} /> WebVTT</button></div>
            </div>
            <p className="mt-2 text-[10px] text-text-muted">Timeline export includes only imported caption clips, not ordinary title text. {exportCues.length} cues selected.</p>
          </div>
          {error && <p role="alert" className="rounded-lg border border-red-500/25 bg-red-500/10 px-3 py-2 text-[11px] text-red-200">{error}</p>}
          {notice && <p role="status" className="rounded-lg border border-accent-blue/20 bg-accent-blue/5 px-3 py-2 text-[11px] text-accent-blue">{notice}</p>}
        </div>
      </section>
    </div>
  )
}
