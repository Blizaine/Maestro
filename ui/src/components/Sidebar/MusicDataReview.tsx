import { draftMusicLyrics, musicRecordingUrl, reviewMusicTrack } from '../../api/musicTraining'
import type { MusicProject, MusicTrack } from '../../api/musicTraining'

const field = 'w-full rounded-lg border border-border bg-bg-tertiary p-2 text-sm text-text-primary'
const button = 'rounded-lg border border-border px-3 py-2 text-xs hover:bg-bg-tertiary disabled:opacity-40'
const captionHint = 'Describe what you hear: genre and pace; vocal tone and delivery; instruments and rhythm; mood and production. Use a distinct caption for each recording.'

export function MusicTrackFields({track, onChange}: {track: MusicTrack; onChange: (changes: Partial<MusicTrack>) => void}) {
  const filename = track.audio_path.replace(/\\/g, '/').split('/').pop() || ''
  const edit = (changes: Partial<MusicTrack>) => onChange({...changes, reviewed: false})
  return <>
    <audio controls preload="metadata" src={track.preview_url || `/api/v1/uploads/audio/${encodeURIComponent(filename)}`} className="w-full" aria-label={`Recording ${track.name}`}/>
    <label className="block text-xs">Original song<input className={`${field} mt-1`} value={track.source_song || ''} maxLength={200} onChange={event => edit({source_song: event.target.value})} placeholder="Use the same name for excerpts of one song"/></label>
    <label className="block text-xs">Style caption<textarea aria-label="Style caption" rows={3} className={`${field} mt-1`} value={track.style} maxLength={2000} onChange={event => edit({style: event.target.value})} placeholder="e.g. Dry, rapid male rap; sparse piano loop, punchy kick and snare; tense, close vocal mix"/></label>
    <p className="text-[11px] text-text-muted">{captionHint}</p>
    <label className="block text-xs">Full lyrics and sections<textarea aria-label="Full lyrics and sections" rows={5} className={`${field} mt-1`} value={track.lyrics} maxLength={40000} onChange={event => edit({lyrics: event.target.value})} placeholder="[Verse]… [Chorus]… or [Instrumental]"/></label>
    <p className="text-[11px] text-text-muted">Match this recording or excerpt: include repetitions, spoken intros and omitted verses correctly. Do not paste the full song's lyrics for a short excerpt.</p>
    <label className="flex items-center gap-2 text-xs"><input type="checkbox" checked={track.holdout} onChange={event => edit({holdout: event.target.checked})}/>Check only — do not train on this song (held out)</label>
    <p className="text-[11px] text-text-muted">Use a different song for this check. It measures whether training helps on unseen audio; it does not teach the voice or supply a reference for generation.</p>
    <label className="flex items-center gap-2 text-xs"><input type="checkbox" checked={!!track.reviewed} disabled={!track.style.trim() || !track.lyrics.trim()} onChange={event => onChange({reviewed: event.target.checked})}/>I listened and checked this caption and these lyrics</label>
  </>
}

export function MusicDataReview({project, busy, run, onEdit}: {
  project: MusicProject; busy: boolean; run: (action: () => Promise<void>) => Promise<void>; onEdit: (tracks: MusicTrack[]) => void
}) {
  const reviewed = project.reviewed_track_ids || []
  const active = ['queued', 'training', 'preparing', 'auditioning'].includes(project.status)
  return <details className="space-y-3 rounded-lg border border-border p-3">
    <summary className="cursor-pointer text-sm">Recordings, captions & lyrics · {reviewed.length}/{project.tracks.length} reviewed</summary>
    <p className="text-xs text-text-muted">{captionHint} Corrections create a new project; existing checkpoints and recordings are preserved.</p>
    <div className="flex flex-wrap gap-2">
      <button type="button" className={button} disabled={busy || active} onClick={() => onEdit(project.tracks.map(track => ({...track, reviewed: reviewed.includes(track.id!)})))}>Edit captions and lyrics in a new project</button>
      <button type="button" className={button} disabled={busy || active} onClick={() => void run(async () => {await draftMusicLyrics(project.id)})}>Draft lyrics from recordings</button>
    </div>
    <p className="text-[11px] text-text-muted">Whisper listens locally and queues behind GPU jobs. Its drafts need review, especially for singing. It does not write instrument or voice descriptions.</p>
    {project.tracks.map(track => {
      const draft = project.review_drafts?.[track.id!]
      const coverage = project.sequence_coverage?.find(row => row.track_id === track.id)
      return <div key={track.id} className="space-y-2 border-t border-border pt-3">
        <p className="break-words text-xs">{track.name} · {track.holdout ? 'held out' : 'training'}{track.source_song ? ` · ${track.source_song}` : ''}</p>
        <audio controls preload="metadata" src={musicRecordingUrl(project.id, track.id!)} aria-label={`Recording ${track.name}`} className="w-full"/>
        <p className="whitespace-pre-wrap text-xs">{track.style}</p>
        <textarea readOnly rows={4} aria-label={`Saved lyrics for ${track.name}`} value={track.lyrics} className={field}/>
        {coverage && <p className={`text-xs ${coverage.truncated ? 'text-amber-400' : 'text-text-muted'}`}>{coverage.training_seconds.toFixed(1)}s of {coverage.source_seconds.toFixed(1)}s fit the training sequence{coverage.truncated ? '. The ending will be omitted; use shorter excerpts with matching lyrics.' : '.'}</p>}
        <label className="flex items-center gap-2 text-xs"><input type="checkbox" checked={reviewed.includes(track.id!)} disabled={busy || active} onChange={event => void run(async () => {await reviewMusicTrack(project.id, track.id!, event.target.checked)})}/>I listened and checked this caption and these lyrics</label>
        {draft && <details className="rounded border border-border p-2 text-xs">
          <summary className="cursor-pointer">Local lyric draft · {draft.language} · needs review</summary>
          <p className="my-2 text-text-muted">{draft.note}</p>
          <div className="max-h-60 space-y-1 overflow-y-auto">{draft.segments.map((segment, index) => <p key={index} className={segment.uncertain ? 'text-amber-400' : ''}>{segment.start.toFixed(1)}–{segment.end.toFixed(1)}s: {segment.text}{segment.uncertain ? ' (uncertain)' : ''}</p>)}</div>
          <button type="button" className={`${button} mt-2`} disabled={busy || active || !draft.text.trim()} onClick={() => onEdit(project.tracks.map(item => ({...item, reviewed: false, lyrics: item.id === track.id ? draft.text : item.lyrics})))}>Review draft in an edited copy</button>
        </details>}
      </div>
    })}
  </details>
}
