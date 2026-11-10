import { useId, useState } from 'react'
import { AudioLines, ChevronDown } from 'lucide-react'
import { useStore, hasStudioSourceAudioTranscriptDriver } from '../../stores/useStore'

/** Optional transcript context for an exact source soundtrack in Studio Video. */
export function SourceAudioTranscriptControls() {
  const panelId = useId()
  const [expanded, setExpanded] = useState(false)
  const hasDriver = useStore(s => hasStudioSourceAudioTranscriptDriver({
    generationMode: s.generationMode,
    studioVideoWorkflow: s.studioVideoWorkflow,
    params: s.params,
  }))
  const includeInPrompt = useStore(s => s.params.source_audio_transcript_in_prompt === true)
  const useForEnhancement = useStore(s => s.params.source_audio_transcript_for_enhancement === true)
  const isEnhancing = useStore(s => s.isEnhancing)
  const enabled = includeInPrompt || useForEnhancement
  const setParam = useStore(s => s.setParam)

  if (!hasDriver) return null

  return (
    <section aria-label="Source audio transcript context" className="rounded-xl border border-border bg-bg-secondary/40">
      <button type="button" aria-expanded={expanded} aria-controls={panelId}
        onClick={() => setExpanded(value => !value)}
        className="flex min-h-9 w-full items-center gap-2 px-2.5 text-left text-[11px] text-text-secondary hover:text-text-primary">
        <AudioLines size={14} className="shrink-0 text-accent-blue" />
        <span className="min-w-0 flex-1 font-medium">Audio context</span>
        {enabled && <span className="rounded-full bg-accent-blue/15 px-1.5 py-0.5 text-[9px] font-medium text-accent-blue">On</span>}
        <ChevronDown size={13} className={`shrink-0 transition-transform ${expanded ? 'rotate-180' : ''}`} />
      </button>
      <div id={panelId} hidden={!expanded} className="space-y-2 border-t border-border px-2.5 pb-2.5 pt-2">
        <label className="flex min-h-7 cursor-pointer items-center gap-2 text-[11px] text-text-secondary">
          <input type="checkbox" role="switch" checked={includeInPrompt} disabled={isEnhancing}
            onChange={event => setParam('source_audio_transcript_in_prompt', event.currentTarget.checked)}
            className="h-3.5 w-3.5 shrink-0 accent-accent-blue" />
          <span>Include transcript in video prompts</span>
        </label>
        <label className="flex min-h-7 cursor-pointer items-center gap-2 text-[11px] text-text-secondary">
          <input type="checkbox" role="switch" checked={useForEnhancement} disabled={isEnhancing}
            onChange={event => setParam('source_audio_transcript_for_enhancement', event.currentTarget.checked)}
            className="h-3.5 w-3.5 shrink-0 accent-accent-blue" />
          <span>Use transcript for prompt enhancement</span>
        </label>
        <p className="text-[10px] leading-relaxed text-text-muted">
          Each window gets the matching source words or lyrics. Automatic transcription can mishear, so treat it as context only; it does not add dialogue or subtitles, and the source audio stays unchanged.
        </p>
        <p className="text-[9px] leading-relaxed text-text-muted">
          Uses local Whisper small (downloaded if needed). Runs on CPU to avoid VRAM; later uses are cached.
        </p>
      </div>
    </section>
  )
}
