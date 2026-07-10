import { Mic } from 'lucide-react'
import { useStore } from '../../stores/useStore'

const AUDIO_ACCEPT = '.wav,.mp3,.flac,.ogg,.m4a'

/**
 * Voice Reference (ID-LoRA) drop zone.
 *
 * Used by Studio Video mode and Director mode to attach a ~5-second
 * voice sample. The sample is sent to LTX-2 alongside the video prompt
 * and (when combined with an ID-LoRA) keeps the speaker's voice
 * consistent across clips.
 *
 * The component is gated by the global `voice_reference_enabled` flag
 * in services config — when the toggle in Settings → Services is off,
 * the section renders nothing.
 *
 * State lives on `directorVoiceRef` / `directorVoiceRefPath` /
 * `directorIdentityGuidanceScale` in the store. Despite the
 * "director" prefix, the submission path uses these for any video
 * generation (Studio mode or Director mode), so this single component
 * works in both contexts.
 */
export function VoiceRefSection() {
  const enabled = useStore(s => !!s.servicesConfig?.voice_reference_enabled)
  const voiceRef = useStore(s => s.directorVoiceRef)
  const setVoiceRef = useStore(s => s.setDirectorVoiceRef)
  const identityScale = useStore(s => s.directorIdentityGuidanceScale)
  const setIdentityScale = useStore(s => s.setDirectorIdentityGuidanceScale)

  if (!enabled) return null

  return (
    <div className="bg-bg-tertiary border border-border rounded-lg px-3 py-2.5">
      <div className="flex items-center justify-between mb-1.5">
        <span className="text-[11px] text-text-muted uppercase tracking-wider flex items-center gap-1">
          <Mic size={10} />
          Voice Reference (ID-LoRA)
        </span>
        {!voiceRef ? (
          <label className="cursor-pointer text-[10px] text-accent-blue hover:underline">
            + Add audio
            <input
              type="file"
              accept={AUDIO_ACCEPT}
              className="hidden"
              onChange={e => {
                const f = e.target.files?.[0]
                if (f) setVoiceRef(f)
                e.target.value = ''
              }}
            />
          </label>
        ) : (
          <button
            onClick={() => setVoiceRef(null)}
            className="text-[10px] text-red-400 hover:text-red-300 transition-colors"
          >
            Remove
          </button>
        )}
      </div>
      {voiceRef ? (
        <div className="space-y-2">
          <div className="flex items-center gap-1.5 bg-bg-secondary rounded px-2 py-1">
            <Mic size={10} className="text-accent-blue shrink-0" />
            <span className="text-[10px] text-text-secondary truncate">{voiceRef.name}</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-[10px] text-text-muted whitespace-nowrap">Identity scale</span>
            <input
              type="range"
              min={0}
              max={10}
              step={0.5}
              value={identityScale}
              onChange={e => setIdentityScale(parseFloat(e.target.value))}
              className="flex-1 h-1 accent-accent-blue"
            />
            <span className="text-[10px] text-text-muted w-6 text-right">{identityScale}</span>
          </div>
        </div>
      ) : (
        <p className="text-[10px] text-text-muted italic">
          ~5-second voice sample. Combined with an active ID-LoRA, keeps the speaker's voice consistent across video clips.
        </p>
      )}
    </div>
  )
}
