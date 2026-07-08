import { Music } from 'lucide-react'
import { useStore } from '../../stores/useStore'

// Director Music Video — "Generate a track" up-front options. The description
// itself is typed into the bottom chat (its Send button kicks off the whole
// write-song → render → analyze → video chain), so this panel only holds the
// small choices: instrumental + length. The LLM writes the Style + Lyrics
// internally — power users who want to hand-edit those use Studio → Audio → Music.
export function DirectorSongSetup() {
  const instrumental = useStore(s => s.directorSongInstrumental)
  const setInstrumental = useStore(s => s.setDirectorSongInstrumental)
  const duration = useStore(s => s.directorSongDuration)
  const setDuration = useStore(s => s.setDirectorSongDuration)

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <label className="text-[11px] text-text-muted uppercase tracking-wider flex items-center gap-1.5">
          <Music size={12} /> Generate a song
        </label>
        <label className="flex items-center gap-1.5 cursor-pointer text-[10px] text-text-secondary hover:text-text-primary transition-colors">
          <input
            type="checkbox"
            checked={instrumental}
            onChange={e => setInstrumental(e.target.checked)}
            className="accent-accent-blue"
          />
          Instrumental
        </label>
      </div>

      <div>
        <div className="flex items-center justify-between mb-1">
          <label className="text-[11px] text-text-muted uppercase tracking-wider">Length</label>
          <span className="text-[10px] text-text-secondary tabular-nums">{duration}s</span>
        </div>
        <input
          type="range"
          min={30}
          max={300}
          step={10}
          value={duration}
          onChange={e => setDuration(Number(e.target.value))}
          className="w-full accent-accent-blue"
        />
      </div>

      <p className="text-[10px] text-text-muted leading-snug">
        Describe your music video in the box below and hit Generate — the song
        {instrumental ? '' : ' + lyrics'} is written for you, then the full video is
        produced. For hands-on control of style and lyrics, use Studio → Audio → Music.
      </p>
    </div>
  )
}
