import { useEffect } from 'react'
import { Music } from 'lucide-react'
import { useStore } from '../../stores/useStore'

const DIRECTOR_MUSIC_MODEL_ORDER = [
  'ace_step_v1_5_xl_sft_lm_4b',
  'minimax_music3',
]

// Director Music Video — "Generate a track" up-front options. The description
// itself is typed into the bottom chat (its Send button kicks off the whole
// write-song → render → analyze → video chain), so this panel only holds the
// generator, instrumental mode, and length. The LLM writes the model-specific
// Style + Lyrics internally; power users can hand-edit those in Studio Audio.
export function DirectorSongSetup() {
  const models = useStore(s => s.models)
  const enabledModels = useStore(s => s.enabledModels)
  const nsfwMode = useStore(s => s.servicesConfig?.nsfw_mode ?? false)
  const musicModel = useStore(s => s.directorMusicModel)
  const setMusicModel = useStore(s => s.setDirectorMusicModel)
  const instrumental = useStore(s => s.directorSongInstrumental)
  const setInstrumental = useStore(s => s.setDirectorSongInstrumental)
  const duration = useStore(s => s.directorSongDuration)
  const setDuration = useStore(s => s.setDirectorSongDuration)

  const musicModels = DIRECTOR_MUSIC_MODEL_ORDER
    .map(modelType => models.find(model => model.model_type === modelType))
    .filter(model => model != null)
    .filter(model => enabledModels.has(model.model_type))
    .filter(model => !model.nsfw_only || nsfwMode)
  const effectiveModel = musicModels.some(model => model.model_type === musicModel)
    ? musicModel
    : (musicModels[0]?.model_type || '')
  const selectedModel = musicModels.find(model => model.model_type === effectiveModel)
  const isMusic3 = selectedModel?.architecture === 'minimax_music3'
  const maximumDuration = isMusic3 ? 300 : 360

  useEffect(() => {
    if (effectiveModel && effectiveModel !== musicModel) {
      setMusicModel(effectiveModel)
    }
  }, [effectiveModel, musicModel, setMusicModel])

  useEffect(() => {
    const bounded = Math.min(maximumDuration, Math.max(5, duration))
    if (bounded !== duration) setDuration(bounded)
  }, [duration, maximumDuration, setDuration])

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
        <label className="text-[11px] text-text-muted uppercase tracking-wider mb-1.5 block">
          Music model
        </label>
        {musicModels.length > 0 ? (
          <>
            <select
              value={effectiveModel}
              onChange={e => setMusicModel(e.target.value)}
              className="w-full bg-bg-tertiary border border-border rounded-lg px-3 py-2 text-xs text-text-primary focus:outline-none focus:border-accent-blue"
            >
              {musicModels.map(model => (
                <option key={model.model_type} value={model.model_type}>
                  {model.name}
                </option>
              ))}
            </select>
            <p className="text-[10px] text-text-muted leading-snug mt-1.5">
              {selectedModel?.selector_help || selectedModel?.description}
              {selectedModel?.is_downloaded === false ? ' Downloads on first use.' : ''}
            </p>
          </>
        ) : (
          <p className="text-[10px] text-amber-400 leading-snug">
            Enable ACE-Step or MiniMax-Music3 in Settings → System → Enabled Models.
          </p>
        )}
      </div>

      <div>
        <div className="flex items-center justify-between mb-1">
          <label className="text-[11px] text-text-muted uppercase tracking-wider">Length</label>
          <span className="text-[10px] text-text-secondary tabular-nums">{duration}s</span>
        </div>
        <input
          type="range"
          min={5}
          max={maximumDuration}
          step={5}
          value={duration}
          onChange={e => setDuration(Number(e.target.value))}
          className="w-full accent-accent-blue"
        />
      </div>

      <p className="text-[10px] text-text-muted leading-snug">
        Describe your music video in the box below and hit Generate — the song
        {instrumental ? '' : ' + lyrics'} is written for you, then the full video is
        produced with {isMusic3 ? 'MiniMax-Music3' : 'ACE-Step'}. For hands-on
        control of style and lyrics, use Studio → Audio → Music.
      </p>
    </div>
  )
}
