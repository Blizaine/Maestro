import { Image, Video, AudioLines } from 'lucide-react'
import { useStore } from '../../stores/useStore'

type StudioMediaMode = 'video' | 'image' | 'audio'

const modes: { value: StudioMediaMode; label: string; icon: typeof Image }[] = [
  { value: 'video', label: 'Video', icon: Video },
  { value: 'image', label: 'Image', icon: Image },
  { value: 'audio', label: 'Audio', icon: AudioLines },
]

export function GenerationModeSelector() {
  const generationMode = useStore(s => s.generationMode)
  const setGenerationMode = useStore(s => s.setGenerationMode)
  const toolsTool = useStore(s => s.toolsTool)
  const audioSubMode = useStore(s => s.audioSubMode)
  const videoWorkflow = useStore(s => s.studioVideoWorkflow)
  const setVideoWorkflow = useStore(s => s.setStudioVideoWorkflow)
  const imageWorkflow = useStore(s => s.studioImageWorkflow)
  const setImageWorkflow = useStore(s => s.setStudioImageWorkflow)
  const toolsUpscaleMedia = useStore(s => s.toolsUpscaleMedia)
  const setToolsTool = useStore(s => s.setToolsTool)

  const activeMode: StudioMediaMode = generationMode === 'avatar'
    ? 'video'
    : generationMode === 'tools'
      ? (toolsTool === 'upscale' ? toolsUpscaleMedia : toolsTool === 'film_grain' ? 'video' : 'audio')
      : generationMode

  const selectMode = (mode: StudioMediaMode) => {
    if (mode === activeMode) return
    if (mode === 'video') {
      setVideoWorkflow(videoWorkflow)
      return
    }
    if (mode === 'image') {
      setImageWorkflow(imageWorkflow)
      return
    }
    if (mode === 'audio' && audioSubMode === 'revoice') {
      setToolsTool('revoice')
      setGenerationMode('tools')
      return
    }
    setGenerationMode(mode)
  }

  return (
    <div className="flex bg-bg-tertiary rounded-lg p-0.5 border border-border">
      {modes.map(m => {
        const Icon = m.icon
        const active = activeMode === m.value
        return (
          <button
            key={m.value}
            onClick={() => selectMode(m.value)}
            className={`flex-1 flex items-center justify-center gap-1.5 text-xs py-2 rounded-md transition-all ${
              active
                ? 'bg-bg-active text-text-primary'
                : 'text-text-secondary hover:text-text-primary'
            }`}
          >
            <Icon size={14} />
            <span>{m.label}</span>
          </button>
        )
      })}
    </div>
  )
}
