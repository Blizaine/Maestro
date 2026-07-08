import { Mic, Music, Zap, Layers } from 'lucide-react'
import { useStore } from '../../stores/useStore'
import type { AudioSubMode } from '../../types'

const modes: { value: AudioSubMode; label: string; icon: typeof Mic }[] = [
  { value: 'speech', label: 'Speech', icon: Mic },
  { value: 'music', label: 'Music', icon: Music },
  { value: 'sfx', label: 'SFX', icon: Zap },
  { value: 'mixer', label: 'Mixer', icon: Layers },
]

export function AudioSubModeToggle() {
  const audioSubMode = useStore(s => s.audioSubMode)
  const setAudioSubMode = useStore(s => s.setAudioSubMode)

  return (
    <div className="flex bg-bg-tertiary rounded-lg p-0.5 border border-border">
      {modes.map(m => {
        const Icon = m.icon
        const active = audioSubMode === m.value
        return (
          <button
            key={m.value}
            onClick={() => setAudioSubMode(m.value)}
            className={`flex-1 flex items-center justify-center gap-1.5 text-xs py-1.5 rounded-md transition-all ${
              active
                ? 'bg-bg-active text-text-primary'
                : 'text-text-secondary hover:text-text-primary'
            }`}
          >
            <Icon size={13} />
            <span>{m.label}</span>
          </button>
        )
      })}
    </div>
  )
}
