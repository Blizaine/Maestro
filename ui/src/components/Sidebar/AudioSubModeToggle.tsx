import { AudioLines, Layers, Mic, Music, Zap } from 'lucide-react'
import { useStore } from '../../stores/useStore'
import type { AudioSubMode } from '../../types'
import { StudioWorkflowSelect, type StudioWorkflowGroup } from './StudioWorkflowSelect'

const AUDIO_WORKFLOW_GROUPS: StudioWorkflowGroup<AudioSubMode>[] = [
  {
    label: 'Create',
    options: [
      { value: 'speech', label: 'Speech', description: 'Generate speech and cloned voices', icon: Mic },
      { value: 'music', label: 'Music', description: 'Create songs and instrumental music', icon: Music },
      { value: 'sfx', label: 'Sound Effects', description: 'Generate synchronized sound effects', icon: Zap },
    ],
  },
  {
    label: 'Process',
    options: [
      { value: 'mixer', label: 'Mixer', description: 'Combine and balance audio tracks', icon: Layers },
      { value: 'revoice', label: 'Revoice', description: 'Replace voices in an existing video', icon: AudioLines },
    ],
  },
]

const AUDIO_OPTIONS = Object.fromEntries(
  AUDIO_WORKFLOW_GROUPS.flatMap(group => group.options).map(option => [option.value, option]),
) as Record<AudioSubMode, (typeof AUDIO_WORKFLOW_GROUPS)[number]['options'][number]>

export function AudioSubModeToggle() {
  const generationMode = useStore(state => state.generationMode)
  const audioSubMode = useStore(state => state.audioSubMode)
  const toolsTool = useStore(state => state.toolsTool)
  const setAudioSubMode = useStore(state => state.setAudioSubMode)
  const setGenerationMode = useStore(state => state.setGenerationMode)
  const setToolsTool = useStore(state => state.setToolsTool)

  const activeMode: AudioSubMode = generationMode === 'tools' && toolsTool === 'revoice'
    ? 'revoice'
    : audioSubMode

  const selectWorkflow = (mode: AudioSubMode) => {
    if (mode === 'revoice') {
      if (audioSubMode !== 'revoice') setAudioSubMode('revoice')
      setToolsTool('revoice')
      if (generationMode !== 'tools') setGenerationMode('tools')
      return
    }

    // Switch back to the real Audio generation mode before selecting its
    // model-backed workflow. Zustand updates synchronously, so the second
    // action sees the restored Audio params rather than the Tools snapshot.
    if (generationMode !== 'audio') setGenerationMode('audio')
    useStore.getState().setAudioSubMode(mode)
  }

  return (
    <StudioWorkflowSelect<AudioSubMode>
      value={activeMode}
      activeOption={AUDIO_OPTIONS[activeMode]}
      groups={AUDIO_WORKFLOW_GROUPS}
      hint="Create · Process"
      onChange={selectWorkflow}
    />
  )
}
