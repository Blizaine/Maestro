import { useStore } from '../../stores/useStore'
import { DurationPresetControl } from './DurationPresetControl'
import { formatDuration } from '../../lib/durationPlanning'

export function AudioDurationControl() {
  const modelOptions = useStore(s => s.modelOptions)
  const duration = useStore(s => s.durationSeconds)
  const setDuration = useStore(s => s.setDurationSeconds)
  const slider = modelOptions?.duration_slider
  if (!slider) return null

  const minimum = Number(slider.min || 1)
  const maximum = Number(slider.max || 60)
  const isLongForm = maximum >= 60 * 60
  return (
    <DurationPresetControl
      label={slider.label || 'Duration'}
      value={duration}
      onChange={setDuration}
      minSeconds={minimum}
      maxSeconds={maximum}
      showSingleWindow={false}
      quantizeToWindows={false}
      modelLimitLabel={isLongForm
        ? 'Long speech is synthesized in bounded chunks and assembled automatically.'
        : `This generator supports up to ${formatDuration(maximum)} per output.`}
    />
  )
}
