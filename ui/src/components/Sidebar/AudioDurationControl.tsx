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
  const segmentMaximum = modelOptions?.audio_segment_max_seconds
  return (
    <DurationPresetControl
      label={slider.label || 'Duration'}
      value={duration}
      onChange={setDuration}
      minSeconds={minimum}
      maxSeconds={maximum}
      showSingleWindow={false}
      quantizeToWindows={false}
      durationIsMaximum={!!segmentMaximum}
      durationPresets={segmentMaximum ? [
        { label: '15s', seconds: 15 }, { label: '30s', seconds: 30 },
        { label: '45s', seconds: 45 }, { label: '1m', seconds: 60 },
        { label: '2m', seconds: 120 }, { label: '3m', seconds: 180 },
        { label: '4m', seconds: 240 }, { label: '5m', seconds: 300 },
      ] : undefined}
      modelLimitLabel={segmentMaximum
        ? `Up to ${formatDuration(segmentMaximum)} per segment, ${formatDuration(maximum)} per output. Long scripts split automatically; short speech ends when the script finishes.`
        : isLongForm
        ? 'Long speech is synthesized in bounded chunks and assembled automatically.'
        : `This generator supports up to ${formatDuration(maximum)} per output.`}
    />
  )
}
