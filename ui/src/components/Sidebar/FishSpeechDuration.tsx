import { useRef } from 'react'
import { useStore } from '../../stores/useStore'
import { DurationPresetControl } from './DurationPresetControl'

export function FishSpeechDuration() {
  const duration = useStore(s => s.durationSeconds)
  const maximum = useStore(s => s.modelOptions?.duration_slider?.max ?? 300)
  const longform = useStore(s => s.modelOptions?.speech_longform === true)
  const setDuration = useStore(s => s.setDurationSeconds)
  const lastMaximum = useRef(duration > 0 ? duration : 30)
  const automatic = duration === 0
  return <div className="space-y-3">
    <div className="text-[11px] uppercase tracking-wider text-text-muted">Speech length</div>
    <div className="grid grid-cols-2 gap-1 rounded-lg border border-border p-1">
      {(['Auto', 'Maximum length'] as const).map((label, index) => <button key={label} type="button"
        aria-pressed={index === 0 ? automatic : !automatic}
        onClick={() => {
          if (index === 0) {
            if (duration > 0) lastMaximum.current = duration
            setDuration(0)
          } else if (automatic) setDuration(lastMaximum.current)
        }}
        className={`rounded-md px-3 py-2 text-xs ${((index === 0) === automatic)
          ? 'bg-accent-blue/15 text-accent-blue' : 'text-text-secondary hover:bg-bg-tertiary'}`}>
        {label}
      </button>)}
    </div>
    {automatic ? <p className="text-xs leading-relaxed text-text-muted">
      Ends naturally. Write follows the length in your brief. {longform ? 'Long scripts are generated in parts and joined into one file. ' : ''}Maestro uses a generous script-based safety limit to bound runaway silence or repetition. Pauses and delivery still come from the model.
    </p> : <DurationPresetControl label="Maximum speech duration" value={duration}
      onChange={value => { lastMaximum.current = value; setDuration(value) }} minSeconds={1} maxSeconds={maximum}
      showSingleWindow={false} quantizeToWindows={false} durationIsMaximum
      durationPresets={[{label:'5s',seconds:5},{label:'10s',seconds:10},{label:'15s',seconds:15},
        {label:'30s',seconds:30},{label:'1m',seconds:60},{label:'2m',seconds:120},{label:'5m',seconds:300},
        ...(longform ? [{label:'10m',seconds:600},{label:'20m',seconds:1200},{label:'1h',seconds:3600}] : [])]}
      modelLimitLabel="Write uses this duration as its draft target unless your brief or writer override specifies a length. For generation, it is a maximum; speech can finish earlier." />}
  </div>
}
