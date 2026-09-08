import { useEffect, useRef, useState } from 'react'
import {
  LONG_FORM_DURATION_PRESETS,
  durationWindowPlan,
  formatDuration,
  formatTimecode,
  maximumWholeWindowCount,
  nearestWholeWindowDuration,
  nativeDurationSlider,
  parseTimecode,
  recommendAutoDuration,
  wholeWindowDuration,
  type AutoDurationPlanningStyle,
  type DurationPlanningMode,
} from '../../lib/durationPlanning'

type PresetSelection = 'single' | number | null

interface DurationPresetControlProps {
  value: number
  onChange: (seconds: number) => void
  minSeconds?: number
  maxSeconds: number
  windowSeconds?: number | null
  /** Fresh output contributed by pass one when source context occupies part of it. */
  firstWindowSeconds?: number | null
  overlapSeconds?: number
  discardSeconds?: number
  showSingleWindow?: boolean
  label?: string
  disabled?: boolean
  modelLimitLabel?: string
  durationPresets?: ReadonlyArray<{ label: string; seconds: number }>
  durationIsMaximum?: boolean
  quantizeToWindows?: boolean
  /** Add Auto / Time / Window selection for long-form video planning. */
  enablePlanningModes?: boolean
  planningMode?: DurationPlanningMode
  onPlanningModeChange?: (mode: DurationPlanningMode) => void
  autoPrompt?: string
  autoPlanningStyle?: AutoDurationPlanningStyle
  autoSourceSeconds?: number | null
  autoSourceLabel?: string
  autoMediaOnly?: boolean
  autoManualWindowCount?: number | null
  autoMaximumInferredWindows?: number
  nativeTiming?: { minimumFrames: number; frameStep: number; fps: number }
  /** Stable capacity for Auto; independent of Time's duration-following window. */
  autoWindowSeconds?: number
  autoFirstWindowSeconds?: number
}

const WINDOW_COUNT_PRESETS = [1, 2, 3, 4, 6, 8] as const

export function DurationPresetControl({
  value,
  onChange,
  minSeconds = 1,
  maxSeconds,
  windowSeconds = null,
  firstWindowSeconds = null,
  overlapSeconds = 0,
  discardSeconds = 0,
  showSingleWindow = true,
  label = 'Duration',
  disabled = false,
  modelLimitLabel,
  durationPresets = LONG_FORM_DURATION_PRESETS,
  durationIsMaximum = false,
  quantizeToWindows = true,
  enablePlanningModes = false,
  planningMode,
  onPlanningModeChange,
  autoPrompt = '',
  autoPlanningStyle = 'faithful',
  autoSourceSeconds = null,
  autoSourceLabel,
  autoMediaOnly = false,
  autoManualWindowCount = null,
  autoMaximumInferredWindows = 8,
  nativeTiming,
  autoWindowSeconds,
  autoFirstWindowSeconds,
}: DurationPresetControlProps) {
  const inputRef = useRef<HTMLInputElement>(null)
  const [customText, setCustomText] = useState<string | null>(null)
  const [selectedPreset, setSelectedPreset] = useState<PresetSelection>(null)
  const [internalPlanningMode, setInternalPlanningMode] = useState<DurationPlanningMode>('auto')
  const effectivePlanningMode = enablePlanningModes
    ? (planningMode ?? internalPlanningMode)
    : 'duration'
  const effectiveWindow = Math.max(minSeconds, windowSeconds || value || minSeconds)
  const effectiveFirstWindow = Math.max(
    minSeconds,
    Math.min(effectiveWindow, firstWindowSeconds ?? effectiveWindow),
  )
  const hasReducedFirstWindow = effectiveFirstWindow < effectiveWindow - 0.01
  const exactPlan = durationWindowPlan(
    value,
    effectiveWindow,
    overlapSeconds,
    discardSeconds,
    effectiveFirstWindow,
  )
  const isSequence = exactPlan.windowCount > 1
  const maximumWindowCount = maximumWholeWindowCount(
    effectiveWindow,
    overlapSeconds,
    discardSeconds,
    maxSeconds,
    effectiveFirstWindow,
  )
  const autoPlan = recommendAutoDuration(
    autoWindowSeconds ?? effectiveWindow,
    overlapSeconds,
    discardSeconds,
    {
      prompt: autoPrompt,
      planningStyle: autoPlanningStyle,
      sourceSeconds: autoSourceSeconds,
      sourceLabel: autoSourceLabel,
      manualWindowCount: autoManualWindowCount,
      minimumSeconds: minSeconds,
      maximumSeconds: maxSeconds,
      maximumInferredWindows: autoMaximumInferredWindows,
      firstWindowSeconds: autoFirstWindowSeconds ?? effectiveFirstWindow,
    },
  )

  const selectPreset = (preset: Exclude<PresetSelection, null>) => {
    setSelectedPreset(preset)
    const next = preset === 'single'
      ? Math.min(maxSeconds, effectiveFirstWindow)
      : quantizeToWindows
        ? nearestWholeWindowDuration(
          preset,
          effectiveWindow,
          overlapSeconds,
          discardSeconds,
          maxSeconds,
          effectiveFirstWindow,
        ).generatedSeconds
        : Math.min(maxSeconds, Math.max(minSeconds, preset))
    if (Math.abs(next - value) > 0.05) onChange(next)
  }

  const slider = nativeTiming ? nativeDurationSlider(
    nativeTiming.minimumFrames, nativeTiming.frameStep, nativeTiming.fps, maxSeconds,
  ) : null

  useEffect(() => {
    if (effectivePlanningMode !== 'auto') return
    // Timed media and explicit prompt durations keep their exact requested
    // length (the final pass is trimmed). Story/manual recommendations use a
    // complete native-window duration.
    const next = autoPlan.requestedSeconds
    if (Math.abs(next - value) > 0.05) onChange(next)
  }, [autoPlan.requestedSeconds, effectivePlanningMode, onChange, value])

  const setPlanningMode = (mode: DurationPlanningMode) => {
    setSelectedPreset(null)
    if (onPlanningModeChange) onPlanningModeChange(mode)
    else setInternalPlanningMode(mode)
    if (mode === 'windows') {
      const capacity = autoWindowSeconds ?? effectiveWindow
      const firstCapacity = autoFirstWindowSeconds ?? effectiveFirstWindow
      const count = durationWindowPlan(value, capacity, overlapSeconds, discardSeconds, firstCapacity).windowCount
      const plan = wholeWindowDuration(
        count,
        capacity,
        overlapSeconds,
        discardSeconds,
        maxSeconds,
        firstCapacity,
      )
      if (Math.abs(plan.generatedSeconds - value) > 0.05) onChange(plan.generatedSeconds)
    }
  }

  const setWindowCount = (count: number) => {
    const plan = wholeWindowDuration(
      count,
      effectiveWindow,
      overlapSeconds,
      discardSeconds,
      maxSeconds,
      effectiveFirstWindow,
    )
    onChange(plan.generatedSeconds)
  }

  const commitCustom = () => {
    const parsed = parseTimecode(customText ?? formatTimecode(value))
    if (parsed == null) {
      setCustomText(null)
      return
    }
    const bounded = Math.min(maxSeconds, Math.max(minSeconds, parsed))
    setSelectedPreset(null)
    onChange(bounded)
    setCustomText(null)
  }

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between gap-2">
        <label className="text-[11px] text-text-muted uppercase tracking-wider">{label}</label>
        <span className="text-xs text-text-secondary tabular-nums">{formatDuration(value, true)}</span>
      </div>

      {enablePlanningModes && (
        <div className="grid grid-cols-3 rounded-lg border border-border bg-bg-secondary p-0.5" aria-label={`${label} planning mode`}>
          {(['auto', 'duration', 'windows'] as DurationPlanningMode[]).map(mode => (
            <button
              key={mode}
              type="button"
              disabled={disabled}
              onClick={() => setPlanningMode(mode)}
              title={mode === 'duration'
                ? 'Choose a target runtime'
                : mode === 'windows'
                  ? 'Choose the exact number of native generation windows'
                  : `Let Maestro infer duration from timed media, manual prompt lines, exact dialogue, and story scope (vague concepts max ${autoMaximumInferredWindows} windows; explicit scripts may run longer)`}
              className={`rounded-md px-2 py-1.5 text-[10px] capitalize transition-colors disabled:opacity-40 ${
                effectivePlanningMode === mode
                  ? 'bg-accent-blue/15 text-text-primary shadow-sm'
                  : 'text-text-muted hover:text-text-secondary'
              }`}
            >
              {mode === 'auto' ? (
                <span className="flex flex-col items-center leading-3">
                  <span>Auto</span>
                  <span className="text-[9px] normal-case tabular-nums">{formatDuration(autoPlan.requestedSeconds, true)}</span>
                </span>
              ) : mode === 'duration' ? 'Time' : 'Window'}
            </button>
          ))}
        </div>
      )}

      {effectivePlanningMode === 'duration' && (
        <>
          {slider && (
            <div className="space-y-1 rounded-lg border border-border bg-bg-secondary px-3 py-2">
              <input type="range" min={0} max={slider.count} step={1}
                value={slider.indexAt(value)} disabled={disabled}
                aria-label={`${label} model duration`} aria-valuetext={formatDuration(value, true)}
                onChange={event => {
                  setSelectedPreset(null)
                  setCustomText(null)
                  onChange(slider.secondsAt(Number(event.target.value)))
                }} className="w-full min-w-0" />
              <div className="flex justify-between text-[9px] text-text-muted tabular-nums">
                <span>{formatDuration(slider.secondsAt(0), true)}</span>
                <span>Model steps · up to {formatDuration(Math.min(maxSeconds, 300))}</span>
              </div>
              {value > slider.secondsAt(slider.count) + 0.05 && (
                <p className="text-[9px] text-text-muted">Long preset selected · {formatDuration(value, true)}. Moving the slider selects a duration within five minutes.</p>
              )}
            </div>
          )}
          <div className={showSingleWindow
            ? 'grid grid-cols-[repeat(14,minmax(0,1fr))] gap-1.5'
            : 'grid grid-cols-6 gap-1.5'}>
            {showSingleWindow && (
              <button
                type="button"
                disabled={disabled}
                title={hasReducedFirstWindow
                  ? `One continuation pass: ${formatDuration(Math.min(maxSeconds, effectiveFirstWindow), true)} of new footage; ${formatDuration(effectiveWindow, true)} native pass includes source-tail context.`
                  : `One complete model window: ${formatDuration(Math.min(maxSeconds, effectiveWindow), true)}`}
                onClick={() => selectPreset('single')}
                className={`col-span-4 w-full rounded-md border px-2 py-1 text-[9px] transition-colors disabled:opacity-40 ${
                  selectedPreset === 'single'
                    ? 'border-accent-blue bg-accent-blue/15 text-text-primary'
                    : 'border-border bg-bg-tertiary text-text-muted hover:text-text-secondary'
                }`}
              >
                1 window
              </button>
            )}
            {durationPresets.map(preset => {
              const plan = quantizeToWindows
                ? nearestWholeWindowDuration(
                    preset.seconds,
                    effectiveWindow,
                    overlapSeconds,
                    discardSeconds,
                    maxSeconds,
                    effectiveFirstWindow,
                  )
                : {
                    generatedSeconds: Math.min(maxSeconds, preset.seconds),
                    windowCount: 1,
                  }
              const unavailable = preset.seconds > maxSeconds + 0.0001
              return (
                <button
                  key={preset.label}
                  type="button"
                  disabled={disabled || unavailable}
                  title={unavailable
                    ? `${preset.label} exceeds this model's ${formatDuration(maxSeconds)} native maximum.`
                    : quantizeToWindows
                      ? `${preset.label} preset: ${formatDuration(plan.generatedSeconds, true)} actual, ${plan.windowCount} ${plan.windowCount === 1 ? 'window' : 'windows'}`
                      : `${durationIsMaximum ? 'Up to' : 'Output'} ${formatDuration(plan.generatedSeconds, true)}`}
                  onClick={() => selectPreset(preset.seconds)}
                  className={`${showSingleWindow ? 'col-span-2' : ''} w-full rounded-md border px-1.5 py-1 text-[9px] transition-colors disabled:opacity-30 disabled:cursor-not-allowed ${
                    selectedPreset === preset.seconds
                      ? 'border-accent-blue bg-accent-blue/15 text-text-primary'
                      : 'border-border bg-bg-tertiary text-text-muted hover:text-text-secondary'
                  }`}
                >
                  {preset.label}
                </button>
              )
            })}
            <button
              type="button"
              disabled={disabled}
              onClick={() => {
                setSelectedPreset(null)
                inputRef.current?.focus()
                inputRef.current?.select()
              }}
              className={`${showSingleWindow ? 'col-span-4' : 'col-span-2'} w-full rounded-md border px-2 py-1 text-[9px] transition-colors disabled:opacity-40 ${
                selectedPreset == null
                  ? 'border-accent-blue bg-accent-blue/15 text-text-primary'
                  : 'border-border bg-bg-tertiary text-text-muted hover:text-text-secondary'
              }`}
            >
              Custom
            </button>
          </div>
          <div className="flex items-center gap-2">
            <input
              ref={inputRef}
              type="text"
              inputMode="decimal"
              aria-label={`${label} timecode`}
              disabled={disabled}
              value={customText ?? formatTimecode(value)}
              onChange={event => setCustomText(event.target.value)}
              onFocus={() => {
                setSelectedPreset(null)
                setCustomText(formatTimecode(value))
              }}
              onBlur={commitCustom}
              onKeyDown={event => {
                if (event.key === 'Enter') {
                  commitCustom()
                  inputRef.current?.blur()
                }
              }}
              placeholder="HH:MM:SS"
              className="w-[104px] rounded-md border border-border bg-bg-secondary px-2 py-1.5 text-[10px] text-text-primary tabular-nums focus:outline-none focus:border-accent-blue disabled:opacity-50"
            />
            <div className="min-w-0 text-[9px] leading-snug text-text-muted">
              {!quantizeToWindows
                ? `${durationIsMaximum ? 'Maximum output' : 'Exact requested output'} · ${formatDuration(value, true)}`
                : isSequence
                ? hasReducedFirstWindow
                  ? `${exactPlan.windowCount} continuation passes; first adds ${formatDuration(effectiveFirstWindow, true)}, later passes add ${formatDuration(exactPlan.strideSeconds, true)}; final output ${formatDuration(value, true)}${exactPlan.trimSeconds > 0.05 ? ` (trim ${formatDuration(exactPlan.trimSeconds, true)})` : ''}`
                  : `${exactPlan.windowCount} windows × ${formatDuration(effectiveWindow, true)}; final output ${formatDuration(value, true)}${exactPlan.trimSeconds > 0.05 ? ` (trim ${formatDuration(exactPlan.trimSeconds, true)})` : ''}`
                : hasReducedFirstWindow
                  ? `Single continuation pass · ${formatDuration(value, true)} new footage; source-tail context uses the rest of the ${formatDuration(effectiveWindow, true)} pass`
                  : `Single window · ${formatDuration(value, true)}`}
              {modelLimitLabel && <span className="block">{modelLimitLabel}</span>}
            </div>
          </div>
        </>
      )}

      {effectivePlanningMode === 'windows' && (
        <div className="space-y-2 rounded-lg border border-border bg-bg-secondary p-2.5">
          <div className="grid grid-cols-6 gap-1.5">
            {WINDOW_COUNT_PRESETS.map(count => (
              <button
                key={count}
                type="button"
                disabled={disabled || count > maximumWindowCount}
                onClick={() => setWindowCount(count)}
                className={`rounded-md border px-1 py-1.5 text-[10px] tabular-nums transition-colors disabled:cursor-not-allowed disabled:opacity-30 ${
                  exactPlan.windowCount === count
                    ? 'border-accent-blue bg-accent-blue/15 text-text-primary'
                    : 'border-border bg-bg-tertiary text-text-muted hover:text-text-secondary'
                }`}
              >
                {count}
              </button>
            ))}
          </div>
          <div className="flex items-center gap-2">
            <input
              type="range"
              min={1}
              max={maximumWindowCount}
              step={1}
              disabled={disabled}
              value={Math.min(maximumWindowCount, exactPlan.windowCount)}
              onChange={event => setWindowCount(Number(event.target.value))}
              className="min-w-0 flex-1 h-1"
              aria-label="Exact window count"
            />
            <input
              type="number"
              inputMode="numeric"
              min={1}
              max={maximumWindowCount}
              step={1}
              disabled={disabled}
              value={exactPlan.windowCount}
              onChange={event => {
                if (event.target.value !== '') setWindowCount(Number(event.target.value))
              }}
              className="w-16 rounded-md border border-border bg-bg-tertiary px-2 py-1 text-center text-[10px] text-text-primary tabular-nums focus:border-accent-blue focus:outline-none disabled:opacity-50"
              aria-label="Window count"
            />
          </div>
          <div className="text-[9px] leading-snug text-text-muted">
            <span className="text-text-secondary">{exactPlan.windowCount} exact {exactPlan.windowCount === 1 ? 'window' : 'windows'}</span>
            {' · '}{formatDuration(value, true)} final output
            {hasReducedFirstWindow
              ? ` · first pass adds ${formatDuration(effectiveFirstWindow, true)}; later passes add ${formatDuration(exactPlan.strideSeconds, true)}`
              : ` · each pass up to ${formatDuration(effectiveWindow, true)}`}
            {modelLimitLabel && <span className="block mt-0.5">{modelLimitLabel}</span>}
          </div>
        </div>
      )}

      {effectivePlanningMode === 'auto' && (
        <div className="rounded-lg border border-accent-blue/25 bg-accent-blue/5 p-2.5 text-[10px] leading-relaxed">
          <div className="flex items-center justify-between gap-2">
            <span className="font-medium text-text-primary">Auto recommendation</span>
            <span className="text-accent-blue tabular-nums">
              {autoPlan.windowCount} {autoPlan.windowCount === 1 ? 'window' : 'windows'} · {formatDuration(autoPlan.requestedSeconds, true)}
            </span>
          </div>
          <p className="mt-1 text-text-secondary">{autoMediaOnly
            ? autoSourceSeconds ? 'Follows the control video. Choose Time or Window to change the runtime.' : 'Upload a control video to match its duration automatically.'
            : autoPlan.reason}</p>
          {!autoMediaOnly && <p className="mt-1 text-[9px] text-text-muted">
            Timed media, explicit durations, manual prompt lines, and explicit screenplay dialogue are honored. Vague inferred concepts are capped at {autoPlan.inferredWindowLimit} windows; choose Time or Window for a different runtime.
          </p>}
          {modelLimitLabel && <p className="mt-1 text-[9px] text-text-muted">{modelLimitLabel}</p>}
        </div>
      )}
    </div>
  )
}
