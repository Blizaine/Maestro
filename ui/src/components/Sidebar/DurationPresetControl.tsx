import { useEffect, useRef, useState } from 'react'
import {
  LONG_FORM_DURATION_PRESETS,
  durationWindowPlan,
  formatDuration,
  formatTimecode,
  maximumWholeWindowCount,
  nearestWholeWindowDuration,
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
  overlapSeconds?: number
  discardSeconds?: number
  showSingleWindow?: boolean
  label?: string
  disabled?: boolean
  modelLimitLabel?: string
  quantizeToWindows?: boolean
  /** Add Duration / Windows / Auto selection for long-form video planning. */
  enablePlanningModes?: boolean
  planningMode?: DurationPlanningMode
  onPlanningModeChange?: (mode: DurationPlanningMode) => void
  autoPrompt?: string
  autoPlanningStyle?: AutoDurationPlanningStyle
  autoSourceSeconds?: number | null
  autoSourceLabel?: string
  autoManualWindowCount?: number | null
  autoMaximumInferredWindows?: number
}

const WINDOW_COUNT_PRESETS = [1, 2, 3, 4, 6, 8] as const

export function DurationPresetControl({
  value,
  onChange,
  minSeconds = 1,
  maxSeconds,
  windowSeconds = null,
  overlapSeconds = 0,
  discardSeconds = 0,
  showSingleWindow = true,
  label = 'Duration',
  disabled = false,
  modelLimitLabel,
  quantizeToWindows = true,
  enablePlanningModes = false,
  planningMode,
  onPlanningModeChange,
  autoPrompt = '',
  autoPlanningStyle = 'faithful',
  autoSourceSeconds = null,
  autoSourceLabel,
  autoManualWindowCount = null,
  autoMaximumInferredWindows = 8,
}: DurationPresetControlProps) {
  const inputRef = useRef<HTMLInputElement>(null)
  const [customText, setCustomText] = useState<string | null>(null)
  const [selectedPreset, setSelectedPreset] = useState<PresetSelection>(null)
  const [internalPlanningMode, setInternalPlanningMode] = useState<DurationPlanningMode>('duration')
  const effectivePlanningMode = enablePlanningModes
    ? (planningMode ?? internalPlanningMode)
    : 'duration'
  const effectiveWindow = Math.max(minSeconds, windowSeconds || value || minSeconds)
  const exactPlan = durationWindowPlan(value, effectiveWindow, overlapSeconds, discardSeconds)
  const isSequence = exactPlan.windowCount > 1
  const maximumWindowCount = maximumWholeWindowCount(
    effectiveWindow,
    overlapSeconds,
    discardSeconds,
    maxSeconds,
  )
  const autoPlan = recommendAutoDuration(
    effectiveWindow,
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
    },
  )

  useEffect(() => {
    if (effectivePlanningMode !== 'duration' || selectedPreset == null) return
    const next = selectedPreset === 'single'
      ? Math.min(maxSeconds, effectiveWindow)
      : quantizeToWindows
        ? nearestWholeWindowDuration(
          selectedPreset,
          effectiveWindow,
          overlapSeconds,
          discardSeconds,
          maxSeconds,
        ).generatedSeconds
        : Math.min(maxSeconds, Math.max(minSeconds, selectedPreset))
    if (Math.abs(next - value) > 0.05) onChange(next)
  }, [discardSeconds, effectivePlanningMode, effectiveWindow, maxSeconds, minSeconds, onChange, overlapSeconds, quantizeToWindows, selectedPreset, value])

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
      const plan = wholeWindowDuration(
        exactPlan.windowCount,
        effectiveWindow,
        overlapSeconds,
        discardSeconds,
        maxSeconds,
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
          {(['duration', 'windows', 'auto'] as DurationPlanningMode[]).map(mode => (
            <button
              key={mode}
              type="button"
              disabled={disabled}
              onClick={() => setPlanningMode(mode)}
              title={mode === 'duration'
                ? 'Choose a target runtime'
                : mode === 'windows'
                  ? 'Choose the exact number of native generation windows'
                  : `Let Maestro infer a bounded duration from timed media, manual prompt lines, exact dialogue, and story scope (inferred ideas max ${autoMaximumInferredWindows} windows)`}
              className={`rounded-md px-2 py-1.5 text-[10px] capitalize transition-colors disabled:opacity-40 ${
                effectivePlanningMode === mode
                  ? 'bg-accent-blue/15 text-text-primary shadow-sm'
                  : 'text-text-muted hover:text-text-secondary'
              }`}
            >
              {mode === 'auto' ? 'Auto · Beta' : mode}
            </button>
          ))}
        </div>
      )}

      {effectivePlanningMode === 'duration' && (
        <>
          <div className={showSingleWindow
            ? 'grid grid-cols-[repeat(14,minmax(0,1fr))] gap-1.5'
            : 'grid grid-cols-6 gap-1.5'}>
            {showSingleWindow && (
              <button
                type="button"
                disabled={disabled}
                title={`One complete model window: ${formatDuration(Math.min(maxSeconds, effectiveWindow), true)}`}
                onClick={() => setSelectedPreset('single')}
                className={`col-span-4 w-full rounded-md border px-2 py-1 text-[9px] transition-colors disabled:opacity-40 ${
                  selectedPreset === 'single'
                    ? 'border-accent-blue bg-accent-blue/15 text-text-primary'
                    : 'border-border bg-bg-tertiary text-text-muted hover:text-text-secondary'
                }`}
              >
                1 window
              </button>
            )}
            {LONG_FORM_DURATION_PRESETS.map(preset => {
              const plan = quantizeToWindows
                ? nearestWholeWindowDuration(
                    preset.seconds,
                    effectiveWindow,
                    overlapSeconds,
                    discardSeconds,
                    maxSeconds,
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
                      : `${preset.label} output: ${formatDuration(plan.generatedSeconds, true)}`}
                  onClick={() => setSelectedPreset(preset.seconds)}
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
                ? `Exact requested output · ${formatDuration(value, true)}`
                : isSequence
                ? `${exactPlan.windowCount} windows × ${formatDuration(effectiveWindow, true)}; final output ${formatDuration(value, true)}${exactPlan.trimSeconds > 0.05 ? ` (trim ${formatDuration(exactPlan.trimSeconds, true)})` : ''}`
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
            {' · '}each pass up to {formatDuration(effectiveWindow, true)}
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
          <p className="mt-1 text-text-secondary">{autoPlan.reason}</p>
          <p className="mt-1 text-[9px] text-text-muted">
            Timed media, explicit durations, and manual prompt lines are honored exactly. Inferred story scope is capped at {autoPlan.inferredWindowLimit} windows; choose Duration or Windows for longer jobs.
          </p>
          {modelLimitLabel && <p className="mt-1 text-[9px] text-text-muted">{modelLimitLabel}</p>}
        </div>
      )}
    </div>
  )
}
