import { useId } from 'react'
import { RotateCcw, SunMedium } from 'lucide-react'
import type { EditorColorSettings } from '../types'
import { DEFAULT_EDITOR_COLOR_SETTINGS, normalizeEditorColorSettings } from './editorColor'

function ColorSlider({
  color,
  property,
  label,
  min,
  max,
  step,
  displayValue,
  disabled,
  onChange,
}: {
  color: EditorColorSettings
  property: keyof EditorColorSettings
  label: string
  min: number
  max: number
  step: number
  displayValue: string
  disabled: boolean
  onChange: (property: keyof EditorColorSettings, value: number) => void
}) {
  const inputId = useId()
  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between gap-2">
        <label htmlFor={inputId} className="text-[11px] font-medium text-text-secondary">{label}</label>
        <span className="font-mono text-[10px] text-text-muted">{displayValue}</span>
      </div>
      <input
        id={inputId}
        type="range"
        min={min}
        max={max}
        step={step}
        value={color[property]}
        disabled={disabled}
        onChange={event => onChange(property, Number(event.target.value))}
        className="w-full accent-[var(--color-accent-blue)] disabled:opacity-40"
      />
    </div>
  )
}

export function EditorColorControls({
  color,
  disabled = false,
  onChange,
}: {
  color?: EditorColorSettings
  disabled?: boolean
  onChange: (color: EditorColorSettings) => void
}) {
  const current = normalizeEditorColorSettings(color)
  const patch = (property: keyof EditorColorSettings, value: number) => {
    const next = normalizeEditorColorSettings({ ...current, [property]: value })
    onChange(next)
  }

  return (
    <section className="space-y-3 rounded-xl border border-border bg-bg-tertiary p-3" aria-label="Color adjustments">
      <div className="flex items-center justify-between gap-2">
        <div>
          <h3 className="flex items-center gap-1.5 text-[12px] font-semibold text-text-primary"><SunMedium size={14} className="text-accent-blue" /> Color</h3>
          <p className="mt-0.5 text-[10px] text-text-muted">Non-destructive clip adjustments.</p>
        </div>
        <button
          type="button"
          onClick={() => onChange({ ...DEFAULT_EDITOR_COLOR_SETTINGS })}
          disabled={disabled}
          title="Reset exposure, contrast, saturation, and temperature"
          aria-label="Reset color adjustments"
          className="inline-flex items-center gap-1.5 rounded-md border border-border px-2 py-1.5 text-[10px] font-medium text-text-secondary hover:bg-bg-hover disabled:cursor-not-allowed disabled:opacity-45"
        >
          <RotateCcw size={11} /> Reset
        </button>
      </div>
      <div className="space-y-3">
        <ColorSlider
          color={current}
          property="exposure"
          label="Exposure"
          min={-2}
          max={2}
          step={0.01}
          displayValue={`${current.exposure.toFixed(2)} EV`}
          disabled={disabled}
          onChange={patch}
        />
        <ColorSlider
          color={current}
          property="contrast"
          label="Contrast"
          min={0}
          max={2}
          step={0.01}
          displayValue={current.contrast.toFixed(2)}
          disabled={disabled}
          onChange={patch}
        />
        <ColorSlider
          color={current}
          property="saturation"
          label="Saturation"
          min={0}
          max={2}
          step={0.01}
          displayValue={current.saturation.toFixed(2)}
          disabled={disabled}
          onChange={patch}
        />
        <ColorSlider
          color={current}
          property="temperature"
          label="Temperature"
          min={-1}
          max={1}
          step={0.01}
          displayValue={current.temperature < 0
            ? `${Math.abs(current.temperature).toFixed(2)} cool`
            : current.temperature > 0 ? `${current.temperature.toFixed(2)} warm` : 'Neutral'}
          disabled={disabled}
          onChange={patch}
        />
      </div>
      {disabled && <p className="text-[10px] text-amber-300">Unlock this track to edit color.</p>}
    </section>
  )
}
