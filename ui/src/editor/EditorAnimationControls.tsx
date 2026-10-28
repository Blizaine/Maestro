import { useId, useState } from 'react'
import { ChevronLeft, ChevronRight, KeyRound, Plus, Trash2 } from 'lucide-react'
import type { EditorAnimatedProperty, EditorKeyframe, EditorTimelineItem } from '../types'
import {
  clampEditorAnimatedValue,
  deleteEditorKeyframe,
  EDITOR_ANIMATED_PROPERTY_SETTINGS,
  EDITOR_KEYFRAME_EASINGS,
  editorClipRelativeTime,
  evaluateEditorProperty,
  getEditorAnimatedBaseValue,
  getEditorKeyframes,
  updateEditorKeyframe,
  upsertEditorKeyframe,
} from './editorAnimation'

interface EditorAnimationControlsProps {
  item: EditorTimelineItem
  properties: EditorAnimatedProperty[]
  absolutePlayhead: number
  fps: number
  disabled: boolean
  onUpdate: (patch: Partial<EditorTimelineItem>) => void
  onNavigate: (absoluteTime: number) => void
}

function formatTime(value: number): string {
  return `${Math.max(0, value).toFixed(2)}s`
}

function MotionNumberField({
  label,
  value,
  min,
  max,
  step,
  disabled,
  onChange,
}: {
  label: string
  value: number
  min: number
  max: number
  step: number
  disabled: boolean
  onChange: (value: number) => void
}) {
  const inputId = useId()
  return (
    <div className="min-w-0 space-y-1">
      <label htmlFor={inputId} className="block text-[10px] font-medium uppercase tracking-wider text-text-muted">{label}</label>
      <input
        id={inputId}
        type="number"
        value={Number.isFinite(value) ? Number(value.toFixed(4)) : 0}
        min={min}
        max={max}
        step={step}
        disabled={disabled}
        onChange={event => onChange(Number(event.target.value))}
        className="w-full rounded-md border border-border bg-bg-secondary px-2 py-2 text-[12px] text-text-primary outline-none focus:border-accent-blue/60 disabled:cursor-not-allowed disabled:opacity-50"
      />
    </div>
  )
}

export function EditorAnimationControls({
  item,
  properties,
  absolutePlayhead,
  fps,
  disabled,
  onUpdate,
  onNavigate,
}: EditorAnimationControlsProps) {
  const [property, setProperty] = useState<EditorAnimatedProperty>(properties[0] || 'x')
  const propertySelectId = useId()
  const firstProperty = properties[0]
  const selectedProperty = (properties.includes(property) ? property : firstProperty) || 'x'
  const relativeTime = editorClipRelativeTime(item, absolutePlayhead)
  const frames = getEditorKeyframes(item, selectedProperty)
  const settings = EDITOR_ANIMATED_PROPERTY_SETTINGS[selectedProperty]
  const value = evaluateEditorProperty(item, selectedProperty, absolutePlayhead)
  const totalKeyCount = properties.reduce((count, nextProperty) => count + getEditorKeyframes(item, nextProperty).length, 0)
  const currentKeyIndex = frames.findIndex(frame => Math.abs(frame.time - relativeTime) < 1e-4)
  const previousKey = frames.filter(frame => frame.time < relativeTime - 1e-4).at(-1)
  const nextKey = frames.find(frame => frame.time > relativeTime + 1e-4)
  const frameStep = Number.isFinite(fps) && fps > 0 ? 1 / fps : 1 / 24

  const applyItem = (next: EditorTimelineItem) => onUpdate({ keyframes: next.keyframes })
  const setCurrentKey = () => {
    const updated = upsertEditorKeyframe(item, selectedProperty, relativeTime, value)
    applyItem(updated)
  }
  const updateKey = (index: number, patch: Partial<EditorKeyframe>) => {
    applyItem(updateEditorKeyframe(item, selectedProperty, index, patch))
  }
  const removeKey = (index: number) => {
    applyItem(deleteEditorKeyframe(item, selectedProperty, index))
  }

  return (
    <section className="space-y-3 rounded-xl border border-border bg-bg-tertiary p-3" aria-label="Motion keyframes">
      <div className="flex items-start justify-between gap-2">
        <div>
          <h3 className="text-[12px] font-semibold text-text-primary">Motion</h3>
          <p className="mt-0.5 text-[10px] text-text-muted">{totalKeyCount} key{totalKeyCount === 1 ? '' : 's'} across {properties.length} animated properties</p>
        </div>
        <span className="shrink-0 rounded-md bg-bg-secondary px-2 py-1 font-mono text-[10px] text-text-secondary" title="Playhead position relative to this clip">
          {formatTime(relativeTime)} / {formatTime(item.duration)}
        </span>
      </div>

      <div className="space-y-1">
        <label htmlFor={propertySelectId} className="block text-[10px] font-medium uppercase tracking-wider text-text-muted">Animated property</label>
        <select
          id={propertySelectId}
          value={selectedProperty}
          onChange={event => setProperty(event.target.value as EditorAnimatedProperty)}
          className="w-full rounded-md border border-border bg-bg-secondary px-2 py-2 text-[12px] text-text-primary outline-none focus:border-accent-blue/60"
        >
          {properties.map(option => (
            <option key={option} value={option}>{EDITOR_ANIMATED_PROPERTY_SETTINGS[option].label}</option>
          ))}
        </select>
      </div>

      <div className="rounded-lg border border-border bg-bg-secondary p-3">
        <div className="flex items-center justify-between gap-2">
          <div>
            <div className="text-[10px] font-medium uppercase tracking-wider text-text-muted">At playhead</div>
            <div className="mt-0.5 font-mono text-[13px] text-text-primary">
              {value.toFixed(selectedProperty === 'x' || selectedProperty === 'y' || selectedProperty === 'rotation' ? 1 : 2)}{settings.unit || ''}
            </div>
          </div>
          <div className="flex items-center gap-1">
            <button
              type="button"
              onClick={() => previousKey && onNavigate(item.start + previousKey.time)}
              disabled={!previousKey}
              title={previousKey ? `Go to ${formatTime(previousKey.time)}` : 'No previous keyframe'}
              aria-label="Navigate to previous keyframe"
              className="grid h-9 w-9 place-items-center rounded-md border border-border text-text-secondary hover:bg-bg-hover disabled:cursor-not-allowed disabled:opacity-40"
            >
              <ChevronLeft size={15} />
            </button>
            <button
              type="button"
              onClick={() => nextKey && onNavigate(item.start + nextKey.time)}
              disabled={!nextKey}
              title={nextKey ? `Go to ${formatTime(nextKey.time)}` : 'No next keyframe'}
              aria-label="Navigate to next keyframe"
              className="grid h-9 w-9 place-items-center rounded-md border border-border text-text-secondary hover:bg-bg-hover disabled:cursor-not-allowed disabled:opacity-40"
            >
              <ChevronRight size={15} />
            </button>
          </div>
        </div>
        <button
          type="button"
          onClick={setCurrentKey}
          disabled={disabled}
          title={disabled ? 'Unlock this track to edit keyframes' : `${currentKeyIndex >= 0 ? 'Update' : 'Set'} ${settings.label} at ${formatTime(relativeTime)}`}
          className="mt-3 flex w-full items-center justify-center gap-1.5 rounded-lg bg-accent-blue px-3 py-2.5 text-[11px] font-semibold text-white hover:brightness-110 disabled:cursor-not-allowed disabled:opacity-45"
        >
          {currentKeyIndex >= 0 ? <KeyRound size={13} /> : <Plus size={13} />}
          {currentKeyIndex >= 0 ? 'Update key' : 'Set key'} at {formatTime(relativeTime)}
        </button>
      </div>

      {frames.length ? (
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <h4 className="text-[10px] font-semibold uppercase tracking-wider text-text-muted">{settings.label} keys</h4>
            <span className="text-[10px] text-text-muted">{frames.length} total</span>
          </div>
          {frames.map((frame, index) => (
            <div key={`${frame.time}-${index}`} className="space-y-2 rounded-lg border border-border bg-bg-secondary p-2.5">
              <div className="flex items-center justify-between gap-2">
                <span className="inline-flex items-center gap-1.5 text-[10px] font-medium text-text-secondary">
                  <KeyRound size={11} className="text-accent-blue" /> Key {index + 1} · {formatTime(frame.time)}
                </span>
                <button
                  type="button"
                  onClick={() => removeKey(index)}
                  disabled={disabled}
                  title={`Delete keyframe at ${formatTime(frame.time)}`}
                  aria-label={`Delete keyframe ${index + 1} at ${formatTime(frame.time)}`}
                  className="grid h-8 w-8 place-items-center rounded-md text-text-muted hover:bg-red-500/10 hover:text-red-300 disabled:cursor-not-allowed disabled:opacity-40"
                >
                  <Trash2 size={13} />
                </button>
              </div>
              <div className="grid grid-cols-2 gap-2">
                <MotionNumberField
                  label={`Key ${index + 1} time`}
                  value={frame.time}
                  min={0}
                  max={item.duration}
                  step={frameStep}
                  disabled={disabled}
                  onChange={time => updateKey(index, { time })}
                />
                <MotionNumberField
                  label={`Key ${index + 1} value`}
                  value={frame.value}
                  min={settings.min}
                  max={settings.max}
                  step={settings.step}
                  disabled={disabled}
                  onChange={keyValue => updateKey(index, { value: clampEditorAnimatedValue(selectedProperty, keyValue, getEditorAnimatedBaseValue(item, selectedProperty)) })}
                />
              </div>
              <div className="space-y-1">
                <label htmlFor={`${propertySelectId}-easing-${index}`} className="block text-[10px] font-medium uppercase tracking-wider text-text-muted">Easing to next key</label>
                <select
                  id={`${propertySelectId}-easing-${index}`}
                  value={frame.easing}
                  disabled={disabled}
                  onChange={event => updateKey(index, { easing: event.target.value as EditorKeyframe['easing'] })}
                  className="w-full rounded-md border border-border bg-bg-secondary px-2 py-2 text-[11px] text-text-primary outline-none focus:border-accent-blue/60 disabled:cursor-not-allowed disabled:opacity-50"
                >
                  {EDITOR_KEYFRAME_EASINGS.map(easing => <option key={easing.value} value={easing.value}>{easing.label}</option>)}
                </select>
              </div>
            </div>
          ))}
        </div>
      ) : (
        <p className="rounded-lg border border-dashed border-border px-3 py-3 text-[11px] leading-relaxed text-text-muted">
          No {settings.label.toLowerCase()} keys yet. Set a key at the playhead; the clip’s current value is retained from time zero.
        </p>
      )}

      {disabled && <p className="text-[10px] text-amber-300">Unlock this track to edit its motion.</p>}
    </section>
  )
}
