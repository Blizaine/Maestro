import { useState } from 'react'
import { RotateCcw, Sparkles } from 'lucide-react'
import { useStore } from '../../stores/useStore'
import {
  countKreaIdentityReferences,
  isKreaIdentityEdit,
  KREA_IDENTITY_DEFAULTS,
  normalizeKreaIdentitySettings,
  type KreaIdentitySettings,
} from '../../lib/kreaIdentityControls'

type SettingKey = keyof KreaIdentitySettings

function NumberRangeSetting({
  label,
  value,
  min,
  max,
  rangeStep,
  numberStep,
  suffix,
  onChange,
}: {
  label: string
  value: number
  min: number
  max: number
  rangeStep: number
  numberStep: number
  suffix?: string
  onChange: (value: number) => void
}) {
  const [draftState, setDraftState] = useState<{ baseValue: number; text: string } | null>(null)
  // A draft is valid only for the value it was typed against. If a slider,
  // reset, or model switch changes the committed value, render that value
  // immediately without synchronizing state from an effect.
  const draft = draftState?.baseValue === value ? draftState.text : String(value)

  const commitDraft = () => {
    const parsed = draft.trim() === '' ? Number.NaN : Number(draft)
    if (Number.isFinite(parsed)) onChange(parsed)
    setDraftState(null)
  }

  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between gap-3">
        <label htmlFor={`krea-${label.toLowerCase().replace(/[^a-z0-9]+/g, '-')}`} className="text-[11px] text-text-secondary">
          {label}
        </label>
        <div className="flex items-center gap-1">
          <input
            id={`krea-${label.toLowerCase().replace(/[^a-z0-9]+/g, '-')}`}
            type="number"
            min={min}
            max={max}
            step={numberStep}
            value={draft}
            onChange={event => setDraftState({ baseValue: value, text: event.currentTarget.value })}
            onBlur={commitDraft}
            onKeyDownCapture={event => {
              // SidebarDialog owns a native keydown listener on its panel,
              // which runs before React's bubbled onKeyDown handler. Cancel
              // here so Escape restores this draft without closing the panel.
              if (event.key === 'Escape') {
                event.preventDefault()
                event.stopPropagation()
                setDraftState(null)
              }
            }}
            onKeyDown={event => {
              if (event.key === 'Enter') {
                event.currentTarget.blur()
              }
            }}
            aria-label={`${label} value`}
            className="w-[4.5rem] rounded-md border border-border bg-bg-primary px-1.5 py-1 text-right text-xs tabular-nums text-text-primary focus:outline-none focus:border-accent-blue"
          />
          {suffix && <span className="w-6 text-[10px] text-text-muted">{suffix}</span>}
        </div>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={rangeStep}
        value={value}
        onChange={event => onChange(Number(event.currentTarget.value))}
        aria-label={`${label} slider`}
        className="w-full accent-accent-blue"
      />
    </div>
  )
}

export function KreaIdentityControls() {
  const params = useStore(state => state.params)
  const modelOptions = useStore(state => state.modelOptions)
  const selectedModel = useStore(state => state.models.find(model => model.model_type === state.params.model_type))
  const generationMode = useStore(state => state.generationMode)
  const workflow = useStore(state => state.studioImageWorkflow)
  const imageRefs = useStore(state => state.imageRefs)
  const imageSourcePath = useStore(state => state.imageWorkflowSourcePath)
  const imageSourceFile = useStore(state => state.imageWorkflowSourceFile)
  const setParam = useStore(state => state.setParam)
  const [resetVersion, setResetVersion] = useState(0)

  const activeModel = selectedModel || { model_type: params.model_type, architecture: modelOptions?.architecture }
  if (generationMode !== 'image' || workflow === 'upscale' || !isKreaIdentityEdit(activeModel)) return null

  const sourceImagePresent = Boolean(
    imageSourcePath
    || imageSourceFile
    || (Number(params.image_mode) === 2 && (params.image_start || params.image_guide)),
  )
  const referenceCount = countKreaIdentityReferences({
    generationMode,
    workflow,
    imageMode: params.image_mode,
    sourceImagePresent,
    referenceCount: imageRefs.length,
    editReferencesSupported: modelOptions?.image_ref_inpaint === true,
  })
  const twoReferences = referenceCount >= 2
  const settings = normalizeKreaIdentitySettings(params.custom_settings)
  const customSettings = params.custom_settings && typeof params.custom_settings === 'object' && !Array.isArray(params.custom_settings)
    ? params.custom_settings
    : {}

  const update = (patch: Partial<KreaIdentitySettings>) => {
    setParam('custom_settings', { ...customSettings, ...patch })
  }
  const setValue = (key: SettingKey, value: number) => {
    const normalized = normalizeKreaIdentitySettings({ ...settings, [key]: value })
    update({ [key]: normalized[key] })
  }
  const resetSettings = () => {
    update(KREA_IDENTITY_DEFAULTS)
    setResetVersion(version => version + 1)
  }
  const inputKey = `${activeModel.model_type}:${resetVersion}`
  const subjectLabel = twoReferences ? 'Subject likeness · Image 2' : 'Subject likeness'

  return (
    <section aria-label="Krea Identity Edit" className="space-y-3 rounded-lg border border-border bg-bg-tertiary/40 p-3">
      <div className="flex items-center justify-between gap-2">
        <span className="text-[11px] uppercase tracking-wider text-text-muted">Krea Identity Edit</span>
        <button type="button" onClick={resetSettings}
          className="inline-flex min-h-7 items-center gap-1 rounded-md px-2 text-[10px] text-text-muted hover:bg-bg-hover hover:text-text-primary">
          <RotateCcw size={11} /> Reset defaults
        </button>
      </div>

      {referenceCount > 0 ? (
        <>
          {twoReferences && <p className="text-[10px] leading-relaxed text-text-muted">
            Image 1 is the scene/reference. Image 2 is the subject whose identity should stay consistent.
          </p>}
          <NumberRangeSetting key={`${inputKey}:subject`} label={subjectLabel} value={settings.krea2_ref_boost} min={0} max={10}
            rangeStep={0.1} numberStep={0.1} onChange={value => setValue('krea2_ref_boost', value)} />
          {twoReferences && <NumberRangeSetting key={`${inputKey}:scene`} label="Scene/reference likeness · Image 1" value={settings.krea2_ref_boost_a}
            min={0} max={10} rangeStep={0.1} numberStep={0.1}
            onChange={value => setValue('krea2_ref_boost_a', value)} />}
          <p className="text-[10px] leading-relaxed text-text-muted">
            1 keeps likeness neutral; 4 is strong. Above about 8, Krea can resist other edit instructions.
          </p>
          <button type="button" onClick={() => setValue('krea2_ref_boost', 4)}
            className="inline-flex min-h-8 items-center gap-1.5 rounded-md border border-border px-2.5 text-[10px] text-text-secondary hover:border-accent-blue/50 hover:text-text-primary">
            <Sparkles size={11} /> Strong likeness (4)
          </button>
        </>
      ) : (
        <p className="text-[10px] leading-relaxed text-text-muted">Add a reference image to tune subject likeness.</p>
      )}

      <div className="border-t border-border/70 pt-3">
        <NumberRangeSetting key={`${inputKey}:grounding`} label="Grounding size · longer edge" value={settings.krea2_grounding_px}
          min={384} max={1536} rangeStep={64} numberStep={1} suffix="px"
          onChange={value => setValue('krea2_grounding_px', value)} />
        <p className="mt-1.5 text-[10px] leading-relaxed text-text-muted">
          Default 768px. Sets the longer edge of the image encoder&apos;s grounding input, not output resolution. 384–768px is the trained range; higher values are experimental and cost more time and memory.
        </p>
      </div>
    </section>
  )
}
