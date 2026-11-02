import { Fragment } from 'react'
import type { GenerationJob } from '../../types'
import { useStore } from '../../stores/useStore'
import { modelDisplayName } from '../../lib/modelDisplay'
import { generationSettingLabels, generationSettingValue } from '../../lib/generationDetails'

export function JobSettingsDetails({ job }: { job: GenerationJob }) {
  const models = useStore(state => state.models)
  const details = job.generationDetails
  const params = details?.params
  const loras = Array.isArray(params?.activated_loras)
    ? params.activated_loras.filter((value): value is string => typeof value === 'string') : null
  const weights = typeof params?.loras_multipliers === 'string'
    ? params.loras_multipliers.trim().split(/\s+/) : []
  const scalarSettings = Object.entries(params || {}).filter(([key, value]) => (
    !['prompt', 'negative_prompt', 'model_type', 'activated_loras', 'loras_multipliers'].includes(key)
    && ['string', 'number', 'boolean'].includes(typeof value)
  ))
  const modelType = typeof params?.model_type === 'string' ? params.model_type : ''

  return <details className="border-t border-border bg-bg-secondary/70 text-[11px]" data-job-settings>
    <summary className="cursor-pointer px-3 py-2 text-text-secondary select-none">Settings &amp; LoRAs</summary>
    <div className="px-3 pb-3 space-y-3">
      <p className="text-[10px] text-text-muted">Saved settings for this attempt. Current sidebar changes do not change this record.</p>
      {!params ? <p className="text-text-muted">Settings were not captured for this older job.</p> : <>
        <dl className="grid grid-cols-[auto_minmax(0,1fr)] gap-x-3 gap-y-1">
          <dt className="text-text-muted">Job</dt><dd className="break-all text-text-secondary">{job.id}</dd>
          {details?.workspace && <><dt className="text-text-muted">Workspace</dt><dd className="break-words text-text-secondary">{details.workspace}</dd></>}
          {modelType && <><dt className="text-text-muted">Model</dt><dd className="break-words text-text-secondary">{modelDisplayName(modelType, models)}</dd></>}
          {scalarSettings.map(([key, value]) => <Fragment key={key}>
            <dt className="text-text-muted">{generationSettingLabels[key] || key}</dt>
            <dd className="break-words text-text-secondary">{generationSettingValue(value)}</dd>
          </Fragment>)}
        </dl>
        <section>
          <div className="mb-1 font-medium text-text-muted">Selected LoRAs</div>
          {loras === null ? <p className="text-text-muted">LoRA selection was not recorded.</p> : loras.length === 0
            ? <p className="text-text-secondary">None</p>
            : <ul className="space-y-1">{loras.map((name, index) => <li key={`${name}-${index}`} className="flex gap-2">
              <span className="min-w-0 flex-1 break-all text-text-secondary">{name}</span>
              <span className="shrink-0 text-text-muted">{weights[index] || '1'}×</span>
            </li>)}</ul>}
        </section>
        {(['prompt', 'negative_prompt'] as const).map(key => typeof params[key] === 'string' && params[key] !== '' && <section key={key}>
          <div className="mb-1 font-medium text-text-muted">{key === 'prompt' ? 'Prompt' : 'Negative prompt'}</div>
          <p className="max-h-40 overflow-y-auto whitespace-pre-wrap break-words text-text-secondary">{params[key]}</p>
        </section>)}
        {details?.performance_plan && Object.keys(details.performance_plan).length > 0 && <section>
          <div className="mb-1 font-medium text-text-muted">Applied performance plan</div>
          <dl className="grid grid-cols-[auto_minmax(0,1fr)] gap-x-3 gap-y-1">
            {Object.entries(details.performance_plan).map(([key, value]) => <Fragment key={key}>
              <dt className="text-text-muted">{generationSettingLabels[key] || key}</dt>
              <dd className="break-words text-text-secondary">{generationSettingValue(value)}</dd>
            </Fragment>)}
          </dl>
        </section>}
      </>}
    </div>
  </details>
}
