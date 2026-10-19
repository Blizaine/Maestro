import { AlertTriangle, Gauge } from 'lucide-react'
import { useStore } from '../../stores/useStore'
import { estimateH3DurationGuidance } from '../../lib/h3DurationGuidance'
import { h3RecommendationLabel } from '../../lib/h3Memory'
import { h3PerformanceAdvice } from '../../lib/h3PerformanceAdvice'

const secondsLabel = (frames: number, fps: number) => `${(frames / fps).toFixed(1)}s`
const rowsLabel = (rows: number) => Math.round(rows / 100) * 100

/** Advisory for the largest generated pass, separate from the window capacity. */
export function H3DurationGuidance({
  frames, selectedFrames, minimumFrames, maximumFrames, continuationFrames,
}: {
  frames: number
  selectedFrames: number
  minimumFrames: number
  maximumFrames: number
  continuationFrames: number
}) {
  const options = useStore(s => s.modelOptions)
  const params = useStore(s => s.params)
  const generationMode = useStore(s => s.generationMode)
  const totalVramGb = useStore(s => s.systemStats?.gpu.vram_total_gb ?? 0)
  const totalRamGb = useStore(s => s.systemStats?.ram?.total_gb ?? 0)
  const autoEnabled = useStore(s => s.servicesConfig?.auto_performance)
  const guidance = estimateH3DurationGuidance({
    options,
    resolution: params.resolution,
    frames,
    totalVramGb,
    totalRamGb,
    references: options?.omni_reference ? params.minimax_h3_references : undefined,
    referenceDetail: params.minimax_h3_reference_detail,
    continuationFrames,
    firstLastImageCount: options?.omni_reference ? 0
      : Number(Array.isArray(params.image_start) ? params.image_start.some(Boolean) : Boolean(params.image_start))
        + Number(Array.isArray(params.image_end) ? params.image_end.some(Boolean) : Boolean(params.image_end)),
    videoGuide: Boolean(params.video_guide || (!options?.omni_reference && params.image_refs?.length)),
    audioGuide: Boolean(params.audio_guide),
    prompt: params.prompt,
  })
  const advice = h3PerformanceAdvice({
    options, resolution: params.resolution, frames, totalVramGb, totalRamGb,
    steps: params.num_inference_steps, attention: params.override_attention ?? 'auto',
    loraCount: params.activated_loras?.length ?? 0,
    referenceCount: params.minimax_h3_references?.length ?? 0,
    videoGuide: Boolean(params.video_guide), continuationFrames,
  })
  if (generationMode !== 'video' || !guidance.supported) return null

  const fps = options?.fps ?? 24
  const nativeFrames = Math.min(345, options?.frames_maximum ?? 345)
  const extendedLimit = selectedFrames > nativeFrames
  const nativePercent = Math.max(0, Math.min(100,
    (nativeFrames - minimumFrames) / Math.max(1, maximumFrames - minimumFrames) * 100))
  const fastPercent = guidance.fastMaxFrames == null ? 0 : Math.max(0, Math.min(nativePercent,
    (guidance.fastMaxFrames - minimumFrames) / Math.max(1, maximumFrames - minimumFrames) * 100))
  const pathLabel = guidance.path === 'fast' ? 'Native normalization path'
    : guidance.path === 'chunked' ? 'Chunked normalization path' : 'Normalization path uncertain'
  const pathColor = guidance.path === 'fast' ? 'text-emerald-400'
    : guidance.path === 'chunked' ? 'text-amber-400' : 'text-text-secondary'
  const basisLabel = guidance.recommendationFrames != null
    ? h3RecommendationLabel(guidance.recommendationBasis) : null
  const statusLabel = guidance.beyondRecommended ? 'Experimental duration'
    : guidance.exceedsRecommendation ? 'Above recommended starting point'
      : basisLabel ?? 'No automatic memory recommendation'
  const riskLabel = basisLabel && statusLabel !== basisLabel
    ? `${statusLabel} · ${basisLabel}` : statusLabel
  const riskColor = guidance.risk === 'high' || guidance.risk === 'caution' || guidance.exceedsRecommendation
    ? 'text-amber-400' : 'text-text-muted'
  const rangeLabel = guidance.rowsLow != null && guidance.rowsHigh != null
    ? `${rowsLabel(guidance.rowsLow).toLocaleString()}–${rowsLabel(guidance.rowsHigh).toLocaleString()} estimated packed rows`
    : guidance.rowsLow != null
      ? `About ${rowsLabel(guidance.rowsLow).toLocaleString()} packed rows before the unmeasured inputs`
      : 'Packed-row estimate unavailable for these inputs'

  return <div className="min-w-0 mt-2 space-y-2 text-[10px] leading-relaxed"
    data-testid="h3-duration-guidance" data-speed-path={guidance.path} data-memory-risk={guidance.risk}>
    <div aria-label="H3 window duration bands" className="space-y-1">
      <div className="relative h-1.5 overflow-hidden rounded-full bg-bg-tertiary" aria-hidden="true">
        <div className="absolute inset-y-0 left-0 bg-amber-400/50" style={{width: `${nativePercent}%`}} />
        {fastPercent > 0 && <div className="absolute inset-y-0 left-0 bg-emerald-400/70" style={{width: `${fastPercent}%`}} />}
        {maximumFrames > nativeFrames && <>
          <div className="absolute inset-y-0 right-0 bg-amber-400/60" style={{width: `${100 - nativePercent}%`}} />
          <div className="absolute inset-y-0 w-px bg-text-primary" style={{left: `${nativePercent}%`}} />
        </>}
      </div>
      <div className="flex flex-wrap justify-between gap-x-2 text-[9px] text-text-muted">
        <span>{guidance.fastMaxFrames != null
          ? `Native normalization estimate ≤ ${secondsLabel(guidance.fastMaxFrames, fps)}` : 'Normalization path depends on inputs'}</span>
        <span>{secondsLabel(nativeFrames, fps)} native limit</span>
        {maximumFrames > nativeFrames && <span className="text-amber-400">30s experimental</span>}
      </div>
    </div>

    <div className="rounded-lg border border-border bg-bg-tertiary/50 px-2.5 py-2 space-y-1">
      <div className={`flex flex-wrap items-center gap-x-2 gap-y-0.5 ${pathColor}`}>
        <Gauge size={12} aria-hidden="true" />
        <strong className="font-medium">{pathLabel}</strong>
        <span className="text-text-muted">for {secondsLabel(frames, fps)} per pass</span>
      </div>
      <div className={riskColor}>
        {guidance.risk === 'high' || guidance.risk === 'caution' || guidance.exceedsRecommendation
          ? <AlertTriangle size={11} className="inline mr-1" aria-hidden="true" /> : null}
        {riskLabel}{totalVramGb > 0 ? ` · ${totalVramGb.toFixed(0)} GB GPU`
          : guidance.recommendationFrames == null ? ' · VRAM unavailable' : ''}
        {guidance.recommendationFrames != null ? ` · ${secondsLabel(guidance.recommendationFrames, fps)}` : ''}
      </div>
      {advice && <div className="space-y-1 text-text-secondary" data-testid="h3-head-split-guidance"
        data-recommended-split={advice.split ?? 'unknown'} data-speed-caution={advice.slow}>
        <p>Suggested head splitting: <strong className="font-medium">{advice.label}</strong>.
          {' '}{autoEnabled === false ? 'Auto-tune is off; your saved setting stays in control.'
            : 'Auto-tune applies this when it manages head splitting.'}</p>
        {advice.slow && <p className="text-amber-400">This warning reflects dense/Sage2 tests, not an SLA or Veda sparse-attention ETA. Shorter 720p windows are recommended for dense/Sage2.</p>}
        {advice.experimental && <p className="text-amber-400">This combination needs testing; memory fit is unverified.</p>}
      </div>}
      {(extendedLimit || guidance.beyondRecommended) && <p className="text-amber-400" data-testid="h3-extended-warning">
        {guidance.beyondRecommended ? 'This pass exceeds' : 'The selected window limit exceeds'} H3’s {secondsLabel(nativeFrames, fps)} native duration envelope.
        {' '}Longer H3 durations are experimental; memory fit and render time depend on hardware and generation inputs. Shorter windows stay within the native envelope.
      </p>}
      <details>
        <summary className="cursor-pointer text-text-muted">Why this estimate?</summary>
        <div className="pt-1 space-y-1 text-text-muted">
          <p>{rangeLabel}. Sequences up to 75,000 rows use native RMSNorm; larger sequences use bounded normalization chunks. This describes the normalization path and does not predict whole-render speed.</p>
          <p>Resolution, reference media and carried history affect the workload. Use the memory recommendation as a starting point; it does not guarantee that a render will fit.</p>
          {guidance.recommendationNote && <p data-testid="h3-recommendation-note">{guidance.recommendationNote}</p>}
          {guidance.recommendationVersion != null && <p>Guidance updated {guidance.recommendationVersion}.</p>}
          {guidance.notes.map((note, index) => <p key={index}>{note}</p>)}
          {advice?.notes.map((note, index) => <p key={`performance-${index}`}>{note}</p>)}
          <p>Step count, attention choice and checkpoint offloading also affect render time. The generation log reports actual packed rows; full-render speed has not been calibrated across GPUs.</p>
        </div>
      </details>
    </div>
  </div>
}
