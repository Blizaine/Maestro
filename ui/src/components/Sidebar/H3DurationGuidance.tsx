import { AlertTriangle, Gauge } from 'lucide-react'
import { useStore } from '../../stores/useStore'
import { estimateH3DurationGuidance } from '../../lib/h3DurationGuidance'

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
  const guidance = estimateH3DurationGuidance({
    options,
    resolution: params.resolution,
    frames,
    totalVramGb,
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
  if (generationMode !== 'video' || !guidance.supported) return null

  const fps = options?.fps ?? 24
  const nativeFrames = Math.min(345, options?.frames_maximum ?? 345)
  const extendedLimit = selectedFrames > nativeFrames
  const nativePercent = Math.max(0, Math.min(100,
    (nativeFrames - minimumFrames) / Math.max(1, maximumFrames - minimumFrames) * 100))
  const fastPercent = guidance.fastMaxFrames == null ? 0 : Math.max(0, Math.min(nativePercent,
    (guidance.fastMaxFrames - minimumFrames) / Math.max(1, maximumFrames - minimumFrames) * 100))
  const pathLabel = guidance.path === 'fast' ? 'Faster path expected'
    : guidance.path === 'chunked' ? 'Slower path expected' : 'Speed path uncertain'
  const pathColor = guidance.path === 'fast' ? 'text-emerald-400'
    : guidance.path === 'chunked' ? 'text-amber-400' : 'text-text-secondary'
  const riskLabel = guidance.risk === 'high' ? 'Potential out of memory'
    : guidance.risk === 'caution' ? 'Memory caution'
      : guidance.risk === 'within-profile' ? 'Within GPU recommendation' : 'VRAM assessment unavailable'
  const riskColor = guidance.risk === 'high' ? 'text-red-400'
    : guidance.risk === 'caution' ? 'text-amber-400' : 'text-text-muted'
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
          <div className="absolute inset-y-0 right-0 bg-red-400/60" style={{width: `${100 - nativePercent}%`}} />
          <div className="absolute inset-y-0 w-px bg-text-primary" style={{left: `${nativePercent}%`}} />
        </>}
      </div>
      <div className="flex flex-wrap justify-between gap-x-2 text-[9px] text-text-muted">
        <span>{guidance.fastMaxFrames != null
          ? `Faster path estimate ≤ ${secondsLabel(guidance.fastMaxFrames, fps)}` : 'Speed estimate depends on inputs'}</span>
        <span>{secondsLabel(nativeFrames, fps)} recommended limit</span>
        {maximumFrames > nativeFrames && <span className="text-red-400">30s experimental</span>}
      </div>
    </div>

    <div className="rounded-lg border border-border bg-bg-tertiary/50 px-2.5 py-2 space-y-1">
      <div className={`flex flex-wrap items-center gap-x-2 gap-y-0.5 ${pathColor}`}>
        <Gauge size={12} aria-hidden="true" />
        <strong className="font-medium">{pathLabel}</strong>
        <span className="text-text-muted">for {secondsLabel(frames, fps)} per pass</span>
      </div>
      <div className={riskColor}>
        {guidance.risk === 'high' || guidance.risk === 'caution'
          ? <AlertTriangle size={11} className="inline mr-1" aria-hidden="true" /> : null}
        {riskLabel}{totalVramGb > 0 ? ` · ${totalVramGb.toFixed(0)} GB GPU` : ''}
        {guidance.recommendationFrames != null ? ` · up to ${secondsLabel(guidance.recommendationFrames, fps)}` : ''}
      </div>
      {extendedLimit && <p className="text-amber-400" data-testid="h3-extended-warning">
        {guidance.beyondRecommended ? 'This pass exceeds' : 'The selected window limit exceeds'} H3’s {secondsLabel(nativeFrames, fps)} recommended duration.
        {' '}{guidance.beyondRecommended ? 'It' : 'A pass using that extra length'} may be much slower or run out of memory. Shorter windows are recommended.
      </p>}
      <details>
        <summary className="cursor-pointer text-text-muted">Why this estimate?</summary>
        <div className="pt-1 space-y-1 text-text-muted">
          <p>{rangeLabel}. Up to 75,000 rows uses the faster normalization path; larger sequences use memory-bounded chunks. This describes one part of generation, and does not guarantee the fastest render.</p>
          <p>Resolution, reference media and carried history affect the workload. The GPU recommendation uses this checkpoint’s VRAM profile; it is an estimate, not a guarantee that the job will fit.</p>
          {guidance.notes.map((note, index) => <p key={index}>{note}</p>)}
          <p>Step count, attention choice and checkpoint offloading also affect render time. Speed estimates assume the default chunk settings. The generation log reports the actual packed rows. Extended-window slowdown has not been calibrated across GPUs.</p>
        </div>
      </details>
    </div>
  </div>
}
