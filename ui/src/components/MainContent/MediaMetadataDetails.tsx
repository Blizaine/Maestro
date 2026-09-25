import type { ReactNode } from 'react'
import type { OutputFile, OutputMetadata, OutputMediaInfo, OutputProcessingInfo } from '../../types'
import { formatBytes, formatGenerationDuration } from '../../lib/format'
import { getMediaTimestamp } from '../../lib/mediaTimestamp'

function resolution(info?: OutputMediaInfo) {
  return info?.width && info?.height ? `${info.width} × ${info.height}` : null
}

function finite(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value)
}

function basename(value: string) {
  return value.replace(/\\/g, '/').split('/').filter(Boolean).pop() || value
}

function legacyProcessing(metadata?: OutputMetadata | null): OutputProcessingInfo | undefined {
  if (!metadata) return undefined
  const params = metadata.params || {}
  const isUpscale = metadata.tool === 'upscale' || params.edit_sub_mode === 'upscale'
  if (!isUpscale) return undefined
  const method = [params.method, params.spatial_upsampling, params.upscale_method]
    .find(value => typeof value === 'string' && value.length > 0) as string | undefined
  const multiplier = [params.multiplier, params.scale, params.upscale_multiplier]
    .find(finite)
  return {
    method,
    method_label: method ? legacyMethodLabel(method) : 'Upscale (legacy record)',
    multiplier,
    temporal_method: typeof params.temporal_upsampling === 'string' ? params.temporal_upsampling : undefined,
    source_name: metadata.tool_source || (typeof params.media_path === 'string' ? params.media_path : undefined),
  }
}

function legacyMethodLabel(method: string) {
  const normalized = method.trim().toLowerCase()
  const dlss = /^dlss5\*(\d+(?:\.\d+)?)$/.exec(normalized)
  if (dlss) return `DLSS 5 Neural Rendering ${dlss[1]}×`
  const flashTwoPass = /^flashvsr\d+pass(\d+)$/.exec(normalized)
  if (flashTwoPass) return `FlashVSR two-pass ${flashTwoPass[1]}×`
  const flash = /^flashvsr(\d+)$/.exec(normalized)
  if (flash) return `FlashVSR ${flash[1]}×`
  const lanczos = /^lanczos(\d+(?:\.\d+)?)$/.exec(normalized)
  if (lanczos) return `Lanczos ${lanczos[1]}×`
  return method
}

function DefinitionRow({ label, children }: { label: string; children: ReactNode }) {
  return <><dt className="text-text-muted">{label}</dt><dd className="min-w-0 break-words text-text-secondary">{children}</dd></>
}

function mediaRows(info: OutputMediaInfo | undefined, listedSize: number) {
  if (!info) return null
  const sizeBytes = finite(info.size_bytes) ? info.size_bytes : listedSize
  const dimensions = resolution(info)
  return <>
    {dimensions && <DefinitionRow label="Resolution">{dimensions}</DefinitionRow>}
    {finite(info.fps) && <DefinitionRow label="Frame rate">{info.fps} fps</DefinitionRow>}
    {finite(info.frames) && <DefinitionRow label="Frames">{info.frames.toLocaleString()}</DefinitionRow>}
    {finite(info.duration_seconds) && <DefinitionRow label="Duration">{formatGenerationDuration(info.duration_seconds)}</DefinitionRow>}
    {finite(sizeBytes) && <DefinitionRow label="File size">{formatBytes(sizeBytes)}</DefinitionRow>}
  </>
}

const processingOptionLabels: Record<string, string> = {
  dlss_intensity: 'Enhancement strength',
  dlss_depth: 'Depth setting',
  dlss_motion: 'Motion setting',
}

function ProcessingDetails({ processing, metadata }: { processing?: OutputProcessingInfo; metadata?: OutputMetadata | null }) {
  const legacy = !processing ? legacyProcessing(metadata) : undefined
  const details = processing || legacy
  if (!details) return null
  const method = details.method_label || (details.method ? legacyMethodLabel(details.method) : 'Processing')
  const inputResolution = resolution(details.input)
  const outputResolution = resolution(details.output)
  const inputFps = finite(details.input?.fps) ? details.input!.fps : undefined
  const outputFps = finite(details.output?.fps) ? details.output!.fps : undefined
  const sourceName = details.source_name
  const completed = finite(details.completed_at) ? getMediaTimestamp({ value: details.completed_at, kind: 'processed' }) : null
  const options = details.options && typeof details.options === 'object' && !Array.isArray(details.options)
    ? details.options
    : null
  const optionEntries = options ? Object.entries(options) : []

  return <section className="mt-3 rounded-lg border border-accent-blue/20 bg-accent-blue/5 p-2.5">
    <div className="mb-2 text-[10px] font-medium uppercase tracking-wide text-text-muted">Processing</div>
    <dl className="grid grid-cols-[auto_minmax(0,1fr)] gap-x-3 gap-y-1 text-[11px]">
      <DefinitionRow label="Method">{method}</DefinitionRow>
      {finite(details.multiplier) && <DefinitionRow label="Spatial scale">{details.multiplier}×</DefinitionRow>}
      {(inputResolution || outputResolution) && <DefinitionRow label="Resolution change">
        {inputResolution || 'Unknown'} → {outputResolution || 'Unknown'}
      </DefinitionRow>}
      {(inputFps != null || outputFps != null) && <DefinitionRow label="Frame rate change">
        {inputFps != null ? `${inputFps} fps` : 'Unknown'} → {outputFps != null ? `${outputFps} fps` : 'Unknown'}
      </DefinitionRow>}
      {(details.temporal_label || details.temporal_method) && <DefinitionRow label="Temporal method">
        {details.temporal_label || details.temporal_method}
      </DefinitionRow>}
      {finite(details.frame_multiplier) && <DefinitionRow label="Frame multiplier">{details.frame_multiplier}×</DefinitionRow>}
      {sourceName && <DefinitionRow label="Source"><span title={sourceName}>{basename(sourceName)}</span></DefinitionRow>}
      {finite(details.elapsed_seconds) && <DefinitionRow label="Processing time">{formatGenerationDuration(details.elapsed_seconds)}</DefinitionRow>}
      {completed && <DefinitionRow label="Completed"><span title={completed.exact}>{completed.exact}</span></DefinitionRow>}
    </dl>
    {optionEntries.length > 0 && <details className="mt-2 text-[10px] text-text-muted">
      <summary className="cursor-pointer select-none">Processing options</summary>
      <dl className="mt-1 grid grid-cols-[auto_minmax(0,1fr)] gap-x-3 gap-y-1">
        {optionEntries.map(([key, value]) => <DefinitionRow key={key} label={processingOptionLabels[key] || key}>
          {typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean'
            ? String(value)
            : JSON.stringify(value)}
        </DefinitionRow>)}
      </dl>
    </details>}
  </section>
}

export function MediaMetadataDetails({ file, metadata }: { file: OutputFile; metadata?: OutputMetadata | null }) {
  const timestamp = getMediaTimestamp(metadata?.timestamp, file.created_at)
  const mediaInfo = metadata?.media_info
  const hasProcessing = !!metadata?.processing || !!legacyProcessing(metadata)

  return <section className="rounded-lg border border-border bg-bg-tertiary/70 p-2.5">
    <div className="mb-2 text-[10px] font-medium uppercase tracking-wide text-text-muted">Media details</div>
    <dl className="grid grid-cols-[auto_minmax(0,1fr)] gap-x-3 gap-y-1 text-[11px]">
      <DefinitionRow label="File"><span title={file.name}>{file.name}</span></DefinitionRow>
      <DefinitionRow label="Type">{file.type}</DefinitionRow>
      {timestamp && <DefinitionRow label={timestamp.label}>
        <span title={timestamp.exact}>{timestamp.exact}</span>
      </DefinitionRow>}
      {mediaRows(mediaInfo, file.size)}
      {!mediaInfo && finite(file.size) && <DefinitionRow label="File size">{formatBytes(file.size)}</DefinitionRow>}
    </dl>
    {hasProcessing && <ProcessingDetails processing={metadata?.processing} metadata={metadata} />}
  </section>
}
