import { useEffect, useState } from 'react'

export const dlssSpatialOptions = [1, 1.5, 1.724, 2, 3].map(scale => ({
  value: `dlss5*${scale}`, label: `DLSS 5 ${scale}x${scale === 1 ? ' (native refinement)' : ''}`,
}))

type Capabilities = {
  neural_rendering: { available: boolean; reason: string }
  frame_generation: { available: boolean; reason: string; factors: number[] }
}

export function MediaFinishingControls({ spatial, temporal, onTemporal, options, onOptions, image = false }: {
  spatial: string; temporal: string; onTemporal: (value: string) => void
  options: Record<string, unknown>; onOptions: (value: Record<string, unknown>) => void; image?: boolean
}) {
  const [capabilities, setCapabilities] = useState<Capabilities | null>(null)
  const [error, setError] = useState('')
  const refresh = (force = false) => {
    setError('')
    fetch(`/api/v1/media-flow/capabilities?refresh=${force}`)
      .then(async response => { if (!response.ok) throw new Error('Could not check media capabilities'); return response.json() })
      .then(setCapabilities).catch(e => setError(String(e.message)))
  }
  useEffect(() => { refresh() }, [])
  const hasNeural = spatial.startsWith('dlss5*')
  const hasFrameGen = temporal.startsWith('dlssg*') && !image
  const selectClass = 'w-full bg-bg-tertiary border border-border rounded-lg px-2 py-2 text-xs text-text-primary'
  const patch = (key: string, value: unknown) => onOptions({ ...options, [key]: value })
  return <div className="space-y-3 text-xs">
    {!image && <label className="block space-y-1">
      <span className="text-text-muted">Temporal Upsampling</span>
      <select value={temporal} onChange={e => onTemporal(e.target.value)} className={selectClass}>
        <option value="">Original frame rate</option>
        {[2, 3, 4].map(n => <option key={`rife${n}`} value={`rife${n}`}>RIFE 4.26 ×{n}</option>)}
        {[2, 3, 4, 5, 6].map(n => <option key={`dlssg${n}`} value={`dlssg*${n}`}
          disabled={!capabilities?.frame_generation.factors.includes(n)}>DLSS Frame Generation ×{n}{n > 4 ? ' (RTX 50)' : ''}</option>)}
      </select>
    </label>}
    {(hasNeural || hasFrameGen) && <div className="space-y-2">
      {hasNeural && <>
        <label className="block">Neural Rendering intensity: {Number(options.dlss_intensity ?? 1).toFixed(2)}
          <input className="w-full" type="range" min={0} max={2} step={0.05} value={Number(options.dlss_intensity ?? 1)} onChange={e => patch('dlss_intensity', Number(e.target.value))} />
        </label>
        <label className="block">Depth precision
          <select className={selectClass} value={String(options.dlss_depth ?? 'half')} onChange={e => patch('dlss_depth', e.target.value)}>
            <option value="quarter">Quarter resolution (faster)</option><option value="half">Half resolution</option><option value="full">Full resolution</option>
          </select>
        </label>
      </>}
      <label className="block">Motion estimation
        <select className={selectClass} value={String(options.dlss_motion ?? 'original')} onChange={e => patch('dlss_motion', e.target.value)}>
          <option value="original">OpenCV DIS (faster)</option><option value="raft">RAFT (more detail)</option>
        </select>
      </label>
      <p className="text-text-muted text-[11px]">Depth and motion are estimated from your footage. Results depend on the content.</p>
      {hasNeural && capabilities && !capabilities.neural_rendering.available && <p className="text-indicator-warning">DLSS Neural Rendering: {capabilities.neural_rendering.reason}. See docs/DLSS5.md in the Maestro project.</p>}
      {hasFrameGen && capabilities && !capabilities.frame_generation.available && <p className="text-indicator-warning">DLSS Frame Generation: {capabilities.frame_generation.reason}.</p>}
    </div>}
    {error && <p className="text-indicator-warning">{error}</p>}
    <button type="button" className="text-text-muted underline text-[10px]" onClick={() => refresh(true)}>Refresh DLSS availability</button>
  </div>
}
