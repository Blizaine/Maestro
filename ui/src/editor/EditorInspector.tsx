import { SlidersHorizontal, Trash2 } from 'lucide-react'
import type { EditorTimelineItem } from '../types'
import { editorCanvasLabel } from './editorUtils'
import { useEditorStore } from './useEditorStore'

function FieldLabel({ children }: { children: React.ReactNode }) {
  return <label className="text-[9px] font-medium uppercase tracking-wider text-text-muted">{children}</label>
}

function NumberField({
  label,
  value,
  min,
  max,
  step = 0.1,
  onChange,
}: {
  label: string
  value: number
  min?: number
  max?: number
  step?: number
  onChange: (value: number) => void
}) {
  return (
    <div className="space-y-1">
      <FieldLabel>{label}</FieldLabel>
      <input
        type="number"
        value={Number.isFinite(value) ? Number(value.toFixed(3)) : 0}
        min={min}
        max={max}
        step={step}
        onChange={event => onChange(Number(event.target.value))}
        className="w-full rounded-md border border-border bg-bg-tertiary px-2 py-1.5 text-[10px] text-text-primary outline-none focus:border-accent-blue/60"
      />
    </div>
  )
}

function SliderField({
  label,
  value,
  min,
  max,
  step,
  onChange,
}: {
  label: string
  value: number
  min: number
  max: number
  step: number
  onChange: (value: number) => void
}) {
  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between">
        <FieldLabel>{label}</FieldLabel>
        <span className="font-mono text-[9px] text-text-secondary">{value.toFixed(step < 0.1 ? 2 : 1)}</span>
      </div>
      <input type="range" min={min} max={max} step={step} value={value} onChange={event => onChange(Number(event.target.value))} className="w-full accent-[var(--color-accent-blue)]" />
    </div>
  )
}

export function EditorInspector({ compact = false }: { compact?: boolean }) {
  const project = useEditorStore(state => state.project)
  const selectedItemId = useEditorStore(state => state.selectedItemId)
  const updateItem = useEditorStore(state => state.updateItem)
  const deleteSelected = useEditorStore(state => state.deleteSelected)
  const setCanvas = useEditorStore(state => state.setCanvas)
  const setExportSettings = useEditorStore(state => state.setExportSettings)

  const location = project && selectedItemId
    ? project.tracks.flatMap(track => track.items.map(item => ({ track, item }))).find(entry => entry.item.id === selectedItemId)
    : null
  const item = location?.item
  const track = location?.track
  const asset = item?.asset_id ? project?.assets[item.asset_id] : null

  const patchTransform = (key: keyof EditorTimelineItem['transform'], value: number) => {
    if (!item) return
    updateItem(item.id, { transform: { ...item.transform, [key]: value } })
  }
  const patchStyle = (key: 'x' | 'y' | 'font_size' | 'color', value: number | string) => {
    if (!item) return
    updateItem(item.id, {
      style: {
        x: item.style?.x ?? 0,
        y: item.style?.y ?? 0,
        font_size: item.style?.font_size ?? 64,
        color: item.style?.color ?? '#ffffff',
        [key]: value,
      },
    })
  }

  if (!project) return null

  return (
    <aside className={`min-h-0 overflow-y-auto bg-bg-secondary ${compact ? 'h-full' : 'w-[270px] shrink-0 border-l border-border'}`}>
      <div className="sticky top-0 z-10 flex h-10 items-center gap-2 border-b border-border bg-bg-secondary/95 px-3 backdrop-blur">
        <SlidersHorizontal size={12} className="text-accent-blue" />
        <h2 className="text-[10px] font-semibold uppercase tracking-[0.12em] text-text-secondary">{item ? 'Clip inspector' : 'Project inspector'}</h2>
      </div>

      {item && track ? (
        <div className="space-y-4 p-3">
          <div>
            <FieldLabel>Name</FieldLabel>
            <input
              value={item.name}
              onChange={event => updateItem(item.id, { name: event.target.value.slice(0, 140) })}
              className="mt-1 w-full rounded-md border border-border bg-bg-tertiary px-2 py-1.5 text-[10px] text-text-primary outline-none focus:border-accent-blue/60"
            />
            <p className="mt-1 truncate text-[8px] text-text-muted">{track.name}{asset ? ` · ${asset.width || '—'}×${asset.height || '—'}` : ''}</p>
          </div>

          {track.type === 'text' && (
            <div className="space-y-2">
              <FieldLabel>Text</FieldLabel>
              <textarea
                value={item.text || ''}
                onChange={event => updateItem(item.id, { text: event.target.value })}
                rows={3}
                className="w-full resize-none rounded-md border border-border bg-bg-tertiary px-2 py-1.5 text-[10px] text-text-primary outline-none focus:border-accent-blue/60"
              />
              <div className="grid grid-cols-2 gap-2">
                <NumberField label="Font size" value={item.style?.font_size ?? 64} min={8} max={400} step={1} onChange={value => patchStyle('font_size', value)} />
                <div className="space-y-1">
                  <FieldLabel>Color</FieldLabel>
                  <input type="color" value={item.style?.color || '#ffffff'} onChange={event => patchStyle('color', event.target.value)} className="h-[30px] w-full rounded-md border border-border bg-bg-tertiary p-1" />
                </div>
                <NumberField label="Text X" value={item.style?.x ?? 0} step={1} onChange={value => patchStyle('x', value)} />
                <NumberField label="Text Y" value={item.style?.y ?? 0} step={1} onChange={value => patchStyle('y', value)} />
              </div>
            </div>
          )}

          <div className="space-y-2">
            <FieldLabel>Timing</FieldLabel>
            <div className="grid grid-cols-2 gap-2">
              <NumberField label="Start" value={item.start} min={0} step={1 / project.canvas.fps} onChange={value => updateItem(item.id, { start: value })} />
              <NumberField label="Duration" value={item.duration} min={1 / project.canvas.fps} step={1 / project.canvas.fps} onChange={value => updateItem(item.id, { duration: value })} />
              {track.type !== 'text' && <NumberField label="Source in" value={item.source_in} min={0} step={1 / project.canvas.fps} onChange={value => updateItem(item.id, { source_in: value })} />}
              {track.type !== 'text' && <NumberField label="Speed" value={item.speed} min={0.1} max={8} step={0.05} onChange={value => updateItem(item.id, { speed: value })} />}
            </div>
          </div>

          {track.type !== 'audio' && (
            <div className="space-y-3">
              <FieldLabel>Transform</FieldLabel>
              <div className="grid grid-cols-2 gap-2">
                <NumberField label="X" value={item.transform.x} step={1} onChange={value => patchTransform('x', value)} />
                <NumberField label="Y" value={item.transform.y} step={1} onChange={value => patchTransform('y', value)} />
              </div>
              <SliderField label="Scale" value={item.transform.scale} min={0.1} max={3} step={0.01} onChange={value => patchTransform('scale', value)} />
              <SliderField label="Rotation" value={item.transform.rotation} min={-180} max={180} step={1} onChange={value => patchTransform('rotation', value)} />
              {track.type === 'video' && (
                <div className="grid grid-cols-2 gap-1 rounded-lg bg-bg-tertiary p-1">
                  {(['contain', 'cover'] as const).map(fit => (
                    <button key={fit} type="button" onClick={() => updateItem(item.id, { fit })} className={`rounded-md py-1 text-[9px] capitalize ${item.fit === fit ? 'bg-bg-active text-text-primary' : 'text-text-muted hover:text-text-secondary'}`}>{fit}</button>
                  ))}
                </div>
              )}
            </div>
          )}

          <div className="space-y-3">
            {track.type !== 'text' && <SliderField label="Volume" value={item.volume} min={0} max={2} step={0.01} onChange={value => updateItem(item.id, { volume: value })} />}
            {track.type !== 'audio' && <SliderField label="Opacity" value={item.opacity} min={0} max={1} step={0.01} onChange={value => updateItem(item.id, { opacity: value })} />}
          </div>

          <button type="button" onClick={deleteSelected} className="flex w-full items-center justify-center gap-1.5 rounded-lg border border-red-500/20 bg-red-500/5 py-2 text-[10px] text-red-400 hover:bg-red-500/10">
            <Trash2 size={11} /> Remove clip
          </button>
        </div>
      ) : (
        <div className="space-y-4 p-3">
          <div className="rounded-lg border border-border bg-bg-tertiary p-2.5">
            <div className="text-[10px] font-medium text-text-primary">{editorCanvasLabel(project.canvas.width, project.canvas.height)} canvas</div>
            <div className="mt-0.5 text-[8px] text-text-muted">{project.canvas.width} × {project.canvas.height} · {project.canvas.fps} fps</div>
          </div>
          <div className="space-y-2">
            <FieldLabel>Canvas</FieldLabel>
            <div className="grid grid-cols-2 gap-2">
              <NumberField label="Width" value={project.canvas.width} min={64} max={7680} step={2} onChange={width => setCanvas({ width: Math.round(width / 2) * 2 })} />
              <NumberField label="Height" value={project.canvas.height} min={64} max={4320} step={2} onChange={height => setCanvas({ height: Math.round(height / 2) * 2 })} />
              <NumberField label="FPS" value={project.canvas.fps} min={1} max={120} step={1} onChange={fps => setCanvas({ fps })} />
              <div className="space-y-1">
                <FieldLabel>Background</FieldLabel>
                <input type="color" value={project.canvas.background} onChange={event => setCanvas({ background: event.target.value })} className="h-[30px] w-full rounded-md border border-border bg-bg-tertiary p-1" />
              </div>
            </div>
          </div>
          <div className="space-y-2">
            <FieldLabel>Export</FieldLabel>
            <select value={project.export.quality} onChange={event => setExportSettings({ quality: event.target.value as 'draft' | 'balanced' | 'high' })} className="w-full rounded-md border border-border bg-bg-tertiary px-2 py-1.5 text-[10px] text-text-primary outline-none">
              <option value="draft">Draft · fastest</option>
              <option value="balanced">Balanced</option>
              <option value="high">High quality</option>
            </select>
            <label className="flex items-center justify-between rounded-md border border-border bg-bg-tertiary px-2 py-2 text-[9px] text-text-secondary">
              Include audio
              <input type="checkbox" checked={project.export.include_audio} onChange={event => setExportSettings({ include_audio: event.target.checked })} className="accent-[var(--color-accent-blue)]" />
            </label>
          </div>
          <p className="text-[9px] leading-relaxed text-text-muted">Select a clip to edit timing, framing, speed, volume, opacity, and text. All edits remain non-destructive until export.</p>
        </div>
      )}
    </aside>
  )
}
