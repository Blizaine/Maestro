import { ArrowDown, ArrowUp, ChevronDown, ImagePlus, Plus, Square, Trash2, Type } from 'lucide-react'
import { useState } from 'react'
import type { EditorAsset, EditorGraphicElement, EditorTimelineItem } from '../types'
import { DEFAULT_EDITOR_FONT, EDITOR_FONT_OPTIONS } from './editorFonts'

const fieldClass = 'w-full rounded-md border border-border bg-bg-secondary px-2 py-2 text-[11px] text-text-primary outline-none focus:border-accent-blue/60 disabled:opacity-50'
const colorClass = 'h-9 w-full rounded-md border border-border bg-bg-secondary p-1 disabled:opacity-50'

function FieldLabel({ children }: { children: React.ReactNode }) {
  return <div className="text-[9px] font-medium uppercase tracking-wider text-text-muted">{children}</div>
}

function GraphicNumberField({
  element,
  property,
  value,
  disabled,
  onChange,
}: {
  element: EditorGraphicElement
  property: 'x' | 'y' | 'width' | 'height'
  value: number
  disabled: boolean
  onChange: (value: number) => void
}) {
  const label = `${element.type} ${element.id.slice(-4)} ${property.toUpperCase()}`
  return (
    <label className="space-y-1">
      <FieldLabel>{property}</FieldLabel>
      <input
        type="number"
        min={property === 'width' || property === 'height' ? 1 : undefined}
        step={1}
        value={Number.isFinite(value) ? value : 0}
        aria-label={label}
        className={fieldClass}
        disabled={disabled}
        onChange={event => {
          const next = Number(event.target.value)
          if (Number.isFinite(next) && ((property !== 'width' && property !== 'height') || next > 0)) onChange(next)
        }}
      />
    </label>
  )
}

function GraphicOpacity({ value, disabled, label, onChange }: {
  value: number
  disabled: boolean
  label: string
  onChange: (value: number) => void
}) {
  return (
    <label className="block space-y-1">
      <span className="flex items-center justify-between gap-2"><FieldLabel>{label}</FieldLabel><span className="font-mono text-[9px] text-text-muted">{Math.round(value * 100)}%</span></span>
      <input type="range" min={0} max={1} step={0.01} value={value} aria-label={label} disabled={disabled}
        onChange={event => onChange(Number(event.target.value))} className="w-full accent-[var(--color-accent-blue)]" />
    </label>
  )
}

function newElementId(): string {
  return `graphic-el-${globalThis.crypto?.randomUUID?.() || `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 9)}`}`
}

export function EditorGraphicControls({
  item,
  assets,
  disabled,
  onUpdate,
}: {
  item: EditorTimelineItem
  assets: Record<string, EditorAsset>
  disabled: boolean
  onUpdate: (patch: Partial<EditorTimelineItem>) => void
}) {
  const graphic = item.graphic
  const [expandedElementIds, setExpandedElementIds] = useState<Set<string>>(() => new Set(graphic?.elements.slice(0, 1).map(element => element.id) || []))
  if (!graphic) return null
  const imageAssets = Object.values(assets).filter(asset => asset.type === 'image')
  const patchElement = (id: string, patch: Partial<EditorGraphicElement>) => {
    const elements = graphic.elements.map(element => (
      element.id === id ? { ...element, ...patch } as EditorGraphicElement : element
    ))
    onUpdate({ graphic: { ...graphic, elements } })
  }
  const addElement = (element: EditorGraphicElement) => {
    onUpdate({ graphic: { ...graphic, elements: [...graphic.elements, element] } })
    setExpandedElementIds(current => new Set(current).add(element.id))
  }
  const moveElement = (index: number, delta: -1 | 1) => {
    const target = index + delta
    if (target < 0 || target >= graphic.elements.length) return
    const elements = [...graphic.elements]
    ;[elements[index], elements[target]] = [elements[target], elements[index]]
    onUpdate({ graphic: { ...graphic, elements } })
  }

  return (
    <section className="space-y-3 rounded-xl border border-border bg-bg-tertiary p-3" aria-label="Graphic overlay elements">
      <div>
        <div className="text-[11px] font-semibold text-text-primary">Graphic overlay</div>
        <p className="mt-1 text-[9px] leading-relaxed text-text-muted">Build this composition from editable text, panels, and project images. Coordinates use a 1344 × 768 design canvas and scale to the project. Drag the selected group in the preview or set its position above.</p>
      </div>

      <div className="grid grid-cols-3 gap-1">
        <button type="button" disabled={disabled || graphic.elements.length >= 40} onClick={() => addElement({
          id: newElementId(), type: 'text', x: 128, y: 128, width: 680, height: 108,
          text: 'New text', font_family: DEFAULT_EDITOR_FONT, font_size: 48, color: '#ffffff', opacity: 1, align: 'left', bold: false,
        })} className="flex items-center justify-center gap-1 rounded-md border border-border bg-bg-secondary px-1 py-2 text-[9px] text-text-secondary hover:bg-bg-hover disabled:opacity-40">
          <Type size={11} /> Text
        </button>
        <button type="button" disabled={disabled || graphic.elements.length >= 40} onClick={() => addElement({
          id: newElementId(), type: 'rect', x: 88, y: 88, width: 500, height: 300,
          fill: '#101827', opacity: 0.9, border_color: '#ffffff', border_width: 0, radius: 16,
        })} className="flex items-center justify-center gap-1 rounded-md border border-border bg-bg-secondary px-1 py-2 text-[9px] text-text-secondary hover:bg-bg-hover disabled:opacity-40">
          <Square size={11} /> Panel
        </button>
        <button type="button" disabled={disabled || graphic.elements.length >= 40 || imageAssets.length === 0} title={imageAssets.length ? 'Add a project image' : 'Add an image to the project first'} onClick={() => {
          const firstImage = imageAssets[0]
          if (!firstImage) return
          addElement({ id: newElementId(), type: 'image', x: 762, y: 194, width: 470, height: 380, asset_id: firstImage.id, opacity: 1, fit: 'contain' })
        }} className="flex items-center justify-center gap-1 rounded-md border border-border bg-bg-secondary px-1 py-2 text-[9px] text-text-secondary hover:bg-bg-hover disabled:opacity-40">
          <ImagePlus size={11} /> Image
        </button>
      </div>
      {imageAssets.length === 0 && <p className="-mt-2 text-[9px] text-text-muted">Add an image to the project to use it in a graphic overlay.</p>}
      {graphic.elements.length >= 40 && <p className="-mt-2 text-[9px] text-accent-warm">This overlay has reached the 40 element limit.</p>}

      <div className="space-y-2">
        {graphic.elements.map((element, index) => (
          <article key={element.id} className="space-y-2 rounded-lg border border-border bg-bg-secondary p-2.5" aria-label={`${element.type} element ${index + 1}`}>
            <div className="flex items-center gap-1.5">
              <button type="button" aria-label={`${expandedElementIds.has(element.id) ? 'Collapse' : 'Expand'} layer ${index + 1}`} aria-expanded={expandedElementIds.has(element.id)}
                onClick={() => setExpandedElementIds(current => {
                  const next = new Set(current)
                  if (next.has(element.id)) next.delete(element.id)
                  else next.add(element.id)
                  return next
                })} className="rounded p-1 text-text-muted hover:bg-bg-hover">
                <ChevronDown size={12} className={expandedElementIds.has(element.id) ? '' : '-rotate-90'} />
              </button>
              <span className="min-w-0 flex-1 truncate text-[10px] font-medium text-text-primary" title={element.type === 'text' ? element.text : undefined}>
                Layer {index + 1} · {element.type === 'text'
                  ? `Text · ${element.text.trim().split('\n')[0] || 'Empty'}`
                  : element.type === 'rect' ? 'Panel' : 'Image'}
              </span>
              <button type="button" aria-label={`Move element ${index + 1} up`} title="Move forward one layer" disabled={disabled || index === 0}
                onClick={() => moveElement(index, -1)} className="rounded p-1 text-text-muted hover:bg-bg-hover disabled:opacity-30"><ArrowUp size={12} /></button>
              <button type="button" aria-label={`Move element ${index + 1} down`} title="Move back one layer" disabled={disabled || index === graphic.elements.length - 1}
                onClick={() => moveElement(index, 1)} className="rounded p-1 text-text-muted hover:bg-bg-hover disabled:opacity-30"><ArrowDown size={12} /></button>
              <button type="button" aria-label={`Remove element ${index + 1}`} title="Remove element" disabled={disabled}
                onClick={() => onUpdate({ graphic: { ...graphic, elements: graphic.elements.filter(candidate => candidate.id !== element.id) } })}
                className="rounded p-1 text-text-muted hover:bg-red-500/10 hover:text-red-300 disabled:opacity-30"><Trash2 size={12} /></button>
            </div>

            {expandedElementIds.has(element.id) && <>
            <div className="grid grid-cols-4 gap-1.5">
              {(['x', 'y', 'width', 'height'] as const).map(property => (
                <GraphicNumberField key={property} element={element} property={property} value={element[property]} disabled={disabled}
                  onChange={value => patchElement(element.id, { [property]: value })} />
              ))}
            </div>

            {element.type === 'text' && (
              <div className="space-y-2">
                <label className="block space-y-1"><FieldLabel>Text</FieldLabel>
                  <textarea value={element.text} disabled={disabled} aria-label={`Text element ${index + 1} content`} rows={2}
                    onChange={event => patchElement(element.id, { text: event.target.value })} className={`${fieldClass} resize-y`} />
                </label>
                <label className="block space-y-1"><FieldLabel>Font</FieldLabel>
                  <select value={element.font_family} disabled={disabled} aria-label={`Text element ${index + 1} font`}
                    onChange={event => patchElement(element.id, { font_family: event.target.value })} className={fieldClass}>
                    {EDITOR_FONT_OPTIONS.map(option => <option key={option.value} value={option.value}>{option.label}</option>)}
                  </select>
                </label>
                <div className="grid grid-cols-2 gap-2">
                  <label className="space-y-1"><FieldLabel>Font size</FieldLabel>
                    <input type="number" min={1} max={500} step={1} value={element.font_size} disabled={disabled} aria-label={`Text element ${index + 1} font size`}
                      onChange={event => { const value = Number(event.target.value); if (Number.isFinite(value) && value > 0) patchElement(element.id, { font_size: value }) }} className={fieldClass} />
                  </label>
                  <label className="space-y-1"><FieldLabel>Color</FieldLabel>
                    <input type="color" value={element.color} disabled={disabled} aria-label={`Text element ${index + 1} color`}
                      onChange={event => patchElement(element.id, { color: event.target.value })} className={colorClass} />
                  </label>
                </div>
                <label className="block space-y-1"><FieldLabel>Alignment</FieldLabel>
                  <select value={element.align || 'left'} disabled={disabled} aria-label={`Text element ${index + 1} alignment`}
                    onChange={event => patchElement(element.id, { align: event.target.value as 'left' | 'center' | 'right' })} className={fieldClass}>
                    <option value="left">Left</option><option value="center">Center</option><option value="right">Right</option>
                  </select>
                </label>
                <label className="flex items-center justify-between rounded-md border border-border px-2 py-2 text-[10px] text-text-secondary">
                  Bold <input type="checkbox" checked={Boolean(element.bold)} disabled={disabled} aria-label={`Text element ${index + 1} bold`}
                    onChange={event => patchElement(element.id, { bold: event.target.checked })} className="accent-[var(--color-accent-blue)]" />
                </label>
                <GraphicOpacity value={element.opacity ?? 1} disabled={disabled} label={`Text element ${index + 1} opacity`}
                  onChange={value => patchElement(element.id, { opacity: value })} />
              </div>
            )}

            {element.type === 'rect' && (
              <div className="space-y-2">
                <div className="grid grid-cols-2 gap-2">
                  <label className="space-y-1"><FieldLabel>Fill</FieldLabel>
                    <input type="color" value={element.fill} disabled={disabled} aria-label={`Panel element ${index + 1} fill`}
                      onChange={event => patchElement(element.id, { fill: event.target.value })} className={colorClass} />
                  </label>
                  <label className="space-y-1"><FieldLabel>Border</FieldLabel>
                    <input type="color" value={element.border_color || '#ffffff'} disabled={disabled} aria-label={`Panel element ${index + 1} border color`}
                      onChange={event => patchElement(element.id, { border_color: event.target.value })} className={colorClass} />
                  </label>
                </div>
                <div className="grid grid-cols-2 gap-2">
                  <label className="space-y-1"><FieldLabel>Corner radius</FieldLabel>
                    <input type="number" min={0} max={600} step={1} value={element.radius ?? 0} disabled={disabled} aria-label={`Panel element ${index + 1} corner radius`}
                      onChange={event => { const value = Number(event.target.value); if (Number.isFinite(value) && value >= 0) patchElement(element.id, { radius: value }) }} className={fieldClass} />
                  </label>
                  <label className="space-y-1"><FieldLabel>Border width</FieldLabel>
                    <input type="number" min={0} max={250} step={1} value={element.border_width ?? 0} disabled={disabled} aria-label={`Panel element ${index + 1} border width`}
                      onChange={event => { const value = Number(event.target.value); if (Number.isFinite(value) && value >= 0) patchElement(element.id, { border_width: value }) }} className={fieldClass} />
                  </label>
                </div>
                <GraphicOpacity value={element.opacity ?? 1} disabled={disabled} label={`Panel element ${index + 1} opacity`}
                  onChange={value => patchElement(element.id, { opacity: value })} />
              </div>
            )}

            {element.type === 'image' && (
              <div className="space-y-2">
                <label className="block space-y-1"><FieldLabel>Project image</FieldLabel>
                  <select value={element.asset_id} disabled={disabled} aria-label={`Image element ${index + 1} project image`}
                    onChange={event => patchElement(element.id, { asset_id: event.target.value })} className={fieldClass}>
                    {imageAssets.map(asset => <option key={asset.id} value={asset.id}>{asset.display_name || asset.name}</option>)}
                  </select>
                </label>
                <label className="block space-y-1"><FieldLabel>Fit</FieldLabel>
                  <select value={element.fit || 'contain'} disabled={disabled} aria-label={`Image element ${index + 1} fit`}
                    onChange={event => patchElement(element.id, { fit: event.target.value as 'contain' | 'cover' })} className={fieldClass}>
                    <option value="contain">Contain</option><option value="cover">Cover</option>
                  </select>
                </label>
                <GraphicOpacity value={element.opacity ?? 1} disabled={disabled} label={`Image element ${index + 1} opacity`}
                  onChange={value => patchElement(element.id, { opacity: value })} />
              </div>
            )}
            </>}
          </article>
        ))}
      </div>
      {graphic.elements.length === 0 && <p className="rounded-md border border-accent-warm/25 bg-accent-warm/5 px-2.5 py-2 text-[9px] leading-relaxed text-accent-warm">This graphic is empty. Add text, a panel, or an image to make it visible.</p>}
      <button type="button" disabled={disabled} onClick={() => onUpdate({ graphic: { ...graphic, elements: [] } })}
        className="flex w-full items-center justify-center gap-1 rounded-md py-1.5 text-[9px] text-text-muted hover:bg-bg-hover disabled:opacity-40">
        <Plus size={10} /> Clear all elements
      </button>
    </section>
  )
}
