import { useEffect, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import { X, LayoutTemplate, Plus, Trash2 } from 'lucide-react'
import { useStore } from '../../stores/useStore'
import { parseFlux3Layout, serializeFlux3Layout, validBox, type Flux3Element, type LayoutBox } from '../../lib/flux3Layout'

export function Flux3PromptHelper() {
  const prompt = useStore(s => s.params.prompt)
  const setParam = useStore(s => s.setParam)
  const refs = useStore(s => s.imageRefs)
  const resolution = useStore(s => s.params.resolution)
  const [open, setOpen] = useState(false)
  const [caption, setCaption] = useState('')
  const [elements, setElements] = useState<Flux3Element[]>([])
  const [selected, setSelected] = useState(-1)
  const [sourceView, setSourceView] = useState(false)
  const [error, setError] = useState('')
  const [urls, setUrls] = useState<string[]>([])
  const [sourceAspect, setSourceAspect] = useState(1)
  const canvas = useRef<HTMLDivElement>(null)
  const gesture = useRef<{ start: [number, number]; box: LayoutBox; mode: 'draw' | 'move' | 'resize'; index: number } | null>(null)
  const current = elements[selected]
  const currentBox = sourceView ? current?.src_bbox : current?.from !== undefined ? current.tgt_bbox : current?.bbox
  const sourceIndex = Number(current?.from?.replace('ref_image_', ''))
  const sourceUrl = sourceView && Number.isInteger(sourceIndex) ? urls[sourceIndex] : undefined
  const [width, height] = String(resolution || '1024x1024').split('x').map(Number)

  useEffect(() => {
    if (!open) return
    const values = refs.map(file => URL.createObjectURL(file))
    setUrls(values)
    return () => values.forEach(url => URL.revokeObjectURL(url))
  }, [refs, open])
  useEffect(() => {
    if (!open) return
    const close = (event: KeyboardEvent) => { if (event.key === 'Escape') setOpen(false) }
    window.addEventListener('keydown', close)
    return () => window.removeEventListener('keydown', close)
  }, [open])

  function patchCurrent(update: Partial<Flux3Element>) {
    setElements(values => values.map((value, index) => index === selected ? { ...value, ...update } : value))
  }
  function addElement() {
    let number = elements.length + 1
    while (elements.some(element => element.id === `element_${number}`)) number++
    const id = `element_${number}`
    setElements(values => [...values, { id, desc: '', bbox: [200, 200, 800, 800] }])
    setCaption(value => `${value.trimEnd()} <${id}>`)
    setSelected(elements.length)
  }
  function point(event: React.PointerEvent): [number, number] {
    const rect = canvas.current!.getBoundingClientRect()
    return [Math.round(Math.max(0, Math.min(1000, (event.clientY - rect.top) / rect.height * 1000))), Math.round(Math.max(0, Math.min(1000, (event.clientX - rect.left) / rect.width * 1000)))]
  }
  const fieldClass = 'w-full bg-bg-tertiary border border-border rounded p-2 text-sm'

  return <>
    <button type="button" onClick={() => {
      try { const parsed = parseFlux3Layout(prompt); setCaption(parsed.caption); setElements(parsed.elements); setSelected(parsed.elements.length ? 0 : -1); setError(''); setOpen(true) }
      catch (cause) { setError(String((cause as Error).message)) }
    }} className="flex items-center gap-1 text-xs text-text-secondary p-1.5" title="Draw Flux 3 layout guidance"><LayoutTemplate size={14} />Prompt Helper</button>
    {!open && error && <p role="alert" className="text-xs text-red-400">{error}</p>}
    {open && createPortal(<div className="fixed inset-0 z-[100] bg-black/75 flex items-center justify-center p-2" onClick={event => event.stopPropagation()}>
      <section role="dialog" aria-modal="true" aria-label="Flux 3 Prompt Helper" className="bg-bg-primary border border-border rounded-xl w-full max-w-5xl max-h-[94dvh] overflow-auto p-4 text-text-primary">
        <div className="flex justify-between items-center mb-3"><h2 className="font-semibold">Flux 3 Prompt Helper</h2><button aria-label="Close prompt helper" onClick={() => setOpen(false)}><X size={20} /></button></div>
        <p className="text-xs text-text-muted mb-3">Boxes guide composition; they are not exact pixel masks. Coordinates: top, left, bottom, right on a 0–1000 grid.</p>
        <div className="grid md:grid-cols-2 gap-4">
          <div>
            <div className="flex gap-2 mb-2"><button onClick={addElement} className="flex gap-1 items-center text-sm"><Plus size={14} />Add element</button>
              <button disabled={!current} onClick={() => { setElements(values => values.filter((_, index) => index !== selected)); setSelected(-1) }} className="flex gap-1 items-center text-sm disabled:opacity-40"><Trash2 size={14} />Delete</button>
              <select aria-label="Layout view" value={sourceView ? 'source' : 'target'} onChange={event => setSourceView(event.target.value === 'source')} className="bg-bg-tertiary text-sm"><option value="target">Output layout</option><option value="source">Source region</option></select></div>
            <div ref={canvas} data-testid="flux3-layout-canvas" className="relative border border-border bg-bg-tertiary touch-none overflow-hidden" style={{ aspectRatio: sourceView ? sourceAspect : width / height }}
              onPointerDown={event => {
                if (!current || (sourceView && !sourceUrl)) return
                event.currentTarget.setPointerCapture(event.pointerId)
                const start = point(event)
                const inside = validBox(currentBox) && start[0] >= currentBox[0] && start[0] <= currentBox[2] && start[1] >= currentBox[1] && start[1] <= currentBox[3]
                const corner = inside && Math.abs(start[0] - currentBox![2]) < 45 && Math.abs(start[1] - currentBox![3]) < 45
                gesture.current = { start, box: validBox(currentBox) ? currentBox : [start[0], start[1], start[0], start[1]], mode: corner ? 'resize' : inside && !event.altKey ? 'move' : 'draw', index: selected }
              }} onPointerMove={event => {
                const active = gesture.current
                if (!active) return
                const end = point(event)
                let box: LayoutBox
                if (active.mode === 'move') {
                  const dy = Math.max(-active.box[0], Math.min(1000 - active.box[2], end[0] - active.start[0]))
                  const dx = Math.max(-active.box[1], Math.min(1000 - active.box[3], end[1] - active.start[1]))
                  box = [active.box[0] + dy, active.box[1] + dx, active.box[2] + dy, active.box[3] + dx]
                } else if (active.mode === 'resize') box = [active.box[0], active.box[1], Math.max(active.box[0] + 1, end[0]), Math.max(active.box[1] + 1, end[1])]
                else box = [Math.min(active.start[0], end[0]), Math.min(active.start[1], end[1]), Math.max(active.start[0], end[0]), Math.max(active.start[1], end[1])]
                patchCurrent(sourceView ? { src_bbox: box } : current.from !== undefined ? { tgt_bbox: box } : { bbox: box })
              }} onPointerUp={() => { gesture.current = null }} onPointerCancel={() => { gesture.current = null }}>
              {sourceUrl && <img src={sourceUrl} alt="Selected reference" className="absolute inset-0 w-full h-full" draggable={false} onLoad={event => setSourceAspect(event.currentTarget.naturalWidth / event.currentTarget.naturalHeight)} />}
              {elements.map((element, index) => {
                const box = sourceView ? index === selected ? element.src_bbox : null : element.from !== undefined ? element.tgt_bbox : element.bbox
                return validBox(box) && <div key={index} className={`absolute border-2 ${index === selected ? 'border-orange-400 bg-orange-400/15' : 'border-blue-400/60'} pointer-events-none`} style={{ top: `${box[0] / 10}%`, left: `${box[1] / 10}%`, height: `${(box[2] - box[0]) / 10}%`, width: `${(box[3] - box[1]) / 10}%` }}><span className="text-xs bg-black/60 text-white">{element.id}</span></div>
              })}
            </div>
            <p className="text-xs text-text-muted my-2">Select an element, then drag to draw. Drag inside to move; drag the bottom-right corner to resize. Alt-drag draws a new box.</p>
            <div className="flex flex-wrap gap-2">{elements.map((element, index) => <button className={`text-xs border rounded px-2 py-1 ${index === selected ? 'border-orange-400' : 'border-border'}`} key={index} onClick={() => setSelected(index)}>{element.id}</button>)}</div>
          </div>
          <div className="space-y-2">
            <label className="block text-sm">Scene caption<textarea aria-label="Scene caption" value={caption} onChange={event => setCaption(event.target.value)} className={fieldClass} rows={5} /></label>
            {current && <>
              <label className="block text-sm">Element ID<input aria-label="Element ID" value={current.id} onChange={event => {
                const id = event.target.value; setCaption(value => value.replaceAll(`<${current.id}>`, `<${id}>`)); patchCurrent({ id })
              }} className={fieldClass} /></label>
              <label className="block text-sm">Description<textarea aria-label="Element description" value={current.desc} onChange={event => patchCurrent({ desc: event.target.value })} className={fieldClass} /></label>
              <label className="block text-sm">Content source<select aria-label="Content source" value={current.from || 'generated'} className={fieldClass} onChange={event => {
                const from = event.target.value
                if (from === 'generated') { const next = { ...current, bbox: current.tgt_bbox || current.bbox || [200, 200, 800, 800] }; delete next.from; delete next.src_bbox; delete next.tgt_bbox; setElements(values => values.map((value, index) => index === selected ? next : value)) }
                else patchCurrent({ from, src_bbox: from === 'target' ? null : [0, 0, 1000, 1000], tgt_bbox: current.bbox || current.tgt_bbox || [200, 200, 800, 800], bbox: undefined })
              }}><option value="generated">Generated element</option><option value="target">New content in reference edit</option>{refs.map((file, index) => <option key={index} value={`ref_image_${index}`}>Image {index + 1}: {file.name}</option>)}</select></label>
              {current.from !== undefined && <label className="flex gap-2 text-sm"><input type="checkbox" checked={current.tgt_bbox === null} onChange={event => patchCurrent({ tgt_bbox: event.target.checked ? null : [200, 200, 800, 800] })} />Remove this reference element</label>}
              <p className="text-xs text-text-muted">Output: {JSON.stringify(current.from !== undefined ? current.tgt_bbox : current.bbox)}<br />{current.from && `Source: ${JSON.stringify(current.src_bbox)}`}</p>
            </>}
            {error && <p role="alert" className="text-sm text-red-400">{error}</p>}
            <button className="bg-orange-500 text-white px-4 py-2 rounded" onClick={() => {
              try {
                for (const element of elements) if (element.from?.startsWith('ref_image_') && Number(element.from.slice(10)) >= refs.length) throw new Error(`Upload the reference used by ${element.id}.`)
                setParam('prompt', serializeFlux3Layout(caption, elements)); setOpen(false); setError('')
              } catch (cause) { setError((cause as Error).message) }
            }}>Save prompt</button>
          </div>
        </div>
      </section>
    </div>, document.body)}
  </>
}
