import { useEffect, useLayoutEffect, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import { X, LayoutTemplate, Plus, Trash2 } from 'lucide-react'
import { useStore } from '../../stores/useStore'
import { parseFlux3Layout, serializeFlux3Layout, validBox, type Flux3Element, type LayoutBox } from '../../lib/flux3Layout'

function referenceImageIndex(from: string | undefined): number | null {
  const match = /^ref_image_(\d+)$/.exec(from || '')
  if (!match) return null
  const index = Number(match[1])
  return Number.isSafeInteger(index) && index >= 0 ? index : null
}

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
  const [displayedReference, setDisplayedReference] = useState(0)
  const [error, setError] = useState('')
  const [urls, setUrls] = useState<string[]>([])
  const [sourceImage, setSourceImage] = useState<{ url: string; aspect: number; loaded: boolean; error: boolean } | null>(null)
  const canvas = useRef<HTMLDivElement>(null)
  const activeSourceUrl = useRef<string | undefined>(undefined)
  const gesture = useRef<{ start: [number, number]; box: LayoutBox; mode: 'draw' | 'move' | 'resize'; index: number } | null>(null)
  const current = elements[selected]
  const currentUsesDisplayedReference = current?.from === `ref_image_${displayedReference}`
  const currentBox = sourceView ? currentUsesDisplayedReference ? current?.src_bbox : null : current?.from !== undefined ? current.tgt_bbox : current?.bbox
  const sourceUrl = sourceView ? urls[displayedReference] : undefined
  const activeSourceImage = sourceImage?.url === sourceUrl ? sourceImage : null
  const sourceAspect = activeSourceImage?.aspect ?? 1
  const sourceReady = Boolean(activeSourceImage?.loaded)
  const sourceLoadError = Boolean(activeSourceImage?.error)
  const [width, height] = String(resolution || '1024x1024').split('x').map(Number)

  useEffect(() => {
    if (!open) {
      setUrls([])
      setSourceImage(null)
      return
    }
    const values = refs.map(file => URL.createObjectURL(file))
    setUrls(values)
    return () => values.forEach(url => URL.revokeObjectURL(url))
  }, [refs, open])
  useLayoutEffect(() => { activeSourceUrl.current = sourceUrl }, [sourceUrl])
  useEffect(() => {
    if (!refs.length) {
      setSourceView(false)
      setDisplayedReference(0)
      return
    }
    setDisplayedReference(index => Math.min(index, refs.length - 1))
  }, [refs])
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
    const reference = sourceView && displayedReference < refs.length ? displayedReference : null
    const element: Flux3Element = reference === null
      ? { id, desc: '', bbox: [200, 200, 800, 800] }
      : { id, desc: '', from: `ref_image_${reference}`, src_bbox: null, tgt_bbox: [200, 200, 800, 800] }
    setElements(values => [...values, element])
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
      try {
        const parsed = parseFlux3Layout(prompt)
        const referenceElement = parsed.elements.findIndex(element => {
          const index = referenceImageIndex(element.from)
          return index !== null && index < refs.length
        })
        setCaption(parsed.caption)
        setElements(parsed.elements)
        setSelected(referenceElement >= 0 ? referenceElement : parsed.elements.length ? 0 : -1)
        setDisplayedReference(referenceElement >= 0 ? referenceImageIndex(parsed.elements[referenceElement].from) ?? 0 : 0)
        setSourceView(refs.length > 0)
        setError('')
        setOpen(true)
      }
      catch (cause) { setError(String((cause as Error).message)) }
    }} className="flex items-center gap-1 text-xs text-text-secondary p-1.5" title="Draw Flux 3 layout guidance"><LayoutTemplate size={14} />Prompt Helper</button>
    {!open && error && <p role="alert" className="text-xs text-red-400">{error}</p>}
    {open && createPortal(<div className="fixed inset-0 z-[100] bg-black/75 flex items-center justify-center p-2" onClick={event => event.stopPropagation()}>
      <section role="dialog" aria-modal="true" aria-label="Flux 3 Prompt Helper" className="bg-bg-primary border border-border rounded-xl w-full max-w-5xl max-h-[94dvh] overflow-auto p-4 text-text-primary">
        <div className="flex justify-between items-center mb-3"><h2 className="font-semibold">Flux 3 Prompt Helper</h2><button aria-label="Close prompt helper" onClick={() => setOpen(false)}><X size={20} /></button></div>
        <p className="text-xs text-text-muted mb-3">Boxes guide composition; they are not exact pixel masks. Coordinates: top, left, bottom, right on a 0–1000 grid.</p>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="min-w-0">
            <div className="flex flex-wrap items-center gap-2 mb-2"><button onClick={addElement} className="flex gap-1 items-center text-sm"><Plus size={14} />Add element</button>
              <button disabled={!current} onClick={() => { setElements(values => values.filter((_, index) => index !== selected)); setSelected(-1) }} className="flex gap-1 items-center text-sm disabled:opacity-40"><Trash2 size={14} />Delete</button>
              <label className="flex min-w-0 max-w-full items-center gap-1 text-sm">Reference image<select aria-label="Reference image" value={refs.length ? String(displayedReference) : ''} disabled={!refs.length} onChange={event => { setDisplayedReference(Number(event.target.value)); setSelected(-1) }} className="min-w-0 max-w-full flex-1 bg-bg-tertiary text-sm disabled:opacity-50">
                <option value="" disabled>{refs.length ? 'Choose reference' : 'No reference images'}</option>{refs.map((file, index) => <option key={index} value={index}>{`Image ${index + 1}: ${file.name}`}</option>)}
              </select></label>
              <label className="flex items-center gap-1 text-sm">Layout view<select aria-label="Layout view" value={sourceView ? 'source' : 'target'} onChange={event => setSourceView(event.target.value === 'source')} className="bg-bg-tertiary text-sm"><option value="target">Output layout</option><option value="source" disabled={!refs.length}>Source region</option></select></label></div>
            <div className="flex justify-center">
            <div ref={canvas} data-testid="flux3-layout-canvas" className="relative border border-border bg-bg-tertiary touch-none overflow-hidden" style={{ aspectRatio: sourceView ? sourceAspect : width / height, width: sourceView && sourceUrl ? `min(100%, ${(55 * sourceAspect).toFixed(3)}dvh)` : '100%' }}
              onPointerDown={event => {
                if (!current || (sourceView && (!sourceReady || !currentUsesDisplayedReference))) return
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
              {sourceUrl && <img key={sourceUrl} src={sourceUrl} alt="Selected reference" className="absolute inset-0 w-full h-full object-contain" draggable={false} onLoad={event => {
                if (activeSourceUrl.current !== sourceUrl) return
                const { naturalWidth, naturalHeight } = event.currentTarget
                if (naturalWidth > 0 && naturalHeight > 0) setSourceImage({ url: sourceUrl, aspect: naturalWidth / naturalHeight, loaded: true, error: false })
              }} onError={() => {
                if (activeSourceUrl.current === sourceUrl) setSourceImage({ url: sourceUrl, aspect: 1, loaded: false, error: true })
              }} />}
              {sourceView && !sourceUrl && <div className="absolute inset-0 flex items-center justify-center p-3 text-center text-xs text-text-muted">Upload a reference image to view source regions.</div>}
              {sourceUrl && !sourceReady && <div role={sourceLoadError ? 'alert' : 'status'} className="absolute inset-0 flex items-center justify-center p-3 text-center text-xs text-text-muted">{sourceLoadError ? 'This reference image could not be loaded.' : 'Loading reference image…'}</div>}
              {elements.map((element, index) => {
                const box = sourceView ? sourceReady && index === selected && currentUsesDisplayedReference ? element.src_bbox : null : element.from !== undefined ? element.tgt_bbox : element.bbox
                return validBox(box) && <div key={index} className={`absolute border-2 ${index === selected ? 'border-orange-400 bg-orange-400/15' : 'border-blue-400/60'} pointer-events-none`} style={{ top: `${box[0] / 10}%`, left: `${box[1] / 10}%`, height: `${(box[2] - box[0]) / 10}%`, width: `${(box[3] - box[1]) / 10}%` }}><span className="text-xs bg-black/60 text-white">{element.id}</span></div>
              })}
            </div>
            </div>
            <p className="text-xs text-text-muted my-2">{sourceView
              ? !refs.length ? 'Upload a reference image to view source regions.'
                : !sourceUrl ? 'Loading reference image…'
                  : sourceLoadError ? 'This reference image could not be loaded.'
                    : !sourceReady ? 'Loading reference image…'
                      : currentUsesDisplayedReference ? 'Select an element linked to this image, then drag to draw its source region. Add element binds a new element to the selected reference.'
                        : 'Add an element to bind it to this reference, or select an element linked to it.'
              : 'Select an element, then drag to draw. Drag inside to move; drag the bottom-right corner to resize. Alt-drag draws a new box.'}</p>
            <div className="flex flex-wrap gap-2">{elements.map((element, index) => <button className={`text-xs border rounded px-2 py-1 ${index === selected ? 'border-orange-400' : 'border-border'}`} key={index} onClick={() => {
              setSelected(index)
              const reference = referenceImageIndex(element.from)
              if (reference !== null && reference < refs.length) { setDisplayedReference(reference); setSourceView(true) }
              else setSourceView(false)
            }}>{element.id}</button>)}</div>
          </div>
          <div className="min-w-0 space-y-2">
            <label className="block text-sm">Scene caption<textarea aria-label="Scene caption" value={caption} onChange={event => setCaption(event.target.value)} className={fieldClass} rows={5} /></label>
            {current && <>
              <label className="block text-sm">Element ID<input aria-label="Element ID" value={current.id} onChange={event => {
                const id = event.target.value; setCaption(value => value.replaceAll(`<${current.id}>`, `<${id}>`)); patchCurrent({ id })
              }} className={fieldClass} /></label>
              <label className="block text-sm">Description<textarea aria-label="Element description" value={current.desc} onChange={event => patchCurrent({ desc: event.target.value })} className={fieldClass} /></label>
              <label className="block text-sm">Content source<select aria-label="Content source" value={current.from || 'generated'} className={fieldClass} onChange={event => {
                const from = event.target.value
                if (from === 'generated') { const next = { ...current, bbox: current.tgt_bbox || current.bbox || [200, 200, 800, 800] }; delete next.from; delete next.src_bbox; delete next.tgt_bbox; setElements(values => values.map((value, index) => index === selected ? next : value)) }
                else {
                  const reference = referenceImageIndex(from)
                  if (reference !== null) setDisplayedReference(reference)
                  patchCurrent({ from, src_bbox: from === 'target' ? null : [0, 0, 1000, 1000], tgt_bbox: current.bbox || current.tgt_bbox || [200, 200, 800, 800], bbox: undefined })
                }
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
