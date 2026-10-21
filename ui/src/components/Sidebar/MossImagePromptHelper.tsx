import { useEffect, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import { LayoutTemplate, Plus, Trash2, X } from 'lucide-react'
import { useStore } from '../../stores/useStore'
import {
  emptyMossImageDocument,
  fitMossImageBoxToCanvas,
  mossCanvasPoint,
  mossImageBoxProblem,
  mossImageBoxStyle,
  parseMossImageLayout,
  parseMossImageSize,
  serializeMossImageLayout,
  validMossImageBox,
  type MossImageBox,
  type MossImageDocument,
  type MossImageItem,
} from '../../lib/mossImageLayout'

type GestureMode = 'draw' | 'move' | 'n' | 'ne' | 'e' | 'se' | 's' | 'sw' | 'w' | 'nw'
type Gesture = {
  start: [number, number]
  box?: MossImageBox
  mode: GestureMode
  index: number
  selectedBefore: number
}

function clamp(value: number, low: number, high: number): number {
  return Math.max(low, Math.min(high, value))
}

function handles(box: MossImageBox): Array<[GestureMode, number, number]> {
  const [top, left, bottom, right] = box
  return [
    ['nw', top, left], ['n', top, (left + right) / 2], ['ne', top, right],
    ['e', (top + bottom) / 2, right], ['se', bottom, right], ['s', bottom, (left + right) / 2],
    ['sw', bottom, left], ['w', (top + bottom) / 2, left],
  ]
}

function pointInside(point: [number, number], box: MossImageBox): boolean {
  return point[0] >= box[0] && point[0] <= box[2] && point[1] >= box[1] && point[1] <= box[3]
}

function itemTitle(item: MossImageItem, index: number): string {
  const title = item.text || item.caption || item.label || 'New item'
  return `${index + 1}. ${title}`
}

function boxFromInput(value: string): MossImageBox | null {
  try {
    const parsed: unknown = JSON.parse(value)
    return Array.isArray(parsed) && parsed.length === 4 && parsed.every(Number.isFinite)
      ? parsed as MossImageBox
      : null
  } catch {
    return null
  }
}

export function MossImagePromptHelper() {
  const prompt = useStore(state => state.params.prompt)
  const setParam = useStore(state => state.setParam)
  const resolution = useStore(state => state.params.resolution)
  const size = parseMossImageSize(typeof resolution === 'string' ? resolution : undefined)
  const [open, setOpen] = useState(false)
  const [layout, setLayout] = useState<MossImageDocument>(() => emptyMossImageDocument())
  const [comments, setComments] = useState<string[]>([])
  const [selected, setSelected] = useState(-1)
  const [boxDraft, setBoxDraft] = useState('')
  const [error, setError] = useState('')
  const baseline = useRef('')
  const canvas = useRef<HTMLDivElement>(null)
  const dialog = useRef<HTMLElement>(null)
  const gesture = useRef<Gesture | null>(null)
  const current = layout.items[selected]

  useEffect(() => {
    if (!open) return
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        event.preventDefault()
        setOpen(false)
      }
    }
    window.addEventListener('keydown', closeOnEscape)
    dialog.current?.focus()
    return () => window.removeEventListener('keydown', closeOnEscape)
  }, [open])

  function selectItem(index: number, source = layout) {
    setSelected(index)
    const box = source.items[index]?.bbox
    setBoxDraft(Array.isArray(box) ? JSON.stringify(box) : '')
    setError('')
  }

  function patchItem(index: number, update: Partial<MossImageItem>) {
    setLayout(value => ({
      ...value,
      items: value.items.map((item, itemIndex) => itemIndex === index ? { ...item, ...update } : item),
    }))
  }

  function openHelper() {
    try {
      const parsed = parseMossImageLayout(prompt || '')
      baseline.current = prompt || ''
      setLayout(parsed.document)
      setComments(parsed.comments)
      setError('')
      selectItem(parsed.document.items.length ? 0 : -1, parsed.document)
      setOpen(true)
    } catch (cause) {
      setError((cause as Error).message)
    }
  }

  function addItem() {
    const halfHeight = Math.max(1, Math.round(size.height / 8))
    const halfWidth = Math.max(1, Math.round(size.width / 8))
    const box: MossImageBox = [-halfHeight, -halfWidth, halfHeight, halfWidth]
    const next = { ...layout, items: [...layout.items, { label: 'object', caption: '', bbox: box }] }
    setLayout(next)
    selectItem(next.items.length - 1, next)
  }

  function deleteItem() {
    if (!current) return
    const next = { ...layout, items: layout.items.filter((_, index) => index !== selected) }
    const index = Math.min(selected, next.items.length - 1)
    setLayout(next)
    selectItem(index, next)
  }

  function updateBox(index: number, box: MossImageBox) {
    patchItem(index, { bbox: box })
    if (index === selected) setBoxDraft(JSON.stringify(box))
  }

  function handleCanvasPointerDown(event: React.PointerEvent<HTMLDivElement>) {
    if (event.button !== 0 || !canvas.current) return
    const rect = canvas.current.getBoundingClientRect()
    const point = mossCanvasPoint(event.clientX, event.clientY, rect, size)
    const scaleX = size.width / Math.max(rect.width, 1)
    const scaleY = size.height / Math.max(rect.height, 1)
    const radiusX = 8 * scaleX
    const radiusY = 8 * scaleY
    let hit: { index: number; mode: GestureMode; box: MossImageBox } | null = null

    for (let index = layout.items.length - 1; index >= 0; index--) {
      const box = layout.items[index].bbox
      if (!validMossImageBox(box, size)) continue
      const typedBox = box as MossImageBox
      const handle = handles(typedBox).find(([, y, x]) => Math.abs(point[0] - y) <= radiusY && Math.abs(point[1] - x) <= radiusX)
      if (handle) {
        hit = { index, mode: handle[0], box: typedBox }
        break
      }
      if (pointInside(point, typedBox)) {
        hit = { index, mode: 'move', box: typedBox }
        break
      }
    }

    const selectedBefore = selected
    if (hit) {
      selectItem(hit.index)
      gesture.current = { start: point, box: [...hit.box], mode: hit.mode, index: hit.index, selectedBefore }
    } else {
      gesture.current = { start: point, mode: 'draw', index: -1, selectedBefore }
    }
    canvas.current.setPointerCapture(event.pointerId)
    event.preventDefault()
  }

  function handleCanvasPointerMove(event: React.PointerEvent<HTMLDivElement>) {
    const active = gesture.current
    if (!active || !canvas.current) return
    const rect = canvas.current.getBoundingClientRect()
    const point = mossCanvasPoint(event.clientX, event.clientY, rect, size)
    let box: MossImageBox

    if (active.mode === 'draw') {
      box = [Math.min(active.start[0], point[0]), Math.min(active.start[1], point[1]), Math.max(active.start[0], point[0]), Math.max(active.start[1], point[1])]
      if (box[0] === box[2] || box[1] === box[3]) return
      if (active.index < 0) {
        const next = { ...layout, items: [...layout.items, { label: 'object', caption: '', bbox: box }] }
        active.index = next.items.length - 1
        setLayout(next)
        selectItem(active.index, next)
      } else updateBox(active.index, box)
      return
    }

    if (!active.box) return
    const [top, left, bottom, right] = active.box
    let nextTop = top
    let nextLeft = left
    let nextBottom = bottom
    let nextRight = right
    if (active.mode === 'move') {
      const dy = point[0] - active.start[0]
      const dx = point[1] - active.start[1]
      const allowedY = -size.height / 2 - top <= size.height / 2 - bottom
      const allowedX = -size.width / 2 - left <= size.width / 2 - right
      const moveY = allowedY ? clamp(dy, -size.height / 2 - top, size.height / 2 - bottom) : 0
      const moveX = allowedX ? clamp(dx, -size.width / 2 - left, size.width / 2 - right) : 0
      nextTop += moveY
      nextBottom += moveY
      nextLeft += moveX
      nextRight += moveX
    } else {
      const resizeYmin = active.mode.includes('n')
      const resizeYmax = active.mode.includes('s')
      const resizeXmin = active.mode.includes('w')
      const resizeXmax = active.mode.includes('e')
      if (resizeYmin) nextTop = clamp(point[0], -size.height / 2, bottom - 1)
      if (resizeYmax) nextBottom = clamp(point[0], top + 1, size.height / 2)
      if (resizeXmin) nextLeft = clamp(point[1], -size.width / 2, right - 1)
      if (resizeXmax) nextRight = clamp(point[1], left + 1, size.width / 2)
    }
    updateBox(active.index, [nextTop, nextLeft, nextBottom, nextRight])
  }

  function endCanvasGesture(event: React.PointerEvent<HTMLDivElement>) {
    const active = gesture.current
    if (!active) return
    if (event.type === 'pointercancel') {
      if (active.mode === 'draw' && active.index >= 0) {
        setLayout(value => ({ ...value, items: value.items.filter((_, index) => index !== active.index) }))
      } else if (active.box) {
        updateBox(active.index, active.box)
      }
      selectItem(active.selectedBefore)
    }
    gesture.current = null
    if (canvas.current?.hasPointerCapture(event.pointerId)) canvas.current.releasePointerCapture(event.pointerId)
  }

  function fitBoxes() {
    let changed = 0
    let unfit = 0
    const items = layout.items.map(item => {
      const box = fitMossImageBoxToCanvas(item.bbox, size)
      if (!box) {
        if (item.bbox !== undefined) unfit++
        return item
      }
      if (JSON.stringify(box) !== JSON.stringify(item.bbox)) changed++
      return { ...item, bbox: box }
    })
    const next = { ...layout, items }
    setLayout(next)
    if (selected >= 0) setBoxDraft(Array.isArray(items[selected]?.bbox) ? JSON.stringify(items[selected].bbox) : '')
    setError(unfit ? `Fitted ${changed} box${changed === 1 ? '' : 'es'}. ${unfit} item${unfit === 1 ? ' has' : 's have'} a box that needs four finite numbers entered manually.` : `Fitted ${changed} box${changed === 1 ? '' : 'es'} to the ${size.width} × ${size.height} canvas, preserving size where possible.`)
  }

  function savePrompt() {
    if ((prompt || '') !== baseline.current) {
      setError('The main prompt changed while this helper was open. Close and reopen the helper to load the latest prompt.')
      return
    }
    let next = layout
    if (selected >= 0) {
      const box = boxFromInput(boxDraft)
      if (!box) {
        setError('Enter four finite numbers in [ymin, xmin, ymax, xmax] order for the selected item.')
        return
      }
      next = { ...layout, items: layout.items.map((item, index) => index === selected ? { ...item, bbox: box } : item) }
    }
    if (!next.summary.caption.trim()) {
      setError('Add a scene description in Summary caption.')
      return
    }
    for (let index = 0; index < next.items.length; index++) {
      const item = next.items[index]
      if (!['text', 'object', 'character'].includes(item.label || '')) {
        selectItem(index, next)
        setError(`Item ${index + 1}: choose Text, Object or Character for its label.`)
        return
      }
      if (!item.caption?.trim()) {
        selectItem(index, next)
        setError(`Item ${index + 1}: add a caption describing it.`)
        return
      }
      if (item.label === 'text' && !item.text?.trim()) {
        selectItem(index, next)
        setError(`Item ${index + 1}: add the exact visible wording in Text.`)
        return
      }
      const problem = mossImageBoxProblem(item.bbox, size)
      if (problem) {
        selectItem(index, next)
        setError(`Item ${index + 1}: ${problem} Use Fit Boxes to Canvas or edit its coordinates.`)
        return
      }
    }
    setParam('prompt', serializeMossImageLayout(next, comments))
    setOpen(false)
    setError('')
  }

  const fieldClass = 'w-full bg-bg-tertiary border border-border rounded p-2 text-sm'

  return <>
    <button type="button" onClick={openHelper} className="flex items-center gap-1 text-xs text-text-secondary p-1.5" title="Draw MOSS Image layout guidance">
      <LayoutTemplate size={14} />Prompt Helper
    </button>
    {!open && error && <p role="alert" className="text-xs text-red-400">{error}</p>}
    {open && createPortal(
      <div className="fixed inset-0 z-[100] bg-black/75 flex items-start justify-center overflow-y-auto p-2" onClick={event => { if (event.target === event.currentTarget) setOpen(false) }}>
        <section ref={dialog} role="dialog" aria-modal="true" aria-labelledby="moss-image-helper-title" tabIndex={-1} className="bg-bg-primary border border-border rounded-xl w-full max-w-5xl max-h-[94dvh] overflow-auto p-4 text-text-primary">
          <div className="flex justify-between items-center mb-3">
            <h2 id="moss-image-helper-title" className="font-semibold">MOSS Image Prompt Helper</h2>
            <button type="button" aria-label="Close prompt helper" onClick={() => setOpen(false)}><X size={20} /></button>
          </div>
          <p className="text-xs text-text-muted mb-3">Text-to-image layout only. Pixel corners use [ymin, xmin, ymax, xmax], measured from the canvas center; X increases right and Y increases down. Current canvas: {size.width} × {size.height}.</p>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="min-w-0">
              <div className="flex flex-wrap items-center gap-2 mb-2">
                <button type="button" onClick={addItem} className="flex gap-1 items-center text-sm"><Plus size={14} />Add item</button>
                <button type="button" disabled={!current} onClick={deleteItem} className="flex gap-1 items-center text-sm disabled:opacity-40"><Trash2 size={14} />Delete</button>
                <button type="button" disabled={!layout.items.length} onClick={fitBoxes} className="text-sm disabled:opacity-40">Fit Boxes to Canvas</button>
              </div>
              <div className="flex justify-center">
                <div
                  ref={canvas}
                  data-testid="moss-image-layout-canvas"
                  className="relative border border-border bg-bg-tertiary touch-none overflow-hidden w-full"
                  style={{ aspectRatio: `${size.width} / ${size.height}`, width: `min(100%, ${(55 * size.width / size.height).toFixed(3)}dvh)` }}
                  onPointerDown={handleCanvasPointerDown}
                  onPointerMove={handleCanvasPointerMove}
                  onPointerUp={endCanvasGesture}
                  onPointerCancel={endCanvasGesture}
                >
                  {layout.items.map((item, index) => {
                    const box = item.bbox
                    if (!Array.isArray(box) || box.length !== 4 || !box.every(Number.isFinite) || box[0] >= box[2] || box[1] >= box[3]) return null
                    const typedBox = box as MossImageBox
                    const selectedItem = index === selected
                    return <div key={index} className={`absolute border-2 ${selectedItem ? 'border-orange-400 bg-orange-400/15' : 'border-blue-400/70 bg-blue-400/10'} pointer-events-none`} style={mossImageBoxStyle(typedBox, size)}>
                      <span className="text-xs bg-black/60 text-white px-0.5">{item.text || item.label || `Item ${index + 1}`}</span>
                      {selectedItem && handles(typedBox).map(([mode, y, x]) => <span key={mode} className="absolute w-2 h-2 bg-orange-400 border border-white rounded-sm" style={{ top: `${((y - box[0]) / (box[2] - box[0])) * 100}%`, left: `${((x - box[1]) / (box[3] - box[1])) * 100}%`, transform: 'translate(-50%, -50%)' }} />)}
                    </div>
                  })}
                </div>
              </div>
              <p className="text-xs text-text-muted my-2">Drag to draw an item. Select a box and drag inside to move it; drag any of its eight handles to resize.</p>
              <div className="flex flex-wrap gap-2">
                {layout.items.map((item, index) => <button type="button" className={`text-xs border rounded px-2 py-1 ${index === selected ? 'border-orange-400' : 'border-border'}`} key={index} onClick={() => selectItem(index)}>{itemTitle(item, index)}</button>)}
              </div>
            </div>
            <div className="min-w-0 space-y-2">
              <label className="block text-sm">Summary caption<textarea aria-label="Summary caption" value={layout.summary.caption} onChange={event => setLayout(value => ({ ...value, summary: { ...value.summary, caption: event.target.value } }))} className={fieldClass} rows={4} /></label>
              <label className="block text-sm">Canvas arrangement<select aria-label="Canvas arrangement" value={layout.summary.canvas} onChange={event => {
                const arrangement = event.currentTarget.value
                setLayout(value => ({ ...value, summary: { ...value.summary, canvas: arrangement } }))
              }} className={fieldClass}>
                {!['single', 'multi_panel', 'collage'].includes(layout.summary.canvas) && <option value={layout.summary.canvas}>{layout.summary.canvas}</option>}
                <option value="single">Single image</option><option value="multi_panel">Multiple panels</option><option value="collage">Collage</option>
              </select></label>
              {current && <>
                <label className="block text-sm">Label<select aria-label="Item label" value={current.label || ''} onChange={event => patchItem(selected, { label: event.target.value })} className={fieldClass}>
                  <option value="" disabled>Choose a label</option>
                  {!['text', 'object', 'character', ''].includes(current.label || '') && <option value={current.label}>{current.label}</option>}
                  <option value="object">Object</option><option value="character">Character</option><option value="text">Text</option>
                </select></label>
                <label className="block text-sm">Item caption<textarea aria-label="Item caption" value={current.caption || ''} onChange={event => patchItem(selected, { caption: event.target.value })} className={fieldClass} rows={3} /></label>
                {current.label === 'text' && <label className="block text-sm">Exact visible text<textarea aria-label="Exact visible text" value={current.text || ''} onChange={event => patchItem(selected, { text: event.target.value })} className={fieldClass} rows={2} /></label>}
                <label className="block text-sm">Bounding box [ymin, xmin, ymax, xmax]<textarea aria-label="Bounding box coordinates" value={boxDraft} onChange={event => {
                  const value = event.target.value
                  setBoxDraft(value)
                  const box = boxFromInput(value)
                  if (box) patchItem(selected, { bbox: box })
                }} className={fieldClass} rows={2} spellCheck={false} /></label>
                {mossImageBoxProblem(boxFromInput(boxDraft), size) && <p className="text-xs text-amber-400">{mossImageBoxProblem(boxFromInput(boxDraft), size)}</p>}
              </>}
              {error && <p role="alert" className="text-sm text-red-400">{error}</p>}
              <button type="button" onClick={savePrompt} className="bg-orange-500 text-white px-4 py-2 rounded">Save prompt</button>
            </div>
          </div>
        </section>
      </div>, window.document.body)}
  </>
}
