import { useRef, useState, useEffect, useCallback, useMemo } from 'react'
import { useStore } from '../../stores/useStore'

type Aspect = '16:9' | '9:16' | '1:1' | '4:3' | '3:4' | 'source'

interface Props {
  videoUrl: string
  /** Source pixel dimensions parsed from `editVideoResolution` (e.g. "1280x720"). */
  srcW: number
  srcH: number
}

/**
 * Interactive composer for the outpaint canvas.
 *
 * The black outer rectangle represents the FINAL output canvas at the chosen
 * aspect ratio. The video preview inside is the source clip — user drags it
 * to reposition and uses the corner handles to resize. The pixel-space
 * pads (top/bottom/left/right) are derived from the box position+size on
 * submit (see useStore.submit, outpaint branch).
 *
 * State model: the video box is stored in canvas-relative coordinates
 * (0–1 fractions). When the user picks a new aspect, we re-fit the box so
 * the source sits centered, fully visible (letterboxed), at native scale.
 *
 * Display: the canvas is rendered at a fixed display height with width
 * computed from aspect. We track the rendered pixel size via a ResizeObserver
 * so pointer deltas convert correctly to canvas-relative deltas during drag.
 */
export function OutpaintCanvas({ videoUrl, srcW, srcH }: Props) {
  const aspect = useStore(s => s.outpaintAspect)
  const setAspect = useStore(s => s.setOutpaintAspect)
  const box = useStore(s => s.outpaintVideoBox)
  const setBox = useStore(s => s.setOutpaintVideoBox)

  const canvasRef = useRef<HTMLDivElement>(null)
  const [renderedSize, setRenderedSize] = useState({ w: 0, h: 0 })
  const [drag, setDrag] = useState<null | {
    mode: 'move' | 'nw' | 'ne' | 'sw' | 'se'
    startPx: { x: number; y: number }
    startBox: { x: number; y: number; w: number; h: number }
  }>(null)

  // Compute canvas pixel dimensions in source-pixel-space (matches the
  // server-side computation in useStore.submit). Used to letterbox-fit the
  // video on aspect change and to display the actual output dimensions.
  const canvasPx = useMemo(() => {
    if (srcW <= 0 || srcH <= 0) return { w: 0, h: 0 }
    if (aspect === 'source') return { w: srcW, h: srcH }
    const [aw, ah] = aspect.split(':').map(Number)
    const target = aw / ah
    const srcRatio = srcW / srcH
    if (srcRatio > target) return { w: srcW, h: Math.round(srcW / target) }
    return { w: Math.round(srcH * target), h: srcH }
  }, [aspect, srcW, srcH])

  // Re-fit the video to the new canvas whenever aspect changes. Centered,
  // at native source scale relative to the canvas. (User can then drag/
  // resize off the default position.)
  useEffect(() => {
    if (canvasPx.w <= 0 || canvasPx.h <= 0) return
    const w = srcW / canvasPx.w
    const h = srcH / canvasPx.h
    setBox({ x: (1 - w) / 2, y: (1 - h) / 2, w, h })
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [aspect, srcW, srcH])

  // Track rendered canvas pixel size for accurate drag math.
  useEffect(() => {
    if (!canvasRef.current) return
    const el = canvasRef.current
    const ro = new ResizeObserver(entries => {
      const { width, height } = entries[0].contentRect
      setRenderedSize({ w: width, h: height })
    })
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  const aspectStyle = useMemo(() => {
    // Constrain canvas to fit the sidebar width AND a reasonable max height.
    // 200px height keeps the composer compact; portrait aspects clamp to 280
    // so 9:16 is still proportionally readable.
    const ratio = canvasPx.w > 0 && canvasPx.h > 0 ? canvasPx.w / canvasPx.h : 1
    const maxHeight = ratio < 1 ? 280 : 220
    return { aspectRatio: `${ratio}`, maxHeight: `${maxHeight}px`, margin: '0 auto' as const }
  }, [canvasPx])

  const onPointerDown = useCallback(
    (mode: 'move' | 'nw' | 'ne' | 'sw' | 'se') => (e: React.PointerEvent) => {
      e.preventDefault()
      e.stopPropagation()
      ;(e.target as HTMLElement).setPointerCapture(e.pointerId)
      setDrag({
        mode,
        startPx: { x: e.clientX, y: e.clientY },
        startBox: { ...box },
      })
    },
    [box]
  )

  useEffect(() => {
    if (!drag) return
    const handleMove = (e: PointerEvent) => {
      if (renderedSize.w <= 0 || renderedSize.h <= 0) return
      const dx = (e.clientX - drag.startPx.x) / renderedSize.w
      const dy = (e.clientY - drag.startPx.y) / renderedSize.h
      const start = drag.startBox
      let nb = { ...start }
      if (drag.mode === 'move') {
        nb.x = clamp(start.x + dx, 0, 1 - start.w)
        nb.y = clamp(start.y + dy, 0, 1 - start.h)
      } else {
        // Resize from the named corner. Aspect ratio of the video preview is
        // locked to source aspect (resizing scales uniformly so the preview
        // doesn't get squished — matches what the server does on submit).
        const sourceAspect = srcW / srcH
        const canvasAspect = canvasPx.w > 0 ? canvasPx.w / canvasPx.h : 1
        // Convert canvas-relative aspect: a square-ish box in canvas units
        // corresponds to a source-aspect rect when canvas isn't square.
        const boxAspect = sourceAspect / canvasAspect

        // Use the dominant axis delta to drive uniform scale.
        const dominant = Math.abs(dx) > Math.abs(dy) ? dx : dy
        let scale = 1
        // Direction: NE/SE positive-x grow, NW/SW negative-x grow, etc.
        if (drag.mode === 'se') scale = 1 + dominant / Math.max(start.w, 0.01)
        else if (drag.mode === 'sw') scale = 1 - dominant / Math.max(start.w, 0.01)
        else if (drag.mode === 'ne') scale = 1 + dominant / Math.max(start.w, 0.01)
        else if (drag.mode === 'nw') scale = 1 - dominant / Math.max(start.w, 0.01)
        scale = Math.max(0.05, Math.min(2, scale))

        const nw = clamp(start.w * scale, 0.05, 1)
        const nh = clamp(nw / boxAspect, 0.05, 1)
        // Anchor the OPPOSITE corner so the resize feels natural.
        if (drag.mode === 'se') {
          nb = { x: start.x, y: start.y, w: nw, h: nh }
        } else if (drag.mode === 'sw') {
          nb = { x: start.x + start.w - nw, y: start.y, w: nw, h: nh }
        } else if (drag.mode === 'ne') {
          nb = { x: start.x, y: start.y + start.h - nh, w: nw, h: nh }
        } else if (drag.mode === 'nw') {
          nb = { x: start.x + start.w - nw, y: start.y + start.h - nh, w: nw, h: nh }
        }
        // Clamp inside canvas
        nb.x = clamp(nb.x, 0, 1 - nb.w)
        nb.y = clamp(nb.y, 0, 1 - nb.h)
      }
      setBox(nb)
    }
    const handleUp = () => setDrag(null)
    window.addEventListener('pointermove', handleMove)
    window.addEventListener('pointerup', handleUp)
    return () => {
      window.removeEventListener('pointermove', handleMove)
      window.removeEventListener('pointerup', handleUp)
    }
  }, [drag, renderedSize, srcW, srcH, canvasPx, setBox])

  // Padding readout (in source-pixel-space, same as what we submit).
  const pads = useMemo(() => {
    if (canvasPx.w <= 0) return null
    return {
      top: Math.max(0, Math.round(box.y * canvasPx.h)),
      left: Math.max(0, Math.round(box.x * canvasPx.w)),
      bottom: Math.max(0, Math.round((1 - box.y - box.h) * canvasPx.h)),
      right: Math.max(0, Math.round((1 - box.x - box.w) * canvasPx.w)),
    }
  }, [box, canvasPx])

  const ASPECTS: { v: Aspect; label: string }[] = [
    { v: 'source', label: 'Source' },
    { v: '16:9', label: '16:9' },
    { v: '9:16', label: '9:16' },
    { v: '1:1', label: '1:1' },
    { v: '4:3', label: '4:3' },
    { v: '3:4', label: '3:4' },
  ]

  return (
    <div className="space-y-2">
      {/* Canvas */}
      <div
        ref={canvasRef}
        className="relative bg-black rounded-md overflow-hidden border border-border select-none touch-none"
        style={aspectStyle}
      >
        {/* Subtle padding-region tint to make the outpaint area visible */}
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,rgba(255,255,255,0.03),transparent_70%)] pointer-events-none" />

        {/* Video preview */}
        <div
          className="absolute"
          style={{
            left: `${box.x * 100}%`,
            top: `${box.y * 100}%`,
            width: `${box.w * 100}%`,
            height: `${box.h * 100}%`,
          }}
        >
          <video
            src={videoUrl}
            muted
            loop
            autoPlay
            playsInline
            className="w-full h-full object-fill pointer-events-none"
            draggable={false}
          />
          {/* Move handle (the whole frame is draggable) */}
          <div
            className="absolute inset-0 cursor-move"
            onPointerDown={onPointerDown('move')}
          />
          {/* Frame border */}
          <div className="absolute inset-0 border-2 border-accent-blue/60 pointer-events-none" />
          {/* Corner resize handles */}
          {(['nw', 'ne', 'sw', 'se'] as const).map(corner => (
            <div
              key={corner}
              onPointerDown={onPointerDown(corner)}
              className={`absolute w-3 h-3 bg-accent-blue border border-white rounded-sm
                ${corner === 'nw' ? '-top-1.5 -left-1.5 cursor-nwse-resize' : ''}
                ${corner === 'ne' ? '-top-1.5 -right-1.5 cursor-nesw-resize' : ''}
                ${corner === 'sw' ? '-bottom-1.5 -left-1.5 cursor-nesw-resize' : ''}
                ${corner === 'se' ? '-bottom-1.5 -right-1.5 cursor-nwse-resize' : ''}`}
            />
          ))}
        </div>

        {/* Padding readout overlay */}
        {pads && (pads.top > 0 || pads.bottom > 0 || pads.left > 0 || pads.right > 0) && (
          <div className="absolute bottom-1 left-1 right-1 flex justify-between text-[9px] text-white/70 pointer-events-none font-mono">
            <span>{pads.left > 0 ? `← ${pads.left}px` : ''}</span>
            <span className="flex flex-col items-center">
              {pads.top > 0 && <span>↑ {pads.top}px</span>}
              {pads.bottom > 0 && <span>↓ {pads.bottom}px</span>}
            </span>
            <span>{pads.right > 0 ? `${pads.right}px →` : ''}</span>
          </div>
        )}
      </div>

      {/* Aspect ratio buttons */}
      <div className="grid grid-cols-6 gap-1">
        {ASPECTS.map(a => (
          <button
            key={a.v}
            onClick={() => setAspect(a.v)}
            className={`px-1 py-1 rounded text-[10px] border transition-colors ${
              aspect === a.v
                ? 'border-accent-blue bg-accent-blue/15 text-accent-blue'
                : 'border-border text-text-muted hover:border-border-light hover:text-text-secondary'
            }`}
          >
            {a.label}
          </button>
        ))}
      </div>

      {/* Output dimensions readout */}
      {canvasPx.w > 0 && (
        <p className="text-[9px] text-text-muted text-center font-mono">
          Canvas: {canvasPx.w}×{canvasPx.h}px
        </p>
      )}
    </div>
  )
}

function clamp(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, v))
}
