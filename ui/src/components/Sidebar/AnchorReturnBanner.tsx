import { ArrowLeft, X, Check, SkipForward } from 'lucide-react'
import { useStore } from '../../stores/useStore'

/**
 * Persistent banner that drives the Edit Anything → Image Mode round-trip
 * for a single boundary anchor at a time. Mounted at the top of the
 * sidebar whenever `editReturnTarget` is set.
 *
 * Each round-trip is independent — start and end anchors are populated
 * by separate "Edit X in Image Mode" buttons in EditAnythingControls.
 *
 * Actions:
 *   - Apply: take the most recent Image-mode output and store it as the
 *     anchor identified by editReturnTarget.anchor, then return to Edit
 *     Anything mode.
 *   - Skip: return to Edit Anything without setting the anchor. The
 *     model will fall back to extracting the source frame at generation
 *     time (the morph-from-source default).
 *   - Cancel (×): same as Skip — return without applying.
 */
export function AnchorReturnBanner() {
  const target = useStore(s => s.editReturnTarget)
  const outputs = useStore(s => s.outputs)
  const apply = useStore(s => s.applyOutputAsAnchor)
  const skip = useStore(s => s.skipAnchorPhase)
  const cancel = useStore(s => s.cancelAnchorReturn)

  if (!target) return null

  const anchorLabel = target.anchor === 'start' ? 'Start' : 'End'

  // Latest image output (newest first, type === 'image')
  const latestImage = outputs.find(o => o.type === 'image')
  const hasLatestImage = !!latestImage

  return (
    <div className="px-3 py-2 bg-accent-blue/10 border-b border-accent-blue/30">
      <div className="flex items-center gap-2 mb-1.5">
        <ArrowLeft size={12} className="text-accent-blue shrink-0" />
        <span className="text-[10px] font-semibold text-accent-blue">
          Editing {anchorLabel} Anchor
        </span>
        <button
          onClick={cancel}
          title="Cancel — return to Edit Anything without setting this anchor"
          className="ml-auto p-0.5 rounded hover:bg-accent-blue/20 text-accent-blue/80 hover:text-accent-blue"
        >
          <X size={11} />
        </button>
      </div>
      <p className="text-[9px] text-text-muted leading-snug mb-2">
        Edit the {anchorLabel.toLowerCase()} frame in Image Mode. Apply your
        result to use it as the {anchorLabel.toLowerCase()} boundary anchor,
        or skip to fall back to the source frame.
      </p>
      <div className="flex gap-1.5">
        <button
          onClick={() => void apply()}
          disabled={!hasLatestImage}
          className="flex-1 flex items-center justify-center gap-1 px-2 py-1 rounded bg-accent-blue text-white hover:bg-accent-blue/90 disabled:opacity-40 disabled:cursor-not-allowed text-[10px] transition-colors"
        >
          <Check size={11} />
          Apply &amp; return
        </button>
        <button
          onClick={skip}
          title={`Skip ${anchorLabel} anchor — fall back to source frame`}
          className="flex items-center justify-center gap-1 px-2 py-1 rounded border border-border text-text-secondary hover:bg-bg-hover text-[10px] transition-colors"
        >
          <SkipForward size={11} />
          Skip
        </button>
      </div>
      {!hasLatestImage && (
        <p className="text-[9px] text-text-muted mt-1.5 italic">
          Generate an image first, then click Apply.
        </p>
      )}
    </div>
  )
}
