import { ArrowLeft, X, Check, SkipForward } from 'lucide-react'
import { useStore } from '../../stores/useStore'

/**
 * Persistent banner that drives Edit Anything/Recast/Repaint → Image Mode
 * round-trips for one boundary anchor or reference at a time.
 * Mounted at the top of the sidebar whenever `editReturnTarget` is set.
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

  const isRecast = target.anchor === 'recast'
  const isRepaint = target.anchor === 'repaint'
  const anchorLabel = target.anchor === 'start'
    ? 'Start'
    : target.anchor === 'end'
      ? 'End'
      : isRepaint
        ? 'Repaint First Frame'
        : 'Recast Reference'

  // Latest image output (newest first, type === 'image')
  const latestImage = outputs.find(o => o.type === 'image')
  const hasLatestImage = !!latestImage

  return (
    <div className="px-3 py-2 bg-accent-blue/10 border-b border-accent-blue/30">
      <div className="flex items-center gap-2 mb-1.5">
        <ArrowLeft size={12} className="text-accent-blue shrink-0" />
        <span className="text-[10px] font-semibold text-accent-blue">
          {isRecast
            ? 'Editing Recast Reference'
            : isRepaint
              ? 'Editing Repaint First Frame'
              : `Editing ${anchorLabel} Anchor`}
        </span>
        <button
          onClick={cancel}
          title={isRecast
            ? 'Cancel — return to Recast without changing its reference'
            : isRepaint
              ? 'Cancel — return to Repaint without changing its edited frame'
              : 'Cancel — return to Edit Anything without setting this anchor'}
          className="ml-auto p-0.5 rounded hover:bg-accent-blue/20 text-accent-blue/80 hover:text-accent-blue"
        >
          <X size={11} />
        </button>
      </div>
      <p className="text-[9px] text-text-muted leading-snug mb-2">
        {isRecast
          ? 'Edit the selected trim-start frame in Image Mode. Apply the result to use it as Recast’s replacement reference, or return unchanged.'
          : isRepaint
            ? 'Repaint the selected first frame in Image Mode. Apply the result to animate that exact new scene with the source video’s motion.'
            : `Edit the ${anchorLabel.toLowerCase()} frame in Image Mode. Apply your result to use it as the ${anchorLabel.toLowerCase()} boundary anchor, or skip to fall back to the source frame.`}
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
          title={isRecast
            ? 'Return to Recast without changing its reference'
            : isRepaint
              ? 'Return to Repaint without changing its edited frame'
              : `Skip ${anchorLabel} anchor — fall back to source frame`}
          className="flex items-center justify-center gap-1 px-2 py-1 rounded border border-border text-text-secondary hover:bg-bg-hover text-[10px] transition-colors"
        >
          <SkipForward size={11} />
          {isRecast || isRepaint ? 'Return unchanged' : 'Skip'}
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
