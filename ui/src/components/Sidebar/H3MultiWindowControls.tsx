import { Info } from 'lucide-react'
import { useStore } from '../../stores/useStore'

/**
 * Shared long-form controls for MiniMax H3 and every LTX video family.
 *
 * First / Last uses H3's native sliding-window continuation. Omni keeps its
 * canonical reference manifest in every pass and can either carry native
 * motion/audio overlap or create independent hard-cut clips. LTX uses its
 * existing native rolling-window path with the same explicit UX contract.
 */
export function H3MultiWindowControls() {
  const params = useStore(s => s.params)
  const modelOptions = useStore(s => s.modelOptions)
  const setParam = useStore(s => s.setParam)

  const isH3 = String(modelOptions?.architecture || '').startsWith('minimax_h3')
  const isLtx = modelOptions?.multi_window_sequence_controls === true
  if (!isH3 && !isLtx) return null

  const isOmni = isH3 && modelOptions?.omni_reference === true
  const enabled = isLtx
    ? params.ltx_multi_window === true
    : isOmni
      ? params.minimax_h3_reference_sequence === true
      : params.minimax_h3_multi_window === true
  const promptMode = isLtx
    ? (params.ltx_window_prompt_mode === 'manual' ? 'manual' : 'auto')
    : isOmni
      ? (params.minimax_h3_sequence_prompt_mode === 'manual' ? 'manual' : 'auto')
      : params.minimax_h3_sequence_prompt_mode === 'manual'
        ? 'manual'
        : params.minimax_h3_sequence_prompt_mode === 'auto'
          ? 'auto'
          : (params.minimax_h3_window_storyboard === false ? 'manual' : 'auto')

  const setEnabled = (checked: boolean) => {
    if (isLtx) {
      setParam('ltx_multi_window', checked)
    } else if (isOmni) {
      setParam('minimax_h3_reference_sequence', checked)
    } else {
      setParam('minimax_h3_multi_window', checked)
    }
  }

  const setPromptMode = (mode: 'auto' | 'manual') => {
    if (isLtx) {
      setParam('ltx_window_prompt_mode', mode)
    } else if (isOmni) {
      setParam('minimax_h3_sequence_prompt_mode', mode)
    } else {
      // Retain the legacy storyboard switch for backend compatibility while
      // also persisting the explicit UI choice in new sidecars.
      setParam('minimax_h3_sequence_prompt_mode', mode)
      setParam('minimax_h3_window_storyboard', mode === 'auto')
    }
  }

  return (
    <div className="rounded-lg border border-border bg-bg-tertiary/50 px-2.5 py-2 space-y-1.5">
      <label className="flex items-center gap-2 cursor-pointer">
        <input
          type="checkbox"
          checked={enabled}
          onChange={event => setEnabled(event.target.checked)}
          className="accent-accent-blue"
        />
        <span className="text-[10px] text-text-secondary">Multi-window sequence</span>
        {isOmni && (
          <span className="text-[8px] text-amber-300/90 border border-amber-400/30 rounded px-1 py-0.5">
            Experimental
          </span>
        )}
        <span
          title={isOmni
            ? 'Extends Duration beyond one native Omni pass. Canonical references remain available in every pass.'
            : isLtx
              ? 'Extends Duration beyond one native LTX pass. Motion and, on audio-capable LTX models, synchronized audio continue into each new window.'
              : 'Extends Duration beyond one native First / Last pass while carrying recent motion and synchronized audio into the next window.'}
          className="text-text-muted cursor-help"
        >
          <Info size={11} />
        </span>
      </label>

      {enabled && (
        <>
          <label className="flex items-center justify-between gap-2 text-[9px] text-text-muted">
            <span className="flex items-center gap-1">
              Window prompts
              <span
                title="Auto plan expands one overall idea with Maestro's LLM. Manual uses each non-empty line in the prompt box for the matching window."
                className="text-text-muted cursor-help"
              >
                <Info size={10} />
              </span>
            </span>
            <select
              value={promptMode}
              onChange={event => setPromptMode(
                event.target.value === 'manual' ? 'manual' : 'auto',
              )}
              className="min-w-[138px] rounded border border-border bg-bg-secondary px-2 py-1 text-[9px] text-text-secondary focus:outline-none focus:border-accent-blue"
            >
              <option value="auto">Auto plan (AI)</option>
              <option value="manual">Manual - one per line</option>
            </select>
          </label>

          {isOmni && (
            <label className="flex items-start gap-2 text-[9px] text-text-muted cursor-pointer">
              <input
                type="checkbox"
                checked={params.minimax_h3_sequence_continuity !== false}
                onChange={event => setParam('minimax_h3_sequence_continuity', event.target.checked)}
                className="accent-accent-blue mt-0.5"
              />
              <span>
                Carry motion and sound between windows
                <span className="block text-[8px] mt-0.5">
                  Uses native Ref2VA overlap for smooth continuation. Turn off for independent hard-cut clips.
                </span>
              </span>
            </label>
          )}
        </>
      )}
    </div>
  )
}
