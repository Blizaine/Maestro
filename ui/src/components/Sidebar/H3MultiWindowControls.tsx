import { Info } from 'lucide-react'
import { useStore } from '../../stores/useStore'
import type { WindowPromptMode } from '../../types'

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
  const explicitPromptMode = isLtx
    ? params.ltx_window_prompt_mode
    : params.minimax_h3_sequence_prompt_mode
  const promptMode = isLtx
    ? (explicitPromptMode === 'auto' || explicitPromptMode === 'creative' || explicitPromptMode === 'manual'
        ? explicitPromptMode
        : enabled ? 'auto' : 'manual')
    : isOmni
      ? (explicitPromptMode === 'auto' || explicitPromptMode === 'creative' || explicitPromptMode === 'manual'
          ? explicitPromptMode
          : enabled ? 'auto' : 'manual')
      : explicitPromptMode === 'manual'
        ? 'manual'
        : explicitPromptMode === 'creative'
          ? 'creative'
          : explicitPromptMode === 'auto'
            ? 'auto'
            : enabled
              ? (params.minimax_h3_window_storyboard === false ? 'manual' : 'auto')
              : 'manual'

  const setPromptMode = (mode: WindowPromptMode) => {
    if (isLtx) {
      setParam('ltx_window_prompt_mode', mode)
    } else if (isOmni) {
      setParam('minimax_h3_sequence_prompt_mode', mode)
    } else {
      // Retain the legacy storyboard switch for backend compatibility while
      // also persisting the explicit UI choice in new sidecars.
      setParam('minimax_h3_sequence_prompt_mode', mode)
      setParam('minimax_h3_window_storyboard', mode !== 'manual')
    }
  }

  return (
    <div className="rounded-lg border border-border bg-bg-tertiary/50 px-2.5 py-2 space-y-1.5">
      <div className="flex items-center gap-2">
        <span className="text-[10px] text-text-secondary">
          {enabled ? 'Long sequence · automatic' : 'Prompt writing'}
        </span>
        <span
          title={isOmni
            ? 'Duration automatically becomes a native Omni sequence when it exceeds one model/GPU-aware window. Canonical references remain available in every pass.'
            : isLtx
              ? 'Duration automatically becomes a rolling LTX sequence when it exceeds one window. Motion and synchronized audio continue into each new window.'
              : 'Duration automatically becomes a First / Last sequence when it exceeds one window, carrying recent motion and synchronized audio forward.'}
          className="text-text-muted cursor-help"
        >
          <Info size={11} />
        </span>
      </div>

      <label className="flex items-center justify-between gap-2 text-[9px] text-text-muted">
        <span className="flex items-center gap-1">
          {enabled ? 'Window prompts' : 'Prompt writing'}
          <span
            title={enabled
              ? "AI - Faithful preserves and distributes only your supplied events and dialogue. AI - Creative treats your prompt as a brief and writes a complete scene with story beats, camera coverage, and dialogue. Manual uses each non-empty line for the matching window."
              : "AI - Faithful expands only what you supplied. AI - Creative may invent supporting story beats and dialogue. Manual sends the visible prompt as written."}
            className="text-text-muted cursor-help"
          >
            <Info size={10} />
          </span>
        </span>
        <select
          value={promptMode}
          onChange={event => setPromptMode(
            event.target.value === 'manual'
              ? 'manual'
              : event.target.value === 'creative'
                ? 'creative'
                : 'auto',
          )}
          className="min-w-[158px] rounded border border-border bg-bg-secondary px-2 py-1 text-[9px] text-text-secondary focus:outline-none focus:border-accent-blue"
        >
          <option value="auto">AI - Faithful</option>
          <option value="creative">AI - Creative story + dialogue</option>
          <option value="manual">{enabled ? 'Manual - one per line' : 'Manual'}</option>
        </select>
      </label>

      {!enabled && promptMode !== 'manual' && (
        <p className="text-[8px] leading-relaxed text-text-muted">
          {promptMode === 'creative'
            ? 'Maestro treats your prompt as a creative brief and writes the scene automatically. Exact quoted lines stay locked; add “only these lines” when no extra dialogue should be written.'
            : 'Maestro expands this prompt without inventing new story events or dialogue.'}{' '}
          Queued prompts wait for their own turn so the LLM never competes with an active generation. Use the sparkle button first only when you want to review the result.
        </p>
      )}

      {enabled && isOmni && (
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
    </div>
  )
}
