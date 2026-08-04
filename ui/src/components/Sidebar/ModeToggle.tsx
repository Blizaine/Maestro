import { useEffect } from 'react'
import { useStore } from '../../stores/useStore'

/** Studio Video sub-modes and what a model must support to run them.
 *
 *  Frames     — text-only or start/end frame conditioning. Every video model.
 *  Multi-Shot — one generation per clip, each optionally anchored by a start
 *               image, so the model must accept "S".
 *  Extend     — continues an existing clip from a video source ("V", or a
 *               model that declares video_continuation).
 *  Blend      — generates a transition pinned by a first AND last frame, so
 *               the model must accept both "S" and "E".
 *
 *  Without this every model is offered all four tabs, including ones whose
 *  conditioning it cannot accept. MiniMax H3 Ref2VA allows only "T" — it
 *  conditions on reference images and audio, not on timeline positions — so
 *  it lands on Frames alone rather than showing three tabs that would fail
 *  at generation time.
 */
interface ModeCaps {
  allowed: string
  videoContinuation: boolean
}

const MODES = [
  { value: 0, label: 'Frames', supported: () => true },
  { value: 2, label: 'Multi-Shot', supported: (c: ModeCaps) => c.allowed.includes('S') },
  { value: 3, label: 'Extend', supported: (c: ModeCaps) => c.allowed.includes('V') || c.videoContinuation },
  { value: 4, label: 'Blend', supported: (c: ModeCaps) => c.allowed.includes('S') && c.allowed.includes('E') },
]

export function ModeToggle() {
  const imageMode = useStore(s => s.params.image_mode)
  const setParam = useStore(s => s.setParam)
  const modelOptions = useStore(s => s.modelOptions)

  // A backend that doesn't send the letters is treated as "everything
  // allowed", so this can only ever hide a tab that was known to be unusable.
  const allowed = modelOptions?.image_prompt_types_allowed ?? 'TSEVL'
  const caps: ModeCaps = { allowed, videoContinuation: modelOptions?.video_continuation ?? false }
  const modes = MODES.filter(m => m.supported(caps))

  // Switching to a model that can't do the active sub-mode would otherwise
  // strand the sidebar on a tab with no button left to leave it by.
  useEffect(() => {
    if (!modes.some(m => m.value === imageMode)) setParam('image_mode', 0)
  }, [allowed, imageMode]) // eslint-disable-line react-hooks/exhaustive-deps

  if (modes.length <= 1) return null

  return (
    <div className="flex bg-bg-tertiary rounded-lg p-0.5 border border-border">
      {modes.map(m => (
        <button
          key={m.value}
          onClick={() => setParam('image_mode', m.value)}
          className={`flex-1 text-xs py-1.5 rounded-md transition-all ${
            imageMode === m.value
              ? 'bg-bg-active text-text-primary'
              : 'text-text-secondary hover:text-text-primary'
          }`}
        >
          {m.label}
        </button>
      ))}
    </div>
  )
}
