import { useState } from 'react'
import { Sparkles, Download, Cpu, ShieldAlert, X } from 'lucide-react'

const SEEN_KEY = 'maestro_welcome_seen_v1'

/**
 * WelcomeModal — a one-time first-run intro. Sets the expectations that
 * most surprise new users (model weights download on first use, not at
 * install; Director is planned by a local LLM; mature mode is an opt-in
 * in Settings). Shown once ever, tracked in localStorage.
 *
 * Deliberately not tied to any backend call — it's pure orientation, so
 * it can render instantly on first paint.
 */
export function WelcomeModal() {
  const [open, setOpen] = useState(() => localStorage.getItem(SEEN_KEY) !== '1')

  if (!open) return null

  const dismiss = () => {
    localStorage.setItem(SEEN_KEY, '1')
    setOpen(false)
  }

  return (
    <div className="fixed inset-0 z-[120] flex items-center justify-center bg-black/60 p-4" onClick={dismiss}>
      <div
        className="bg-bg-secondary border border-border rounded-2xl shadow-2xl w-[520px] max-w-[94vw] max-h-[88vh] overflow-y-auto"
        onClick={e => e.stopPropagation()}
      >
        {/* Header */}
        <div className="px-6 pt-6 pb-4 flex items-start gap-3">
          <div className="w-11 h-11 rounded-xl bg-accent-blue/15 flex items-center justify-center shrink-0">
            <Sparkles size={22} className="text-accent-blue" />
          </div>
          <div className="flex-1">
            <h2 className="text-base font-semibold text-text-primary">Welcome to Maestro</h2>
            <p className="text-xs text-text-muted mt-0.5">
              A local AI studio for video, images, and music — including a Director mode
              that plans a whole music video or short film from a sentence.
            </p>
          </div>
          <button onClick={dismiss} className="p-1 rounded text-text-muted hover:text-text-primary" aria-label="Close">
            <X size={16} />
          </button>
        </div>

        {/* Points */}
        <div className="px-6 pb-2 space-y-3.5">
          <Row icon={<Download size={16} className="text-accent-blue" />} title="Models download on first use">
            The install set up the app, not the model weights. The first time you use a
            given model, its files download once — often <span className="text-text-secondary">tens of GB</span> for
            a video model — before anything appears. Later runs are fast. Watch progress
            at the bottom-right; the very first video can take a while.
          </Row>
          <Row icon={<Cpu size={16} className="text-accent-blue" />} title="Director runs on a local LLM">
            Director mode plans your video with a local language model (Gemma 4, ~5 GB,
            downloaded on first Director use) — nothing is sent to the cloud. You can
            switch models, or plug in an external API, under Settings → Services.
          </Row>
          <Row icon={<ShieldAlert size={16} className="text-red-400" />} title="Mature mode is off by default">
            Adult content generation is an explicit opt-in behind a disclaimer in
            Settings → Services. Leave it off and content stays PG-13.
          </Row>
        </div>

        {/* Footer */}
        <div className="px-6 py-4 flex items-center justify-end gap-3">
          <button
            onClick={dismiss}
            className="px-5 py-2 text-xs bg-accent-blue text-white rounded-lg hover:bg-accent-blue-hover transition-colors font-medium"
          >
            Get started
          </button>
        </div>
      </div>
    </div>
  )
}

function Row({ icon, title, children }: { icon: React.ReactNode; title: string; children: React.ReactNode }) {
  return (
    <div className="flex items-start gap-3">
      <div className="w-8 h-8 rounded-lg bg-bg-tertiary/60 flex items-center justify-center shrink-0 mt-0.5">
        {icon}
      </div>
      <div className="flex-1">
        <div className="text-xs font-medium text-text-primary">{title}</div>
        <div className="text-[11px] text-text-muted mt-0.5 leading-relaxed">{children}</div>
      </div>
    </div>
  )
}
