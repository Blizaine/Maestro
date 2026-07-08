import { useState } from 'react'
import { BookMarked, Loader2 } from 'lucide-react'

/**
 * SaveRecipeDialog — turns the current gallery output into a reusable
 * recipe. The output's sidecar supplies model + LoRAs + settings and the
 * media supplies the thumbnail; the user just names it.
 */
export function SaveRecipeDialog({ defaultNsfw, onSave, onCancel }: {
  defaultNsfw: boolean
  onSave: (name: string, description: string, nsfw: boolean) => Promise<void>
  onCancel: () => void
}) {
  const [name, setName] = useState('')
  const [description, setDescription] = useState('')
  const [nsfw, setNsfw] = useState(defaultNsfw)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const submit = async () => {
    if (!name.trim() || saving) return
    setSaving(true); setError(null)
    try {
      await onSave(name.trim(), description.trim(), nsfw)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Save failed')
      setSaving(false)
    }
  }

  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/60 p-4" onClick={onCancel}>
      <div className="bg-bg-secondary border border-border rounded-xl shadow-2xl w-[420px] max-w-[94vw] p-5"
        onClick={e => e.stopPropagation()}>
        <div className="flex items-center gap-2 mb-3">
          <BookMarked size={16} className="text-accent-blue" />
          <h2 className="text-sm font-semibold text-text-primary">Save as Recipe</h2>
        </div>
        <p className="text-[11px] text-text-muted mb-3 leading-snug">
          Captures this generation's model, LoRAs, and settings as a one-click
          preset. Its thumbnail comes from this output. Applying a recipe later
          prepopulates the prompt so you just edit the subject.
        </p>

        <label className="text-[10px] text-text-muted uppercase tracking-wider mb-1 block">Name</label>
        <input
          autoFocus
          value={name}
          onChange={e => setName(e.target.value)}
          onKeyDown={e => { if (e.key === 'Enter') submit() }}
          placeholder="e.g. Cinematic Film Look"
          className="w-full bg-bg-tertiary border border-border rounded-lg px-3 py-2 text-sm text-text-primary focus:outline-none focus:border-accent-blue mb-3"
        />

        <label className="text-[10px] text-text-muted uppercase tracking-wider mb-1 block">Description (optional)</label>
        <textarea
          value={description}
          onChange={e => setDescription(e.target.value)}
          placeholder="When to use it, what it's good for…"
          rows={2}
          className="w-full bg-bg-tertiary border border-border rounded-lg px-3 py-2 text-sm text-text-primary focus:outline-none focus:border-accent-blue resize-none mb-3"
        />

        <label className="flex items-center gap-2 cursor-pointer mb-4">
          <input type="checkbox" checked={nsfw} onChange={e => setNsfw(e.target.checked)} className="accent-accent-blue" />
          <span className="text-[11px] text-text-secondary">Mature recipe (hidden unless mature mode is on)</span>
        </label>

        {error && <div className="text-[11px] text-red-400 mb-3">{error}</div>}

        <div className="flex items-center justify-end gap-2">
          <button onClick={onCancel} disabled={saving}
            className="px-4 py-2 text-xs text-text-secondary hover:text-text-primary border border-border rounded-lg hover:border-border-light transition-colors disabled:opacity-40">
            Cancel
          </button>
          <button onClick={submit} disabled={!name.trim() || saving}
            className="px-4 py-2 text-xs bg-accent-blue text-white rounded-lg hover:bg-accent-blue-hover transition-colors disabled:opacity-40 flex items-center gap-1.5">
            {saving ? <><Loader2 size={12} className="animate-spin" /> Saving…</> : 'Save Recipe'}
          </button>
        </div>
      </div>
    </div>
  )
}
