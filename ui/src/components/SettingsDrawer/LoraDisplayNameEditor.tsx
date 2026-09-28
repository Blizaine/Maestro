import { useState } from 'react'
import { Check, Loader2, Pencil, RotateCcw, X } from 'lucide-react'
import { updateLoraDisplayName, type LoraDisplayNameResult, type LoraDisplayNameScope } from '../../api/client'
import { notifyLoraDisplayNameChanged } from '../../lib/loraDisplayNames'

export function LoraDisplayNameEditor({ filename, displayName, displayNameOverride, modelType, directory, onSaved }: {
  filename: string
  displayName: string
  displayNameOverride?: string | null
  modelType?: string
  directory?: string
  onSaved: (result: LoraDisplayNameResult) => void
}) {
  const [editing, setEditing] = useState(false)
  const [draft, setDraft] = useState(displayName)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState('')
  const scope: LoraDisplayNameScope | null = modelType
    ? { modelType }
    : directory ? { directory } : null
  const hasScope = scope !== null

  const begin = (event: React.MouseEvent) => {
    event.stopPropagation()
    setDraft(displayName)
    setError('')
    setEditing(true)
  }
  const cancel = (event?: React.SyntheticEvent) => {
    event?.stopPropagation()
    setEditing(false)
    setError('')
  }
  const save = async (value: string | null, event?: React.SyntheticEvent) => {
    event?.stopPropagation()
    if (saving || !hasScope) return
    const clean = value?.trim() || null
    if (value !== null && !clean) {
      setError('Enter a display name or reset to automatic.')
      return
    }
    setSaving(true)
    setError('')
    try {
      if (!scope) return
      const result = await updateLoraDisplayName(filename, clean, scope)
      onSaved(result)
      notifyLoraDisplayNameChanged({ model_type: modelType, directory })
      setEditing(false)
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason))
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="relative inline-flex shrink-0 items-center" onClick={event => event.stopPropagation()}
      onKeyDown={event => event.stopPropagation()}>
      {!editing ? (
        <button type="button" disabled={!hasScope} onClick={begin}
          className="rounded p-0.5 text-text-muted opacity-50 hover:bg-bg-hover hover:text-text-primary hover:opacity-100 focus-visible:opacity-100 disabled:cursor-not-allowed"
          aria-label={`Edit display name for ${displayName}`} title={`Edit display name · file: ${filename}`}>
          <Pencil size={11}/>
        </button>
      ) : (
        <div className="inline-flex items-center gap-0.5 rounded-md border border-border bg-bg-secondary p-0.5 shadow-lg">
          <input autoFocus type="text" value={draft} maxLength={120}
            onChange={event => setDraft(event.target.value)}
            onKeyDown={event => {
              if (event.key === 'Enter') { event.preventDefault(); void save(draft, event) }
              if (event.key === 'Escape') { event.preventDefault(); cancel(event) }
            }}
            disabled={saving} aria-label={`Display name for ${filename}`} title={`Original filename: ${filename}`}
            className="w-16 min-w-0 rounded border border-border bg-bg-tertiary px-1.5 py-1 text-[10px] text-text-primary focus:border-accent-blue focus:outline-none disabled:opacity-50 sm:w-24" />
          <button type="button" disabled={saving || !draft.trim()} onClick={event => void save(draft, event)}
            aria-label="Save LoRA display name" title="Save display name"
            className="rounded p-1 text-text-secondary hover:bg-bg-hover hover:text-indicator-success disabled:opacity-40">
            {saving ? <Loader2 size={11} className="animate-spin"/> : <Check size={11}/>}
          </button>
          <button type="button" disabled={saving || !displayNameOverride} onClick={event => void save(null, event)}
            aria-label="Reset automatic LoRA name" title="Reset to automatic name"
            className="rounded p-1 text-text-secondary hover:bg-bg-hover hover:text-accent-blue disabled:opacity-40">
            <RotateCcw size={11}/>
          </button>
          <button type="button" disabled={saving} onClick={cancel} aria-label="Cancel display name edit"
            title="Cancel" className="rounded p-1 text-text-secondary hover:bg-bg-hover hover:text-text-primary disabled:opacity-40">
            <X size={11}/>
          </button>
          {error && <span role="alert" className="absolute right-0 top-full z-40 mt-1 w-44 rounded-md border border-red-500/30 bg-bg-secondary p-1.5 text-[10px] leading-snug text-red-400 shadow-xl">{error}</span>}
        </div>
      )}
    </div>
  )
}
