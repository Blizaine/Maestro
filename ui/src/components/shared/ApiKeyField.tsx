import { useId, useState } from 'react'

export function ApiKeyField({ label, maskedValue, isSet, onSave }: {
  label: string
  maskedValue: string
  isSet: boolean
  onSave: (value: string) => Promise<void>
}) {
  const id = useId()
  const [editing, setEditing] = useState(false)
  const [value, setValue] = useState('')
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState('')

  const save = async () => {
    if (saving) return
    setSaving(true)
    setError('')
    try {
      await onSave(value)
      setEditing(false)
      setValue('')
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not save the API key. Please try again.')
    } finally {
      setSaving(false)
    }
  }

  return (
    <div>
      <label htmlFor={editing ? id : undefined} className="text-[11px] text-text-muted uppercase tracking-wider mb-1.5 block">
        {label}
      </label>
      {editing ? (
        <div className="flex gap-2">
          <input
            id={id}
            type="password"
            value={value}
            disabled={saving}
            aria-invalid={!!error}
            aria-describedby={error ? `${id}-error` : undefined}
            onChange={e => { setValue(e.target.value); setError('') }}
            onKeyDown={e => { if (e.key === 'Enter') { e.preventDefault(); void save() } }}
            placeholder="Paste API key..."
            className="flex-1 min-w-0 bg-bg-tertiary border border-border rounded-lg px-3 py-2 text-sm text-text-primary focus:outline-none focus:border-accent-blue"
            autoFocus
          />
          <button onClick={() => void save()} disabled={saving}
            className="px-3 py-2 bg-accent-blue text-white text-xs rounded-lg hover:bg-accent-blue-hover disabled:opacity-50">
            {saving ? 'Saving…' : 'Save'}
          </button>
          <button onClick={() => { setEditing(false); setValue(''); setError('') }} disabled={saving}
            className="px-3 py-2 border border-border text-xs rounded-lg text-text-secondary hover:text-text-primary disabled:opacity-50">
            Cancel
          </button>
        </div>
      ) : (
        <div className="flex gap-2 items-center">
          <div className="flex-1 bg-bg-tertiary border border-border rounded-lg px-3 py-2 text-sm text-text-muted font-mono">
            {isSet ? maskedValue : 'Not set'}
          </div>
          <button onClick={() => { setError(''); setEditing(true) }}
            className="px-3 py-2 border border-border text-xs rounded-lg text-text-secondary hover:text-text-primary hover:border-border-light transition-colors">
            {isSet ? 'Change' : 'Set'}
          </button>
        </div>
      )}
      {error && <p id={`${id}-error`} role="alert" className="text-xs text-red-400 mt-2">{error}</p>}
    </div>
  )
}
