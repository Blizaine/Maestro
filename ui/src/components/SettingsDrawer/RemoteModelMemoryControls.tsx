import { useId, useState } from 'react'
import { fetchRemoteLoadedModels, unloadRemoteModel, type RemoteLoadedModels } from '../../api/client'

export function RemoteModelMemoryControls({ modelId }: { modelId: string }) {
  const selectId = useId()
  const [snapshot, setSnapshot] = useState<RemoteLoadedModels | null>(null)
  const [instanceId, setInstanceId] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [message, setMessage] = useState('')

  const refresh = async () => {
    setBusy(true)
    setError('')
    setMessage('')
    setSnapshot(null)
    setInstanceId('')
    try {
      const result = await fetchRemoteLoadedModels()
      setSnapshot(result)
      const matches = result.instances.filter(item => modelId === item.instance_id || modelId === item.model_key)
      if (matches.length === 1) setInstanceId(matches[0].instance_id)
      if (!result.instances.length) setMessage('LM Studio has no language models loaded.')
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Could not list loaded models.')
    } finally {
      setBusy(false)
    }
  }

  const unload = async () => {
    if (!snapshot || !instanceId || busy) return
    setBusy(true)
    setError('')
    setMessage('')
    try {
      const result = await unloadRemoteModel({ instance_id: instanceId, server_url: snapshot.server_url })
      setSnapshot({ ...snapshot, instances: snapshot.instances.filter(item => item.instance_id !== instanceId) })
      setInstanceId('')
      setMessage(result.status === 'unloaded'
        ? 'LM Studio confirmed that the selected model instance was unloaded.'
        : 'The selected model instance is already unloaded.')
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Could not unload the selected model.')
      // An accepted request may have failed verification. Require a fresh list
      // before retrying so the button cannot act on an uncertain old selection.
      setSnapshot(null)
      setInstanceId('')
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="space-y-2 border border-border rounded-lg p-3">
      <div className="flex items-center justify-between gap-3">
        <span className="text-xs text-text-primary">LM Studio memory</span>
        <button
          type="button"
          disabled={busy}
          onClick={refresh}
          className="text-[10px] text-accent-blue hover:text-accent-blue-hover disabled:opacity-50"
        >
          {busy ? 'Working…' : 'Refresh loaded models'}
        </button>
      </div>
      <p className="text-[10px] text-text-muted">
        Free memory before image or video generation. Requires LM Studio 0.4 or newer.
        Select a loaded model to unload it from the configured server.
      </p>
      {!!snapshot?.instances.length && (
        <>
          <label htmlFor={selectId} className="text-[11px] text-text-muted block">Loaded model instance</label>
          <select
            id={selectId}
            value={instanceId}
            disabled={busy}
            onChange={event => setInstanceId(event.target.value)}
            className="w-full bg-bg-tertiary border border-border rounded-lg px-3 py-2 text-xs text-text-primary focus:outline-none focus:border-accent-blue"
          >
            <option value="">Choose an instance</option>
            {snapshot.instances.map(item => <option key={item.instance_id} value={item.instance_id}>
              {item.display_name} ({item.instance_id})
            </option>)}
          </select>
          <button
            type="button"
            disabled={busy || !instanceId}
            onClick={unload}
            className="px-3 py-1.5 text-xs text-text-secondary border border-border rounded-lg hover:text-text-primary disabled:opacity-50"
          >
            Unload selected model
          </button>
        </>
      )}
      {error && <p role="alert" className="text-[10px] text-red-400">{error}</p>}
      {message && <p role="status" className="text-[10px] text-text-secondary">{message}</p>}
    </div>
  )
}
