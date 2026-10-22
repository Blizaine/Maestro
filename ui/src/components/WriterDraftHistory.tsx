import { useState } from 'react'

export interface WriterDraft {
  index: number
  status: string
  response_text: string
  response_complete?: boolean
  error?: string
  model_id?: string
  rejections?: Array<{stage: string; reasons: string[]}>
}

export function WriterDraftHistory({drafts}: {drafts: WriterDraft[]}) {
  if (!drafts.length) return null
  return <details className="my-3 rounded-lg border border-border p-3">
    <summary className="cursor-pointer text-xs font-medium">Writer drafts ({drafts.length})</summary>
    <p className="my-2 text-[11px] text-text-muted">Received drafts are intermediate writer responses, not approval of their content.</p>
    {drafts.map(draft => <details key={draft.index} className="mb-2">
      <summary className="cursor-pointer text-xs">Draft {draft.index} · {draft.status}{draft.response_complete === false ? ' · incomplete response' : ''}</summary>
      {draft.rejections?.map((decision, index) => <div key={index} className="my-2 text-xs text-indicator-warning">
        <p>{decision.stage}</p>
        <ul className="list-disc pl-4">{decision.reasons.map((reason, reasonIndex) => <li key={reasonIndex}>{reason}</li>)}</ul>
      </div>)}
      {draft.error && <p className="my-2 text-xs text-indicator-warning">{draft.error}</p>}
      <pre className="max-h-96 overflow-auto whitespace-pre-wrap break-words text-xs leading-relaxed">{draft.response_text || '(No response text received)'}</pre>
    </details>)}
  </details>
}

export function LoadWriterDraftHistory({load}: {load: () => Promise<WriterDraft[]>}) {
  const [drafts, setDrafts] = useState<WriterDraft[] | null>(null)
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  return <div>
    <button type="button" disabled={busy} className="my-2 text-xs text-accent-blue hover:underline disabled:opacity-40"
      onClick={async () => {
        setBusy(true); setError('')
        try { setDrafts(await load()) }
        catch (reason) { setError(reason instanceof Error ? reason.message : String(reason)) }
        finally { setBusy(false) }
      }}>{busy ? 'Loading writer drafts…' : 'Load saved writer drafts and rejection reasons'}</button>
    {error && <p role="alert" className="text-xs text-indicator-warning">{error}</p>}
    {drafts && (drafts.length ? <WriterDraftHistory drafts={drafts}/> : <p className="text-xs text-text-muted">No saved writer drafts. Older runs did not retain this history.</p>)}
  </div>
}
