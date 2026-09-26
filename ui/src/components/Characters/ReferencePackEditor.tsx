import { useEffect, useRef, useState } from 'react'
import * as api from '../../api/client'
import type { ReferencePackView, SavedOmniCharacter } from '../../types'
import { PACK_VIEWS } from '../../lib/referencePacks'
import { SidebarDialog } from '../Sidebar/SidebarPanels'

const field = 'w-full min-w-0 rounded-lg border border-border bg-bg-primary p-2 text-base sm:text-xs text-text-primary'
const button = 'min-h-10 rounded-lg border border-border px-3 py-2 text-xs text-accent-blue hover:bg-bg-hover disabled:opacity-40'

export function ReferencePackButton({ character, disabled = false }: { character?: SavedOmniCharacter; disabled?: boolean }) {
  const [open, setOpen] = useState(false)
  return <>
    <button type="button" disabled={disabled} onClick={() => setOpen(true)} className={button}>
      {character ? 'Pack views & revisions' : 'Import five-view pack'}
    </button>
    {open && <ReferencePackEditor initial={character} onClose={() => setOpen(false)} />}
  </>
}

function FilePreview({ file }: { file: File }) {
  const image = useRef<HTMLImageElement>(null)
  useEffect(() => {
    const value = URL.createObjectURL(file)
    if (image.current) image.current.src = value
    return () => URL.revokeObjectURL(value)
  }, [file])
  return <img ref={image} alt="Selected replacement" className="h-40 w-full object-contain" />
}

export function ReferencePackEditor({ initial, onClose }: { initial?: SavedOmniCharacter; onClose: () => void }) {
  const [character, setCharacter] = useState(initial)
  const [name, setName] = useState(initial?.name || '')
  const [selectedId, setSelectedId] = useState(initial?.reference_pack?.revisions.at(-1)?.id || '')
  const [editing, setEditing] = useState(!initial)
  const [files, setFiles] = useState<Partial<Record<ReferencePackView, File>>>({})
  const [label, setLabel] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const revisions = character?.reference_pack?.revisions || []
  const selected = revisions.find(item => item.id === selectedId)
  const changed = Object.keys(files).length
  const apply = (result: SavedOmniCharacter) => {
    setCharacter(result)
    window.dispatchEvent(new Event('maestro-characters-changed'))
  }
  const save = async () => {
    setBusy(true); setError('')
    try {
      const images: Record<string, string> = {}
      for (const [view, file] of Object.entries(files)) images[view] = (await api.uploadImage(file)).path
      const result = await api.saveReferencePack({ name, images, label, base_revision_id: selected?.id }, character?.id)
      apply(result); setSelectedId(result.reference_pack!.revisions.at(-1)!.id)
      setFiles({}); setLabel(''); setEditing(false)
    } catch (err) { setError(err instanceof Error ? err.message : 'Could not import pack.') }
    finally { setBusy(false) }
  }
  const review = async () => {
    if (!character || !selected) return
    setBusy(true); setError('')
    try { apply(await api.reviewReferencePack(character.id, selected.id, !selected.approved)) }
    catch (err) { setError(err instanceof Error ? err.message : 'Could not save review.') }
    finally { setBusy(false) }
  }
  return <SidebarDialog open title={character ? `${character.name} — Reference Pack` : 'Import five-view Reference Pack'} variant="center" onClose={() => { if (!busy) onClose() }}>
    <div className="p-3 space-y-4">
      <p className="text-xs text-text-secondary">Five complementary views of one character. An approved revision uses five H3 image slots. Review face, proportions, hair and outfit across every view before approving.</p>
      {!character && <label className="block text-xs text-text-secondary">Character name<input aria-label="Pack character name" value={name} onChange={event => setName(event.target.value)} disabled={busy} className={field} /></label>}
      {revisions.length > 0 && <label className="block text-xs text-text-secondary">{editing ? 'Base revision (unchanged views are copied)' : 'Review revision'}
        <select aria-label="Pack revision" value={selectedId} disabled={busy || editing} onChange={event => { setSelectedId(event.target.value); setError('') }} className={field}>
          {revisions.map(revision => <option key={revision.id} value={revision.id}>v{revision.number} · {revision.label || 'Untitled'} · {revision.approved ? 'Approved' : 'Draft'}{revision.parent_revision_id ? ` · from v${revisions.find(parent => parent.id === revision.parent_revision_id)?.number}` : ''}</option>)}
        </select>
      </label>}
      {editing && <label className="block text-xs text-text-secondary">Revision label<input aria-label="Revision label" value={label} onChange={event => setLabel(event.target.value)} disabled={busy} placeholder="e.g. Blue jacket, corrected face" className={field} /></label>}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
        {PACK_VIEWS.map(([view, title]) => <div key={view} className="min-w-0 rounded-lg border border-border bg-bg-primary p-2 space-y-2">
          <p className="text-xs font-medium text-text-primary">{title}</p>
          {files[view] ? <FilePreview file={files[view]!} /> : selected && <a href={selected.images[view].url} target="_blank" rel="noreferrer" title={`Open ${title} full size`}><img src={selected.images[view].url} alt={title} className="h-40 w-full object-contain" /></a>}
          {editing && <input aria-label={title} type="file" accept=".png,.jpg,.jpeg,.webp,.bmp,.tif,.tiff" disabled={busy} onChange={event => setFiles(current => { const next = { ...current }; if (event.target.files?.[0]) next[view] = event.target.files[0]; else delete next[view]; return next })} className="w-full text-xs text-text-secondary" />}
          {editing && selected && !files[view] && <p className="text-[11px] text-text-muted">Inherited from v{selected.number}</p>}
        </div>)}
      </div>
      {error && <p role="alert" className="text-xs text-indicator-error">{error}</p>}
      {busy && <p role="status" className="text-xs text-text-secondary">Saving Reference Pack…</p>}
      <div className="flex flex-wrap gap-2">
        {editing ? <>
          <button type="button" disabled={busy || !name.trim() || (selected ? !changed : changed !== 5)} onClick={() => void save()} className={button}>Save draft revision</button>
          {character && <button type="button" disabled={busy} onClick={() => { setEditing(false); setFiles({}) }} className={button}>Cancel changes</button>}
        </> : <>
          <button type="button" disabled={busy} onClick={() => void review()} className={button}>{selected?.approved ? 'Unapprove revision' : 'Approve all five views'}</button>
          <button type="button" disabled={busy} onClick={() => { setFiles({}); setEditing(true); setError('') }} className={button}>New revision from this one</button>
        </>}
        <button type="button" disabled={busy} onClick={onClose} className={button}>Done</button>
      </div>
      <p className="text-[11px] text-text-muted">Revisions keep their own images. New revisions start as drafts and never replace an active selection. To switch revisions in H3, remove the active pack and add the approved revision you want.</p>
    </div>
  </SidebarDialog>
}
