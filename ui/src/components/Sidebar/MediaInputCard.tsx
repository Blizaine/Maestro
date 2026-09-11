import { Eye, FileAudio, Image as ImageIcon, Plus, X } from 'lucide-react'
import { useRef, useState, type HTMLAttributes } from 'react'
import { SidebarDialog } from './SidebarPanels'

/** A reference thumbnail; its parent places the editor below the active row. */
export function MediaInputCard({ title, subtitle, preview, mediaUrl, kind = 'image', disabled, onRemove, onEarlier, onLater, expanded, onEdit, editorId, ...drag }: {
  title: string; subtitle?: string; preview?: string; mediaUrl?: string
  kind?: 'image' | 'video' | 'audio'; onRemove: () => void
  disabled?: boolean
  onEarlier?: () => void; onLater?: () => void
  expanded: boolean; onEdit: () => void; editorId: string
} & Omit<HTMLAttributes<HTMLDivElement>, 'title' | 'children'>) {
  const [viewing, setViewing] = useState(false)
  return <div {...drag} className={`media-input-card group relative min-w-0 overflow-hidden rounded-xl border bg-bg-tertiary ${expanded ? 'border-accent-blue' : 'border-border'}`}>
    <button type="button" aria-label={`Edit ${title} reference`} aria-expanded={expanded} aria-controls={expanded ? editorId : undefined}
      aria-keyshortcuts="Alt+ArrowLeft Alt+ArrowRight" title="Edit reference · Drag to reorder, or use Alt + Left/Right"
      onClick={onEdit} onKeyDown={event => {
        if (!event.altKey || !['ArrowLeft', 'ArrowRight'].includes(event.key)) return
        event.preventDefault()
        if (event.key === 'ArrowLeft') onEarlier?.()
        else onLater?.()
      }} className="block w-full rounded-xl p-1.5 text-left hover:bg-bg-hover focus-visible:outline-2 focus-visible:-outline-offset-2 focus-visible:outline-accent-blue">
      <div className="flex h-14 items-center justify-center overflow-hidden rounded-lg bg-bg-primary text-text-muted">
        {preview ? <img src={preview} alt="" draggable={false} loading="lazy" className="h-full w-full object-cover"/>
          : kind === 'video' && mediaUrl ? <video src={`${mediaUrl}#t=0.1`} muted playsInline preload="metadata" className="h-full w-full object-cover"/>
          : kind === 'audio' ? <FileAudio size={25}/> : <ImageIcon size={25}/>}
      </div>
      <div className="mt-1 truncate text-[11px] font-medium text-text-primary" title={title}>{title}</div>
      {subtitle && <p className="mt-0.5 truncate text-[10px] text-text-muted" title={subtitle}>{subtitle}</p>}
    </button>
    <button type="button" disabled={disabled} aria-label={`Remove ${title}`} onClick={event => { event.stopPropagation(); onRemove() }}
      className="absolute right-1.5 top-1.5 rounded-full bg-black/65 p-1.5 text-white hover:bg-black/85"><X size={12}/></button>
    {mediaUrl && <button type="button" aria-label={`Preview ${title}`} title={`Preview ${title}`} onClick={() => setViewing(true)}
      className="absolute left-1.5 top-1.5 rounded-full bg-black/65 p-1.5 text-white hover:bg-black/85"><Eye size={12}/></button>}
    <SidebarDialog open={viewing} title={`${title} preview`} onClose={() => setViewing(false)}>
      {kind === 'video' ? <video src={viewing ? mediaUrl : undefined} controls playsInline className="w-full max-h-[65dvh]"/>
        : kind === 'audio' ? <audio src={viewing ? mediaUrl : undefined} controls className="w-full"/>
        : <img src={mediaUrl || preview} alt={title} className="w-full max-h-[65dvh] object-contain"/>}
    </SidebarDialog>
  </div>
}

export function MediaAddTile({ label = 'Add reference', hint, disabled, busy, accept, multiple = true, onFiles }: {
  label?: string; hint?: string; disabled?: boolean; busy?: boolean; accept?: string; multiple?: boolean; onFiles: (files: File[]) => void
}) {
  const input = useRef<HTMLInputElement>(null)
  const [dragging, setDragging] = useState(false)
  return <div className={`studio-media-add min-h-[100px] rounded-xl border border-dashed ${dragging ? 'border-accent-blue bg-accent-blue/10' : 'border-border bg-bg-tertiary/30'}`}
    onDragOver={event => { event.preventDefault(); if (!disabled) setDragging(true) }} onDragLeave={() => setDragging(false)}
    onDrop={event => { event.preventDefault(); setDragging(false); if (!disabled && !busy) onFiles(Array.from(event.dataTransfer.files)) }}>
    <button type="button" disabled={disabled || busy} onClick={() => input.current?.click()} aria-label={label}
      className="flex min-h-[100px] h-full w-full flex-col items-center justify-center gap-1 rounded-xl px-2 py-2 text-center text-text-secondary hover:text-text-primary hover:bg-bg-hover disabled:opacity-40">
      <Plus size={18}/><span className="text-[11px]">{busy ? 'Uploading…' : label}</span>
      {hint && <span className="text-[10px] text-text-muted">{hint}</span>}
    </button>
    <input ref={input} type="file" multiple={multiple} accept={accept} disabled={disabled || busy} className="hidden" aria-label={`${label} files`}
      onChange={event => { onFiles(Array.from(event.target.files || [])); event.currentTarget.value = '' }}/>
  </div>
}
