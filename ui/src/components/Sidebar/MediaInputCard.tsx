import { ArrowDown, ArrowUp, ChevronDown, FileAudio, Image as ImageIcon, Plus, X } from 'lucide-react'
import { useRef, useState, type HTMLAttributes, type ReactNode } from 'react'
import { SidebarDialog } from './SidebarPanels'

/** A visible media role with its detailed settings kept beside the input. */
export function MediaInputCard({ title, subtitle, preview, mediaUrl, kind = 'image', disabled, onRemove, onEarlier, onLater, children, ...drag }: {
  title: string; subtitle?: string; preview?: string; mediaUrl?: string
  kind?: 'image' | 'video' | 'audio'; onRemove: () => void
  disabled?: boolean
  onEarlier?: () => void; onLater?: () => void; children: ReactNode
} & Omit<HTMLAttributes<HTMLDetailsElement>, 'title' | 'children'>) {
  const [viewing, setViewing] = useState(false)
  return <details {...drag} className="media-input-card group min-w-0 overflow-hidden rounded-xl border border-border bg-bg-tertiary">
    <summary className="relative cursor-pointer list-none select-none p-2 [&::-webkit-details-marker]:hidden">
      <div className="flex h-20 items-center justify-center overflow-hidden rounded-lg bg-bg-primary text-text-muted">
        {preview ? <img src={preview} alt="" loading="lazy" className="h-full w-full object-cover"/>
          : kind === 'video' && mediaUrl ? <video src={`${mediaUrl}#t=0.1`} muted playsInline preload="metadata" className="h-full w-full object-cover"/>
          : kind === 'audio' ? <FileAudio size={25}/> : <ImageIcon size={25}/>}
      </div>
      <button type="button" disabled={disabled} aria-label={`Remove ${title}`} onClick={event => { event.preventDefault(); event.stopPropagation(); onRemove() }}
        className="absolute right-2.5 top-2.5 rounded-full bg-black/65 p-1.5 text-white hover:bg-black/85"><X size={12}/></button>
      <div className="mt-2 flex items-center gap-1 text-xs font-medium text-text-primary"><span className="min-w-0 flex-1 truncate" title={title}>{title}</span><ChevronDown size={12}/></div>
      {subtitle && <p className="mt-0.5 truncate text-[10px] text-text-muted" title={subtitle}>{subtitle}</p>}
    </summary>
    <div className="space-y-2 border-t border-border p-2">
      <div className="flex flex-wrap items-center gap-2 text-[10px] text-text-secondary">
        {mediaUrl && <button type="button" onClick={() => setViewing(true)} className="mr-auto rounded-lg border border-border px-2 py-1.5 hover:bg-bg-hover">Preview</button>}
        <button type="button" disabled={!onEarlier} onClick={onEarlier} aria-label={`Move ${title} earlier`} className="rounded-lg p-1.5 hover:bg-bg-hover disabled:opacity-30"><ArrowUp size={13}/></button>
        <button type="button" disabled={!onLater} onClick={onLater} aria-label={`Move ${title} later`} className="rounded-lg p-1.5 hover:bg-bg-hover disabled:opacity-30"><ArrowDown size={13}/></button>
      </div>
      {children}
    </div>
    <SidebarDialog open={viewing} title={`${title} preview`} onClose={() => setViewing(false)}>
      {kind === 'video' ? <video src={viewing ? mediaUrl : undefined} controls playsInline className="w-full max-h-[65dvh]"/>
        : kind === 'audio' ? <audio src={viewing ? mediaUrl : undefined} controls className="w-full"/>
        : <img src={mediaUrl || preview} alt={title} className="w-full max-h-[65dvh] object-contain"/>}
    </SidebarDialog>
  </details>
}

export function MediaAddTile({ label = 'Add reference', hint, disabled, busy, accept, multiple = true, onFiles }: {
  label?: string; hint?: string; disabled?: boolean; busy?: boolean; accept?: string; multiple?: boolean; onFiles: (files: File[]) => void
}) {
  const input = useRef<HTMLInputElement>(null)
  const [dragging, setDragging] = useState(false)
  return <div className={`min-h-[128px] rounded-xl border border-dashed ${dragging ? 'border-accent-blue bg-accent-blue/10' : 'border-border bg-bg-tertiary/30'}`}
    onDragOver={event => { event.preventDefault(); if (!disabled) setDragging(true) }} onDragLeave={() => setDragging(false)}
    onDrop={event => { event.preventDefault(); setDragging(false); if (!disabled && !busy) onFiles(Array.from(event.dataTransfer.files)) }}>
    <button type="button" disabled={disabled || busy} onClick={() => input.current?.click()} aria-label={label}
      className="flex min-h-[128px] h-full w-full flex-col items-center justify-center gap-2 rounded-xl px-3 py-4 text-center text-text-secondary hover:text-text-primary hover:bg-bg-hover disabled:opacity-40">
      <Plus size={20}/><span className="text-xs">{busy ? 'Uploading…' : label}</span>
      {hint && <span className="text-[10px] text-text-muted">{hint}</span>}
    </button>
    <input ref={input} type="file" multiple={multiple} accept={accept} disabled={disabled || busy} className="hidden" aria-label={`${label} files`}
      onChange={event => { onFiles(Array.from(event.target.files || [])); event.currentTarget.value = '' }}/>
  </div>
}
