import { useMemo, useRef, useState } from 'react'
import {
  AudioLines,
  Film,
  Image as ImageIcon,
  Loader2,
  Plus,
  RefreshCw,
  Search,
  Type,
  Upload,
} from 'lucide-react'
import type { EditorAsset, EditorMediaType } from '../types'
import { useEditorStore } from './useEditorStore'

type Filter = 'all' | EditorMediaType

function MediaPreview({ asset }: { asset: EditorAsset }) {
  if (asset.type === 'image') {
    return <img src={asset.url} alt="" className="h-full w-full object-cover" loading="lazy" />
  }
  if (asset.type === 'video') {
    return <video src={asset.url} className="h-full w-full object-cover" muted preload="metadata" />
  }
  return (
    <div className="flex h-full w-full items-center justify-center bg-gradient-to-br from-accent-blue/20 to-accent-cool-to/10">
      <AudioLines size={20} className="text-accent-blue" />
    </div>
  )
}

export function EditorMediaBin({ compact = false }: { compact?: boolean }) {
  const inputRef = useRef<HTMLInputElement>(null)
  const [search, setSearch] = useState('')
  const [filter, setFilter] = useState<Filter>('all')
  const [uploading, setUploading] = useState(false)
  const library = useEditorStore(state => state.library)
  const refreshLibrary = useEditorStore(state => state.refreshLibrary)
  const uploadMedia = useEditorStore(state => state.uploadMedia)
  const addMedia = useEditorStore(state => state.addMedia)
  const addTitle = useEditorStore(state => state.addTitle)

  const visible = useMemo(() => {
    const query = search.trim().toLowerCase()
    return library.filter(asset => (
      (filter === 'all' || asset.type === filter)
      && (!query || asset.name.toLowerCase().includes(query))
    ))
  }, [filter, library, search])

  const handleFiles = async (files: FileList | null) => {
    if (!files?.length) return
    setUploading(true)
    try {
      for (const file of Array.from(files)) await uploadMedia(file)
    } finally {
      setUploading(false)
      if (inputRef.current) inputRef.current.value = ''
    }
  }

  return (
    <section className={`flex min-h-0 flex-col bg-bg-secondary ${compact ? 'h-full' : 'w-[260px] shrink-0 border-r border-border'}`}>
      <div className="space-y-2 border-b border-border p-2.5">
        <div className="flex items-center justify-between">
          <h2 className="text-[11px] font-semibold uppercase tracking-[0.12em] text-text-secondary">Media</h2>
          <button type="button" onClick={() => void refreshLibrary()} className="rounded p-1 text-text-muted hover:bg-bg-hover hover:text-text-primary" title="Refresh media">
            <RefreshCw size={12} />
          </button>
        </div>
        <div className="relative">
          <Search size={12} className="absolute left-2.5 top-1/2 -translate-y-1/2 text-text-muted" />
          <input
            value={search}
            onChange={event => setSearch(event.target.value)}
            placeholder="Search media"
            className="w-full rounded-lg border border-border bg-bg-tertiary py-1.5 pl-7 pr-2 text-[10px] text-text-primary outline-none placeholder:text-text-muted focus:border-accent-blue/60"
          />
        </div>
        <div className="flex items-center gap-1">
          {(['all', 'video', 'image', 'audio'] as Filter[]).map(value => {
            const Icon = value === 'video' ? Film : value === 'image' ? ImageIcon : value === 'audio' ? AudioLines : null
            return (
              <button
                key={value}
                type="button"
                onClick={() => setFilter(value)}
                className={`flex-1 rounded-md px-1.5 py-1 text-[9px] capitalize transition-colors ${filter === value ? 'bg-accent-blue/15 text-accent-blue' : 'text-text-muted hover:bg-bg-hover hover:text-text-secondary'}`}
              >
                {Icon ? <Icon size={10} className="mx-auto" /> : 'All'}
              </button>
            )
          })}
        </div>
        <div className="grid grid-cols-2 gap-1.5">
          <button
            type="button"
            onClick={() => inputRef.current?.click()}
            disabled={uploading}
            className="flex items-center justify-center gap-1 rounded-lg border border-dashed border-border-light bg-bg-tertiary px-2 py-1.5 text-[9px] text-text-secondary hover:border-accent-blue/50 hover:text-accent-blue disabled:opacity-50"
          >
            {uploading ? <Loader2 size={11} className="animate-spin" /> : <Upload size={11} />} Upload
          </button>
          <button
            type="button"
            onClick={() => addTitle()}
            className="flex items-center justify-center gap-1 rounded-lg border border-border bg-bg-tertiary px-2 py-1.5 text-[9px] text-text-secondary hover:border-accent-blue/40 hover:text-accent-blue"
          >
            <Type size={11} /> Title
          </button>
          <input
            ref={inputRef}
            type="file"
            accept="video/*,image/*,audio/*,.mkv,.webm,.mov,.mp4,.wav,.mp3,.m4a,.flac,.ogg"
            multiple
            className="hidden"
            onChange={event => void handleFiles(event.target.files)}
          />
        </div>
      </div>

      <div className={`min-h-0 flex-1 overflow-y-auto p-2 ${compact ? 'grid auto-rows-max grid-cols-2 gap-2 sm:grid-cols-3' : 'space-y-1.5'}`}>
        {visible.map(asset => (
          <article
            key={`${asset.origin}:${asset.name}`}
            draggable
            onDragStart={event => {
              event.dataTransfer.effectAllowed = 'copy'
              event.dataTransfer.setData('application/x-maestro-editor-asset', JSON.stringify(asset))
            }}
            onDoubleClick={() => void addMedia(asset)}
            className={`group overflow-hidden rounded-lg border border-border bg-bg-tertiary transition-colors hover:border-border-light ${compact ? '' : 'flex items-center gap-2 p-1.5'}`}
          >
            <div className={`relative shrink-0 overflow-hidden rounded-md bg-media-canvas ${compact ? 'aspect-video w-full' : 'h-12 w-16'}`}>
              <MediaPreview asset={asset} />
              <span className="absolute bottom-1 left-1 rounded bg-black/65 px-1 py-0.5 text-[7px] uppercase text-white/80">{asset.origin}</span>
            </div>
            <div className={`min-w-0 ${compact ? 'p-1.5' : 'flex-1'}`}>
              <div className="truncate text-[9px] font-medium text-text-secondary" title={asset.name}>{asset.name}</div>
              <div className="mt-0.5 flex items-center justify-between gap-1 text-[8px] text-text-muted">
                <span className="capitalize">{asset.type}</span>
                <button
                  type="button"
                  onClick={() => void addMedia(asset)}
                  className="rounded bg-accent-blue/10 p-1 text-accent-blue opacity-80 hover:bg-accent-blue/20 group-hover:opacity-100"
                  title="Add at playhead"
                >
                  <Plus size={10} />
                </button>
              </div>
            </div>
          </article>
        ))}
        {visible.length === 0 && (
          <div className="col-span-full flex min-h-40 flex-col items-center justify-center gap-2 px-4 text-center text-text-muted">
            <Film size={24} className="opacity-50" />
            <p className="text-[10px] leading-relaxed">Upload media or generate it in Studio. Your workspace media appears here automatically.</p>
          </div>
        )}
      </div>
    </section>
  )
}
